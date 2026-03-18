"""
Post-training 阈值校准扫描
Calibration Sweep for Post-processing Thresholds

A0: 从现有 vectors/ + masks/ 重建 pre-filter 候选集 (candidates.gpkg)
A1: 在候选集上扫描 post_conf / min_area / max_elongation 组合
A2: (手动) 用最优配置重跑 detect_and_evaluate.py 做 end-to-end 确认

用法:
  python calibration_sweep.py --step a0          # 导出 pre-filter candidates + baseline
  python calibration_sweep.py --step a1 --dry    # 打印搜索空间
  python calibration_sweep.py --step a1          # 运行 sweep
  python calibration_sweep.py --step a1 --top 10 # 运行后打印 top-10 合格组合
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import detect_and_evaluate as pipeline
from core.grid_utils import normalize_grid_id

# ════════════════════════════════════════════════════════════════════════
# 常量
# ════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRIDS = ["G1189", "G1190", "G1238"]
SWEEP_DIR = BASE_DIR / "results" / "calibration_sweep"

# 搜索空间
SEARCH_SPACE = {
    "post_conf_threshold": [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "min_object_area": [1.0, 1.5, 2.0, 3.0, 5.0],
    "max_elongation": [3.0, 4.0, 5.0, 8.0],
}

# 5-20m² size bucket definition
SIZE_BINS = [0, 5, 20, 50, 100, float("inf")]
SIZE_LABELS = ["<5m2", "5-20m2", "20-50m2", "50-100m2", ">100m2"]


# ════════════════════════════════════════════════════════════════════════
# A0: 重建 pre-filter 候选集
# ════════════════════════════════════════════════════════════════════════

def _load_vectors_for_grid(grid_id: str) -> list[gpd.GeoDataFrame]:
    """加载某个 grid 的所有 per-tile vector GeoJSON 并回填 confidence + RGB。"""
    vectors_dir = BASE_DIR / "results" / grid_id / "vectors"
    masks_dir = BASE_DIR / "results" / grid_id / "masks"
    tiles_dir = BASE_DIR / "tiles" / grid_id

    vector_files = sorted(vectors_dir.glob("*_vectors.geojson"))
    if not vector_files:
        raise FileNotFoundError(f"No vector files in {vectors_dir}")

    import rasterstats

    all_gdfs = []
    for vf in vector_files:
        tile_name = vf.stem.replace("_vectors", "")  # e.g. G1189_3_0_geo
        mask_path = masks_dir / f"{tile_name}_mask.tif"
        tile_path = tiles_dir / f"{tile_name}.tif"

        gdf = gpd.read_file(str(vf))
        if len(gdf) == 0:
            continue

        # 回填 confidence (mask band 2)
        if mask_path.exists():
            try:
                conf_stats = rasterstats.zonal_stats(
                    gdf, str(mask_path), band=2, stats=["mean"], nodata=0,
                )
                gdf["confidence"] = [
                    (s["mean"] / 255.0) if s["mean"] is not None else 0.0
                    for s in conf_stats
                ]
            except Exception as e:
                print(f"  [WARN] {tile_name} confidence 回填失败: {e}")
                gdf["confidence"] = 0.5
        else:
            print(f"  [WARN] mask not found: {mask_path}")
            gdf["confidence"] = 0.5

        # 回填 RGB (for shadow/bright filtering)
        if tile_path.exists():
            try:
                for band_idx, col_name in [(1, "mean_r"), (2, "mean_g"), (3, "mean_b")]:
                    stats = rasterstats.zonal_stats(
                        gdf, str(tile_path), band=band_idx, stats=["mean"], nodata=0,
                    )
                    gdf[col_name] = [
                        s["mean"] if s["mean"] is not None else 0 for s in stats
                    ]

                # 阴影 + 过曝过滤 (same logic as detect_and_evaluate.py)
                shadow_thresh = pipeline.SHADOW_RGB_THRESH
                is_shadow = (
                    (gdf["mean_r"] < shadow_thresh)
                    & (gdf["mean_g"] < shadow_thresh)
                    & (gdf["mean_b"] < shadow_thresh)
                )
                is_too_bright = (
                    (gdf["mean_r"] > 250) & (gdf["mean_g"] > 250) & (gdf["mean_b"] > 250)
                )
                pre_count = len(gdf)
                gdf = gdf[~(is_shadow | is_too_bright)].copy()
                removed = pre_count - len(gdf)
                if removed > 0:
                    print(f"    {tile_name}: 阴影/过曝移除 {removed} 个")
            except Exception as e:
                print(f"  [WARN] {tile_name} RGB过滤失败: {e}")

        if len(gdf) > 0:
            gdf["source_tile"] = tile_name
            all_gdfs.append(gdf)

    return all_gdfs


def build_candidates(grid_id: str) -> gpd.GeoDataFrame:
    """为单个 grid 重建 pre-filter 候选集。"""
    import geoai

    print(f"\n[A0] {grid_id}: 加载 per-tile vectors...")
    all_gdfs = _load_vectors_for_grid(grid_id)

    if not all_gdfs:
        raise RuntimeError(f"{grid_id}: 未找到任何有效的 vector 数据")

    combined = pd.concat(all_gdfs, ignore_index=True)
    print(f"  合并后: {len(combined)} 个候选多边形")

    # 确保 CRS 并转换到 metric
    combined = pipeline.ensure_crs(combined, assumed_crs=pipeline.INPUT_CRS, label=f"{grid_id} candidates")
    combined = combined.to_crs(pipeline.METRIC_CRS)

    # 添加几何属性 (area_m2, elongation, solidity)
    combined = geoai.add_geometric_properties(combined)

    # Spatial NMS (same as detection pipeline)
    pre_nms = len(combined)
    combined = pipeline.spatial_nms(combined, iou_threshold=0.5)
    print(f"  NMS: {pre_nms} → {len(combined)} 个多边形")

    # 确保 confidence 列存在
    if "confidence" not in combined.columns:
        combined["confidence"] = 0.5

    return combined


def read_baseline_metrics(grid_id: str) -> dict:
    """从现有评估结果文件中读取 baseline 指标。"""
    result_dir = BASE_DIR / "results" / grid_id

    # Merge F1@IoU0.3 from presence_metrics.csv
    presence_csv = result_dir / "presence_metrics.csv"
    presence_data = pd.read_csv(presence_csv, encoding="utf-8-sig")
    presence_r = float(presence_data["recall"].iloc[0])
    presence_f1 = float(presence_data["f1"].iloc[0])

    # Size-stratified recall from size_stratified_metrics.csv
    size_csv = result_dir / "size_stratified_metrics.csv"
    size_data = pd.read_csv(size_csv, encoding="utf-8-sig")
    # 5-20m² at IoU 0.3
    row_5_20 = size_data[
        (size_data["IoU_Threshold"] == 0.3)
        & (size_data["size_class"] == "5-20m2")
    ]
    size_5_20_recall = float(row_5_20["recall"].iloc[0]) if len(row_5_20) > 0 else None

    # Merge F1@IoU0.3 — compute dynamically from predictions + GT
    pipeline.set_grid_context(normalize_grid_id(grid_id))
    gt = pipeline.load_ground_truth()
    pred_path = result_dir / "predictions_metric.gpkg"
    pred = gpd.read_file(str(pred_path))
    m03 = pipeline.iou_matching(gt, pred, iou_threshold=0.3, merge_preds=True)
    merge_f1 = round(m03["f1"], 4)

    return {
        "merge_f1_iou03": merge_f1,
        "presence_r_iou01": round(presence_r, 4),
        "presence_f1_iou01": round(presence_f1, 4),
        "size_5_20_recall_iou03": round(size_5_20_recall, 4) if size_5_20_recall is not None else None,
    }


def step_a0():
    """Step A0: 导出 pre-filter candidates + baseline_snapshot.json."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    baseline = {"grids": {}, "macro_mean_f1": None}

    for grid_id in GRIDS:
        grid_sweep_dir = SWEEP_DIR / grid_id
        grid_sweep_dir.mkdir(parents=True, exist_ok=True)

        # 重建候选集
        candidates = build_candidates(grid_id)
        out_path = grid_sweep_dir / "candidates.gpkg"
        candidates.to_file(str(out_path), driver="GPKG")
        print(f"  [OK] {out_path} ({len(candidates)} candidates)")

        # 读取 baseline 指标
        metrics = read_baseline_metrics(grid_id)
        baseline["grids"][grid_id] = metrics
        print(f"  [BASELINE] {grid_id}: {metrics}")

    # 计算 macro mean
    f1_values = [baseline["grids"][g]["merge_f1_iou03"] for g in GRIDS]
    baseline["macro_mean_f1"] = round(sum(f1_values) / len(f1_values), 4)

    # 保存 baseline snapshot
    snapshot_path = SWEEP_DIR / "baseline_snapshot.json"
    snapshot_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[OK] baseline snapshot: {snapshot_path}")
    print(f"  Macro mean F1@IoU0.3: {baseline['macro_mean_f1']}")

    return baseline


# ════════════════════════════════════════════════════════════════════════
# A1: Post-processing sweep
# ════════════════════════════════════════════════════════════════════════

def _compute_size_bucket_recall(
    gt: gpd.GeoDataFrame,
    pred: gpd.GeoDataFrame,
    iou_threshold: float = 0.3,
    bucket: str = "5-20m2",
) -> float | None:
    """纯逻辑计算特定 size bucket 的 recall，无文件写入副作用。"""
    gt_metric = gt.to_crs(pipeline.METRIC_CRS).copy()
    gt_metric["area_m2"] = gt_metric.geometry.area
    gt_metric["size_class"] = pd.cut(
        gt_metric["area_m2"], bins=SIZE_BINS, labels=SIZE_LABELS, include_lowest=True,
    )

    metrics = pipeline.iou_matching(gt_metric, pred, iou_threshold=iou_threshold)
    gt_metric["_matched"] = gt_metric.index.isin(list(metrics["matched_gt_indices"]))

    subset = gt_metric[gt_metric["size_class"] == bucket]
    if len(subset) == 0:
        return None
    return int(subset["_matched"].sum()) / len(subset)


def _evaluate_combo(
    candidates: gpd.GeoDataFrame,
    gt: gpd.GeoDataFrame,
    post_conf: float,
    min_area: float,
    max_elong: float,
) -> dict:
    """对单个参数组合做过滤 + 评估，返回指标 dict。"""
    # 过滤
    filtered = candidates.copy()
    filtered = filtered[filtered["confidence"] >= post_conf]
    if "area_m2" in filtered.columns:
        filtered = filtered[filtered["area_m2"] >= min_area]
    if "elongation" in filtered.columns and max_elong < 999:
        filtered = filtered[filtered["elongation"] <= max_elong]

    n_pred = len(filtered)
    if n_pred == 0:
        return {
            "n_pred": 0,
            "merge_f1_iou03": 0.0,
            "presence_r_iou01": 0.0,
            "size_5_20_recall_iou03": 0.0,
        }

    # Merge F1@IoU0.3
    m03 = pipeline.iou_matching(gt, filtered, iou_threshold=0.3, merge_preds=True)

    # Presence recall@IoU0.1
    m01 = pipeline.iou_matching(gt, filtered, iou_threshold=0.1, merge_preds=True)

    # 5-20m² recall@IoU0.3 (inline, no side effects)
    size_5_20_r = _compute_size_bucket_recall(gt, filtered, iou_threshold=0.3, bucket="5-20m2")

    return {
        "n_pred": n_pred,
        "merge_f1_iou03": round(m03["f1"], 4),
        "merge_p_iou03": round(m03["precision"], 4),
        "merge_r_iou03": round(m03["recall"], 4),
        "presence_r_iou01": round(m01["recall"], 4),
        "presence_f1_iou01": round(m01["f1"], 4),
        "size_5_20_recall_iou03": round(size_5_20_r, 4) if size_5_20_r is not None else None,
    }


def _check_constraints(row: dict, baseline: dict) -> tuple[bool, str]:
    """检查单行结果是否满足所有硬约束。返回 (pass, reason)."""
    reasons = []

    # Constraint 1: G1189 F1@IoU0.3 >= baseline
    g1189_f1 = row.get("G1189_merge_f1_iou03", 0.0)
    g1189_base = baseline["grids"]["G1189"]["merge_f1_iou03"]
    if g1189_f1 < g1189_base:
        reasons.append(f"G1189 F1 {g1189_f1:.4f} < {g1189_base:.4f}")

    # Constraint 2: Presence recall drop <= 0.01 per grid
    for g in GRIDS:
        pr = row.get(f"{g}_presence_r_iou01", 0.0)
        pr_base = baseline["grids"][g]["presence_r_iou01"]
        if pr_base - pr > 0.01:
            reasons.append(f"{g} pres_R drop {pr_base - pr:.4f} > 0.01")

    # Constraint 3: 5-20m² recall
    for g in GRIDS:
        sr = row.get(f"{g}_size_5_20_recall_iou03")
        sr_base = baseline["grids"][g].get("size_5_20_recall_iou03")
        if sr is None or sr_base is None:
            continue
        if g == "G1189":
            # G1189: must not decrease
            if sr < sr_base:
                reasons.append(f"G1189 5-20m² R {sr:.4f} < {sr_base:.4f}")
        else:
            # G1190/G1238: drop <= 0.02
            if sr_base - sr > 0.02:
                reasons.append(f"{g} 5-20m² R drop {sr_base - sr:.4f} > 0.02")

    passed = len(reasons) == 0
    return passed, "; ".join(reasons) if reasons else "OK"


def step_a1(dry_run: bool = False, top_n: int = 5):
    """Step A1: Post-processing sweep."""
    combos = list(itertools.product(
        SEARCH_SPACE["post_conf_threshold"],
        SEARCH_SPACE["min_object_area"],
        SEARCH_SPACE["max_elongation"],
    ))
    print(f"搜索空间: {len(combos)} 组合 × {len(GRIDS)} grids = {len(combos) * len(GRIDS)} 次评估")

    if dry_run:
        for i, (pc, ma, me) in enumerate(combos, 1):
            print(f"  [{i:03d}] post_conf={pc}, min_area={ma}, max_elong={me}")
        return

    # 加载 baseline
    snapshot_path = SWEEP_DIR / "baseline_snapshot.json"
    if not snapshot_path.exists():
        print("[ERROR] baseline_snapshot.json not found. Run --step a0 first.")
        sys.exit(1)
    baseline = json.loads(snapshot_path.read_text(encoding="utf-8"))

    # 预加载 candidates 和 GT
    print("\n加载候选集和真值数据...")
    candidates_cache = {}
    gt_cache = {}
    for grid_id in GRIDS:
        cand_path = SWEEP_DIR / grid_id / "candidates.gpkg"
        if not cand_path.exists():
            print(f"[ERROR] {cand_path} not found. Run --step a0 first.")
            sys.exit(1)
        candidates_cache[grid_id] = gpd.read_file(str(cand_path))
        print(f"  {grid_id}: {len(candidates_cache[grid_id])} candidates")

        pipeline.set_grid_context(normalize_grid_id(grid_id))
        gt_cache[grid_id] = pipeline.load_ground_truth()
        print(f"  {grid_id}: {len(gt_cache[grid_id])} GT polygons")

    # 执行 sweep
    results = []
    summary_path = SWEEP_DIR / "postproc_summary.csv"
    t0 = time.time()

    for i, (pc, ma, me) in enumerate(combos, 1):
        row = {
            "post_conf_threshold": pc,
            "min_object_area": ma,
            "max_elongation": me,
        }

        for grid_id in GRIDS:
            metrics = _evaluate_combo(
                candidates_cache[grid_id], gt_cache[grid_id], pc, ma, me,
            )
            for k, v in metrics.items():
                row[f"{grid_id}_{k}"] = v

        # Macro mean F1
        f1_values = [row[f"{g}_merge_f1_iou03"] for g in GRIDS]
        row["macro_mean_f1"] = round(sum(f1_values) / len(f1_values), 4)

        # 约束检查
        passed, reason = _check_constraints(row, baseline)
        row["constraints_pass"] = passed
        row["constraint_reason"] = reason

        results.append(row)

        if i % 20 == 0 or i == len(combos):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(combos)}] {elapsed:.1f}s elapsed")

    df = pd.DataFrame(results)
    df.to_csv(str(summary_path), index=False, encoding="utf-8-sig")
    print(f"\n[OK] sweep 完成: {summary_path}")
    print(f"  总计 {len(df)} 组合, 约束通过: {df['constraints_pass'].sum()}")

    # 打印 top-N
    passed_df = df[df["constraints_pass"]].sort_values("macro_mean_f1", ascending=False)
    if len(passed_df) > 0:
        print(f"\nTop-{top_n} 合格组合 (macro mean F1@IoU0.3):")
        display_cols = [
            "post_conf_threshold", "min_object_area", "max_elongation",
            "macro_mean_f1",
            "G1189_merge_f1_iou03", "G1190_merge_f1_iou03", "G1238_merge_f1_iou03",
            "G1189_presence_r_iou01", "G1190_presence_r_iou01", "G1238_presence_r_iou01",
        ]
        print(passed_df.head(top_n)[display_cols].to_string(index=False))

        # 保存 best config
        best = passed_df.iloc[0]
        best_config = {
            "post_conf_threshold": float(best["post_conf_threshold"]),
            "min_object_area": float(best["min_object_area"]),
            "max_elongation": float(best["max_elongation"]),
            "macro_mean_f1": float(best["macro_mean_f1"]),
            "per_grid": {
                g: {
                    "merge_f1_iou03": float(best[f"{g}_merge_f1_iou03"]),
                    "presence_r_iou01": float(best[f"{g}_presence_r_iou01"]),
                    "size_5_20_recall_iou03": (
                        float(best[f"{g}_size_5_20_recall_iou03"])
                        if best[f"{g}_size_5_20_recall_iou03"] is not None
                        else None
                    ),
                }
                for g in GRIDS
            },
            "baseline": baseline,
        }
        best_path = SWEEP_DIR / "best_config.json"
        best_path.write_text(
            json.dumps(best_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n[OK] best config: {best_path}")
    else:
        print("\n[WARN] 无任何组合通过所有约束!")
        print("最接近的组合:")
        near = df.sort_values("macro_mean_f1", ascending=False).head(5)
        print(near[["post_conf_threshold", "min_object_area", "max_elongation",
                     "macro_mean_f1", "constraint_reason"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-training 阈值校准扫描"
    )
    parser.add_argument(
        "--step", required=True, choices=["a0", "a1"],
        help="执行步骤: a0=导出候选集, a1=sweep",
    )
    parser.add_argument("--dry", action="store_true", help="只打印搜索空间")
    parser.add_argument("--top", type=int, default=5, help="打印 top-N 合格组合")
    args = parser.parse_args()

    try:
        if args.step == "a0":
            step_a0()
        elif args.step == "a1":
            step_a1(dry_run=args.dry, top_n=args.top)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
