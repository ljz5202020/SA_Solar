"""
参数网格搜索：扫描 chip_size / overlap / min_object_area 组合
Parameter Grid Search for Solar Panel Detection

用法：
  python param_search.py          # 运行全部组合
  python param_search.py --dry    # 只打印组合，不实际运行

每组实验输出到 results/<GRID_ID>/param_search/<experiment_id>/
汇总表输出到 results/<GRID_ID>/param_search/summary.csv
"""

import itertools
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

# 复用主流水线的函数
from detect_and_evaluate import (
    GRID_ID,
    OUTPUT_DIR,
    TARGET_CRS,
    DEFAULT_IOU,
    detect_solar_panels,
    load_ground_truth,
    iou_matching,
)

# ════════════════════════════════════════════════════════════════════════
# 搜索空间
# GSD ≈ 0.08 m/pixel，chip_size=400 只覆盖 ~30m，大阵列容易被切断
# 偏向大 chip 搜索；同时搜索 epsilon（orthogonalize 简化强度）
# ════════════════════════════════════════════════════════════════════════
PARAM_GRID = {
    "chip_size": [(400, 400), (640, 640), (800, 800), (1024, 1024)],
    "overlap": [0.25, 0.35],
    "min_object_area": [1.0, 2.0],
}

SEARCH_DIR = OUTPUT_DIR / "param_search"


def run_search(dry_run: bool = False):
    """遍历参数组合，每组跑检测 + 评估。"""
    combos = list(itertools.product(
        PARAM_GRID["chip_size"],
        PARAM_GRID["overlap"],
        PARAM_GRID["min_object_area"],
    ))
    print(f"参数搜索: {len(combos)} 组实验")
    print(f"输出目录: {SEARCH_DIR}\n")

    if dry_run:
        for i, (cs, ov, ma) in enumerate(combos, 1):
            print(f"  [{i:02d}] chip_size={cs}, overlap={ov}, min_area={ma}")
        return

    SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # 预加载 GT（只加载一次）
    gt = load_ground_truth()

    results = []
    summary_path = SEARCH_DIR / "summary.csv"

    for i, (cs, ov, ma) in enumerate(combos, 1):
        exp_id = f"cs{cs[0]}_ov{ov:.2f}_ma{ma:.1f}"
        exp_dir = SEARCH_DIR / exp_id

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(combos)}] {exp_id}")
        print(f"  chip_size={cs}, overlap={ov}, min_object_area={ma}")

        # 跳过已完成的实验
        pred_path = exp_dir / "predictions.geojson"
        if pred_path.exists():
            print(f"  [SKIP] 已存在 predictions.geojson，直接评估")
        else:
            t0 = time.time()
            try:
                detect_solar_panels(
                    chip_size=cs,
                    overlap=ov,
                    min_object_area=ma,
                    output_dir=str(exp_dir),
                )
            except Exception as e:
                print(f"  [ERROR] 检测失败: {e}")
                results.append({
                    "experiment_id": exp_id,
                    "chip_size": cs[0],
                    "overlap": ov,
                    "min_object_area": ma,
                    "status": "error",
                    "error": str(e),
                })
                continue
            elapsed = time.time() - t0
            print(f"  检测耗时: {elapsed:.1f}s")

        # 评估
        try:
            import geopandas as gpd
            pred = gpd.read_file(str(pred_path))
            if pred.crs is None:
                pred = pred.set_crs("EPSG:4326")
            pred = pred.to_crs(TARGET_CRS)
            pred = pred[pred.geometry.notnull() & pred.is_valid].copy()

            metrics = iou_matching(gt, pred, iou_threshold=DEFAULT_IOU)

            row = {
                "experiment_id": exp_id,
                "chip_size": cs[0],
                "overlap": ov,
                "min_object_area": ma,
                "n_pred": len(pred),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "status": "done",
            }
            results.append(row)
            print(f"  P={row['precision']:.4f}  R={row['recall']:.4f}  "
                  f"F1={row['f1']:.4f}  (TP={row['tp']} FP={row['fp']} FN={row['fn']})")

        except Exception as e:
            print(f"  [ERROR] 评估失败: {e}")
            results.append({
                "experiment_id": exp_id,
                "chip_size": cs[0],
                "overlap": ov,
                "min_object_area": ma,
                "status": "eval_error",
                "error": str(e),
            })

        # 每组实验后增量保存汇总
        pd.DataFrame(results).to_csv(
            str(summary_path), index=False, encoding="utf-8-sig"
        )

    # 最终汇总
    df = pd.DataFrame(results)
    df.to_csv(str(summary_path), index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"参数搜索完成! 汇总表: {summary_path}")
    if "f1" in df.columns:
        done = df[df["status"] == "done"].sort_values("f1", ascending=False)
        if len(done) > 0:
            print(f"\nTop 5 by F1:")
            print(done.head(5)[["experiment_id", "precision", "recall", "f1",
                                "tp", "fp", "fn"]].to_string(index=False))


if __name__ == "__main__":
    dry_run = "--dry" in sys.argv
    run_search(dry_run=dry_run)
