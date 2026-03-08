"""
太阳能板检测与评估完整流水线
Solar Panel Detection & Evaluation Pipeline

功能：
  1. 使用 geoai SolarPanelDetector 对 tiles/ 中的 GeoTIFF 进行检测
  2. 加载 g1238.gpkg / g1238.geojson 真值数据
  3. 基于 IoU 匹配进行多阈值评估
  4. 生成可视化图表
  5. 输出汇总报告和逐 tile CSV

依赖安装：
  pip install geoai-py geopandas shapely scikit-learn matplotlib seaborn rasterio
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# 配置常量
# ════════════════════════════════════════════════════════════════════════
BASE_DIR      = Path(__file__).parent
GRID_ID       = "G1238"
TILES_DIR     = BASE_DIR / "tiles" / GRID_ID
GT_GPKG       = BASE_DIR / "data" / "annotations" / f"{GRID_ID}.gpkg"
GT_GEOJSON    = BASE_DIR / "data" / "annotations" / f"{GRID_ID.lower()}.geojson"
OUTPUT_DIR    = BASE_DIR / "results" / GRID_ID
MASKS_DIR     = OUTPUT_DIR / "masks"
VECTORS_DIR   = OUTPUT_DIR / "vectors"
BUILDINGS_GPKG = BASE_DIR / "buildings.gpkg"

# geoai 检测参数
CONFIDENCE_THRESHOLD = 0.3
MASK_THRESHOLD       = 0.3
MIN_OBJECT_AREA      = 2      # 后处理面积过滤（m²），<2m² 几乎全是碎片
MAX_ELONGATION       = 4.0    # 后处理长宽比过滤，>4 几乎全是误检
MIN_SOLIDITY         = 0.0    # 暂不限制 solidity（TP/FP 分布重叠太大）
SHADOW_RGB_THRESH    = 60     # RGB 三通道均 < 此值视为阴影
POST_CONF_THRESHOLD  = 0.70   # 后处理置信度过滤（基于 mask band2 回填值）
OVERLAP              = 0.25
CHIP_SIZE            = (400, 400)
BATCH_SIZE           = 4

# 评估参数
IOU_THRESHOLDS       = [0.1, 0.2, 0.3, 0.5, 0.7]
DEFAULT_IOU          = 0.3
TARGET_CRS           = "EPSG:4326"  # WGS 84（开普敦适用）

# 输出文件路径
PREDICTIONS_PATH         = OUTPUT_DIR / "predictions.geojson"
CONFIDENCE_HIST_PATH     = OUTPUT_DIR / "confidence_histogram.png"
PR_CURVE_PATH            = OUTPUT_DIR / "precision_recall_curve.png"
IOU_METRICS_PATH         = OUTPUT_DIR / "iou_threshold_metrics.png"
EVALUATION_CSV_PATH      = OUTPUT_DIR / "evaluation_per_tile.csv"


# ════════════════════════════════════════════════════════════════════════
# 辅助函数：空间 NMS 去重
# ════════════════════════════════════════════════════════════════════════
def spatial_nms(gdf: gpd.GeoDataFrame, iou_threshold: float = 0.5) -> gpd.GeoDataFrame:
    """
    空间非极大值抑制：合并 IoU > threshold 的重复检测。
    保留面积较大的多边形（通常更完整）。
    """
    if len(gdf) <= 1:
        return gdf

    keep = [True] * len(gdf)
    sindex = gdf.sindex

    for i in range(len(gdf)):
        if not keep[i]:
            continue
        geom_i = gdf.iloc[i].geometry
        candidates = list(sindex.intersection(geom_i.bounds))
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            geom_j = gdf.iloc[j].geometry
            try:
                inter = geom_i.intersection(geom_j).area
                union = geom_i.area + geom_j.area - inter
                if union > 0 and (inter / union) > iou_threshold:
                    # 保留面积较大的
                    if geom_i.area >= geom_j.area:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
            except Exception:
                continue

    pre_nms = len(gdf)
    result = gdf[keep].copy()
    if pre_nms > len(result):
        print(f"\n空间 NMS: 移除了 {pre_nms - len(result)} 个重复检测")
    return result


# ════════════════════════════════════════════════════════════════════════
# 第一步：检测太阳能板
# ════════════════════════════════════════════════════════════════════════
def detect_solar_panels(
    chip_size=None,
    overlap=None,
    min_object_area=None,
    confidence_threshold=None,
    mask_threshold=None,
    output_dir=None,
) -> gpd.GeoDataFrame:
    """
    使用 geoai SolarPanelDetector 对每张 GeoTIFF 进行检测。
    参数可覆盖模块级常量，用于参数搜索。
    """
    _chip_size = chip_size or CHIP_SIZE
    _overlap = overlap if overlap is not None else OVERLAP
    _min_object_area = min_object_area if min_object_area is not None else MIN_OBJECT_AREA
    _confidence_threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    _mask_threshold = mask_threshold if mask_threshold is not None else MASK_THRESHOLD
    _output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    _masks_dir = _output_dir / "masks"
    _vectors_dir = _output_dir / "vectors"
    _predictions_path = _output_dir / "predictions.geojson"

    _output_dir.mkdir(parents=True, exist_ok=True)
    _masks_dir.mkdir(parents=True, exist_ok=True)
    _vectors_dir.mkdir(parents=True, exist_ok=True)

    # 只处理带有地理参考信息的 _geo.tif 文件
    geo_tifs = sorted(TILES_DIR.glob(f"{GRID_ID}_*_*_geo.tif"))
    if not geo_tifs:
        # 回退到普通 .tif（不包含 _geo 后缀且不含 mosaic/mask 等关键词）
        geo_tifs = sorted([
            f for f in TILES_DIR.glob(f"{GRID_ID}_*_*.tif")
            if "_geo" not in f.stem and "mosaic" not in f.stem and "mask" not in f.stem
        ])

    if not geo_tifs:
        print("[ERROR] tiles/ 目录下未找到任何 GeoTIFF 文件")
        sys.exit(1)

    print(f"找到 {len(geo_tifs)} 张待检测的 GeoTIFF 文件")

    # ── 自动检测 GPU/CUDA ─────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n[GPU] GPU acceleration enabled: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = "cpu"
            print("\n[WARN] CUDA GPU not found, using CPU (slower)")
    except ImportError:
        device = "cpu"
        print("\n[WARN] PyTorch not installed, using CPU")

    # ── 路径 A：geoai SolarPanelDetector ──────────────────────────────
    try:
        import geoai

        print(f"\n[路径 A] 使用 geoai.SolarPanelDetector 进行检测 (device={device})...")
        detector = geoai.SolarPanelDetector(device=device)

        # 预加载建筑轮廓（避免每个 tile 重复加载）
        bldg_union_cache = None
        bldg_crs_cache = None
        if BUILDINGS_GPKG.exists():
            try:
                _buildings = gpd.read_file(str(BUILDINGS_GPKG))
                if len(_buildings) > 0:
                    bldg_utm = _buildings.to_crs(TARGET_CRS)
                    bldg_utm["geometry"] = bldg_utm.geometry.buffer(2)
                    bldg_crs_cache = _buildings.crs  # 原始 CRS
                    bldg_union_cache = {}  # 按 CRS 缓存
                    print(f"[建筑掩膜] 已加载 {len(_buildings)} 个建筑轮廓 (buffer=2m)")
                del _buildings
            except Exception as e:
                print(f"[WARN] 加载建筑轮廓失败: {e}")

        all_gdfs = []
        for idx, tif_path in enumerate(geo_tifs, 1):
            tile_name = tif_path.stem
            print(f"  [{idx}/{len(geo_tifs)}] 检测中: {tile_name}")

            try:
                # 生成掩膜 → 矢量化
                mask_path = _masks_dir / f"{tile_name}_mask.tif"
                masks_result = detector.generate_masks(
                    str(tif_path),
                    output_path=str(mask_path),
                    confidence_threshold=_confidence_threshold,
                    mask_threshold=_mask_threshold,
                    min_object_area=_min_object_area,
                    overlap=_overlap,
                    chip_size=_chip_size,
                    batch_size=BATCH_SIZE,
                    verbose=False,
                )

                # 矢量化：正交化多边形
                vector_path = _vectors_dir / f"{tile_name}_vectors.geojson"
                gdf_tile = geoai.orthogonalize(
                    input_path=masks_result,
                    output_path=str(vector_path),
                    epsilon=0.2,
                )

                if gdf_tile is not None and len(gdf_tile) > 0:
                    # --- 从 mask band 2 回填 confidence ---
                    try:
                        import rasterstats as _rs
                        _conf_stats = _rs.zonal_stats(
                            gdf_tile, str(mask_path), band=2,
                            stats=["mean"], nodata=0,
                        )
                        gdf_tile["confidence"] = [
                            (s["mean"] / 255.0) if s["mean"] is not None else 0.0
                            for s in _conf_stats
                        ]
                    except Exception as _e:
                        print(f"    [WARN] confidence 回填失败: {_e}")

                    # --- 颜色过滤：去除阴影和反光 ---
                    try:
                        import rasterstats
                        stats_r = rasterstats.zonal_stats(gdf_tile, str(tif_path), band=1, stats=['mean'], nodata=0)
                        stats_g = rasterstats.zonal_stats(gdf_tile, str(tif_path), band=2, stats=['mean'], nodata=0)
                        stats_b = rasterstats.zonal_stats(gdf_tile, str(tif_path), band=3, stats=['mean'], nodata=0)
                        gdf_tile["mean_r"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_r]
                        gdf_tile["mean_g"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_g]
                        gdf_tile["mean_b"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_b]

                        # 阴影过滤（RGB 三通道均低于阈值）+ 过曝过滤
                        is_shadow = ((gdf_tile["mean_r"] < SHADOW_RGB_THRESH)
                                     & (gdf_tile["mean_g"] < SHADOW_RGB_THRESH)
                                     & (gdf_tile["mean_b"] < SHADOW_RGB_THRESH))
                        is_too_bright = (gdf_tile["mean_r"] > 250) & (gdf_tile["mean_g"] > 250) & (gdf_tile["mean_b"] > 250)
                        valid_mask = ~(is_shadow | is_too_bright)

                        pre_count = len(gdf_tile)
                        gdf_tile = gdf_tile[valid_mask].copy()
                        if pre_count > len(gdf_tile):
                            print(f"    -> 颜色过滤: 移除了 {pre_count - len(gdf_tile)} 个极端阴影/反光")
                    except Exception as e:
                        print(f"    [WARN] RGB过滤失败: {e}")

                    # --- 建筑掩膜已禁用（OSM 数据滞后） ---

                    if len(gdf_tile) > 0:
                        # 添加几何属性用于后续过滤
                        gdf_tile = geoai.add_geometric_properties(gdf_tile)
                        # 添加来源 tile 信息
                        gdf_tile["source_tile"] = tile_name
                        all_gdfs.append(gdf_tile)
                        print(f"    → 检测到 {len(gdf_tile)} 个候选特征")
                    else:
                        print(f"    → 颜色过滤后无多边形剩余")
                else:
                    print(f"    → 未检测到太阳能板")

            except Exception as e:
                print(f"    [WARNING] 处理 {tile_name} 时出错: {e}")
                continue

        if not all_gdfs:
            print("[ERROR] 所有 tile 均未检测到太阳能板")
            sys.exit(1)

        # 合并所有检测结果
        pred_gdf = pd.concat(all_gdfs, ignore_index=True)

        # 空间 NMS 去重：chip 重叠导致同一目标被重复检测
        pred_gdf = spatial_nms(pred_gdf, iou_threshold=0.5)

        # 面积过滤
        pre_filter_count = len(pred_gdf)
        if "area_m2" in pred_gdf.columns:
            pred_gdf = pred_gdf[pred_gdf["area_m2"] >= _min_object_area].copy()

        # 长宽比过滤：去除细长条状误检
        if "elongation" in pred_gdf.columns and MAX_ELONGATION < 999:
            pred_gdf = pred_gdf[pred_gdf["elongation"] <= MAX_ELONGATION].copy()

        post_filter_count = len(pred_gdf)
        print(f"\n后处理过滤: {post_filter_count} / {pre_filter_count} 个多边形保留"
              f"（area>={_min_object_area}m² + elongation<={MAX_ELONGATION}）")

        # 确保有 confidence 字段
        if "confidence" not in pred_gdf.columns:
            # geoai orthogonalize 输出的字段可能叫 score 或 probability
            for alt_col in ["score", "probability", "prob", "conf"]:
                if alt_col in pred_gdf.columns:
                    pred_gdf["confidence"] = pred_gdf[alt_col]
                    break
            else:
                print("[INFO] 未找到置信度字段，使用默认值 0.5")
                pred_gdf["confidence"] = 0.5

        # 置信度过滤：去除低置信度预测
        pre_conf_count = len(pred_gdf)
        pred_gdf = pred_gdf[pred_gdf["confidence"] >= POST_CONF_THRESHOLD].copy()
        print(f"置信度过滤: {len(pred_gdf)} / {pre_conf_count} 个多边形保留"
              f"（confidence>={POST_CONF_THRESHOLD}）")

        pred_gdf.to_file(str(_predictions_path), driver="GeoJSON")
        print(f"\n[OK] predictions saved: {_predictions_path}")
        print(f"    总计 {len(pred_gdf)} 个太阳能板检测多边形")
        return pred_gdf

    except ImportError:
        print("[INFO] geoai 包未安装, 尝试路径 B...")
    except Exception as e:
        print(f"[WARNING] geoai 路径 A 出错: {e}\n尝试路径 B...")

    # ── 路径 B：samgeo SAM 模型备选 ──────────────────────────────────
    try:
        from samgeo import SamGeo

        print("\n[路径 B] 使用 samgeo (SAM) + 文字提示 'solar panel' 进行检测...")
        sam = SamGeo(
            model_type="vit_h",
            automatic=False,
            device=device,
        )

        all_gdfs = []
        for idx, tif_path in enumerate(geo_tifs, 1):
            tile_name = tif_path.stem
            print(f"  [{idx}/{len(geo_tifs)}] 检测中: {tile_name}")

            try:
                mask_path = _masks_dir / f"{tile_name}_sam_mask.tif"
                vector_path = _vectors_dir / f"{tile_name}_sam_vectors.geojson"

                sam.set_image(str(tif_path))
                sam.text_predict(
                    text="solar panel",
                    output=str(mask_path),
                    box_threshold=0.24,
                    text_threshold=0.24,
                )
                sam.raster_to_vector(str(mask_path), str(vector_path))

                gdf_tile = gpd.read_file(str(vector_path))
                if len(gdf_tile) > 0:
                    gdf_tile["source_tile"] = tile_name
                    gdf_tile["confidence"] = 0.5  # SAM 不提供逐对象置信度
                    all_gdfs.append(gdf_tile)
                    print(f"    → 检测到 {len(gdf_tile)} 个候选多边形")
                else:
                    print(f"    → 未检测到太阳能板")
            except Exception as e:
                print(f"    [WARNING] 处理 {tile_name} 时出错: {e}")
                continue

        if not all_gdfs:
            print("[ERROR] samgeo 路径也未检测到任何太阳能板")
            sys.exit(1)

        pred_gdf = pd.concat(all_gdfs, ignore_index=True)
        pred_gdf.to_file(str(_predictions_path), driver="GeoJSON")
        print(f"\n[OK] predictions saved: {_predictions_path}")
        return pred_gdf

    except ImportError:
        print("[FATAL] geoai 和 samgeo 均未安装，无法执行检测")
        print("  请运行:  pip install geoai-py")
        print("  或:      pip install segment-geospatial")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# 第二步：加载真值数据
# ════════════════════════════════════════════════════════════════════════
def load_ground_truth() -> gpd.GeoDataFrame:
    """加载真值多边形，统一投影到 UTM"""
    print("\n" + "=" * 60)
    print("加载真值数据 (Ground Truth)...")

    gt = None
    if GT_GPKG.exists():
        try:
            import pyogrio
            layers = pyogrio.list_layers(str(GT_GPKG))
            if len(layers) > 1:
                # 多图层时，读取多边形数量最多的图层
                best_layer, best_count = None, 0
                for layer_name, _ in layers:
                    gdf_tmp = gpd.read_file(str(GT_GPKG), layer=layer_name)
                    print(f"  图层 '{layer_name}': {len(gdf_tmp)} 个多边形")
                    if len(gdf_tmp) > best_count:
                        best_layer, best_count = layer_name, len(gdf_tmp)
                        gt = gdf_tmp
                print(f"  → 选择图层 '{best_layer}' ({best_count} 个多边形)")
            else:
                gt = gpd.read_file(str(GT_GPKG))
                print(f"  已加载 GPKG: {GT_GPKG.name} ({len(gt)} 个多边形)")
        except Exception as e:
            print(f"  [WARNING] 读取 GPKG 失败: {e}")

    if gt is None and GT_GEOJSON.exists():
        try:
            gt = gpd.read_file(str(GT_GEOJSON))
            print(f"  已加载 GeoJSON: {GT_GEOJSON.name} ({len(gt)} 个多边形)")
        except Exception as e:
            print(f"  [ERROR] 读取 GeoJSON 也失败: {e}")
            sys.exit(1)

    if gt is None:
        print("[ERROR] 未找到任何真值文件 (g1238.gpkg 或 g1238.geojson)")
        sys.exit(1)

    # 统一投影
    if gt.crs is None:
        print("  [INFO] 真值数据无 CRS，假设为 EPSG:4326")
        gt = gt.set_crs("EPSG:4326")
    gt = gt.to_crs(TARGET_CRS)

    # 确保都是有效的几何体
    gt = gt[gt.geometry.notnull() & gt.is_valid].copy()
    print(f"  统一投影到 {TARGET_CRS}，有效多边形: {len(gt)} 个")
    return gt


def load_predictions() -> gpd.GeoDataFrame:
    """加载预测结果并统一投影"""
    if not PREDICTIONS_PATH.exists():
        print(f"[ERROR] 预测文件不存在: {PREDICTIONS_PATH}")
        sys.exit(1)

    pred = gpd.read_file(str(PREDICTIONS_PATH))
    print(f"  已加载预测结果: {len(pred)} 个多边形")

    if pred.crs is None:
        pred = pred.set_crs("EPSG:4326")
    pred = pred.to_crs(TARGET_CRS)

    pred = pred[pred.geometry.notnull() & pred.is_valid].copy()
    return pred


# ════════════════════════════════════════════════════════════════════════
# 第三步：IoU 匹配与评估
# ════════════════════════════════════════════════════════════════════════
def compute_iou(geom_a, geom_b) -> float:
    """计算两个几何体的交并比 (IoU)"""
    try:
        if geom_a.is_empty or geom_b.is_empty:
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        union = geom_a.area + geom_b.area - intersection
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return 0.0


def iou_matching(gt: gpd.GeoDataFrame,
                 pred: gpd.GeoDataFrame,
                 iou_threshold: float = 0.3,
                 merge_preds: bool = True,
                 ) -> dict:
    """
    基于空间 IoU 的匹配，支持两种模式：

    merge_preds=False（严格一对一模式）:
      对每个 GT 多边形，找到 IoU 最高的单个预测多边形匹配。

    merge_preds=True（多对一合并模式，默认）:
      对每个 GT 多边形，将所有与之相交的预测多边形 union 合并后再计算 IoU。
      适用于标注者用一个大多边形覆盖屋顶上多组面板，而检测器将其拆分为
      多个小多边形的情况。

    返回:
      {
        "tp": int, "fp": int, "fn": int,
        "precision": float, "recall": float, "f1": float,
        "matched_pred_indices": set,
        "matched_gt_indices": set,
        "iou_scores": list  # 每个 TP 对应的 IoU 值
      }
    """
    pred_sindex = pred.sindex  # 空间索引
    matched_pred = set()
    matched_gt = set()
    iou_scores = []

    if merge_preds:
        # ── 多对一合并模式 ────────────────────────────────────────────
        # 对每个 GT，找出所有与之相交的 pred，合并后计算 IoU
        gt_match_results = []  # (gt_idx, merged_iou, pred_indices_set)

        for gt_idx, gt_row in gt.iterrows():
            gt_geom = gt_row.geometry
            # 空间索引粗筛
            candidate_idxs = list(pred_sindex.intersection(gt_geom.bounds))

            # 精筛：只保留真正相交的
            intersecting_idxs = []
            for pidx in candidate_idxs:
                pred_geom = pred.iloc[pidx].geometry
                try:
                    if gt_geom.intersects(pred_geom):
                        intersecting_idxs.append(pidx)
                except Exception:
                    continue

            if not intersecting_idxs:
                continue

            # 合并所有相交的预测多边形
            merged_pred_geom = unary_union(
                [pred.iloc[pidx].geometry for pidx in intersecting_idxs]
            )

            iou_val = compute_iou(gt_geom, merged_pred_geom)
            if iou_val >= iou_threshold:
                gt_match_results.append(
                    (gt_idx, iou_val, set(intersecting_idxs))
                )

        # 按 IoU 降序处理（贪心），避免 pred 被重复分配
        gt_match_results.sort(key=lambda x: x[1], reverse=True)

        for gt_idx, iou_val, pidx_set in gt_match_results:
            if gt_idx in matched_gt:
                continue
            # 检查是否有至少一个 pred 尚未被分配
            available = pidx_set - matched_pred
            if not available:
                continue
            matched_gt.add(gt_idx)
            matched_pred.update(pidx_set)  # 所有参与合并的 pred 都标记为已匹配
            iou_scores.append(iou_val)

    else:
        # ── 严格一对一模式 ────────────────────────────────────────────
        candidate_pairs = []

        for gt_idx, gt_row in gt.iterrows():
            gt_geom = gt_row.geometry
            candidate_idxs = list(pred_sindex.intersection(gt_geom.bounds))
            for pred_idx in candidate_idxs:
                pred_geom = pred.iloc[pred_idx].geometry
                iou_val = compute_iou(gt_geom, pred_geom)
                if iou_val >= iou_threshold:
                    candidate_pairs.append((gt_idx, pred_idx, iou_val))

        candidate_pairs.sort(key=lambda x: x[2], reverse=True)

        for gt_idx, pred_idx, iou_val in candidate_pairs:
            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                iou_scores.append(iou_val)

    tp = len(matched_gt)
    fn = len(gt) - tp
    fp = len(pred) - len(matched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_pred_indices": matched_pred,
        "matched_gt_indices": matched_gt,
        "iou_scores": iou_scores,
    }


def evaluate_at_multiple_thresholds(gt: gpd.GeoDataFrame,
                                     pred: gpd.GeoDataFrame,
                                     merge_preds: bool = True,
                                     ) -> pd.DataFrame:
    """在多个 IoU 阈值下计算评估指标"""
    results = []
    for iou_thr in IOU_THRESHOLDS:
        metrics = iou_matching(gt, pred, iou_threshold=iou_thr,
                               merge_preds=merge_preds)
        results.append({
            "IoU_Threshold": iou_thr,
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
        })
    return pd.DataFrame(results)


def classify_predictions(gt: gpd.GeoDataFrame,
                          pred: gpd.GeoDataFrame,
                          iou_threshold: float = 0.3
                          ) -> gpd.GeoDataFrame:
    """
    为每个预测多边形标记 TP/FP 分类，以便后续置信度分析。
    """
    pred = pred.copy()
    metrics = iou_matching(gt, pred, iou_threshold=iou_threshold)
    pred["is_tp"] = False
    for pred_idx in metrics["matched_pred_indices"]:
        pred.at[pred.index[pred_idx], "is_tp"] = True
    return pred


# ════════════════════════════════════════════════════════════════════════
# 第四步：逐 Tile 统计
# ════════════════════════════════════════════════════════════════════════
def evaluate_per_tile(gt: gpd.GeoDataFrame,
                      pred: gpd.GeoDataFrame
                      ) -> pd.DataFrame:
    """
    将评估结果按 tile 分解统计。
    使用 tiles/ 的网格参数重建每个 tile 的边界框。
    """
    import rasterio

    geo_tifs = sorted(TILES_DIR.glob(f"{GRID_ID}_*_*_geo.tif"))
    if not geo_tifs:
        geo_tifs = sorted([
            f for f in TILES_DIR.glob(f"{GRID_ID}_*_*.tif")
            if "_geo" not in f.stem and "mosaic" not in f.stem
        ])

    rows = []
    for tif_path in geo_tifs:
        tile_name = tif_path.stem
        try:
            with rasterio.open(str(tif_path)) as src:
                tile_bounds = src.bounds
                tile_crs    = src.crs

            # 创建 tile 边界多边形
            tile_box = gpd.GeoDataFrame(
                geometry=[box(tile_bounds.left, tile_bounds.bottom,
                              tile_bounds.right, tile_bounds.top)],
                crs=tile_crs,
            ).to_crs(TARGET_CRS).geometry[0]

            # 筛选落入该 tile 的 GT 和 Pred
            gt_in_tile   = gt[gt.geometry.intersects(tile_box)]
            pred_in_tile = pred[pred.geometry.intersects(tile_box)]

            if len(gt_in_tile) == 0 and len(pred_in_tile) == 0:
                rows.append({
                    "tile": tile_name,
                    "gt_count": 0, "pred_count": 0,
                    "TP": 0, "FP": 0, "FN": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                })
                continue

            gt_in_tile   = gt_in_tile.reset_index(drop=True)
            pred_in_tile = pred_in_tile.reset_index(drop=True)

            if len(gt_in_tile) > 0 and len(pred_in_tile) > 0:
                m = iou_matching(gt_in_tile, pred_in_tile, iou_threshold=DEFAULT_IOU)
            elif len(pred_in_tile) > 0:
                m = {"tp": 0, "fp": len(pred_in_tile), "fn": 0,
                     "precision": 0.0, "recall": 0.0, "f1": 0.0}
            else:
                m = {"tp": 0, "fp": 0, "fn": len(gt_in_tile),
                     "precision": 0.0, "recall": 0.0, "f1": 0.0}

            rows.append({
                "tile": tile_name,
                "gt_count": len(gt_in_tile),
                "pred_count": len(pred_in_tile),
                "TP": m["tp"], "FP": m["fp"], "FN": m["fn"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            })
        except Exception as e:
            print(f"  [WARNING] 处理 {tile_name} 逐 tile 统计时出错: {e}")
            rows.append({"tile": tile_name, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(str(EVALUATION_CSV_PATH), index=False, encoding="utf-8-sig")
    print(f"\n[OK] per-tile evaluation saved: {EVALUATION_CSV_PATH}")
    return df


# ════════════════════════════════════════════════════════════════════════
# 第五步：可视化
# ════════════════════════════════════════════════════════════════════════
def set_plot_style():
    """设置统一的图表风格"""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
    })


def plot_confidence_histogram(pred_classified: gpd.GeoDataFrame):
    """
    图1: 置信度直方图，TP/FP 分色
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    tp_conf = pred_classified.loc[pred_classified["is_tp"], "confidence"]
    fp_conf = pred_classified.loc[~pred_classified["is_tp"], "confidence"]

    bins = np.linspace(0, 1, 25)

    ax.hist(tp_conf, bins=bins, alpha=0.7, color="#2ecc71", edgecolor="white",
            label=f"TP ({len(tp_conf)})", zorder=3)
    ax.hist(fp_conf, bins=bins, alpha=0.7, color="#e74c3c", edgecolor="white",
            label=f"FP ({len(fp_conf)})", zorder=2)

    ax.set_xlabel("Confidence Score", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Confidence Distribution: TP vs FP", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(str(CONFIDENCE_HIST_PATH))
    plt.close(fig)
    print(f"  [OK] saved: {CONFIDENCE_HIST_PATH.name}")


def plot_precision_recall_curve(gt: gpd.GeoDataFrame,
                                 pred: gpd.GeoDataFrame):
    """
    图2: 以置信度阈值为横轴的 Precision-Recall 曲线
    """
    set_plot_style()

    if "confidence" not in pred.columns:
        print("  [SKIP] 无 confidence 字段，跳过 PR 曲线")
        return

    conf_thresholds = np.arange(0.05, 1.0, 0.05)
    precisions = []
    recalls = []

    for conf_thr in conf_thresholds:
        pred_filtered = pred[pred["confidence"] >= conf_thr].reset_index(drop=True)
        if len(pred_filtered) == 0:
            precisions.append(1.0)  # 无预测 → 精度为 1（无误检）
            recalls.append(0.0)    # 无预测 → 召回为 0
            continue
        m = iou_matching(gt, pred_filtered, iou_threshold=DEFAULT_IOU)
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(conf_thresholds, precisions, "o-", color="#3498db",
            linewidth=2, markersize=5, label="Precision")
    ax.plot(conf_thresholds, recalls, "s-", color="#e67e22",
            linewidth=2, markersize=5, label="Recall")

    ax.set_xlabel("Confidence Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Precision & Recall vs Confidence Threshold (IoU={DEFAULT_IOU})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(str(PR_CURVE_PATH))
    plt.close(fig)
    print(f"  [OK] saved: {PR_CURVE_PATH.name}")


def plot_iou_threshold_metrics(metrics_df: pd.DataFrame):
    """
    图3: IoU 阈值 vs Precision / Recall / F1
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics_df["IoU_Threshold"], metrics_df["Precision"],
            "o-", color="#2ecc71", linewidth=2.5, markersize=8, label="Precision")
    ax.plot(metrics_df["IoU_Threshold"], metrics_df["Recall"],
            "s-", color="#3498db", linewidth=2.5, markersize=8, label="Recall")
    ax.plot(metrics_df["IoU_Threshold"], metrics_df["F1"],
            "D-", color="#9b59b6", linewidth=2.5, markersize=8, label="F1 Score")

    ax.set_xlabel("IoU Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Detection Metrics at Different IoU Thresholds",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(-0.05, 1.05)

    # 添加数据标注
    for _, row in metrics_df.iterrows():
        ax.annotate(f"{row['F1']:.2f}",
                     xy=(row["IoU_Threshold"], row["F1"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="#9b59b6")

    fig.tight_layout()
    fig.savefig(str(IOU_METRICS_PATH))
    plt.close(fig)
    print(f"  [OK] saved: {IOU_METRICS_PATH.name}")


# ════════════════════════════════════════════════════════════════════════
# 误检/漏检分类分析
# ════════════════════════════════════════════════════════════════════════
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.csv"
FN_ANALYSIS_PATH    = OUTPUT_DIR / "fn_analysis.csv"


def analyze_errors(gt: gpd.GeoDataFrame,
                   pred: gpd.GeoDataFrame,
                   pred_classified: gpd.GeoDataFrame) -> None:
    """
    对 FP 和 FN 进行分类分析，输出 CSV 和控制台汇总。
    FP 分类基于 RGB 均值和几何属性；FN 按面积分档。
    """
    print("\n" + "=" * 60)
    print("误检/漏检分类分析 (Error Analysis)...")

    # ── FP 分类 ─────────────────────────────────────────────────────
    fp = pred_classified[~pred_classified["is_tp"]].copy()

    fp["error_type"] = "other"

    # 1) 极暗 → shadow
    is_shadow = (fp["mean_r"] < 50) & (fp["mean_g"] < 50) & (fp["mean_b"] < 50)
    fp.loc[is_shadow, "error_type"] = "shadow"

    # 2) 偏暗 → dark_object
    is_dark = ((fp["mean_r"] < 70) & (fp["mean_g"] < 70) & (fp["mean_b"] < 70)
               & (fp["error_type"] == "other"))
    fp.loc[is_dark, "error_type"] = "dark_object"

    # 3) 蓝色偏高 → blue_object（泳池/蓝车/蓝屋顶）
    has_blue = (fp["mean_b"] > fp["mean_r"] * 1.3) & (fp["error_type"] == "other")
    fp.loc[has_blue, "error_type"] = "blue_object"

    # 4) 面积过小 → small_fragment
    is_small = (fp["area_m2"] < 3) & (fp["error_type"] == "other")
    fp.loc[is_small, "error_type"] = "small_fragment"

    # 5) 过于细长 → elongated
    is_elongated = (fp["elongation"] > 4) & (fp["error_type"] == "other")
    fp.loc[is_elongated, "error_type"] = "elongated"

    # 保存 FP 分析
    fp_out_cols = ["source_tile", "error_type", "mean_r", "mean_g", "mean_b",
                   "area_m2", "elongation", "solidity"]
    fp_out = fp[[c for c in fp_out_cols if c in fp.columns]].copy()
    fp_out.to_csv(str(ERROR_ANALYSIS_PATH), index=False, encoding="utf-8-sig")

    print(f"\n  FP 分类统计 ({len(fp)} 个误检):")
    for etype, count in fp["error_type"].value_counts().items():
        pct = count / len(fp) * 100
        print(f"    {etype:20s}: {count:3d} ({pct:.1f}%)")

    # ── FN 分析 ─────────────────────────────────────────────────────
    metrics = iou_matching(gt, pred, iou_threshold=DEFAULT_IOU)
    fn_gt = gt.loc[~gt.index.isin(metrics["matched_gt_indices"])].copy()

    if len(fn_gt) > 0:
        # 计算面积（投影到 UTM 34S）
        fn_utm = fn_gt.to_crs("EPSG:32734")
        fn_gt["area_m2"] = fn_utm.geometry.area

        fn_gt["size_class"] = pd.cut(
            fn_gt["area_m2"],
            bins=[0, 5, 20, 50, float("inf")],
            labels=["<5m2", "5-20m2", "20-50m2", ">50m2"],
        )

        fn_out = fn_gt[["area_m2", "size_class"]].copy()
        fn_out.to_csv(str(FN_ANALYSIS_PATH), index=False, encoding="utf-8-sig")

        print(f"\n  FN 面积分布 ({len(fn_gt)} 个漏检):")
        for sc, count in fn_gt["size_class"].value_counts().sort_index().items():
            print(f"    {sc:10s}: {count:3d}")
        print(f"    mean area: {fn_gt['area_m2'].mean():.1f} m2, "
              f"median: {fn_gt['area_m2'].median():.1f} m2")
    else:
        print("\n  FN: 0 个漏检")

    print(f"\n  [OK] saved: {ERROR_ANALYSIS_PATH.name}, {FN_ANALYSIS_PATH.name}")


# ════════════════════════════════════════════════════════════════════════
# 第六步：汇总报告
# ════════════════════════════════════════════════════════════════════════
def print_report(gt: gpd.GeoDataFrame,
                 pred: gpd.GeoDataFrame,
                 pred_classified: gpd.GeoDataFrame,
                 metrics_df: pd.DataFrame):
    """输出格式化评估报告"""

    default_metrics = iou_matching(gt, pred, iou_threshold=DEFAULT_IOU)

    tp_conf = pred_classified.loc[pred_classified["is_tp"], "confidence"]
    fp_conf = pred_classified.loc[~pred_classified["is_tp"], "confidence"]

    report = f"""
{'=' * 50}
  太阳能板检测评估报告
  Solar Panel Detection Evaluation Report
{'=' * 50}
真值多边形总数     : {len(gt)}
预测多边形总数     : {len(pred)}
{'─' * 50}"""

    for _, row in metrics_df.iterrows():
        report += f"""
IoU 阈值 = {row['IoU_Threshold']}:
  TP              : {row['TP']}
  FP              : {row['FP']}
  FN              : {row['FN']}
  Precision       : {row['Precision']:.4f}
  Recall          : {row['Recall']:.4f}
  F1 Score        : {row['F1']:.4f}
{'─' * 50}"""

    tp_mean = f"{tp_conf.mean():.4f}" if len(tp_conf) > 0 else "N/A"
    fp_mean = f"{fp_conf.mean():.4f}" if len(fp_conf) > 0 else "N/A"
    tp_std  = f"{tp_conf.std():.4f}" if len(tp_conf) > 1 else "N/A"
    fp_std  = f"{fp_conf.std():.4f}" if len(fp_conf) > 1 else "N/A"

    report += f"""
confidence stats (IoU={DEFAULT_IOU}):
  mean confidence (TP) : {tp_mean}
  mean confidence (FP) : {fp_mean}
  std confidence  (TP) : {tp_std}
  std confidence  (FP) : {fp_std}
{'=' * 50}

输出文件：
  - {PREDICTIONS_PATH}
  - {CONFIDENCE_HIST_PATH}
  - {PR_CURVE_PATH}
  - {IOU_METRICS_PATH}
  - {EVALUATION_CSV_PATH}
"""

    print(report)

    # 同时保存为文本文件
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] report saved: {report_path}")


# ════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════
def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  太阳能板检测 & 评估流水线                             ║")
    print("║  Solar Panel Detection & Evaluation Pipeline          ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    # ── Step 1: 检测 ──────────────────────────────────────────────────
    if PREDICTIONS_PATH.exists():
        print(f"[INFO] 已存在预测文件 {PREDICTIONS_PATH.name}，跳过检测步骤")
        print("       若需重新检测，请删除该文件后重新运行")
        pred = load_predictions()
    else:
        pred = detect_solar_panels()
        pred = load_predictions()  # 重新加载以确保 CRS 统一

    # ── Step 2: 加载真值 ──────────────────────────────────────────────
    gt = load_ground_truth()

    # ── Step 3: 多阈值评估（两种模式对比） ───────────────────────────
    print("\n" + "=" * 60)
    print("IoU 多阈值评估...")
    print("\n>> 模式 1: 多对一合并匹配 (merge_preds=True)")
    print("   适配标注风格: 一个 GT 多边形覆盖多组面板")
    metrics_df = evaluate_at_multiple_thresholds(gt, pred)
    print("\n" + metrics_df.to_string(index=False))

    print("\n>> 模式 2: 严格一对一匹配 (merge_preds=False)")
    print("   传统模式: 每个预测只能匹配一个 GT")
    metrics_df_strict = evaluate_at_multiple_thresholds(
        gt, pred, merge_preds=False
    )
    print("\n" + metrics_df_strict.to_string(index=False))

    # ── Step 4: 分类预测 (TP/FP)（使用合并模式） ─────────────────────
    pred_classified = classify_predictions(gt, pred, iou_threshold=DEFAULT_IOU)

    # ── Step 4b: 误检/漏检分类分析 ────────────────────────────────────
    analyze_errors(gt, pred, pred_classified)

    # ── Step 5: 逐 Tile 评估 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("逐 Tile 评估（合并匹配模式）...")
    tile_df = evaluate_per_tile(gt, pred)
    if "gt_count" in tile_df.columns:
        non_empty = tile_df[tile_df["gt_count"] > 0]
        if len(non_empty) > 0:
            print(f"\n有太阳能板的 tile 数: {len(non_empty)}")
            print(non_empty[["tile", "gt_count", "pred_count", "TP", "FP", "FN",
                             "precision", "recall", "f1"]].to_string(index=False))

    # ── Step 6: 可视化（基于合并模式指标） ────────────────────────────
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    plot_confidence_histogram(pred_classified)
    plot_precision_recall_curve(gt, pred)
    plot_iou_threshold_metrics(metrics_df)

    # ── Step 7: 最终报告 ──────────────────────────────────────────────
    print_report(gt, pred, pred_classified, metrics_df)

    print("\n[DONE] Pipeline finished!")


if __name__ == "__main__":
    main()
