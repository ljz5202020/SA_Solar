"""
标注数据清洗脚本
Annotation Cleanup Script

功能：
1. 合并同一安装的多面板标注（同 Name → union）
2. 统一 ID 体系：{GridID}_{三位序号}
3. 按 Grid 分离 GPKG
4. 添加 land_use 字段（residential / commercial_industrial / unknown）
5. 输出清洗后的独立 GPKG 文件

用法：
    python scripts/annotations/clean_annotations.py [--dry-run]
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
OUTPUT_DIR = ANNOTATIONS_DIR / "cleaned"

# ════════════════════════════════════════════════════════════════════════
# 数据源定义
# ════════════════════════════════════════════════════════════════════════

# G1238: QGIS 标注，已是 installation-level
G1238_SOURCE = {
    "file": ANNOTATIONS_DIR / "G1238.gpkg",
    "layer": "g1238__solar_panel__cape_town_g1238_",
    "grid_id": "G1238",
    "merge_needed": False,  # 已是 installation-level
}

# Google Earth 批量导出：多网格混合，panel-level 需合并
GE_SOURCE = {
    "file": ANNOTATIONS_DIR / "solarpanel_g0001_g1190.gpkg",
    "layer": None,  # default layer
    "merge_needed": True,  # 同 Name 的多面板需 union
}

# JHB: 独立 GPKG，已是 installation-level
JHB_GRIDS = [f"JHB{i:02d}" for i in range(1, 7)]


# ════════════════════════════════════════════════════════════════════════
# SSEG 行政数据：用于匹配 land_use
# ════════════════════════════════════════════════════════════════════════

def load_sseg_points() -> gpd.GeoDataFrame:
    """加载 SSEG 登记点数据，用于 land_use 分类"""
    csv_path = BASE_DIR / "data" / "sseg_registration_geo.csv"
    if not csv_path.exists():
        print("[WARN] SSEG CSV not found, land_use will default to 'unknown'")
        return None

    df = pd.read_csv(csv_path)
    # 过滤有效坐标
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # 简化 customer_type → land_use
    gdf["land_use"] = gdf["customer_type"].map({
        "Residential": "residential",
        "Commercial/Industrial": "commercial_industrial",
    }).fillna("residential")

    # 只保留已安装的记录
    commissioned = gdf["status"].str.contains("commissioned", case=False, na=False)
    gdf = gdf[commissioned].copy()

    print(f"[SSEG] Loaded {len(gdf)} commissioned points "
          f"(residential: {(gdf['land_use']=='residential').sum()}, "
          f"commercial: {(gdf['land_use']=='commercial_industrial').sum()})")
    return gdf[["land_use", "geometry"]]


def assign_land_use(annotations: gpd.GeoDataFrame, sseg: gpd.GeoDataFrame,
                    buffer_m: float = 150.0) -> gpd.GeoDataFrame:
    """
    基于 SSEG 点数据为标注分配 land_use 字段。
    在投影坐标系下做 buffer 匹配（默认 150m，考虑 SSEG 坐标精度约 111m）。

    注意：SSEG 数据仅覆盖开普敦部分区域（lat > -34.14），南部半岛和 JHB 无覆盖。
    无法匹配的标注默认为 'unknown'，可后续在 QGIS 中手动编辑 land_use 字段。
    """
    annotations["land_use"] = "residential"

    if sseg is None or len(sseg) == 0:
        return annotations

    # 确定投影坐标系（根据经度粗略判断 UTM 带）
    centroid_lon = annotations.geometry.centroid.x.mean()
    if centroid_lon > 25:  # JHB — SSEG 无覆盖
        return annotations
    metric_crs = "EPSG:32734"

    # 粗筛：标注区域 bbox 扩大 500m 后是否与 SSEG 有交集
    ann_proj = annotations.to_crs(metric_crs)
    sseg_proj = sseg.to_crs(metric_crs)
    ann_bounds = ann_proj.total_bounds  # xmin, ymin, xmax, ymax
    sseg_in_region = sseg_proj.cx[
        ann_bounds[0] - 500 : ann_bounds[2] + 500,
        ann_bounds[1] - 500 : ann_bounds[3] + 500,
    ]
    if len(sseg_in_region) == 0:
        return annotations

    # 为每个标注创建 buffer，与 SSEG 点做空间关联
    ann_buffered = ann_proj.copy()
    ann_buffered["geometry"] = ann_proj.geometry.centroid.buffer(buffer_m)

    joined = gpd.sjoin(ann_buffered, sseg_in_region, how="left", predicate="contains")

    # 如果多个 SSEG 点匹配同一标注，取众数
    land_use_votes = joined.groupby(joined.index)["land_use"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"
    )

    annotations["land_use"] = land_use_votes.reindex(annotations.index).fillna("residential")
    matched = (annotations["land_use"] != "unknown").sum()
    if matched > 0:
        print(f"    SSEG matched {matched}/{len(annotations)} annotations")
    return annotations


# ════════════════════════════════════════════════════════════════════════
# 清洗逻辑
# ════════════════════════════════════════════════════════════════════════

def process_g1238() -> gpd.GeoDataFrame:
    """处理 G1238：已是 installation-level，只需统一 ID"""
    src = G1238_SOURCE
    gdf = gpd.read_file(src["file"], layer=src["layer"])
    print(f"[G1238] Loaded {len(gdf)} polygons from {src['layer']}")

    # 统一字段
    gdf = gdf.sort_values("panel_id").reset_index(drop=True)
    gdf["annotation_id"] = [f"G1238_{i+1:03d}" for i in range(len(gdf))]
    gdf["grid_id"] = "G1238"
    gdf["source"] = "qgis_aerial"

    # 去掉 Z 坐标（如有）
    gdf["geometry"] = gdf.geometry.map(
        lambda g: gpd.GeoSeries([g]).set_crs(gdf.crs).geometry.iloc[0]
    )

    gdf["num_parts"] = 1  # G1238 已是 installation-level，每个都是单多边形

    return gdf[["annotation_id", "grid_id", "source", "num_parts", "geometry"]]


def process_google_earth() -> dict[str, gpd.GeoDataFrame]:
    """处理 Google Earth 导出：按 Grid 分离，同 Name 合并"""
    src = GE_SOURCE
    gdf = gpd.read_file(src["file"])
    print(f"[GE] Loaded {len(gdf)} polygons from {src['file'].name}")

    # 提取 grid_id from Name (e.g., "G1189_001" → "G1189")
    gdf["grid_id_parsed"] = gdf["Name"].str.extract(r"(G\d+)")[0]
    grids = gdf["grid_id_parsed"].dropna().unique()

    results = {}
    for grid_id in sorted(grids):
        subset = gdf[gdf["grid_id_parsed"] == grid_id].copy()

        if src["merge_needed"]:
            # 按 Name 分组，合并同一安装的多面板
            merged_rows = []
            for name, group in subset.groupby("Name"):
                if len(group) > 1:
                    merged_geom = unary_union(group.geometry)
                    print(f"  [{grid_id}] Merged {len(group)} panels for {name}")
                else:
                    merged_geom = group.geometry.iloc[0]
                merged_rows.append({"name_orig": name, "geometry": merged_geom})

            merged = gpd.GeoDataFrame(merged_rows, crs=gdf.crs)
        else:
            merged = subset[["Name", "geometry"]].rename(columns={"Name": "name_orig"})

        # 统一 ID
        merged = merged.reset_index(drop=True)
        merged["annotation_id"] = [f"{grid_id}_{i+1:03d}" for i in range(len(merged))]
        merged["grid_id"] = grid_id
        merged["source"] = "google_earth"

        # 记录每个安装包含的面板数
        merged["num_parts"] = merged.geometry.apply(
            lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1
        )

        results[grid_id] = merged[["annotation_id", "grid_id", "source", "num_parts", "geometry"]]
        multi = merged[merged["num_parts"] > 1]
        if len(multi) > 0:
            print(f"  [{grid_id}] {len(multi)} installations have multiple parts")
        print(f"  [{grid_id}] {len(subset)} panels → {len(merged)} installations")

    return results


def process_jhb() -> dict[str, gpd.GeoDataFrame]:
    """处理 JHB 标注：已是 installation-level，只需统一 ID"""
    results = {}
    for grid_id in JHB_GRIDS:
        fpath = ANNOTATIONS_DIR / f"{grid_id}.gpkg"
        if not fpath.exists():
            continue
        gdf = gpd.read_file(fpath)
        if len(gdf) == 0:
            continue

        gdf = gdf.sort_values("panel_id").reset_index(drop=True)
        gdf["annotation_id"] = [f"{grid_id}_{i+1:03d}" for i in range(len(gdf))]
        gdf["grid_id"] = grid_id
        gdf["source"] = "google_earth"

        gdf["num_parts"] = 1  # JHB 已是 installation-level

        results[grid_id] = gdf[["annotation_id", "grid_id", "source", "num_parts", "geometry"]]
        print(f"[{grid_id}] {len(gdf)} installations (already installation-level)")

    return results


# ════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clean and unify annotation data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 SSEG 数据
    sseg = load_sseg_points()

    # 收集所有网格
    all_grids: dict[str, gpd.GeoDataFrame] = {}

    # 1. G1238
    all_grids["G1238"] = process_g1238()

    # 2. Google Earth 导出（多网格）
    ge_grids = process_google_earth()
    all_grids.update(ge_grids)

    # 3. JHB
    jhb_grids = process_jhb()
    all_grids.update(jhb_grids)

    # 4. 分配 land_use
    print("\n[land_use] Assigning land use from SSEG data...")
    for grid_id, gdf in all_grids.items():
        all_grids[grid_id] = assign_land_use(gdf, sseg)
        counts = gdf["land_use"].value_counts().to_dict() if "land_use" in gdf.columns else {}
        # Re-read after assign
        counts = all_grids[grid_id]["land_use"].value_counts().to_dict()
        print(f"  [{grid_id}] {counts}")

    # 5. 汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    for grid_id in sorted(all_grids.keys()):
        gdf = all_grids[grid_id]
        n = len(gdf)
        total += n
        lu = gdf["land_use"].value_counts().to_dict()
        print(f"  {grid_id}: {n:>4d} installations  {lu}")
    print(f"  {'TOTAL':>6s}: {total:>4d} installations")

    # 6. 写出
    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    for grid_id, gdf in all_grids.items():
        out_path = OUTPUT_DIR / f"{grid_id}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        print(f"  Written: {out_path} ({len(gdf)} features)")

    # 7. 写合并版本（方便查看）
    all_combined = pd.concat(all_grids.values(), ignore_index=True)
    all_combined = gpd.GeoDataFrame(all_combined, crs="EPSG:4326")
    combined_path = OUTPUT_DIR / "all_annotations_cleaned.gpkg"
    all_combined.to_file(combined_path, driver="GPKG")
    print(f"  Written: {combined_path} ({len(all_combined)} features)")

    print("\n[DONE] Cleanup complete!")


if __name__ == "__main__":
    main()
