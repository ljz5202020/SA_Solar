"""
建筑轮廓预处理：下载 + 合并 + Tile 过滤清单
Building Footprint Preprocessing: Download, Merge, Tile Manifest

数据源：
  1. OSM (via osmnx) — 优先
  2. Microsoft Global ML Building Footprints — 补充
合并策略：Microsoft 为底，OSM 中不重叠的建筑补充进来

输出：
  - buildings.gpkg  — 合并后的建筑轮廓
  - tile_manifest.csv — 每个 tile 是否有建筑

依赖：pip install osmnx geopandas requests
"""

import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
BUILDINGS_GPKG = BASE_DIR / "buildings.gpkg"
TILE_MANIFEST = BASE_DIR / "tile_manifest.csv"

# 网格参数（与 tiles/build_vrt.py 一致）
XMIN, YMIN = 18.370850, -34.203447
XMAX, YMAX = 18.381972, -34.194205
TILE_SIZE = 0.0016  # degrees
N_COLS = 7
N_ROWS = 6

# 合并去重阈值
DEDUP_IOU_THRESHOLD = 0.8

# Microsoft Building Footprints 索引 URL
MS_INDEX_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


# ════════════════════════════════════════════════════════════════════════
# 第一步：下载 OSM 建筑轮廓
# ════════════════════════════════════════════════════════════════════════
def download_osm_buildings() -> gpd.GeoDataFrame:
    """通过 osmnx 从 OSM Overpass API 下载建筑轮廓"""
    try:
        import osmnx as ox

        print("[OSM] 正在下载建筑轮廓...")
        # osmnx features_from_bbox: bbox=(south, west, north, east) in newer versions
        # or bbox=(north, south, east, west) depending on version
        gdf = ox.features_from_bbox(
            bbox=(YMIN, YMAX, XMIN, XMAX),
            tags={"building": True},
        )

        if gdf is None or len(gdf) == 0:
            print("[OSM] 未找到建筑数据")
            return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

        # 只保留 Polygon/MultiPolygon
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf = gdf[["geometry"]].reset_index(drop=True)
        gdf.crs = "EPSG:4326"
        gdf["source"] = "osm"
        print(f"[OSM] 下载到 {len(gdf)} 个建筑轮廓")
        return gdf

    except ImportError:
        print("[OSM] osmnx 未安装，跳过 OSM 数据源")
        return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")
    except Exception as e:
        print(f"[OSM] 下载失败: {e}")
        return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")


# ════════════════════════════════════════════════════════════════════════
# 第二步：下载 Microsoft 建筑轮廓
# ════════════════════════════════════════════════════════════════════════
def lat_lon_to_quadkey(lat: float, lon: float, level: int = 9) -> str:
    """将经纬度转换为 Bing Maps quadkey"""
    import math

    x = (lon + 180.0) / 360.0
    sin_lat = math.sin(lat * math.pi / 180.0)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)

    quadkey = ""
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        pixel_x = int(x * (1 << level) * 256)
        pixel_y = int(y * (1 << level) * 256)
        tile_x = pixel_x // 256
        tile_y = pixel_y // 256
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def download_microsoft_buildings() -> gpd.GeoDataFrame:
    """从 Microsoft Global ML Building Footprints 下载南非建筑数据"""
    try:
        import requests

        print("[Microsoft] 正在获取数据索引...")
        resp = requests.get(MS_INDEX_URL, timeout=30)
        resp.raise_for_status()

        # 解析索引 CSV，筛选南非 (ZAF) 数据
        import io
        index_df = pd.read_csv(io.StringIO(resp.text))

        # 筛选南非
        zaf_rows = index_df[index_df["Location"].str.contains("SouthAfrica|ZAF|South Africa", case=False, na=False)]
        if len(zaf_rows) == 0:
            # 尝试其他列名
            for col in index_df.columns:
                zaf_rows = index_df[index_df[col].astype(str).str.contains("SouthAfrica|ZAF|South Africa", case=False, na=False)]
                if len(zaf_rows) > 0:
                    break

        if len(zaf_rows) == 0:
            print("[Microsoft] 未找到南非数据索引")
            print(f"  可用区域: {index_df.iloc[:, 0].unique()[:10]}...")
            return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")

        print(f"[Microsoft] 找到 {len(zaf_rows)} 个南非数据分片")

        # 计算项目区域的 quadkey 来筛选相关分片
        center_lat = (YMIN + YMAX) / 2
        center_lon = (XMIN + XMAX) / 2
        target_qk = lat_lon_to_quadkey(center_lat, center_lon, level=9)
        print(f"  项目区域 quadkey 前缀: {target_qk[:6]}")

        # 下载匹配的 GeoJSON 分片
        all_buildings = []
        url_col = None
        for col in ["Url", "url", "URL", "Link", "link"]:
            if col in zaf_rows.columns:
                url_col = col
                break

        if url_col is None:
            print("[Microsoft] 无法找到 URL 列")
            return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")

        # 尝试按 quadkey 筛选，或下载所有南非分片中覆盖项目区域的
        project_bbox = box(XMIN, YMIN, XMAX, YMAX)

        for _, row in zaf_rows.iterrows():
            url = row[url_col]
            # 检查 quadkey 是否匹配（URL 中通常包含 quadkey）
            qk_col = None
            for col in ["QuadKey", "quadkey", "Quadkey"]:
                if col in row.index:
                    qk_col = col
                    break

            if qk_col and str(row[qk_col])[:4] != target_qk[:4]:
                continue  # 跳过不相关的分片

            try:
                print(f"  下载分片: {url[-40:]}...")
                gdf_chunk = gpd.read_file(url)
                # 只保留与项目区域相交的建筑
                gdf_chunk = gdf_chunk[gdf_chunk.geometry.intersects(project_bbox)]
                if len(gdf_chunk) > 0:
                    all_buildings.append(gdf_chunk[["geometry"]])
                    print(f"    → {len(gdf_chunk)} 个建筑在项目区域内")
            except Exception as e:
                print(f"    [WARN] 下载失败: {e}")
                continue

        if not all_buildings:
            print("[Microsoft] 项目区域内未找到建筑数据")
            return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")

        result = pd.concat(all_buildings, ignore_index=True)
        result = gpd.GeoDataFrame(result, crs="EPSG:4326")
        result["source"] = "microsoft"
        print(f"[Microsoft] 共获取 {len(result)} 个建筑轮廓")
        return result

    except ImportError:
        print("[Microsoft] requests 未安装，跳过 Microsoft 数据源")
        return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")
    except Exception as e:
        print(f"[Microsoft] 下载失败: {e}")
        return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")


# ════════════════════════════════════════════════════════════════════════
# 第三步：合并去重
# ════════════════════════════════════════════════════════════════════════
def merge_and_dedup(
    osm: gpd.GeoDataFrame,
    ms: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    合并两个数据源，以 Microsoft 为底，OSM 中不重叠的补充进来。
    重叠判定：IoU > DEDUP_IOU_THRESHOLD
    """
    if len(ms) == 0 and len(osm) == 0:
        print("[合并] 两个数据源均为空")
        return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")

    if len(ms) == 0:
        print("[合并] 仅使用 OSM 数据")
        return osm

    if len(osm) == 0:
        print("[合并] 仅使用 Microsoft 数据")
        return ms

    print(f"[合并] Microsoft: {len(ms)}, OSM: {len(osm)}")

    # 以 Microsoft 为底，检查 OSM 中哪些不重叠
    ms_sindex = ms.sindex
    osm_to_add = []

    for idx, osm_row in osm.iterrows():
        osm_geom = osm_row.geometry
        candidates = list(ms_sindex.intersection(osm_geom.bounds))

        is_duplicate = False
        for cidx in candidates:
            ms_geom = ms.iloc[cidx].geometry
            try:
                intersection = osm_geom.intersection(ms_geom).area
                union = osm_geom.area + ms_geom.area - intersection
                if union > 0 and (intersection / union) > DEDUP_IOU_THRESHOLD:
                    is_duplicate = True
                    break
            except Exception:
                continue

        if not is_duplicate:
            osm_to_add.append(idx)

    osm_unique = osm.loc[osm_to_add].copy()
    osm_unique["source"] = "osm"

    merged = pd.concat([ms, osm_unique], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs="EPSG:4326")
    print(f"[合并] 去重后共 {len(merged)} 个建筑 (Microsoft: {len(ms)}, OSM 补充: {len(osm_unique)})")
    return merged


# ════════════════════════════════════════════════════════════════════════
# 第四步：生成 Tile 清单
# ════════════════════════════════════════════════════════════════════════
def generate_tile_manifest(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    将建筑轮廓与 tile 网格做空间交叉，
    生成 tile_manifest.csv（tile_id, has_buildings, building_count, source）
    """
    print("\n[Manifest] 生成 tile 清单...")

    rows = []
    buildings_sindex = buildings.sindex if len(buildings) > 0 else None

    for col in range(N_COLS):
        for row in range(N_ROWS):
            tile_id = f"G1238_{col}_{row}"

            # 重建 tile 边界框
            txmin = XMIN + col * TILE_SIZE
            txmax = min(txmin + TILE_SIZE, XMAX)
            tymax = YMAX - row * TILE_SIZE
            tymin = YMAX - (row + 1) * TILE_SIZE
            tile_box = box(txmin, tymin, txmax, tymax)

            if buildings_sindex is None or len(buildings) == 0:
                rows.append({
                    "tile_id": tile_id,
                    "has_buildings": False,
                    "building_count": 0,
                    "source": "",
                })
                continue

            # 空间查询
            candidates = list(buildings_sindex.intersection(tile_box.bounds))
            intersecting = []
            sources = set()
            for cidx in candidates:
                bldg = buildings.iloc[cidx]
                try:
                    if bldg.geometry.intersects(tile_box):
                        intersecting.append(cidx)
                        if "source" in buildings.columns:
                            sources.add(bldg["source"])
                except Exception:
                    continue

            has_bldg = len(intersecting) > 0
            source_str = "/".join(sorted(sources)) if sources else ""

            rows.append({
                "tile_id": tile_id,
                "has_buildings": has_bldg,
                "building_count": len(intersecting),
                "source": source_str,
            })

    df = pd.DataFrame(rows)
    df.to_csv(str(TILE_MANIFEST), index=False)
    print(f"[Manifest] 已保存: {TILE_MANIFEST}")

    n_with = df["has_buildings"].sum()
    n_without = len(df) - n_with
    print(f"  有建筑的 tile: {n_with}, 无建筑的 tile: {n_without}")
    print(f"  可跳过 {n_without} 个 tile 的下载和检测")
    return df


# ════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════
def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  建筑轮廓预处理 Building Footprint Preprocessing      ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    # 1. 下载 OSM
    osm_buildings = download_osm_buildings()

    # 2. 下载 Microsoft
    ms_buildings = download_microsoft_buildings()

    # 3. 合并去重
    merged = merge_and_dedup(osm_buildings, ms_buildings)

    if len(merged) == 0:
        print("\n[WARNING] 未获取到任何建筑数据，tile_manifest 将全部标记为无建筑")

    # 4. 保存 buildings.gpkg
    if len(merged) > 0:
        merged.to_file(str(BUILDINGS_GPKG), driver="GPKG")
        print(f"\n[OK] 建筑轮廓已保存: {BUILDINGS_GPKG} ({len(merged)} 个建筑)")
    else:
        # 保存空文件以避免下游报错
        empty = gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")
        empty.to_file(str(BUILDINGS_GPKG), driver="GPKG")
        print(f"\n[OK] 空建筑轮廓已保存: {BUILDINGS_GPKG}")

    # 5. 生成 tile 清单
    manifest = generate_tile_manifest(merged)

    print("\n[DONE] 建筑轮廓预处理完成!")
    print(f"  buildings.gpkg: {BUILDINGS_GPKG}")
    print(f"  tile_manifest.csv: {TILE_MANIFEST}")
    return merged, manifest


if __name__ == "__main__":
    main()
