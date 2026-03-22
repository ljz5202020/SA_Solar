"""
OSM 建筑过滤：grid 级 + tile 级过滤，跳过纯荒地。

策略：
1. 一次性从 Overpass API 下载整个研究区的建筑中心点
2. Grid 级：统计每个 grid 内建筑数量，过滤无建筑 grid
3. Tile 级：对有建筑的 grid，标记哪些 tile 包含建筑
4. Buffer 策略：含建筑的 tile 的上下左右邻居也强制保留（grid 范围内）

输出：
  cache/grid_osm_buildings.csv          — grid 级统计
  cache/tile_download_mask.csv          — tile 级下载掩膜 (grid_id, col, row, reason)
  cache/osm_buildings_centroids.gpkg    — 建筑中心点缓存

Usage:
  python scripts/imagery/filter_grids_osm.py
  python scripts/imagery/filter_grids_osm.py --min-buildings 5
  python scripts/imagery/filter_grids_osm.py --buffer-rings 2   # 扩展 2 圈邻居
  python scripts/imagery/filter_grids_osm.py --refresh           # 重新下载 OSM 数据
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import TILES_ROOT, TASK_GRID_GPKG, get_grid_spec, get_tile_bounds

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BUILDING_CACHE = Path("cache/osm_buildings_centroids.gpkg")


def download_building_centroids(bounds, timeout=300):
    """Download building centroids from Overpass for the entire study area."""
    w, s, e, n = bounds
    query = f"""
[out:json][timeout:{timeout}];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out center;
"""
    print(f"Querying Overpass API for buildings in "
          f"({s:.4f},{w:.4f},{n:.4f},{e:.4f})...")
    print(f"This may take 1-2 minutes for the Cape Town area...")

    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout + 30)
    resp.raise_for_status()
    data = resp.json()

    elements = data.get("elements", [])
    print(f"Received {len(elements)} building elements from OSM")

    points = []
    for el in elements:
        center = el.get("center")
        if center:
            points.append(Point(center["lon"], center["lat"]))
        elif "lat" in el and "lon" in el:
            points.append(Point(el["lon"], el["lat"]))

    if not points:
        print("WARNING: No building centroids extracted!")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry",
                                crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    print(f"Extracted {len(gdf)} building centroids")
    return gdf


def compute_tile_mask(grid_id, spec, buildings_gdf, buffer_rings=1):
    """Compute which tiles to download for a grid.

    Returns list of (col, row, reason) where reason is:
      'building' — tile contains building(s)
      'buffer'   — neighbor of a tile with buildings
    """
    # Build tile geometries
    tiles = {}
    for col in range(spec.n_cols):
        for row in range(spec.n_rows):
            txmin, tymin, txmax, tymax = get_tile_bounds(spec, col, row)
            tiles[(col, row)] = box(txmin, tymin, txmax, tymax)

    # Create tile GeoDataFrame for spatial join
    tile_records = []
    for (col, row), geom in tiles.items():
        tile_records.append({"col": col, "row": row, "geometry": geom})
    tiles_gdf = gpd.GeoDataFrame(tile_records, crs="EPSG:4326")

    # Clip buildings to grid extent
    grid_bounds = box(spec.xmin, spec.ymin, spec.xmax, spec.ymax)
    bldg_in_grid = buildings_gdf[buildings_gdf.within(grid_bounds)]

    if len(bldg_in_grid) == 0:
        return []

    # Spatial join: which tiles have buildings?
    joined = gpd.sjoin(tiles_gdf, bldg_in_grid, how="inner", predicate="contains")
    building_tiles = set(zip(joined["col"], joined["row"]))

    # Expand with buffer rings (neighbors)
    all_tiles = {}
    for col, row in building_tiles:
        all_tiles[(col, row)] = "building"

    current_frontier = set(building_tiles)
    for ring in range(buffer_rings):
        next_frontier = set()
        for col, row in current_frontier:
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nc, nr = col + dc, row + dr
                if 0 <= nc < spec.n_cols and 0 <= nr < spec.n_rows:
                    if (nc, nr) not in all_tiles:
                        all_tiles[(nc, nr)] = "buffer"
                        next_frontier.add((nc, nr))
        current_frontier = next_frontier

    result = [(col, row, reason) for (col, row), reason in sorted(all_tiles.items())]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Filter grids and tiles by OSM building presence")
    parser.add_argument("--grid-output", default="cache/grid_osm_buildings.csv")
    parser.add_argument("--tile-output", default="cache/tile_download_mask.csv")
    parser.add_argument("--min-buildings", type=int, default=1,
                        help="Minimum buildings for a grid to be included")
    parser.add_argument("--buffer-rings", type=int, default=1,
                        help="Number of neighbor rings around building tiles (default 1)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download OSM data")
    args = parser.parse_args()

    grid_out = Path(args.grid_output)
    tile_out = Path(args.tile_output)
    grid_out.parent.mkdir(parents=True, exist_ok=True)

    # Load task grid
    tg = gpd.read_file(TASK_GRID_GPKG)
    print(f"Total grids in task_grid: {len(tg)}")
    bounds = tg.total_bounds

    # Download or load cached building centroids
    if BUILDING_CACHE.exists() and not args.refresh:
        print(f"Loading cached building centroids from {BUILDING_CACHE}")
        buildings = gpd.read_file(BUILDING_CACHE)
        print(f"  {len(buildings)} centroids loaded")
    else:
        buildings = download_building_centroids(bounds)
        if len(buildings) > 0:
            BUILDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
            buildings.to_file(BUILDING_CACHE, driver="GPKG")
            print(f"Cached to {BUILDING_CACHE}")

    # ── Grid-level stats ──
    print("\n[1/2] Grid-level: counting buildings per grid...")
    joined = gpd.sjoin(buildings, tg[["gridcell_id", "geometry"]],
                       how="inner", predicate="within")
    counts = joined.groupby("gridcell_id").size().reset_index(name="n_buildings")

    all_grids = tg[["gridcell_id"]].copy()
    grid_result = all_grids.merge(counts, on="gridcell_id", how="left")
    grid_result["n_buildings"] = grid_result["n_buildings"].fillna(0).astype(int)
    grid_result["has_buildings"] = grid_result["n_buildings"] >= args.min_buildings
    grid_result = grid_result.rename(columns={"gridcell_id": "grid_id"})
    grid_result = grid_result.sort_values("grid_id").reset_index(drop=True)
    grid_result.to_csv(grid_out, index=False)

    grids_with = grid_result[grid_result["has_buildings"]]["grid_id"].tolist()
    grids_without = len(grid_result) - len(grids_with)

    print(f"  Has buildings: {len(grids_with)} grids")
    print(f"  No buildings:  {grids_without} grids (skipped)")

    # ── Tile-level mask ──
    print(f"\n[2/2] Tile-level: computing download mask "
          f"(buffer={args.buffer_rings} rings)...")

    tile_rows = []
    total_tiles = 0
    total_selected = 0

    for i, grid_id in enumerate(grids_with):
        spec = get_grid_spec(grid_id)
        grid_total = spec.n_cols * spec.n_rows
        total_tiles += grid_total

        mask = compute_tile_mask(grid_id, spec, buildings, args.buffer_rings)
        for col, row, reason in mask:
            tile_rows.append({
                "grid_id": grid_id,
                "col": col,
                "row": row,
                "reason": reason,
            })
        total_selected += len(mask)

        n_bldg = sum(1 for _, _, r in mask if r == "building")
        n_buf = sum(1 for _, _, r in mask if r == "buffer")

        if (i + 1) % 100 == 0 or i == len(grids_with) - 1:
            print(f"  Processed {i+1}/{len(grids_with)} grids...")

    tile_df = pd.DataFrame(tile_rows)
    tile_df.to_csv(tile_out, index=False)

    # Also count tiles in skipped (no-building) grids
    for grid_id in grid_result[~grid_result["has_buildings"]]["grid_id"]:
        try:
            spec = get_grid_spec(grid_id)
            total_tiles += spec.n_cols * spec.n_rows
        except Exception:
            pass

    # ── Summary ──
    n_bldg_tiles = len(tile_df[tile_df["reason"] == "building"])
    n_buf_tiles = len(tile_df[tile_df["reason"] == "buffer"])
    skipped_tiles = total_tiles - total_selected

    print(f"\n{'='*55}")
    print(f"Grid-level results:  {grid_out}")
    print(f"Tile-level mask:     {tile_out}")
    print(f"")
    print(f"Grid summary:")
    print(f"  Total grids:       {len(grid_result)}")
    print(f"  With buildings:    {len(grids_with)} ({100*len(grids_with)/len(grid_result):.1f}%)")
    print(f"  Skipped (empty):   {grids_without} ({100*grids_without/len(grid_result):.1f}%)")
    print(f"")
    print(f"Tile summary (across {len(grids_with)} grids with buildings):")
    print(f"  Total tiles:       {total_tiles}")
    print(f"  Selected:          {total_selected} ({100*total_selected/total_tiles:.1f}%)")
    print(f"    Building tiles:  {n_bldg_tiles}")
    print(f"    Buffer tiles:    {n_buf_tiles}")
    print(f"  Skipped:           {skipped_tiles} ({100*skipped_tiles/total_tiles:.1f}%)")

    # Per-grid breakdown for top grids
    if len(tile_df) > 0:
        per_grid = tile_df.groupby("grid_id").agg(
            selected=("reason", "count"),
            building=("reason", lambda x: (x == "building").sum()),
            buffer=("reason", lambda x: (x == "buffer").sum()),
        )
        # merge with grid total tiles
        grid_totals = []
        for gid in per_grid.index:
            spec = get_grid_spec(gid)
            grid_totals.append(spec.n_cols * spec.n_rows)
        per_grid["total"] = pd.array(grid_totals, dtype="int64")
        per_grid["pct"] = (100 * per_grid["selected"] / per_grid["total"]).round(1)
        per_grid = per_grid.sort_values("selected", ascending=False)

        print(f"\nTop 10 grids by selected tiles:")
        print(f"  {'Grid':>7s}  {'Total':>5s}  {'Select':>6s}  {'Bldg':>4s}  {'Buf':>4s}  {'%':>5s}")
        for gid, r in per_grid.head(10).iterrows():
            print(f"  {gid:>7s}  {int(r['total']):5d}  {int(r['selected']):6d}  "
                  f"{int(r['building']):4d}  {int(r['buffer']):4d}  {r['pct']:5.1f}%")


if __name__ == "__main__":
    main()
