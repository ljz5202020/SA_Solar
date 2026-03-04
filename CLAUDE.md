# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cape Town rooftop solar panel detection and evaluation pipeline. Uses computer vision (geoai / SAM) to detect solar panels from aerial imagery GeoTIFFs, then evaluates detection quality against hand-labeled ground truth using IoU-based matching.

The project supports a QGIS-based annotation workflow where operators label solar panels grid-by-grid on Cape Town municipal aerial imagery (WMS).

## Key Files

- `detect_and_evaluate.py` — Main pipeline: detection → building mask → NMS → geometric/color filtering → evaluation → visualization
- `building_filter.py` — Building footprint preprocessing: downloads OSM + Microsoft footprints, merges, generates `buildings.gpkg` and `tile_manifest.csv`
- `tiles/build_vrt.py` — Georeferences raw WMS tiles and builds VRT mosaic; supports tile manifest filtering

## Running the Pipeline

```bash
# Install dependencies
pip install geoai-py geopandas shapely scikit-learn matplotlib seaborn rasterio rasterstats osmnx requests

# Step 1: Download building footprints and generate tile manifest
python building_filter.py

# Step 2: (Optional) Rebuild VRT with tile filtering
python tiles/build_vrt.py

# Step 3: Run detection and evaluation (requires GPU)
python detect_and_evaluate.py
```

The detection script skips if `results/predictions.geojson` exists. Delete it to re-run.

## Path Configuration

All scripts use `Path(__file__).parent` for relative paths. Works on both Windows and WSL/Linux without modification. `tiles/build_vrt.py` auto-detects GDAL tool paths by platform.

## Architecture Notes

- Detection has two fallback paths: Path A (geoai.SolarPanelDetector, preferred) → Path B (samgeo SAM model)
- Post-detection filtering pipeline (in order):
  1. RGB color stats — removes shadows (<45), pools (blue-dominant), reflections (>240), vegetation (green-dominant)
  2. Building footprint mask — only keeps detections intersecting building polygons (2m buffer)
  3. Spatial NMS — removes duplicate detections from chip overlap (IoU>0.5)
  4. Geometric filtering — area (>8m²), elongation (<6), solidity (>0.7)
- `building_filter.py` downloads OSM (osmnx) + Microsoft ML footprints, deduplicates by IoU>0.8, outputs `buildings.gpkg`
- `tile_manifest.csv` marks tiles with/without buildings; used by `build_vrt.py` to skip empty tiles
- Evaluation supports two matching modes: **merge_preds** (default) and **strict one-to-one**
- CRS is unified to `EPSG:32734` (UTM Zone 34S) for all metric calculations
- Tile naming convention: `G{grid_id}_{col}_{row}_geo.tif` (geo-referenced) vs `G{grid_id}_{col}_{row}.tif` (raw)

## Data Layout

- `tiles/` — Input GeoTIFF tiles (42 tiles for grid G1238, ~2000x2000px each)
- `results/` — Detection masks, vectorized predictions, merged `predictions.geojson`, evaluation CSV, charts
- `buildings.gpkg` — Merged building footprints (OSM + Microsoft)
- `tile_manifest.csv` — Per-tile building presence info
- `G1238.gpkg` / `g1238.geojson` — Ground truth annotations
- `task grid.gpkg` — Grid cells for task management in QGIS
