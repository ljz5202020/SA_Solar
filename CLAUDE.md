# CLAUDE.md

Cape Town rooftop solar panel detection & evaluation pipeline. Uses geoai/SAM to detect solar panels from aerial GeoTIFFs, evaluates against hand-labeled ground truth (weak supervision).

## Directory Structure

```
data/
  task_grid.gpkg              — Grid 编号集合
  annotations/                — 弱监督标注（详见 annotations/README.md）
    G1238.gpkg                — QGIS 航测图标注
    solarpanel_g0001_g1190.gpkg — Google Earth 标注（已校准）
tiles/<GridID>/               — 各 Grid 的航测瓦片 + VRT
results/<GridID>/             — 检测结果、评估报告、图表
  masks/                      — per-tile 检测掩膜
  vectors/                    — per-tile 矢量化结果
docs/                         — 工作流文档
```

## Scripts

- `detect_and_evaluate.py` — 主流程（检测→过滤→评估→可视化），GRID_ID 变量控制目标 Grid
- `building_filter.py` — OSM+Microsoft 建筑轮廓 → buildings.gpkg + tile_manifest.csv
- `tiles/build_vrt.py` — WMS 瓦片配准 + VRT 拼接，GRID_ID 变量控制目标 Grid

## Running

```bash
python building_filter.py
python tiles/build_vrt.py
python detect_and_evaluate.py   # requires GPU
```

Detection skips if `results/<GridID>/predictions.geojson` exists. Delete to re-run.
