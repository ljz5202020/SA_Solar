# Cape Town Rooftop Solar Panel Detection

基于航测影像的开普敦屋顶太阳能板检测与评估流水线。

## 项目进度

| Grid | 底图 | 标注 | 检测 | 评估 | 备注 |
|------|------|------|------|------|------|
| G1238 | done | done (QGIS) | done | done | 首个完整流程 Grid |
| G0001-G1190 | - | done (Google Earth, 已校准) | - | - | 弱监督标注，待分配底图 |

## 快速开始

```bash
python building_filter.py          # 下载建筑轮廓
python tiles/build_vrt.py          # 瓦片配准 + VRT 拼接
python detect_and_evaluate.py      # 检测 + 评估（需 GPU）
```

## 目录结构

```
data/               GIS 数据（task grid、标注）
tiles/<GridID>/     各 Grid 的航测瓦片
results/<GridID>/   各 Grid 的检测结果与评估
docs/               工作流文档
```
