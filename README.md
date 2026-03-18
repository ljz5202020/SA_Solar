# Cape Town Rooftop Solar Panel Detection

基于航测影像的开普敦屋顶太阳能安装检测与评估流水线。

**V1.2 任务定义**: installation-level footprint segmentation — 每个 polygon 表示一个太阳能安装的轮廓，非逐 panel 分割。

## 项目进度

| Grid | 底图 | 标注 | 检测 | 评估 | 备注 |
|------|------|------|------|------|------|
| G1238 | done | done (QGIS) | done | done | 首个完整流程 Grid |
| G1189 | done | done (Google Earth, 已校准) | done | done | Fine-tuned F1≈0.595 |
| G1190 | done | done (Google Earth, 已校准) | done | done | Fine-tuned F1≈0.649 |

详细进度见 `STATUS.md` 和 `ROADMAP.md`。

## 快速开始

```bash
./scripts/bootstrap_env.sh         # 首次创建/更新 .venv
source scripts/activate_env.sh
./scripts/check_env.sh             # 检查依赖 + CUDA
python building_filter.py          # 下载建筑轮廓
python detect_and_evaluate.py      # 检测 + 评估（需 GPU）
```

## 文档导航

| 文档 | 内容 |
|------|------|
| [`docs/architecture.md`](docs/architecture.md) | 目录结构、路径映射、CRS 约定 |
| [`docs/workflows.md`](docs/workflows.md) | 推理、微调、分析完整命令序列 |
| [`docs/governance/repo-rules.md`](docs/governance/repo-rules.md) | Git 大文件保护、目录治理 |
| [`data/annotations/ANNOTATION_SPEC.md`](data/annotations/ANNOTATION_SPEC.md) | V1.2 标注规范 |
| [`STATUS.md`](STATUS.md) | 当前状态 + 里程碑摘要 |
| [`ROADMAP.md`](ROADMAP.md) | 版本详细历史 + 决策记录 |

## 本地环境

- 虚拟环境固定在 `./.venv`
- 运行时缓存固定在仓库内：`.cache/`、`.config/`、`.local/`、`.tmp/`
- `requirements.lock.txt` 为环境快照，重建时优先使用
- `train.py` 强制验证 CUDA；`./scripts/check_env.sh` 显示 `cuda_available=False` 时训练不会启动
