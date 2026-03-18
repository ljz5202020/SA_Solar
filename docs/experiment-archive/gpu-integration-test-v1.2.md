# GPU Integration Test Log — V1.2

Archived from STATUS.md. These are the GPU-validated full-grid integration test results for V1.2 installation profile.

---

## 2026-03-17 — G1189 full-grid installation profile

- Command run on GPU:
  `./.venv/bin/python detect_and_evaluate.py --grid-id G1189 --model-path checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth --evaluation-profile installation --force`
- Runtime confirmed CUDA execution on `NVIDIA GeForce RTX 4070 Laptop GPU`.
- Detection summary: 64 final prediction polygons after post-processing and confidence filtering.
- Ground truth: 58 installation polygons from `solarpanel_g0001_g1190.gpkg`.

| Metric | Value |
|--------|-------|
| Presence P@IoU0.1 | 0.6667 |
| Presence R@IoU0.1 | 0.7241 |
| Presence F1@IoU0.1 | 0.6942 |
| Merge F1@IoU0.3 | 0.5950 |
| Merge F1@IoU0.5 | 0.4628 |
| Mean IoU | 0.5439 |
| Median IoU | 0.5972 |
| IoU>=0.3 match rate | 85.7% |
| IoU>=0.5 match rate | 66.7% |

Acceptance check for G1189:

- `G1189 >= 0.595` full-grid F1@IoU0.3: met exactly (`0.5950`)
- presence recall@IoU0.1 baseline guard: currently acceptable for V1.2 tracking
- 5-20m² bucket reporting: available, with 12 FN in the `5-20m²` bucket
- area-error baseline: established via `results/G1189/area_error_metrics.csv`

Remaining V1.2 integration work:

- None for the current V1.2 release gate

## 2026-03-17 — G1190 full-grid installation profile

- Command run on GPU:
  `./.venv/bin/python detect_and_evaluate.py --grid-id G1190 --model-path checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth --evaluation-profile installation --force`
- Runtime confirmed CUDA execution on `NVIDIA GeForce RTX 4070 Laptop GPU`.
- Detection summary: 75 final prediction polygons after post-processing and confidence filtering.
- Ground truth: 76 installation polygons from `solarpanel_g0001_g1190.gpkg`.

| Metric | Value |
|--------|-------|
| Presence P@IoU0.1 | 0.7600 |
| Presence R@IoU0.1 | 0.7500 |
| Presence F1@IoU0.1 | 0.7550 |
| Merge F1@IoU0.3 | 0.6490 |
| Merge F1@IoU0.5 | 0.5033 |
| Mean IoU | 0.5410 |
| Median IoU | 0.6020 |
| IoU>=0.3 match rate | 86.0% |
| IoU>=0.5 match rate | 66.7% |

Acceptance check for G1190:

- `G1190 >= 0.649` full-grid F1@IoU0.3: met exactly (`0.6490`)
- presence recall@IoU0.1 baseline guard: passed (`0.7500`)
- 5-20m² bucket reporting: available, with 23 FN in the `5-20m²` bucket
- area-error baseline: established via `results/G1190/area_error_metrics.csv`

## 2026-03-17 — G1238 full-grid installation profile

- Command run on GPU:
  `./.venv/bin/python detect_and_evaluate.py --grid-id G1238 --model-path checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth --evaluation-profile installation --force`
- Runtime confirmed CUDA execution on `NVIDIA GeForce RTX 4070 Laptop GPU`.
- Detection summary: 184 final prediction polygons after post-processing and confidence filtering.
- Ground truth: 123 installation polygons from layer `g1238__solar_panel__cape_town_g1238_`.

| Metric | Value |
|--------|-------|
| Presence P@IoU0.1 | 0.7037 |
| Presence R@IoU0.1 | 0.9268 |
| Presence F1@IoU0.1 | 0.8000 |
| Merge F1@IoU0.3 | 0.7789 |
| Merge F1@IoU0.5 | 0.5878 |
| Mean IoU | 0.6476 |
| Median IoU | 0.6812 |
| IoU>=0.3 match rate | 97.4% |
| IoU>=0.5 match rate | 76.3% |

Acceptance check for G1238:

- no-regression guard: passed; full-grid merge F1@IoU0.3 is well above the historical baseline (`0.7789` vs `0.64`)
- presence recall@IoU0.1 baseline guard: passed (`0.9268`)
- area-error baseline: established via `results/G1238/area_error_metrics.csv`
