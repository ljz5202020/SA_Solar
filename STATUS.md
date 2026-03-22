# Project Status — Cape Town Solar Panel Detection

**Last Updated**: 2026-03-20

## Current Phase

V2 — SAM2 Annotation & Retrain (COMPLETE; primary metric switched to F1@IoU0.5)

---

## Progress Tracker
<!-- progress:status:start -->
### Recent Updates
- 2026-03-20: V2 SAM2 annotation + retrain complete (val_AP50=0.6889, +0.268 vs V1)
- 2026-03-20: Model-assisted annotation script for semi-auto QGIS workflow
- 2026-03-20: Evaluation GT switched to SAM2 annotations; primary metric → F1@IoU0.5
- 2026-03-18: Add annotation cleanup, docs restructure, JHB fine-tuned evaluation, and project configs
- 2026-03-18: Restructure project: move shared utils to core/, group scripts by domain

### Current Ops Focus
- Semi-automatic annotation: expand to more grids via export_hints.py + SAM2 click-segment
- FP reduction: hard negative mining, confidence threshold tuning
<!-- progress:status:end -->

## Evaluation Results Summary (V2, SAM2 GT, F1@IoU0.5)
<!-- progress:results:start -->
| Grid | P@IoU0.5 | R@IoU0.5 | F1@IoU0.5 | mean IoU | Status |
|------|----------|----------|-----------|----------|--------|
| G1238 | 0.509 | 0.569 | 0.537 | 0.691 | evaluated |
| G1190 | 0.559 | 0.717 | 0.628 | 0.742 | evaluated |
| G1189 | 0.555 | 0.560 | 0.557 | 0.729 | evaluated |
| JHB01–06 | — | — | — | — | not evaluated |
<!-- progress:results:end -->

## V0: Baseline Detection Pipeline — COMPLETE

| Grid | Precision | Recall | F1 | Notes |
|------|-----------|--------|----|-------|
| G1238 | 0.62 | 0.66 | 0.64 | Best grid, 124 annotations |
| G1189 | — | 0.33 | — | Low recall |
| G1190 | — | 0.39 | — | Low recall |
| JHB01–06 | — | 0.28 (macro) | — | Cross-city transfer |

## V1: Cape Town Fine-Tune — SUPERSEDED BY V2

Fine-tuned Mask R-CNN on 257 Cape Town annotations (legacy weak-supervision).

**Best checkpoint**: `checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth` (val_AP50=0.4205)

Full details → see `ROADMAP.md` V1 section.

---

## V1.2: Evaluation Profile & Annotation Alignment — COMPLETE

Task frozen as **installation-level footprint segmentation** (not panel-level). No model change this round.

### Work Package Progress

| WP | Description | Status |
|----|-------------|--------|
| Pre | STATUS.md progress tracker | Done |
| WP0 | Annotation spec (ANNOTATION_SPEC.md) | Done |
| WP1 | Annotation manifest + bootstrap script | Done |
| WP2 | Evaluation profile (presence/footprint/area) | Done |
| WP3 | config.json extension (evaluation_config) | Done |
| WP4 | COCO export manifest-aware filtering | Done |
| WP5 | train.py minor update | Done |
| WP6 | Documentation updates | Done |
| -- | GPU integration test (3 grids) | Done |

### V1.2 Acceptance Criteria

- full-grid F1@IoU0.3: G1189 >= 0.595, G1190 >= 0.649, G1238 no regression
- presence recall@IoU0.1: no regression from fine-tuned baseline
- 5-20m² bucket recall: must be reported; G1189/G1190 no regression
- area error: establish baseline (no hard gate yet)

Acceptance status: all current V1.2 gates satisfied on GPU-validated full-grid runs.

GPU integration test details → [`docs/experiment-archive/gpu-integration-test-v1.2.md`](docs/experiment-archive/gpu-integration-test-v1.2.md)

---

## V2: SAM2 Annotation & Retrain — COMPLETE

Re-annotated all 3 grids with SAM2.1 (GeoSAM/QGIS), 456 polygons total (T1 quality).

**Best checkpoint**: `checkpoints/v2_sam2_260320/best_model.pth` (val_AP50=0.6889)

| Metric | V1 | V2 | Change |
|--------|-----|-----|--------|
| val_AP50 | 0.4205 | 0.6889 | **+0.268** |
| Annotations | 257 (T2) | 456 (T1 SAM2) | +77% |
| Primary metric | F1@IoU0.1 | **F1@IoU0.5** | upgraded |

Key: precision (~0.63) is now the bottleneck, not recall. Semi-auto annotation workflow ready for scaling.

---

## V3: Future Directions — NOT STARTED

- Semi-auto annotation expansion to more grids
- FP reduction (hard negatives, confidence tuning)
- Detector + SAM2 inference pipeline
- JHB cross-city transfer
- Stronger backbone (Swin, ConvNeXt)
- Active learning
