# Project Status — Cape Town Solar Panel Detection

**Last Updated**: 2026-03-18

## Current Phase

V1.2 — Evaluation Profile & Annotation Alignment (COMPLETE; installation profile validated on GPU for G1189/G1190/G1238)

---

## Progress Tracker
<!-- progress:status:start -->
### Recent Updates
- 2026-03-18: Restructure project: move shared utils to core/, group scripts by domain
- 2026-03-18: Added low-resolution grid preview batching for G1240+ screening and contact-sheet generation.
- 2026-03-18: Added browser review UI for keep/exclude/review decisions with WSL-friendly LAN access hints.
- 2026-03-18: Completed the first 100-grid preview batch from G1240 and finished manual screening for that batch.

### Current Ops Focus
- Repository structure cleanup: reduce root-level script clutter and group workflows by purpose.
- Export reviewed keep/exclude decisions into a reusable grid manifest for later tile downloads.
<!-- progress:status:end -->

## Evaluation Results Summary
<!-- progress:results:start -->
| Grid | Precision | Recall | F1 | Status |
|------|-----------|--------|----|--------|
| G1238 | 0.7273 | 0.8455 | 0.7820 | evaluated |
| G1189 | 0.7188 | 0.7931 | 0.7541 | evaluated |
| G1190 | 0.7922 | 0.8026 | 0.7974 | evaluated |
| JHB01 | — | — | — | not evaluated |
| JHB02 | — | — | — | not evaluated |
| JHB03 | — | — | — | not evaluated |
| JHB04 | — | — | — | not evaluated |
| JHB05 | — | — | — | not evaluated |
| JHB06 | — | — | — | not evaluated |
<!-- progress:results:end -->

## V0: Baseline Detection Pipeline — COMPLETE

| Grid | Precision | Recall | F1 | Notes |
|------|-----------|--------|----|-------|
| G1238 | 0.62 | 0.66 | 0.64 | Best grid, 124 annotations |
| G1189 | — | 0.33 | — | Low recall |
| G1190 | — | 0.39 | — | Low recall |
| JHB01–06 | — | 0.28 (macro) | — | Cross-city transfer |

## V1: Cape Town Fine-Tune — MINIMUM V1 COMPLETE

Fine-tuned Mask R-CNN on 257 Cape Town annotations across 3 grids.

**Best checkpoint**: `checkpoints/v1_ft_cs400_tileval_20260317_r4/best_model.pth` (val_AP50=0.4205)

### Full-Grid Fine-Tuned Results (GPU-verified, installation profile)

| Grid | F1@IoU0.3 |
|------|-----------|
| G1189 | 0.5950 |
| G1190 | 0.6490 |
| G1238 | 0.7789 |

### Residual Issues

- G1189 small-panel recall regressed: 0.3077 → 0.2308 (val split)
- Post-training calibration sweep completed; inference bundle freeze still pending
- Leave-one-grid-out cross-validation pending
- JHB transfer evaluation not performed
- Parameter freeze for v1 inference bundle pending

Full val-split tables, training data/config, and size-stratified gains → see `ROADMAP.md`.

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

## V2: Future Directions — NOT STARTED

- Additional Cape Town grids
- JHB annotations + fine-tuning
- 2025 satellite imagery evaluation
- Stronger backbone (Swin Transformer)
- Active learning
- Temporal analysis (installation time estimation)
