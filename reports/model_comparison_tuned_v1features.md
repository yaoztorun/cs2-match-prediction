# Model Comparison - Phase-3 V1 Features (Validation Only)

All six models use the identical chronological split, the identical 17 Phase-3 features, and the identical train-only mirrored augmentation. All three tuned models used the **same** TRAIN-only expanding-window CV folds. Every number for a model other than LR V2 is read from that model's saved metadata JSON (verified complete before use); none were re-run or modified.

**LR V1 = UNTUNED · LR V2 = TUNED · RF V1 = UNTUNED · RF V2 = TUNED · XGB V1 = UNTUNED · XGB V2 = TUNED**

## All six models (validation)

| metric | majority | LR V1 | LR V2 | RF V1 | RF V2 | XGB V1 | XGB V2 |
|---|---|---|---|---|---|---|---|
| Accuracy | 0.5532 | 0.6110 | 0.6131 | 0.5927 | 0.6068 | 0.5948 | 0.6117 |
| ROC-AUC | 0.5000 | 0.6431 | 0.6412 | 0.6278 | 0.6566 | 0.6327 | 0.6504 |
| F1 | - | 0.6738 | 0.6761 | 0.6564 | 0.6690 | 0.6563 | 0.6742 |
| Log loss | - | 0.6564 | 0.6581 | 0.6679 | 0.6514 | 0.6608 | 0.6542 |
| Brier | - | 0.2322 | 0.2329 | 0.2365 | 0.2298 | 0.2345 | 0.2311 |

## TUNED-ONLY comparison: LR V2 vs RF V2 vs XGB V2

This is the project's **first approximately fair algorithm comparison**: same 17 features, same chronological history, same main validation partition, and the same TRAIN-only temporal CV folds used to tune each one.

| metric | LR V2 (tuned) | RF V2 (tuned) | XGB V2 (tuned) | best |
|---|---|---|---|---|
| Accuracy | 0.6131 | 0.6068 | 0.6117 | **LR V2** |
| ROC-AUC | 0.6412 | 0.6566 | 0.6504 | **RF V2** |
| F1 | 0.6761 | 0.6690 | 0.6742 | **LR V2** |
| Log loss | 0.6581 | 0.6514 | 0.6542 | **RF V2** |
| Brier | 0.2329 | 0.2298 | 0.2311 | **RF V2** |

## Train -> validation ROC-AUC gaps

| model | train AUC | val AUC | gap |
|---|---|---|---|
| LR V1 (untuned) | 0.6210 | 0.6431 | -0.0220 |
| LR V2 (tuned) | 0.6234 | 0.6412 | -0.0177 |
| RF V1 (untuned) | 0.9995 | 0.6278 | +0.3716 |
| RF V2 (tuned) | 0.7116 | 0.6566 | +0.0550 |
| XGB V1 (untuned) | 0.7514 | 0.6327 | +0.1187 |
| XGB V2 (tuned) | 0.6577 | 0.6504 | +0.0073 |

## LR V2 deltas

| metric | LR V2 - LR V1 | LR V2 - RF V2 | LR V2 - XGB V2 |
|---|---|---|---|
| Accuracy | +0.0021 | +0.0063 | +0.0014 |
| ROC-AUC | -0.0019 | -0.0154 | -0.0093 |
| F1 | +0.0023 | +0.0071 | +0.0019 |
| Log loss | +0.0017 | +0.0068 | +0.0040 |
| Brier | +0.0007 | +0.0031 | +0.0018 |

For log loss and Brier, **more negative is better**; for accuracy/ROC-AUC/F1, more positive is better.

## No final project model is declared

- Richer features (map-level, player-level) have not been evaluated.
- The internal test partition remains **sealed**.
- Cologne 2026 remains an **external, untouched** holdout.
- These are single-validation-period results on 1,419 matches; differences of a few thousandths should not be treated as decisive.
