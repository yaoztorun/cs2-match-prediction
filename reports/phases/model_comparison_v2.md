# Model Comparison V2 (Validation Only)

All models use the identical chronological split, the identical 17 Phase-3 features, and the identical train-only mirrored augmentation policy. LR/RF/XGB-V1 numbers are read from their saved metadata JSONs (verified present and complete before use); none were re-run or modified.

**Tuning status (essential for fair reading):**

- **LR V1** = UNTUNED (from-scratch Logistic Regression baseline)
- **RF V1** = UNTUNED Random Forest baseline
- **RF V2** = TUNED (chronological TRAIN-only CV)
- **XGB V1** = UNTUNED fixed baseline
- **XGB V2** = TUNED (chronological TRAIN-only CV, three-stage temporal separation)

Logistic Regression has still never been tuned, so nothing below establishes algorithm superiority.

## Validation metrics

| metric | majority | LR V1 (untuned) | RF V1 (untuned) | RF V2 (tuned) | XGB V1 (untuned) | XGB V2 (tuned) |
|---|---|---|---|---|---|---|
| Accuracy | 0.5532 | 0.6110 | 0.5927 | 0.6068 | 0.5948 | 0.6117 |
| ROC-AUC | 0.5000 | 0.6431 | 0.6278 | 0.6566 | 0.6327 | 0.6504 |
| F1 | - | 0.6738 | 0.6564 | 0.6690 | 0.6563 | 0.6742 |
| Log loss | - | 0.6564 | 0.6679 | 0.6514 | 0.6608 | 0.6542 |
| Brier | - | 0.2322 | 0.2365 | 0.2298 | 0.2345 | 0.2311 |

## Train ROC-AUC, validation ROC-AUC, and the gap

| model | train AUC | val AUC | gap |
|---|---|---|---|
| LR V1 (untuned) | 0.6210 | 0.6431 | -0.0220 |
| RF V1 (untuned) | 0.9995 | 0.6278 | +0.3716 |
| RF V2 (tuned) | 0.7116 | 0.6566 | +0.0550 |
| XGB V1 (untuned) | 0.7514 | 0.6327 | +0.1187 |
| XGB V2 (tuned) | 0.6577 | 0.6504 | +0.0073 |

## XGB V2 deltas

| metric | XGB V2 - XGB V1 | XGB V2 - RF V2 | XGB V2 - LR V1 |
|---|---|---|---|
| Accuracy | +0.0169 | +0.0049 | +0.0007 |
| ROC-AUC | +0.0178 | -0.0061 | +0.0074 |
| F1 | +0.0179 | +0.0051 | +0.0004 |
| Log loss | -0.0066 | +0.0028 | -0.0023 |
| Brier | -0.0034 | +0.0013 | -0.0011 |

For log loss and Brier, **more negative is better**; for accuracy/ROC-AUC/F1, more positive is better.

On validation ROC-AUC, **the tuned XGBoost (XGB V2) outperformed the untuned Logistic Regression baseline (LR V1)**. This is not a claim that XGBoost is definitively better than Logistic Regression - LR has never been tuned, so the comparison is not like-for-like.

**No final project model is declared.** Logistic Regression remains untuned and the internal test set has not been used for this or any comparison.
