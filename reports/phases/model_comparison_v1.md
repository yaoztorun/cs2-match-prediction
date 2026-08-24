# Model Comparison V1 (Validation Only)

All models use the identical chronological split, the identical 17 Phase-3 features, and the identical train-only mirrored augmentation policy. LR/RF numbers are read from their saved metadata JSONs (verified present and complete before use); none of those models were re-run or modified.

**Tuning status matters for fair reading:**

- **LR V1** = untuned (from-scratch Logistic Regression baseline)
- **RF V1** = untuned Random Forest baseline
- **RF V2** = chronologically tuned Random Forest (TRAIN-only expanding-window CV)
- **XGBoost V1** = untuned fixed baseline

Only RF V2 has received any hyperparameter search. Comparisons below must not be read as algorithm-superiority claims.

## Validation metrics

| metric | majority | LR V1 (untuned) | RF V1 (untuned) | RF V2 (tuned) | XGBoost V1 (untuned) |
|---|---|---|---|---|---|
| Accuracy | 0.5532 | 0.6110 | 0.5927 | 0.6068 | 0.5948 |
| ROC-AUC | 0.5000 | 0.6431 | 0.6278 | 0.6566 | 0.6327 |
| F1 | - | 0.6738 | 0.6564 | 0.6690 | 0.6563 |
| Log loss | - | 0.6564 | 0.6679 | 0.6514 | 0.6608 |
| Brier | - | 0.2322 | 0.2365 | 0.2298 | 0.2345 |

## Train metrics and train -> validation gaps

| model | train acc | val acc | acc gap | train AUC | val AUC | AUC gap |
|---|---|---|---|---|---|---|
| LR V1 (untuned) | 0.5909 | 0.6110 | -0.0201 | 0.6210 | 0.6431 | -0.0220 |
| RF V1 (untuned) | 0.9863 | 0.5927 | +0.3936 | 0.9995 | 0.6278 | +0.3716 |
| RF V2 (tuned) | 0.6460 | 0.6068 | +0.0393 | 0.7116 | 0.6566 | +0.0550 |
| XGBoost V1 (untuned) | 0.6738 | 0.5948 | +0.0790 | 0.7514 | 0.6327 | +0.1187 |

## XGBoost V1 deltas

| metric | XGB V1 - LR V1 | XGB V1 - RF V1 | XGB V1 - RF V2 |
|---|---|---|---|
| Accuracy | -0.0162 | +0.0021 | -0.0120 |
| ROC-AUC | -0.0104 | +0.0048 | -0.0239 |
| F1 | -0.0175 | -0.0001 | -0.0127 |
| Log loss | +0.0043 | -0.0071 | +0.0094 |
| Brier | +0.0022 | -0.0021 | +0.0046 |

For log loss and Brier, **more negative is better**; for accuracy/ROC-AUC/F1, more positive is better.

**No final project model is declared.** Only RF V2 has been tuned; LR and XGBoost remain untuned baselines, and the internal test set has not been used for this or any comparison.
