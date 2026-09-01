# Model Comparison: Logistic Regression V1 vs. Random Forest V1 (Validation)

Logistic Regression's numbers are read from `models/logistic_regression_scratch_v1.json` (verified present and complete before use, not re-run or modified). Both models use the exact same chronological split, the same 17 Phase-3 features, and the same train-only mirrored augmentation policy.

## Validation metrics

| metric | majority baseline | Logistic Regression V1 | Random Forest V1 |
|---|---|---|---|
| Accuracy | 0.5532 | 0.6110 | 0.5927 |
| ROC-AUC | 0.5000 | 0.6431 | 0.6278 |
| F1 | - | 0.6738 | 0.6564 |
| Log loss | - | 0.6564 | 0.6679 |
| Brier score | - | 0.2322 | 0.2365 |

## Train-validation gap (overfitting signal)

| metric | LR train | LR val | LR gap | RF train | RF val | RF gap |
|---|---|---|---|---|---|---|
| Accuracy | 0.5909 | 0.6110 | -0.0201 | 0.9863 | 0.5927 | +0.3936 |
| ROC-AUC | 0.6210 | 0.6431 | -0.0220 | 0.9995 | 0.6278 | +0.3716 |
| Log loss | 0.6662 | 0.6564 | +0.0097 | 0.2065 | 0.6679 | -0.4614 |

## Discussion

- **Discrimination**: Random Forest's validation ROC-AUC is 0.6278 vs. Logistic Regression's 0.6431 (-0.0152). Logistic Regression discriminates better on validation.
- **Calibration**: see each model's own calibration diagnostic (`logistic_regression_calibration_v1.png` / `random_forest_calibration_v1.png`) and Brier/log-loss above - lower is better-calibrated/sharper.
- **Train-validation gap**: see the table above; a large Random Forest gap (if present) reflects the untuned `max_depth=None`/`min_samples_leaf=1` baseline configuration overfitting relative to Logistic Regression's much smaller gap, which is expected of an unregularized deep forest and is not treated as a bug in this phase.
- **Confidence/extremeness of probabilities**: see `random_forest_probability_distribution_v1.png` vs. Logistic Regression's narrower/wider spread (Phase 4A report) for a visual comparison.
- **Interpretability**: Logistic Regression's coefficients (Phase 4A) give a single global, signed direction per feature; Random Forest's impurity/permutation importances (`random_forest_feature_importance_v1.csv` / `random_forest_permutation_importance_v1.csv`) only rank feature usefulness and cannot express direction or interactions directly - trees can, importances alone don't show them.

**No final project model is declared here.** XGBoost has not yet been evaluated, and the internal test set has not been used for this or any comparison.
