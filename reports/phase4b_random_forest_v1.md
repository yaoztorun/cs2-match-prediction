# Phase 4B - Model 2: Random Forest Classifier V1

## 1. Why Random Forest is Model 2

Model 2 in the project proposal's model lineup, testing whether a nonlinear tree ensemble improves over the linear Logistic Regression baseline (Model 1) when given exactly the same leakage-safe historical information.

## 2. Why a tree ensemble may improve over linear Logistic Regression

Logistic Regression can only combine features additively/linearly (in the standardized feature space). A Random Forest can learn interactions and thresholds automatically, e.g. `elo_diff` mattering differently depending on `bestOf`, or on whether both teams have enough history to trust their historical stats - relationships Logistic Regression cannot represent without explicit interaction terms.

## 3. Exact fixed V1 configuration

`sklearn.ensemble.RandomForestClassifier` is explicitly permitted for this model (not required from scratch, unlike Model 1). Single fixed baseline, **no tuning**:

```
n_estimators=300, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, class_weight=None, random_state=42, n_jobs=-1
```

## 4. Reuse of the identical chronological split

`data/modeling/series_split_v1.csv` from Phase 4A is reused byte-for-byte, **not regenerated**: train=6,619 unique historical matches, validation=1,419, test=1,418 (sealed, not loaded as a dataframe anywhere in this phase). This guarantees Logistic Regression and Random Forest are compared on exactly the same historical matches.

## 5. Mirroring methodology

Identical raw-mirroring policy to Logistic Regression (shared implementation in `scripts/preprocessing_common.py`): 6,619 unique historical training matches are each mirrored once (directional diffs negated, symmetric/context columns unchanged, target flipped) and concatenated with the originals, producing **13,238 training observations** - important to say precisely: this is 13,238 augmented *observations* fed to the model, not 13,238 independent matches; there are still only 6,619 unique underlying historical matches in the training partition. Augmented target mean: **0.5** (exactly 0.5). Validation and test are never mirrored.

## 6. Random-Forest-specific preprocessing

Same 17 raw Phase-3 whitelist features, same `bestOf`/`tier` reference-dummy encoding as Logistic Regression, but **no standardization** - tree splits are invariant to monotonic per-feature rescaling. Only train medians (fit on the augmented training set only) are used, for imputing `days_since_last_match_diff`'s missingness. 19 transformed features (same names/order as Logistic Regression's 19). Saved to `data/modeling/random_forest_preprocessing_v1.json`.

## 7. Train metrics (unmirrored original orientation)

- n = 6,619
- Accuracy: 0.9863 (majority-class reference: 0.5463)
- Precision: 0.9806
- Recall: 0.9945
- F1: 0.9875
- ROC-AUC: 0.9995
- Log loss: 0.2065
- Brier score: 0.0394
- Confusion matrix [[TN,FP],[FN,TP]]: [[2932, 71], [20, 3596]]

## 8. Validation metrics

- n = 1,419
- Accuracy: 0.5927 (majority-class reference: 0.5532)
- Precision: 0.6154
- Recall: 0.7032
- F1: 0.6564
- ROC-AUC: 0.6278
- Log loss: 0.6679
- Brier score: 0.2365
- Confusion matrix [[TN,FP],[FN,TP]]: [[289, 345], [233, 552]]

## 9. Train-validation overfitting analysis

- Accuracy gap (train - val): **+0.3936**
- ROC-AUC gap (train - val): **+0.3716**
- Log loss gap (val - train): **+0.4614**
- Brier gap (val - train): **+0.1971**

The fixed baseline uses `max_depth=None, min_samples_leaf=1` - unrestricted trees that can grow until every leaf is (near-)pure, which can memorize the training set. A large gap here is expected and is evidence of overfitting in this untuned baseline, documented as observed behavior - **this is not treated as a bug, and V1 does not react by changing `max_depth`/`min_samples_leaf`/etc.; that is explicitly deferred to a future tuning phase.**

## 10. ROC interpretation

`reports/figures/random_forest_roc_v1.png` (validation only). ROC-AUC = 0.6278 vs. chance = 0.5 and Logistic Regression's 0.6431.

## 11. Calibration interpretation

`reports/figures/random_forest_calibration_v1.png`, diagnostic only - no isotonic/Platt/temperature correction applied. Per-bin mean predicted probability vs. empirical win rate:

| bin | n | mean predicted | empirical win rate |
|---|---|---|---|
| [0.0,0.1) | 3 | 0.040 | 1.000 |
| [0.1,0.2) | 9 | 0.173 | 0.222 |
| [0.2,0.3) | 46 | 0.259 | 0.283 |
| [0.3,0.4) | 163 | 0.355 | 0.429 |
| [0.4,0.5) | 286 | 0.449 | 0.486 |
| [0.5,0.6) | 451 | 0.549 | 0.530 |
| [0.6,0.7) | 343 | 0.646 | 0.673 |
| [0.7,0.8) | 103 | 0.737 | 0.718 |
| [0.8,0.9) | 12 | 0.829 | 0.917 |
| [0.9,1.0) | 3 | 0.921 | 1.000 |

Validation Brier score: 0.2365. Validation log loss: 0.6679.

## 12. Probability-distribution interpretation

`reports/figures/random_forest_probability_distribution_v1.png` (validation only). Percentiles of predicted P(team1 wins):

| min | p5 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| 0.024 | 0.310 | 0.447 | 0.543 | 0.627 | 0.728 | 0.934 |

## 13. Feature importance (impurity)

`reports/tables/random_forest_feature_importance_v1.csv`. **Impurity importance is NOT causal importance and may be biased toward variables offering many possible split points** (e.g. high-cardinality continuous features over binary flags). Top 8:

| rank | feature | impurity_importance |
|---|---|---|
| 1 | elo_diff | 0.1309 |
| 2 | overall_win_rate_diff | 0.0964 |
| 3 | days_since_last_match_diff | 0.0953 |
| 4 | format_win_rate_diff | 0.0947 |
| 5 | total_matches_before_diff | 0.0946 |
| 6 | history_matches_sum | 0.0835 |
| 7 | history_matches_min | 0.0749 |
| 8 | avg_series_margin_last_10_diff | 0.0712 |

## 14. Permutation importance (validation, ROC-AUC)

`reports/tables/random_forest_permutation_importance_v1.csv` (`sklearn.inspection.permutation_importance`, `scoring='roc_auc'`, `n_repeats=10`, `random_state=42`). Top 8:

| rank | feature | mean_importance | std_importance |
|---|---|---|---|
| 1 | elo_diff | 0.0614 | 0.0141 |
| 2 | total_matches_before_diff | 0.0296 | 0.0095 |
| 3 | avg_series_margin_last_10_diff | 0.0084 | 0.0051 |
| 4 | overall_win_rate_diff | 0.0057 | 0.0042 |
| 5 | avg_series_margin_last_5_diff | 0.0022 | 0.0017 |
| 6 | format_win_rate_diff | 0.0018 | 0.0035 |
| 7 | both_teams_have_5_matches | 0.0018 | 0.0011 |
| 8 | both_teams_have_10_matches | 0.0014 | 0.0011 |

Unlike Logistic Regression, where symmetric/context features (`bestOf`, `tier`, history confidence flags) received essentially zero standalone coefficients (mirroring mathematically cancels their additive effect), a tree ensemble can use these through interactions (e.g. `bestOf==BO3 AND elo_diff>threshold`) - so they may show non-zero importance here even though they carry no independent directional signal on their own. No features are removed based on these importances; this is interpretation only.

## 15. Side-symmetry diagnostic

Full detail in `reports/random_forest_symmetry_v1.md`. Summary: mean symmetry error **0.0311**, median **0.0267**, p95 **0.0733**, max **0.1533** (`abs(P(A beats B) - (1 - P(B beats A)))` on validation, using the same fitted preprocessing artifact for both orientations). Diagnostic only - no correction applied in V1.

## 16. Comparison to Logistic Regression V1

Full detail in `reports/model_comparison_lr_vs_rf_v1.md`. Headline: validation ROC-AUC 0.6278 (RF) vs. 0.6431 (LR), accuracy 0.5927 (RF) vs. 0.6110 (LR).

## Interpretation question: did Random Forest gain anything from nonlinear relationships/interactions?

**Not on this validation evidence** - Random Forest's ROC-AUC is -0.0152 relative to Logistic Regression's, i.e. worse; any nonlinear capacity gained was offset by the untuned baseline's overfitting (Section 9) rather than showing up as better validation discrimination. This conclusion is based only on the validation evidence above (ROC-AUC/accuracy delta, calibration, feature importances touching `bestOf`/`tier`/history-confidence, and the train-validation gap) - it is not assumed a priori.

## 17. Limitations

- Untuned baseline: `max_depth=None`/`min_samples_leaf=1` likely overfits (Section 9) - not fixed here.
- Impurity importance is biased toward high-cardinality continuous features; permutation importance is the more trustworthy of the two but is still only a ranking, not a causal statement.
- Side-symmetry is not exact (Section 15) - relevant for any future real-match prediction use.
- Same 17 series-level features as Logistic Regression - no map/player-level detail.
- Single chronological train/validation split - no cross-validation.
- `n_jobs=-1` and sklearn's internal tie-breaking mean results are reproducible on this machine given `random_state=42`, but bitwise reproducibility across different sklearn/BLAS versions or hardware is not guaranteed.

## 18. What is deferred to tuning

`n_estimators`, `max_depth`, `min_samples_split`/`min_samples_leaf`, `max_features`, `class_weight`, and `criterion` search; probability calibration (isotonic/Platt/temperature); symmetrized-probability correction; feature selection based on importances.

## Status

- **INTERNAL TEST = SEALED** - not opened or scored in this phase.
- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.
- **XGBOOST = NOT STARTED**.
- **NO RANDOM FOREST HYPERPARAMETER TUNING PERFORMED**.
