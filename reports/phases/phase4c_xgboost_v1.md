# Phase 4C - Model 3: XGBoost Classifier V1

Terminology: **LR V1** = untuned from-scratch Logistic Regression. **RF V1** = untuned Random Forest. **RF V2** = chronologically tuned Random Forest. **XGBoost V1** = untuned fixed gradient-boosting baseline (this phase).

## Fixed V1 configuration

`xgboost.XGBClassifier` (library implementation - Model 1 was from-scratch only because it was adapted from the course lab). Single fixed baseline, **no tuning, no early stopping, no `eval_set`**:

```
objective=binary:logistic, eval_metric=logloss, n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method=hist
```

Installed `xgboost==3.4.0`; none required - the specified configuration ran as-is on xgboost 3.4.0. This is a deliberately moderate baseline (shallower trees than RF V1, shrinkage via `learning_rate`, row/feature subsampling, default-style L2) and **must not be described as an optimized configuration**.

## Data, split and mirroring

Reused `data/modeling/series_split_v1.csv` unchanged: **6,619 unique historical training matches**, 1,419 validation matches, 1,418 test matches (SEALED - never loaded). Mirroring applied to TRAIN only, producing **13,238 augmented training observations** (never described as 13,238 independent matches - each mirrored row is a synthetic relabeling of an already-counted match). Augmented target mean = **0.5** (exactly 0.5). Validation and test are never mirrored.

## Preprocessing and the missing-value decision

Same 17-feature whitelist and the same deterministic `bestOf`/`tier` reference-dummy encoding as LR/RF, giving the same 19 transformed columns. **No standardization** (tree splits are scale-invariant). **NaN preserved rather than median-imputed** (`missing_value_policy = preserve_nan_native_xgboost`): XGBoost natively learns a default split direction for missing values, so a cold-start team's unknown `days_since_last_match_diff` stays a distinguishable signal instead of collapsing onto the median. This affects exactly one column (926 NaN values in augmented train, 72 in validation; no other whitelist column has any missingness). A verified side benefit: because `mirror_raw_rows` negates NaN to NaN, preserving NaN makes the augmented directional-diff means **exactly 0.0**, whereas LR/RF median imputation left a small residual asymmetry. No validation/test information is used to fill anything.

## Train metrics (unmirrored)

- n = 6,619
- Accuracy: 0.6738
- Precision: 0.6912
- Recall: 0.7284
- F1: 0.7093
- ROC-AUC: 0.7514
- Log loss: 0.6000
- Brier: 0.2066
- Confusion matrix [[TN,FP],[FN,TP]]: [[1826, 1177], [982, 2634]]

## Validation metrics

- n = 1,419
- Accuracy: 0.5948 (majority-class reference: 0.5532)
- Precision: 0.6182
- Recall: 0.6994
- F1: 0.6563
- ROC-AUC: 0.6327
- Log loss: 0.6608
- Brier: 0.2345
- Confusion matrix [[TN,FP],[FN,TP]]: [[295, 339], [236, 549]]

## Interpretation questions

### A. Did boosting improve validation discrimination relative to LR V1 and RF V1?

XGBoost V1 validation ROC-AUC = **0.6327**, vs. LR V1 0.6431 (-0.0104) and RF V1 0.6278 (+0.0048). Boosting did not clearly improve discrimination over both untuned baselines.

### B. How does XGBoost V1 compare with tuned RF V2?

RF V2 validation ROC-AUC 0.6566 vs. XGBoost V1 0.6327 (-0.0239 for XGBoost). RF V2 remains ahead on discrimination - but note this is an **untuned** XGBoost against a **chronologically tuned** Random Forest, so the comparison is not like-for-like in tuning effort in either direction.

### C. Is there evidence of overfitting?

XGBoost V1 train->validation gaps: accuracy **+0.0790**, ROC-AUC **+0.1187** (train acc 0.6738 / AUC 0.7514). For context: LR V1 -0.0201 acc / -0.0220 AUC, RF V1 +0.3936 / +0.3716, RF V2 +0.0393 / +0.0550. XGBoost V1 shows a clearly smaller gap than untuned RF V1, indicating the shallower depth/shrinkage/subsampling controlled overfitting substantially. **No configuration change was made in response** - tuning is out of scope for Phase 4C.

### D. Are XGBoost probabilities better or worse (log loss / Brier / calibration)?

XGBoost V1 validation log loss **0.6608** and Brier **0.2345**, vs. LR V1 0.6564/0.2322, RF V1 0.6679/0.2365, RF V2 0.6514/0.2298 (lower is better for both). This matters because these probabilities will later feed the tournament simulator. Per-bin calibration (`reports/figures/xgboost_calibration_v1.png`, diagnostic only - no isotonic/Platt/temperature scaling applied):

| bin | n | mean predicted | empirical win rate |
|---|---|---|---|
| [0.0,0.1) | 1 | 0.100 | 0.000 |
| [0.1,0.2) | 18 | 0.175 | 0.111 |
| [0.2,0.3) | 63 | 0.261 | 0.349 |
| [0.3,0.4) | 179 | 0.354 | 0.441 |
| [0.4,0.5) | 270 | 0.453 | 0.493 |
| [0.5,0.6) | 328 | 0.551 | 0.537 |
| [0.6,0.7) | 337 | 0.644 | 0.626 |
| [0.7,0.8) | 153 | 0.745 | 0.680 |
| [0.8,0.9) | 66 | 0.837 | 0.818 |
| [0.9,1.0) | 4 | 0.921 | 1.000 |

### E. Which features appear most useful (gain vs. validation permutation importance)?

`gain` = average improvement in the split criterion contributed by splits on that feature; `weight` = how many times the feature was used as a split. `normalized_gain` is each feature's share of total gain. **Neither is causal importance**, and gain (like RF's impurity importance) is computed on training structure - validation **permutation importance** is the stronger generalization-oriented diagnostic. Top 8 by gain:

| rank | feature | raw_gain | normalized_gain | weight |
|---|---|---|---|---|
| 1 | both_teams_have_5_matches | 18.1802 | 0.1554 | 5 |
| 2 | elo_diff | 12.7463 | 0.1090 | 506 |
| 3 | both_teams_have_10_matches | 8.2347 | 0.0704 | 5 |
| 4 | both_teams_have_history | 8.2328 | 0.0704 | 1 |
| 5 | history_matches_min | 6.6940 | 0.0572 | 313 |
| 6 | total_matches_before_diff | 6.6881 | 0.0572 | 496 |
| 7 | overall_win_rate_diff | 5.0846 | 0.0435 | 432 |
| 8 | avg_series_margin_last_10_diff | 4.9295 | 0.0421 | 246 |

Top 8 by validation permutation importance (ROC-AUC, n_repeats=10):

| rank | feature | mean_importance | std_importance |
|---|---|---|---|
| 1 | elo_diff | 0.0901 | 0.0159 |
| 2 | total_matches_before_diff | 0.0523 | 0.0071 |
| 3 | avg_series_margin_last_5_diff | 0.0113 | 0.0027 |
| 4 | history_matches_min | 0.0086 | 0.0040 |
| 5 | avg_series_margin_last_10_diff | 0.0046 | 0.0020 |
| 6 | days_since_last_match_diff | 0.0044 | 0.0038 |
| 7 | history_matches_sum | 0.0020 | 0.0024 |
| 8 | overall_win_rate_diff | 0.0012 | 0.0036 |

No features were removed - this is interpretation only.

### F. Do ELO and historical experience remain the strongest robust signals?

By validation permutation importance, `elo_diff` ranks **#1** and `total_matches_before_diff` ranks **#2** (top 3 overall: ['elo_diff', 'total_matches_before_diff', 'avg_series_margin_last_5_diff']). Yes - ELO and historical-experience remain the strongest robust signals, consistent with LR V1's largest coefficients and RF V1/V2's permutation rankings.

### G. Did XGBoost make useful use of nonlinear interactions that LR cannot represent?

On this validation evidence, **no**: XGBoost V1's ROC-AUC is -0.0104 relative to LR V1's, i.e. worse. Whatever additional nonlinear capacity boosting has did not convert into better validation discrimination here.
 No claim beyond what the validation numbers support is made.

## Probability distribution

| min | p5 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| 0.100 | 0.289 | 0.438 | 0.555 | 0.653 | 0.799 | 0.948 |

Compare qualitatively with LR V1, RF V1 and RF V2's distributions (see their respective probability-distribution figures) to judge whether boosting produces more extreme or more conservative probabilities. See `reports/figures/xgboost_probability_distribution_v1.png`.

## Side-symmetry diagnostic

mean **0.034423**, median **0.026287**, p95 **0.098018**, max **0.214942**. Tree ensembles are not mathematically constrained to satisfy `P(A beats B) = 1 - P(B beats A)` - determinism does not imply antisymmetry - so no expected value was assumed in advance. Detail in `reports/xgboost_symmetry_v1.md`. **No symmetrization applied.**

## Comparison summary

Full five-way table in `reports/model_comparison_v1.md`.

## Limitations / deferred

- XGBoost V1 is **untuned**; RF V2 is the only tuned model so far, so cross-algorithm rankings here are not fair algorithm-superiority evidence.
- No probability calibration and no probability symmetrization applied.
- Same 17 series-level features - no map-level or player-level detail.
- Single chronological train/validation split; no cross-validation in this phase.
- Future XGBoost tuning must use chronological CV **inside TRAIN**, exactly as RF V2 did - never the main validation set, and never early stopping against it.

## Status

- **TEST = SEALED** - not opened or scored in this phase.
- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.
- No early stopping and no `eval_set` were used at any point.
- Exactly one fixed XGBoost configuration was trained; no tuning performed.
