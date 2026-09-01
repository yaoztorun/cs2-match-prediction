# Phase 4C.1 - XGBoost V2 Results (Chronologically Tuned)

**LR V1 = UNTUNED. RF V1 = UNTUNED. RF V2 = TUNED. XGB V1 = UNTUNED. XGB V2 = TUNED.**

## 1. Why XGBoost tuning was performed

XGBoost V1 was a fixed, untuned baseline with a train->validation ROC-AUC gap of +0.1187 (0.7514 -> 0.6327) and validation log loss 0.6608. This phase asks only whether chronological tuning can improve XGBoost's generalization and probability quality using exactly the same Phase-3 V1 information - no feature changes.

## 2. Temporal CV methodology

Outer folds are the **same** expanding-window folds used for RF V2 (`data/modeling/random_forest_cv_folds_v2.csv`, reused byte-identically). Within each outer fold the training history is split chronologically again into INNER FIT (earliest ~82.5%) and INNER EARLY STOP (latest ~17.5%), giving three strictly ordered temporal stages:

```
INNER FIT  <  INNER EARLY STOP  <  OUTER FOLD VALIDATION
```

Asserted for every candidate x fold, with no exact-timestamp group crossing either boundary. Mirroring applies to inner-fit only; preprocessing is fit on the mirrored inner-fit only; inner-early-stop and outer-fold-validation are never mirrored. Fold TRAIN metrics use the **original unmirrored inner-fit rows**.

## 3. Why early stopping was allowed only inside TRAIN folds

Early stopping needs a holdout to decide when to stop. Using the *outer fold-validation* block for that would make the candidate's score optimistic, because the model would already have consulted that block. So a separate inner early-stop block - still entirely inside the global TRAIN period - serves that role, and the outer fold-validation block is used **only** for scoring. The 1,419-match main validation partition was never loaded by the tuning script at all.

## 4. Candidate search strategy

40 candidates = 10 deterministic anchors + 30 `RandomState(42)` random draws over the specified space, de-duplicated. `n_estimators` was **not** a search dimension: each fold used `n_estimators=2000` as a cap with `early_stopping_rounds=100`. API used: xgboost 3.4.0: early_stopping_rounds passed as an XGBClassifier constructor keyword + fit(eval_set=[(inner_early_stop)]).

The anchor `xgb_v1_structure_reference` reproduces XGBoost V1's structural hyperparameters and **was eligible for selection** - CV could legitimately have concluded V1's structure was fine and only its fixed 300-round count was suboptimal.

## 5. Hyperparameter-selection criterion

1) PRIMARY: lowest mean CV log loss. 2) EQUIVALENCE: candidates within 0.002 of the best mean log loss are treated as essentially equivalent. 3) SECONDARY: highest mean CV ROC-AUC. 4) TERTIARY: lower CV log-loss standard deviation. 5) COMPLEXITY (deterministic, in order): lower max_depth -> higher min_child_weight -> higher gamma -> higher reg_lambda -> higher reg_alpha -> lower median effective tree count -> candidate_id lexicographic order. subsample/colsample_bytree are tunable predictive hyperparameters and are deliberately NOT used as a model-complexity ranking.

Accuracy was explicitly not an objective (the project consumes probabilities for tournament simulation, so log loss leads).

## 6. Selected structural parameters

Candidate **`random_002`**, chosen via secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC).

```
learning_rate=0.02, max_depth=4, min_child_weight=20, subsample=0.6, colsample_bytree=0.9, gamma=2.0, reg_alpha=0.01, reg_lambda=1.0
```

CV: log loss 0.6651 ± 0.0069, ROC-AUC 0.6269 ± 0.0141, Brier 0.2363, accuracy 0.5976.

## 7. Best-iteration behavior across folds

`best_iteration` by fold (1-4): **[72, 80, 149, 115]**; median = 97.5.

## 8. final_n_estimators derivation

Rule (fixed before the search): `round(median(best_iteration + 1 across the 4 folds))` -> round(median([73, 81, 150, 116])) = **98**. Derived from CV only; the main validation partition played no part.

## 9. Train metrics (original unmirrored)

- n = 6,619
- Accuracy: 0.6170
- Precision: 0.6336
- Recall: 0.7091
- F1: 0.6692
- ROC-AUC: 0.6577
- Log loss: 0.6535
- Brier: 0.2308
- Confusion matrix [[TN,FP],[FN,TP]]: [[1520, 1483], [1052, 2564]]

## 10. Main validation metrics (one-shot)

Refit on 6,619 unique historical training matches -> 13,238 augmented training observations, `n_estimators=98`, no early stopping, no eval_set. Evaluated exactly once:

- n = 1,419
- Accuracy: 0.6117
- Precision: 0.6291
- Recall: 0.7261
- F1: 0.6742
- ROC-AUC: 0.6504
- Log loss: 0.6542
- Brier: 0.2311
- Confusion matrix [[TN,FP],[FN,TP]]: [[298, 336], [215, 570]]

## 11. XGB V1 vs XGB V2

| metric | XGB V1 (untuned) | XGB V2 (tuned) | delta |
|---|---|---|---|
| Accuracy | 0.5948 | 0.6117 | +0.0169 |
| ROC-AUC | 0.6327 | 0.6504 | +0.0178 |
| F1 | 0.6563 | 0.6742 | +0.0179 |
| Log loss | 0.6608 | 0.6542 | -0.0066 |
| Brier | 0.2345 | 0.2311 | -0.0034 |

## 12. XGB V2 vs RF V2 (both tuned)

| metric | RF V2 (tuned) | XGB V2 (tuned) | delta |
|---|---|---|---|
| Accuracy | 0.6068 | 0.6117 | +0.0049 |
| ROC-AUC | 0.6566 | 0.6504 | -0.0061 |
| F1 | 0.6690 | 0.6742 | +0.0051 |
| Log loss | 0.6514 | 0.6542 | +0.0028 |
| Brier | 0.2298 | 0.2311 | +0.0013 |

This is the one genuinely like-for-like comparison so far: both received chronological TRAIN-only tuning on the same folds.

## 13. XGB V2 vs LR V1

Validation ROC-AUC 0.6504 (XGB V2, tuned) vs 0.6431 (LR V1, untuned): +0.0074. **The tuned XGBoost outperformed the untuned Logistic Regression baseline.** LR has never been tuned, so this is not an algorithm-superiority claim.

## 14. Overfitting change

- XGB V1 train AUC 0.7514 -> validation 0.6327, gap **+0.1187**
- XGB V2 train AUC 0.6577 -> validation 0.6504, gap **+0.0073**

Did tuning reduce the gap? **Yes** (-0.1114). Did it improve validation ROC-AUC? **Yes**. Log loss? **Yes**. Brier? **Yes**. A successful V2 does not need higher training performance - lower train AUC with higher validation AUC is evidence of better generalization, not worse fitting.

## 15. Calibration

`reports/figures/xgboost_v2_calibration.png` (validation only, no correction applied):

| bin | n | mean predicted | empirical win rate |
|---|---|---|---|
| [0.2,0.3) | 23 | 0.277 | 0.217 |
| [0.3,0.4) | 151 | 0.360 | 0.338 |
| [0.4,0.5) | 339 | 0.449 | 0.469 |
| [0.5,0.6) | 491 | 0.556 | 0.552 |
| [0.6,0.7) | 338 | 0.639 | 0.716 |
| [0.7,0.8) | 77 | 0.728 | 0.740 |

Validation log loss 0.6542 and Brier 0.2311 vs. XGB V1 0.6608/0.2345, RF V2 0.6514/0.2298, LR V1 0.6564/0.2322.

## 16. Probability spread

| percentile | XGB V1 | XGB V2 |
|---|---|---|
| min | 0.100 | 0.232 |
| p5 | 0.289 | 0.349 |
| p25 | 0.438 | 0.452 |
| p50 | 0.555 | 0.550 |
| p75 | 0.653 | 0.610 |
| p95 | 0.799 | 0.704 |
| max | 0.948 | 0.777 |

p5-p95 spread: XGB V1 0.510 vs XGB V2 0.355 - regularization made V2's probabilities **more conservative (narrower)**. Whether that is *better* is judged by log loss/Brier above, not by spread alone.

## 17. Feature importance (gain)

`gain` = average improvement from splits on the feature; `weight` = number of splits; `normalized_gain` = share of total gain. Neither is causal importance. Top 8 (all 19 features are in `reports/tables/xgboost_feature_importance_v2.csv`):

| rank | feature | raw_gain | normalized_gain | weight |
|---|---|---|---|---|
| 1 | elo_diff | 34.6290 | 0.2015 | 284 |
| 2 | both_teams_have_5_matches | 15.1917 | 0.0884 | 8 |
| 3 | total_matches_before_diff | 14.2006 | 0.0826 | 191 |
| 4 | both_teams_have_10_matches | 14.1603 | 0.0824 | 1 |
| 5 | history_matches_min | 11.9335 | 0.0694 | 167 |
| 6 | overall_win_rate_diff | 9.8571 | 0.0573 | 77 |
| 7 | avg_series_margin_last_10_diff | 9.4269 | 0.0548 | 42 |
| 8 | win_rate_last_5_diff | 8.5605 | 0.0498 | 32 |

## 18. Permutation importance (validation, ROC-AUC)

| rank | feature | mean_importance | std_importance |
|---|---|---|---|
| 1 | elo_diff | 0.0989 | 0.0193 |
| 2 | total_matches_before_diff | 0.0360 | 0.0058 |
| 3 | history_matches_min | 0.0083 | 0.0031 |
| 4 | days_since_last_match_diff | 0.0027 | 0.0014 |
| 5 | avg_series_margin_last_10_diff | 0.0021 | 0.0008 |
| 6 | overall_win_rate_diff | 0.0012 | 0.0007 |
| 7 | history_matches_sum | 0.0012 | 0.0008 |
| 8 | avg_series_margin_last_5_diff | 0.0010 | 0.0007 |

### V1 vs V2 reliance on specific features

| feature | V1 norm. gain | V2 norm. gain | V1 perm | V2 perm |
|---|---|---|---|---|
| both_teams_have_5_matches | 0.1554 | 0.0884 | 0.0007 | 0.0003 |
| both_teams_have_10_matches | 0.0704 | 0.0824 | 0.0005 | 0.0001 |
| both_teams_have_history | 0.0704 | 0.0000 | 0.0000 | 0.0000 |
| elo_diff | 0.1090 | 0.2015 | 0.0901 | 0.0989 |
| total_matches_before_diff | 0.0572 | 0.0826 | 0.0523 | 0.0360 |

Of particular interest: `both_teams_have_5_matches`/`both_teams_have_10_matches` were gain-heavy but weakly generalizing in V1 (high gain, near-zero validation permutation importance). The table shows whether the more regularized V2 reduced that reliance. **No feature selection is performed.**

## 19. Symmetry

XGB V2: mean **0.009317**, median **0.007694**, p95 **0.023101**, max **0.058808** (XGB V1: 0.034423 / 0.026287 / 0.098018 / 0.214942). No assumption was made that tuning improves symmetry; detail in `reports/xgboost_symmetry_v2.md`. **Not symmetrized.**

## 20. Limitations

- Logistic Regression has still never been tuned; only RF V2 and XGB V2 are tuned, so no algorithm-superiority conclusion is available.
- Same 17 series-level features - no map-level or player-level information.
- Four chronological folds over ~2.6 years; CV differences of a few thousandths in log loss are within noise, which is exactly why the 0.002 equivalence epsilon and deterministic tie-break ladder exist.
- `final_n_estimators` comes from a median across only 4 folds.
- No probability calibration, no symmetrization, no threshold tuning.
- The main validation partition has now been used once for XGB V2; repeated future use would erode its independence.

## Status

- **TEST = SEALED** - not opened or scored.
- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.
- Main validation was evaluated exactly once, after the configuration was frozen.
- No tuning iteration followed the main-validation result.
- No final project model declared.
