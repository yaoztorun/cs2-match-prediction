# Phase 5B.1 - Paired V1 vs V2 Map-Pool Feature Evaluation (TRAIN-only CV)

**Framing.** The 4 chronological CV folds used below (`data/modeling/random_forest_cv_folds_v2.csv`) are the SAME folds RF V2's and XGB V2's frozen hyperparameters were originally selected against in Phase 4B.1/4C.1. This report is therefore a **paired feature ablation** under a fixed, previously-selected model configuration - it answers *"with the model configuration held fixed, do V2 features improve performance relative to V1 features on the same development folds used during tuning?"* - not a fresh, unbiased estimate of future generalization. **The main 1,419-match VALIDATION partition was never loaded in this script and remains untouched.**

**XGBoost methodology note.** The original `xgboost_tuning_v2.py` used an inner early-stopping split with fold-specific `best_iteration`. This script instead uses XGB V2's single FROZEN final configuration (`n_estimators=98`, no early stopping, full outer-fold training history) identically for both the V1 and V2 arms - holding the model configuration identical across feature sets is what makes this comparison paired and feature-attributable. XGB-V1 numbers below are **not** expected to reproduce `reports/tables/xgboost_tuning_v2.csv` row-for-row; that is by design. RF has no early-stopping distinction, so RF-V1 below is expected to closely reproduce `random_forest_tuning_v2.csv`'s frozen-candidate row.

## Frozen configurations (loaded, never altered)

- **RF V2** (`random_009`): `{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 20, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True, 'criterion': 'gini'}`, called as `RandomForestClassifier(**params, random_state=42, n_jobs=-1)`.
- **XGB V2** (`random_002`): `{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 20, 'subsample': 0.6, 'colsample_bytree': 0.9, 'gamma': 2.0, 'reg_alpha': 0.01, 'reg_lambda': 1.0}` + fixed `{'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1}`, `n_estimators=98`, no early stopping, no eval_set - identical for V1 and V2.

## Random Forest (frozen RF V2 config)

| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |
|---|---|---|---|---|---|---|---|
| V1 | 0.6638±0.0075 | 0.6290±0.0153 | 0.2356±0.0036 | 0.5999 | 0.6490 | 0.7317 | +0.1027 |
| V2 | 0.6622±0.0073 | 0.6315±0.0157 | 0.2349±0.0035 | 0.5959 | 0.6479 | 0.7590 | +0.1274 |

### Paired fold-wise deltas (V2 - V1; negative=better for Log Loss/Brier, positive=better for ROC-AUC/Accuracy/F1)

| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| 1 | -0.0019 | +0.0021 | -0.0009 | -0.0151 | -0.0084 |
| 2 | -0.0026 | +0.0068 | -0.0013 | -0.0060 | -0.0026 |
| 3 | -0.0018 | +0.0004 | -0.0008 | +0.0008 | +0.0013 |
| 4 | +0.0003 | +0.0009 | +0.0001 | +0.0045 | +0.0053 |
| **mean** | **-0.0015** | **+0.0026** | **-0.0007** | **-0.0040** | **-0.0011** |

Log loss improved (V2 better) in **3/4** folds; ROC-AUC improved in **4/4** folds; Brier improved in **3/4** folds.

Differences of only a few thousandths in mean CV log loss/ROC-AUC should not be over-interpreted - this is a single paired ablation on 4 folds, not a significance test.

## XGBoost (frozen XGB V2 config)

| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |
|---|---|---|---|---|---|---|---|
| V1 | 0.6649±0.0073 | 0.6277±0.0182 | 0.2362±0.0035 | 0.5974 | 0.6490 | 0.6672 | +0.0394 |
| V2 | 0.6632±0.0065 | 0.6306±0.0150 | 0.2354±0.0031 | 0.5974 | 0.6482 | 0.6901 | +0.0595 |

### Paired fold-wise deltas (V2 - V1; negative=better for Log Loss/Brier, positive=better for ROC-AUC/Accuracy/F1)

| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| 1 | -0.0032 | +0.0083 | -0.0015 | +0.0030 | -0.0021 |
| 2 | -0.0025 | +0.0041 | -0.0012 | +0.0060 | +0.0128 |
| 3 | +0.0008 | -0.0052 | +0.0004 | -0.0136 | -0.0117 |
| 4 | -0.0020 | +0.0042 | -0.0009 | +0.0045 | -0.0024 |
| **mean** | **-0.0017** | **+0.0029** | **-0.0008** | **-0.0000** | **-0.0008** |

Log loss improved (V2 better) in **3/4** folds; ROC-AUC improved in **3/4** folds; Brier improved in **3/4** folds.

Differences of only a few thousandths in mean CV log loss/ROC-AUC should not be over-interpreted - this is a single paired ablation on 4 folds, not a significance test.

## V2-only feature importance (descriptive, no feature selection performed)

RF impurity importance and XGB gain both computed from a full-augmented-TRAIN refit of the frozen V2 configuration (never touching validation/test); permutation importance is the **average across the 4 CV folds' own fold-validation slices** (still entirely inside TRAIN-only CV, never the main validation partition).

### RF: top 10 by impurity importance

| rank | feature | family | impurity importance | fold-val permutation (mean±std) |
|---|---|---|---|---|
| 1 | elo_diff | A_original_v1 | 0.1245 | 0.0087±0.0024 |
| 2 | map_pool_best_elo_diff | B_pool_depth | 0.0690 | 0.0028±0.0019 |
| 3 | map_pool_total_matches_diff | B_pool_depth | 0.0597 | 0.0052±0.0043 |
| 4 | map_pool_second_best_elo_diff | B_pool_depth | 0.0459 | 0.0006±0.0014 |
| 5 | total_matches_before_diff | A_original_v1 | 0.0434 | 0.0023±0.0037 |
| 6 | map_matchup_mean_elo_advantage | C_same_map_matchup | 0.0396 | -0.0004±0.0019 |
| 7 | map_pool_mean_elo_diff | B_pool_depth | 0.0352 | -0.0000±0.0023 |
| 8 | overall_win_rate_diff | A_original_v1 | 0.0328 | 0.0008±0.0034 |
| 9 | map_pool_third_best_elo_diff | B_pool_depth | 0.0323 | -0.0005±0.0006 |
| 10 | map_pool_mean_normalized_margin_diff | B_pool_depth | 0.0275 | 0.0016±0.0014 |

### XGB: top 10 by gain

| rank | feature | family | gain | weight | fold-val permutation (mean±std) |
|---|---|---|---|---|---|
| 1 | elo_diff | A_original_v1 | 44.7661 | 184 | 0.0292±0.0098 |
| 2 | map_pool_total_matches_diff | B_pool_depth | 26.7213 | 63 | 0.0110±0.0061 |
| 3 | map_pool_best_elo_diff | B_pool_depth | 23.5136 | 91 | 0.0035±0.0037 |
| 4 | map_pool_experienced_maps_diff | B_pool_depth | 17.2246 | 41 | 0.0031±0.0045 |
| 5 | map_pool_size_diff | B_pool_depth | 15.4206 | 61 | 0.0017±0.0029 |
| 6 | map_pool_second_best_elo_diff | B_pool_depth | 13.9462 | 12 | 0.0003±0.0017 |
| 7 | map_matchup_positive_advantage_balance | C_same_map_matchup | 13.1311 | 5 | -0.0001±0.0001 |
| 8 | map_pool_total_matches_min | D_coverage_confidence | 12.3056 | 29 | 0.0001±0.0006 |
| 9 | map_pool_mean_elo_diff | B_pool_depth | 12.2190 | 13 | 0.0000±0.0015 |
| 10 | map_matchup_median_smoothed_wr_advantage | C_same_map_matchup | 12.2109 | 47 | -0.0003±0.0006 |

### Do any map features beat the two named V1 references?

Specifically checking whether any `map_pool_*`/`map_matchup_*`/coverage feature outranks `elo_diff` and `total_matches_before_diff` in validation-fold permutation importance:

- **RF**: `elo_diff`=0.0087, `total_matches_before_diff`=0.0023 (fold-val permutation mean). 2 of 30 map-derived features exceed the lower of the two: `map_pool_total_matches_diff`, `map_pool_best_elo_diff`
- **XGB**: `elo_diff`=0.0292, `total_matches_before_diff`=0.0020 (fold-val permutation mean). 4 of 30 map-derived features exceed the lower of the two: `map_pool_total_matches_diff`, `map_pool_best_elo_diff`, `map_pool_experienced_maps_diff`, `map_pool_mean_normalized_margin_diff`

## Family analysis (which map-pool families carry meaningful validation-fold permutation importance)

A = original 17 Phase-3 features, B = pool-depth/order-statistics (`map_pool_*` diffs), C = same-map matchup advantages (`map_matchup_*` advantages), D = map coverage/confidence (pool-size/union/shared-count/coverage flags). Not causal - descriptive only.

**RF** - mean / summed fold-validation permutation importance by family:

| family | n features | mean permutation importance | summed permutation importance |
|---|---|---|---|
| A_original_v1 | 19 | 0.0008 | 0.0157 |
| B_pool_depth | 14 | 0.0008 | 0.0107 |
| C_same_map_matchup | 6 | -0.0002 | -0.0013 |
| D_coverage_confidence | 10 | -0.0000 | -0.0002 |

**XGB** - mean / summed fold-validation permutation importance by family:

| family | n features | mean permutation importance | summed permutation importance |
|---|---|---|---|
| A_original_v1 | 19 | 0.0018 | 0.0348 |
| B_pool_depth | 14 | 0.0015 | 0.0216 |
| C_same_map_matchup | 6 | -0.0003 | -0.0015 |
| D_coverage_confidence | 10 | 0.0000 | 0.0000 |

## Answering the brief

**RF**: mean CV log loss improved (-0.0015, 3/4 folds better); mean CV ROC-AUC improved (+0.0026, 4/4 folds better); Brier improved (-0.0007, 3/4 folds better).
**XGB**: mean CV log loss improved (-0.0017, 3/4 folds better); mean CV ROC-AUC improved (+0.0029, 3/4 folds better); Brier improved (-0.0008, 3/4 folds better).

## Do the map-pool features appear to add real predictive information beyond Feature Set V1?

RF verdict: **HELP**. XGB verdict: **HELP**. Combined: **HELP**. Reported using cautious language given differences on the order of a few thousandths are within the noise this small a fold count can resolve, and given the framing note above (paired ablation on the same folds used for hyperparameter selection, not an independent generalization estimate).

## Conclusion

**MAP FEATURES HELP**

- **TEST = SEALED**
- **COLOGNE = UNTOUCHED**
- **MAIN VALIDATION = NOT USED**
