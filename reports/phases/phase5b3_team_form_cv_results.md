# Phase 5B.3 - Paired V2 vs V3 Team-Form Feature Evaluation (TRAIN-only CV)

**Framing.** The 4 chronological CV folds below (`data/modeling/random_forest_cv_folds_v2.csv`) are the SAME folds RF V2's and XGB V2's frozen hyperparameters were originally selected against in Phase 4B.1/4C.1 - **before V3 existed**. This is a paired development-set feature ablation under a fixed, previously-selected model configuration, not an independent estimate of future generalization. The main 1,419-match VALIDATION partition was never loaded here.

**Asymmetric interpretation (read before the verdict below).** If V3 improves under these frozen configurations, that IS evidence the new form information adds predictive signal without needing retuning. If V3 does NOT improve, that does **not** mean the form features carry no useful signal - the correct conclusion is only that *V3 did not improve performance under the frozen, previously-selected model configurations*. A later V3-specific tune could use the richer feature space differently.

**Regression parity.** V2-arm fold metrics here are required (by `scripts/validate_phase5b3.py`) to match Phase 5B.1's own saved V2-arm rows in `reports/tables/series_feature_v1_v2_cv_comparison.csv` within a strict numeric tolerance - both use the identical data/configuration/protocol combination.

## Frozen configurations (loaded, never altered)

- **RF V2** (`random_009`): `{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 20, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True, 'criterion': 'gini'}`.
- **XGB V2** (`random_002`): `{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 20, 'subsample': 0.6, 'colsample_bytree': 0.9, 'gamma': 2.0, 'reg_alpha': 0.01, 'reg_lambda': 1.0}` + fixed `{'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1}`, `n_estimators=98`, no early stopping - identical for V2 and V3.

## Random Forest (frozen RF V2 config)

| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |
|---|---|---|---|---|---|---|---|
| V2 | 0.6622±0.0073 | 0.6315±0.0157 | 0.2349±0.0035 | 0.5959 | 0.6479 | 0.7590 | +0.1274 |
| V3 | 0.6583±0.0056 | 0.6392±0.0103 | 0.2331±0.0027 | 0.6063 | 0.6570 | 0.7705 | +0.1313 |

### Paired fold-wise deltas (V3 - V2; negative=better for Log Loss/Brier, positive=better for ROC-AUC/Accuracy/F1)

| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| 1 | -0.0067 | +0.0168 | -0.0032 | +0.0377 | +0.0334 |
| 2 | -0.0012 | +0.0008 | -0.0005 | -0.0023 | -0.0021 |
| 3 | -0.0039 | +0.0068 | -0.0018 | +0.0023 | -0.0042 |
| 4 | -0.0037 | +0.0062 | -0.0017 | +0.0038 | +0.0092 |
| **mean** | **-0.0039** | **+0.0077** | **-0.0018** | **+0.0104** | **+0.0091** |

Log loss improved (V3 better) in **4/4** folds; ROC-AUC improved in **4/4** folds; Brier improved in **4/4** folds.

## XGBoost (frozen XGB V2 config)

| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |
|---|---|---|---|---|---|---|---|
| V2 | 0.6632±0.0065 | 0.6306±0.0150 | 0.2354±0.0031 | 0.5974 | 0.6482 | 0.6901 | +0.0595 |
| V3 | 0.6603±0.0072 | 0.6368±0.0154 | 0.2340±0.0034 | 0.6037 | 0.6586 | 0.6950 | +0.0582 |

### Paired fold-wise deltas (V3 - V2; negative=better for Log Loss/Brier, positive=better for ROC-AUC/Accuracy/F1)

| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| 1 | -0.0019 | +0.0062 | -0.0009 | +0.0211 | +0.0273 |
| 2 | -0.0044 | +0.0092 | -0.0021 | +0.0008 | +0.0013 |
| 3 | -0.0024 | +0.0034 | -0.0011 | +0.0053 | +0.0070 |
| 4 | -0.0028 | +0.0059 | -0.0013 | -0.0023 | +0.0063 |
| **mean** | **-0.0029** | **+0.0062** | **-0.0014** | **+0.0062** | **+0.0105** |

Log loss improved (V3 better) in **4/4** folds; ROC-AUC improved in **4/4** folds; Brier improved in **4/4** folds.

## V3-only feature importance (descriptive, no feature selection performed)

RF impurity importance and XGB gain both from a full-augmented-TRAIN refit of the frozen V3 configuration; permutation importance is the average across the 4 CV folds' own fold-validation slices (still entirely inside TRAIN-only CV).

**Correlated-feature caveat.** Several of the 12 new form features are highly correlated (Phase 5B.2's own quality report found r up to 0.93 among `time_weighted_win_rate_diff`, `time_weighted_series_margin_diff` and `performance_residual_all_diff`). Permutation importance may be *shared* across correlated features - a near-zero INDIVIDUAL score does not prove a correlated feature or family carries no signal. Family-level GROUPED permutation importance (section below) is reported specifically to address this.

### RF: top 10 by impurity importance

| rank | feature | family | impurity importance | fold-val permutation (mean±std) |
|---|---|---|---|---|
| 1 | elo_diff | A_original_v1 | 0.0872 | 0.0038±0.0031 |
| 2 | map_pool_best_elo_diff | B_pool_depth | 0.0553 | 0.0015±0.0012 |
| 3 | avg_opponent_elo_last_10_diff | E_opponent_strength | 0.0495 | 0.0031±0.0011 |
| 4 | performance_residual_all_diff | E_opponent_strength | 0.0464 | 0.0016±0.0036 |
| 5 | map_pool_total_matches_diff | B_pool_depth | 0.0464 | 0.0037±0.0035 |
| 6 | time_weighted_series_margin_diff | F_time_decayed | 0.0379 | 0.0013±0.0011 |
| 7 | total_matches_before_diff | A_original_v1 | 0.0369 | 0.0023±0.0039 |
| 8 | map_pool_second_best_elo_diff | B_pool_depth | 0.0329 | 0.0004±0.0013 |
| 9 | map_pool_mean_elo_diff | B_pool_depth | 0.0288 | -0.0004±0.0013 |
| 10 | avg_opponent_elo_last_5_diff | E_opponent_strength | 0.0282 | 0.0025±0.0026 |

### XGB: top 10 by gain

| rank | feature | family | gain | weight | fold-val permutation (mean±std) |
|---|---|---|---|---|---|
| 1 | elo_diff | A_original_v1 | 62.0833 | 112 | 0.0094±0.0046 |
| 2 | map_pool_total_matches_diff | B_pool_depth | 32.0672 | 58 | 0.0102±0.0055 |
| 3 | performance_residual_all_diff | E_opponent_strength | 24.4144 | 26 | 0.0027±0.0050 |
| 4 | map_pool_best_elo_diff | B_pool_depth | 23.6805 | 71 | 0.0024±0.0024 |
| 5 | map_pool_experienced_maps_diff | B_pool_depth | 18.3018 | 33 | 0.0018±0.0022 |
| 6 | avg_opponent_elo_last_10_diff | E_opponent_strength | 17.4039 | 94 | 0.0031±0.0008 |
| 7 | map_pool_size_diff | B_pool_depth | 17.1291 | 49 | 0.0012±0.0021 |
| 8 | total_matches_before_diff | A_original_v1 | 15.5275 | 32 | 0.0013±0.0023 |
| 9 | map_pool_mean_smoothed_wr_diff | B_pool_depth | 15.1990 | 13 | -0.0002±0.0005 |
| 10 | map_matchup_shared_coverage | D_coverage_confidence | 14.1119 | 4 | -0.0001±0.0002 |

### Do any of the 12 new form features beat the four named references?

- **RF**: reference minimum = 0.0015 (of {elo_diff=0.0038, map_pool_total_matches_diff=0.0037, map_pool_best_elo_diff=0.0015, total_matches_before_diff=0.0023}). 3 of 12 new form features individually exceed it: `avg_opponent_elo_last_10_diff`, `avg_opponent_elo_last_5_diff`, `performance_residual_all_diff`
- **XGB**: reference minimum = 0.0013 (of {elo_diff=0.0094, map_pool_total_matches_diff=0.0102, map_pool_best_elo_diff=0.0024, total_matches_before_diff=0.0013}). 3 of 12 new form features individually exceed it: `avg_opponent_elo_last_10_diff`, `avg_opponent_elo_last_5_diff`, `performance_residual_all_diff`

## Family-level GROUPED permutation importance (E, F, G only; descriptive, NOT feature selection)

Each family's columns are permuted JOINTLY (one shared row-permutation across the whole group per repeat, 10 repeats, `random_state=42`), measuring the ROC-AUC decrease when that entire family's signal is destroyed at once - this is the correct way to read importance for a set of correlated features, since individually permuting each one leaves the others to compensate.

**RF**:

| family | mean ROC-AUC decrease (across 4 folds) | std across folds |
|---|---|---|
| E_opponent_strength | 0.0091 | 0.0043 |
| F_time_decayed | 0.0033 | 0.0034 |
| G_form_confidence | 0.0006 | 0.0011 |

**XGB**:

| family | mean ROC-AUC decrease (across 4 folds) | std across folds |
|---|---|---|
| E_opponent_strength | 0.0121 | 0.0043 |
| F_time_decayed | 0.0036 | 0.0035 |
| G_form_confidence | 0.0006 | 0.0008 |

Not used for feature selection, model changes, or tuning - descriptive only.

## Answering the brief

**RF**: mean CV log loss improved (-0.0039, 4/4 folds better); ROC-AUC improved (+0.0077, 4/4 folds better); Brier improved (-0.0018, 4/4 folds better).
**XGB**: mean CV log loss improved (-0.0029, 4/4 folds better); ROC-AUC improved (+0.0062, 4/4 folds better); Brier improved (-0.0014, 4/4 folds better).

## Conclusion

RF: **HELP**. XGB: **HELP**. Combined: **HELP**.

Both models improved on all three primary metrics under the frozen, previously-selected configurations - evidence the new form information adds predictive signal without needing retuning.

**MAP AND FORM FEATURES HELP** (this phase's verdict is about the 12 new form features specifically; family A-D behavior is unchanged from Phase 5B.1).

- **MAIN VALIDATION = NOT USED**
- **TEST = SEALED**
- **COLOGNE = UNTOUCHED**
