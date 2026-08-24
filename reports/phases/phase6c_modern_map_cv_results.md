# Phase 6C, Stage B - Known-Map V2-rich vs V3-modern-map TRAIN-only Ablation

**MAP VALIDATION = NOT USED. TEST = SEALED. COLOGNE = UNTOUCHED.** This script never opens `data/modeling/map_split_v1.csv` - only the TRAIN-only `data/modeling/map_cv_folds_v1.csv` manifest.

Identical, frozen Phase 6B configurations used for BOTH arms: RF `random_007` (max_depth=7, min_samples_leaf=5, min_samples_split=10, max_features=sqrt, criterion=log_loss, n_estimators=400, bootstrap=True), XGB `random_013` (learning_rate=0.03, max_depth=2, min_child_weight=10, subsample=0.75, colsample_bytree=0.85, gamma=5.0, reg_alpha=0.01, reg_lambda=10.0, `n_estimators=124`, no early stopping). Neither model was retuned for V3.

## Paired fold-wise deltas (V3 - V2)

Negative `delta_log_loss`/`delta_brier` and positive `delta_roc_auc`/`delta_accuracy`/`delta_f1` favor V3.

| fold | model | V2 log loss | V3 log loss | Δ log loss | V2 ROC-AUC | V3 ROC-AUC | Δ ROC-AUC | Δ accuracy | Δ Brier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | random_forest | 0.6699 | 0.6698 | -0.0001 | 0.6194 | 0.6210 | +0.0016 | -0.0039 | -0.0001 |
| 1 | xgboost | 0.6721 | 0.6708 | -0.0013 | 0.6149 | 0.6199 | +0.0050 | +0.0104 | -0.0007 |
| 2 | random_forest | 0.6756 | 0.6765 | +0.0009 | 0.5974 | 0.5932 | -0.0042 | +0.0011 | +0.0004 |
| 2 | xgboost | 0.6781 | 0.6772 | -0.0009 | 0.5917 | 0.5932 | +0.0016 | +0.0000 | -0.0004 |
| 3 | random_forest | 0.6657 | 0.6652 | -0.0006 | 0.6280 | 0.6279 | -0.0001 | +0.0099 | -0.0002 |
| 3 | xgboost | 0.6666 | 0.6666 | +0.0001 | 0.6276 | 0.6255 | -0.0021 | +0.0025 | +0.0001 |
| 4 | random_forest | 0.6610 | 0.6603 | -0.0007 | 0.6281 | 0.6308 | +0.0026 | +0.0050 | -0.0004 |
| 4 | xgboost | 0.6605 | 0.6597 | -0.0008 | 0.6338 | 0.6367 | +0.0028 | +0.0025 | -0.0004 |

Mean fold deltas:

| model | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| random_forest | -0.0001 | -0.0000 | -0.0001 | +0.0030 | +0.0026 |
| xgboost | -0.0007 | +0.0018 | -0.0004 | +0.0038 | +0.0048 |

## Pooled TRAIN-only OOF metrics

| arm | model | n | accuracy | ROC-AUC | log loss | Brier | series-macro log loss | series-macro accuracy |
|---|---|---|---|---|---|---|---|---|
| v2 | random_forest | 6541 | 0.5834 | 0.6179 | 0.6682 | 0.2380 | 0.6632 | 0.5921 |
| v2 | xgboost | 6541 | 0.5843 | 0.6164 | 0.6695 | 0.2385 | 0.6645 | 0.5968 |
| v3 | random_forest | 6541 | 0.5865 | 0.6177 | 0.6681 | 0.2379 | 0.6635 | 0.5965 |
| v3 | xgboost | 6541 | 0.5880 | 0.6183 | 0.6688 | 0.2381 | 0.6637 | 0.6010 |

## Additional correct maps (pooled TRAIN-only OOF)

| model | n | V2 correct | V3 correct | additional correct | pp accuracy gain |
|---|---|---|---|---|---|
| random_forest | 6541 | 3816 | 3836 | +20 | +0.306 |
| xgboost | 6541 | 3822 | 3846 | +24 | +0.367 |

## Coverage diagnostic: high_evidence (both_recent_map_history & roster_map_players_with_history_min>=5)

Predefined using ONLY pre-match V3 evidence. Descriptive only - no subgroup-specific model is trained.

| model | n | V2 log loss | V3 log loss | V2 ROC-AUC | V3 ROC-AUC | V2 accuracy | V3 accuracy |
|---|---|---|---|---|---|---|---|
| random_forest | 4979 | 0.6692 | 0.6692 | 0.6157 | 0.6157 | 0.5828 | 0.5863 |
| xgboost | 4979 | 0.6713 | 0.6705 | 0.6119 | 0.6144 | 0.5818 | 0.5832 |

## V3-only grouped permutation importance (TRAIN-only CV, fold-validation)

Each family's columns are permuted jointly. No feature is selected or removed based on this.

| model | family | label | n features | AUC decrease |
|---|---|---|---|---|
| random_forest | O | current-roster selected-map player performance | 9 | +0.0022 |
| random_forest | M | recent/opponent-adjusted selected-map team features | 8 | +0.0020 |
| random_forest | P | current-core selected-map continuity | 3 | +0.0007 |
| random_forest | N | map specialization (relative to overall/pool strength) | 5 | +0.0001 |
| xgboost | M | recent/opponent-adjusted selected-map team features | 8 | +0.0033 |
| xgboost | O | current-roster selected-map player performance | 9 | +0.0020 |
| xgboost | P | current-core selected-map continuity | 3 | +0.0005 |
| xgboost | N | map specialization (relative to overall/pool strength) | 5 | -0.0005 |

## Strongest new individual features (TRAIN-only CV permutation importance)

**Random Forest** (top 10):

| rank | feature | permutation | impurity |
|---|---|---|---|
| 1 | avg_opponent_elo_last_10_diff | +0.0024 | 0.0256 |
| 2 | map_pool_total_matches_diff | +0.0018 | 0.0204 |
| 3 | time_weighted_map_opponent_elo_diff | +0.0017 | 0.0225 |
| 4 | elo_diff | +0.0016 | 0.0316 |
| 5 | roster_mean_kast_diff | +0.0016 | 0.0217 |
| 6 | avg_opponent_elo_last_5_diff | +0.0011 | 0.0223 |
| 7 | total_matches_before_diff | +0.0010 | 0.0139 |
| 8 | map_pool_experienced_maps_diff | +0.0010 | 0.0085 |
| 9 | roster_map_mean_history_mass_diff | +0.0008 | 0.0175 |
| 10 | current_core_map_history_mass_diff | +0.0007 | 0.0159 |

**XGBoost** (top 10):

| rank | feature | permutation | gain |
|---|---|---|---|
| 1 | elo_diff | +0.0061 | 45.8959 |
| 2 | roster_mean_kast_diff | +0.0047 | 18.7699 |
| 3 | avg_opponent_elo_last_10_diff | +0.0038 | 21.4907 |
| 4 | time_weighted_map_opponent_elo_diff | +0.0032 | 18.5698 |
| 5 | map_pool_best_elo_diff | +0.0027 | 28.4960 |
| 6 | map_pool_total_matches_diff | +0.0026 | 27.2352 |
| 7 | roster_top_kd_balance_diff | +0.0016 | 11.5617 |
| 8 | avg_opponent_elo_last_5_diff | +0.0011 | 12.9203 |
| 9 | roster_map_mean_history_mass_diff | +0.0009 | 15.0234 |
| 10 | total_matches_before_diff | +0.0008 | 13.2838 |

## Interpretation (asymmetric, per the framing established in every prior ablation)

RF mean fold Δlog loss -0.0001, ΔROC-AUC -0.0000. XGB mean fold Δlog loss -0.0007, ΔROC-AUC +0.0018. If V3 improves under these frozen, previously-selected configurations, that is evidence the new features add signal without retuning. If V3 does not improve, the correct conclusion is only that it did not improve under the frozen configurations used here - not that the new selected-map information carries no signal at all. No feature is added, removed or reweighted based on these results; this is the final feature-engineering experiment for the known-map task.

## Status

- **MAP VALIDATION = NOT USED**
- **TEST = SEALED**
- **COLOGNE = UNTOUCHED**
- **NO POST-RESULT FEATURE CHANGES**
- **SRC = UNCHANGED**
