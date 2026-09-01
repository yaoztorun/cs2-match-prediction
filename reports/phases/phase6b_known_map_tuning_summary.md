# Phase 6B - Known-Map Tuning Summary (TRAIN-only)

Everything in this document was produced **before the main map validation partition was ever opened**. Nothing here is a generalization estimate: the four chronological folds are the same folds the configurations were selected against, so these are development numbers.

**Task: predict the winner of one specific, user-selected map (`team1_map_win`).** These figures are not comparable with the pre-veto series models' accuracies - different target, different task.

## TRAIN-CV reference baselines (never tuned)

| baseline | n | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|
| 0.5 constant | 6541 | 0.5314 | 0.5000 | 0.6931 | 0.2500 |
| overall ELO | 6541 | 0.5791 | 0.6077 | 0.6757 | 0.2413 |
| map ELO | 6541 | 0.5438 | 0.5649 | 0.6872 | 0.2470 |

## Selected configurations (frozen before validation)

**Random Forest** - `random_007`, via secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC):

```
max_depth=7, min_samples_leaf=5, min_samples_split=10, max_features=sqrt, criterion=log_loss, n_estimators=400, bootstrap=True
```
CV mean log loss 0.6681 ± 0.0054, CV mean ROC-AUC 0.6182.

**XGBoost** - `random_013`, via secondary (log-loss tie within epsilon, resolved by highest mean ROC-AUC):

```
learning_rate=0.03, max_depth=2, min_child_weight=10, subsample=0.75, colsample_bytree=0.85, gamma=5.0, reg_alpha=0.01, reg_lambda=10.0
```
`best_iteration` by fold [68, 361, 74, 172] -> **final_n_estimators = 124** (round(median(best_iteration + 1 across the 4 outer folds))), frozen. CV mean log loss 0.6709 ± 0.0075, CV mean ROC-AUC 0.6161.

## Selected-config out-of-fold metrics (pooled, TRAIN-only)

Each selected configuration re-run across the same four folds in its **final deployment form** (XGBoost with the frozen tree count and no early stopping), so these predictions match how the final models are actually fitted.

| model | n | accuracy | precision | recall | F1 | ROC-AUC | log loss | Brier | series-macro log loss | series-macro Brier | series-macro accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 6541 | 0.5834 | 0.5949 | 0.6772 | 0.6334 | 0.6179 | 0.6682 | 0.2380 | 0.6632 | 0.2356 | 0.5921 |
| xgboost | 6541 | 0.5843 | 0.5956 | 0.6784 | 0.6343 | 0.6164 | 0.6695 | 0.2385 | 0.6645 | 0.2360 | 0.5968 |
| ensemble | 6541 | 0.5846 | 0.5961 | 0.6772 | 0.6341 | 0.6184 | 0.6683 | 0.2379 | 0.6633 | 0.2355 | 0.5962 |

Series-macro figures average each series' own per-map mean, then average those equally across match_ids, so a BO5 does not outweigh a BO1. Map-level metrics remain PRIMARY - the task is map prediction. No per-series ROC-AUC is computed: most series have too few maps for it to mean anything.

## Frozen RF/XGB probability ensemble

`p_ensemble = w * p_rf + (1 - w) * p_xgb` over the 11 predefined weights ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], selected on pooled TRAIN-only OOF log loss alone.

| w_rf | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|
| 0.0 | 0.5843 | 0.6164 | 0.6695 | 0.2385 |
| 0.1 | 0.5837 | 0.6170 | 0.6692 | 0.2383 |
| 0.2 | 0.5826 | 0.6175 | 0.6690 | 0.2382 |
| 0.3 | 0.5840 | 0.6179 | 0.6687 | 0.2381 |
| 0.4 | 0.5840 | 0.6182 | 0.6686 | 0.2380 |
| 0.5 | 0.5842 | 0.6183 | 0.6684 | 0.2380 |
| 0.6 **<-- selected** | 0.5846 | 0.6184 | 0.6683 | 0.2379 |
| 0.7 | 0.5851 | 0.6184 | 0.6682 | 0.2379 |
| 0.8 | 0.5855 | 0.6184 | 0.6682 | 0.2379 |
| 0.9 | 0.5829 | 0.6182 | 0.6682 | 0.2379 |
| 1.0 | 0.5834 | 0.6179 | 0.6682 | 0.2380 |

**Frozen: w_rf = 0.6** via secondary (log-loss tie within epsilon, resolved by highest ROC-AUC). Saved to `data/modeling/map_ensemble_v1_config.json` before any validation data was opened.

## Provisional development winner

Probability quality first: log loss -> ROC-AUC -> Brier -> accuracy, on TRAIN-only OOF.

| rank | model | log loss | ROC-AUC | Brier | accuracy |
|---|---|---|---|---|---|
| 1 | random_forest | 0.6682 | 0.6179 | 0.2380 | 0.5834 |
| 2 | ensemble | 0.6683 | 0.6184 | 0.2379 | 0.5846 |
| 3 | xgboost | 0.6695 | 0.6164 | 0.2385 | 0.5843 |

Provisional winner: **random_forest**. This is NOT a final project model - the internal TEST partition remains the final unbiased internal evaluation and has not been opened.

## Feature-family grouped permutation importance (TRAIN-only CV)

Each family's columns are permuted **jointly** (one shared row permutation per repeat), so correlated features inside a family are broken together rather than one at a time. Values are the mean fold-validation ROC-AUC decrease, averaged over the four folds.

| family | label | n features | RF AUC decrease | XGB AUC decrease |
|---|---|---|---|---|
| A | original series V1 (ELO / win rate / activity) | 15 | +0.0036 | +0.0093 |
| B | map-pool depth and order statistics | 14 | +0.0119 | +0.0126 |
| C | same-map matchup advantage | 6 | +0.0019 | +0.0010 |
| D | map-pool confidence | 10 | -0.0000 | +0.0001 |
| E | opponent-strength / residual form | 5 | +0.0093 | +0.0124 |
| F | time-decayed form | 3 | +0.0013 | +0.0017 |
| G | form confidence | 4 | -0.0002 | -0.0000 |
| H | player performance | 10 | +0.0072 | +0.0114 |
| I | roster stability | 4 | +0.0007 | +0.0000 |
| J | roster / player confidence | 7 | +0.0008 | -0.0005 |
| K | map-specific historical strength (selected map) | 14 | +0.0032 | +0.0018 |
| L | categorical map / bestOf / tier context | 14 | +0.0000 | +0.0000 |

### Does exact selected-map history add importance beyond overall ELO/form?

Family **K** (map-specific historical strength, 14 features) scores +0.0032 (RF) and +0.0018 (XGB) AUC decrease, against family **A** (the original series-level ELO/form block) at +0.0036 (RF) and +0.0093 (XGB). Read descriptively: a grouped permutation measures how much a model *relies on* a family given everything else it can see, not how much information that family contains in isolation - correlated families mask one another. No feature is removed in this phase.

## Top individual features by TRAIN-only CV permutation importance

**Random Forest** (top 12 by fold-validation permutation importance):

| rank | feature | family | permutation | impurity_importance_mean |
|---|---|---|---|---|
| 1 | avg_opponent_elo_last_10_diff | E | +0.0039 | 0.0328 |
| 2 | map_pool_total_matches_diff | B | +0.0026 | 0.0246 |
| 3 | avg_opponent_elo_last_5_diff | E | +0.0019 | 0.0274 |
| 4 | roster_mean_kast_diff | H | +0.0018 | 0.0258 |
| 5 | elo_diff | A | +0.0017 | 0.0372 |
| 6 | total_matches_before_diff | A | +0.0013 | 0.0185 |
| 7 | map_pool_best_elo_diff | B | +0.0012 | 0.0300 |
| 8 | map_matches_before_diff | K | +0.0012 | 0.0153 |
| 9 | performance_residual_all_diff | E | +0.0011 | 0.0261 |
| 10 | map_pool_experienced_maps_diff | B | +0.0009 | 0.0118 |
| 11 | roster_top_kast_diff | H | +0.0009 | 0.0173 |
| 12 | days_since_map_played_diff | K | +0.0008 | 0.0141 |

**XGBoost** (top 12 by fold-validation permutation importance):

| rank | feature | family | permutation | gain_mean |
|---|---|---|---|---|
| 1 | elo_diff | A | +0.0088 | 37.1298 |
| 2 | avg_opponent_elo_last_10_diff | E | +0.0068 | 19.1567 |
| 3 | roster_mean_kast_diff | H | +0.0061 | 18.9011 |
| 4 | map_pool_best_elo_diff | B | +0.0039 | 29.3578 |
| 5 | map_pool_total_matches_diff | B | +0.0029 | 24.1164 |
| 6 | avg_opponent_elo_last_5_diff | E | +0.0013 | 12.1802 |
| 7 | roster_top_kd_balance_diff | H | +0.0012 | 11.9717 |
| 8 | total_matches_before_diff | A | +0.0010 | 11.8612 |
| 9 | map_pool_experienced_maps_diff | B | +0.0010 | 16.3484 |
| 10 | map_matches_before_diff | K | +0.0009 | 12.6501 |
| 11 | map_matchup_median_smoothed_wr_advantage | C | +0.0008 | 11.5901 |
| 12 | roster_top_kast_diff | H | +0.0006 | 11.9899 |

Full tables in `reports/tables/map_rf_feature_importance_v1.csv` and `map_xgb_feature_importance_v1.csv`. Diagnostic only - no feature selection is performed in Phase 6B.

## Status at this point

- RF configuration: **FROZEN**
- XGBoost structural configuration and `final_n_estimators`: **FROZEN**
- Ensemble weight: **FROZEN**
- Main map VALIDATION: **not yet opened**
- TEST: **SEALED**
- Cologne: **UNTOUCHED**
