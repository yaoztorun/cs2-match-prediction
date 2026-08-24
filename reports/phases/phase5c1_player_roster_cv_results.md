# Phase 5C.1 - Paired V3 vs V4 Player/Roster Feature Evaluation (TRAIN-only CV)

**Framing.** The 4 chronological CV folds below (`data/modeling/random_forest_cv_folds_v2.csv`) are the SAME folds RF V2's and XGB V2's frozen hyperparameters were originally selected against, before V3 or V4 existed. This is a paired development-set feature ablation under a fixed, previously-selected model configuration, not an independent estimate of future generalization. The main 1,419-match VALIDATION partition was never loaded here.

**Asymmetric interpretation.** If V4 improves, that IS evidence the player/roster information adds signal without retuning. If overall improvement is small but the coverage-rich subgroup (`roster_form_players_min >= 5`) improves substantially, the correct reading is that player features appear useful where historical coverage exists but incomplete source coverage dilutes the aggregate gain - NOT that the information has no value. If V4 does not improve anywhere, the only correct conclusion is that it did not improve under these frozen, previously-selected configurations; a later V4-specific tune could use the feature space differently.

**Regression parity.** V3-arm fold metrics here are required (by `scripts/validate_phase5c1.py`) to match Phase 5B.3's own saved V3-arm rows in `reports/tables/series_feature_v2_v3_cv_comparison.csv` within a strict numeric tolerance.

## Frozen configurations (loaded, never altered)

- **RF V2** (`random_009`): `{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 20, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True, 'criterion': 'gini'}`.
- **XGB V2** (`random_002`): `{'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 20, 'subsample': 0.6, 'colsample_bytree': 0.9, 'gamma': 2.0, 'reg_alpha': 0.01, 'reg_lambda': 1.0}` + fixed `{'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1}`, `n_estimators=98`, no early stopping - identical for V3 and V4.

## Full-data results

### Random Forest (frozen RF V2 config)

| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |
|---|---|---|---|---|---|---|---|
| V3 | 0.6583±0.0056 | 0.6392±0.0103 | 0.2331±0.0027 | 0.6063 | 0.6570 | 0.7705 | +0.1313 |
| V4 | 0.6573±0.0066 | 0.6409±0.0121 | 0.2327±0.0031 | 0.6093 | 0.6580 | 0.7782 | +0.1372 |

Paired fold-wise deltas (V4 - V3):

| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| 1 | +0.0006 | -0.0012 | +0.0003 | -0.0106 | -0.0120 |
| 2 | -0.0027 | +0.0054 | -0.0012 | +0.0091 | +0.0060 |
| 3 | +0.0001 | +0.0001 | +0.0001 | +0.0068 | +0.0058 |
| 4 | -0.0023 | +0.0026 | -0.0009 | +0.0068 | +0.0043 |
| **mean** | **-0.0011** | **+0.0017** | **-0.0004** | **+0.0030** | **+0.0010** |

Log loss improved in **2/4** folds; ROC-AUC improved in **3/4**; Brier improved in **2/4**.

**Pooled TRAIN-only out-of-fold result** (all 4 folds' validation predictions concatenated):

- n = 5296 (V3) / 5296 (V4) | Accuracy 0.6063 -> 0.6093 (+0.30 pp) | ROC-AUC 0.6393 -> 0.6412 (+0.0018) | Log Loss 0.6583 -> 0.6573 (-0.0011) | Brier 0.2331 -> 0.2327 (-0.0004) | F1 0.6571 -> 0.6583 (+0.0012)
- Correct predictions: V3 = 3211, V4 = 3227, **V4 - V3 = +16 additional correct matches**.

### XGBoost (frozen XGB V2 config)

| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |
|---|---|---|---|---|---|---|---|
| V3 | 0.6603±0.0072 | 0.6368±0.0154 | 0.2340±0.0034 | 0.6037 | 0.6586 | 0.6950 | +0.0582 |
| V4 | 0.6594±0.0068 | 0.6380±0.0140 | 0.2336±0.0032 | 0.6091 | 0.6609 | 0.7057 | +0.0677 |

Paired fold-wise deltas (V4 - V3):

| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |
|---|---|---|---|---|---|
| 1 | -0.0016 | +0.0034 | -0.0007 | +0.0106 | +0.0099 |
| 2 | -0.0002 | -0.0007 | +0.0000 | +0.0053 | -0.0007 |
| 3 | -0.0006 | +0.0008 | -0.0003 | +0.0015 | -0.0027 |
| 4 | -0.0013 | +0.0012 | -0.0006 | +0.0045 | +0.0026 |
| **mean** | **-0.0009** | **+0.0012** | **-0.0004** | **+0.0055** | **+0.0023** |

Log loss improved in **4/4** folds; ROC-AUC improved in **3/4**; Brier improved in **3/4**.

**Pooled TRAIN-only out-of-fold result** (all 4 folds' validation predictions concatenated):

- n = 5296 (V3) / 5296 (V4) | Accuracy 0.6037 -> 0.6091 (+0.55 pp) | ROC-AUC 0.6366 -> 0.6381 (+0.0015) | Log Loss 0.6603 -> 0.6594 (-0.0009) | Brier 0.2340 -> 0.2336 (-0.0004) | F1 0.6590 -> 0.6611 (+0.0021)
- Correct predictions: V3 = 3197, V4 = 3226, **V4 - V3 = +29 additional correct matches**.

## Coverage-stratified diagnostic (descriptive only)

Defined ENTIRELY from a pre-match feature (`roster_form_players_min`), never the target. Coverage-rich = both inferred rosters have >=5 players with usable prior history. Never used to tune, select, or modify V4.

### RF

**Coverage-rich (roster_form_players_min >= 5)**

- Fold-mean (n≈880/fold): Δ log loss -0.0005, Δ ROC-AUC +0.0011, Δ Brier -0.0002, Δ accuracy +0.0010, Δ F1 -0.0021
- Pooled OOF (n=3519/3519): Accuracy 0.6016 -> 0.6024 (+0.09 pp), ROC-AUC 0.6295 -> 0.6305 (+0.0010), Log Loss 0.6622 -> 0.6617 (-0.0005), correct: 2117 -> 2120 (+3)

**Cold-start (roster_form_players_min == 0)**

- Fold-mean (n≈346/fold): Δ log loss -0.0030, Δ ROC-AUC +0.0061, Δ Brier -0.0013, Δ accuracy +0.0064, Δ F1 +0.0072
- Pooled OOF (n=1386/1386): Accuracy 0.6198 -> 0.6263 (+0.65 pp), ROC-AUC 0.6555 -> 0.6608 (+0.0053), Log Loss 0.6512 -> 0.6484 (-0.0028), correct: 859 -> 868 (+9)

### XGB

**Coverage-rich (roster_form_players_min >= 5)**

- Fold-mean (n≈880/fold): Δ log loss -0.0011, Δ ROC-AUC +0.0015, Δ Brier -0.0005, Δ accuracy +0.0057, Δ F1 +0.0007
- Pooled OOF (n=3519/3519): Accuracy 0.5953 -> 0.6013 (+0.60 pp), ROC-AUC 0.6260 -> 0.6282 (+0.0022), Log Loss 0.6645 -> 0.6634 (-0.0012), correct: 2095 -> 2116 (+21)

**Cold-start (roster_form_players_min == 0)**

- Fold-mean (n≈346/fold): Δ log loss -0.0008, Δ ROC-AUC -0.0004, Δ Brier -0.0004, Δ accuracy +0.0053, Δ F1 +0.0046
- Pooled OOF (n=1386/1386): Accuracy 0.6169 -> 0.6219 (+0.51 pp), ROC-AUC 0.6553 -> 0.6546 (-0.0007), Log Loss 0.6527 -> 0.6522 (-0.0006), correct: 855 -> 862 (+7)

## V4-only feature importance (descriptive, no feature selection performed)

**Correlated-feature caveat.** Several new player features are highly correlated (Phase 5C's own quality report found r=0.965 between core-5 concentration and continuity). Permutation importance may be *shared* across correlated features - a near-zero individual score does not prove a correlated feature or family carries no signal.

### RF: top 10 by impurity importance

| rank | feature | family | impurity importance | fold-val permutation (mean±std) |
|---|---|---|---|---|
| 1 | elo_diff | A_original_v1 | 0.0795 | 0.0033±0.0023 |
| 2 | avg_opponent_elo_last_10_diff | E_opponent_strength | 0.0473 | 0.0037±0.0010 |
| 3 | map_pool_best_elo_diff | B_pool_depth | 0.0461 | 0.0008±0.0007 |
| 4 | map_pool_total_matches_diff | B_pool_depth | 0.0411 | 0.0039±0.0047 |
| 5 | performance_residual_all_diff | E_opponent_strength | 0.0356 | 0.0009±0.0027 |
| 6 | time_weighted_series_margin_diff | F_time_decayed | 0.0322 | 0.0014±0.0012 |
| 7 | map_pool_second_best_elo_diff | B_pool_depth | 0.0290 | -0.0002±0.0007 |
| 8 | total_matches_before_diff | A_original_v1 | 0.0289 | 0.0022±0.0040 |
| 9 | roster_mean_player_history_mass_diff | J_confidence_evidence | 0.0280 | -0.0001±0.0029 |
| 10 | map_matchup_mean_elo_advantage | C_same_map_matchup | 0.0250 | -0.0004±0.0012 |

### XGB: top 10 by gain

| rank | feature | family | gain | weight | fold-val permutation (mean±std) |
|---|---|---|---|---|---|
| 1 | elo_diff | A_original_v1 | 67.1943 | 104 | 0.0106±0.0037 |
| 2 | map_pool_total_matches_diff | B_pool_depth | 34.6292 | 41 | 0.0065±0.0049 |
| 3 | map_pool_best_elo_diff | B_pool_depth | 27.4619 | 56 | 0.0016±0.0010 |
| 4 | performance_residual_all_diff | E_opponent_strength | 22.0462 | 22 | 0.0025±0.0057 |
| 5 | map_pool_size_diff | B_pool_depth | 21.3229 | 25 | 0.0005±0.0011 |
| 6 | avg_opponent_elo_last_10_diff | E_opponent_strength | 20.4033 | 85 | 0.0034±0.0002 |
| 7 | map_pool_experienced_maps_diff | B_pool_depth | 19.4155 | 32 | 0.0017±0.0032 |
| 8 | roster_form_players_min | J_confidence_evidence | 18.5801 | 1 | -0.0000±0.0001 |
| 9 | total_matches_before_diff | A_original_v1 | 18.1744 | 27 | 0.0022±0.0014 |
| 10 | recent_unique_players_10_maps_diff | I_roster_stability | 18.0081 | 14 | 0.0004±0.0007 |

## Family-level GROUPED permutation importance: full-data vs. coverage-rich

H = player performance (ADR/KAST/KD-balance/assists-per-round), I = roster stability, J = confidence/evidence. Each family's columns are permuted JOINTLY per repeat (10 repeats, `random_state=42`); reported is the ROC-AUC decrease averaged across folds. Coverage-rich is computed on the same fitted fold model, scored on that fold's coverage-rich validation slice only (still descriptive, no retraining).

**RF - H/I/J:**

| family | scope | mean ROC-AUC decrease | std across folds |
|---|---|---|---|
| H_player_performance | full_data | 0.0077 | 0.0036 |
| H_player_performance | coverage_rich | 0.0110 | 0.0075 |
| I_roster_stability | full_data | 0.0017 | 0.0011 |
| I_roster_stability | coverage_rich | 0.0001 | 0.0004 |
| J_confidence_evidence | full_data | -0.0007 | 0.0040 |
| J_confidence_evidence | coverage_rich | -0.0010 | 0.0038 |

**RF - H1-H4 subfamilies:**

| subfamily | scope | mean ROC-AUC decrease | std across folds |
|---|---|---|---|
| H1_adr | full_data | 0.0004 | 0.0003 |
| H1_adr | coverage_rich | 0.0004 | 0.0008 |
| H2_kast | full_data | 0.0044 | 0.0037 |
| H2_kast | coverage_rich | 0.0047 | 0.0053 |
| H3_kd_balance | full_data | 0.0004 | 0.0013 |
| H3_kd_balance | coverage_rich | 0.0001 | 0.0020 |
| H4_assists_per_round | full_data | 0.0001 | 0.0003 |
| H4_assists_per_round | coverage_rich | -0.0000 | 0.0004 |

**XGB - H/I/J:**

| family | scope | mean ROC-AUC decrease | std across folds |
|---|---|---|---|
| H_player_performance | full_data | 0.0095 | 0.0050 |
| H_player_performance | coverage_rich | 0.0140 | 0.0083 |
| I_roster_stability | full_data | 0.0008 | 0.0010 |
| I_roster_stability | coverage_rich | -0.0002 | 0.0004 |
| J_confidence_evidence | full_data | -0.0007 | 0.0027 |
| J_confidence_evidence | coverage_rich | -0.0018 | 0.0035 |

**XGB - H1-H4 subfamilies:**

| subfamily | scope | mean ROC-AUC decrease | std across folds |
|---|---|---|---|
| H1_adr | full_data | -0.0000 | 0.0011 |
| H1_adr | coverage_rich | 0.0000 | 0.0011 |
| H2_kast | full_data | 0.0073 | 0.0054 |
| H2_kast | coverage_rich | 0.0085 | 0.0063 |
| H3_kd_balance | full_data | 0.0003 | 0.0013 |
| H3_kd_balance | coverage_rich | 0.0006 | 0.0031 |
| H4_assists_per_round | full_data | -0.0003 | 0.0005 |
| H4_assists_per_round | coverage_rich | -0.0001 | 0.0002 |

Not used for feature selection, model changes, or tuning - descriptive only.

## Verdict

RF full-data: **HELP**. XGB full-data: **HELP**.

**PLAYER FEATURES HELP CLEARLY**

Both models improved on all three primary metrics under the frozen, previously-selected configurations, across the full TRAIN-only development set.

- **MAIN VALIDATION = NOT USED**
- **TEST = SEALED**
- **COLOGNE = UNTOUCHED**
