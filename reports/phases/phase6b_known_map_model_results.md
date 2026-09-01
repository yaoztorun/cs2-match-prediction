# Phase 6B - Known-Map Model Results

## What this model predicts

Given Team A, Team B, the series format (BO1/BO3/BO5) and a **user-selected map**, estimate P(Team A wins that map), using only information available before the series starts. The selected map is legitimate input here because the user supplies it.

**This is a different prediction task from the pre-veto series model.** The earlier ~61% figures were SERIES-winner accuracies; everything below is MAP-outcome accuracy. A map accuracy and a series accuracy cannot be differenced or described as one being N percentage points better than the other - they are different targets. Converting map probabilities into BO3/BO5 series probabilities is an application-level simulation step that is not part of this phase.

## Data

- TRAIN: **7,762** unique historical maps -> 15,524 augmented training observations after side mirroring (mirrored rows are re-labellings of already-counted maps, never additional maps)
- VALIDATION: **1,129** maps - opened **exactly once**, in this script, after every configuration was frozen
- TEST: **1,427** maps - **SEALED**, never loaded
- Maps of one series never cross a partition (enforced by `data/modeling/map_split_v1.csv`)

## Frozen configurations

**Random Forest** `random_007`: `max_depth=7`, `min_samples_leaf=5`, `min_samples_split=10`, `max_features=sqrt`, `criterion=log_loss`, `n_estimators=400`, `bootstrap=True`

**XGBoost** `random_013`: `learning_rate=0.03`, `max_depth=2`, `min_child_weight=10`, `subsample=0.75`, `colsample_bytree=0.85`, `gamma=5.0`, `reg_alpha=0.01`, `reg_lambda=10.0`, `n_estimators=124` (= round(median(best_iteration + 1 across the 4 outer folds)), from fold best_iterations [68, 361, 74, 172]), fitted with **no eval_set and no early stopping**

**Ensemble** `p = 0.6 * p_rf + 0.4 * p_xgb`, weight selected from TRAIN-only OOF via secondary (log-loss tie within epsilon, resolved by highest ROC-AUC)

## Main map validation - map-level metrics (opened once)

| model | n | accuracy | precision | recall | F1 | ROC-AUC | log loss | Brier | confusion [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 1129 | 0.5881 | 0.6016 | 0.7145 | 0.6532 | 0.6141 | 0.6667 | 0.2372 | [[226, 290], [175, 438]] |
| XGBoost | 1129 | 0.5961 | 0.6065 | 0.7292 | 0.6622 | 0.6203 | 0.6649 | 0.2363 | [[226, 290], [166, 447]] |
| RF/XGB ensemble | 1129 | 0.5890 | 0.6011 | 0.7227 | 0.6563 | 0.6171 | 0.6656 | 0.2367 | [[222, 294], [170, 443]] |
| baseline: half | 1129 | 0.5430 | 0.5430 | 1.0000 | 0.7038 | 0.5000 | 0.6931 | 0.2500 | - |
| baseline: overall_elo | 1129 | 0.5899 | 0.6027 | 0.7178 | 0.6552 | 0.6155 | 0.6801 | 0.2422 | - |
| baseline: map_elo | 1129 | 0.5492 | 0.5795 | 0.6183 | 0.5983 | 0.5892 | 0.6783 | 0.2429 | - |

Train-validation gaps (overfitting control):

| model | train accuracy | validation accuracy | train ROC-AUC | validation ROC-AUC | AUC gap |
|---|---|---|---|---|---|
| Random Forest | 0.6475 | 0.5881 | 0.7096 | 0.6141 | +0.0955 |
| XGBoost | 0.5917 | 0.5961 | 0.6323 | 0.6203 | +0.0121 |
| RF/XGB ensemble | 0.6221 | 0.5890 | 0.6793 | 0.6171 | +0.0622 |

## TRAIN-only out-of-fold vs. main validation

| model | OOF log loss | validation log loss | OOF ROC-AUC | validation ROC-AUC | OOF accuracy | validation accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 0.6682 | 0.6667 | 0.6179 | 0.6141 | 0.5834 | 0.5881 |
| XGBoost | 0.6695 | 0.6649 | 0.6164 | 0.6203 | 0.5843 | 0.5961 |
| RF/XGB ensemble | 0.6683 | 0.6656 | 0.6184 | 0.6171 | 0.5846 | 0.5890 |

The OOF column is development evidence over the four TRAIN folds; the validation column is a single held-out period. They measure different things and are shown side by side only to expose how much the picture moved.

## Series-macro diagnostics

Multiple maps of one series are dependent observations. Map-level metrics above remain PRIMARY (the task is map prediction); these average each series' own per-map mean and then average those equally across match_ids, so BO3/BO5 series do not receive disproportionate weight. No per-series ROC-AUC is computed - most series have too few maps, often of a single class, for it to be meaningful.

| model | population | n series | series-macro log loss | series-macro Brier | series-macro accuracy |
|---|---|---|---|---|---|
| Random Forest | pooled_train_oof | 3115 | 0.6632 | 0.2356 | 0.5921 |
| Random Forest | validation | 522 | 0.6543 | 0.2314 | 0.6101 |
| XGBoost | pooled_train_oof | 3115 | 0.6645 | 0.2360 | 0.5968 |
| XGBoost | validation | 522 | 0.6535 | 0.2308 | 0.6133 |
| RF/XGB ensemble | pooled_train_oof | 3115 | 0.6633 | 0.2355 | 0.5962 |
| RF/XGB ensemble | validation | 522 | 0.6536 | 0.2310 | 0.6080 |
| baseline: half | validation | 522 | 0.6931 | 0.2500 | 0.5438 |
| baseline: overall_elo | validation | 522 | 0.6623 | 0.2343 | 0.6096 |
| baseline: map_elo | validation | 522 | 0.6725 | 0.2401 | 0.5617 |

## Per-map validation diagnostics

Predefined before validation was opened. Small samples are marked; ROC-AUC is reported only where both target classes are present. Low-history maps such as Train and Overpass are shown as-is and are **not** removed, and nothing here is used to retune anything.

**Random Forest**

| map | n | small sample | accuracy | log loss | Brier | ROC-AUC |
|---|---|---|---|---|---|---|
| Mirage | 197 | no | 0.5685 | 0.6813 | 0.2441 | 0.5799 |
| Dust2 | 188 | no | 0.5851 | 0.6542 | 0.2319 | 0.6352 |
| Inferno | 180 | no | 0.5833 | 0.6500 | 0.2299 | 0.6547 |
| Nuke | 172 | no | 0.6337 | 0.6802 | 0.2420 | 0.6289 |
| Ancient | 168 | no | 0.5238 | 0.6843 | 0.2460 | 0.5632 |
| Train | 109 | no | 0.5872 | 0.6912 | 0.2481 | 0.5735 |
| Overpass | 100 | no | 0.6800 | 0.6239 | 0.2176 | 0.6209 |
| Anubis | 15 | yes | 0.5333 | 0.5891 | 0.2015 | 0.7222 |

**XGBoost**

| map | n | small sample | accuracy | log loss | Brier | ROC-AUC |
|---|---|---|---|---|---|---|
| Mirage | 197 | no | 0.5736 | 0.6784 | 0.2428 | 0.5835 |
| Dust2 | 188 | no | 0.6117 | 0.6591 | 0.2337 | 0.6329 |
| Inferno | 180 | no | 0.5778 | 0.6534 | 0.2311 | 0.6548 |
| Nuke | 172 | no | 0.6105 | 0.6602 | 0.2335 | 0.6626 |
| Ancient | 168 | no | 0.5595 | 0.6844 | 0.2457 | 0.5722 |
| Train | 109 | no | 0.5872 | 0.6862 | 0.2461 | 0.5783 |
| Overpass | 100 | no | 0.6800 | 0.6282 | 0.2186 | 0.6222 |
| Anubis | 15 | yes | 0.6667 | 0.6250 | 0.2180 | 0.7222 |

**RF/XGB ensemble**

| map | n | small sample | accuracy | log loss | Brier | ROC-AUC |
|---|---|---|---|---|---|---|
| Mirage | 197 | no | 0.5787 | 0.6798 | 0.2434 | 0.5811 |
| Dust2 | 188 | no | 0.5904 | 0.6559 | 0.2325 | 0.6345 |
| Inferno | 180 | no | 0.5833 | 0.6510 | 0.2301 | 0.6560 |
| Nuke | 172 | no | 0.6279 | 0.6714 | 0.2384 | 0.6432 |
| Ancient | 168 | no | 0.5238 | 0.6840 | 0.2457 | 0.5674 |
| Train | 109 | no | 0.5872 | 0.6888 | 0.2472 | 0.5722 |
| Overpass | 100 | no | 0.6700 | 0.6251 | 0.2177 | 0.6186 |
| Anubis | 15 | yes | 0.5333 | 0.6028 | 0.2077 | 0.7222 |

## Coverage diagnostics (descriptive only)

Subgroups predefined before validation was opened. No subgroup-specific model is built in Phase 6B.

| subgroup | model | n | % of validation | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|---|---|
| A_both_teams_have_map_history | Random Forest | 1023 | 90.6% | 0.5934 | 0.6178 | 0.6635 | 0.2358 |
| A_both_teams_have_map_history | XGBoost | 1023 | 90.6% | 0.5982 | 0.6250 | 0.6615 | 0.2347 |
| A_both_teams_have_map_history | RF/XGB ensemble | 1023 | 90.6% | 0.5953 | 0.6212 | 0.6623 | 0.2351 |
| B_both_teams_have_5_map_matches | Random Forest | 815 | 72.2% | 0.5939 | 0.6244 | 0.6601 | 0.2343 |
| B_both_teams_have_5_map_matches | XGBoost | 815 | 72.2% | 0.5951 | 0.6296 | 0.6598 | 0.2339 |
| B_both_teams_have_5_map_matches | RF/XGB ensemble | 815 | 72.2% | 0.5926 | 0.6270 | 0.6595 | 0.2339 |
| C_roster_form_players_min_ge_5 | Random Forest | 1016 | 90.0% | 0.5837 | 0.6131 | 0.6672 | 0.2375 |
| C_roster_form_players_min_ge_5 | XGBoost | 1016 | 90.0% | 0.5935 | 0.6212 | 0.6646 | 0.2361 |
| C_roster_form_players_min_ge_5 | RF/XGB ensemble | 1016 | 90.0% | 0.5846 | 0.6168 | 0.6657 | 0.2367 |
| D_map_cold_start_at_least_one_side | Random Forest | 106 | 9.4% | 0.5377 | 0.5829 | 0.6978 | 0.2514 |
| D_map_cold_start_at_least_one_side | XGBoost | 106 | 9.4% | 0.5755 | 0.5829 | 0.6975 | 0.2517 |
| D_map_cold_start_at_least_one_side | RF/XGB ensemble | 106 | 9.4% | 0.5283 | 0.5850 | 0.6972 | 0.2513 |

## Calibration (diagnostic only - no calibration is fitted in Phase 6B)

Figures: `reports/figures/map_{rf,xgb,ensemble}_v1_calibration.png`, plus ROC curves and probability-distribution histograms under the same stems. No isotonic or Platt calibration is applied.

**Random Forest** reliability (10 bins):

| bin | n | mean predicted | empirical map win rate |
|---|---|---|---|
| [0.1,0.2) | 2 | 0.187 | 0.000 |
| [0.2,0.3) | 28 | 0.258 | 0.214 |
| [0.3,0.4) | 85 | 0.360 | 0.376 |
| [0.4,0.5) | 286 | 0.454 | 0.479 |
| [0.5,0.6) | 380 | 0.552 | 0.571 |
| [0.6,0.7) | 239 | 0.639 | 0.598 |
| [0.7,0.8) | 96 | 0.743 | 0.688 |
| [0.8,0.9) | 13 | 0.838 | 0.923 |

**XGBoost** reliability (10 bins):

| bin | n | mean predicted | empirical map win rate |
|---|---|---|---|
| [0.2,0.3) | 22 | 0.281 | 0.182 |
| [0.3,0.4) | 103 | 0.359 | 0.340 |
| [0.4,0.5) | 267 | 0.456 | 0.476 |
| [0.5,0.6) | 386 | 0.554 | 0.575 |
| [0.6,0.7) | 257 | 0.640 | 0.619 |
| [0.7,0.8) | 94 | 0.723 | 0.702 |

**RF/XGB ensemble** reliability (10 bins):

| bin | n | mean predicted | empirical map win rate |
|---|---|---|---|
| [0.2,0.3) | 27 | 0.265 | 0.185 |
| [0.3,0.4) | 89 | 0.359 | 0.371 |
| [0.4,0.5) | 276 | 0.453 | 0.478 |
| [0.5,0.6) | 388 | 0.552 | 0.575 |
| [0.6,0.7) | 247 | 0.639 | 0.587 |
| [0.7,0.8) | 96 | 0.737 | 0.729 |
| [0.8,0.9) | 6 | 0.821 | 0.833 |

Predicted-probability spread on validation:

| model | min | p5 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.177 | 0.347 | 0.460 | 0.545 | 0.618 | 0.744 | 0.885 |
| XGBoost | 0.244 | 0.346 | 0.469 | 0.552 | 0.618 | 0.717 | 0.762 |
| RF/XGB ensemble | 0.211 | 0.347 | 0.463 | 0.550 | 0.616 | 0.733 | 0.836 |

## Side-symmetry diagnostic

Each validation matchup is scored as A vs B on map X and again as B vs A on the same map X; the error is `|P(A wins) - (1 - P(B wins))|`. **Measured, not corrected** - Phase 6B deliberately records raw behaviour rather than symmetrizing predictions.

| model | mean | median | p95 | max |
|---|---|---|---|---|
| Random Forest | 0.0091 | 0.0078 | 0.0229 | 0.0411 |
| XGBoost | 0.0070 | 0.0058 | 0.0174 | 0.0415 |
| RF/XGB ensemble | 0.0060 | 0.0049 | 0.0153 | 0.0298 |

## Verdict

On this single held-out validation period, **XGBoost** has the lowest map-level log loss (0.6649 vs 0.6783 for the map-ELO baseline and 0.6801 for the overall-ELO baseline), with ROC-AUC 0.6203 against 0.5892 / 0.6155 respectively.

No final project model is declared. The internal TEST partition is the final unbiased internal evaluation and remains sealed; the external Cologne protocol follows after that. No hyperparameter, feature, threshold, ensemble weight, preprocessing rule, map category or calibration was changed after these numbers were seen.

## Status

- **MAIN MAP VALIDATION = USED ONCE AFTER FREEZE**
- **TEST = SEALED**
- **COLOGNE = UNTOUCHED**
- **NO POST-VALIDATION RETUNING**
- **SRC = UNCHANGED**
