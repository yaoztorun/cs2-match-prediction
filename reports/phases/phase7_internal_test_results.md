# Phase 7 - Sealed Internal TEST Results

**This is the first and only unbiased internal evaluation of the fully frozen known-map system.** The model, features, preprocessing, threshold (0.5) and selection procedure were all frozen before this partition was opened (Phase 6D). No fitting, tuning, calibration, symmetrization, or ensemble construction occurs in this phase or after it based on these results.

## 1. Final model TEST performance

n=1427, Accuracy=0.6132, Precision=0.6458, Recall=0.7109, F1=0.6768, ROC-AUC=0.6489, Log Loss=0.6521, Brier=0.2301

Series-macro: Log Loss=0.6424, Brier=0.2254, Accuracy=0.6291

Side-symmetry (diagnostic only, never corrected): mean=0.0080, median=0.0067, p95=0.0198, max=0.0445

## 2. Baselines on the same TEST rows

| model | n | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|
| baseline_constant_05 | 1427 | 0.5697 | 0.5000 | 0.6931 | 0.2500 |
| baseline_overall_elo | 1427 | 0.6132 | 0.6477 | 0.6549 | 0.2314 |
| baseline_map_elo | 1427 | 0.5613 | 0.5863 | 0.6786 | 0.2428 |
| final_xgb_v3 | 1427 | 0.6132 | 0.6489 | 0.6521 | 0.2301 |

Final XGB vs overall-ELO: Δlog loss -0.0028, ΔROC-AUC +0.0012. Final XGB vs map-ELO: Δlog loss -0.0265, ΔROC-AUC +0.0626. (Point estimates only - see `reports/phase7_internal_test_uncertainty.md` for cluster-bootstrap intervals on these deltas.)

## 3. Series-macro (all models)

| model | n series | series-macro log loss | series-macro Brier | series-macro accuracy |
|---|---|---|---|---|
| final_xgb_v3 | 688 | 0.6424 | 0.2254 | 0.6291 |
| baseline_constant_05 | 688 | 0.6931 | 0.2500 | 0.5871 |
| baseline_overall_elo | 688 | 0.6385 | 0.2242 | 0.6307 |
| baseline_map_elo | 688 | 0.6745 | 0.2410 | 0.5670 |

## 4. Performance by CS2 Map

Sorted alphabetically (never by performance) to avoid cherry-picking. `n < 30` is flagged SMALL SAMPLE - INTERPRET CAUTIOUSLY.

| map | n | accuracy | ROC-AUC | log loss | Brier | team1 rate | mean p |
|---|---|---|---|---|---|---|---|
| Ancient | 221 | 0.6471 | 0.6582 | 0.6464 | 0.2273 | 0.579 | 0.538 |
| Anubis | 97 | 0.6392 | 0.6917 | 0.6442 | 0.2259 | 0.546 | 0.534 |
| Dust2 | 315 | 0.6476 | 0.6996 | 0.6323 | 0.2206 | 0.562 | 0.538 |
| Inferno | 154 | 0.5455 | 0.5791 | 0.6809 | 0.2439 | 0.597 | 0.539 |
| Mirage | 286 | 0.5734 | 0.6025 | 0.6709 | 0.2392 | 0.566 | 0.540 |
| Nuke | 204 | 0.6078 | 0.6429 | 0.6559 | 0.2319 | 0.569 | 0.537 |
| Overpass | 149 | 0.6242 | 0.6794 | 0.6365 | 0.2229 | 0.570 | 0.537 |
| Train **[SMALL SAMPLE]** | 1 | 1.0000 | n/a (single class) | 0.6393 | 0.2231 | 0.000 | 0.472 |

Descriptively (not a ranking claim): `Dust2` has the lowest observed TEST log loss (0.6323, n=315); `Inferno` has the highest (0.6809, n=154). Every per-map estimate here should be read together with its own `n` - a striking result on a small map subsample is far less certain than a similar result on a well-populated one (see the map-level figures and `reports/phase7_internal_test_uncertainty.md`). No map is removed and no map-specific model is created based on this table.

## 5. Performance by BO1/BO3/BO5

| bestOf | n | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|
| BO1 | 54 | 0.7037 | 0.7579 | 0.6028 | 0.2059 |
| BO3 | 1326 | 0.6086 | 0.6392 | 0.6553 | 0.2317 |
| BO5 | 47 | 0.6383 | 0.7529 | 0.6175 | 0.2135 |

## 6. Performance by tier

| tier | n | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|
| tier1 | 897 | 0.6243 | 0.6552 | 0.6502 | 0.2292 |
| tier2 | 416 | 0.6010 | 0.6430 | 0.6574 | 0.2325 |
| tier3 | 114 | 0.5702 | 0.6257 | 0.6481 | 0.2283 |

## 7. Coverage / evidence diagnostics (descriptive only)

| subgroup | n | % of TEST | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|---|
| A_recent_map_evidence | 1223 | 85.7% | 0.6051 | 0.6445 | 0.6542 | 0.2311 |
| B_trusted_adjusted_evidence | 1223 | 85.7% | 0.6051 | 0.6445 | 0.6542 | 0.2311 |
| C_roster_map_evidence_ge3 | 1192 | 83.5% | 0.5948 | 0.6359 | 0.6608 | 0.2343 |
| D_strong_roster_map_evidence_ge5 | 1084 | 76.0% | 0.5969 | 0.6360 | 0.6602 | 0.2340 |
| E_roster_map_cold_start | 200 | 14.0% | 0.7050 | 0.6872 | 0.6093 | 0.2095 |
| F_team_map_cold_start | 204 | 14.3% | 0.6618 | 0.6746 | 0.6396 | 0.2239 |
| G_current_core_evidence | 1141 | 80.0% | 0.5960 | 0.6396 | 0.6584 | 0.2331 |
| H_high_evidence | 1141 | 80.0% | 0.5968 | 0.6405 | 0.6584 | 0.2332 |

No subgroup-specific model is trained; these are descriptive slices of the single frozen model's predictions.

## 8. Calibration (fixed 10-bin reliability, raw probabilities, no calibration fitted)

| bin | n | mean predicted | empirical Team1 win rate |
|---|---|---|---|
| [0.2,0.3) | 23 | 0.279 | 0.348 |
| [0.3,0.4) | 153 | 0.355 | 0.346 |
| [0.4,0.5) | 356 | 0.453 | 0.489 |
| [0.5,0.6) | 433 | 0.549 | 0.575 |
| [0.6,0.7) | 355 | 0.649 | 0.693 |
| [0.7,0.8) | 107 | 0.724 | 0.776 |

## 9. Development vs TEST (context only)

| source | model identity | n | accuracy | ROC-AUC | log loss | Brier |
|---|---|---|---|---|---|---|
| Phase 6D TRAIN-only selected-model OOF (development) | map_xgboost_v3_final (the SAME frozen model evaluated on TEST here) | 6541 | 0.5901 | 0.6200 | 0.6682 | 0.2379 |
| Phase 6B consumed main VALIDATION (EARLIER model/version, V2 features, random_013 @ 124 trees - NOT the Phase 6D final model) | map_xgboost_v2_random_013 (an earlier, different model) | 1129 | 0.5961 | 0.6203 | 0.6649 | 0.2363 |
| Phase 7 sealed internal TEST (this evaluation) | map_xgboost_v3_final (the frozen model) | 1427 | 0.6132 | 0.6489 | 0.6521 | 0.2301 |

**Direct generalization comparison** (same model, same features): Phase 6D TRAIN-only OOF development -> Phase 7 sealed TEST: Δaccuracy +0.0231, ΔROC-AUC +0.0289, Δlog loss -0.0161, ΔBrier -0.0078. The Phase 6B validation row above used an EARLIER model/version (V2 features, a different XGB configuration) and is shown only as broader historical context, never as an evaluation of the exact Phase 6D final model.

## Interpretation

1. **Generalization**: see the direct comparison above and the cluster-bootstrap intervals in `reports/phase7_internal_test_uncertainty.md` for whether the TEST result is inside, above, or below the development-era range once sampling uncertainty is accounted for.

2. **Vs. baselines**: final XGB log loss 0.6521 against constant-0.5 0.6931, overall-ELO 0.6549, map-ELO 0.6786; ROC-AUC 0.6489 against 0.6477 / 0.5863.

3. **Uncertainty**: see `reports/phase7_internal_test_uncertainty.md` for 2,000-replicate series-cluster bootstrap intervals.

4. **Probability quality**: Brier 0.2301 vs constant-0.5's 0.2500; see the calibration table above and `reports/figures/map_xgb_v3_test_calibration.png`.

5. **Consistency across maps/formats/coverage**: see sections 4-7 above and the map-level figures - read every subgroup number together with its own `n`.

6. **Cold-start behavior**: compare coverage subgroups E/F (cold start) against A/B/C/D/H (evidenced) in section 7.

7. **Side-orientation stability**: see the side-symmetry statistics in section 1 - diagnostic only, no prediction is symmetrized.


No model change is proposed in this section. Any improvement discussion belongs in limitations/future-work only, and does not alter the frozen system evaluated here.

## Status

- **FINAL MODEL = FROZEN BEFORE TEST**
- **TEST = OPENED FOR FINAL INTERNAL EVALUATION**
- **TEST = NOT USED FOR MODEL DEVELOPMENT**
- **NO POST-TEST RETUNING**
- **THRESHOLD = 0.5**
- **NO CALIBRATION**
- **NO NEW ENSEMBLE**
- **COLOGNE = UNTOUCHED**
- **SRC = UNCHANGED**
