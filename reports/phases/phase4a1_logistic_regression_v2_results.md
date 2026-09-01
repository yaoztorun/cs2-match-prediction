# Phase 4A.1 - Logistic Regression V2 Results (Chronologically Tuned)

**LR V1 = UNTUNED · LR V2 = TUNED · RF V1 = UNTUNED · RF V2 = TUNED · XGB V1 = UNTUNED · XGB V2 = TUNED**

## 1. Why LR tuning was performed

RF and XGBoost had both received chronological tuning while Logistic Regression had not, so no fair algorithm comparison existed. LR V1's cost curve was also still descending at its fixed 10,000 iterations, so it was both untuned *and* under-converged.

## 2. Why lambda is the predictive/model hyperparameter

L2 strength changes the hypothesis the model settles on - it trades training fit against coefficient magnitude and therefore changes predictions. It is the only thing searched here.

## 3. Why alpha and iterations are optimization controls

`alpha=0.01` and `max_iterations=20000` were fixed for every candidate and were never selected by comparing validation predictions - they exist purely so gradient descent converges reliably. Because all lambda candidates share them, the lambda comparison is clean.

**Disclosure of a real confound**: LR V1 used `alpha=0.001` with a fixed 10,000 iterations and was still descending; LR V2 uses `alpha=0.01` with a convergence criterion. So the LR V1 -> LR V2 comparison mixes *regularization* with *better convergence*. The `lambda=0` row of the tuning table is the clean reference for isolating the pure regularization effect.

## 4. Temporal CV methodology

The **same** four expanding-window folds used to tune RF V2 and XGB V2 (`data/modeling/random_forest_cv_folds_v2.csv`, byte-identical), all inside the global TRAIN partition, chronology re-verified at runtime. The main validation partition was never loaded by the tuning script.

## 5. Mirroring

Per fold: mirror fold-train only; fold-validation never mirrored; fold TRAIN metrics computed on the original unmirrored fold-train rows. For the final refit: 6,619 unique historical training matches -> **13,238 augmented training observations** (never described as independent matches), target mean exactly 0.5.

## 6. Preprocessing

Identical LR V1 semantics (standardization + train-median imputation + deterministic bestOf/tier dummies, 19 transformed features), refit independently inside every fold and again on the augmented full TRAIN for the final model. No validation statistics anywhere.

## 7. Convergence methodology

TRAINING-OBJECTIVE ONLY. min_iterations=1000, checked every 100 iterations. Converged if EITHER (A) relative training-cost improvement over the preceding 100-iteration window is < 1e-07 on 3 consecutive checks, OR (B) the regularized gradient norm sqrt(||dj_dw||^2 + dj_db^2) < 1e-05. Validation data is never consulted.

Final refit: **8200 iterations**, converged=**True** via `relative_cost_plateau`, J 0.692701 -> 0.665374, final gradient norm 2.362e-04. See `reports/figures/logistic_regression_cost_v2.png`.

**Objective comparability caveat**: with lambda=50.0 the plotted objective *includes* the L2 penalty `(lambda/(2m))*sum(w^2)`, so its numeric value is **not** directly comparable to LR V1's unregularized objective. Only the unregularized log loss reported in the metrics tables is comparable across the two.

## 8. Lambda candidate performance

Full table in `reports/phase4a1_logistic_regression_tuning.md`. Two honest caveats carried forward from that report:

- **Every** lambda candidate fell inside the predefined 0.002 log-loss equivalence band, so L2 strength is close to *irrelevant* on this problem rather than lambda=50 being meaningfully superior.
- The convergence gate was degenerate (only one candidate converged in all four folds), but a sensitivity check re-applying the ladder **without** the gate selects the same lambda, so the outcome is not an artifact of the gate.

## 9. Selected lambda

**lambda = 50.0**, via primary (lowest mean CV log loss, unique). CV log loss 0.66993 ± 0.01099, CV ROC-AUC 0.61967 ± 0.01596.

## 10. Train metrics (original unmirrored)

- n = 6,619
- Accuracy: 0.5936
- Precision: 0.6175
- Recall: 0.6728
- F1: 0.6440
- ROC-AUC: 0.6234
- Log loss: 0.6650
- Brier: 0.2364
- Confusion matrix [[TN,FP],[FN,TP]]: [[1496, 1507], [1183, 2433]]

## 11. Main validation metrics (one-shot)

- n = 1,419
- Accuracy: 0.6131
- Precision: 0.6297
- Recall: 0.7299
- F1: 0.6761
- ROC-AUC: 0.6412
- Log loss: 0.6581
- Brier: 0.2329
- Confusion matrix [[TN,FP],[FN,TP]]: [[297, 337], [212, 573]]

## 12. LR V1 vs LR V2

| metric | LR V1 (untuned) | LR V2 (tuned) | delta |
|---|---|---|---|
| Accuracy | 0.6110 | 0.6131 | +0.0021 |
| ROC-AUC | 0.6431 | 0.6412 | -0.0019 |
| F1 | 0.6738 | 0.6761 | +0.0023 |
| Log loss | 0.6564 | 0.6581 | +0.0017 |
| Brier | 0.2322 | 0.2329 | +0.0007 |

Train->validation ROC-AUC gap: LR V1 -0.0220 -> LR V2 -0.0177. Remember this comparison confounds regularization with the improved convergence settings (Section 3).

## 13. LR V2 vs RF V2

| metric | LR V2 | RF V2 | delta (LR V2 - RF V2) |
|---|---|---|---|
| Accuracy | 0.6131 | 0.6068 | +0.0063 |
| ROC-AUC | 0.6412 | 0.6566 | -0.0154 |
| F1 | 0.6761 | 0.6690 | +0.0071 |
| Log loss | 0.6581 | 0.6514 | +0.0068 |
| Brier | 0.2329 | 0.2298 | +0.0031 |

## 14. LR V2 vs XGB V2

| metric | LR V2 | XGB V2 | delta (LR V2 - XGB V2) |
|---|---|---|---|
| Accuracy | 0.6131 | 0.6117 | +0.0014 |
| ROC-AUC | 0.6412 | 0.6504 | -0.0093 |
| F1 | 0.6761 | 0.6742 | +0.0019 |
| Log loss | 0.6581 | 0.6542 | +0.0040 |
| Brier | 0.2329 | 0.2311 | +0.0018 |

Full three-way tuned comparison in `reports/model_comparison_tuned_v1features.md`.

## 15. Calibration

`reports/figures/logistic_regression_calibration_v2.png` (validation only, no correction applied):

| bin | n | mean predicted | empirical win rate |
|---|---|---|---|
| [0.0,0.1) | 2 | 0.095 | 0.000 |
| [0.1,0.2) | 8 | 0.164 | 0.125 |
| [0.2,0.3) | 82 | 0.252 | 0.305 |
| [0.3,0.4) | 150 | 0.360 | 0.440 |
| [0.4,0.5) | 267 | 0.456 | 0.449 |
| [0.5,0.6) | 375 | 0.550 | 0.565 |
| [0.6,0.7) | 301 | 0.648 | 0.635 |
| [0.7,0.8) | 161 | 0.744 | 0.708 |
| [0.8,0.9) | 68 | 0.835 | 0.765 |
| [0.9,1.0) | 5 | 0.910 | 0.800 |

Probability spread (validation): min 0.094 | p5 0.275 | p25 0.455 | median 0.554 | p75 0.656 | p95 0.802 | max 0.927.

## 16. Coefficient shrinkage

Bias `b = 0.000000` (never regularized, matching the lab). Coefficient L2 norm `||w||_2 = 0.4465`. Top 8 by magnitude:

| feature | coefficient | abs_coefficient |
|---|---|---|
| elo_diff | 0.4033 | 0.4033 |
| total_matches_before_diff | 0.1494 | 0.1494 |
| avg_series_margin_last_10_diff | 0.0727 | 0.0727 |
| avg_series_margin_last_5_diff | 0.0696 | 0.0696 |
| win_rate_last_10_diff | -0.0439 | 0.0439 |
| format_win_rate_diff | -0.0378 | 0.0378 |
| win_rate_last_5_diff | -0.0236 | 0.0236 |
| days_since_last_match_diff | -0.0137 | 0.0137 |

V1 vs V2 comparison (`reports/tables/logistic_regression_coefficient_comparison_v1_v2.csv`), largest absolute changes:

| feature | v1_coefficient | v2_coefficient | absolute_change | relative_change |
|---|---|---|---|---|
| elo_diff | 0.2695 | 0.4033 | +0.1337 | +49.62% |
| format_win_rate_diff | 0.01294 | -0.03784 | -0.0508 | -392.38% |
| win_rate_last_10_diff | 0.006068 | -0.04386 | -0.0499 | -822.75% |
| overall_win_rate_diff | 0.05914 | 0.01235 | -0.0468 | -79.12% |
| avg_series_margin_last_5_diff | 0.03732 | 0.06959 | +0.0323 | +86.48% |
| avg_series_margin_last_10_diff | 0.04515 | 0.07266 | +0.0275 | +60.93% |
| win_rate_last_5_diff | 0.002579 | -0.02362 | -0.0262 | -1015.85% |
| matches_last_30_days_diff | 0.0215 | 0.002235 | -0.0193 | -89.61% |

`relative_change` is deliberately `n/a` where V1's coefficient was numerically ~0 (the symmetric/context features, ~1e-18 in V1) - a ratio against ~0 is meaningless, not informative. Note that V2's coefficients are generally **larger** in magnitude than V1's despite L2 shrinkage: V1 was under-converged (still descending at 10,000 iterations at alpha=0.001), so its coefficients had not yet grown to their fitted values. Shrinkage is visible *within* the tuning sweep (mean ||w||_2 falls monotonically as lambda rises), which is the clean comparison. **Coefficients are not interpreted causally.**

## 17. Symmetry

mean **4.186e-17**, median **0.000e+00**, p95 **1.110e-16**, max **2.220e-16** - measured, not assumed. Detail in `reports/logistic_regression_symmetry_v2.md`. Not corrected.

## 18. Limitations

- The selected lambda (50.0) sits at the **edge of the predefined grid**, so the optimum may lie beyond it; the grid is deliberately not extended, since re-searching after seeing results is what this protocol forbids.
- Every candidate fell within the 0.002 equivalence band - L2 strength barely matters here, so lambda=50 should not be read as meaningfully better than lambda=0.
- The convergence gate admitted only one candidate; the sensitivity check shows the outcome is unchanged without it, but the gate as configured did little useful work, and criterion (B) (gradient norm) never fired at all.
- LR V1 -> LR V2 confounds regularization with convergence quality (Section 3).
- Same 17 series-level features; no map-level or player-level information.
- Single chronological validation period of 1,419 matches; no probability calibration, no threshold tuning.
- The main validation partition has now been used once for LR V2; repeated future use would erode its independence.

## Status

- **TEST = SEALED** - not opened or scored.
- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.
- Main validation evaluated exactly once, after the configuration was frozen.
- No tuning iteration followed the main-validation result.
- No final project model declared.
