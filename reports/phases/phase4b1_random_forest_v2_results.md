# Phase 4B.1 - Random Forest V2 Results (Chronologically Tuned)

Terminology used consistently throughout: **RF V1** = untuned baseline (`max_depth=None, min_samples_leaf=1`, Phase 4B). **RF V2** = chronologically-tuned Random Forest, selected via TRAIN-only expanding-window CV (this phase). **LR V1** = untuned Logistic Regression baseline (Phase 4A, also not yet tuned).

## Selected RF V2 configuration

Candidate `random_009`, selected via: secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC) (CV mean log loss 0.6638 ± 0.0075, CV mean ROC-AUC 0.6290 ± 0.0153). Full search detail in `reports/phase4b1_random_forest_tuning.md`.

```
n_estimators=300, max_depth=8, min_samples_leaf=20, min_samples_split=10, max_features=sqrt, bootstrap=True, criterion=gini
```

## Refit on full TRAIN

Trained on the full augmented TRAIN partition: 6,619 unique historical matches -> 13,238 augmented training **observations** after mirroring (never described as 13,238 matches). Preprocessing fit on this augmented full-train only, saved to `data/modeling/random_forest_preprocessing_v2.json`. Evaluated **exactly once** on the 1,419-row main validation partition - no iteration followed.

## Train metrics (unmirrored)

- n = 6,619
- Accuracy: 0.6460
- Precision: 0.6592
- Recall: 0.7290
- F1: 0.6923
- ROC-AUC: 0.7116
- Log loss: 0.6291
- Brier score: 0.2193
- Confusion matrix [[TN,FP],[FN,TP]]: [[1640, 1363], [980, 2636]]

## Validation metrics

- n = 1,419
- Accuracy: 0.6068
- Precision: 0.6260
- Recall: 0.7185
- F1: 0.6690
- ROC-AUC: 0.6566
- Log loss: 0.6514
- Brier score: 0.2298
- Confusion matrix [[TN,FP],[FN,TP]]: [[297, 337], [221, 564]]

## Four-way validation comparison

| metric | majority baseline | LR V1 (untuned) | RF V1 (untuned) | RF V2 (tuned) |
|---|---|---|---|---|
| Accuracy | 0.5532 | 0.6110 | 0.5927 | 0.6068 |
| ROC-AUC | 0.5000 | 0.6431 | 0.6278 | 0.6566 |
| F1 | - | 0.6738 | 0.6564 | 0.6690 |
| Log loss | - | 0.6564 | 0.6679 | 0.6514 |
| Brier score | - | 0.2322 | 0.2365 | 0.2298 |

## Metric deltas

| metric | RF V2 - RF V1 | RF V2 - LR V1 |
|---|---|---|
| Accuracy | +0.0141 | -0.0042 |
| ROC-AUC | +0.0287 | +0.0135 |
| F1 | +0.0127 | -0.0047 |
| Log loss | -0.0165 | -0.0051 |
| Brier score | -0.0067 | -0.0024 |

## Train-validation gap: V1 vs. V2 (overfitting control)

| model | accuracy gap | ROC-AUC gap |
|---|---|---|
| LR V1 | -0.0201 | -0.0220 |
| RF V1 (untuned) | +0.3936 | +0.3716 |
| RF V2 (tuned) | +0.0393 | +0.0550 |

Controlling complexity (`max_depth`, `min_samples_leaf`, `min_samples_split` no longer at their most permissive values) reduced the train-validation accuracy gap from +0.3936 (V1) to +0.0393 (V2) and the ROC-AUC gap from +0.3716 to +0.0550 - a substantial reduction in overfitting, evaluated honestly on the real held-out validation period, not just on CV.

## Probability diagnostics (validation only)

`reports/figures/random_forest_v2_roc.png`, `random_forest_v2_calibration.png`, `random_forest_v2_probability_distribution.png`. No calibration correction applied. Per-bin calibration:

| bin | n | mean predicted | empirical win rate |
|---|---|---|---|
| [0.1,0.2) | 1 | 0.192 | 0.000 |
| [0.2,0.3) | 31 | 0.269 | 0.290 |
| [0.3,0.4) | 184 | 0.360 | 0.391 |
| [0.4,0.5) | 302 | 0.451 | 0.464 |
| [0.5,0.6) | 439 | 0.555 | 0.528 |
| [0.6,0.7) | 370 | 0.642 | 0.711 |
| [0.7,0.8) | 88 | 0.735 | 0.739 |
| [0.8,0.9) | 4 | 0.816 | 1.000 |

Probability percentiles: | min | p5 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| 0.192 | 0.336 | 0.450 | 0.551 | 0.621 | 0.713 | 0.829 |

## Side-symmetry diagnostic

mean **0.0086**, median **0.0071**, p95 **0.0216**, max **0.0380** - diagnostic only, no correction applied, same definition as RF V1.

## Feature importance: V2 vs. V1

Specifically checking whether V2 relies less on V1 features that had high impurity importance but near-zero/negative validation permutation importance:

| feature | V1 impurity | V1 permutation | V2 impurity | V2 permutation |
|---|---|---|---|---|
| days_since_last_match_diff | 0.0953 | -0.0024 | 0.0513 | 0.0019 |
| history_matches_sum | 0.0835 | -0.0025 | 0.0541 | 0.0012 |
| history_matches_min | 0.0749 | -0.0021 | 0.0663 | 0.0039 |

Top 5 V2 impurity importances:

| rank | feature | impurity_importance |
|---|---|---|
| 1 | elo_diff | 0.2665 |
| 2 | total_matches_before_diff | 0.1422 |
| 3 | overall_win_rate_diff | 0.1013 |
| 4 | format_win_rate_diff | 0.0756 |
| 5 | history_matches_min | 0.0663 |

Top 5 V2 permutation importances (validation, ROC-AUC):

| rank | feature | mean_importance | std_importance |
|---|---|---|---|
| 1 | elo_diff | 0.0712 | 0.0148 |
| 2 | total_matches_before_diff | 0.0353 | 0.0056 |
| 3 | avg_series_margin_last_10_diff | 0.0053 | 0.0022 |
| 4 | history_matches_min | 0.0039 | 0.0019 |
| 5 | matches_last_30_days_diff | 0.0027 | 0.0010 |

Full tables: `reports/tables/random_forest_v2_feature_importance.csv` / `random_forest_v2_permutation_importance.csv`. Diagnostic only - no feature selection performed.

## Verdict

RF V2 vs. RF V1 (validation ROC-AUC): +0.0287 - RF V2 improves on RF V1 on this held-out validation evidence, alongside a much smaller train-validation gap (Section above).
RF V2 vs. LR V1 (validation ROC-AUC): +0.0135 - **the tuned Random Forest (RF V2) outperformed the untuned Logistic Regression baseline (LR V1)** on this validation evidence. This is *not* a declaration that Random Forest is definitively superior to Logistic Regression as an algorithm - Logistic Regression has not yet been tuned, and a fair algorithm-vs-algorithm comparison requires comparable tuning effort on both sides.

No final project model is declared. XGBoost has not been evaluated, Logistic Regression has not been tuned, and the internal test set has not been used for this or any comparison.

## Status

- **TEST = SEALED** - not opened or scored in this phase.
- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.
- No configuration iteration occurred after seeing this validation result.
