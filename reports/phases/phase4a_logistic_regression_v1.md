# Phase 4A - Model 1: Logistic Regression From Scratch (V1 baseline)

## 1. Why Logistic Regression is Model 1

The project proposal specifies Model 1 as Logistic Regression, adapted from the course lab. It serves as the interpretable linear baseline that every later, more flexible model (Random Forest, XGBoost) must be compared against.

## 2. Implemented from scratch

The model is implemented entirely from scratch in `scripts/models/logistic_regression_scratch.py`, adapted from `reference/Lab_2_Logistic_Regression_answer.ipynb`. No `sklearn.linear_model.LogisticRegression`, `statsmodels`, `scipy.optimize`, or any other ML estimator was used to fit this model - `sklearn.metrics` is used only after our own `predict_proba`/`predict` produced predictions.

## 3. Lab functions adapted

From `reference/Lab_2_Logistic_Regression_answer.ipynb`: `sigmoid` (cell 15, `UNQ_C1`), `compute_cost` (cell 23, `UNQ_C2`), `compute_gradient` (cell 32, `UNQ_C3`), `gradient_descent` (cell 40), `predict` (cell 48, `UNQ_C4`), `compute_cost_reg` (cell 71, `UNQ_C5`), `compute_gradient_reg` (cell 78, `UNQ_C6`) - all preserved with the lab's original mathematics (binary cross-entropy cost, `dj_dw = Err@X/m`, `dj_db = sum(Err)/m`, `w -= alpha*dj_dw`, `b -= alpha*dj_db`, L2 penalty `(lambda_/(2m))*sum(w^2)` added to cost and `(lambda_/m)*w` added to `dj_dw`, bias never regularized).

## 4. Project-specific modifications

Clearly marked `[PROJECT ADAPTATION]`/`[PROJECT ADDITION]` in the source: numerical-stability probability clipping in `compute_cost`/`compute_cost_reg` (prevents `log(0)` on the larger, more separable CS2 feature set - does not change the BCE objective); divergence detection inside `gradient_descent` (raises `GradientDescentDivergenceError` on non-finite or catastrophically increasing cost, so a broken run stops loudly); `predict_proba` (factored out of the lab's `predict`); a configurable `threshold` parameter on `predict` (defaults to the lab's hardcoded 0.5); the entire `scripts/preprocessing_logistic_v1.py` module (imputation/standardization/reference encoding, mirrored augmentation); model serialization to `.npz`/`.json`.

## 5. Chronological split methodology

`scripts/build_series_split_v1.py` splits the 9,456-row Phase 3 development set by exact `datetime` group into train/validation/test, choosing the group boundary closest to the 70%/85% cumulative-row-count marks: train=6,619 (2023-01-10 09:30:00 to 2025-08-30 14:00:00), validation=1,419 (2025-08-30 15:00:00 to 2026-01-27 12:30:00). Full detail in `reports/phase4a_split_summary.md`.

## 6. Why validation/test are chronological

A random split would let the model be evaluated on matches that occurred *before* some of its training data chronologically - exactly the kind of leakage Phase 3's historical-feature engine was built to prevent. A chronological split mirrors genuine deployment: predicting future matches from past information only.

## 7. Why same-timestamp groups are not split

Phase 3's engine already treats every match sharing an exact timestamp as mutually invisible to each other (neither can see the other's result). If such a group were divided across partitions, a validation/test row could end up in the same simultaneous batch as a training row it was never meant to be distinguishable from at prediction time - keeping the whole group in one partition preserves that guarantee.

## 8. Train-only preprocessing

All imputation medians, means, and standard deviations for the 12 continuous features, and the `bestOf`/`tier` reference-category encoding, are fit exclusively on the training partition (specifically the augmented train+mirror set - see Section 9) and saved to `data/modeling/logistic_preprocessing_v1.json`. Validation (and later test) are only ever *transformed* with these already-fitted values, never used to compute them.

## 9. Mirrored training augmentation and the Team1 orientation bias

Phase 1/2.5 established a persistent ~55% Team1 win-rate artifact from how `team1`/`team2` are assigned, unrelated to real skill. **Mirroring is applied to the raw (pre-preprocessing) training rows, before preprocessing is fit** - not by negating already-standardized values. This was a deliberate correction during planning: fitting standardization on the original (biased) training mean and then separately negating+re-standardizing a mirrored raw row would not give exact negatives whenever that mean is non-zero (`(-x-mean)/std != -(x-mean)/std`), silently reintroducing the very bias mirroring exists to cancel. Instead: 6,619 raw training rows are mirrored (every directional diff feature negated, symmetric/context features unchanged, target flipped) and concatenated with the originals into a 13,238-row augmented set; preprocessing is fit on *that*. Verified: augmented target mean = **0.5** (exactly 0.5), and every directional diff feature's augmented raw mean is ~0 by construction (`assert_augmented_symmetry`, tight tolerance for the fully-populated features, a documented looser tolerance for `days_since_last_match_diff` because NaN-negated-is-still-NaN means a missing value's mirrored pair shares one imputed value rather than becoming exact negatives - a small, bounded, expected deviation, not a bug). A dedicated future-inference symmetry test (`tests/test_preprocessing_logistic.py`) proves the *same fitted artifact* transforms a genuinely reversed future matchup consistently with this training-time scheme. Validation and test are never mirrored.

## 10. Gradient descent configuration

`alpha=0.001`, `num_iters=10000`, `lambda_=0.0` (unregularized V1 baseline) - a single, non-tuned configuration. `num_iters=10000` is lab-style. `alpha=0.001` is not just a conservative project guess: cell 42 of `reference/Lab_2_Logistic_Regression_answer.ipynb` - the lab's own **unregularized** (`lambda_=0`) example - literally uses `alpha=0.001, iterations=10000`, verified by direct inspection of that cell rather than assumed. (The lab's separate *regularized* example, cell 84, uses `alpha=0.01, lambda_=0.01` - a different configuration for a different, regularized run, not reused here since V1 is unregularized.) `w` initialized to zeros, `b=0`, batch gradient descent only, no other optimizer.

## 11. Cost convergence

Initial cost: **0.693102**. Final cost: **0.666181**. Absolute decrease: **0.026921**. Relative decrease: **3.88%**. Cost history is finite throughout (`reports/figures/logistic_regression_cost_v1.png`); no divergence was encountered at this configuration.

## 12. Train metrics (unmirrored original orientation)

Evaluated on the original, unmirrored training rows (not the 2x augmented matrix used to fit the model) so train and validation are directly comparable.

- n = 6,619
- Accuracy: 0.5909 (majority-class reference: 0.5463)
- Precision: 0.6158
- Recall: 0.6679
- F1: 0.6408
- ROC-AUC: 0.6210
- Log loss: 0.6662
- Brier score: 0.2369
- Confusion matrix [[TN,FP],[FN,TP]]: [[1496, 1507], [1201, 2415]]

## 13. Validation metrics

- n = 1,419
- Accuracy: 0.6110 (majority-class reference: 0.5532)
- Precision: 0.6284
- Recall: 0.7261
- F1: 0.6738
- ROC-AUC: 0.6431
- Log loss: 0.6564
- Brier score: 0.2322
- Confusion matrix [[TN,FP],[FN,TP]]: [[297, 337], [215, 570]]

## 14. Coefficient interpretation

`reports/tables/logistic_regression_coefficients_v1.csv` (feature, coefficient, abs_coefficient), bias/intercept `b = 0.0000`. Standardized-continuous features (the 10 diffs + `history_matches_min`/`history_matches_sum`) have coefficients in standardized units; `both_teams_have_*` are binary; `bestOf_BO3`/`bestOf_BO5`/`tier_tier2`/`tier_tier3` are one-hot relative to the BO1/tier1 reference category. Top features by absolute magnitude:

| feature | coefficient | abs_coefficient |
|---|---|---|
| elo_diff | 0.2695 | 0.2695 |
| total_matches_before_diff | 0.1563 | 0.1563 |
| overall_win_rate_diff | 0.0591 | 0.0591 |
| avg_series_margin_last_10_diff | 0.0452 | 0.0452 |
| avg_series_margin_last_5_diff | 0.0373 | 0.0373 |
| matches_last_30_days_diff | 0.0215 | 0.0215 |
| days_since_last_match_diff | -0.0156 | 0.0156 |
| format_win_rate_diff | 0.0129 | 0.0129 |

Sign and relative direction only - **no causal claims**. For example, a positive `elo_diff` coefficient means higher Team1 historical ELO relative to Team2 is *associated with* higher predicted Team1 win probability, holding the model's other inputs fixed; it does not mean ELO *causes* wins.

**Observation**: the 5 symmetric confidence features (`history_matches_min`/`_sum`, `both_teams_have_*`) and the 4 one-hot `bestOf`/`tier` context columns all learned coefficients on the order of `1e-18` - effectively zero - while the bias `b` also stayed at essentially 0. This is an expected consequence of the mirrored augmentation, not a training failure: at initialization (`w=0,b=0`) every feature's error term is `sigmoid(0)-y = 0.5-y` for a row and `0.5-(1-y) = -(0.5-y)` for its mirror, so a symmetric feature (identical in both rows) receives exactly cancelling gradient contributions from each mirrored pair, while a directional feature (sign-flipped in the mirror) receives *reinforcing* contributions - and this near-cancellation is self-sustaining as long as the symmetric-feature weights and `b` stay close to 0. In other words, the augmentation doesn't just balance the target class - it structurally suppresses any coefficient on a feature that carries no side-relative signal, which is exactly the desired behavior for confidence/context features that were never meant to indicate *which side* wins.

## 15. Probability / calibration observations (validation, diagnostic only)

`reports/figures/logistic_regression_roc_v1.png` and `logistic_regression_calibration_v1.png`. No calibration correction (isotonic/Platt) is applied at this stage - the plot is diagnostic only. Per-bin mean predicted probability vs. empirical win rate:

| bin | n | mean predicted | empirical win rate |
|---|---|---|---|
| [0.1,0.2) | 7 | 0.172 | 0.286 |
| [0.2,0.3) | 68 | 0.262 | 0.279 |
| [0.3,0.4) | 147 | 0.358 | 0.449 |
| [0.4,0.5) | 290 | 0.454 | 0.441 |
| [0.5,0.6) | 390 | 0.550 | 0.564 |
| [0.6,0.7) | 311 | 0.644 | 0.640 |
| [0.7,0.8) | 167 | 0.745 | 0.713 |
| [0.8,0.9) | 39 | 0.834 | 0.821 |

## 16. Limitations of this V1 baseline

- Purely linear decision boundary in the 19 transformed features - no interaction terms.
- No regularization tuning yet (`lambda_=0` fixed); no learning-rate or iteration-count tuning.
- Mirrored augmentation neutralizes the orientation bias's effect on training, but does not explain or fix its unknown root cause (Phase 2.5).
- Only the 17 Phase-3 series-level features - no map-level or player-level detail yet.
- 463 train rows have `both_teams_have_history==0` (at least one side is a cold-start team with no prior match) - these add noise the model cannot fully distinguish from a genuinely even matchup beyond the confidence flags.
- Decision threshold fixed at the lab's default 0.5, not tuned for any downstream use case.
- Single chronological train/validation split - no cross-validation.
- BO5 is a small sample in both train and validation - metrics on that subset are less reliable.

## Status

- **Internal test partition: SEALED** - not opened or scored in this phase.
- **Cologne 2026 / post-Cologne: UNTOUCHED** - structurally absent from `series_features_v1.parquet`.
- No hyperparameter tuning performed.
- Random Forest and XGBoost have not been trained.
