# Logistic Regression V2 - L2 Regularization Tuning (Chronological, TRAIN-only)

## Temporal CV methodology

The **same** 4 expanding-window folds used for RF V2 and XGB V2 (`data/modeling/random_forest_cv_folds_v2.csv`, reused byte-identically), so the later tuned-model comparison is directly comparable. Fold chronology (`max(fold-train datetime) < min(fold-validation datetime)`) is re-verified at runtime, and every fold id lies inside the global TRAIN partition.

Unlike XGBoost, Logistic Regression needs **no inner early-stopping split**: optimization stops on convergence of the *training objective*, never on validation performance.

## Proof the main validation partition was absent

This script never opens the main split manifest - it reads only the fold manifest, which by construction contains only TRAIN match_ids. Lambda selection therefore could not have been influenced by the 1,419-match main validation partition. (AST-verified in `scripts/validate_phase4a1.py`.)

## Mirroring and preprocessing (per fold, independent)

For each fold: mirror the fold-training rows only (directional diffs negated, symmetric/context unchanged, target flipped), fit preprocessing on **that augmented fold-train only**, then transform (a) augmented fold-train for fitting, (b) the original unmirrored fold-train for TRAIN metrics, (c) fold-validation, never mirrored. Mirrored rows are augmented *observations*, never additional independent matches.

## Lambda search space (fixed before any results)

`[0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]` - 11 candidates x 4 folds = 44 scratch fits. **`lambda=0.0` is the unregularized structural reference and is fully eligible for selection** - if chronological CV finds regularization does not help, LR V2 may legitimately keep it.

## Optimization settings (NOT predictive hyperparameters)

`alpha=0.01` and `max_iterations=20000` are fixed for **every** candidate. They exist only to make gradient descent converge reliably and were never chosen by comparing validation predictions. Because all candidates share them, the lambda comparison is clean.

**Note on comparing to LR V1**: V1 used `alpha=0.001` with a fixed 10,000 iterations and its cost curve was still descending. LR V2 uses `alpha=0.01` with a convergence criterion, so even a selected `lambda=0` would not reproduce LR V1 - it would be *V1's regularization, properly converged*. The V1->V2 comparison therefore confounds regularization with convergence; the `lambda=0` row below is the correct reference for isolating the pure regularization effect.

## Convergence criterion (training-objective only, dual)

TRAINING-OBJECTIVE ONLY. min_iterations=1000, checked every 100 iterations. Converged if EITHER (A) relative training-cost improvement over the preceding 100-iteration window is < 1e-07 on 3 consecutive checks, OR (B) the regularized gradient norm sqrt(||dj_dw||^2 + dj_db^2) < 1e-05. Validation data is never consulted.

Criterion (B) - the regularized gradient-norm condition - exists so a fit that has genuinely reached a flat, small-gradient optimum is not excluded merely because the plateau tolerance in (A) is unnecessarily strict. Both conditions read only the training objective and its gradient.

## Learning-rate safety check (alpha=0.01, TRAIN-only)

Run on fold 1's augmented training data only. No validation data touched, no validation metrics compared. If this had failed, the run would have STOPPED rather than silently trying other alphas.

| lambda | costs finite | weights finite | objective decreased | iterations | converged (by) | initial cost | final cost | final \|grad\| |
|---|---|---|---|---|---|---|---|---|
| 0.0 | True | True | True | 20000 | False (None) | 0.692877 | 0.674107 | 1.067e-03 |
| 1.0 | True | True | True | 20000 | False (None) | 0.692877 | 0.674149 | 9.893e-04 |
| 10.0 | True | True | True | 20000 | False (None) | 0.692877 | 0.674453 | 5.009e-04 |

## Selection rule (fixed before the search)

0) ELIGIBILITY: a lambda is selectable only if ALL 4 folds stayed finite AND met the convergence criterion (either the sustained relative-cost plateau OR the regularized gradient-norm condition). 1) PRIMARY: lowest mean CV log loss. 2) EQUIVALENCE: candidates within 0.002 of the best mean log loss are treated as essentially equivalent. 3) SECONDARY: highest mean CV ROC-AUC. 4) TERTIARY: lower CV log-loss standard deviation. 5) FINAL TIE-BREAK: LARGER lambda (stronger regularization = simpler, more constrained model among predictively equivalent options), then ascending lambda order. Accuracy is never an objective.

## All lambda candidates by mean CV log loss

| rank | lambda | log loss (mean±std) | ROC-AUC (mean±std) | Brier | acc | F1 | acc gap | AUC gap | median iters | all folds converged | mean \|grad\| | mean \|\|w\|\|2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50.0 | 0.66993±0.01099 | 0.61967±0.01596 | 0.23832 | 0.5942 | 0.6491 | -0.0077 | -0.0046 | 11450 | True | 2.45e-04 | 0.3717 |
| 2 | 25.0 | 0.67049±0.01159 | 0.61947±0.01643 | 0.23853 | 0.5937 | 0.6483 | -0.0073 | -0.0043 | 18050 | False | 3.09e-04 | 0.3987 |
| 3 | 10.0 | 0.67089±0.01202 | 0.61926±0.01685 | 0.23869 | 0.5939 | 0.6484 | -0.0066 | -0.0039 | 20000 | False | 4.79e-04 | 0.4242 |
| 4 | 5.0 | 0.67105±0.01218 | 0.61913±0.01710 | 0.23875 | 0.5937 | 0.6483 | -0.0075 | -0.0039 | 20000 | False | 5.86e-04 | 0.4355 |
| 5 | 2.0 | 0.67115±0.01228 | 0.61911±0.01718 | 0.23879 | 0.5939 | 0.6483 | -0.0074 | -0.0038 | 20000 | False | 6.65e-04 | 0.4434 |
| 6 | 1.0 | 0.67118±0.01231 | 0.61909±0.01721 | 0.23881 | 0.5939 | 0.6483 | -0.0076 | -0.0038 | 20000 | False | 6.95e-04 | 0.4462 |
| 7 | 0.5 | 0.67120±0.01233 | 0.61907±0.01725 | 0.23881 | 0.5939 | 0.6483 | -0.0079 | -0.0038 | 20000 | False | 7.11e-04 | 0.4477 |
| 8 | 0.1 | 0.67121±0.01234 | 0.61910±0.01726 | 0.23882 | 0.5942 | 0.6486 | -0.0083 | -0.0038 | 20000 | False | 7.24e-04 | 0.4489 |
| 9 | 0.01 | 0.67121±0.01234 | 0.61909±0.01727 | 0.23882 | 0.5942 | 0.6486 | -0.0083 | -0.0038 | 20000 | False | 7.27e-04 | 0.4492 |
| 10 | 0.001 | 0.67121±0.01234 | 0.61909±0.01727 | 0.23882 | 0.5942 | 0.6486 | -0.0083 | -0.0038 | 20000 | False | 7.27e-04 | 0.4492 |
| 11 | 0.0 | 0.67121±0.01234 | 0.61909±0.01727 | 0.23882 | 0.5942 | 0.6486 | -0.0083 | -0.0038 | 20000 | False | 7.27e-04 | 0.4492 |

Convergence trigger counts across all 44 fits: {'NOT_CONVERGED': 29, 'relative_cost_plateau': 15}. Non-converged fits: 29. Eligible candidates (all 4 folds converged): **1 of 11**.

### Convergence-criterion issues in this run (reported, not hidden)

**1. Criterion (B) never fired.** The smallest final regularized gradient norm observed across all 44 fits was `2.381e-04`, while the predefined tolerance was `1e-05` - roughly 24x larger. As implemented, the gradient-norm condition was therefore stricter than the cost-plateau condition and never rescued a candidate, so it did **not** serve its intended purpose of preventing exclusion of effectively-optimized fits. The tolerance was fixed before the search and has deliberately **not** been loosened afterwards, since changing a convergence rule after seeing results would turn it into a results-driven choice.

**2. The eligibility gate was degenerate.** 29 of 44 fits reached `max_iterations=20000` without meeting the plateau criterion, leaving only 1 eligible candidate(s). A gate that admits one option is not meaningfully selecting.

### Sensitivity of the selection to the eligibility gate

Because this matters for trusting the result, the same ladder was re-applied **ignoring the convergence gate entirely**: all 11 of 11 candidates fall inside the 0.002 log-loss equivalence band (spread is only 0.00128), so the secondary ROC-AUC rule decides, and it picks lambda=[50.0]. The gated selection picked lambda=[50.0]. **These agree, so the degenerate gate did not drive the outcome.**

### Two further honest caveats

- **The selected lambda sits at the edge of the search grid** (`50.0` is the largest value searched), so the true optimum may lie beyond it. The grid was fixed in advance and is deliberately **not** extended here, since re-searching after seeing results is exactly what this phase's protocol forbids. This is flagged as a limitation for a future phase.
- **Regularization barely matters on this problem.** The entire lambda sweep moves mean CV log loss by only 0.00128 and mean CV ROC-AUC by 0.00060. Every candidate is within the predefined equivalence band, so the honest conclusion is that L2 strength is close to irrelevant here rather than that lambda=50 is meaningfully superior.

## Did L2 regularization improve anything?

- **Log loss**: best lambda=50.0 at 0.66993 vs lambda=0 at 0.67121 (**-0.00128**).
- **ROC-AUC**: best lambda=50.0 at 0.61967 vs lambda=0 at 0.61909 (**+0.00058**).
- **Brier**: best lambda=50.0 at 0.23832 vs lambda=0 at 0.23882 (**-0.00050**).
- **Coefficient shrinkage**: mean ||w||_2 falls from 0.4492 at lambda=0 to 0.3717 at lambda=50.0 - L2 is demonstrably shrinking the coefficients as intended.
- **Generalization gap**: mean train-validation ROC-AUC gap moves from -0.0038 (lambda=0) to -0.0046 (lambda=50.0).
- **Convergence speed**: median iterations moves from 20000 (lambda=0) to 11450 (lambda=50.0).

Differences of a few ten-thousandths in mean CV log loss across only 4 folds should not be over-interpreted - that is exactly why the 0.002 equivalence epsilon and the deterministic tie-break ladder exist.

## Selected configuration (FROZEN)

**lambda = 50.0**, selected via: primary (lowest mean CV log loss, unique).

- CV log loss 0.66993 ± 0.01099
- CV ROC-AUC 0.61967 ± 0.01596
- CV Brier 0.23832 | CV accuracy 0.5942
- all folds converged: True | median iterations 11450

Frozen in `data/modeling/logistic_regression_v2_selected_config.json`. Only now may the main validation partition be evaluated, exactly once, in `scripts/train_logistic_regression_v2.py`.
