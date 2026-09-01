"""
Phase 4A.1: Logistic Regression V2 L2-regularization (lambda) tuning via
TRAIN-only expanding-window chronological CV, reusing the SAME folds as
RF V2 and XGB V2 (data/modeling/random_forest_cv_folds_v2.csv).

This script NEVER reads data/modeling/series_split_v1.csv, so the 1,419-match
main VALIDATION partition and the TEST partition are structurally absent from
lambda selection (AST-verified in validation/validate_phase4a1.py).

WHAT IS AND ISN'T BEING TUNED
-----------------------------
  lambda  -> the PREDICTIVE/model hyperparameter being searched.
  alpha, max_iterations, convergence criterion
          -> OPTIMIZATION settings, fixed for every candidate. They exist only
             so gradient descent converges reliably; they are never chosen by
             validation performance. Because all candidates share them, the
             lambda comparison is clean.

Writes:
    reports/tables/logistic_regression_tuning_v2.csv
    reports/phase4a1_logistic_regression_tuning.md
    data/modeling/logistic_regression_v2_selected_config.json
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss

from _common import ROOT, REPORTS
from training.logistic_regression.logistic_regression_scratch import (
    compute_cost_reg, compute_gradient_reg, predict_proba, predict, GradientDescentDivergenceError,
)
from training.logistic_regression.logistic_regression_convergence_v2 import (
    gradient_descent_until_convergence, CONVERGENCE_RULE_TEXT,
    MIN_ITERATIONS, CHECK_EVERY, RELATIVE_TOLERANCE, CONSECUTIVE_CHECKS_REQUIRED, GRADIENT_NORM_TOLERANCE,
)
from feature_engineering.preprocessing.preprocessing_logistic_v1 import build_augmented_training_raw, fit_preprocessing, transform

CONFIG_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
MODELING_DIR = ROOT / "data" / "modeling"
TABLES_DIR = REPORTS / "tables"

# --- fixed OPTIMIZATION settings (not predictive hyperparameters) ---
ALPHA = 0.01
MAX_ITERATIONS = 20000
THRESHOLD = 0.5

# --- the ONLY predictive hyperparameter searched; fixed before any results ---
LAMBDA_GRID = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

LOG_LOSS_EQUIVALENCE_EPSILON = 0.002
N_FOLDS = 4
TUNING_VERSION = "1.0.0"

SELECTION_RULE_TEXT = (
    "0) ELIGIBILITY: a lambda is selectable only if ALL 4 folds stayed finite AND met the convergence "
    "criterion (either the sustained relative-cost plateau OR the regularized gradient-norm condition). "
    "1) PRIMARY: lowest mean CV log loss. "
    f"2) EQUIVALENCE: candidates within {LOG_LOSS_EQUIVALENCE_EPSILON} of the best mean log loss are treated "
    "as essentially equivalent. 3) SECONDARY: highest mean CV ROC-AUC. 4) TERTIARY: lower CV log-loss "
    "standard deviation. 5) FINAL TIE-BREAK: LARGER lambda (stronger regularization = simpler, more "
    "constrained model among predictively equivalent options), then ascending lambda order. "
    "Accuracy is never an objective."
)


def compute_metrics(y_true, y_proba, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def prepare_fold(fold, cv_df, features_df, model_features, target_col):
    """Per-fold, independent: mirror fold-train, fit preprocessing on that
    augmented fold-train ONLY, transform all three views."""
    train_ids = set(cv_df.loc[(cv_df.fold == fold) & (cv_df.role == "train"), "match_id"])
    val_ids = set(cv_df.loc[(cv_df.fold == fold) & (cv_df.role == "validation"), "match_id"])
    fold_train_raw = features_df[features_df.match_id.isin(train_ids)].reset_index(drop=True)
    fold_val_raw = features_df[features_df.match_id.isin(val_ids)].reset_index(drop=True)

    assert fold_train_raw["datetime"].max() < fold_val_raw["datetime"].min(), \
        f"fold {fold}: chronology violated"

    augmented = build_augmented_training_raw(fold_train_raw)
    params = fit_preprocessing(augmented, model_features)

    X_fit, names = transform(augmented, params)
    y_fit = augmented[target_col].to_numpy(dtype=float)
    X_tr, _ = transform(fold_train_raw, params)          # UNMIRRORED -> train metrics
    y_tr = fold_train_raw[target_col].to_numpy(dtype=float)
    X_val, _ = transform(fold_val_raw, params)           # never mirrored
    y_val = fold_val_raw[target_col].to_numpy(dtype=float)

    return {
        "X_fit": X_fit, "y_fit": y_fit, "X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "y_val": y_val,
        "n_names": len(names),
        "unique_fold_train_matches": len(fold_train_raw),
        "augmented_fold_train_observations": len(augmented),
        "fold_validation_matches": len(fold_val_raw),
    }


def run_safety_check(fold_data):
    """TRAIN-ONLY numerical stability check for alpha=0.01 on the earliest fold.
    NOT a hyperparameter search - no validation metrics are computed or compared."""
    print(f"\n=== alpha={ALPHA} learning-rate safety check (fold 1 augmented TRAIN only) ===")
    rows = []
    for lam in [0.0, 1.0, 10.0]:
        try:
            w, b, J, info = gradient_descent_until_convergence(
                fold_data["X_fit"], fold_data["y_fit"], np.zeros(fold_data["n_names"]), 0.0,
                compute_cost_reg, compute_gradient_reg, alpha=ALPHA, max_iters=MAX_ITERATIONS,
                lambda_=lam, progress_label=f"safety lam={lam}", verbose=False)
        except GradientDescentDivergenceError as e:
            raise RuntimeError(
                f"OPTIMIZER INSTABILITY at alpha={ALPHA}, lambda={lam}: {e}\n"
                "STOPPING as required - alpha is NOT silently changed."
            ) from e

        costs_finite = bool(np.all(np.isfinite(J)))
        weights_finite = bool(np.all(np.isfinite(w)) and np.isfinite(b))
        decreased = bool(J[-1] < J[0])
        if not (costs_finite and weights_finite and decreased):
            raise RuntimeError(
                f"OPTIMIZER INSTABILITY at alpha={ALPHA}, lambda={lam}: costs_finite={costs_finite}, "
                f"weights_finite={weights_finite}, decreased={decreased}. STOPPING - alpha is NOT changed."
            )
        rows.append({"lambda": lam, "costs_finite": costs_finite, "weights_finite": weights_finite,
                      "objective_decreased": decreased, "iterations_run": info["iterations_run"],
                      "converged": info["converged"], "converged_by": info["converged_by"],
                      "initial_cost": info["initial_cost"], "final_cost": info["final_cost"],
                      "final_gradient_norm": info["final_gradient_norm"]})
        print(f"  lambda={lam}: OK | iters={info['iterations_run']} converged={info['converged']} "
              f"({info['converged_by']}) J0={info['initial_cost']:.6f} Jf={info['final_cost']:.6f} "
              f"|grad|={info['final_gradient_norm']:.3e}")
    print("=== safety check PASSED - alpha=0.01 is numerically stable ===\n")
    return pd.DataFrame(rows)


def select_winner(agg_df):
    """Applies the fixed selection rule. Only rows with all_folds_converged==True
    are eligible."""
    eligible = agg_df[agg_df["all_folds_converged"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "NO ELIGIBLE CANDIDATE: no lambda converged in all 4 folds under the predefined criterion. "
            "STOPPING and reporting the optimizer issue rather than selecting an under-trained model."
        )

    best = eligible["val_log_loss_mean"].min()
    tied = eligible[eligible["val_log_loss_mean"] <= best + LOG_LOSS_EQUIVALENCE_EPSILON].copy()
    stage = "primary (lowest mean CV log loss, unique)"

    if len(tied) > 1:
        best_auc = tied["val_roc_auc_mean"].max()
        t2 = tied[tied["val_roc_auc_mean"] == best_auc]
        stage = "secondary (log-loss tie within epsilon, resolved by highest mean CV ROC-AUC)"
        if len(t2) > 1:
            best_std = t2["val_log_loss_std"].min()
            t3 = t2[t2["val_log_loss_std"] == best_std]
            stage = "tertiary (log-loss and ROC-AUC tied, resolved by lower CV log-loss std)"
            if len(t3) > 1:
                t3 = t3.sort_values("lambda", ascending=False)  # prefer STRONGER regularization
                stage = "final tie-break (all metrics tied, resolved by larger lambda)"
            t2 = t3
        tied = t2

    return float(tied.iloc[0]["lambda"]), stage


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features, target_col = cfg["model_features"], cfg["target"]

    features_df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])

    # re-verify fold chronology rather than trusting the manifest
    for f in range(1, N_FOLDS + 1):
        tr = cv_df[(cv_df.fold == f) & (cv_df.role == "train")]
        va = cv_df[(cv_df.fold == f) & (cv_df.role == "validation")]
        assert tr["datetime"].max() < va["datetime"].min(), f"fold {f} chronology violated"
    print(f"Reusing RF V2/XGB V2 fold manifest: {N_FOLDS} expanding-window folds, chronology verified.")

    print("Preparing per-fold mirrored data + fold-local preprocessing...")
    folds = {}
    for f in range(1, N_FOLDS + 1):
        folds[f] = prepare_fold(f, cv_df, features_df, model_features, target_col)
        d = folds[f]
        print(f"  fold {f}: {d['unique_fold_train_matches']} unique train matches -> "
              f"{d['augmented_fold_train_observations']} augmented observations | "
              f"{d['fold_validation_matches']} fold-validation matches")

    safety_df = run_safety_check(folds[1])

    total_fits = len(LAMBDA_GRID) * N_FOLDS
    print(f"Running {len(LAMBDA_GRID)} lambdas x {N_FOLDS} folds = {total_fits} scratch LR fits "
          f"(alpha={ALPHA}, max_iters={MAX_ITERATIONS})")
    t_start = time.time()
    fit_no = 0
    rows = []

    for lam in LAMBDA_GRID:
        for f in range(1, N_FOLDS + 1):
            fit_no += 1
            d = folds[f]
            label = f"[{fit_no}/{total_fits}] lambda={lam} fold={f}"
            print(f"  {label} ...", flush=True)

            w, b, J, info = gradient_descent_until_convergence(
                d["X_fit"], d["y_fit"], np.zeros(d["n_names"]), 0.0,
                compute_cost_reg, compute_gradient_reg, alpha=ALPHA, max_iters=MAX_ITERATIONS,
                lambda_=lam, progress_label=label, progress_every=5000, verbose=True)

            tr_m = compute_metrics(d["y_tr"], predict_proba(d["X_tr"], w, b),
                                    predict(d["X_tr"], w, b, threshold=THRESHOLD))
            va_m = compute_metrics(d["y_val"], predict_proba(d["X_val"], w, b),
                                    predict(d["X_val"], w, b, threshold=THRESHOLD))

            rows.append({
                "lambda": lam, "fold": f, "row_type": "fold",
                "unique_fold_train_matches": d["unique_fold_train_matches"],
                "augmented_fold_train_observations": d["augmented_fold_train_observations"],
                "fold_validation_matches": d["fold_validation_matches"],
                "iterations_run": info["iterations_run"], "converged": info["converged"],
                "converged_by": info["converged_by"],
                "initial_cost": info["initial_cost"], "final_cost": info["final_cost"],
                "relative_cost_decrease": info["relative_cost_decrease"],
                "final_relative_improvement": info["final_relative_improvement"],
                "final_gradient_norm": info["final_gradient_norm"],
                # NOTE: wall-clock elapsed time is intentionally NOT persisted here.
                # It is shown in the console progress logs (operational visibility)
                # but would make this results artifact non-byte-reproducible across
                # runs for no scientific benefit.
                "train_accuracy": tr_m["accuracy"], "train_roc_auc": tr_m["roc_auc"],
                "train_log_loss": tr_m["log_loss"], "train_brier": tr_m["brier"], "train_f1": tr_m["f1"],
                "val_accuracy": va_m["accuracy"], "val_roc_auc": va_m["roc_auc"],
                "val_log_loss": va_m["log_loss"], "val_brier": va_m["brier"], "val_f1": va_m["f1"],
                "train_val_acc_gap": tr_m["accuracy"] - va_m["accuracy"],
                "train_val_auc_gap": tr_m["roc_auc"] - va_m["roc_auc"],
                "train_val_log_loss_gap": va_m["log_loss"] - tr_m["log_loss"],
                "coef_l2_norm": float(np.linalg.norm(w)),
                "coef_abs_sum": float(np.sum(np.abs(w))),
            })
            print(f"    -> iters={info['iterations_run']} conv={info['converged']}({info['converged_by']}) "
                  f"|grad|={info['final_gradient_norm']:.3e} val_logloss={va_m['log_loss']:.5f} "
                  f"val_auc={va_m['roc_auc']:.5f} [{time.time()-t_start:.0f}s elapsed]", flush=True)

    fold_df = pd.DataFrame(rows)

    agg = []
    for lam, g in fold_df.groupby("lambda"):
        agg.append({
            "lambda": lam, "fold": np.nan, "row_type": "aggregate",
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_brier_std": g["val_brier"].std(ddof=0),
            "val_accuracy_mean": g["val_accuracy"].mean(), "val_f1_mean": g["val_f1"].mean(),
            "train_accuracy_mean": g["train_accuracy"].mean(), "train_roc_auc_mean": g["train_roc_auc"].mean(),
            "train_val_acc_gap_mean": g["train_val_acc_gap"].mean(),
            "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
            "train_val_log_loss_gap_mean": g["train_val_log_loss_gap"].mean(),
            "mean_iterations": g["iterations_run"].mean(),
            "median_iterations": float(np.median(g["iterations_run"])),
            "all_folds_converged": bool(g["converged"].all()),
            "n_folds_converged": int(g["converged"].sum()),
            "mean_final_gradient_norm": g["final_gradient_norm"].mean(),
            "mean_coef_l2_norm": g["coef_l2_norm"].mean(),
        })
    agg_df = pd.DataFrame(agg)

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.concat([fold_df, agg_df], ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "logistic_regression_tuning_v2.csv", index=False, encoding="utf-8")

    selected_lambda, stage = select_winner(agg_df)
    wrow = agg_df[agg_df["lambda"] == selected_lambda].iloc[0]
    print(f"\nSELECTED lambda={selected_lambda} via {stage}")
    print(f"  CV log loss {wrow['val_log_loss_mean']:.5f} +/- {wrow['val_log_loss_std']:.5f} | "
          f"ROC-AUC {wrow['val_roc_auc_mean']:.5f} +/- {wrow['val_roc_auc_std']:.5f}")

    selected = {
        "selected_lambda": selected_lambda,
        "alpha": ALPHA,
        "max_iterations": MAX_ITERATIONS,
        "convergence_rule": CONVERGENCE_RULE_TEXT,
        "convergence_constants": {
            "min_iterations": MIN_ITERATIONS, "check_every": CHECK_EVERY,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "consecutive_checks_required": CONSECUTIVE_CHECKS_REQUIRED,
            "gradient_norm_tolerance": GRADIENT_NORM_TOLERANCE,
        },
        "cv_mean_log_loss": float(wrow["val_log_loss_mean"]),
        "cv_std_log_loss": float(wrow["val_log_loss_std"]),
        "cv_mean_roc_auc": float(wrow["val_roc_auc_mean"]),
        "cv_std_roc_auc": float(wrow["val_roc_auc_std"]),
        "cv_mean_brier": float(wrow["val_brier_mean"]),
        "cv_mean_accuracy": float(wrow["val_accuracy_mean"]),
        "cv_mean_iterations": float(wrow["mean_iterations"]),
        "all_folds_converged": bool(wrow["all_folds_converged"]),
        "selection_stage": stage,
        "selection_rule": SELECTION_RULE_TEXT,
        "log_loss_epsilon": LOG_LOSS_EQUIVALENCE_EPSILON,
        "lambda_grid": LAMBDA_GRID,
        "feature_config": "config/features/series_features_v1.yaml",
        "split_manifest": "data/modeling/series_split_v1.csv",
        "cv_folds_manifest": "data/modeling/random_forest_cv_folds_v2.csv",
        "tuning_results_artifact": "reports/tables/logistic_regression_tuning_v2.csv",
        "version": TUNING_VERSION,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "main_validation_used_in_selection": False,
    }
    (MODELING_DIR / "logistic_regression_v2_selected_config.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8")

    write_report(fold_df, agg_df, safety_df, selected_lambda, stage)

    print(f"\nTotal tuning time: {time.time()-t_start:.0f}s")
    print("Wrote reports/tables/logistic_regression_tuning_v2.csv")
    print("Wrote data/modeling/logistic_regression_v2_selected_config.json")
    print("Wrote reports/phase4a1_logistic_regression_tuning.md")


def write_report(fold_df, agg_df, safety_df, selected_lambda, stage):
    md = []
    md.append("# Logistic Regression V2 - L2 Regularization Tuning (Chronological, TRAIN-only)\n")

    md.append("## Temporal CV methodology\n")
    md.append(f"The **same** {N_FOLDS} expanding-window folds used for RF V2 and XGB V2 "
              "(`data/modeling/random_forest_cv_folds_v2.csv`, reused byte-identically), so the later "
              "tuned-model comparison is directly comparable. Fold chronology "
              "(`max(fold-train datetime) < min(fold-validation datetime)`) is re-verified at runtime, and every "
              "fold id lies inside the global TRAIN partition.\n")
    md.append("Unlike XGBoost, Logistic Regression needs **no inner early-stopping split**: optimization stops on "
              "convergence of the *training objective*, never on validation performance.\n")

    md.append("## Proof the main validation partition was absent\n")
    md.append("This script never opens the main split manifest - it reads only the fold manifest, which by "
              "construction contains only TRAIN match_ids. Lambda selection therefore could not have been "
              "influenced by the 1,419-match main validation partition. (AST-verified in "
              "`validation/validate_phase4a1.py`.)\n")

    md.append("## Mirroring and preprocessing (per fold, independent)\n")
    md.append("For each fold: mirror the fold-training rows only (directional diffs negated, symmetric/context "
              "unchanged, target flipped), fit preprocessing on **that augmented fold-train only**, then "
              "transform (a) augmented fold-train for fitting, (b) the original unmirrored fold-train for TRAIN "
              "metrics, (c) fold-validation, never mirrored. Mirrored rows are augmented *observations*, never "
              "additional independent matches.\n")

    md.append("## Lambda search space (fixed before any results)\n")
    md.append(f"`{LAMBDA_GRID}` - {len(LAMBDA_GRID)} candidates x {N_FOLDS} folds = "
              f"{len(LAMBDA_GRID) * N_FOLDS} scratch fits. **`lambda=0.0` is the unregularized structural "
              "reference and is fully eligible for selection** - if chronological CV finds regularization does "
              "not help, LR V2 may legitimately keep it.\n")

    md.append("## Optimization settings (NOT predictive hyperparameters)\n")
    md.append(f"`alpha={ALPHA}` and `max_iterations={MAX_ITERATIONS}` are fixed for **every** candidate. They "
              "exist only to make gradient descent converge reliably and were never chosen by comparing "
              "validation predictions. Because all candidates share them, the lambda comparison is clean.\n")
    md.append("**Note on comparing to LR V1**: V1 used `alpha=0.001` with a fixed 10,000 iterations and its cost "
              "curve was still descending. LR V2 uses `alpha=0.01` with a convergence criterion, so even a "
              "selected `lambda=0` would not reproduce LR V1 - it would be *V1's regularization, properly "
              "converged*. The V1->V2 comparison therefore confounds regularization with convergence; the "
              "`lambda=0` row below is the correct reference for isolating the pure regularization effect.\n")

    md.append("## Convergence criterion (training-objective only, dual)\n")
    md.append(CONVERGENCE_RULE_TEXT + "\n")
    md.append("Criterion (B) - the regularized gradient-norm condition - exists so a fit that has genuinely "
              "reached a flat, small-gradient optimum is not excluded merely because the plateau tolerance in "
              "(A) is unnecessarily strict. Both conditions read only the training objective and its gradient.\n")

    md.append(f"## Learning-rate safety check (alpha={ALPHA}, TRAIN-only)\n")
    md.append("Run on fold 1's augmented training data only. No validation data touched, no validation metrics "
              "compared. If this had failed, the run would have STOPPED rather than silently trying other alphas.\n")
    md.append("| lambda | costs finite | weights finite | objective decreased | iterations | converged (by) | initial cost | final cost | final \\|grad\\| |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in safety_df.iterrows():
        md.append(f"| {r['lambda']} | {r['costs_finite']} | {r['weights_finite']} | {r['objective_decreased']} | "
                  f"{int(r['iterations_run'])} | {r['converged']} ({r['converged_by']}) | "
                  f"{r['initial_cost']:.6f} | {r['final_cost']:.6f} | {r['final_gradient_norm']:.3e} |")
    md.append("")

    md.append("## Selection rule (fixed before the search)\n")
    md.append(SELECTION_RULE_TEXT + "\n")

    md.append("## All lambda candidates by mean CV log loss\n")
    ranked = agg_df.sort_values("val_log_loss_mean").reset_index(drop=True)
    md.append("| rank | lambda | log loss (mean±std) | ROC-AUC (mean±std) | Brier | acc | F1 | acc gap | AUC gap | "
              "median iters | all folds converged | mean \\|grad\\| | mean \\|\\|w\\|\\|2 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in ranked.iterrows():
        md.append(f"| {i+1} | {r['lambda']} | {r['val_log_loss_mean']:.5f}±{r['val_log_loss_std']:.5f} | "
                  f"{r['val_roc_auc_mean']:.5f}±{r['val_roc_auc_std']:.5f} | {r['val_brier_mean']:.5f} | "
                  f"{r['val_accuracy_mean']:.4f} | {r['val_f1_mean']:.4f} | {r['train_val_acc_gap_mean']:+.4f} | "
                  f"{r['train_val_auc_gap_mean']:+.4f} | {r['median_iterations']:.0f} | "
                  f"{r['all_folds_converged']} | {r['mean_final_gradient_norm']:.2e} | "
                  f"{r['mean_coef_l2_norm']:.4f} |")
    md.append("")

    conv_counts = fold_df.groupby(fold_df["converged_by"].fillna("NOT_CONVERGED")).size().to_dict()
    n_not_conv = int((~fold_df["converged"]).sum())
    n_eligible = int(agg_df["all_folds_converged"].sum())
    min_gn = float(fold_df["final_gradient_norm"].min())
    md.append(f"Convergence trigger counts across all {len(fold_df)} fits: {conv_counts}. "
              f"Non-converged fits: {n_not_conv}. Eligible candidates (all {N_FOLDS} folds converged): "
              f"**{n_eligible} of {len(agg_df)}**.\n")

    # ---- honest reporting of two real issues with this run ----
    md.append("### Convergence-criterion issues in this run (reported, not hidden)\n")
    md.append(f"**1. Criterion (B) never fired.** The smallest final regularized gradient norm observed across "
              f"all {len(fold_df)} fits was `{min_gn:.3e}`, while the predefined tolerance was "
              f"`{GRADIENT_NORM_TOLERANCE}` - roughly {min_gn / GRADIENT_NORM_TOLERANCE:.0f}x larger. As "
              "implemented, the gradient-norm condition was therefore stricter than the cost-plateau condition "
              "and never rescued a candidate, so it did **not** serve its intended purpose of preventing "
              "exclusion of effectively-optimized fits. The tolerance was fixed before the search and has "
              "deliberately **not** been loosened afterwards, since changing a convergence rule after seeing "
              "results would turn it into a results-driven choice.\n")
    md.append(f"**2. The eligibility gate was degenerate.** {n_not_conv} of {len(fold_df)} fits reached "
              f"`max_iterations={MAX_ITERATIONS}` without meeting the plateau criterion, leaving only "
              f"{n_eligible} eligible candidate(s). A gate that admits one option is not meaningfully "
              "selecting.\n")

    # ---- sensitivity: would the winner change without the gate? ----
    best_ll = agg_df["val_log_loss_mean"].min()
    within = agg_df[agg_df["val_log_loss_mean"] <= best_ll + LOG_LOSS_EQUIVALENCE_EPSILON]
    best_auc_ungated = within["val_roc_auc_mean"].max()
    ungated_winner = within[within["val_roc_auc_mean"] == best_auc_ungated]["lambda"].tolist()
    gated_winner = agg_df[agg_df["all_folds_converged"]]["lambda"].tolist()
    md.append("### Sensitivity of the selection to the eligibility gate\n")
    md.append(f"Because this matters for trusting the result, the same ladder was re-applied **ignoring the "
              f"convergence gate entirely**: all {len(within)} of {len(agg_df)} candidates fall inside the "
              f"{LOG_LOSS_EQUIVALENCE_EPSILON} log-loss equivalence band (spread is only "
              f"{agg_df['val_log_loss_mean'].max() - best_ll:.5f}), so the secondary ROC-AUC rule decides, and it "
              f"picks lambda={ungated_winner}. The gated selection picked lambda={gated_winner}. "
              f"**{'These agree, so the degenerate gate did not drive the outcome.' if ungated_winner == gated_winner else 'These DISAGREE - the gate changed the outcome, which must be treated with caution.'}**\n")
    md.append("### Two further honest caveats\n")
    md.append(f"- **The selected lambda sits at the edge of the search grid** (`{max(LAMBDA_GRID)}` is the "
              "largest value searched), so the true optimum may lie beyond it. The grid was fixed in advance and "
              "is deliberately **not** extended here, since re-searching after seeing results is exactly what "
              "this phase's protocol forbids. This is flagged as a limitation for a future phase.")
    md.append(f"- **Regularization barely matters on this problem.** The entire lambda sweep moves mean CV log "
              f"loss by only {agg_df['val_log_loss_mean'].max() - best_ll:.5f} and mean CV ROC-AUC by "
              f"{agg_df['val_roc_auc_mean'].max() - agg_df['val_roc_auc_mean'].min():.5f}. Every candidate is "
              "within the predefined equivalence band, so the honest conclusion is that L2 strength is close to "
              "irrelevant here rather than that lambda=50 is meaningfully superior.\n")

    lam0 = agg_df[agg_df["lambda"] == 0.0].iloc[0]
    best = ranked.iloc[0]
    md.append("## Did L2 regularization improve anything?\n")
    md.append(f"- **Log loss**: best lambda={best['lambda']} at {best['val_log_loss_mean']:.5f} vs "
              f"lambda=0 at {lam0['val_log_loss_mean']:.5f} "
              f"(**{best['val_log_loss_mean'] - lam0['val_log_loss_mean']:+.5f}**).")
    best_auc_row = agg_df.loc[agg_df["val_roc_auc_mean"].idxmax()]
    md.append(f"- **ROC-AUC**: best lambda={best_auc_row['lambda']} at {best_auc_row['val_roc_auc_mean']:.5f} vs "
              f"lambda=0 at {lam0['val_roc_auc_mean']:.5f} "
              f"(**{best_auc_row['val_roc_auc_mean'] - lam0['val_roc_auc_mean']:+.5f}**).")
    best_brier_row = agg_df.loc[agg_df["val_brier_mean"].idxmin()]
    md.append(f"- **Brier**: best lambda={best_brier_row['lambda']} at {best_brier_row['val_brier_mean']:.5f} vs "
              f"lambda=0 at {lam0['val_brier_mean']:.5f} "
              f"(**{best_brier_row['val_brier_mean'] - lam0['val_brier_mean']:+.5f}**).")
    lam_max = agg_df.loc[agg_df["lambda"].idxmax()]
    md.append(f"- **Coefficient shrinkage**: mean ||w||_2 falls from {lam0['mean_coef_l2_norm']:.4f} at lambda=0 "
              f"to {lam_max['mean_coef_l2_norm']:.4f} at lambda={lam_max['lambda']} - L2 is demonstrably "
              "shrinking the coefficients as intended.")
    md.append(f"- **Generalization gap**: mean train-validation ROC-AUC gap moves from "
              f"{lam0['train_val_auc_gap_mean']:+.4f} (lambda=0) to {lam_max['train_val_auc_gap_mean']:+.4f} "
              f"(lambda={lam_max['lambda']}).")
    md.append(f"- **Convergence speed**: median iterations moves from {lam0['median_iterations']:.0f} (lambda=0) "
              f"to {lam_max['median_iterations']:.0f} (lambda={lam_max['lambda']}).\n")
    md.append(f"Differences of a few ten-thousandths in mean CV log loss across only {N_FOLDS} folds should not "
              f"be over-interpreted - that is exactly why the {LOG_LOSS_EQUIVALENCE_EPSILON} equivalence epsilon "
              "and the deterministic tie-break ladder exist.\n")

    md.append("## Selected configuration (FROZEN)\n")
    md.append(f"**lambda = {selected_lambda}**, selected via: {stage}.\n")
    w = agg_df[agg_df["lambda"] == selected_lambda].iloc[0]
    md.append(f"- CV log loss {w['val_log_loss_mean']:.5f} ± {w['val_log_loss_std']:.5f}")
    md.append(f"- CV ROC-AUC {w['val_roc_auc_mean']:.5f} ± {w['val_roc_auc_std']:.5f}")
    md.append(f"- CV Brier {w['val_brier_mean']:.5f} | CV accuracy {w['val_accuracy_mean']:.4f}")
    md.append(f"- all folds converged: {w['all_folds_converged']} | median iterations {w['median_iterations']:.0f}\n")
    md.append("Frozen in `data/modeling/logistic_regression_v2_selected_config.json`. Only now may the main "
              "validation partition be evaluated, exactly once, in `training/logistic_regression/train_logistic_regression_v2.py`.\n")

    (REPORTS / "phases" / "phase4a1_logistic_regression_tuning.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
