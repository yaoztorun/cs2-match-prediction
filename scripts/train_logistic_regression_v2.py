"""
Phase 4A.1: refit the ONE frozen Logistic Regression V2 configuration
(lambda selected via TRAIN-only chronological CV in
scripts/logistic_regression_tuning_v2.py) on the FULL augmented TRAIN
partition, and evaluate it EXACTLY ONCE on the 1,419-match main VALIDATION.

The mathematical core is still the FROZEN from-scratch lab-adapted
implementation (scripts/models/logistic_regression_scratch.py) - only the
convergence-aware loop wrapper is new. sklearn is used for metrics only.

No validation-based stopping of any kind: optimization stops on the
training-objective convergence criterion alone. TEST is never loaded.
Cologne/post-Cologne are structurally absent from series_features_v1.parquet.
LR V1 / RF V1 / RF V2 / XGB V1 / XGB V2 artifacts are only read, never written.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, brier_score_loss, confusion_matrix, roc_curve,
)

from _common import ROOT, REPORTS
from models.logistic_regression_scratch import (
    compute_cost_reg, compute_gradient_reg, predict_proba, predict,
)
from logistic_regression_convergence_v2 import (
    gradient_descent_until_convergence, CONVERGENCE_RULE_TEXT,
)
from preprocessing_logistic_v1 import (
    build_augmented_training_raw, fit_preprocessing, transform, save_preprocessing,
    assert_augmented_symmetry, mirror_raw_rows, PREPROCESSING_VERSION,
)

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = REPORTS / "figures"
TABLES_DIR = REPORTS / "tables"
SELECTED_CONFIG_PATH = MODELING_DIR / "logistic_regression_v2_selected_config.json"

LR1_META = MODELS_DIR / "logistic_regression_scratch_v1.json"
RF1_META = MODELS_DIR / "random_forest_v1.json"
RF2_META = MODELS_DIR / "random_forest_v2.json"
XGB1_META = MODELS_DIR / "xgboost_v1_metadata.json"
XGB2_META = MODELS_DIR / "xgboost_v2_metadata.json"
LR1_COEFS = TABLES_DIR / "logistic_regression_coefficients_v1.csv"

EXPECTED_TRAIN_N, EXPECTED_VAL_N, EXPECTED_TEST_N = 6619, 1419, 1418
EXPECTED_AUGMENTED_N = 13238
THRESHOLD = 0.5


def compute_metrics(y_true, y_proba, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "majority_class_accuracy": float(max(np.mean(y_true), 1 - np.mean(y_true))),
    }


def load_verified(path, label):
    required = ["accuracy", "roc_auc", "f1", "log_loss", "brier_score", "n", "majority_class_accuracy"]
    if not path.exists():
        raise RuntimeError(f"Cannot build comparison: {path} missing ({label}).")
    meta = json.loads(path.read_text(encoding="utf-8"))
    for k in ["train_metrics", "validation_metrics"]:
        if k not in meta:
            raise RuntimeError(f"Cannot build comparison: '{k}' missing from {path} ({label}).")
        miss = [m for m in required if m not in meta[k]]
        if miss:
            raise RuntimeError(f"Cannot build comparison: {k} in {path} ({label}) missing {miss}.")
    return meta


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features, target_col = cfg["model_features"], cfg["target"]

    if not SELECTED_CONFIG_PATH.exists():
        raise RuntimeError(f"{SELECTED_CONFIG_PATH} missing - run scripts/logistic_regression_tuning_v2.py first.")
    selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    lam = float(selected["selected_lambda"])
    alpha = float(selected["alpha"])
    max_iters = int(selected["max_iterations"])
    print(f"Frozen config: lambda={lam}, alpha={alpha}, max_iterations={max_iters}")
    print(f"  selection stage: {selected['selection_stage']}")
    print(f"  CV log loss {selected['cv_mean_log_loss']:.5f} +/- {selected['cv_std_log_loss']:.5f} | "
          f"CV ROC-AUC {selected['cv_mean_roc_auc']:.5f} +/- {selected['cv_std_roc_auc']:.5f}")

    lr1 = load_verified(LR1_META, "LR V1")
    rf1 = load_verified(RF1_META, "RF V1")
    rf2 = load_verified(RF2_META, "RF V2")
    xgb1 = load_verified(XGB1_META, "XGB V1")
    xgb2 = load_verified(XGB2_META, "XGB V2")
    print("Verified LR V1 / RF V1 / RF V2 / XGB V1 / XGB V2 metrics exist and are complete.")

    df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    split = pd.read_csv(SPLIT_PATH)
    df = df.merge(split[["match_id", "split"]], on="match_id", how="inner")

    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    test_count = int((df["split"] == "test").sum())
    # TEST IS NEVER LOADED AS A DATAFRAME HERE.

    assert len(train_raw) == EXPECTED_TRAIN_N
    assert len(val_raw) == EXPECTED_VAL_N
    assert test_count == EXPECTED_TEST_N
    print(f"Reused split: train={len(train_raw)}, validation={len(val_raw)}, test={test_count} (sealed, not loaded)")

    augmented = build_augmented_training_raw(train_raw)
    assert len(augmented) == EXPECTED_AUGMENTED_N
    mirrored_target_mean = float(augmented[target_col].mean())
    assert abs(mirrored_target_mean - 0.5) < 1e-12
    assert_augmented_symmetry(augmented)
    print(f"train: {len(train_raw)} unique historical matches | augmented: {len(augmented)} training "
          f"OBSERVATIONS (never independent matches) | target mean {mirrored_target_mean}")

    params = fit_preprocessing(augmented, model_features)
    save_preprocessing(params, MODELING_DIR / "logistic_preprocessing_v2.json")

    X_fit, feature_names = transform(augmented, params)
    y_fit = augmented[target_col].to_numpy(dtype=float)
    X_tr, _ = transform(train_raw, params)          # UNMIRRORED -> train metrics
    y_tr = train_raw[target_col].to_numpy(dtype=float)
    X_val, _ = transform(val_raw, params)
    y_val = val_raw[target_col].to_numpy(dtype=float)

    n_features = X_fit.shape[1]
    assert n_features == 19, f"expected 19 transformed features, got {n_features}"

    print(f"\nTraining scratch Logistic Regression V2 (lambda={lam}, alpha={alpha}, max_iters={max_iters})...")
    w, b, J_history, info = gradient_descent_until_convergence(
        X_fit, y_fit, np.zeros(n_features), 0.0, compute_cost_reg, compute_gradient_reg,
        alpha=alpha, max_iters=max_iters, lambda_=lam,
        progress_label="full-train", progress_every=2000, verbose=True)
    print(f"  iterations={info['iterations_run']} converged={info['converged']} ({info['converged_by']}) "
          f"J0={info['initial_cost']:.6f} Jf={info['final_cost']:.6f} |grad|={info['final_gradient_norm']:.3e}")

    assert np.isfinite(w).all() and np.isfinite(b)
    assert all(np.isfinite(J_history))

    proba_tr = predict_proba(X_tr, w, b)
    pred_tr = predict(X_tr, w, b, threshold=THRESHOLD)
    train_metrics = compute_metrics(y_tr, proba_tr, pred_tr)

    proba_val = predict_proba(X_val, w, b)
    pred_val = predict(X_val, w, b, threshold=THRESHOLD)
    val_metrics = compute_metrics(y_val, proba_val, pred_val)

    assert np.isfinite(proba_val).all() and (proba_val >= 0).all() and (proba_val <= 1).all()

    print("TRAIN metrics:", json.dumps(train_metrics, indent=2))
    print("MAIN VALIDATION metrics (FIRST AND ONLY evaluation for LR V2):", json.dumps(val_metrics, indent=2))

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "validation", **val_metrics}]).to_csv(
        TABLES_DIR / "logistic_regression_metrics_v2.csv", index=False, encoding="utf-8")

    # ---- cost / convergence figure ----
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(range(len(J_history)), J_history, linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("regularized training objective J(w,b)" if lam > 0 else "training objective J(w,b)")
    ax.set_title(f"Logistic Regression V2 - training objective (lambda={lam})")
    note = (f"lambda={lam}, alpha={alpha}\niterations={info['iterations_run']}/{max_iters}\n"
            f"converged={info['converged']} ({info['converged_by']})\n"
            f"J0={info['initial_cost']:.6f} -> Jf={info['final_cost']:.6f}")
    ax.annotate(note, xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logistic_regression_cost_v2.png", dpi=150)
    plt.close(fig)

    # ---- ROC ----
    fpr, tpr, _ = roc_curve(y_val, proba_val)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"Logistic Regression V2 (AUC={val_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC curve - Logistic Regression V2"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "logistic_regression_roc_v2.png", dpi=150); plt.close(fig)

    # ---- calibration ----
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(proba_val, bins) - 1, 0, 9)
    cal_rows = []
    for i in range(10):
        m = bin_idx == i
        if m.sum() == 0:
            continue
        cal_rows.append({"bin": f"[{bins[i]:.1f},{bins[i+1]:.1f})", "n": int(m.sum()),
                          "mean_predicted": float(proba_val[m].mean()),
                          "empirical_win_rate": float(y_val[m].mean())})
    cal_df = pd.DataFrame(cal_rows)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(cal_df["mean_predicted"], cal_df["empirical_win_rate"], s=np.clip(cal_df["n"], 10, None), alpha=0.8)
    ax.set_xlabel("mean predicted P(team1 wins)"); ax.set_ylabel("empirical team1 win rate")
    ax.set_title("Validation calibration diagnostic (10 bins) - Logistic Regression V2")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "logistic_regression_calibration_v2.png", dpi=150); plt.close(fig)

    # ---- probability distribution ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(proba_val, bins=30, range=(0, 1))
    ax.set_xlabel("P(team1 wins)"); ax.set_ylabel("count")
    ax.set_title("Validation predicted-probability distribution - Logistic Regression V2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logistic_regression_probability_distribution_v2.png", dpi=150); plt.close(fig)

    pct = [0, 5, 25, 50, 75, 95, 100]
    proba_percentiles = {p: float(np.percentile(proba_val, p)) for p in pct}

    # ---- coefficients + V1/V2 comparison ----
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": w, "abs_coefficient": np.abs(w)})
    coef_df.to_csv(TABLES_DIR / "logistic_regression_coefficients_v2.csv", index=False, encoding="utf-8")
    coef_sorted = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    v1c = pd.read_csv(LR1_COEFS).set_index("feature")["coefficient"]
    comp_rows = []
    REL_CHANGE_MIN_V1 = 1e-6  # below this, a ratio is meaningless (V1's ~1e-18 coefficients)
    for i, f in enumerate(feature_names):
        v1v = float(v1c.get(f, np.nan))
        v2v = float(w[i])
        rel = ((v2v - v1v) / abs(v1v)) if (np.isfinite(v1v) and abs(v1v) >= REL_CHANGE_MIN_V1) else np.nan
        comp_rows.append({"feature": f, "v1_coefficient": v1v, "v2_coefficient": v2v,
                           "absolute_change": v2v - v1v, "relative_change": rel})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(TABLES_DIR / "logistic_regression_coefficient_comparison_v1_v2.csv",
                    index=False, encoding="utf-8")

    # ---- side symmetry ----
    val_mirrored = mirror_raw_rows(val_raw)
    X_val_m, _ = transform(val_mirrored, params)
    proba_val_m = predict_proba(X_val_m, w, b)
    sym_err = np.abs(proba_val - (1 - proba_val_m))
    sym = {"mean": float(sym_err.mean()), "median": float(np.median(sym_err)),
           "p95": float(np.percentile(sym_err, 95)), "max": float(sym_err.max())}
    print("Side-symmetry (validation):", sym)

    # ---- artifacts ----
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    np.savez(MODELS_DIR / "logistic_regression_scratch_v2.npz",
             w=w, b=np.array([b]), feature_names=np.array(feature_names), J_history=np.array(J_history))

    model_meta = {
        "model_type": "logistic_regression_from_scratch",
        "implementation_source": "adapted from course Lab 2 Logistic Regression (frozen core, unmodified)",
        "version": "v2",
        "selected_from_cv": True,
        "lambda_": lam,
        "alpha": alpha,
        "max_iterations": max_iters,
        "decision_threshold": THRESHOLD,
        "convergence_rule": CONVERGENCE_RULE_TEXT,
        "iterations_run": info["iterations_run"],
        "converged": info["converged"],
        "converged_by": info["converged_by"],
        "initial_cost": info["initial_cost"],
        "final_cost": info["final_cost"],
        "relative_cost_decrease": info["relative_cost_decrease"],
        "final_gradient_norm": info["final_gradient_norm"],
        "validation_based_stopping_used": False,
        "main_validation_used_in_selection": False,
        "cv_selected_config_artifact": "data/modeling/logistic_regression_v2_selected_config.json",
        "cv_tuning_artifact": "reports/tables/logistic_regression_tuning_v2.csv",
        "cv_folds_manifest": "data/modeling/random_forest_cv_folds_v2.csv",
        "cv_mean_log_loss": selected["cv_mean_log_loss"],
        "cv_std_log_loss": selected["cv_std_log_loss"],
        "cv_mean_roc_auc": selected["cv_mean_roc_auc"],
        "cv_std_roc_auc": selected["cv_std_roc_auc"],
        "bias_b": float(b),
        "coef_l2_norm": float(np.linalg.norm(w)),
        "training_cutoff_datetime": str(train_raw["datetime"].max()),
        "unique_training_matches": int(len(train_raw)),
        "augmented_training_observations": int(len(augmented)),
        "mirrored_train_target_mean": mirrored_target_mean,
        "feature_count": int(n_features),
        "feature_config_path": "config/series_features_v1.yaml",
        "preprocessing_artifact_path": "data/modeling/logistic_preprocessing_v2.json",
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_manifest_path": "data/modeling/series_split_v1.csv",
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_probability_percentiles": proba_percentiles,
        "validation_symmetry_error": sym,
        "test_status": "SEALED - not evaluated in Phase 4A.1",
        "cologne_status": "UNTOUCHED - not present in series_features_v1.parquet",
    }
    (MODELS_DIR / "logistic_regression_scratch_v2.json").write_text(
        json.dumps(model_meta, indent=2), encoding="utf-8")

    write_symmetry(sym_err, sym)
    write_comparison(lr1, rf1, rf2, xgb1, xgb2, train_metrics, val_metrics)
    write_results(selected, train_raw, val_raw, augmented, feature_names, info,
                   train_metrics, val_metrics, coef_sorted, comp_df, cal_df,
                   proba_percentiles, sym, model_meta, lr1, rf2, xgb2, lam)

    print("\nWrote models/logistic_regression_scratch_v2.{npz,json}")
    print("Wrote data/modeling/logistic_preprocessing_v2.json")
    print("Wrote reports/tables/logistic_regression_{metrics,coefficients}_v2.csv")
    print("Wrote reports/tables/logistic_regression_coefficient_comparison_v1_v2.csv")
    print("Wrote reports/figures/logistic_regression_{cost,roc,calibration,probability_distribution}_v2.png")
    print("Wrote reports/logistic_regression_symmetry_v2.md")
    print("Wrote reports/model_comparison_tuned_v1features.md")
    print("Wrote reports/phase4a1_logistic_regression_v2_results.md")


def write_symmetry(sym_err, sym):
    md = []
    md.append("# Logistic Regression V2 - Side-Symmetry Diagnostic\n")
    md.append("Same diagnostic as RF and XGBoost: for every validation matchup A-vs-B, the raw mirrored B-vs-A "
              "form is built with the same `mirror_raw_rows`, transformed with the SAME fitted V2 preprocessing "
              "artifact, and scored with the same model.\n")
    md.append("Logistic Regression's linear form combined with mirrored training makes near-exact antisymmetry "
              "theoretically plausible, but this is **measured rather than assumed** - residual deviation can "
              "still arise from preprocessing (median imputation of a mirrored NaN pair shares one value) and "
              "floating-point effects.\n")
    md.append(f"- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n={len(sym_err)} validation rows")
    md.append(f"- mean: {sym['mean']:.3e}")
    md.append(f"- median: {sym['median']:.3e}")
    md.append(f"- 95th percentile: {sym['p95']:.3e}")
    md.append(f"- max: {sym['max']:.3e}\n")
    md.append("**No external symmetry correction is applied.**\n")
    (REPORTS / "logistic_regression_symmetry_v2.md").write_text("\n".join(md), encoding="utf-8")


def write_comparison(lr1, rf1, rf2, xgb1, xgb2, lr2_train, lr2_val):
    l1v, l1t = lr1["validation_metrics"], lr1["train_metrics"]
    r1v, r1t = rf1["validation_metrics"], rf1["train_metrics"]
    r2v, r2t = rf2["validation_metrics"], rf2["train_metrics"]
    x1v, x1t = xgb1["validation_metrics"], xgb1["train_metrics"]
    x2v, x2t = xgb2["validation_metrics"], xgb2["train_metrics"]
    maj = l1v["majority_class_accuracy"]

    md = []
    md.append("# Model Comparison - Phase-3 V1 Features (Validation Only)\n")
    md.append("All six models use the identical chronological split, the identical 17 Phase-3 features, and the "
              "identical train-only mirrored augmentation. All three tuned models used the **same** TRAIN-only "
              "expanding-window CV folds. Every number for a model other than LR V2 is read from that model's "
              "saved metadata JSON (verified complete before use); none were re-run or modified.\n")
    md.append("**LR V1 = UNTUNED · LR V2 = TUNED · RF V1 = UNTUNED · RF V2 = TUNED · XGB V1 = UNTUNED · "
              "XGB V2 = TUNED**\n")

    md.append("## All six models (validation)\n")
    md.append("| metric | majority | LR V1 | LR V2 | RF V1 | RF V2 | XGB V1 | XGB V2 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        base = f"{maj:.4f}" if key == "accuracy" else ("0.5000" if key == "roc_auc" else "-")
        md.append(f"| {label} | {base} | {l1v[key]:.4f} | {lr2_val[key]:.4f} | {r1v[key]:.4f} | "
                  f"{r2v[key]:.4f} | {x1v[key]:.4f} | {x2v[key]:.4f} |")
    md.append("")

    md.append("## TUNED-ONLY comparison: LR V2 vs RF V2 vs XGB V2\n")
    md.append("This is the project's **first approximately fair algorithm comparison**: same 17 features, same "
              "chronological history, same main validation partition, and the same TRAIN-only temporal CV folds "
              "used to tune each one.\n")
    md.append("| metric | LR V2 (tuned) | RF V2 (tuned) | XGB V2 (tuned) | best |")
    md.append("|---|---|---|---|---|")
    for key, label, lower_better in [("accuracy", "Accuracy", False), ("roc_auc", "ROC-AUC", False),
                                      ("f1", "F1", False), ("log_loss", "Log loss", True),
                                      ("brier_score", "Brier", True)]:
        vals = {"LR V2": lr2_val[key], "RF V2": r2v[key], "XGB V2": x2v[key]}
        best = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        md.append(f"| {label} | {lr2_val[key]:.4f} | {r2v[key]:.4f} | {x2v[key]:.4f} | **{best}** |")
    md.append("")

    md.append("## Train -> validation ROC-AUC gaps\n")
    md.append("| model | train AUC | val AUC | gap |")
    md.append("|---|---|---|---|")
    for label, t, v in [("LR V1 (untuned)", l1t, l1v), ("LR V2 (tuned)", lr2_train, lr2_val),
                         ("RF V1 (untuned)", r1t, r1v), ("RF V2 (tuned)", r2t, r2v),
                         ("XGB V1 (untuned)", x1t, x1v), ("XGB V2 (tuned)", x2t, x2v)]:
        md.append(f"| {label} | {t['roc_auc']:.4f} | {v['roc_auc']:.4f} | {t['roc_auc']-v['roc_auc']:+.4f} |")
    md.append("")

    md.append("## LR V2 deltas\n")
    md.append("| metric | LR V2 - LR V1 | LR V2 - RF V2 | LR V2 - XGB V2 |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {lr2_val[key]-l1v[key]:+.4f} | {lr2_val[key]-r2v[key]:+.4f} | "
                  f"{lr2_val[key]-x2v[key]:+.4f} |")
    md.append("\nFor log loss and Brier, **more negative is better**; for accuracy/ROC-AUC/F1, more positive is "
              "better.\n")

    md.append("## No final project model is declared\n")
    md.append("- Richer features (map-level, player-level) have not been evaluated.")
    md.append("- The internal test partition remains **sealed**.")
    md.append("- Cologne 2026 remains an **external, untouched** holdout.")
    md.append("- These are single-validation-period results on 1,419 matches; differences of a few thousandths "
              "should not be treated as decisive.\n")
    (REPORTS / "model_comparison_tuned_v1features.md").write_text("\n".join(md), encoding="utf-8")


def write_results(selected, train_raw, val_raw, augmented, feature_names, info,
                   train_metrics, val_metrics, coef_sorted, comp_df, cal_df,
                   proba_percentiles, sym, model_meta, lr1, rf2, xgb2, lam):
    l1v, l1t = lr1["validation_metrics"], lr1["train_metrics"]
    r2v = rf2["validation_metrics"]
    x2v = xgb2["validation_metrics"]
    v1_gap = l1t["roc_auc"] - l1v["roc_auc"]
    v2_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]

    md = []
    md.append("# Phase 4A.1 - Logistic Regression V2 Results (Chronologically Tuned)\n")
    md.append("**LR V1 = UNTUNED · LR V2 = TUNED · RF V1 = UNTUNED · RF V2 = TUNED · XGB V1 = UNTUNED · "
              "XGB V2 = TUNED**\n")

    md.append("## 1. Why LR tuning was performed\n")
    md.append("RF and XGBoost had both received chronological tuning while Logistic Regression had not, so no "
              "fair algorithm comparison existed. LR V1's cost curve was also still descending at its fixed "
              "10,000 iterations, so it was both untuned *and* under-converged.\n")

    md.append("## 2. Why lambda is the predictive/model hyperparameter\n")
    md.append("L2 strength changes the hypothesis the model settles on - it trades training fit against "
              "coefficient magnitude and therefore changes predictions. It is the only thing searched here.\n")

    md.append("## 3. Why alpha and iterations are optimization controls\n")
    md.append(f"`alpha={selected['alpha']}` and `max_iterations={selected['max_iterations']}` were fixed for "
              "every candidate and were never selected by comparing validation predictions - they exist purely "
              "so gradient descent converges reliably. Because all lambda candidates share them, the lambda "
              "comparison is clean.\n")
    md.append("**Disclosure of a real confound**: LR V1 used `alpha=0.001` with a fixed 10,000 iterations and "
              "was still descending; LR V2 uses `alpha=0.01` with a convergence criterion. So the LR V1 -> LR V2 "
              "comparison mixes *regularization* with *better convergence*. The `lambda=0` row of the tuning "
              "table is the clean reference for isolating the pure regularization effect.\n")

    md.append("## 4. Temporal CV methodology\n")
    md.append("The **same** four expanding-window folds used to tune RF V2 and XGB V2 "
              "(`data/modeling/random_forest_cv_folds_v2.csv`, byte-identical), all inside the global TRAIN "
              "partition, chronology re-verified at runtime. The main validation partition was never loaded by "
              "the tuning script.\n")

    md.append("## 5. Mirroring\n")
    md.append(f"Per fold: mirror fold-train only; fold-validation never mirrored; fold TRAIN metrics computed on "
              f"the original unmirrored fold-train rows. For the final refit: {len(train_raw):,} unique "
              f"historical training matches -> **{len(augmented):,} augmented training observations** (never "
              f"described as independent matches), target mean exactly "
              f"{model_meta['mirrored_train_target_mean']}.\n")

    md.append("## 6. Preprocessing\n")
    md.append(f"Identical LR V1 semantics (standardization + train-median imputation + deterministic bestOf/tier "
              f"dummies, {len(feature_names)} transformed features), refit independently inside every fold and "
              "again on the augmented full TRAIN for the final model. No validation statistics anywhere.\n")

    md.append("## 7. Convergence methodology\n")
    md.append(selected["convergence_rule"] + "\n")
    md.append(f"Final refit: **{info['iterations_run']} iterations**, converged=**{info['converged']}** via "
              f"`{info['converged_by']}`, J {info['initial_cost']:.6f} -> {info['final_cost']:.6f}, final "
              f"gradient norm {info['final_gradient_norm']:.3e}. See "
              "`reports/figures/logistic_regression_cost_v2.png`.\n")
    if lam > 0:
        md.append(f"**Objective comparability caveat**: with lambda={lam} the plotted objective *includes* the "
                  f"L2 penalty `(lambda/(2m))*sum(w^2)`, so its numeric value is **not** directly comparable to "
                  "LR V1's unregularized objective. Only the unregularized log loss reported in the metrics "
                  "tables is comparable across the two.\n")

    md.append("## 8. Lambda candidate performance\n")
    md.append("Full table in `reports/phase4a1_logistic_regression_tuning.md`. Two honest caveats carried "
              "forward from that report:\n")
    md.append("- **Every** lambda candidate fell inside the predefined 0.002 log-loss equivalence band, so L2 "
              "strength is close to *irrelevant* on this problem rather than lambda=50 being meaningfully "
              "superior.")
    md.append("- The convergence gate was degenerate (only one candidate converged in all four folds), but a "
              "sensitivity check re-applying the ladder **without** the gate selects the same lambda, so the "
              "outcome is not an artifact of the gate.\n")

    md.append("## 9. Selected lambda\n")
    md.append(f"**lambda = {lam}**, via {selected['selection_stage']}. CV log loss "
              f"{selected['cv_mean_log_loss']:.5f} ± {selected['cv_std_log_loss']:.5f}, CV ROC-AUC "
              f"{selected['cv_mean_roc_auc']:.5f} ± {selected['cv_std_roc_auc']:.5f}.\n")

    def mlines(m):
        return "\n".join([
            f"- n = {m['n']:,}", f"- Accuracy: {m['accuracy']:.4f}", f"- Precision: {m['precision']:.4f}",
            f"- Recall: {m['recall']:.4f}", f"- F1: {m['f1']:.4f}", f"- ROC-AUC: {m['roc_auc']:.4f}",
            f"- Log loss: {m['log_loss']:.4f}", f"- Brier: {m['brier_score']:.4f}",
            f"- Confusion matrix [[TN,FP],[FN,TP]]: {m['confusion_matrix']}",
        ])

    md.append("## 10. Train metrics (original unmirrored)\n")
    md.append(mlines(train_metrics) + "\n")
    md.append("## 11. Main validation metrics (one-shot)\n")
    md.append(mlines(val_metrics) + "\n")

    md.append("## 12. LR V1 vs LR V2\n")
    md.append("| metric | LR V1 (untuned) | LR V2 (tuned) | delta |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {l1v[key]:.4f} | {val_metrics[key]:.4f} | {val_metrics[key]-l1v[key]:+.4f} |")
    md.append(f"\nTrain->validation ROC-AUC gap: LR V1 {v1_gap:+.4f} -> LR V2 {v2_gap:+.4f}. Remember this "
              "comparison confounds regularization with the improved convergence settings (Section 3).\n")

    md.append("## 13. LR V2 vs RF V2\n")
    md.append("| metric | LR V2 | RF V2 | delta (LR V2 - RF V2) |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {val_metrics[key]:.4f} | {r2v[key]:.4f} | {val_metrics[key]-r2v[key]:+.4f} |")
    md.append("")

    md.append("## 14. LR V2 vs XGB V2\n")
    md.append("| metric | LR V2 | XGB V2 | delta (LR V2 - XGB V2) |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {val_metrics[key]:.4f} | {x2v[key]:.4f} | {val_metrics[key]-x2v[key]:+.4f} |")
    md.append("\nFull three-way tuned comparison in `reports/model_comparison_tuned_v1features.md`.\n")

    md.append("## 15. Calibration\n")
    md.append("`reports/figures/logistic_regression_calibration_v2.png` (validation only, no correction "
              "applied):\n")
    md.append("| bin | n | mean predicted | empirical win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['empirical_win_rate']:.3f} |")
    p = proba_percentiles
    md.append(f"\nProbability spread (validation): min {p[0]:.3f} | p5 {p[5]:.3f} | p25 {p[25]:.3f} | "
              f"median {p[50]:.3f} | p75 {p[75]:.3f} | p95 {p[95]:.3f} | max {p[100]:.3f}.\n")

    md.append("## 16. Coefficient shrinkage\n")
    md.append(f"Bias `b = {model_meta['bias_b']:.6f}` (never regularized, matching the lab). "
              f"Coefficient L2 norm `||w||_2 = {model_meta['coef_l2_norm']:.4f}`. Top 8 by magnitude:\n")
    md.append("| feature | coefficient | abs_coefficient |")
    md.append("|---|---|---|")
    for _, r in coef_sorted.head(8).iterrows():
        md.append(f"| {r['feature']} | {r['coefficient']:.4f} | {r['abs_coefficient']:.4f} |")
    md.append("\nV1 vs V2 comparison (`reports/tables/logistic_regression_coefficient_comparison_v1_v2.csv`), "
              "largest absolute changes:\n")
    top_ch = comp_df.reindex(comp_df["absolute_change"].abs().sort_values(ascending=False).index).head(8)
    md.append("| feature | v1_coefficient | v2_coefficient | absolute_change | relative_change |")
    md.append("|---|---|---|---|---|")
    for _, r in top_ch.iterrows():
        rel = "n/a" if not np.isfinite(r["relative_change"]) else f"{r['relative_change']:+.2%}"
        md.append(f"| {r['feature']} | {r['v1_coefficient']:.4g} | {r['v2_coefficient']:.4g} | "
                  f"{r['absolute_change']:+.4f} | {rel} |")
    md.append("\n`relative_change` is deliberately `n/a` where V1's coefficient was numerically ~0 (the "
              "symmetric/context features, ~1e-18 in V1) - a ratio against ~0 is meaningless, not informative. "
              "Note that V2's coefficients are generally **larger** in magnitude than V1's despite L2 shrinkage: "
              "V1 was under-converged (still descending at 10,000 iterations at alpha=0.001), so its "
              "coefficients had not yet grown to their fitted values. Shrinkage is visible *within* the tuning "
              "sweep (mean ||w||_2 falls monotonically as lambda rises), which is the clean comparison. "
              "**Coefficients are not interpreted causally.**\n")

    md.append("## 17. Symmetry\n")
    md.append(f"mean **{sym['mean']:.3e}**, median **{sym['median']:.3e}**, p95 **{sym['p95']:.3e}**, max "
              f"**{sym['max']:.3e}** - measured, not assumed. Detail in "
              "`reports/logistic_regression_symmetry_v2.md`. Not corrected.\n")

    md.append("## 18. Limitations\n")
    md.append(f"- The selected lambda ({lam}) sits at the **edge of the predefined grid**, so the optimum may "
              "lie beyond it; the grid is deliberately not extended, since re-searching after seeing results is "
              "what this protocol forbids.")
    md.append("- Every candidate fell within the 0.002 equivalence band - L2 strength barely matters here, so "
              "lambda=50 should not be read as meaningfully better than lambda=0.")
    md.append("- The convergence gate admitted only one candidate; the sensitivity check shows the outcome is "
              "unchanged without it, but the gate as configured did little useful work, and criterion (B) "
              "(gradient norm) never fired at all.")
    md.append("- LR V1 -> LR V2 confounds regularization with convergence quality (Section 3).")
    md.append("- Same 17 series-level features; no map-level or player-level information.")
    md.append("- Single chronological validation period of 1,419 matches; no probability calibration, no "
              "threshold tuning.")
    md.append("- The main validation partition has now been used once for LR V2; repeated future use would "
              "erode its independence.\n")

    md.append("## Status\n")
    md.append("- **TEST = SEALED** - not opened or scored.")
    md.append("- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.")
    md.append("- Main validation evaluated exactly once, after the configuration was frozen.")
    md.append("- No tuning iteration followed the main-validation result.")
    md.append("- No final project model declared.\n")

    (REPORTS / "phase4a1_logistic_regression_v2_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
