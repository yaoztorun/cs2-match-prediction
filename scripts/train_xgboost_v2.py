"""
Phase 4C.1: refit the ONE frozen XGBoost V2 configuration (selected via
TRAIN-only three-stage chronological CV in scripts/xgboost_tuning_v2.py) on
the FULL augmented TRAIN partition, and evaluate it EXACTLY ONCE on the
1,419-match main VALIDATION partition.

No early stopping and no eval_set are used in this final refit: there is no
TRAIN-only holdout left for that purpose, and the main validation partition
must never serve that role. `n_estimators` comes from the frozen
CV-derived `final_n_estimators`.

TEST is never loaded. Cologne/post-Cologne are structurally absent from
series_features_v1.parquet. LR / RF V1 / RF V2 / XGB V1 artifacts are only
read here, never written.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, brier_score_loss, confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier

from _common import ROOT, REPORTS
from preprocessing_xgboost_v1 import (
    build_augmented_training_raw, fit_preprocessing, transform, save_preprocessing,
    assert_augmented_symmetry, mirror_raw_rows, PREPROCESSING_VERSION, MISSING_VALUE_POLICY,
)

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = REPORTS / "figures"
TABLES_DIR = REPORTS / "tables"
SELECTED_CONFIG_PATH = MODELING_DIR / "xgboost_v2_selected_config.json"

LR_META = MODELS_DIR / "logistic_regression_scratch_v1.json"
RF1_META = MODELS_DIR / "random_forest_v1.json"
RF2_META = MODELS_DIR / "random_forest_v2.json"
XGB1_META = MODELS_DIR / "xgboost_v1_metadata.json"
XGB1_IMPORTANCE = TABLES_DIR / "xgboost_feature_importance_v1.csv"
XGB1_PERM = TABLES_DIR / "xgboost_permutation_importance_v1.csv"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418
EXPECTED_AUGMENTED_N = 13238


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
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    if not SELECTED_CONFIG_PATH.exists():
        raise RuntimeError(f"{SELECTED_CONFIG_PATH} missing - run scripts/xgboost_tuning_v2.py first.")
    selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    hp = selected["params"]
    fixed = selected["fixed_params"]
    final_n_estimators = int(selected["final_n_estimators"])
    print(f"Frozen config: {selected['selected_candidate_id']} | {hp}")
    print(f"  selection stage: {selected['selection_stage']}")
    print(f"  best_iterations by fold: {selected['best_iterations_by_fold']} -> "
          f"final_n_estimators = {final_n_estimators}")

    lr_meta = load_verified(LR_META, "LR V1")
    rf1_meta = load_verified(RF1_META, "RF V1")
    rf2_meta = load_verified(RF2_META, "RF V2")
    xgb1_meta = load_verified(XGB1_META, "XGB V1")
    print("Verified LR V1 / RF V1 / RF V2 / XGB V1 metrics exist and are complete.")

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

    augmented_train_raw = build_augmented_training_raw(train_raw)
    assert len(augmented_train_raw) == EXPECTED_AUGMENTED_N
    mirrored_target_mean = float(augmented_train_raw[target_col].mean())
    assert abs(mirrored_target_mean - 0.5) < 1e-12
    assert_augmented_symmetry(augmented_train_raw)
    print(f"train: {len(train_raw)} unique historical matches | augmented: {len(augmented_train_raw)} training "
          f"OBSERVATIONS (not independent matches) | target mean {mirrored_target_mean}")

    params = fit_preprocessing(augmented_train_raw, model_features)
    save_preprocessing(params, MODELING_DIR / "xgboost_preprocessing_v2.json")

    X_train_aug, feature_names = transform(augmented_train_raw, params)
    y_train_aug = augmented_train_raw[target_col].to_numpy(dtype=float)
    X_train_orig, _ = transform(train_raw, params)
    y_train_orig = train_raw[target_col].to_numpy(dtype=float)
    X_val, _ = transform(val_raw, params)
    y_val = val_raw[target_col].to_numpy(dtype=float)

    n_features = X_train_aug.shape[1]
    assert n_features == 19
    n_nan_train = int(np.isnan(X_train_aug).sum())
    n_nan_val = int(np.isnan(X_val).sum())

    # ---- single refit: fixed n_estimators, NO early stopping, NO eval_set ----
    model = XGBClassifier(n_estimators=final_n_estimators, **hp, **fixed)
    model.fit(X_train_aug, y_train_aug)
    trees_used = int(model.get_booster().num_boosted_rounds())
    print(f"refit with n_estimators={final_n_estimators} (no early stopping) -> trees built: {trees_used}")
    assert trees_used == final_n_estimators

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    model.save_model(str(MODELS_DIR / "xgboost_v2.json"))

    proba_train = model.predict_proba(X_train_orig)[:, 1]
    pred_train = model.predict(X_train_orig)
    train_metrics = compute_metrics(y_train_orig, proba_train, pred_train)

    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)
    val_metrics = compute_metrics(y_val, proba_val, pred_val)

    assert np.isfinite(proba_val).all() and (proba_val >= 0).all() and (proba_val <= 1).all()

    print("TRAIN metrics:", json.dumps(train_metrics, indent=2))
    print("MAIN VALIDATION metrics (FIRST AND ONLY evaluation for XGB V2):", json.dumps(val_metrics, indent=2))

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "validation", **val_metrics}]).to_csv(
        TABLES_DIR / "xgboost_metrics_v2.csv", index=False, encoding="utf-8")

    # ---- ROC ----
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fpr, tpr, _ = roc_curve(y_val, proba_val)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"XGBoost V2 (AUC={val_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC curve - XGBoost V2"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "xgboost_v2_roc.png", dpi=150); plt.close(fig)

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
    ax.set_title("Validation calibration diagnostic (10 bins) - XGBoost V2")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "xgboost_v2_calibration.png", dpi=150); plt.close(fig)

    # ---- probability distribution ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(proba_val, bins=30, range=(0, 1))
    ax.set_xlabel("P(team1 wins)"); ax.set_ylabel("count")
    ax.set_title("Validation predicted-probability distribution - XGBoost V2")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "xgboost_v2_probability_distribution.png", dpi=150); plt.close(fig)

    pct = [0, 5, 25, 50, 75, 95, 100]
    proba_percentiles = {p: float(np.percentile(proba_val, p)) for p in pct}

    # ---- importance ----
    booster = model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    gain_by = {n: float(gain_raw.get(f"f{i}", 0.0)) for i, n in enumerate(feature_names)}
    weight_by = {n: float(weight_raw.get(f"f{i}", 0.0)) for i, n in enumerate(feature_names)}
    total_gain = sum(gain_by.values())
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "raw_gain": [gain_by[n] for n in feature_names],
        "normalized_gain": [(gain_by[n] / total_gain) if total_gain > 0 else 0.0 for n in feature_names],
        "weight": [weight_by[n] for n in feature_names],
    }).sort_values("raw_gain", ascending=False).reset_index(drop=True)
    importance_df["rank_gain"] = importance_df.index + 1
    importance_df.to_csv(TABLES_DIR / "xgboost_feature_importance_v2.csv", index=False, encoding="utf-8")

    perm = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": feature_names, "mean_importance": perm.importances_mean,
                             "std_importance": perm.importances_std}) \
        .sort_values("mean_importance", ascending=False).reset_index(drop=True)
    perm_df["rank"] = perm_df.index + 1
    perm_df.to_csv(TABLES_DIR / "xgboost_permutation_importance_v2.csv", index=False, encoding="utf-8")

    # ---- symmetry ----
    val_mirrored = mirror_raw_rows(val_raw)
    X_val_m, _ = transform(val_mirrored, params)
    proba_val_m = model.predict_proba(X_val_m)[:, 1]
    sym_err = np.abs(proba_val - (1 - proba_val_m))
    sym = {"mean": float(sym_err.mean()), "median": float(np.median(sym_err)),
           "p95": float(np.percentile(sym_err, 95)), "max": float(sym_err.max())}
    print("Side-symmetry (validation):", sym)

    model_meta = {
        "model_type": "xgboost_classifier",
        "implementation": "xgboost.XGBClassifier",
        "version": "v2",
        "xgboost_version": __import__("xgboost").__version__,
        "selected_from_cv": True,
        "cv_candidate_id": selected["selected_candidate_id"],
        "cv_selection_stage": selected["selection_stage"],
        "cv_log_loss_epsilon": selected["log_loss_epsilon"],
        "cv_mean_log_loss": selected["cv_mean_log_loss"],
        "cv_std_log_loss": selected["cv_std_log_loss"],
        "cv_mean_roc_auc": selected["cv_mean_roc_auc"],
        "cv_std_roc_auc": selected["cv_std_roc_auc"],
        "best_iterations_by_fold": selected["best_iterations_by_fold"],
        "median_best_iteration": selected["median_best_iteration"],
        "final_n_estimators": final_n_estimators,
        "trees_actually_built": trees_used,
        "cv_folds_artifact": selected["cv_folds_artifact"],
        "cv_search_artifact": selected["tuning_results_artifact"],
        "selected_config_artifact": "data/modeling/xgboost_v2_selected_config.json",
        **hp, **fixed,
        "n_estimators": final_n_estimators,
        "early_stopping_used_in_final_refit": False,
        "eval_set_used_in_final_refit": False,
        "main_validation_used_in_selection": False,
        "training_cutoff": str(train_raw["datetime"].max()),
        "unique_training_matches": int(len(train_raw)),
        "augmented_training_observations": int(len(augmented_train_raw)),
        "mirrored_train_target_mean": mirrored_target_mean,
        "feature_count": int(n_features),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "nan_count_augmented_train": n_nan_train,
        "nan_count_validation": n_nan_val,
        "feature_config": "config/series_features_v1.yaml",
        "preprocessing_artifact": "data/modeling/xgboost_preprocessing_v2.json",
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_manifest": "data/modeling/series_split_v1.csv",
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_probability_percentiles": proba_percentiles,
        "validation_symmetry_error": sym,
        "test_status": "SEALED - not evaluated in Phase 4C.1",
        "cologne_status": "UNTOUCHED - not present in series_features_v1.parquet",
    }
    (MODELS_DIR / "xgboost_v2_metadata.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    write_symmetry(sym_err, sym, xgb1_meta)
    write_comparison(lr_meta, rf1_meta, rf2_meta, xgb1_meta, train_metrics, val_metrics)
    write_results(selected, train_raw, val_raw, augmented_train_raw, feature_names,
                   train_metrics, val_metrics, importance_df, perm_df, cal_df,
                   proba_percentiles, sym, model_meta, lr_meta, rf1_meta, rf2_meta, xgb1_meta,
                   final_n_estimators, n_nan_train, n_nan_val)

    print("\nWrote models/xgboost_v2.json + models/xgboost_v2_metadata.json")
    print("Wrote data/modeling/xgboost_preprocessing_v2.json")
    print("Wrote reports/tables/xgboost_metrics_v2.csv + feature/permutation importance v2")
    print("Wrote reports/figures/xgboost_v2_{roc,calibration,probability_distribution}.png")
    print("Wrote reports/xgboost_symmetry_v2.md")
    print("Wrote reports/model_comparison_v2.md")
    print("Wrote reports/phase4c1_xgboost_v2_results.md")


def write_symmetry(sym_err, sym, xgb1_meta):
    v1 = xgb1_meta.get("validation_symmetry_error", {})
    md = []
    md.append("# XGBoost V2 - Side-Symmetry Diagnostic\n")
    md.append("Same diagnostic as XGBoost V1 and Random Forest: for every validation matchup A-vs-B, the raw "
              "mirrored B-vs-A form is built with the same `mirror_raw_rows`, transformed with the SAME fitted "
              "V2 preprocessing artifact, and scored with the same model.\n")
    md.append("**No assumption is made that tuning improves symmetry.** A gradient-boosted tree ensemble is not "
              "mathematically constrained to satisfy `P(A beats B) = 1 - P(B beats A)`; determinism does not "
              "imply antisymmetry. Measured, reported, and left uncorrected.\n")
    md.append(f"- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n={len(sym_err)} validation rows\n")
    md.append("| statistic | XGB V1 | XGB V2 | change |")
    md.append("|---|---|---|---|")
    for k, label in [("mean", "mean"), ("median", "median"), ("p95", "95th percentile"), ("max", "max")]:
        v1v = v1.get(k, float("nan"))
        md.append(f"| {label} | {v1v:.6f} | {sym[k]:.6f} | {sym[k]-v1v:+.6f} |")
    md.append("")
    better = sym["mean"] < v1.get("mean", float("inf"))
    md.append(f"**Observation**: V2's mean symmetry error is {'lower' if better else 'not lower'} than V1's. "
              "Reported as observed - no symmetrization applied in either version.\n")
    (REPORTS / "xgboost_symmetry_v2.md").write_text("\n".join(md), encoding="utf-8")


def write_comparison(lr_meta, rf1_meta, rf2_meta, xgb1_meta, xgb2_train, xgb2_val):
    lr_v, lr_t = lr_meta["validation_metrics"], lr_meta["train_metrics"]
    rf1_v, rf1_t = rf1_meta["validation_metrics"], rf1_meta["train_metrics"]
    rf2_v, rf2_t = rf2_meta["validation_metrics"], rf2_meta["train_metrics"]
    x1_v, x1_t = xgb1_meta["validation_metrics"], xgb1_meta["train_metrics"]
    maj = lr_v["majority_class_accuracy"]

    md = []
    md.append("# Model Comparison V2 (Validation Only)\n")
    md.append("All models use the identical chronological split, the identical 17 Phase-3 features, and the "
              "identical train-only mirrored augmentation policy. LR/RF/XGB-V1 numbers are read from their saved "
              "metadata JSONs (verified present and complete before use); none were re-run or modified.\n")
    md.append("**Tuning status (essential for fair reading):**\n")
    md.append("- **LR V1** = UNTUNED (from-scratch Logistic Regression baseline)")
    md.append("- **RF V1** = UNTUNED Random Forest baseline")
    md.append("- **RF V2** = TUNED (chronological TRAIN-only CV)")
    md.append("- **XGB V1** = UNTUNED fixed baseline")
    md.append("- **XGB V2** = TUNED (chronological TRAIN-only CV, three-stage temporal separation)\n")
    md.append("Logistic Regression has still never been tuned, so nothing below establishes algorithm "
              "superiority.\n")

    md.append("## Validation metrics\n")
    md.append("| metric | majority | LR V1 (untuned) | RF V1 (untuned) | RF V2 (tuned) | XGB V1 (untuned) | XGB V2 (tuned) |")
    md.append("|---|---|---|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        base = f"{maj:.4f}" if key == "accuracy" else ("0.5000" if key == "roc_auc" else "-")
        md.append(f"| {label} | {base} | {lr_v[key]:.4f} | {rf1_v[key]:.4f} | {rf2_v[key]:.4f} | "
                  f"{x1_v[key]:.4f} | {xgb2_val[key]:.4f} |")
    md.append("")

    md.append("## Train ROC-AUC, validation ROC-AUC, and the gap\n")
    md.append("| model | train AUC | val AUC | gap |")
    md.append("|---|---|---|---|")
    for label, t, v in [("LR V1 (untuned)", lr_t, lr_v), ("RF V1 (untuned)", rf1_t, rf1_v),
                         ("RF V2 (tuned)", rf2_t, rf2_v), ("XGB V1 (untuned)", x1_t, x1_v),
                         ("XGB V2 (tuned)", xgb2_train, xgb2_val)]:
        md.append(f"| {label} | {t['roc_auc']:.4f} | {v['roc_auc']:.4f} | {t['roc_auc']-v['roc_auc']:+.4f} |")
    md.append("")

    md.append("## XGB V2 deltas\n")
    md.append("| metric | XGB V2 - XGB V1 | XGB V2 - RF V2 | XGB V2 - LR V1 |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {xgb2_val[key]-x1_v[key]:+.4f} | {xgb2_val[key]-rf2_v[key]:+.4f} | "
                  f"{xgb2_val[key]-lr_v[key]:+.4f} |")
    md.append("\nFor log loss and Brier, **more negative is better**; for accuracy/ROC-AUC/F1, more positive is "
              "better.\n")

    d_lr = xgb2_val["roc_auc"] - lr_v["roc_auc"]
    if d_lr > 0.005:
        md.append("On validation ROC-AUC, **the tuned XGBoost (XGB V2) outperformed the untuned Logistic "
                  "Regression baseline (LR V1)**. This is not a claim that XGBoost is definitively better than "
                  "Logistic Regression - LR has never been tuned, so the comparison is not like-for-like.\n")
    elif d_lr < -0.005:
        md.append("On validation ROC-AUC, the tuned XGBoost (XGB V2) still did **not** outperform the untuned "
                  "Logistic Regression baseline (LR V1). Reported honestly as observed.\n")
    else:
        md.append("On validation ROC-AUC, the tuned XGBoost (XGB V2) and the untuned Logistic Regression "
                  "baseline (LR V1) are essentially tied.\n")

    md.append("**No final project model is declared.** Logistic Regression remains untuned and the internal test "
              "set has not been used for this or any comparison.\n")
    (REPORTS / "model_comparison_v2.md").write_text("\n".join(md), encoding="utf-8")


def write_results(selected, train_raw, val_raw, augmented_train_raw, feature_names,
                   train_metrics, val_metrics, importance_df, perm_df, cal_df,
                   proba_percentiles, sym, model_meta, lr_meta, rf1_meta, rf2_meta, xgb1_meta,
                   final_n_estimators, n_nan_train, n_nan_val):
    lr_v = lr_meta["validation_metrics"]
    rf2_v = rf2_meta["validation_metrics"]
    x1_v, x1_t = xgb1_meta["validation_metrics"], xgb1_meta["train_metrics"]
    x1_gap = x1_t["roc_auc"] - x1_v["roc_auc"]
    v2_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]
    hp = selected["params"]

    md = []
    md.append("# Phase 4C.1 - XGBoost V2 Results (Chronologically Tuned)\n")
    md.append("**LR V1 = UNTUNED. RF V1 = UNTUNED. RF V2 = TUNED. XGB V1 = UNTUNED. XGB V2 = TUNED.**\n")

    md.append("## 1. Why XGBoost tuning was performed\n")
    md.append(f"XGBoost V1 was a fixed, untuned baseline with a train->validation ROC-AUC gap of {x1_gap:+.4f} "
              f"({x1_t['roc_auc']:.4f} -> {x1_v['roc_auc']:.4f}) and validation log loss {x1_v['log_loss']:.4f}. "
              "This phase asks only whether chronological tuning can improve XGBoost's generalization and "
              "probability quality using exactly the same Phase-3 V1 information - no feature changes.\n")

    md.append("## 2. Temporal CV methodology\n")
    md.append("Outer folds are the **same** expanding-window folds used for RF V2 "
              "(`data/modeling/random_forest_cv_folds_v2.csv`, reused byte-identically). Within each outer fold "
              "the training history is split chronologically again into INNER FIT (earliest ~82.5%) and INNER "
              "EARLY STOP (latest ~17.5%), giving three strictly ordered temporal stages:\n")
    md.append("```\nINNER FIT  <  INNER EARLY STOP  <  OUTER FOLD VALIDATION\n```\n")
    md.append("Asserted for every candidate x fold, with no exact-timestamp group crossing either boundary. "
              "Mirroring applies to inner-fit only; preprocessing is fit on the mirrored inner-fit only; "
              "inner-early-stop and outer-fold-validation are never mirrored. Fold TRAIN metrics use the "
              "**original unmirrored inner-fit rows**.\n")

    md.append("## 3. Why early stopping was allowed only inside TRAIN folds\n")
    md.append("Early stopping needs a holdout to decide when to stop. Using the *outer fold-validation* block "
              "for that would make the candidate's score optimistic, because the model would already have "
              "consulted that block. So a separate inner early-stop block - still entirely inside the global "
              "TRAIN period - serves that role, and the outer fold-validation block is used **only** for "
              "scoring. The 1,419-match main validation partition was never loaded by the tuning script at all.\n")

    md.append("## 4. Candidate search strategy\n")
    md.append(f"{selected['n_candidates']} candidates = {selected['n_anchor_candidates']} deterministic anchors "
              f"+ {selected['n_random_candidates']} `RandomState(42)` random draws over the specified space, "
              "de-duplicated. `n_estimators` was **not** a search dimension: each fold used "
              f"`n_estimators={selected['n_estimators_cap_during_cv']}` as a cap with "
              f"`early_stopping_rounds={selected['early_stopping_rounds']}`. API used: "
              f"{selected['early_stopping_api']}.\n")
    md.append("The anchor `xgb_v1_structure_reference` reproduces XGBoost V1's structural hyperparameters and "
              "**was eligible for selection** - CV could legitimately have concluded V1's structure was fine and "
              "only its fixed 300-round count was suboptimal.\n")

    md.append("## 5. Hyperparameter-selection criterion\n")
    md.append(selected["selection_rule"] + "\n")
    md.append("Accuracy was explicitly not an objective (the project consumes probabilities for tournament "
              "simulation, so log loss leads).\n")

    md.append("## 6. Selected structural parameters\n")
    md.append(f"Candidate **`{selected['selected_candidate_id']}`**, chosen via {selected['selection_stage']}.\n")
    md.append("```\n" + ", ".join(f"{k}={v}" for k, v in hp.items()) + "\n```\n")
    md.append(f"CV: log loss {selected['cv_mean_log_loss']:.4f} ± {selected['cv_std_log_loss']:.4f}, "
              f"ROC-AUC {selected['cv_mean_roc_auc']:.4f} ± {selected['cv_std_roc_auc']:.4f}, "
              f"Brier {selected['cv_mean_brier']:.4f}, accuracy {selected['cv_mean_accuracy']:.4f}.\n")

    md.append("## 7. Best-iteration behavior across folds\n")
    md.append(f"`best_iteration` by fold (1-4): **{selected['best_iterations_by_fold']}**; "
              f"median = {selected['median_best_iteration']:.1f}.\n")

    md.append("## 8. final_n_estimators derivation\n")
    md.append(f"Rule (fixed before the search): `{selected['final_n_estimators_rule']}` -> "
              f"round(median({[b+1 for b in selected['best_iterations_by_fold']]})) = "
              f"**{final_n_estimators}**. Derived from CV only; the main validation partition played no part.\n")

    def mlines(m):
        return "\n".join([
            f"- n = {m['n']:,}", f"- Accuracy: {m['accuracy']:.4f}", f"- Precision: {m['precision']:.4f}",
            f"- Recall: {m['recall']:.4f}", f"- F1: {m['f1']:.4f}", f"- ROC-AUC: {m['roc_auc']:.4f}",
            f"- Log loss: {m['log_loss']:.4f}", f"- Brier: {m['brier_score']:.4f}",
            f"- Confusion matrix [[TN,FP],[FN,TP]]: {m['confusion_matrix']}",
        ])

    md.append("## 9. Train metrics (original unmirrored)\n")
    md.append(mlines(train_metrics) + "\n")
    md.append("## 10. Main validation metrics (one-shot)\n")
    md.append(f"Refit on {len(train_raw):,} unique historical training matches -> "
              f"{len(augmented_train_raw):,} augmented training observations, `n_estimators="
              f"{final_n_estimators}`, no early stopping, no eval_set. Evaluated exactly once:\n")
    md.append(mlines(val_metrics) + "\n")

    md.append("## 11. XGB V1 vs XGB V2\n")
    md.append("| metric | XGB V1 (untuned) | XGB V2 (tuned) | delta |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {x1_v[key]:.4f} | {val_metrics[key]:.4f} | {val_metrics[key]-x1_v[key]:+.4f} |")
    md.append("")

    md.append("## 12. XGB V2 vs RF V2 (both tuned)\n")
    md.append("| metric | RF V2 (tuned) | XGB V2 (tuned) | delta |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {rf2_v[key]:.4f} | {val_metrics[key]:.4f} | {val_metrics[key]-rf2_v[key]:+.4f} |")
    md.append("\nThis is the one genuinely like-for-like comparison so far: both received chronological "
              "TRAIN-only tuning on the same folds.\n")

    md.append("## 13. XGB V2 vs LR V1\n")
    d_lr = val_metrics["roc_auc"] - lr_v["roc_auc"]
    md.append(f"Validation ROC-AUC {val_metrics['roc_auc']:.4f} (XGB V2, tuned) vs {lr_v['roc_auc']:.4f} "
              f"(LR V1, untuned): {d_lr:+.4f}. "
              + ("**The tuned XGBoost outperformed the untuned Logistic Regression baseline.** LR has never "
                 "been tuned, so this is not an algorithm-superiority claim."
                 if d_lr > 0.005 else
                 "Even after tuning, XGBoost did not outperform the untuned Logistic Regression baseline."
                 if d_lr < -0.005 else
                 "The two are essentially tied.") + "\n")

    md.append("## 14. Overfitting change\n")
    md.append(f"- XGB V1 train AUC {x1_t['roc_auc']:.4f} -> validation {x1_v['roc_auc']:.4f}, gap **{x1_gap:+.4f}**")
    md.append(f"- XGB V2 train AUC {train_metrics['roc_auc']:.4f} -> validation {val_metrics['roc_auc']:.4f}, "
              f"gap **{v2_gap:+.4f}**\n")
    md.append(f"Did tuning reduce the gap? **{'Yes' if v2_gap < x1_gap else 'No'}** "
              f"({v2_gap - x1_gap:+.4f}). "
              f"Did it improve validation ROC-AUC? **{'Yes' if val_metrics['roc_auc'] > x1_v['roc_auc'] else 'No'}**. "
              f"Log loss? **{'Yes' if val_metrics['log_loss'] < x1_v['log_loss'] else 'No'}**. "
              f"Brier? **{'Yes' if val_metrics['brier_score'] < x1_v['brier_score'] else 'No'}**. "
              "A successful V2 does not need higher training performance - lower train AUC with higher "
              "validation AUC is evidence of better generalization, not worse fitting.\n")

    md.append("## 15. Calibration\n")
    md.append("`reports/figures/xgboost_v2_calibration.png` (validation only, no correction applied):\n")
    md.append("| bin | n | mean predicted | empirical win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['empirical_win_rate']:.3f} |")
    md.append(f"\nValidation log loss {val_metrics['log_loss']:.4f} and Brier {val_metrics['brier_score']:.4f} "
              f"vs. XGB V1 {x1_v['log_loss']:.4f}/{x1_v['brier_score']:.4f}, RF V2 "
              f"{rf2_v['log_loss']:.4f}/{rf2_v['brier_score']:.4f}, LR V1 {lr_v['log_loss']:.4f}/"
              f"{lr_v['brier_score']:.4f}.\n")

    md.append("## 16. Probability spread\n")
    p = proba_percentiles
    v1p = xgb1_meta.get("validation_probability_percentiles", {})
    md.append("| percentile | XGB V1 | XGB V2 |")
    md.append("|---|---|---|")
    for k in [0, 5, 25, 50, 75, 95, 100]:
        v1val = v1p.get(str(k), v1p.get(k, float("nan")))
        md.append(f"| {'min' if k==0 else 'max' if k==100 else f'p{k}'} | {v1val:.3f} | {p[k]:.3f} |")
    spread_v1 = (v1p.get("95", v1p.get(95, np.nan)) - v1p.get("5", v1p.get(5, np.nan)))
    spread_v2 = p[95] - p[5]
    md.append(f"\np5-p95 spread: XGB V1 {spread_v1:.3f} vs XGB V2 {spread_v2:.3f} - regularization made V2's "
              f"probabilities **{'more conservative (narrower)' if spread_v2 < spread_v1 else 'more extreme (wider)'}**. "
              "Whether that is *better* is judged by log loss/Brier above, not by spread alone.\n")

    md.append("## 17. Feature importance (gain)\n")
    md.append("`gain` = average improvement from splits on the feature; `weight` = number of splits; "
              "`normalized_gain` = share of total gain. Neither is causal importance. Top 8 (all 19 features are "
              "in `reports/tables/xgboost_feature_importance_v2.csv`):\n")
    md.append("| rank | feature | raw_gain | normalized_gain | weight |")
    md.append("|---|---|---|---|---|")
    for _, r in importance_df.head(8).iterrows():
        md.append(f"| {int(r['rank_gain'])} | {r['feature']} | {r['raw_gain']:.4f} | "
                  f"{r['normalized_gain']:.4f} | {int(r['weight'])} |")
    md.append("")

    md.append("## 18. Permutation importance (validation, ROC-AUC)\n")
    md.append("| rank | feature | mean_importance | std_importance |")
    md.append("|---|---|---|---|")
    for _, r in perm_df.head(8).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['mean_importance']:.4f} | {r['std_importance']:.4f} |")
    md.append("")

    # V1-vs-V2 reliance on gain-heavy but weakly generalizing features
    try:
        v1_imp = pd.read_csv(XGB1_IMPORTANCE).set_index("feature")
        v1_perm = pd.read_csv(XGB1_PERM).set_index("feature")
        v2_imp = importance_df.set_index("feature")
        v2_perm = perm_df.set_index("feature")
        watch = ["both_teams_have_5_matches", "both_teams_have_10_matches", "both_teams_have_history",
                  "elo_diff", "total_matches_before_diff"]
        md.append("### V1 vs V2 reliance on specific features\n")
        md.append("| feature | V1 norm. gain | V2 norm. gain | V1 perm | V2 perm |")
        md.append("|---|---|---|---|---|")
        for f in watch:
            md.append(f"| {f} | {v1_imp.loc[f,'normalized_gain']:.4f} | {v2_imp.loc[f,'normalized_gain']:.4f} | "
                      f"{v1_perm.loc[f,'mean_importance']:.4f} | {v2_perm.loc[f,'mean_importance']:.4f} |")
        md.append("\nOf particular interest: `both_teams_have_5_matches`/`both_teams_have_10_matches` were "
                  "gain-heavy but weakly generalizing in V1 (high gain, near-zero validation permutation "
                  "importance). The table shows whether the more regularized V2 reduced that reliance. "
                  "**No feature selection is performed.**\n")
    except Exception as e:  # never let a reporting nicety break the run
        md.append(f"\n(V1-vs-V2 importance comparison unavailable: {e})\n")

    md.append("## 19. Symmetry\n")
    v1s = xgb1_meta.get("validation_symmetry_error", {})
    md.append(f"XGB V2: mean **{sym['mean']:.6f}**, median **{sym['median']:.6f}**, p95 **{sym['p95']:.6f}**, "
              f"max **{sym['max']:.6f}** (XGB V1: {v1s.get('mean', float('nan')):.6f} / "
              f"{v1s.get('median', float('nan')):.6f} / {v1s.get('p95', float('nan')):.6f} / "
              f"{v1s.get('max', float('nan')):.6f}). No assumption was made that tuning improves symmetry; "
              "detail in `reports/xgboost_symmetry_v2.md`. **Not symmetrized.**\n")

    md.append("## 20. Limitations\n")
    md.append("- Logistic Regression has still never been tuned; only RF V2 and XGB V2 are tuned, so no "
              "algorithm-superiority conclusion is available.")
    md.append("- Same 17 series-level features - no map-level or player-level information.")
    md.append("- Four chronological folds over ~2.6 years; CV differences of a few thousandths in log loss are "
              "within noise, which is exactly why the 0.002 equivalence epsilon and deterministic tie-break "
              "ladder exist.")
    md.append("- `final_n_estimators` comes from a median across only 4 folds.")
    md.append("- No probability calibration, no symmetrization, no threshold tuning.")
    md.append("- The main validation partition has now been used once for XGB V2; repeated future use would "
              "erode its independence.\n")

    md.append("## Status\n")
    md.append("- **TEST = SEALED** - not opened or scored.")
    md.append("- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.")
    md.append("- Main validation was evaluated exactly once, after the configuration was frozen.")
    md.append("- No tuning iteration followed the main-validation result.")
    md.append("- No final project model declared.\n")

    (REPORTS / "phase4c1_xgboost_v2_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
