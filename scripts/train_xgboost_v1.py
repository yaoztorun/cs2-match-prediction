"""
Phase 4C orchestration: trains the fixed XGBoost V1 baseline and evaluates it
on TRAIN and VALIDATION only, reusing the EXACT Phase 4A split
(data/modeling/series_split_v1.csv - never regenerated).

NO early stopping and NO eval_set are used anywhere in this file: the main
validation set never influences fitting in any way. The internal TEST
partition is never loaded as a dataframe. Cologne/post-Cologne rows are not
part of series_features_v1.parquet at all (Phase 3 guarantee).
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

from _common import ROOT, REPORTS
from models.xgboost_v1 import build_model, save_model, load_model, XGB_CONFIG
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

LR_MODEL_JSON_PATH = MODELS_DIR / "logistic_regression_scratch_v1.json"
RF_V1_MODEL_JSON_PATH = MODELS_DIR / "random_forest_v1.json"
RF_V2_MODEL_JSON_PATH = MODELS_DIR / "random_forest_v2.json"

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


def load_verified_metrics(path, label):
    """Reads a previously-saved model metadata JSON and VERIFIES the required
    machine-readable metrics exist before using them - never assumed, never
    approximated, never invented. Raises rather than guessing."""
    required = ["accuracy", "roc_auc", "f1", "log_loss", "brier_score", "n", "majority_class_accuracy"]
    if not path.exists():
        raise RuntimeError(f"Cannot build the comparison: {path} does not exist ({label}).")
    meta = json.loads(path.read_text(encoding="utf-8"))
    for split_key in ["train_metrics", "validation_metrics"]:
        if split_key not in meta:
            raise RuntimeError(f"Cannot build the comparison: '{split_key}' missing from {path} ({label}).")
        missing = [k for k in required if k not in meta[split_key]]
        if missing:
            raise RuntimeError(f"Cannot build the comparison: {split_key} in {path} ({label}) missing {missing}.")
    return meta


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    lr_meta = load_verified_metrics(LR_MODEL_JSON_PATH, "Logistic Regression V1")
    rf1_meta = load_verified_metrics(RF_V1_MODEL_JSON_PATH, "Random Forest V1")
    rf2_meta = load_verified_metrics(RF_V2_MODEL_JSON_PATH, "Random Forest V2")
    print("Verified LR V1 / RF V1 / RF V2 metrics exist and are complete.")

    # ---------------- reuse the EXACT Phase 4A split ----------------
    df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    split = pd.read_csv(SPLIT_PATH)
    df = df.merge(split[["match_id", "split"]], on="match_id", how="inner")

    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    test_count = int((df["split"] == "test").sum())
    # TEST IS DELIBERATELY NEVER LOADED AS A DATAFRAME HERE - only its count is checked.

    assert len(train_raw) == EXPECTED_TRAIN_N, f"train count changed: {len(train_raw)}"
    assert len(val_raw) == EXPECTED_VAL_N, f"validation count changed: {len(val_raw)}"
    assert test_count == EXPECTED_TEST_N, f"test count changed: {test_count}"
    print(f"Reused split: train={len(train_raw)}, validation={len(val_raw)}, test={test_count} (sealed, not loaded)")

    # ---------------- mirror TRAIN ONLY ----------------
    augmented_train_raw = build_augmented_training_raw(train_raw)
    assert len(augmented_train_raw) == 2 * len(train_raw) == EXPECTED_AUGMENTED_N
    mirrored_target_mean = float(augmented_train_raw[target_col].mean())
    assert abs(mirrored_target_mean - 0.5) < 1e-12, f"mirrored target mean must be exactly 0.5, got {mirrored_target_mean}"
    assert_augmented_symmetry(augmented_train_raw)
    print(f"train: {len(train_raw)} unique historical matches | augmented: {len(augmented_train_raw)} training "
          f"OBSERVATIONS after mirroring (NOT {len(augmented_train_raw)} independent matches) | "
          f"mirrored target mean: {mirrored_target_mean}")

    # ---------------- fit preprocessing on augmented TRAIN ONLY ----------------
    params = fit_preprocessing(augmented_train_raw, model_features)
    save_preprocessing(params, MODELING_DIR / "xgboost_preprocessing_v1.json")

    X_train_aug, feature_names = transform(augmented_train_raw, params)
    y_train_aug = augmented_train_raw[target_col].to_numpy(dtype=float)
    X_train_orig, _ = transform(train_raw, params)   # UNMIRRORED, for train metrics
    y_train_orig = train_raw[target_col].to_numpy(dtype=float)
    X_val, _ = transform(val_raw, params)
    y_val = val_raw[target_col].to_numpy(dtype=float)

    n_features = X_train_aug.shape[1]
    assert n_features == 19, f"expected 19 transformed features, got {n_features}"
    n_nan_train = int(np.isnan(X_train_aug).sum())
    n_nan_val = int(np.isnan(X_val).sum())
    print(f"transformed feature count: {n_features} | NaN preserved (not imputed): "
          f"{n_nan_train} in augmented train, {n_nan_val} in validation")

    # ---------------- train (single fixed configuration; NO eval_set, NO early stopping) ----------------
    model = build_model()
    model.fit(X_train_aug, y_train_aug)
    print(f"trained XGBClassifier with fixed config: {XGB_CONFIG}")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    save_model(model, MODELS_DIR / "xgboost_v1.json")

    # ---------------- evaluation: TRAIN (unmirrored) + VALIDATION only ----------------
    proba_train = model.predict_proba(X_train_orig)[:, 1]
    pred_train = model.predict(X_train_orig)
    train_metrics = compute_metrics(y_train_orig, proba_train, pred_train)

    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)
    val_metrics = compute_metrics(y_val, proba_val, pred_val)

    assert np.isfinite(proba_val).all() and (proba_val >= 0).all() and (proba_val <= 1).all()

    print("TRAIN metrics:", json.dumps(train_metrics, indent=2))
    print("VALIDATION metrics:", json.dumps(val_metrics, indent=2))

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "validation", **val_metrics}]).to_csv(
        TABLES_DIR / "xgboost_metrics_v1.csv", index=False, encoding="utf-8")

    # ---------------- ROC curve (validation only) ----------------
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fpr, tpr, _ = roc_curve(y_val, proba_val)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"XGBoost V1 (AUC={val_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC curve - XGBoost V1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_roc_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- calibration diagnostic (validation only, same 10-bin scheme) ----------------
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(proba_val, bins) - 1, 0, 9)
    cal_rows = []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        cal_rows.append({"bin": f"[{bins[i]:.1f},{bins[i+1]:.1f})", "n": int(mask.sum()),
                          "mean_predicted": float(proba_val[mask].mean()),
                          "empirical_win_rate": float(y_val[mask].mean())})
    cal_df = pd.DataFrame(cal_rows)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(cal_df["mean_predicted"], cal_df["empirical_win_rate"], s=np.clip(cal_df["n"], 10, None), alpha=0.8)
    ax.set_xlabel("mean predicted P(team1 wins)")
    ax.set_ylabel("empirical team1 win rate")
    ax.set_title("Validation calibration diagnostic (10 bins) - XGBoost V1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_calibration_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- probability distribution (validation only) ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(proba_val, bins=30, range=(0, 1))
    ax.set_xlabel("P(team1 wins)")
    ax.set_ylabel("count")
    ax.set_title("Validation predicted-probability distribution - XGBoost V1")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_probability_distribution_v1.png", dpi=150)
    plt.close(fig)

    pctiles = [0, 5, 25, 50, 75, 95, 100]
    proba_percentiles = {p: float(np.percentile(proba_val, p)) for p in pctiles}

    # ---------------- XGBoost-native feature importance (gain / normalized gain / weight) ----------------
    booster = model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    # booster keys are f0..f18 positional; map back to real feature names, and
    # fill features never used in any split with 0.0 so all 19 stay visible.
    gain_by_feature = {name: float(gain_raw.get(f"f{i}", 0.0)) for i, name in enumerate(feature_names)}
    weight_by_feature = {name: float(weight_raw.get(f"f{i}", 0.0)) for i, name in enumerate(feature_names)}
    total_gain = sum(gain_by_feature.values())

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "raw_gain": [gain_by_feature[n] for n in feature_names],
        "normalized_gain": [(gain_by_feature[n] / total_gain) if total_gain > 0 else 0.0 for n in feature_names],
        "weight": [weight_by_feature[n] for n in feature_names],
    }).sort_values("raw_gain", ascending=False).reset_index(drop=True)
    importance_df["rank_gain"] = importance_df.index + 1
    importance_df.to_csv(TABLES_DIR / "xgboost_feature_importance_v1.csv", index=False, encoding="utf-8")

    # ---------------- validation permutation importance ----------------
    perm = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": feature_names, "mean_importance": perm.importances_mean,
                             "std_importance": perm.importances_std}) \
        .sort_values("mean_importance", ascending=False).reset_index(drop=True)
    perm_df["rank"] = perm_df.index + 1
    perm_df.to_csv(TABLES_DIR / "xgboost_permutation_importance_v1.csv", index=False, encoding="utf-8")

    # ---------------- side-symmetry diagnostic (validation, diagnostic only) ----------------
    val_mirrored_raw = mirror_raw_rows(val_raw)
    X_val_mirrored, _ = transform(val_mirrored_raw, params)
    proba_val_mirrored = model.predict_proba(X_val_mirrored)[:, 1]  # P(B beats A)
    symmetry_error = np.abs(proba_val - (1 - proba_val_mirrored))
    symmetry_stats = {"mean": float(symmetry_error.mean()), "median": float(np.median(symmetry_error)),
                       "p95": float(np.percentile(symmetry_error, 95)), "max": float(symmetry_error.max())}
    print("Side-symmetry diagnostic (validation):", symmetry_stats)

    # ---------------- metadata ----------------
    model_meta = {
        "model_type": "xgboost_classifier",
        "implementation": "xgboost.XGBClassifier",
        "xgboost_version": __import__("xgboost").__version__,
        "compatibility_adjustments": "none required - the specified configuration ran as-is on xgboost 3.4.0",
        **{k: v for k, v in XGB_CONFIG.items()},
        "early_stopping_used": False,
        "eval_set_used": False,
        "training_cutoff": str(train_raw["datetime"].max()),
        "unique_training_matches": int(len(train_raw)),
        "augmented_training_observations": int(len(augmented_train_raw)),
        "mirrored_train_target_mean": mirrored_target_mean,
        "feature_count": int(n_features),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "nan_count_augmented_train": n_nan_train,
        "nan_count_validation": n_nan_val,
        "feature_config": "config/series_features_v1.yaml",
        "preprocessing_artifact": "data/modeling/xgboost_preprocessing_v1.json",
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_manifest": "data/modeling/series_split_v1.csv",
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_probability_percentiles": proba_percentiles,
        "validation_symmetry_error": symmetry_stats,
        "test_status": "SEALED - not evaluated in Phase 4C",
        "cologne_status": "UNTOUCHED - not present in series_features_v1.parquet",
    }
    (MODELS_DIR / "xgboost_v1_metadata.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    write_symmetry_report(symmetry_error, symmetry_stats)
    write_model_comparison(lr_meta, rf1_meta, rf2_meta, train_metrics, val_metrics)
    write_phase4c_report(train_raw, val_raw, augmented_train_raw, feature_names, params,
                          train_metrics, val_metrics, importance_df, perm_df, cal_df,
                          proba_percentiles, symmetry_stats, model_meta,
                          lr_meta, rf1_meta, rf2_meta, n_nan_train, n_nan_val)

    print("\nWrote models/xgboost_v1.json (native XGBClassifier format)")
    print("Wrote models/xgboost_v1_metadata.json")
    print("Wrote data/modeling/xgboost_preprocessing_v1.json")
    print("Wrote reports/tables/xgboost_metrics_v1.csv")
    print("Wrote reports/tables/xgboost_feature_importance_v1.csv")
    print("Wrote reports/tables/xgboost_permutation_importance_v1.csv")
    print("Wrote reports/figures/xgboost_{roc,calibration,probability_distribution}_v1.png")
    print("Wrote reports/xgboost_symmetry_v1.md")
    print("Wrote reports/model_comparison_v1.md")
    print("Wrote reports/phase4c_xgboost_v1.md")


def write_symmetry_report(symmetry_error, stats):
    md = []
    md.append("# XGBoost V1 - Side-Symmetry Diagnostic\n")
    md.append("For every validation matchup A-vs-B, the raw mirrored B-vs-A form was built with the same "
              "`mirror_raw_rows` used for training augmentation, transformed with the SAME fitted XGBoost "
              "preprocessing artifact, and scored with the same model. For a perfectly side-consistent model, "
              "`P(A beats B) + P(B beats A) == 1`.\n")
    md.append("**No prior claim is made about the expected value of this error.** A gradient-boosted tree "
              "ensemble is *not* mathematically constrained to satisfy antisymmetry, even though its raw "
              "mirrored directional inputs are exact negatives, its symmetric/context inputs are unchanged, its "
              "mirrored training set is exactly balanced, and the model itself is deterministic - determinism "
              "does not imply antisymmetry. This is treated exactly like the Random Forest diagnostic: measured "
              "and reported, never assumed and never corrected in V1.\n")
    md.append(f"- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n={len(symmetry_error)} validation rows")
    md.append(f"- mean: {stats['mean']:.6f}")
    md.append(f"- median: {stats['median']:.6f}")
    md.append(f"- 95th percentile: {stats['p95']:.6f}")
    md.append(f"- max: {stats['max']:.6f}\n")
    if stats["mean"] < 0.02:
        md.append("**Observation**: empirically, mirrored training produced approximately side-consistent "
                  "predictions on this validation set. This is an observed property of this fitted model, not a "
                  "guarantee of the algorithm.\n")
    else:
        md.append("**Observation**: a non-trivial asymmetry is present. A later phase could compare the raw "
                  "probability against an explicit symmetrized probability "
                  "`0.5 * [P(A beats B) + (1 - P(B beats A))]` - not implemented here.\n")
    md.append("**No probability symmetrization is applied in V1.**\n")
    (REPORTS / "xgboost_symmetry_v1.md").write_text("\n".join(md), encoding="utf-8")


def write_model_comparison(lr_meta, rf1_meta, rf2_meta, xgb_train, xgb_val):
    lr_v, lr_t = lr_meta["validation_metrics"], lr_meta["train_metrics"]
    rf1_v, rf1_t = rf1_meta["validation_metrics"], rf1_meta["train_metrics"]
    rf2_v, rf2_t = rf2_meta["validation_metrics"], rf2_meta["train_metrics"]
    maj = lr_v["majority_class_accuracy"]

    md = []
    md.append("# Model Comparison V1 (Validation Only)\n")
    md.append("All models use the identical chronological split, the identical 17 Phase-3 features, and the "
              "identical train-only mirrored augmentation policy. LR/RF numbers are read from their saved "
              "metadata JSONs (verified present and complete before use); none of those models were re-run or "
              "modified.\n")
    md.append("**Tuning status matters for fair reading:**\n")
    md.append("- **LR V1** = untuned (from-scratch Logistic Regression baseline)")
    md.append("- **RF V1** = untuned Random Forest baseline")
    md.append("- **RF V2** = chronologically tuned Random Forest (TRAIN-only expanding-window CV)")
    md.append("- **XGBoost V1** = untuned fixed baseline\n")
    md.append("Only RF V2 has received any hyperparameter search. Comparisons below must not be read as "
              "algorithm-superiority claims.\n")

    md.append("## Validation metrics\n")
    md.append("| metric | majority | LR V1 (untuned) | RF V1 (untuned) | RF V2 (tuned) | XGBoost V1 (untuned) |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| Accuracy | {maj:.4f} | {lr_v['accuracy']:.4f} | {rf1_v['accuracy']:.4f} | {rf2_v['accuracy']:.4f} | {xgb_val['accuracy']:.4f} |")
    md.append(f"| ROC-AUC | 0.5000 | {lr_v['roc_auc']:.4f} | {rf1_v['roc_auc']:.4f} | {rf2_v['roc_auc']:.4f} | {xgb_val['roc_auc']:.4f} |")
    md.append(f"| F1 | - | {lr_v['f1']:.4f} | {rf1_v['f1']:.4f} | {rf2_v['f1']:.4f} | {xgb_val['f1']:.4f} |")
    md.append(f"| Log loss | - | {lr_v['log_loss']:.4f} | {rf1_v['log_loss']:.4f} | {rf2_v['log_loss']:.4f} | {xgb_val['log_loss']:.4f} |")
    md.append(f"| Brier | - | {lr_v['brier_score']:.4f} | {rf1_v['brier_score']:.4f} | {rf2_v['brier_score']:.4f} | {xgb_val['brier_score']:.4f} |\n")

    md.append("## Train metrics and train -> validation gaps\n")
    md.append("| model | train acc | val acc | acc gap | train AUC | val AUC | AUC gap |")
    md.append("|---|---|---|---|---|---|---|")
    for label, t, v in [("LR V1 (untuned)", lr_t, lr_v), ("RF V1 (untuned)", rf1_t, rf1_v),
                         ("RF V2 (tuned)", rf2_t, rf2_v), ("XGBoost V1 (untuned)", xgb_train, xgb_val)]:
        md.append(f"| {label} | {t['accuracy']:.4f} | {v['accuracy']:.4f} | {t['accuracy']-v['accuracy']:+.4f} | "
                  f"{t['roc_auc']:.4f} | {v['roc_auc']:.4f} | {t['roc_auc']-v['roc_auc']:+.4f} |")
    md.append("")

    md.append("## XGBoost V1 deltas\n")
    md.append("| metric | XGB V1 - LR V1 | XGB V1 - RF V1 | XGB V1 - RF V2 |")
    md.append("|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier")]:
        md.append(f"| {label} | {xgb_val[key]-lr_v[key]:+.4f} | {xgb_val[key]-rf1_v[key]:+.4f} | "
                  f"{xgb_val[key]-rf2_v[key]:+.4f} |")
    md.append("\nFor log loss and Brier, **more negative is better**; for accuracy/ROC-AUC/F1, more positive is "
              "better.\n")
    md.append("**No final project model is declared.** Only RF V2 has been tuned; LR and XGBoost remain untuned "
              "baselines, and the internal test set has not been used for this or any comparison.\n")
    (REPORTS / "model_comparison_v1.md").write_text("\n".join(md), encoding="utf-8")


def write_phase4c_report(train_raw, val_raw, augmented_train_raw, feature_names, params,
                          train_metrics, val_metrics, importance_df, perm_df, cal_df,
                          proba_percentiles, symmetry_stats, model_meta,
                          lr_meta, rf1_meta, rf2_meta, n_nan_train, n_nan_val):
    lr_v = lr_meta["validation_metrics"]
    rf1_v, rf1_t = rf1_meta["validation_metrics"], rf1_meta["train_metrics"]
    rf2_v, rf2_t = rf2_meta["validation_metrics"], rf2_meta["train_metrics"]
    lr_t = lr_meta["train_metrics"]

    acc_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    auc_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]

    md = []
    md.append("# Phase 4C - Model 3: XGBoost Classifier V1\n")
    md.append("Terminology: **LR V1** = untuned from-scratch Logistic Regression. **RF V1** = untuned Random "
              "Forest. **RF V2** = chronologically tuned Random Forest. **XGBoost V1** = untuned fixed "
              "gradient-boosting baseline (this phase).\n")

    md.append("## Fixed V1 configuration\n")
    md.append("`xgboost.XGBClassifier` (library implementation - Model 1 was from-scratch only because it was "
              "adapted from the course lab). Single fixed baseline, **no tuning, no early stopping, no "
              "`eval_set`**:\n")
    md.append("```\n" + ", ".join(f"{k}={v}" for k, v in XGB_CONFIG.items()) + "\n```\n")
    md.append(f"Installed `xgboost=={model_meta['xgboost_version']}`; {model_meta['compatibility_adjustments']}. "
              "This is a deliberately moderate baseline (shallower trees than RF V1, shrinkage via "
              "`learning_rate`, row/feature subsampling, default-style L2) and **must not be described as an "
              "optimized configuration**.\n")

    md.append("## Data, split and mirroring\n")
    md.append(f"Reused `data/modeling/series_split_v1.csv` unchanged: **{len(train_raw):,} unique historical "
              f"training matches**, {len(val_raw):,} validation matches, 1,418 test matches (SEALED - never "
              f"loaded). Mirroring applied to TRAIN only, producing **{len(augmented_train_raw):,} augmented "
              f"training observations** (never described as {len(augmented_train_raw):,} independent matches - "
              f"each mirrored row is a synthetic relabeling of an already-counted match). Augmented target mean "
              f"= **{model_meta['mirrored_train_target_mean']}** (exactly 0.5). Validation and test are never "
              "mirrored.\n")

    md.append("## Preprocessing and the missing-value decision\n")
    md.append(f"Same 17-feature whitelist and the same deterministic `bestOf`/`tier` reference-dummy encoding as "
              f"LR/RF, giving the same {len(feature_names)} transformed columns. **No standardization** (tree "
              "splits are scale-invariant). **NaN preserved rather than median-imputed** "
              f"(`missing_value_policy = {model_meta['missing_value_policy']}`): XGBoost natively learns a "
              "default split direction for missing values, so a cold-start team's unknown "
              "`days_since_last_match_diff` stays a distinguishable signal instead of collapsing onto the "
              f"median. This affects exactly one column ({n_nan_train} NaN values in augmented train, "
              f"{n_nan_val} in validation; no other whitelist column has any missingness). A verified side "
              "benefit: because `mirror_raw_rows` negates NaN to NaN, preserving NaN makes the augmented "
              "directional-diff means **exactly 0.0**, whereas LR/RF median imputation left a small residual "
              "asymmetry. No validation/test information is used to fill anything.\n")

    md.append("## Train metrics (unmirrored)\n")
    for line in [f"- n = {train_metrics['n']:,}", f"- Accuracy: {train_metrics['accuracy']:.4f}",
                  f"- Precision: {train_metrics['precision']:.4f}", f"- Recall: {train_metrics['recall']:.4f}",
                  f"- F1: {train_metrics['f1']:.4f}", f"- ROC-AUC: {train_metrics['roc_auc']:.4f}",
                  f"- Log loss: {train_metrics['log_loss']:.4f}", f"- Brier: {train_metrics['brier_score']:.4f}",
                  f"- Confusion matrix [[TN,FP],[FN,TP]]: {train_metrics['confusion_matrix']}"]:
        md.append(line)
    md.append("")

    md.append("## Validation metrics\n")
    for line in [f"- n = {val_metrics['n']:,}",
                  f"- Accuracy: {val_metrics['accuracy']:.4f} (majority-class reference: {val_metrics['majority_class_accuracy']:.4f})",
                  f"- Precision: {val_metrics['precision']:.4f}", f"- Recall: {val_metrics['recall']:.4f}",
                  f"- F1: {val_metrics['f1']:.4f}", f"- ROC-AUC: {val_metrics['roc_auc']:.4f}",
                  f"- Log loss: {val_metrics['log_loss']:.4f}", f"- Brier: {val_metrics['brier_score']:.4f}",
                  f"- Confusion matrix [[TN,FP],[FN,TP]]: {val_metrics['confusion_matrix']}"]:
        md.append(line)
    md.append("")

    md.append("## Interpretation questions\n")

    d_lr = val_metrics["roc_auc"] - lr_v["roc_auc"]
    d_rf1 = val_metrics["roc_auc"] - rf1_v["roc_auc"]
    d_rf2 = val_metrics["roc_auc"] - rf2_v["roc_auc"]

    md.append("### A. Did boosting improve validation discrimination relative to LR V1 and RF V1?\n")
    md.append(f"XGBoost V1 validation ROC-AUC = **{val_metrics['roc_auc']:.4f}**, vs. LR V1 "
              f"{lr_v['roc_auc']:.4f} ({d_lr:+.4f}) and RF V1 {rf1_v['roc_auc']:.4f} ({d_rf1:+.4f}). "
              + ("Boosting improved discrimination over both untuned baselines."
                 if d_lr > 0.005 and d_rf1 > 0.005 else
                 "Boosting improved discrimination over RF V1 but not over LR V1."
                 if d_rf1 > 0.005 and d_lr <= 0.005 else
                 "Boosting improved discrimination over LR V1 but not over RF V1."
                 if d_lr > 0.005 and d_rf1 <= 0.005 else
                 "Boosting did not clearly improve discrimination over both untuned baselines.") + "\n")

    md.append("### B. How does XGBoost V1 compare with tuned RF V2?\n")
    md.append(f"RF V2 validation ROC-AUC {rf2_v['roc_auc']:.4f} vs. XGBoost V1 {val_metrics['roc_auc']:.4f} "
              f"({d_rf2:+.4f} for XGBoost). "
              + ("XGBoost V1 edges out RF V2 on discrimination"
                 if d_rf2 > 0.005 else
                 "RF V2 remains ahead on discrimination"
                 if d_rf2 < -0.005 else
                 "The two are essentially tied on discrimination")
              + " - but note this is an **untuned** XGBoost against a **chronologically tuned** Random Forest, "
              "so the comparison is not like-for-like in tuning effort in either direction.\n")

    md.append("### C. Is there evidence of overfitting?\n")
    md.append(f"XGBoost V1 train->validation gaps: accuracy **{acc_gap:+.4f}**, ROC-AUC **{auc_gap:+.4f}** "
              f"(train acc {train_metrics['accuracy']:.4f} / AUC {train_metrics['roc_auc']:.4f}). For context: "
              f"LR V1 {lr_t['accuracy']-lr_v['accuracy']:+.4f} acc / {lr_t['roc_auc']-lr_v['roc_auc']:+.4f} AUC, "
              f"RF V1 {rf1_t['accuracy']-rf1_v['accuracy']:+.4f} / {rf1_t['roc_auc']-rf1_v['roc_auc']:+.4f}, "
              f"RF V2 {rf2_t['accuracy']-rf2_v['accuracy']:+.4f} / {rf2_t['roc_auc']-rf2_v['roc_auc']:+.4f}. "
              + ("XGBoost V1 shows a clearly smaller gap than untuned RF V1, indicating the shallower "
                 "depth/shrinkage/subsampling controlled overfitting substantially."
                 if auc_gap < (rf1_t['roc_auc']-rf1_v['roc_auc']) / 2 else
                 "XGBoost V1's gap is documented as observed.")
              + " **No configuration change was made in response** - tuning is out of scope for Phase 4C.\n")

    md.append("### D. Are XGBoost probabilities better or worse (log loss / Brier / calibration)?\n")
    md.append(f"XGBoost V1 validation log loss **{val_metrics['log_loss']:.4f}** and Brier "
              f"**{val_metrics['brier_score']:.4f}**, vs. LR V1 {lr_v['log_loss']:.4f}/{lr_v['brier_score']:.4f}, "
              f"RF V1 {rf1_v['log_loss']:.4f}/{rf1_v['brier_score']:.4f}, RF V2 "
              f"{rf2_v['log_loss']:.4f}/{rf2_v['brier_score']:.4f} (lower is better for both). This matters "
              "because these probabilities will later feed the tournament simulator. Per-bin calibration "
              "(`reports/figures/xgboost_calibration_v1.png`, diagnostic only - no isotonic/Platt/temperature "
              "scaling applied):\n")
    md.append("| bin | n | mean predicted | empirical win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['empirical_win_rate']:.3f} |")
    md.append("")

    md.append("### E. Which features appear most useful (gain vs. validation permutation importance)?\n")
    md.append("`gain` = average improvement in the split criterion contributed by splits on that feature; "
              "`weight` = how many times the feature was used as a split. `normalized_gain` is each feature's "
              "share of total gain. **Neither is causal importance**, and gain (like RF's impurity importance) "
              "is computed on training structure - validation **permutation importance** is the stronger "
              "generalization-oriented diagnostic. Top 8 by gain:\n")
    md.append("| rank | feature | raw_gain | normalized_gain | weight |")
    md.append("|---|---|---|---|---|")
    for _, r in importance_df.head(8).iterrows():
        md.append(f"| {int(r['rank_gain'])} | {r['feature']} | {r['raw_gain']:.4f} | "
                  f"{r['normalized_gain']:.4f} | {int(r['weight'])} |")
    md.append("\nTop 8 by validation permutation importance (ROC-AUC, n_repeats=10):\n")
    md.append("| rank | feature | mean_importance | std_importance |")
    md.append("|---|---|---|---|")
    for _, r in perm_df.head(8).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['mean_importance']:.4f} | {r['std_importance']:.4f} |")
    md.append("\nNo features were removed - this is interpretation only.\n")

    top_perm = perm_df.head(3)["feature"].tolist()
    elo_perm_rank = int(perm_df[perm_df["feature"] == "elo_diff"]["rank"].iloc[0])
    exp_perm_rank = int(perm_df[perm_df["feature"] == "total_matches_before_diff"]["rank"].iloc[0])
    md.append("### F. Do ELO and historical experience remain the strongest robust signals?\n")
    md.append(f"By validation permutation importance, `elo_diff` ranks **#{elo_perm_rank}** and "
              f"`total_matches_before_diff` ranks **#{exp_perm_rank}** (top 3 overall: {top_perm}). "
              + ("Yes - ELO and historical-experience remain the strongest robust signals, consistent with "
                 "LR V1's largest coefficients and RF V1/V2's permutation rankings."
                 if elo_perm_rank <= 2 and exp_perm_rank <= 3 else
                 "The ranking differs from the LR/RF pattern - reported as observed.") + "\n")

    md.append("### G. Did XGBoost make useful use of nonlinear interactions that LR cannot represent?\n")
    if d_lr > 0.005:
        md.append(f"On this validation evidence, **yes, tentatively**: XGBoost V1's ROC-AUC exceeds LR V1's by "
                  f"{d_lr:+.4f}, and the boosted trees can represent thresholds/interactions (e.g. `elo_diff` "
                  "mattering differently by `bestOf` or by whether both teams have enough history) that a linear "
                  "model cannot express without explicit interaction terms. This is an association observed on "
                  "one validation period, not a proof that specific interactions were learned.")
    elif d_lr < -0.005:
        md.append(f"On this validation evidence, **no**: XGBoost V1's ROC-AUC is {d_lr:+.4f} relative to LR V1's, "
                  "i.e. worse. Whatever additional nonlinear capacity boosting has did not convert into better "
                  "validation discrimination here.")
    else:
        md.append(f"On this validation evidence, **inconclusive**: the ROC-AUC difference vs. LR V1 ({d_lr:+.4f}) "
                  "is too small to attribute to genuine nonlinear/interaction gains rather than noise.")
    md.append(" No claim beyond what the validation numbers support is made.\n")

    md.append("## Probability distribution\n")
    p = proba_percentiles
    md.append("| min | p5 | p25 | median | p75 | p95 | max |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| {p[0]:.3f} | {p[5]:.3f} | {p[25]:.3f} | {p[50]:.3f} | {p[75]:.3f} | {p[95]:.3f} | {p[100]:.3f} |\n")
    md.append("Compare qualitatively with LR V1, RF V1 and RF V2's distributions (see their respective "
              "probability-distribution figures) to judge whether boosting produces more extreme or more "
              "conservative probabilities. See `reports/figures/xgboost_probability_distribution_v1.png`.\n")

    md.append("## Side-symmetry diagnostic\n")
    md.append(f"mean **{symmetry_stats['mean']:.6f}**, median **{symmetry_stats['median']:.6f}**, p95 "
              f"**{symmetry_stats['p95']:.6f}**, max **{symmetry_stats['max']:.6f}**. Tree ensembles are not "
              "mathematically constrained to satisfy `P(A beats B) = 1 - P(B beats A)` - determinism does not "
              "imply antisymmetry - so no expected value was assumed in advance. Detail in "
              "`reports/xgboost_symmetry_v1.md`. **No symmetrization applied.**\n")

    md.append("## Comparison summary\n")
    md.append("Full five-way table in `reports/model_comparison_v1.md`.\n")

    md.append("## Limitations / deferred\n")
    md.append("- XGBoost V1 is **untuned**; RF V2 is the only tuned model so far, so cross-algorithm rankings "
              "here are not fair algorithm-superiority evidence.")
    md.append("- No probability calibration and no probability symmetrization applied.")
    md.append("- Same 17 series-level features - no map-level or player-level detail.")
    md.append("- Single chronological train/validation split; no cross-validation in this phase.")
    md.append("- Future XGBoost tuning must use chronological CV **inside TRAIN**, exactly as RF V2 did - never "
              "the main validation set, and never early stopping against it.\n")

    md.append("## Status\n")
    md.append("- **TEST = SEALED** - not opened or scored in this phase.")
    md.append("- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.")
    md.append("- No early stopping and no `eval_set` were used at any point.")
    md.append("- Exactly one fixed XGBoost configuration was trained; no tuning performed.\n")

    (REPORTS / "phase4c_xgboost_v1.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
