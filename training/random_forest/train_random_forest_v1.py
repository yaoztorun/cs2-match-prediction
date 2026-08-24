"""
Phase 4B orchestration: trains the Random Forest V1 baseline and evaluates it
on TRAIN and VALIDATION only, reusing the EXACT Phase 4A split
(data/modeling/series_split_v1.csv - never regenerated). The internal TEST
partition is never loaded for scoring. Cologne/post-Cologne rows are not
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

from _common import ROOT, REPORTS, INTERIM
from training.random_forest.random_forest_v1 import build_model, save_model, RF_CONFIG
from feature_engineering.preprocessing.preprocessing_random_forest_v1 import (
    build_augmented_training_raw, fit_preprocessing, transform, save_preprocessing,
    assert_augmented_symmetry, mirror_raw_rows, PREPROCESSING_VERSION,
)

CONFIG_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = REPORTS / "figures"
TABLES_DIR = REPORTS / "tables"
LR_MODEL_JSON_PATH = MODELS_DIR / "series" / "logistic_regression_scratch_v1.json"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418

THRESHOLD = 0.5


def majority_class_accuracy(y):
    p1 = np.mean(y)
    return max(p1, 1 - p1)


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
        "majority_class_accuracy": float(majority_class_accuracy(y_true)),
    }


def load_verified_lr_metrics():
    """Reads models/logistic_regression_scratch_v1.json and VERIFIES the
    required machine-readable train/validation metrics actually exist there
    before using them anywhere - never assumed, never approximated, never
    invented. Raises a clear error if the artifact is missing or incomplete,
    per explicit instruction: stop/report rather than guess."""
    required_keys = ["accuracy", "roc_auc", "f1", "log_loss", "brier_score", "n",
                      "precision", "recall", "confusion_matrix", "majority_class_accuracy"]
    if not LR_MODEL_JSON_PATH.exists():
        raise RuntimeError(
            f"Cannot build the LR-vs-RF comparison: {LR_MODEL_JSON_PATH} does not exist. "
            "Phase 4A must be run first; refusing to invent or approximate Logistic Regression metrics."
        )
    meta = json.loads(LR_MODEL_JSON_PATH.read_text(encoding="utf-8"))
    for split_key in ["train_metrics", "validation_metrics"]:
        if split_key not in meta:
            raise RuntimeError(
                f"Cannot build the LR-vs-RF comparison: '{split_key}' is missing from {LR_MODEL_JSON_PATH}. "
                "Refusing to invent or approximate Logistic Regression metrics."
            )
        missing = [k for k in required_keys if k not in meta[split_key]]
        if missing:
            raise RuntimeError(
                f"Cannot build the LR-vs-RF comparison: {split_key} in {LR_MODEL_JSON_PATH} is missing "
                f"required keys {missing}. Refusing to invent or approximate Logistic Regression metrics."
            )
    return meta


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    # ---------------- verify LR metrics exist BEFORE doing any RF work ----------------
    # (fail fast rather than training everything and only then discovering the
    # comparison report can't be built)
    lr_meta = load_verified_lr_metrics()
    print("Verified Logistic Regression metrics exist and are complete in "
          f"{LR_MODEL_JSON_PATH.relative_to(ROOT)}")

    # ---------------- reuse the EXACT Phase 4A split - never regenerated ----------------
    df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    split = pd.read_csv(SPLIT_PATH)
    df = df.merge(split[["match_id", "split"]], on="match_id", how="inner")
    assert len(df) == len(split), "split file and feature parquet row counts diverged"

    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    test_count = int((df["split"] == "test").sum())
    # TEST IS DELIBERATELY NEVER LOADED AS A DATAFRAME HERE - only its count is checked.

    assert len(train_raw) == EXPECTED_TRAIN_N, f"train count changed: {len(train_raw)} != {EXPECTED_TRAIN_N}"
    assert len(val_raw) == EXPECTED_VAL_N, f"validation count changed: {len(val_raw)} != {EXPECTED_VAL_N}"
    assert test_count == EXPECTED_TEST_N, f"test count changed: {test_count} != {EXPECTED_TEST_N}"
    print(f"Reused split: train={len(train_raw)}, validation={len(val_raw)}, test={test_count} (sealed, not loaded)")

    # ---------------- mirror TRAIN ONLY (same policy as Logistic Regression) ----------------
    augmented_train_raw = build_augmented_training_raw(train_raw)
    assert len(augmented_train_raw) == 2 * len(train_raw) == 13238
    mirrored_target_mean = float(augmented_train_raw[target_col].mean())
    assert abs(mirrored_target_mean - 0.5) < 1e-9, \
        f"mirrored augmented training target mean must be exactly 0.5, got {mirrored_target_mean}"
    assert_augmented_symmetry(augmented_train_raw)
    print(f"train: {len(train_raw)} unique historical matches | "
          f"augmented: {len(augmented_train_raw)} training OBSERVATIONS after mirroring "
          f"(NOT {len(augmented_train_raw)} matches - every mirrored row is a synthetic relabeling of an "
          f"already-counted match) | mirrored target mean: {mirrored_target_mean}")

    # ---------------- fit RF preprocessing on augmented train ONLY (no scaling) ----------------
    params = fit_preprocessing(augmented_train_raw, model_features)
    save_preprocessing(params, MODELING_DIR / "random_forest_preprocessing_v1.json")

    X_train_aug, feature_names = transform(augmented_train_raw, params)
    y_train_aug = augmented_train_raw[target_col].to_numpy(dtype=float)

    X_train_orig, _ = transform(train_raw, params)
    y_train_orig = train_raw[target_col].to_numpy(dtype=float)

    X_val, _ = transform(val_raw, params)
    y_val = val_raw[target_col].to_numpy(dtype=float)

    n_features = X_train_aug.shape[1]
    print(f"transformed feature count: {n_features} -> {feature_names}")
    assert n_features == 19, f"expected 19 transformed features, got {n_features}"

    # ---------------- train (single fixed baseline configuration, no tuning) ----------------
    model = build_model()
    model.fit(X_train_aug, y_train_aug)
    print(f"trained RandomForestClassifier with config: {RF_CONFIG}")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    save_model(model, MODELS_DIR / "series" / "random_forest_v1.joblib")

    # ---------------- evaluation: TRAIN (unmirrored) + VALIDATION only ----------------
    proba_train = model.predict_proba(X_train_orig)[:, 1]
    pred_train = model.predict(X_train_orig)
    train_metrics = compute_metrics(y_train_orig, proba_train, pred_train)

    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)
    val_metrics = compute_metrics(y_val, proba_val, pred_val)

    assert (proba_train >= 0).all() and (proba_train <= 1).all()
    assert (proba_val >= 0).all() and (proba_val <= 1).all()
    assert np.isfinite(proba_val).all()

    print("TRAIN metrics:", json.dumps(train_metrics, indent=2))
    print("VALIDATION metrics:", json.dumps(val_metrics, indent=2))

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    metrics_df = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "validation", **val_metrics},
    ])
    metrics_df.to_csv(TABLES_DIR / "random_forest_metrics_v1.csv", index=False, encoding="utf-8")

    # ---------------- ROC curve (validation only, same style as LR) ----------------
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fpr, tpr, _ = roc_curve(y_val, proba_val)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"Random Forest V1 (AUC={val_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC curve - Random Forest V1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_roc_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- calibration diagnostic (validation only, same 10-bin scheme as LR) ----------------
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(proba_val, bins) - 1, 0, 9)
    cal_rows = []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        cal_rows.append({
            "bin": f"[{bins[i]:.1f},{bins[i+1]:.1f})",
            "n": int(mask.sum()),
            "mean_predicted": float(proba_val[mask].mean()),
            "empirical_win_rate": float(y_val[mask].mean()),
        })
    cal_df = pd.DataFrame(cal_rows)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(cal_df["mean_predicted"], cal_df["empirical_win_rate"],
               s=np.clip(cal_df["n"], 10, None), alpha=0.8)
    ax.set_xlabel("mean predicted P(team1 wins)")
    ax.set_ylabel("empirical team1 win rate")
    ax.set_title("Validation calibration diagnostic (10 bins) - Random Forest V1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_calibration_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- probability distribution (validation only) ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(proba_val, bins=30, range=(0, 1))
    ax.set_xlabel("P(team1 wins)")
    ax.set_ylabel("count")
    ax.set_title("Validation predicted-probability distribution - Random Forest V1")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_probability_distribution_v1.png", dpi=150)
    plt.close(fig)

    pctiles = [0, 5, 25, 50, 75, 95, 100]
    proba_percentiles = {p: float(np.percentile(proba_val, p)) for p in pctiles}

    # ---------------- feature importance (impurity) ----------------
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "impurity_importance": model.feature_importances_,
    }).sort_values("impurity_importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df.to_csv(TABLES_DIR / "random_forest_feature_importance_v1.csv", index=False, encoding="utf-8")

    # ---------------- permutation importance (validation, roc_auc) ----------------
    perm = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": perm.importances_mean,
        "std_importance": perm.importances_std,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    perm_df["rank"] = perm_df.index + 1
    perm_df.to_csv(TABLES_DIR / "random_forest_permutation_importance_v1.csv", index=False, encoding="utf-8")

    # ---------------- side-symmetry diagnostic (validation, diagnostic only) ----------------
    val_mirrored_raw = mirror_raw_rows(val_raw)
    X_val_mirrored, _ = transform(val_mirrored_raw, params)
    proba_val_mirrored = model.predict_proba(X_val_mirrored)[:, 1]  # P(B beats A)
    symmetry_error = np.abs(proba_val - (1 - proba_val_mirrored))
    symmetry_stats = {
        "mean": float(symmetry_error.mean()),
        "median": float(np.median(symmetry_error)),
        "p95": float(np.percentile(symmetry_error, 95)),
        "max": float(symmetry_error.max()),
    }
    print("Side-symmetry diagnostic (validation):", symmetry_stats)

    # ---------------- save model metadata ----------------
    model_meta = {
        "model_type": "random_forest_classifier",
        "implementation": "sklearn.ensemble.RandomForestClassifier",
        "n_estimators": RF_CONFIG["n_estimators"],
        "criterion": RF_CONFIG["criterion"],
        "max_depth": RF_CONFIG["max_depth"],
        "min_samples_split": RF_CONFIG["min_samples_split"],
        "min_samples_leaf": RF_CONFIG["min_samples_leaf"],
        "max_features": RF_CONFIG["max_features"],
        "bootstrap": RF_CONFIG["bootstrap"],
        "class_weight": RF_CONFIG["class_weight"],
        "random_state": RF_CONFIG["random_state"],
        "training_cutoff": str(train_raw["datetime"].max()),
        "original_train_rows": int(len(train_raw)),
        "augmented_train_rows": int(len(augmented_train_raw)),
        "feature_count": int(n_features),
        "preprocessing_artifact": "data/modeling/random_forest_preprocessing_v1.json",
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_config": "config/features/series_features_v1.yaml",
        "split_manifest": "data/modeling/series_split_v1.csv",
        "mirrored_train_target_mean": mirrored_target_mean,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_probability_percentiles": proba_percentiles,
        "validation_symmetry_error": symmetry_stats,
        "test_status": "SEALED - not evaluated in Phase 4B",
        "cologne_status": "UNTOUCHED - not present in series_features_v1.parquet",
    }
    (MODELS_DIR / "series" / "random_forest_v1.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    write_symmetry_report(symmetry_error, symmetry_stats)
    write_comparison_report(lr_meta, val_metrics, train_metrics)
    write_phase4b_report(
        train_raw, val_raw, augmented_train_raw, feature_names, params,
        train_metrics, val_metrics, importance_df, perm_df, cal_df,
        proba_percentiles, symmetry_stats, model_meta, lr_meta,
    )

    print("\nWrote models/random_forest_v1.joblib")
    print("Wrote models/random_forest_v1.json")
    print("Wrote data/modeling/random_forest_preprocessing_v1.json")
    print("Wrote reports/tables/random_forest_metrics_v1.csv")
    print("Wrote reports/tables/random_forest_feature_importance_v1.csv")
    print("Wrote reports/tables/random_forest_permutation_importance_v1.csv")
    print("Wrote reports/figures/random_forest_{roc,calibration,probability_distribution}_v1.png")
    print("Wrote reports/random_forest_symmetry_v1.md")
    print("Wrote reports/model_comparison_lr_vs_rf_v1.md")
    print("Wrote reports/phase4b_random_forest_v1.md")


def write_symmetry_report(symmetry_error, symmetry_stats):
    md = []
    md.append("# Random Forest V1 - Side-Symmetry Diagnostic\n")
    md.append("For every validation matchup A-vs-B, the raw mirrored B-vs-A form was built with the same "
              "`mirror_raw_rows` used for training augmentation, transformed with the SAME fitted Random Forest "
              "preprocessing artifact, and scored with the same model. For a perfectly side-consistent model, "
              "`P(A beats B) + P(B beats A) == 1`. Random Forest is **not** mathematically guaranteed to satisfy "
              "this exactly even when trained on mirrored data (unlike Logistic Regression's linear structure, a "
              "tree ensemble's split boundaries need not be antisymmetric). This is diagnostic only - **no "
              "probability correction is applied in V1**.\n")
    md.append(f"- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n={len(symmetry_error)} validation rows")
    md.append(f"- mean: {symmetry_stats['mean']:.4f}")
    md.append(f"- median: {symmetry_stats['median']:.4f}")
    md.append(f"- 95th percentile: {symmetry_stats['p95']:.4f}")
    md.append(f"- max: {symmetry_stats['max']:.4f}\n")
    if symmetry_stats["mean"] > 0.05:
        md.append("**Observation**: meaningful asymmetry is present. A later phase could compare this raw "
                  "probability against an explicit symmetrized probability `0.5 * [P(A beats B) + (1 - P(B beats A))]` "
                  "- not implemented here.\n")
    else:
        md.append("**Observation**: asymmetry is small in this run, but is still not exactly zero by construction "
                  "(unlike Logistic Regression) - reported for the record, not corrected.\n")
    (REPORTS / "phases" / "random_forest_symmetry_v1.md").write_text("\n".join(md), encoding="utf-8")


def write_comparison_report(lr_meta, rf_val_metrics, rf_train_metrics):
    lr_val = lr_meta["validation_metrics"]
    lr_train = lr_meta["train_metrics"]
    maj = lr_val["majority_class_accuracy"]  # same validation split for both models

    md = []
    md.append("# Model Comparison: Logistic Regression V1 vs. Random Forest V1 (Validation)\n")
    md.append("Logistic Regression's numbers are read from `models/logistic_regression_scratch_v1.json` "
              "(verified present and complete before use, not re-run or modified). Both models use the exact "
              "same chronological split, the same 17 Phase-3 features, and the same train-only mirrored "
              "augmentation policy.\n")

    md.append("## Validation metrics\n")
    md.append("| metric | majority baseline | Logistic Regression V1 | Random Forest V1 |")
    md.append("|---|---|---|---|")
    md.append(f"| Accuracy | {maj:.4f} | {lr_val['accuracy']:.4f} | {rf_val_metrics['accuracy']:.4f} |")
    md.append(f"| ROC-AUC | 0.5000 | {lr_val['roc_auc']:.4f} | {rf_val_metrics['roc_auc']:.4f} |")
    md.append(f"| F1 | - | {lr_val['f1']:.4f} | {rf_val_metrics['f1']:.4f} |")
    md.append(f"| Log loss | - | {lr_val['log_loss']:.4f} | {rf_val_metrics['log_loss']:.4f} |")
    md.append(f"| Brier score | - | {lr_val['brier_score']:.4f} | {rf_val_metrics['brier_score']:.4f} |\n")

    md.append("## Train-validation gap (overfitting signal)\n")
    md.append("| metric | LR train | LR val | LR gap | RF train | RF val | RF gap |")
    md.append("|---|---|---|---|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("log_loss", "Log loss")]:
        lr_gap = lr_train[key] - lr_val[key]
        rf_gap = rf_train_metrics[key] - rf_val_metrics[key]
        md.append(f"| {label} | {lr_train[key]:.4f} | {lr_val[key]:.4f} | {lr_gap:+.4f} | "
                  f"{rf_train_metrics[key]:.4f} | {rf_val_metrics[key]:.4f} | {rf_gap:+.4f} |")
    md.append("")

    auc_delta = rf_val_metrics["roc_auc"] - lr_val["roc_auc"]
    md.append("## Discussion\n")
    md.append(f"- **Discrimination**: Random Forest's validation ROC-AUC is {rf_val_metrics['roc_auc']:.4f} vs. "
              f"Logistic Regression's {lr_val['roc_auc']:.4f} ({auc_delta:+.4f}). "
              f"{'Random Forest discriminates better' if auc_delta > 0.01 else 'Logistic Regression discriminates better' if auc_delta < -0.01 else 'The two models discriminate about equally'} "
              "on validation.")
    md.append("- **Calibration**: see each model's own calibration diagnostic "
              "(`logistic_regression_calibration_v1.png` / `random_forest_calibration_v1.png`) and Brier/log-loss "
              "above - lower is better-calibrated/sharper.")
    md.append("- **Train-validation gap**: see the table above; a large Random Forest gap (if present) reflects "
              "the untuned `max_depth=None`/`min_samples_leaf=1` baseline configuration overfitting relative to "
              "Logistic Regression's much smaller gap, which is expected of an unregularized deep forest and is "
              "not treated as a bug in this phase.")
    md.append("- **Confidence/extremeness of probabilities**: see `random_forest_probability_distribution_v1.png` "
              "vs. Logistic Regression's narrower/wider spread (Phase 4A report) for a visual comparison.")
    md.append("- **Interpretability**: Logistic Regression's coefficients (Phase 4A) give a single global, signed "
              "direction per feature; Random Forest's impurity/permutation importances "
              "(`random_forest_feature_importance_v1.csv` / `random_forest_permutation_importance_v1.csv`) only "
              "rank feature usefulness and cannot express direction or interactions directly - trees can, "
              "importances alone don't show them.\n")
    md.append("**No final project model is declared here.** XGBoost has not yet been evaluated, and the internal "
              "test set has not been used for this or any comparison.\n")

    (REPORTS / "phases" / "model_comparison_lr_vs_rf_v1.md").write_text("\n".join(md), encoding="utf-8")


def write_phase4b_report(train_raw, val_raw, augmented_train_raw, feature_names, params,
                          train_metrics, val_metrics, importance_df, perm_df, cal_df,
                          proba_percentiles, symmetry_stats, model_meta, lr_meta):
    lr_val = lr_meta["validation_metrics"]
    md = []
    md.append("# Phase 4B - Model 2: Random Forest Classifier V1\n")

    md.append("## 1. Why Random Forest is Model 2\n")
    md.append("Model 2 in the project proposal's model lineup, testing whether a nonlinear tree ensemble improves "
              "over the linear Logistic Regression baseline (Model 1) when given exactly the same leakage-safe "
              "historical information.\n")

    md.append("## 2. Why a tree ensemble may improve over linear Logistic Regression\n")
    md.append("Logistic Regression can only combine features additively/linearly (in the standardized feature "
              "space). A Random Forest can learn interactions and thresholds automatically, e.g. `elo_diff` "
              "mattering differently depending on `bestOf`, or on whether both teams have enough history to trust "
              "their historical stats - relationships Logistic Regression cannot represent without explicit "
              "interaction terms.\n")

    md.append("## 3. Exact fixed V1 configuration\n")
    md.append("`sklearn.ensemble.RandomForestClassifier` is explicitly permitted for this model (not required "
              "from scratch, unlike Model 1). Single fixed baseline, **no tuning**:\n")
    md.append("```\nn_estimators=300, criterion='gini', max_depth=None, min_samples_split=2, "
              "min_samples_leaf=1, max_features='sqrt', bootstrap=True, class_weight=None, "
              "random_state=42, n_jobs=-1\n```\n")

    md.append("## 4. Reuse of the identical chronological split\n")
    md.append(f"`data/modeling/series_split_v1.csv` from Phase 4A is reused byte-for-byte, **not regenerated**: "
              f"train={len(train_raw):,} unique historical matches, validation={len(val_raw):,}, "
              "test=1,418 (sealed, not loaded as a dataframe anywhere in this phase). This guarantees Logistic "
              "Regression and Random Forest are compared on exactly the same historical matches.\n")

    md.append("## 5. Mirroring methodology\n")
    md.append(f"Identical raw-mirroring policy to Logistic Regression (shared implementation in "
              f"`feature_engineering/preprocessing/preprocessing_common.py`): {len(train_raw):,} unique historical training matches are "
              f"each mirrored once (directional diffs negated, symmetric/context columns unchanged, target "
              f"flipped) and concatenated with the originals, producing **{len(augmented_train_raw):,} training "
              f"observations** - important to say precisely: this is {len(augmented_train_raw):,} augmented "
              f"*observations* fed to the model, not {len(augmented_train_raw):,} independent matches; there are "
              f"still only {len(train_raw):,} unique underlying historical matches in the training partition. "
              f"Augmented target mean: **{model_meta['mirrored_train_target_mean']}** (exactly 0.5). Validation "
              "and test are never mirrored.\n")

    md.append("## 6. Random-Forest-specific preprocessing\n")
    md.append("Same 17 raw Phase-3 whitelist features, same `bestOf`/`tier` reference-dummy encoding as Logistic "
              "Regression, but **no standardization** - tree splits are invariant to monotonic per-feature "
              "rescaling. Only train medians (fit on the augmented training set only) are used, for imputing "
              f"`days_since_last_match_diff`'s missingness. {len(feature_names)} transformed features (same "
              "names/order as Logistic Regression's 19). Saved to "
              "`data/modeling/random_forest_preprocessing_v1.json`.\n")

    def metrics_lines(m):
        cm = m["confusion_matrix"]
        return "\n".join([
            f"- n = {m['n']:,}",
            f"- Accuracy: {m['accuracy']:.4f} (majority-class reference: {m['majority_class_accuracy']:.4f})",
            f"- Precision: {m['precision']:.4f}",
            f"- Recall: {m['recall']:.4f}",
            f"- F1: {m['f1']:.4f}",
            f"- ROC-AUC: {m['roc_auc']:.4f}",
            f"- Log loss: {m['log_loss']:.4f}",
            f"- Brier score: {m['brier_score']:.4f}",
            f"- Confusion matrix [[TN,FP],[FN,TP]]: {cm}",
        ])

    md.append("## 7. Train metrics (unmirrored original orientation)\n")
    md.append(metrics_lines(train_metrics) + "\n")

    md.append("## 8. Validation metrics\n")
    md.append(metrics_lines(val_metrics) + "\n")

    acc_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    auc_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]
    ll_gap = val_metrics["log_loss"] - train_metrics["log_loss"]
    brier_gap = val_metrics["brier_score"] - train_metrics["brier_score"]
    md.append("## 9. Train-validation overfitting analysis\n")
    md.append(f"- Accuracy gap (train - val): **{acc_gap:+.4f}**")
    md.append(f"- ROC-AUC gap (train - val): **{auc_gap:+.4f}**")
    md.append(f"- Log loss gap (val - train): **{ll_gap:+.4f}**")
    md.append(f"- Brier gap (val - train): **{brier_gap:+.4f}**")
    md.append(f"\nThe fixed baseline uses `max_depth=None, min_samples_leaf=1` - unrestricted trees that can grow "
              "until every leaf is (near-)pure, which can memorize the training set. "
              f"{'A large gap here is expected and is evidence of overfitting in this untuned baseline, documented as observed behavior' if acc_gap > 0.1 or auc_gap > 0.1 else 'The observed gap is documented as-is'} "
              "- **this is not treated as a bug, and V1 does not react by changing `max_depth`/`min_samples_leaf`/"
              "etc.; that is explicitly deferred to a future tuning phase.**\n")

    md.append("## 10. ROC interpretation\n")
    md.append(f"`reports/figures/random_forest_roc_v1.png` (validation only). ROC-AUC = {val_metrics['roc_auc']:.4f} "
              f"vs. chance = 0.5 and Logistic Regression's {lr_val['roc_auc']:.4f}.\n")

    md.append("## 11. Calibration interpretation\n")
    md.append("`reports/figures/random_forest_calibration_v1.png`, diagnostic only - no isotonic/Platt/temperature "
              "correction applied. Per-bin mean predicted probability vs. empirical win rate:\n")
    md.append("| bin | n | mean predicted | empirical win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['empirical_win_rate']:.3f} |")
    md.append(f"\nValidation Brier score: {val_metrics['brier_score']:.4f}. Validation log loss: "
              f"{val_metrics['log_loss']:.4f}.\n")

    md.append("## 12. Probability-distribution interpretation\n")
    md.append("`reports/figures/random_forest_probability_distribution_v1.png` (validation only). Percentiles of "
              "predicted P(team1 wins):\n")
    md.append("| min | p5 | p25 | median | p75 | p95 | max |")
    md.append("|---|---|---|---|---|---|---|")
    p = proba_percentiles
    md.append(f"| {p[0]:.3f} | {p[5]:.3f} | {p[25]:.3f} | {p[50]:.3f} | {p[75]:.3f} | {p[95]:.3f} | {p[100]:.3f} |\n")

    md.append("## 13. Feature importance (impurity)\n")
    md.append("`reports/tables/random_forest_feature_importance_v1.csv`. **Impurity importance is NOT causal "
              "importance and may be biased toward variables offering many possible split points** (e.g. "
              "high-cardinality continuous features over binary flags). Top 8:\n")
    md.append("| rank | feature | impurity_importance |")
    md.append("|---|---|---|")
    for _, r in importance_df.head(8).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['impurity_importance']:.4f} |")
    md.append("")

    md.append("## 14. Permutation importance (validation, ROC-AUC)\n")
    md.append("`reports/tables/random_forest_permutation_importance_v1.csv` "
              "(`sklearn.inspection.permutation_importance`, `scoring='roc_auc'`, `n_repeats=10`, "
              "`random_state=42`). Top 8:\n")
    md.append("| rank | feature | mean_importance | std_importance |")
    md.append("|---|---|---|---|")
    for _, r in perm_df.head(8).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['mean_importance']:.4f} | {r['std_importance']:.4f} |")
    md.append(f"\nUnlike Logistic Regression, where symmetric/context features (`bestOf`, `tier`, history "
              "confidence flags) received essentially zero standalone coefficients (mirroring mathematically "
              "cancels their additive effect), a tree ensemble can use these through interactions (e.g. "
              "`bestOf==BO3 AND elo_diff>threshold`) - so they may show non-zero importance here even though "
              "they carry no independent directional signal on their own. No features are removed based on "
              "these importances; this is interpretation only.\n")

    md.append("## 15. Side-symmetry diagnostic\n")
    md.append(f"Full detail in `reports/random_forest_symmetry_v1.md`. Summary: mean symmetry error "
              f"**{symmetry_stats['mean']:.4f}**, median **{symmetry_stats['median']:.4f}**, p95 "
              f"**{symmetry_stats['p95']:.4f}**, max **{symmetry_stats['max']:.4f}** "
              "(`abs(P(A beats B) - (1 - P(B beats A)))` on validation, using the same fitted preprocessing "
              "artifact for both orientations). Diagnostic only - no correction applied in V1.\n")

    md.append("## 16. Comparison to Logistic Regression V1\n")
    md.append("Full detail in `reports/model_comparison_lr_vs_rf_v1.md`. Headline: validation ROC-AUC "
              f"{val_metrics['roc_auc']:.4f} (RF) vs. {lr_val['roc_auc']:.4f} (LR), accuracy "
              f"{val_metrics['accuracy']:.4f} (RF) vs. {lr_val['accuracy']:.4f} (LR).\n")

    auc_delta = val_metrics["roc_auc"] - lr_val["roc_auc"]
    md.append("## Interpretation question: did Random Forest gain anything from nonlinear relationships/interactions?\n")
    if auc_delta > 0.01:
        verdict = (f"**Yes, on this validation evidence** - Random Forest's ROC-AUC exceeds Logistic Regression's "
                   f"by {auc_delta:+.4f}, consistent with the forest exploiting interactions/nonlinearities the "
                   "linear model cannot represent.")
    elif auc_delta < -0.01:
        verdict = (f"**Not on this validation evidence** - Random Forest's ROC-AUC is {auc_delta:+.4f} relative "
                   "to Logistic Regression's, i.e. worse; any nonlinear capacity gained was offset by the "
                   "untuned baseline's overfitting (Section 9) rather than showing up as better validation "
                   "discrimination.")
    else:
        verdict = (f"**Inconclusive on this validation evidence** - the ROC-AUC difference ({auc_delta:+.4f}) is "
                   "too small to confidently attribute to real nonlinear/interaction gains rather than noise.")
    md.append(verdict + " This conclusion is based only on the validation evidence above (ROC-AUC/accuracy "
              "delta, calibration, feature importances touching `bestOf`/`tier`/history-confidence, and the "
              "train-validation gap) - it is not assumed a priori.\n")

    md.append("## 17. Limitations\n")
    md.append("- Untuned baseline: `max_depth=None`/`min_samples_leaf=1` likely overfits (Section 9) - not fixed here.")
    md.append("- Impurity importance is biased toward high-cardinality continuous features; permutation importance "
              "is the more trustworthy of the two but is still only a ranking, not a causal statement.")
    md.append("- Side-symmetry is not exact (Section 15) - relevant for any future real-match prediction use.")
    md.append("- Same 17 series-level features as Logistic Regression - no map/player-level detail.")
    md.append("- Single chronological train/validation split - no cross-validation.")
    md.append("- `n_jobs=-1` and sklearn's internal tie-breaking mean results are reproducible on this machine "
              "given `random_state=42`, but bitwise reproducibility across different sklearn/BLAS versions or "
              "hardware is not guaranteed.\n")

    md.append("## 18. What is deferred to tuning\n")
    md.append("`n_estimators`, `max_depth`, `min_samples_split`/`min_samples_leaf`, `max_features`, "
              "`class_weight`, and `criterion` search; probability calibration (isotonic/Platt/temperature); "
              "symmetrized-probability correction; feature selection based on importances.\n")

    md.append("## Status\n")
    md.append("- **INTERNAL TEST = SEALED** - not opened or scored in this phase.")
    md.append("- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.")
    md.append("- **XGBOOST = NOT STARTED**.")
    md.append("- **NO RANDOM FOREST HYPERPARAMETER TUNING PERFORMED**.\n")

    (REPORTS / "phases" / "phase4b_random_forest_v1.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
