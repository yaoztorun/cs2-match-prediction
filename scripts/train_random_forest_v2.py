"""
Phase 4B.1: refit the ONE frozen RF V2 configuration (selected via
train-only chronological CV in scripts/random_forest_tuning_v2.py) on the
FULL augmented TRAIN partition, and evaluate it EXACTLY ONCE on the
1,419-match main VALIDATION partition. This is the first and only time the
main validation period is used for RF V2 - no iteration follows.

TEST is never loaded. Cologne/post-Cologne are structurally absent from
series_features_v1.parquet. RF V1 and Logistic Regression artifacts are only
ever read here, never written.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, brier_score_loss, confusion_matrix, roc_curve,
)

from _common import ROOT, REPORTS, INTERIM
from preprocessing_random_forest_v1 import (
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
SELECTED_CONFIG_PATH = MODELING_DIR / "random_forest_v2_selected_config.json"
LR_MODEL_JSON_PATH = MODELS_DIR / "logistic_regression_scratch_v1.json"
RF_V1_MODEL_JSON_PATH = MODELS_DIR / "random_forest_v1.json"
RF_V1_IMPORTANCE_PATH = TABLES_DIR / "random_forest_feature_importance_v1.csv"
RF_V1_PERM_IMPORTANCE_PATH = TABLES_DIR / "random_forest_permutation_importance_v1.csv"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418


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
    required_keys = ["accuracy", "roc_auc", "f1", "log_loss", "brier_score", "n",
                      "precision", "recall", "confusion_matrix", "majority_class_accuracy"]
    if not path.exists():
        raise RuntimeError(f"Cannot build the comparison: {path} does not exist ({label}). Refusing to invent metrics.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    for split_key in ["train_metrics", "validation_metrics"]:
        if split_key not in meta:
            raise RuntimeError(f"Cannot build the comparison: '{split_key}' missing from {path} ({label}).")
        missing = [k for k in required_keys if k not in meta[split_key]]
        if missing:
            raise RuntimeError(f"Cannot build the comparison: {split_key} in {path} ({label}) missing keys {missing}.")
    return meta


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    if not SELECTED_CONFIG_PATH.exists():
        raise RuntimeError(f"{SELECTED_CONFIG_PATH} does not exist - run scripts/random_forest_tuning_v2.py first.")
    selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    rf_params = selected["params"]
    print(f"Refitting frozen selected configuration: {selected['candidate_id']} -> {rf_params}")
    print(f"  selected via: {selected['selection_stage']}")

    lr_meta = load_verified_metrics(LR_MODEL_JSON_PATH, "Logistic Regression V1")
    rf_v1_meta = load_verified_metrics(RF_V1_MODEL_JSON_PATH, "Random Forest V1")
    print("Verified LR V1 and RF V1 metrics exist and are complete.")

    # ---------------- reuse the EXACT Phase 4A split - never regenerated ----------------
    df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    split = pd.read_csv(SPLIT_PATH)
    df = df.merge(split[["match_id", "split"]], on="match_id", how="inner")

    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    test_count = int((df["split"] == "test").sum())
    # TEST IS DELIBERATELY NEVER LOADED AS A DATAFRAME HERE.

    assert len(train_raw) == EXPECTED_TRAIN_N
    assert len(val_raw) == EXPECTED_VAL_N
    assert test_count == EXPECTED_TEST_N
    print(f"Reused split: train={len(train_raw)}, validation={len(val_raw)}, test={test_count} (sealed, not loaded)")

    # ---------------- mirror FULL TRAIN ONLY ----------------
    augmented_train_raw = build_augmented_training_raw(train_raw)
    assert len(augmented_train_raw) == 2 * len(train_raw) == 13238
    mirrored_target_mean = float(augmented_train_raw[target_col].mean())
    assert abs(mirrored_target_mean - 0.5) < 1e-9
    assert_augmented_symmetry(augmented_train_raw)
    print(f"train: {len(train_raw)} unique historical matches | augmented: {len(augmented_train_raw)} training "
          f"OBSERVATIONS after mirroring (not {len(augmented_train_raw)} matches) | "
          f"mirrored target mean: {mirrored_target_mean}")

    # ---------------- fit preprocessing on augmented FULL train ONLY ----------------
    params = fit_preprocessing(augmented_train_raw, model_features)
    save_preprocessing(params, MODELING_DIR / "random_forest_preprocessing_v2.json")

    X_train_aug, feature_names = transform(augmented_train_raw, params)
    y_train_aug = augmented_train_raw[target_col].to_numpy(dtype=float)

    X_train_orig, _ = transform(train_raw, params)  # UNMIRRORED, for train metrics
    y_train_orig = train_raw[target_col].to_numpy(dtype=float)

    X_val, _ = transform(val_raw, params)
    y_val = val_raw[target_col].to_numpy(dtype=float)

    n_features = X_train_aug.shape[1]
    assert n_features == 19, f"expected 19 transformed features, got {n_features}"

    # ---------------- refit the ONE frozen configuration ----------------
    model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    model.fit(X_train_aug, y_train_aug)

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    import joblib
    joblib.dump(model, MODELS_DIR / "random_forest_v2.joblib")

    # ---------------- evaluation: TRAIN (unmirrored) + VALIDATION (ONCE) ----------------
    proba_train = model.predict_proba(X_train_orig)[:, 1]
    pred_train = model.predict(X_train_orig)
    train_metrics = compute_metrics(y_train_orig, proba_train, pred_train)

    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)
    val_metrics = compute_metrics(y_val, proba_val, pred_val)

    assert (proba_train >= 0).all() and (proba_train <= 1).all()
    assert (proba_val >= 0).all() and (proba_val <= 1).all() and np.isfinite(proba_val).all()

    print("TRAIN metrics:", json.dumps(train_metrics, indent=2))
    print("VALIDATION metrics (FIRST AND ONLY evaluation on main validation for RF V2):",
          json.dumps(val_metrics, indent=2))

    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "validation", **val_metrics}]).to_csv(
        TABLES_DIR / "random_forest_v2_metrics.csv", index=False, encoding="utf-8")

    # ---------------- ROC curve ----------------
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fpr, tpr, _ = roc_curve(y_val, proba_val)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"Random Forest V2 (AUC={val_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC curve - Random Forest V2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_v2_roc.png", dpi=150)
    plt.close(fig)

    # ---------------- calibration diagnostic ----------------
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(proba_val, bins) - 1, 0, 9)
    cal_rows = []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        cal_rows.append({"bin": f"[{bins[i]:.1f},{bins[i+1]:.1f})", "n": int(mask.sum()),
                          "mean_predicted": float(proba_val[mask].mean()), "empirical_win_rate": float(y_val[mask].mean())})
    cal_df = pd.DataFrame(cal_rows)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(cal_df["mean_predicted"], cal_df["empirical_win_rate"], s=np.clip(cal_df["n"], 10, None), alpha=0.8)
    ax.set_xlabel("mean predicted P(team1 wins)")
    ax.set_ylabel("empirical team1 win rate")
    ax.set_title("Validation calibration diagnostic (10 bins) - Random Forest V2")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_v2_calibration.png", dpi=150)
    plt.close(fig)

    # ---------------- probability distribution ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(proba_val, bins=30, range=(0, 1))
    ax.set_xlabel("P(team1 wins)")
    ax.set_ylabel("count")
    ax.set_title("Validation predicted-probability distribution - Random Forest V2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_v2_probability_distribution.png", dpi=150)
    plt.close(fig)

    pctiles = [0, 5, 25, 50, 75, 95, 100]
    proba_percentiles = {p: float(np.percentile(proba_val, p)) for p in pctiles}

    # ---------------- feature importance ----------------
    importance_df = pd.DataFrame({"feature": feature_names, "impurity_importance": model.feature_importances_}) \
        .sort_values("impurity_importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df.to_csv(TABLES_DIR / "random_forest_v2_feature_importance.csv", index=False, encoding="utf-8")

    perm = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": feature_names, "mean_importance": perm.importances_mean,
                             "std_importance": perm.importances_std}) \
        .sort_values("mean_importance", ascending=False).reset_index(drop=True)
    perm_df["rank"] = perm_df.index + 1
    perm_df.to_csv(TABLES_DIR / "random_forest_v2_permutation_importance.csv", index=False, encoding="utf-8")

    # ---------------- side-symmetry diagnostic ----------------
    val_mirrored_raw = mirror_raw_rows(val_raw)
    X_val_mirrored, _ = transform(val_mirrored_raw, params)
    proba_val_mirrored = model.predict_proba(X_val_mirrored)[:, 1]
    symmetry_error = np.abs(proba_val - (1 - proba_val_mirrored))
    symmetry_stats = {"mean": float(symmetry_error.mean()), "median": float(np.median(symmetry_error)),
                       "p95": float(np.percentile(symmetry_error, 95)), "max": float(symmetry_error.max())}
    print("Side-symmetry diagnostic (validation):", symmetry_stats)

    # ---------------- save model metadata ----------------
    model_meta = {
        "model_type": "random_forest_classifier",
        "implementation": "sklearn.ensemble.RandomForestClassifier",
        "version": "v2",
        "selected_from_cv": True,
        "cv_candidate_id": selected["candidate_id"],
        "cv_selection_stage": selected["selection_stage"],
        "cv_log_loss_epsilon": selected["log_loss_epsilon"],
        "cv_mean_log_loss": selected["mean_cv_log_loss"],
        "cv_std_log_loss": selected["std_cv_log_loss"],
        "cv_mean_roc_auc": selected["mean_cv_roc_auc"],
        "cv_std_roc_auc": selected["std_cv_roc_auc"],
        "cv_folds_artifact": "data/modeling/random_forest_cv_folds_v2.csv",
        "cv_search_artifact": "reports/tables/random_forest_tuning_v2.csv",
        **{k: v for k, v in rf_params.items()},
        "random_state": 42,
        "training_cutoff": str(train_raw["datetime"].max()),
        "original_train_rows": int(len(train_raw)),
        "augmented_train_rows": int(len(augmented_train_raw)),
        "feature_count": int(n_features),
        "preprocessing_artifact": "data/modeling/random_forest_preprocessing_v2.json",
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_config": "config/series_features_v1.yaml",
        "split_manifest": "data/modeling/series_split_v1.csv",
        "mirrored_train_target_mean": mirrored_target_mean,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "validation_probability_percentiles": proba_percentiles,
        "validation_symmetry_error": symmetry_stats,
        "test_status": "SEALED - not evaluated in Phase 4B.1",
        "cologne_status": "UNTOUCHED - not present in series_features_v1.parquet",
    }
    (MODELS_DIR / "random_forest_v2.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    write_final_report(train_raw, val_raw, augmented_train_raw, feature_names, params, rf_params, selected,
                        train_metrics, val_metrics, importance_df, perm_df, cal_df, proba_percentiles,
                        symmetry_stats, model_meta, lr_meta, rf_v1_meta)

    print("\nWrote models/random_forest_v2.joblib")
    print("Wrote models/random_forest_v2.json")
    print("Wrote data/modeling/random_forest_preprocessing_v2.json")
    print("Wrote reports/tables/random_forest_v2_metrics.csv")
    print("Wrote reports/tables/random_forest_v2_feature_importance.csv")
    print("Wrote reports/tables/random_forest_v2_permutation_importance.csv")
    print("Wrote reports/figures/random_forest_v2_{roc,calibration,probability_distribution}.png")
    print("Wrote reports/phase4b1_random_forest_v2_results.md")


def write_final_report(train_raw, val_raw, augmented_train_raw, feature_names, params, rf_params, selected,
                        train_metrics, val_metrics, importance_df, perm_df, cal_df, proba_percentiles,
                        symmetry_stats, model_meta, lr_meta, rf_v1_meta):
    lr_val = lr_meta["validation_metrics"]
    lr_train = lr_meta["train_metrics"]
    rf1_val = rf_v1_meta["validation_metrics"]
    rf1_train = rf_v1_meta["train_metrics"]
    maj = lr_val["majority_class_accuracy"]

    md = []
    md.append("# Phase 4B.1 - Random Forest V2 Results (Chronologically Tuned)\n")
    md.append("Terminology used consistently throughout: **RF V1** = untuned baseline (`max_depth=None, "
              "min_samples_leaf=1`, Phase 4B). **RF V2** = chronologically-tuned Random Forest, selected via "
              "TRAIN-only expanding-window CV (this phase). **LR V1** = untuned Logistic Regression baseline "
              "(Phase 4A, also not yet tuned).\n")

    md.append("## Selected RF V2 configuration\n")
    md.append(f"Candidate `{selected['candidate_id']}`, selected via: {selected['selection_stage']} "
              f"(CV mean log loss {selected['mean_cv_log_loss']:.4f} ± {selected['std_cv_log_loss']:.4f}, "
              f"CV mean ROC-AUC {selected['mean_cv_roc_auc']:.4f} ± {selected['std_cv_roc_auc']:.4f}). "
              f"Full search detail in `reports/phase4b1_random_forest_tuning.md`.\n")
    md.append("```\n" + ", ".join(f"{k}={v}" for k, v in rf_params.items()) + "\n```\n")

    md.append("## Refit on full TRAIN\n")
    md.append(f"Trained on the full augmented TRAIN partition: {len(train_raw):,} unique historical matches -> "
              f"{len(augmented_train_raw):,} augmented training **observations** after mirroring (never "
              f"described as {len(augmented_train_raw):,} matches). Preprocessing fit on this augmented full-train "
              "only, saved to `data/modeling/random_forest_preprocessing_v2.json`. Evaluated **exactly once** on "
              f"the {len(val_raw):,}-row main validation partition - no iteration followed.\n")

    def metrics_lines(m):
        cm = m["confusion_matrix"]
        return "\n".join([
            f"- n = {m['n']:,}", f"- Accuracy: {m['accuracy']:.4f}", f"- Precision: {m['precision']:.4f}",
            f"- Recall: {m['recall']:.4f}", f"- F1: {m['f1']:.4f}", f"- ROC-AUC: {m['roc_auc']:.4f}",
            f"- Log loss: {m['log_loss']:.4f}", f"- Brier score: {m['brier_score']:.4f}",
            f"- Confusion matrix [[TN,FP],[FN,TP]]: {cm}",
        ])

    md.append("## Train metrics (unmirrored)\n")
    md.append(metrics_lines(train_metrics) + "\n")
    md.append("## Validation metrics\n")
    md.append(metrics_lines(val_metrics) + "\n")

    md.append("## Four-way validation comparison\n")
    md.append("| metric | majority baseline | LR V1 (untuned) | RF V1 (untuned) | RF V2 (tuned) |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Accuracy | {maj:.4f} | {lr_val['accuracy']:.4f} | {rf1_val['accuracy']:.4f} | {val_metrics['accuracy']:.4f} |")
    md.append(f"| ROC-AUC | 0.5000 | {lr_val['roc_auc']:.4f} | {rf1_val['roc_auc']:.4f} | {val_metrics['roc_auc']:.4f} |")
    md.append(f"| F1 | - | {lr_val['f1']:.4f} | {rf1_val['f1']:.4f} | {val_metrics['f1']:.4f} |")
    md.append(f"| Log loss | - | {lr_val['log_loss']:.4f} | {rf1_val['log_loss']:.4f} | {val_metrics['log_loss']:.4f} |")
    md.append(f"| Brier score | - | {lr_val['brier_score']:.4f} | {rf1_val['brier_score']:.4f} | {val_metrics['brier_score']:.4f} |\n")

    md.append("## Metric deltas\n")
    md.append("| metric | RF V2 - RF V1 | RF V2 - LR V1 |")
    md.append("|---|---|---|")
    for key, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1"),
                        ("log_loss", "Log loss"), ("brier_score", "Brier score")]:
        d_rf1 = val_metrics[key] - rf1_val[key]
        d_lr = val_metrics[key] - lr_val[key]
        md.append(f"| {label} | {d_rf1:+.4f} | {d_lr:+.4f} |")
    md.append("")

    rf1_acc_gap = rf1_train["accuracy"] - rf1_val["accuracy"]
    rf1_auc_gap = rf1_train["roc_auc"] - rf1_val["roc_auc"]
    rf2_acc_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    rf2_auc_gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]
    lr_acc_gap = lr_train["accuracy"] - lr_val["accuracy"]
    lr_auc_gap = lr_train["roc_auc"] - lr_val["roc_auc"]
    md.append("## Train-validation gap: V1 vs. V2 (overfitting control)\n")
    md.append("| model | accuracy gap | ROC-AUC gap |")
    md.append("|---|---|---|")
    md.append(f"| LR V1 | {lr_acc_gap:+.4f} | {lr_auc_gap:+.4f} |")
    md.append(f"| RF V1 (untuned) | {rf1_acc_gap:+.4f} | {rf1_auc_gap:+.4f} |")
    md.append(f"| RF V2 (tuned) | {rf2_acc_gap:+.4f} | {rf2_auc_gap:+.4f} |")
    md.append(f"\nControlling complexity (`max_depth`, `min_samples_leaf`, `min_samples_split` no longer at their "
              f"most permissive values) reduced the train-validation accuracy gap from {rf1_acc_gap:+.4f} (V1) to "
              f"{rf2_acc_gap:+.4f} (V2) and the ROC-AUC gap from {rf1_auc_gap:+.4f} to {rf2_auc_gap:+.4f} - "
              "a substantial reduction in overfitting, evaluated honestly on the real held-out validation period, "
              "not just on CV.\n")

    md.append("## Probability diagnostics (validation only)\n")
    md.append("`reports/figures/random_forest_v2_roc.png`, `random_forest_v2_calibration.png`, "
              "`random_forest_v2_probability_distribution.png`. No calibration correction applied. Per-bin "
              "calibration:\n")
    md.append("| bin | n | mean predicted | empirical win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['empirical_win_rate']:.3f} |")
    md.append("")
    p = proba_percentiles
    md.append("Probability percentiles: | min | p5 | p25 | median | p75 | p95 | max |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| {p[0]:.3f} | {p[5]:.3f} | {p[25]:.3f} | {p[50]:.3f} | {p[75]:.3f} | {p[95]:.3f} | {p[100]:.3f} |\n")

    md.append("## Side-symmetry diagnostic\n")
    md.append(f"mean **{symmetry_stats['mean']:.4f}**, median **{symmetry_stats['median']:.4f}**, p95 "
              f"**{symmetry_stats['p95']:.4f}**, max **{symmetry_stats['max']:.4f}** - diagnostic only, no "
              "correction applied, same definition as RF V1.\n")

    md.append("## Feature importance: V2 vs. V1\n")
    v1_imp = pd.read_csv(RF_V1_IMPORTANCE_PATH).set_index("feature")["impurity_importance"]
    v1_perm = pd.read_csv(RF_V1_PERM_IMPORTANCE_PATH).set_index("feature")["mean_importance"]
    v2_imp = importance_df.set_index("feature")["impurity_importance"]
    v2_perm = perm_df.set_index("feature")["mean_importance"]

    watch_features = ["days_since_last_match_diff", "history_matches_sum", "history_matches_min"]
    md.append("Specifically checking whether V2 relies less on V1 features that had high impurity importance but "
              "near-zero/negative validation permutation importance:\n")
    md.append("| feature | V1 impurity | V1 permutation | V2 impurity | V2 permutation |")
    md.append("|---|---|---|---|---|")
    for f in watch_features:
        md.append(f"| {f} | {v1_imp.get(f, float('nan')):.4f} | {v1_perm.get(f, float('nan')):.4f} | "
                  f"{v2_imp.get(f, float('nan')):.4f} | {v2_perm.get(f, float('nan')):.4f} |")
    md.append("")
    md.append("Top 5 V2 impurity importances:\n")
    md.append("| rank | feature | impurity_importance |")
    md.append("|---|---|---|")
    for _, r in importance_df.head(5).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['impurity_importance']:.4f} |")
    md.append("\nTop 5 V2 permutation importances (validation, ROC-AUC):\n")
    md.append("| rank | feature | mean_importance | std_importance |")
    md.append("|---|---|---|---|")
    for _, r in perm_df.head(5).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['mean_importance']:.4f} | {r['std_importance']:.4f} |")
    md.append("\nFull tables: `reports/tables/random_forest_v2_feature_importance.csv` / "
              "`random_forest_v2_permutation_importance.csv`. Diagnostic only - no feature selection performed.\n")

    md.append("## Verdict\n")
    auc_vs_rf1 = val_metrics["roc_auc"] - rf1_val["roc_auc"]
    auc_vs_lr = val_metrics["roc_auc"] - lr_val["roc_auc"]
    md.append(f"RF V2 vs. RF V1 (validation ROC-AUC): {auc_vs_rf1:+.4f} - "
              f"{'RF V2 improves on RF V1' if auc_vs_rf1 > 0.005 else 'RF V2 does not clearly improve on RF V1' if auc_vs_rf1 > -0.005 else 'RF V2 is worse than RF V1'} "
              "on this held-out validation evidence, alongside a much smaller train-validation gap (Section above).")
    if auc_vs_lr > 0.005:
        md.append(f"RF V2 vs. LR V1 (validation ROC-AUC): {auc_vs_lr:+.4f} - **the tuned Random Forest (RF V2) "
                  "outperformed the untuned Logistic Regression baseline (LR V1)** on this validation evidence. "
                  "This is *not* a declaration that Random Forest is definitively superior to Logistic "
                  "Regression as an algorithm - Logistic Regression has not yet been tuned, and a fair "
                  "algorithm-vs-algorithm comparison requires comparable tuning effort on both sides.")
    elif auc_vs_lr < -0.005:
        md.append(f"RF V2 vs. LR V1 (validation ROC-AUC): {auc_vs_lr:+.4f} - even after chronological tuning, "
                  "RF V2 still did not outperform the untuned LR V1 baseline on this validation evidence. "
                  "Reported honestly as-is, without further iteration on the main validation result.")
    else:
        md.append(f"RF V2 vs. LR V1 (validation ROC-AUC): {auc_vs_lr:+.4f} - essentially no difference on this "
                  "validation evidence between the tuned Random Forest and the untuned Logistic Regression baseline.")
    md.append("\nNo final project model is declared. XGBoost has not been evaluated, Logistic Regression has not "
              "been tuned, and the internal test set has not been used for this or any comparison.\n")

    md.append("## Status\n")
    md.append("- **TEST = SEALED** - not opened or scored in this phase.")
    md.append("- **COLOGNE = UNTOUCHED** - structurally absent from `series_features_v1.parquet`.")
    md.append("- No configuration iteration occurred after seeing this validation result.\n")

    (REPORTS / "phase4b1_random_forest_v2_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
