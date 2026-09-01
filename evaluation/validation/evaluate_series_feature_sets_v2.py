"""
Phase 5B.1: Paired V1 vs V2 map-pool feature-set evaluation.

Question answered: with RF V2's and XGB V2's FROZEN hyperparameters held
fixed (loaded from data/modeling/random_forest_v2_selected_config.json and
data/modeling/xgboost_v2_selected_config.json - never tuned here), do the 30
new leakage-safe map-pool features (series_features_v2_map_pool.parquet)
improve series-winner prediction relative to the original 17 Phase-3
features (series_features_v1.parquet)?

FRAMING (read before interpreting the numbers below)
------------------------------------------------------------------------
The 4 chronological CV folds used here (data/modeling/random_forest_cv_folds_v2.csv)
are the SAME folds RF V2 and XGB V2 hyperparameters were originally selected
against. This script is therefore a PAIRED FEATURE ABLATION, not a fresh,
unbiased estimate of future generalization: it answers "with the frozen
model configuration held fixed, do V2 features improve performance relative
to V1 features on the same development folds used during tuning?" - a valid
and useful question, but the resulting CV metrics must NOT be read as an
independent estimate of held-out performance. The main 1,419-match
VALIDATION partition is never loaded here and remains untouched.

XGBoost methodology note: the original xgboost_tuning_v2.py used an inner
early-stopping split with fold-specific best_iteration. This script
deliberately does NOT reproduce that - it uses the single FROZEN final
configuration (n_estimators=98, no early stopping, full outer-fold training
history) for BOTH the V1 and V2 arms, because holding the model configuration
IDENTICAL across feature sets is what makes the comparison paired and
feature-attributable. XGB-V1 numbers here are therefore not expected to
match reports/tables/xgboost_tuning_v2.csv row-for-row - that is by design,
not an implementation error. RF has no early-stopping distinction, so RF-V1
here is expected to closely reproduce random_forest_tuning_v2.csv's frozen
candidate row.

Never touches: TEST, Cologne/post-Cologne, the 1,419-match main VALIDATION
partition, or any Phase 4 / Phase 5A artifact (read-only against all of them).

Writes:
    reports/tables/series_feature_v1_v2_cv_comparison.csv
    reports/tables/series_v2_feature_importance_rf.csv
    reports/tables/series_v2_feature_importance_xgb.csv
    reports/figures/series_feature_v1_v2_rf_cv.png
    reports/figures/series_feature_v1_v2_xgb_cv.png
    reports/phase5b1_series_map_pool_cv_results.md
    data/modeling/phase5b1_fold_preprocessing_audit.json   (for validation/validate_phase5b1.py)
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

from _common import ROOT, REPORTS

import feature_engineering.preprocessing.preprocessing_common as pc1
import feature_engineering.preprocessing.preprocessing_random_forest_v1 as rf1
import feature_engineering.preprocessing.preprocessing_xgboost_v1 as xgb1
import feature_engineering.preprocessing.preprocessing_common_v2_map_pool as pc2
import feature_engineering.preprocessing.preprocessing_random_forest_v2_map_pool as rf2
import feature_engineering.preprocessing.preprocessing_xgboost_v2_map_pool as xgb2

CONFIG_V1_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
CONFIG_V2_PATH = ROOT / "config" / "features" / "series_features_v2_map_pool.yaml"
FEATURES_V1_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
FEATURES_V2_PATH = ROOT / "data" / "features" / "series_features_v2_map_pool.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
MODELING_DIR = ROOT / "data" / "modeling"
RF_SELECTED_CONFIG_PATH = MODELING_DIR / "random_forest_v2_selected_config.json"
XGB_SELECTED_CONFIG_PATH = MODELING_DIR / "xgboost_v2_selected_config.json"
TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"
AUDIT_PATH = MODELING_DIR / "phase5b1_fold_preprocessing_audit.json"

N_FOLDS = 4
EXPECTED_SERIES_ROWS = 9456
EXPECTED_TRAIN_N = 6619
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Family grouping for V2's 30 new features, taken directly from Phase 5A's
# own inventory table (reports/phase5a_map_feature_engineering.md section 8).
# Family A (original V1) = everything NOT listed below.
# ---------------------------------------------------------------------------
FAMILY_B_POOL_DEPTH = [
    "map_pool_size_diff", "map_pool_total_matches_diff", "map_pool_experienced_maps_diff",
    "map_pool_mean_elo_diff", "map_pool_best_elo_diff", "map_pool_second_best_elo_diff",
    "map_pool_third_best_elo_diff", "map_pool_worst_elo_diff", "map_pool_mean_smoothed_wr_diff",
    "map_pool_best_smoothed_wr_diff", "map_pool_second_best_smoothed_wr_diff",
    "map_pool_third_best_smoothed_wr_diff", "map_pool_worst_smoothed_wr_diff",
    "map_pool_mean_normalized_margin_diff",
]
FAMILY_C_MATCHUP = [
    "map_matchup_mean_elo_advantage", "map_matchup_median_elo_advantage",
    "map_matchup_midrange_elo_advantage", "map_matchup_positive_advantage_balance",
    "map_matchup_mean_smoothed_wr_advantage", "map_matchup_median_smoothed_wr_advantage",
]
FAMILY_D_CONFIDENCE = [
    "map_pool_size_min", "map_pool_total_matches_min", "both_teams_have_map_pool_history",
    "both_teams_have_3_recent_maps", "both_teams_have_5_experienced_maps", "union_map_count",
    "shared_recent_map_count", "shared_experienced_map_count", "map_matchup_shared_coverage",
    "map_matchup_elo_advantage_range",
]


def family_of(feature):
    if feature in FAMILY_B_POOL_DEPTH:
        return "B_pool_depth"
    if feature in FAMILY_C_MATCHUP:
        return "C_same_map_matchup"
    if feature in FAMILY_D_CONFIDENCE:
        return "D_coverage_confidence"
    return "A_original_v1"


# ---------------------------------------------------------------------------
# Metrics (same local convention as random_forest_tuning_v2.py / xgboost_tuning_v2.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_proba, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def slice_fold(cv_df, features_df, fold):
    fold_train_ids = set(cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "train"), "match_id"])
    fold_val_ids = set(cv_df.loc[(cv_df["fold"] == fold) & (cv_df["role"] == "validation"), "match_id"])
    fold_train_raw = features_df[features_df["match_id"].isin(fold_train_ids)].reset_index(drop=True)
    fold_val_raw = features_df[features_df["match_id"].isin(fold_val_ids)].reset_index(drop=True)
    return fold_train_raw, fold_val_raw, fold_train_ids, fold_val_ids


def evaluate_model_on_fold(fold_train_raw, fold_val_raw, target_col, augment_fn, fit_fn, transform_fn, build_model_fn):
    """ONE generic per-fold evaluator, shared by every (model family, feature
    set) combination. Mirrors fold-train only; fits preprocessing fresh on
    THIS fold's augmented train only; fits the model with the caller-supplied
    (frozen) hyperparameters. Never touches fold_val_raw except to transform
    and score it."""
    augmented = augment_fn(fold_train_raw)
    params = fit_fn(augmented)

    X_aug, names = transform_fn(augmented, params)
    y_aug = augmented[target_col].to_numpy(dtype=float)

    X_train, _ = transform_fn(fold_train_raw, params)  # UNMIRRORED -> train metrics
    y_train = fold_train_raw[target_col].to_numpy(dtype=float)

    X_val, _ = transform_fn(fold_val_raw, params)  # never mirrored -> scoring only
    y_val = fold_val_raw[target_col].to_numpy(dtype=float)

    model = build_model_fn()
    model.fit(X_aug, y_aug)

    proba_train = model.predict_proba(X_train)[:, 1]
    pred_train = model.predict(X_train)
    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)

    return {
        "model": model, "feature_names": names, "preprocessing_params": params,
        "train_metrics": compute_metrics(y_train, proba_train, pred_train),
        "val_metrics": compute_metrics(y_val, proba_val, pred_val),
        "X_val": X_val, "y_val": y_val,
    }


# ---------------------------------------------------------------------------
# Pre-flight: V1 <-> V2 contract (row count/order/target/datetime/team identity)
# ---------------------------------------------------------------------------

def check_v1_v2_contract(v1_df, v2_df, v1_model_features, target_col):
    assert len(v1_df) == len(v2_df) == EXPECTED_SERIES_ROWS, \
        f"row count mismatch: v1={len(v1_df)} v2={len(v2_df)} expected={EXPECTED_SERIES_ROWS}"
    assert v1_df["match_id"].tolist() == v2_df["match_id"].tolist(), "match_id order differs between V1 and V2"
    assert v1_df[target_col].equals(v2_df[target_col]), "target differs between V1 and V2"
    assert v1_df["datetime"].equals(v2_df["datetime"]), "datetime differs between V1 and V2"
    assert v1_df["team1_canonical"].equals(v2_df["team1_canonical"]), "team1_canonical differs between V1 and V2"
    assert v1_df["team2_canonical"].equals(v2_df["team2_canonical"]), "team2_canonical differs between V1 and V2"
    for c in v1_model_features:
        a, b = v1_df[c], v2_df[c]
        if pd.api.types.is_float_dtype(a):
            assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), f"V1 feature {c} not preserved in V2"
        else:
            assert a.equals(b), f"V1 feature {c} not preserved in V2"
    print(f"Pre-flight OK: V1/V2 share {EXPECTED_SERIES_ROWS} rows, identical order/target/datetime/team identity, "
          f"all {len(v1_model_features)} V1 features preserved value-for-value in V2.")


def main():
    cfg_v1 = yaml.safe_load(CONFIG_V1_PATH.read_text(encoding="utf-8"))
    model_features_v1 = cfg_v1["model_features"]
    target_col = cfg_v1["target"]
    roles_v2 = pc2.load_v2_roles(CONFIG_V2_PATH)
    assert roles_v2["target"] == target_col

    v1_df = pd.read_parquet(FEATURES_V1_PATH, engine="fastparquet")
    v2_df = pd.read_parquet(FEATURES_V2_PATH, engine="fastparquet")
    check_v1_v2_contract(v1_df, v2_df, model_features_v1, target_col)

    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    full_train_ids = set(cv_df["match_id"])
    assert len(full_train_ids) == EXPECTED_TRAIN_N, \
        f"expected the CV fold manifest to cover exactly {EXPECTED_TRAIN_N} TRAIN match_ids, got {len(full_train_ids)}"
    # STRUCTURAL GUARANTEE: this script never reads data/modeling/series_split_v1.csv.
    # "Full TRAIN" is derived entirely from the CV fold manifest (fold 4's train+validation
    # roles alone already cover all 5 chronological blocks = the full TRAIN partition),
    # so the 1,419-match main VALIDATION partition can never be loaded here even by accident.

    if not RF_SELECTED_CONFIG_PATH.exists():
        raise RuntimeError(f"{RF_SELECTED_CONFIG_PATH} missing - run training/random_forest/random_forest_tuning_v2.py first.")
    if not XGB_SELECTED_CONFIG_PATH.exists():
        raise RuntimeError(f"{XGB_SELECTED_CONFIG_PATH} missing - run training/xgboost/xgboost_tuning_v2.py first.")
    rf_selected = json.loads(RF_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    xgb_selected = json.loads(XGB_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    rf_params = dict(rf_selected["params"])
    xgb_hp = dict(xgb_selected["params"])
    xgb_fixed = dict(xgb_selected["fixed_params"])
    xgb_n_estimators = int(xgb_selected["final_n_estimators"])
    print(f"Frozen RF V2 params ({rf_selected['candidate_id']}): {rf_params}")
    print(f"Frozen XGB V2 params ({xgb_selected['selected_candidate_id']}): {xgb_hp} + fixed {xgb_fixed}, "
          f"n_estimators={xgb_n_estimators} (no early stopping - identical for V1 and V2 arms)")

    def build_rf():
        return RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)

    def build_xgb():
        return XGBClassifier(n_estimators=xgb_n_estimators, **xgb_hp, **xgb_fixed)

    # ---- feature-set adapters: (augment_fn, fit_fn, transform_fn) per (model, feature_set) ----
    def v1_adapters():
        return (pc1.build_augmented_training_raw,
                lambda aug: rf1.fit_preprocessing(aug, model_features_v1),
                lambda df, params: rf1.transform(df, params),
                lambda aug: xgb1.fit_preprocessing(aug, model_features_v1),
                lambda df, params: xgb1.transform(df, params))

    def v2_adapters():
        return (lambda df: pc2.build_augmented_training_raw(df, roles_v2),
                lambda aug: rf2.fit_preprocessing(aug, roles_v2),
                lambda df, params: rf2.transform(df, params, roles_v2),
                lambda aug: xgb2.fit_preprocessing(aug, roles_v2),
                lambda df, params: xgb2.transform(df, params, roles_v2))

    feature_sets = {
        "v1": {"df": v1_df, "adapters": v1_adapters()},
        "v2": {"df": v2_df, "adapters": v2_adapters()},
    }

    fold_rows = []
    perm_importance_by_fs = {"v2": {"rf": [], "xgb": []}}
    audit_entries = []

    for fold in range(1, N_FOLDS + 1):
        for fs_name, fs in feature_sets.items():
            augment_fn, rf_fit, rf_transform, xgb_fit, xgb_transform = fs["adapters"]
            fold_train_raw, fold_val_raw, fold_train_ids, fold_val_ids = slice_fold(cv_df, fs["df"], fold)
            assert fold_train_ids.isdisjoint(fold_val_ids)

            for model_name, fit_fn, transform_fn, build_model_fn in [
                ("rf", rf_fit, rf_transform, build_rf),
                ("xgb", xgb_fit, xgb_transform, build_xgb),
            ]:
                result = evaluate_model_on_fold(fold_train_raw, fold_val_raw, target_col,
                                                 augment_fn, fit_fn, transform_fn, build_model_fn)
                tm, vm = result["train_metrics"], result["val_metrics"]
                fold_rows.append({
                    "model": model_name, "feature_set": fs_name, "fold": fold,
                    "n_fold_train": len(fold_train_raw), "n_fold_val": len(fold_val_raw),
                    "train_accuracy": tm["accuracy"], "train_roc_auc": tm["roc_auc"],
                    "train_log_loss": tm["log_loss"], "train_brier": tm["brier"], "train_f1": tm["f1"],
                    "val_accuracy": vm["accuracy"], "val_roc_auc": vm["roc_auc"],
                    "val_log_loss": vm["log_loss"], "val_brier": vm["brier"], "val_f1": vm["f1"],
                    "train_val_auc_gap": tm["roc_auc"] - vm["roc_auc"],
                })

                # preprocessing audit entry (for validation/validate_phase5b1.py's
                # independent per-fold recomputation check)
                stats_key = "train_medians" if "train_medians" in result["preprocessing_params"] else "train_medians_unused_reference"
                audit_entries.append({
                    "model": model_name, "feature_set": fs_name, "fold": fold,
                    "stats_key": stats_key,
                    "fitted_stats": result["preprocessing_params"][stats_key],
                })

                # V2-only: fold-validation permutation importance (still TRAIN-only CV)
                if fs_name == "v2":
                    perm = permutation_importance(result["model"], result["X_val"], result["y_val"],
                                                    scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
                    perm_df = pd.DataFrame({"feature": result["feature_names"],
                                             "importance": perm.importances_mean})
                    perm_importance_by_fs["v2"][model_name].append(perm_df)

            print(f"  fold {fold} / {fs_name}: done")

    fold_df = pd.DataFrame(fold_rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)

    # ---- aggregate mean/std per (model, feature_set) ----
    agg_records = []
    for (model_name, fs_name), g in fold_df.groupby(["model", "feature_set"]):
        agg_records.append({
            "model": model_name, "feature_set": fs_name, "row_type": "aggregate",
            "val_log_loss_mean": g["val_log_loss"].mean(), "val_log_loss_std": g["val_log_loss"].std(ddof=0),
            "val_roc_auc_mean": g["val_roc_auc"].mean(), "val_roc_auc_std": g["val_roc_auc"].std(ddof=0),
            "val_brier_mean": g["val_brier"].mean(), "val_brier_std": g["val_brier"].std(ddof=0),
            "val_accuracy_mean": g["val_accuracy"].mean(), "val_accuracy_std": g["val_accuracy"].std(ddof=0),
            "val_f1_mean": g["val_f1"].mean(), "val_f1_std": g["val_f1"].std(ddof=0),
            "train_roc_auc_mean": g["train_roc_auc"].mean(),
            "train_val_auc_gap_mean": g["train_val_auc_gap"].mean(),
        })
    agg_df = pd.DataFrame(agg_records)

    # ---- paired fold-wise deltas (V2 - V1), same fold matches on both sides ----
    delta_rows = []
    for model_name in ["rf", "xgb"]:
        v1_f = fold_df[(fold_df.model == model_name) & (fold_df.feature_set == "v1")].set_index("fold")
        v2_f = fold_df[(fold_df.model == model_name) & (fold_df.feature_set == "v2")].set_index("fold")
        for fold in range(1, N_FOLDS + 1):
            delta_rows.append({
                "model": model_name, "fold": fold,
                "delta_log_loss": v2_f.loc[fold, "val_log_loss"] - v1_f.loc[fold, "val_log_loss"],
                "delta_roc_auc": v2_f.loc[fold, "val_roc_auc"] - v1_f.loc[fold, "val_roc_auc"],
                "delta_brier": v2_f.loc[fold, "val_brier"] - v1_f.loc[fold, "val_brier"],
                "delta_accuracy": v2_f.loc[fold, "val_accuracy"] - v1_f.loc[fold, "val_accuracy"],
                "delta_f1": v2_f.loc[fold, "val_f1"] - v1_f.loc[fold, "val_f1"],
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_agg = delta_df.groupby("model")[["delta_log_loss", "delta_roc_auc", "delta_brier",
                                            "delta_accuracy", "delta_f1"]].mean().reset_index()
    delta_agg["fold"] = "mean"

    combined = pd.concat([
        fold_df.assign(row_type="fold"),
        agg_df,
        delta_df.assign(row_type="paired_delta"),
        delta_agg.assign(row_type="paired_delta_mean"),
    ], ignore_index=True, sort=False)
    combined.to_csv(TABLES_DIR / "series_feature_v1_v2_cv_comparison.csv", index=False, encoding="utf-8")

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": audit_entries}, f, indent=2)

    # ---- full augmented-TRAIN V2 refits, for stable impurity/gain/weight importance ----
    full_train_df_v2 = v2_df[v2_df["match_id"].isin(full_train_ids)].reset_index(drop=True)
    assert len(full_train_df_v2) == EXPECTED_TRAIN_N
    augmented_full_v2 = pc2.build_augmented_training_raw(full_train_df_v2, roles_v2)
    pc2.assert_augmented_symmetry(augmented_full_v2, roles_v2)

    rf_full_params = rf2.fit_preprocessing(augmented_full_v2, roles_v2)
    X_full_rf, rf_names = rf2.transform(augmented_full_v2, rf_full_params, roles_v2)
    y_full = augmented_full_v2[target_col].to_numpy(dtype=float)
    rf_full_model = build_rf()
    rf_full_model.fit(X_full_rf, y_full)

    xgb_full_params = xgb2.fit_preprocessing(augmented_full_v2, roles_v2)
    X_full_xgb, xgb_names = xgb2.transform(augmented_full_v2, xgb_full_params, roles_v2)
    xgb_full_model = build_xgb()
    xgb_full_model.fit(X_full_xgb, y_full)

    # RF impurity importance
    rf_importance_df = pd.DataFrame({
        "feature": rf_names, "impurity_importance": rf_full_model.feature_importances_,
    })
    rf_perm_avg = pd.concat(perm_importance_by_fs["v2"]["rf"]).groupby("feature")["importance"] \
        .agg(["mean", "std"]).rename(columns={"mean": "fold_val_permutation_mean", "std": "fold_val_permutation_std"})
    rf_importance_df = rf_importance_df.merge(rf_perm_avg, on="feature", how="left")
    rf_importance_df["family"] = rf_importance_df["feature"].map(family_of)
    rf_importance_df = rf_importance_df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)
    rf_importance_df.to_csv(TABLES_DIR / "series_v2_feature_importance_rf.csv", index=False, encoding="utf-8")

    # XGB gain/weight importance
    booster = xgb_full_model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    gain_by = {n: float(gain_raw.get(f"f{i}", 0.0)) for i, n in enumerate(xgb_names)}
    weight_by = {n: float(weight_raw.get(f"f{i}", 0.0)) for i, n in enumerate(xgb_names)}
    xgb_importance_df = pd.DataFrame({
        "feature": xgb_names,
        "gain": [gain_by[n] for n in xgb_names],
        "weight": [weight_by[n] for n in xgb_names],
    })
    xgb_perm_avg = pd.concat(perm_importance_by_fs["v2"]["xgb"]).groupby("feature")["importance"] \
        .agg(["mean", "std"]).rename(columns={"mean": "fold_val_permutation_mean", "std": "fold_val_permutation_std"})
    xgb_importance_df = xgb_importance_df.merge(xgb_perm_avg, on="feature", how="left")
    xgb_importance_df["family"] = xgb_importance_df["feature"].map(family_of)
    xgb_importance_df = xgb_importance_df.sort_values("gain", ascending=False).reset_index(drop=True)
    xgb_importance_df.to_csv(TABLES_DIR / "series_v2_feature_importance_xgb.csv", index=False, encoding="utf-8")

    # ---- figures ----
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    for model_name, fname, title in [("rf", "series_feature_v1_v2_rf_cv.png", "Random Forest V2 config"),
                                       ("xgb", "series_feature_v1_v2_xgb_cv.png", "XGBoost V2 config")]:
        sub = fold_df[fold_df.model == model_name]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, metric, label in zip(axes, ["val_log_loss", "val_roc_auc", "val_brier"],
                                      ["Log Loss (lower better)", "ROC-AUC (higher better)", "Brier (lower better)"]):
            for fs_name, marker in [("v1", "o"), ("v2", "s")]:
                s = sub[sub.feature_set == fs_name].sort_values("fold")
                ax.plot(s["fold"], s[metric], marker=marker, label=f"Feature set {fs_name.upper()}")
            ax.set_xlabel("fold")
            ax.set_ylabel(label)
            ax.set_xticks(range(1, N_FOLDS + 1))
            ax.legend()
        fig.suptitle(f"V1 vs V2 series features, TRAIN-only CV, frozen {title} (paired ablation)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close(fig)

    write_report(fold_df, agg_df, delta_df, delta_agg, rf_importance_df, xgb_importance_df,
                 rf_selected, xgb_selected, rf_params, xgb_hp, xgb_fixed, xgb_n_estimators)

    n_pass_summary = combined.shape[0]
    print(f"\nWrote {n_pass_summary} rows to reports/tables/series_feature_v1_v2_cv_comparison.csv")
    print("Wrote reports/tables/series_v2_feature_importance_rf.csv")
    print("Wrote reports/tables/series_v2_feature_importance_xgb.csv")
    print("Wrote reports/figures/series_feature_v1_v2_rf_cv.png")
    print("Wrote reports/figures/series_feature_v1_v2_xgb_cv.png")
    print("Wrote reports/phase5b1_series_map_pool_cv_results.md")
    print(f"Wrote {AUDIT_PATH.relative_to(ROOT)}")


def verdict_line(mean_delta_ll, mean_delta_auc, mean_delta_brier, folds_ll_negative, folds_auc_positive, n_folds):
    ll_improves = mean_delta_ll < 0
    auc_improves = mean_delta_auc > 0
    brier_improves = mean_delta_brier < 0
    n_agree = sum([ll_improves, auc_improves, brier_improves])
    consistent = folds_ll_negative >= 3 or folds_auc_positive >= 3
    if n_agree == 3 and consistent:
        return "HELP"
    if n_agree == 0:
        return "DO NOT HELP"
    return "MIXED"


def write_report(fold_df, agg_df, delta_df, delta_agg, rf_importance_df, xgb_importance_df,
                  rf_selected, xgb_selected, rf_params, xgb_hp, xgb_fixed, xgb_n_estimators):
    def agg_row(model_name, fs_name):
        return agg_df[(agg_df.model == model_name) & (agg_df.feature_set == fs_name)].iloc[0]

    def delta_row(model_name):
        return delta_agg[delta_agg.model == model_name].iloc[0]

    md = []
    md.append("# Phase 5B.1 - Paired V1 vs V2 Map-Pool Feature Evaluation (TRAIN-only CV)\n")
    md.append("**Framing.** The 4 chronological CV folds used below "
              "(`data/modeling/random_forest_cv_folds_v2.csv`) are the SAME folds RF V2's and XGB V2's frozen "
              "hyperparameters were originally selected against in Phase 4B.1/4C.1. This report is therefore a "
              "**paired feature ablation** under a fixed, previously-selected model configuration - it answers "
              "*\"with the model configuration held fixed, do V2 features improve performance relative to V1 "
              "features on the same development folds used during tuning?\"* - not a fresh, unbiased estimate of "
              "future generalization. **The main 1,419-match VALIDATION partition was never loaded in this script "
              "and remains untouched.**\n")
    md.append("**XGBoost methodology note.** The original `xgboost_tuning_v2.py` used an inner early-stopping "
              "split with fold-specific `best_iteration`. This script instead uses XGB V2's single FROZEN final "
              "configuration (`n_estimators=98`, no early stopping, full outer-fold training history) identically "
              "for both the V1 and V2 arms - holding the model configuration identical across feature sets is what "
              "makes this comparison paired and feature-attributable. XGB-V1 numbers below are **not** expected to "
              "reproduce `reports/tables/xgboost_tuning_v2.csv` row-for-row; that is by design. RF has no "
              "early-stopping distinction, so RF-V1 below is expected to closely reproduce "
              "`random_forest_tuning_v2.csv`'s frozen-candidate row.\n")

    md.append("## Frozen configurations (loaded, never altered)\n")
    md.append(f"- **RF V2** (`{rf_selected['candidate_id']}`): `{rf_params}`, "
              "called as `RandomForestClassifier(**params, random_state=42, n_jobs=-1)`.")
    md.append(f"- **XGB V2** (`{xgb_selected['selected_candidate_id']}`): `{xgb_hp}` + fixed `{xgb_fixed}`, "
              f"`n_estimators={xgb_n_estimators}`, no early stopping, no eval_set - identical for V1 and V2.\n")

    for model_name, label in [("rf", "Random Forest (frozen RF V2 config)"), ("xgb", "XGBoost (frozen XGB V2 config)")]:
        v1a, v2a = agg_row(model_name, "v1"), agg_row(model_name, "v2")
        d = delta_row(model_name)
        dfold = delta_df[delta_df.model == model_name].sort_values("fold")
        md.append(f"## {label}\n")
        md.append("| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | "
                  "mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |")
        md.append("|---|---|---|---|---|---|---|---|")
        for fs_name, a in [("V1", v1a), ("V2", v2a)]:
            md.append(f"| {fs_name} | {a['val_log_loss_mean']:.4f}±{a['val_log_loss_std']:.4f} | "
                      f"{a['val_roc_auc_mean']:.4f}±{a['val_roc_auc_std']:.4f} | "
                      f"{a['val_brier_mean']:.4f}±{a['val_brier_std']:.4f} | {a['val_accuracy_mean']:.4f} | "
                      f"{a['val_f1_mean']:.4f} | {a['train_roc_auc_mean']:.4f} | {a['train_val_auc_gap_mean']:+.4f} |")
        md.append("")
        md.append("### Paired fold-wise deltas (V2 - V1; negative=better for Log Loss/Brier, "
                  "positive=better for ROC-AUC/Accuracy/F1)\n")
        md.append("| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |")
        md.append("|---|---|---|---|---|---|")
        for _, r in dfold.iterrows():
            md.append(f"| {int(r['fold'])} | {r['delta_log_loss']:+.4f} | {r['delta_roc_auc']:+.4f} | "
                      f"{r['delta_brier']:+.4f} | {r['delta_accuracy']:+.4f} | {r['delta_f1']:+.4f} |")
        md.append(f"| **mean** | **{d['delta_log_loss']:+.4f}** | **{d['delta_roc_auc']:+.4f}** | "
                  f"**{d['delta_brier']:+.4f}** | **{d['delta_accuracy']:+.4f}** | **{d['delta_f1']:+.4f}** |\n")

        n_ll_neg = int((dfold["delta_log_loss"] < 0).sum())
        n_auc_pos = int((dfold["delta_roc_auc"] > 0).sum())
        n_brier_neg = int((dfold["delta_brier"] < 0).sum())
        md.append(f"Log loss improved (V2 better) in **{n_ll_neg}/{N_FOLDS}** folds; ROC-AUC improved in "
                  f"**{n_auc_pos}/{N_FOLDS}** folds; Brier improved in **{n_brier_neg}/{N_FOLDS}** folds.\n")
        md.append("Differences of only a few thousandths in mean CV log loss/ROC-AUC should not be over-interpreted "
                  "- this is a single paired ablation on 4 folds, not a significance test.\n")

    md.append("## V2-only feature importance (descriptive, no feature selection performed)\n")
    md.append("RF impurity importance and XGB gain both computed from a full-augmented-TRAIN refit of the frozen "
              "V2 configuration (never touching validation/test); permutation importance is the **average across "
              "the 4 CV folds' own fold-validation slices** (still entirely inside TRAIN-only CV, never the main "
              "validation partition).\n")

    md.append("### RF: top 10 by impurity importance\n")
    md.append("| rank | feature | family | impurity importance | fold-val permutation (mean±std) |")
    md.append("|---|---|---|---|---|")
    for i, r in rf_importance_df.head(10).iterrows():
        md.append(f"| {i+1} | {r['feature']} | {r['family']} | {r['impurity_importance']:.4f} | "
                  f"{r['fold_val_permutation_mean']:.4f}±{r['fold_val_permutation_std']:.4f} |")
    md.append("")

    md.append("### XGB: top 10 by gain\n")
    md.append("| rank | feature | family | gain | weight | fold-val permutation (mean±std) |")
    md.append("|---|---|---|---|---|---|")
    for i, r in xgb_importance_df.head(10).iterrows():
        md.append(f"| {i+1} | {r['feature']} | {r['family']} | {r['gain']:.4f} | {int(r['weight'])} | "
                  f"{r['fold_val_permutation_mean']:.4f}±{r['fold_val_permutation_std']:.4f} |")
    md.append("")

    watch = ["elo_diff", "total_matches_before_diff"]
    md.append("### Do any map features beat the two named V1 references?\n")
    md.append("Specifically checking whether any `map_pool_*`/`map_matchup_*`/coverage feature outranks "
              f"`{watch[0]}` and `{watch[1]}` in validation-fold permutation importance:\n")
    for name, df in [("RF", rf_importance_df), ("XGB", xgb_importance_df)]:
        ref_vals = df.set_index("feature").loc[watch, "fold_val_permutation_mean"]
        map_df = df[df["family"] != "A_original_v1"]
        beat = map_df[map_df["fold_val_permutation_mean"] > ref_vals.min()]
        md.append(f"- **{name}**: `elo_diff`={ref_vals[watch[0]]:.4f}, `total_matches_before_diff`="
                  f"{ref_vals[watch[1]]:.4f} (fold-val permutation mean). "
                  f"{len(beat)} of {len(map_df)} map-derived features exceed the lower of the two: "
                  + (", ".join(f"`{f}`" for f in beat.sort_values('fold_val_permutation_mean', ascending=False)['feature'].head(5))
                     if len(beat) else "none.") + "")
    md.append("")

    md.append("## Family analysis (which map-pool families carry meaningful validation-fold permutation importance)\n")
    md.append("A = original 17 Phase-3 features, B = pool-depth/order-statistics (`map_pool_*` diffs), "
              "C = same-map matchup advantages (`map_matchup_*` advantages), D = map coverage/confidence "
              "(pool-size/union/shared-count/coverage flags). Not causal - descriptive only.\n")
    for name, df in [("RF", rf_importance_df), ("XGB", xgb_importance_df)]:
        fam = df.groupby("family")["fold_val_permutation_mean"].agg(["mean", "sum", "count"]).reindex(
            ["A_original_v1", "B_pool_depth", "C_same_map_matchup", "D_coverage_confidence"])
        md.append(f"**{name}** - mean / summed fold-validation permutation importance by family:\n")
        md.append("| family | n features | mean permutation importance | summed permutation importance |")
        md.append("|---|---|---|---|")
        for fam_name, r in fam.iterrows():
            md.append(f"| {fam_name} | {int(r['count'])} | {r['mean']:.4f} | {r['sum']:.4f} |")
        md.append("")

    md.append("## Answering the brief\n")
    for model_name, label in [("rf", "RF"), ("xgb", "XGB")]:
        v1a, v2a = agg_row(model_name, "v1"), agg_row(model_name, "v2")
        d = delta_row(model_name)
        dfold = delta_df[delta_df.model == model_name].sort_values("fold")
        n_ll = int((dfold["delta_log_loss"] < 0).sum())
        n_auc = int((dfold["delta_roc_auc"] > 0).sum())
        n_br = int((dfold["delta_brier"] < 0).sum())
        md.append(f"**{label}**: mean CV log loss "
                  f"{'improved' if d['delta_log_loss'] < 0 else 'did not improve'} ({d['delta_log_loss']:+.4f}, "
                  f"{n_ll}/{N_FOLDS} folds better); mean CV ROC-AUC "
                  f"{'improved' if d['delta_roc_auc'] > 0 else 'did not improve'} ({d['delta_roc_auc']:+.4f}, "
                  f"{n_auc}/{N_FOLDS} folds better); Brier "
                  f"{'improved' if d['delta_brier'] < 0 else 'did not improve'} ({d['delta_brier']:+.4f}, "
                  f"{n_br}/{N_FOLDS} folds better).")
    md.append("")

    rf_d, xgb_d = delta_row("rf"), delta_row("xgb")
    rf_dfold = delta_df[delta_df.model == "rf"]
    xgb_dfold = delta_df[delta_df.model == "xgb"]
    rf_verdict = verdict_line(rf_d["delta_log_loss"], rf_d["delta_roc_auc"], rf_d["delta_brier"],
                               int((rf_dfold["delta_log_loss"] < 0).sum()), int((rf_dfold["delta_roc_auc"] > 0).sum()), N_FOLDS)
    xgb_verdict = verdict_line(xgb_d["delta_log_loss"], xgb_d["delta_roc_auc"], xgb_d["delta_brier"],
                                int((xgb_dfold["delta_log_loss"] < 0).sum()), int((xgb_dfold["delta_roc_auc"] > 0).sum()), N_FOLDS)
    if rf_verdict == xgb_verdict:
        overall = rf_verdict
    elif "DO NOT HELP" in (rf_verdict, xgb_verdict) and "HELP" in (rf_verdict, xgb_verdict):
        overall = "MIXED"
    else:
        overall = "MIXED"

    md.append("## Do the map-pool features appear to add real predictive information beyond Feature Set V1?\n")
    md.append(f"RF verdict: **{rf_verdict}**. XGB verdict: **{xgb_verdict}**. Combined: **{overall}**. Reported "
              "using cautious language given differences on the order of a few thousandths are within the noise "
              "this small a fold count can resolve, and given the framing note above (paired ablation on the same "
              "folds used for hyperparameter selection, not an independent generalization estimate).\n")

    md.append("## Conclusion\n")
    md.append(f"**MAP FEATURES {overall}**\n")
    md.append("- **TEST = SEALED**")
    md.append("- **COLOGNE = UNTOUCHED**")
    md.append("- **MAIN VALIDATION = NOT USED**\n")

    (REPORTS / "phases" / "phase5b1_series_map_pool_cv_results.md").write_text("\n".join(md), encoding="utf-8")



if __name__ == "__main__":
    main()
