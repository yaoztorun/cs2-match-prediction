"""
Phase 5B.3: Paired V2 vs V3 team-form feature-set evaluation.

Question answered: with RF V2's and XGB V2's FROZEN hyperparameters held
fixed (loaded from data/modeling/random_forest_v2_selected_config.json and
data/modeling/xgboost_v2_selected_config.json - selected BEFORE V3 existed,
never retuned here), do the 12 new opponent-adjusted/recency-weighted team-form
features (series_features_v3_form.parquet, Phase 5B.2) improve series-winner
prediction relative to V2 (series_features_v2_map_pool.parquet) alone?

INTERPRETATION (read before interpreting the numbers below)
------------------------------------------------------------------------
RF V2 / XGB V2's hyperparameters were selected via CV on V2-era features,
before V3 existed. This has an asymmetric consequence for what a result here
can and cannot prove:
  * If V3 improves under these frozen, previously-selected configurations,
    that IS evidence the new form information adds predictive signal without
    needing any retuning.
  * If V3 does NOT improve, that does NOT mean the form features carry no
    useful signal - the correct conclusion is only that "V3 did not improve
    performance under the frozen, previously-selected model configurations."
    A later V3-specific tune could use the richer feature space differently
    (e.g. different regularization, different tree depth/complexity).
This report states results using that language throughout, and the framing
note from Phase 5B.1 still applies: these are the SAME 4 folds RF V2/XGB V2
were originally selected against, so this is a paired development-set
feature ablation, not an independent generalization estimate. The main
1,419-match VALIDATION partition is never loaded here and remains untouched.

REGRESSION PARITY WITH PHASE 5B.1's OWN V2 ARM
------------------------------------------------------------------------
Phase 5B.1 already evaluated V2 + frozen RF V2 config and V2 + frozen XGB V2
config (n_estimators=98, no early stopping) through this identical protocol.
Phase 5B.3's V2 arm uses the exact same data/configuration/protocol
combination, so validation/validate_phase5b3.py requires the V2-arm fold metrics
computed here to match Phase 5B.1's own saved
reports/tables/series_feature_v1_v2_cv_comparison.csv V2-arm rows within a
strict numeric tolerance (machine-level float nondeterminism only) - a
material difference would indicate a preprocessing/evaluation-harness
regression, not a modeling result, and FAILS validation.

Never touches: TEST, Cologne/post-Cologne, the 1,419-match main VALIDATION
partition, or any Phase 4 / Phase 5A / Phase 5B.1 / Phase 5B.2 artifact
(read-only against all of them; no feature is added, removed or changed).

Writes:
    reports/tables/series_feature_v2_v3_cv_comparison.csv
    reports/tables/series_v3_feature_importance_rf.csv
    reports/tables/series_v3_feature_importance_xgb.csv
    reports/tables/series_v3_group_permutation_importance_rf.csv
    reports/tables/series_v3_group_permutation_importance_xgb.csv
    reports/figures/series_feature_v2_v3_rf_cv.png
    reports/figures/series_feature_v2_v3_xgb_cv.png
    reports/phase5b3_team_form_cv_results.md
    data/modeling/phase5b3_fold_preprocessing_audit.json   (for validation/validate_phase5b3.py)
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
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from _common import ROOT, REPORTS

import feature_engineering.preprocessing.preprocessing_common_v2_map_pool as pc2
import feature_engineering.preprocessing.preprocessing_random_forest_v2_map_pool as rf2
import feature_engineering.preprocessing.preprocessing_xgboost_v2_map_pool as xgb2
import feature_engineering.preprocessing.preprocessing_common_v3_form as pc3
from evaluation.validation.evaluate_series_feature_sets_v2 import compute_metrics, slice_fold, evaluate_model_on_fold
from feature_engineering.form.team_form_engine import FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES

CONFIG_V2_PATH = ROOT / "config" / "features" / "series_features_v2_map_pool.yaml"
CONFIG_V3_PATH = ROOT / "config" / "features" / "series_features_v3_form.yaml"
FEATURES_V2_PATH = ROOT / "data" / "features" / "series_features_v2_map_pool.parquet"
FEATURES_V3_PATH = ROOT / "data" / "features" / "series_features_v3_form.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
MODELING_DIR = ROOT / "data" / "modeling"
RF_SELECTED_CONFIG_PATH = MODELING_DIR / "random_forest_v2_selected_config.json"
XGB_SELECTED_CONFIG_PATH = MODELING_DIR / "xgboost_v2_selected_config.json"
TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"
AUDIT_PATH = MODELING_DIR / "phase5b3_fold_preprocessing_audit.json"

N_FOLDS = 4
EXPECTED_SERIES_ROWS = 9456
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Family grouping (7 families, per the Phase 5B.3 brief)
# A = original V1 (inherited, everything not listed below)
# B/C/D = inherited Phase 5A map-pool families (same as Phase 5B.1)
# E/F/G = new Phase 5B.2 team-form families
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
FAMILY_E_OPPONENT_STRENGTH = [
    "avg_opponent_elo_last_5_diff", "avg_opponent_elo_last_10_diff",
    "performance_residual_last_5_diff", "performance_residual_last_10_diff", "performance_residual_all_diff",
]
FAMILY_F_TIME_DECAYED = [
    "time_weighted_win_rate_diff", "time_weighted_performance_residual_diff", "time_weighted_series_margin_diff",
]
FAMILY_G_FORM_CONFIDENCE = [
    "opponent_adjusted_history_min", "both_teams_have_5_adjusted_matches",
    "both_teams_have_10_adjusted_matches", "time_weighted_history_mass_min",
]
GROUPED_FAMILIES = {"E_opponent_strength": FAMILY_E_OPPONENT_STRENGTH,
                     "F_time_decayed": FAMILY_F_TIME_DECAYED,
                     "G_form_confidence": FAMILY_G_FORM_CONFIDENCE}


def family_of(feature):
    if feature in FAMILY_B_POOL_DEPTH:
        return "B_pool_depth"
    if feature in FAMILY_C_MATCHUP:
        return "C_same_map_matchup"
    if feature in FAMILY_D_CONFIDENCE:
        return "D_coverage_confidence"
    if feature in FAMILY_E_OPPONENT_STRENGTH:
        return "E_opponent_strength"
    if feature in FAMILY_F_TIME_DECAYED:
        return "F_time_decayed"
    if feature in FAMILY_G_FORM_CONFIDENCE:
        return "G_form_confidence"
    return "A_original_v1"


def grouped_permutation_importance(model, X, y, feature_names, groups, n_repeats=10, random_state=42):
    """Jointly permutes ALL columns belonging to one family together (one
    shared row-permutation applied across the whole group per repeat, not an
    independent permutation per column), so correlated features within a
    family are broken together rather than one-at-a-time - standard technique
    for descriptive importance under correlated features. Returns
    {group_name: {"mean": ..., "std": ...}} of the ROC-AUC decrease."""
    rng = np.random.RandomState(random_state)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for group_name, cols in groups.items():
        idxs = [name_to_idx[c] for c in cols if c in name_to_idx]
        decreases = []
        for _ in range(n_repeats):
            perm = rng.permutation(X.shape[0])
            X_perm = X.copy()
            X_perm[:, idxs] = X_perm[perm][:, idxs]
            score = roc_auc_score(y, model.predict_proba(X_perm)[:, 1])
            decreases.append(baseline - score)
        out[group_name] = {"mean": float(np.mean(decreases)), "std": float(np.std(decreases))}
    return out


def main():
    cfg_v2 = yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    target_col = cfg_v2["target"]
    roles_v2 = pc2.load_v2_roles(CONFIG_V2_PATH)
    roles_v3 = pc3.load_v3_roles(CONFIG_V3_PATH)
    assert roles_v3["target"] == target_col

    v2_df = pd.read_parquet(FEATURES_V2_PATH, engine="fastparquet")
    v3_df = pd.read_parquet(FEATURES_V3_PATH, engine="fastparquet")

    # ---- pre-flight V2 <-> V3 contract ----
    assert len(v2_df) == len(v3_df) == EXPECTED_SERIES_ROWS
    assert v2_df["match_id"].tolist() == v3_df["match_id"].tolist(), "match_id order differs V2 vs V3"
    assert v2_df[target_col].equals(v3_df[target_col]), "target differs V2 vs V3"
    assert v2_df["datetime"].equals(v3_df["datetime"]), "datetime differs V2 vs V3"
    for c in v2_df.columns:
        if pd.api.types.is_numeric_dtype(v2_df[c]):
            assert np.array_equal(v2_df[c].to_numpy(dtype=float), v3_df[c].to_numpy(dtype=float), equal_nan=True), c
        else:
            assert v2_df[c].equals(v3_df[c]), c
    new_cols = set(v3_df.columns) - set(v2_df.columns)
    assert new_cols == set(FORM_DIRECTIONAL_FEATURES) | set(FORM_SYMMETRIC_FEATURES), \
        f"expected exactly 12 new columns, got {sorted(new_cols)}"
    print(f"Pre-flight OK: V2/V3 share {EXPECTED_SERIES_ROWS} rows, identical order/target/datetime, "
          f"every V2 column preserved, exactly {len(new_cols)} new columns.")

    cv_df = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    full_train_ids = set(cv_df["match_id"])
    assert len(full_train_ids) == 6619

    rf_selected = json.loads(RF_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    xgb_selected = json.loads(XGB_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    rf_params = dict(rf_selected["params"])
    xgb_hp = dict(xgb_selected["params"])
    xgb_fixed = dict(xgb_selected["fixed_params"])
    xgb_n_estimators = int(xgb_selected["final_n_estimators"])
    print(f"Frozen RF V2 params ({rf_selected['candidate_id']}): {rf_params}")
    print(f"Frozen XGB V2 params ({xgb_selected['selected_candidate_id']}): {xgb_hp} + fixed {xgb_fixed}, "
          f"n_estimators={xgb_n_estimators} (no early stopping - identical for V2 and V3 arms)")

    def build_rf():
        return RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)

    def build_xgb():
        return XGBClassifier(n_estimators=xgb_n_estimators, **xgb_hp, **xgb_fixed)

    def v2_adapters():
        return (lambda df: pc2.build_augmented_training_raw(df, roles_v2),
                lambda aug: rf2.fit_preprocessing(aug, roles_v2),
                lambda df, params: rf2.transform(df, params, roles_v2),
                lambda aug: xgb2.fit_preprocessing(aug, roles_v2),
                lambda df, params: xgb2.transform(df, params, roles_v2))

    def v3_adapters():
        return (lambda df: pc2.build_augmented_training_raw(df, roles_v3),
                lambda aug: rf2.fit_preprocessing(aug, roles_v3),
                lambda df, params: rf2.transform(df, params, roles_v3),
                lambda aug: xgb2.fit_preprocessing(aug, roles_v3),
                lambda df, params: xgb2.transform(df, params, roles_v3))

    feature_sets = {"v2": {"df": v2_df, "adapters": v2_adapters()},
                     "v3": {"df": v3_df, "adapters": v3_adapters()}}

    fold_rows = []
    perm_importance_by_model = {"rf": [], "xgb": []}
    group_importance_by_model = {"rf": [], "xgb": []}
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

                stats_key = "train_medians" if "train_medians" in result["preprocessing_params"] else "train_medians_unused_reference"
                audit_entries.append({
                    "model": model_name, "feature_set": fs_name, "fold": fold,
                    "stats_key": stats_key,
                    "fitted_stats": result["preprocessing_params"][stats_key],
                })

                if fs_name == "v3":
                    perm = permutation_importance(result["model"], result["X_val"], result["y_val"],
                                                    scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
                    perm_df = pd.DataFrame({"feature": result["feature_names"], "importance": perm.importances_mean})
                    perm_importance_by_model[model_name].append(perm_df)

                    grouped = grouped_permutation_importance(
                        result["model"], result["X_val"], result["y_val"], result["feature_names"],
                        GROUPED_FAMILIES, n_repeats=10, random_state=RANDOM_STATE)
                    for gname, stats in grouped.items():
                        group_importance_by_model[model_name].append(
                            {"fold": fold, "family": gname, "mean_auc_decrease": stats["mean"], "std_auc_decrease": stats["std"]})

            print(f"  fold {fold} / {fs_name}: done")

    fold_df = pd.DataFrame(fold_rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)

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

    delta_rows = []
    for model_name in ["rf", "xgb"]:
        v2_f = fold_df[(fold_df.model == model_name) & (fold_df.feature_set == "v2")].set_index("fold")
        v3_f = fold_df[(fold_df.model == model_name) & (fold_df.feature_set == "v3")].set_index("fold")
        for fold in range(1, N_FOLDS + 1):
            delta_rows.append({
                "model": model_name, "fold": fold,
                "delta_log_loss": v3_f.loc[fold, "val_log_loss"] - v2_f.loc[fold, "val_log_loss"],
                "delta_roc_auc": v3_f.loc[fold, "val_roc_auc"] - v2_f.loc[fold, "val_roc_auc"],
                "delta_brier": v3_f.loc[fold, "val_brier"] - v2_f.loc[fold, "val_brier"],
                "delta_accuracy": v3_f.loc[fold, "val_accuracy"] - v2_f.loc[fold, "val_accuracy"],
                "delta_f1": v3_f.loc[fold, "val_f1"] - v2_f.loc[fold, "val_f1"],
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_agg = delta_df.groupby("model")[["delta_log_loss", "delta_roc_auc", "delta_brier",
                                            "delta_accuracy", "delta_f1"]].mean().reset_index()
    delta_agg["fold"] = "mean"

    combined = pd.concat([
        fold_df.assign(row_type="fold"), agg_df,
        delta_df.assign(row_type="paired_delta"), delta_agg.assign(row_type="paired_delta_mean"),
    ], ignore_index=True, sort=False)
    combined.to_csv(TABLES_DIR / "series_feature_v2_v3_cv_comparison.csv", index=False, encoding="utf-8")

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": audit_entries}, f, indent=2)

    # ---- full augmented-TRAIN V3 refits, for stable impurity/gain/weight importance ----
    full_train_df_v3 = v3_df[v3_df["match_id"].isin(full_train_ids)].reset_index(drop=True)
    assert len(full_train_df_v3) == 6619
    augmented_full_v3 = pc2.build_augmented_training_raw(full_train_df_v3, roles_v3)
    pc2.assert_augmented_symmetry(augmented_full_v3, roles_v3)

    rf_full_params = rf2.fit_preprocessing(augmented_full_v3, roles_v3)
    X_full_rf, rf_names = rf2.transform(augmented_full_v3, rf_full_params, roles_v3)
    y_full = augmented_full_v3[target_col].to_numpy(dtype=float)
    rf_full_model = build_rf()
    rf_full_model.fit(X_full_rf, y_full)

    xgb_full_params = xgb2.fit_preprocessing(augmented_full_v3, roles_v3)
    X_full_xgb, xgb_names = xgb2.transform(augmented_full_v3, xgb_full_params, roles_v3)
    xgb_full_model = build_xgb()
    xgb_full_model.fit(X_full_xgb, y_full)

    rf_importance_df = pd.DataFrame({"feature": rf_names, "impurity_importance": rf_full_model.feature_importances_})
    rf_perm_avg = pd.concat(perm_importance_by_model["rf"]).groupby("feature")["importance"] \
        .agg(["mean", "std"]).rename(columns={"mean": "fold_val_permutation_mean", "std": "fold_val_permutation_std"})
    rf_importance_df = rf_importance_df.merge(rf_perm_avg, on="feature", how="left")
    rf_importance_df["family"] = rf_importance_df["feature"].map(family_of)
    rf_importance_df = rf_importance_df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)
    rf_importance_df.to_csv(TABLES_DIR / "series_v3_feature_importance_rf.csv", index=False, encoding="utf-8")

    booster = xgb_full_model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    gain_by = {n: float(gain_raw.get(f"f{i}", 0.0)) for i, n in enumerate(xgb_names)}
    weight_by = {n: float(weight_raw.get(f"f{i}", 0.0)) for i, n in enumerate(xgb_names)}
    xgb_importance_df = pd.DataFrame({"feature": xgb_names, "gain": [gain_by[n] for n in xgb_names],
                                       "weight": [weight_by[n] for n in xgb_names]})
    xgb_perm_avg = pd.concat(perm_importance_by_model["xgb"]).groupby("feature")["importance"] \
        .agg(["mean", "std"]).rename(columns={"mean": "fold_val_permutation_mean", "std": "fold_val_permutation_std"})
    xgb_importance_df = xgb_importance_df.merge(xgb_perm_avg, on="feature", how="left")
    xgb_importance_df["family"] = xgb_importance_df["feature"].map(family_of)
    xgb_importance_df = xgb_importance_df.sort_values("gain", ascending=False).reset_index(drop=True)
    xgb_importance_df.to_csv(TABLES_DIR / "series_v3_feature_importance_xgb.csv", index=False, encoding="utf-8")

    group_dfs = {}
    for model_name in ["rf", "xgb"]:
        gdf = pd.DataFrame(group_importance_by_model[model_name])
        agg_g = gdf.groupby("family")[["mean_auc_decrease"]].mean().rename(
            columns={"mean_auc_decrease": "mean_auc_decrease_across_folds"})
        agg_g["std_auc_decrease_across_folds"] = gdf.groupby("family")["mean_auc_decrease"].std(ddof=0)
        agg_g = agg_g.reset_index()
        agg_g.to_csv(TABLES_DIR / f"series_v3_group_permutation_importance_{model_name}.csv",
                     index=False, encoding="utf-8")
        group_dfs[model_name] = agg_g

    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    for model_name, fname, title in [("rf", "series_feature_v2_v3_rf_cv.png", "Random Forest V2 config"),
                                       ("xgb", "series_feature_v2_v3_xgb_cv.png", "XGBoost V2 config")]:
        sub = fold_df[fold_df.model == model_name]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, metric, label in zip(axes, ["val_log_loss", "val_roc_auc", "val_brier"],
                                      ["Log Loss (lower better)", "ROC-AUC (higher better)", "Brier (lower better)"]):
            for fs_name, marker in [("v2", "o"), ("v3", "s")]:
                s = sub[sub.feature_set == fs_name].sort_values("fold")
                ax.plot(s["fold"], s[metric], marker=marker, label=f"Feature set {fs_name.upper()}")
            ax.set_xlabel("fold")
            ax.set_ylabel(label)
            ax.set_xticks(range(1, N_FOLDS + 1))
            ax.legend()
        fig.suptitle(f"V2 vs V3 series features, TRAIN-only CV, frozen {title} (paired ablation)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close(fig)

    write_report(fold_df, agg_df, delta_df, delta_agg, rf_importance_df, xgb_importance_df, group_dfs,
                 rf_selected, xgb_selected, rf_params, xgb_hp, xgb_fixed, xgb_n_estimators)

    print(f"\nWrote {combined.shape[0]} rows to reports/tables/series_feature_v2_v3_cv_comparison.csv")
    print("Wrote reports/tables/series_v3_feature_importance_{rf,xgb}.csv")
    print("Wrote reports/tables/series_v3_group_permutation_importance_{rf,xgb}.csv")
    print("Wrote reports/figures/series_feature_v2_v3_{rf,xgb}_cv.png")
    print("Wrote reports/phase5b3_team_form_cv_results.md")
    print(f"Wrote {AUDIT_PATH.relative_to(ROOT)}")


def interpret(delta_ll, delta_auc, delta_brier, n_ll_neg, n_auc_pos, n_folds):
    ll_improves = delta_ll < 0
    auc_improves = delta_auc > 0
    brier_improves = delta_brier < 0
    n_agree = sum([ll_improves, auc_improves, brier_improves])
    consistent = n_ll_neg >= 3 or n_auc_pos >= 3
    if n_agree == 3 and consistent:
        return "HELP"
    if n_agree == 0:
        return "DO NOT HELP"
    return "MIXED"


def write_report(fold_df, agg_df, delta_df, delta_agg, rf_importance_df, xgb_importance_df, group_dfs,
                  rf_selected, xgb_selected, rf_params, xgb_hp, xgb_fixed, xgb_n_estimators):
    def agg_row(model_name, fs_name):
        return agg_df[(agg_df.model == model_name) & (agg_df.feature_set == fs_name)].iloc[0]

    def delta_row(model_name):
        return delta_agg[delta_agg.model == model_name].iloc[0]

    md = []
    md.append("# Phase 5B.3 - Paired V2 vs V3 Team-Form Feature Evaluation (TRAIN-only CV)\n")
    md.append("**Framing.** The 4 chronological CV folds below (`data/modeling/random_forest_cv_folds_v2.csv`) "
              "are the SAME folds RF V2's and XGB V2's frozen hyperparameters were originally selected against "
              "in Phase 4B.1/4C.1 - **before V3 existed**. This is a paired development-set feature ablation "
              "under a fixed, previously-selected model configuration, not an independent estimate of future "
              "generalization. The main 1,419-match VALIDATION partition was never loaded here.\n")
    md.append("**Asymmetric interpretation (read before the verdict below).** If V3 improves under these "
              "frozen configurations, that IS evidence the new form information adds predictive signal without "
              "needing retuning. If V3 does NOT improve, that does **not** mean the form features carry no "
              "useful signal - the correct conclusion is only that *V3 did not improve performance under the "
              "frozen, previously-selected model configurations*. A later V3-specific tune could use the "
              "richer feature space differently.\n")
    md.append(f"**Regression parity.** V2-arm fold metrics here are required (by "
              "`validation/validate_phase5b3.py`) to match Phase 5B.1's own saved V2-arm rows in "
              "`reports/tables/series_feature_v1_v2_cv_comparison.csv` within a strict numeric tolerance - "
              "both use the identical data/configuration/protocol combination.\n")

    md.append("## Frozen configurations (loaded, never altered)\n")
    md.append(f"- **RF V2** (`{rf_selected['candidate_id']}`): `{rf_params}`.")
    md.append(f"- **XGB V2** (`{xgb_selected['selected_candidate_id']}`): `{xgb_hp}` + fixed `{xgb_fixed}`, "
              f"`n_estimators={xgb_n_estimators}`, no early stopping - identical for V2 and V3.\n")

    for model_name, label in [("rf", "Random Forest (frozen RF V2 config)"), ("xgb", "XGBoost (frozen XGB V2 config)")]:
        v2a, v3a = agg_row(model_name, "v2"), agg_row(model_name, "v3")
        d = delta_row(model_name)
        dfold = delta_df[delta_df.model == model_name].sort_values("fold")
        md.append(f"## {label}\n")
        md.append("| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | "
                  "mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |")
        md.append("|---|---|---|---|---|---|---|---|")
        for fs_name, a in [("V2", v2a), ("V3", v3a)]:
            md.append(f"| {fs_name} | {a['val_log_loss_mean']:.4f}±{a['val_log_loss_std']:.4f} | "
                      f"{a['val_roc_auc_mean']:.4f}±{a['val_roc_auc_std']:.4f} | "
                      f"{a['val_brier_mean']:.4f}±{a['val_brier_std']:.4f} | {a['val_accuracy_mean']:.4f} | "
                      f"{a['val_f1_mean']:.4f} | {a['train_roc_auc_mean']:.4f} | {a['train_val_auc_gap_mean']:+.4f} |")
        md.append("")
        md.append("### Paired fold-wise deltas (V3 - V2; negative=better for Log Loss/Brier, "
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
        md.append(f"Log loss improved (V3 better) in **{n_ll_neg}/{N_FOLDS}** folds; ROC-AUC improved in "
                  f"**{n_auc_pos}/{N_FOLDS}** folds; Brier improved in **{n_brier_neg}/{N_FOLDS}** folds.\n")

    md.append("## V3-only feature importance (descriptive, no feature selection performed)\n")
    md.append("RF impurity importance and XGB gain both from a full-augmented-TRAIN refit of the frozen V3 "
              "configuration; permutation importance is the average across the 4 CV folds' own "
              "fold-validation slices (still entirely inside TRAIN-only CV).\n")
    md.append("**Correlated-feature caveat.** Several of the 12 new form features are highly correlated "
              "(Phase 5B.2's own quality report found r up to 0.93 among `time_weighted_win_rate_diff`, "
              "`time_weighted_series_margin_diff` and `performance_residual_all_diff`). Permutation importance "
              "may be *shared* across correlated features - a near-zero INDIVIDUAL score does not prove a "
              "correlated feature or family carries no signal. Family-level GROUPED permutation importance "
              "(section below) is reported specifically to address this.\n")

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

    watch = ["elo_diff", "map_pool_total_matches_diff", "map_pool_best_elo_diff", "total_matches_before_diff"]
    md.append("### Do any of the 12 new form features beat the four named references?\n")
    for name, df in [("RF", rf_importance_df), ("XGB", xgb_importance_df)]:
        ref_vals = df.set_index("feature").loc[watch, "fold_val_permutation_mean"]
        form_df = df[df["family"].isin(["E_opponent_strength", "F_time_decayed", "G_form_confidence"])]
        beat = form_df[form_df["fold_val_permutation_mean"] > ref_vals.min()]
        ref_str = ", ".join(f"{k}={float(v):.4f}" for k, v in ref_vals.items())
        md.append(f"- **{name}**: reference minimum = {ref_vals.min():.4f} (of {{{ref_str}}}). "
                  f"{len(beat)} of {len(form_df)} new form features individually exceed it: "
                  + (", ".join(f"`{f}`" for f in beat.sort_values('fold_val_permutation_mean', ascending=False)['feature'])
                     if len(beat) else "none - see the grouped importance below before concluding no signal."))
    md.append("")

    md.append("## Family-level GROUPED permutation importance (E, F, G only; descriptive, NOT feature selection)\n")
    md.append("Each family's columns are permuted JOINTLY (one shared row-permutation across the whole group "
              "per repeat, 10 repeats, `random_state=42`), measuring the ROC-AUC decrease when that entire "
              "family's signal is destroyed at once - this is the correct way to read importance for a set of "
              "correlated features, since individually permuting each one leaves the others to compensate.\n")
    for model_name in ["rf", "xgb"]:
        md.append(f"**{model_name.upper()}**:\n")
        md.append("| family | mean ROC-AUC decrease (across 4 folds) | std across folds |")
        md.append("|---|---|---|")
        for _, r in group_dfs[model_name].iterrows():
            md.append(f"| {r['family']} | {r['mean_auc_decrease_across_folds']:.4f} | {r['std_auc_decrease_across_folds']:.4f} |")
        md.append("")
    md.append("Not used for feature selection, model changes, or tuning - descriptive only.\n")

    md.append("## Answering the brief\n")
    for model_name, label in [("rf", "RF"), ("xgb", "XGB")]:
        d = delta_row(model_name)
        dfold = delta_df[delta_df.model == model_name].sort_values("fold")
        n_ll = int((dfold["delta_log_loss"] < 0).sum())
        n_auc = int((dfold["delta_roc_auc"] > 0).sum())
        n_br = int((dfold["delta_brier"] < 0).sum())
        md.append(f"**{label}**: mean CV log loss "
                  f"{'improved' if d['delta_log_loss'] < 0 else 'did not improve'} ({d['delta_log_loss']:+.4f}, "
                  f"{n_ll}/{N_FOLDS} folds better); ROC-AUC "
                  f"{'improved' if d['delta_roc_auc'] > 0 else 'did not improve'} ({d['delta_roc_auc']:+.4f}, "
                  f"{n_auc}/{N_FOLDS} folds better); Brier "
                  f"{'improved' if d['delta_brier'] < 0 else 'did not improve'} ({d['delta_brier']:+.4f}, "
                  f"{n_br}/{N_FOLDS} folds better).")
    md.append("")

    rf_d, xgb_d = delta_row("rf"), delta_row("xgb")
    rf_dfold = delta_df[delta_df.model == "rf"]
    xgb_dfold = delta_df[delta_df.model == "xgb"]
    rf_verdict = interpret(rf_d["delta_log_loss"], rf_d["delta_roc_auc"], rf_d["delta_brier"],
                            int((rf_dfold["delta_log_loss"] < 0).sum()), int((rf_dfold["delta_roc_auc"] > 0).sum()), N_FOLDS)
    xgb_verdict = interpret(xgb_d["delta_log_loss"], xgb_d["delta_roc_auc"], xgb_d["delta_brier"],
                             int((xgb_dfold["delta_log_loss"] < 0).sum()), int((xgb_dfold["delta_roc_auc"] > 0).sum()), N_FOLDS)
    overall = rf_verdict if rf_verdict == xgb_verdict else "MIXED"

    md.append("## Conclusion\n")
    md.append(f"RF: **{rf_verdict}**. XGB: **{xgb_verdict}**. Combined: **{overall}**.\n")
    if overall == "HELP":
        md.append("Both models improved on all three primary metrics under the frozen, previously-selected "
                  "configurations - evidence the new form information adds predictive signal without needing "
                  "retuning.\n")
    else:
        md.append("**This does NOT mean the form features carry no useful signal.** The correct conclusion is "
                  "only that V3 did not improve performance under the frozen, previously-selected RF V2/XGB V2 "
                  "configurations (selected before V3 existed). A later V3-specific tune could use the richer "
                  "feature space differently; the grouped permutation importance above should be read before "
                  "assuming any family is uninformative, given the correlation among the new features.\n")
    md.append(f"**MAP AND FORM FEATURES {overall}** (this phase's verdict is about the 12 new form features "
              "specifically; family A-D behavior is unchanged from Phase 5B.1).\n")
    md.append("- **MAIN VALIDATION = NOT USED**")
    md.append("- **TEST = SEALED**")
    md.append("- **COLOGNE = UNTOUCHED**\n")

    (REPORTS / "phases" / "phase5b3_team_form_cv_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
