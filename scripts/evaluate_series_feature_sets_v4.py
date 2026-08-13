"""
Phase 5C.1: Paired V3 vs V4 player/roster feature-set evaluation.

Question answered: with RF V2's and XGB V2's FROZEN hyperparameters held
fixed (selected before V3 or V4 existed, never retuned here), do the 21 new
player-form/roster-stability features (series_features_v4_roster.parquet,
Phase 5C) improve series-winner prediction relative to V3 alone?

INTERPRETATION (asymmetric, same rule as Phase 5B.3 - read before the verdict)
------------------------------------------------------------------------------
If V4 improves under these frozen, previously-selected configurations, that
IS evidence the player/roster information adds predictive signal without
retuning. If overall improvement is small but the coverage-rich subgroup
(roster_form_players_min >= 5, i.e. both inferred rosters have >=5 players
with usable prior history) improves substantially, the correct reading is
"player features appear useful where historical player coverage exists, but
incomplete source coverage dilutes the aggregate gain" - NOT "no signal". If
V4 does not improve anywhere, the correct conclusion is only that "V4 did not
improve under the frozen, previously-selected model configurations" - not
that the information has zero value; a later V4-specific tune could use the
richer feature space differently. These are the SAME 4 folds RF V2/XGB V2
were originally selected against, so this is a paired development-set
ablation, not an independent generalization estimate. The main 1,419-match
VALIDATION partition is never loaded here.

REGRESSION PARITY WITH PHASE 5B.3's OWN V3 ARM
------------------------------------------------------------------------------
Phase 5B.3 already evaluated V3 + frozen RF V2 config and V3 + frozen XGB V2
config (n_estimators=98, no early stopping) through this identical protocol.
Phase 5C.1's V3 arm uses the exact same data/configuration/protocol
combination, so scripts/validate_phase5c1.py requires the V3-arm fold metrics
computed here to match Phase 5B.3's own saved
reports/tables/series_feature_v2_v3_cv_comparison.csv V3-arm rows within a
strict numeric tolerance - a material difference would indicate a
preprocessing/evaluation-harness regression, not a modeling result, and FAILS
validation.

COVERAGE-STRATIFIED DIAGNOSTIC (descriptive only - never used to tune/select)
------------------------------------------------------------------------------
~30% of rows have no usable prior player history on >=1 side (Phase 5C's own
finding), so player-performance features are structurally absent there. The
coverage-rich/cold-start split is defined ENTIRELY from a pre-match feature
(roster_form_players_min), never from the target, and is reported purely to
see whether the full-data result understates player-feature value where the
source data actually supports it.

Never touches: TEST, Cologne/post-Cologne, the 1,419-match main VALIDATION
partition, or any Phase 1-5C artifact (read-only against all of them; no
feature is added, removed or changed).

Writes:
    reports/tables/series_feature_v3_v4_cv_comparison.csv
    reports/tables/series_feature_v3_v4_coverage_diagnostic.csv
    reports/tables/series_v4_feature_importance_rf.csv
    reports/tables/series_v4_feature_importance_xgb.csv
    reports/tables/series_v4_group_permutation_importance_rf.csv
    reports/tables/series_v4_group_permutation_importance_xgb.csv
    reports/figures/series_feature_v3_v4_rf_cv.png
    reports/figures/series_feature_v3_v4_xgb_cv.png
    reports/phase5c1_player_roster_cv_results.md
    data/modeling/phase5c1_fold_preprocessing_audit.json
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
from xgboost import XGBClassifier

from _common import ROOT, REPORTS

import preprocessing_common_v3_form as pc3
import preprocessing_common_v4_roster as pc4
import preprocessing_random_forest_v2_map_pool as rf2
import preprocessing_xgboost_v2_map_pool as xgb2
import preprocessing_common_v2_map_pool as pc2
from evaluate_series_feature_sets_v2 import compute_metrics, slice_fold, evaluate_model_on_fold
from evaluate_series_feature_sets_v3 import (
    FAMILY_B_POOL_DEPTH, FAMILY_C_MATCHUP, FAMILY_D_CONFIDENCE,
    FAMILY_E_OPPONENT_STRENGTH, FAMILY_F_TIME_DECAYED, FAMILY_G_FORM_CONFIDENCE,
    grouped_permutation_importance,
)
from team_form_engine import FORM_DIRECTIONAL_FEATURES, FORM_SYMMETRIC_FEATURES
from player_roster_feature_engine import (
    ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES, ROSTER_PERFORMANCE_DIFFS,
)

CONFIG_V3_PATH = ROOT / "config" / "series_features_v3_form.yaml"
CONFIG_V4_PATH = ROOT / "config" / "series_features_v4_roster.yaml"
FEATURES_V3_PATH = ROOT / "data" / "features" / "series_features_v3_form.parquet"
FEATURES_V4_PATH = ROOT / "data" / "features" / "series_features_v4_roster.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
MODELING_DIR = ROOT / "data" / "modeling"
RF_SELECTED_CONFIG_PATH = MODELING_DIR / "random_forest_v2_selected_config.json"
XGB_SELECTED_CONFIG_PATH = MODELING_DIR / "xgboost_v2_selected_config.json"
TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"
AUDIT_PATH = MODELING_DIR / "phase5c1_fold_preprocessing_audit.json"

N_FOLDS = 4
EXPECTED_SERIES_ROWS = 9456
RANDOM_STATE = 42
COLD_START_MIN_SAMPLE = 200   # below this, the cold-start pooled diagnostic is flagged unreliable rather than suppressed

# ---------------------------------------------------------------------------
# Family grouping. A-D, E-G inherited unchanged from evaluate_series_feature_sets_v3.
# H/I/J are new (Phase 5C.1). H splits into H1-H4 subfamilies.
# ---------------------------------------------------------------------------
FAMILY_H1_ADR = ["roster_mean_adr_diff", "roster_top_adr_diff", "roster_bottom_adr_diff"]
FAMILY_H2_KAST = ["roster_mean_kast_diff", "roster_top_kast_diff", "roster_bottom_kast_diff"]
FAMILY_H3_KD = ["roster_mean_kd_balance_diff", "roster_top_kd_balance_diff", "roster_bottom_kd_balance_diff"]
FAMILY_H4_APR = ["roster_mean_assists_per_round_diff"]
FAMILY_H_PLAYER_PERFORMANCE = list(ROSTER_PERFORMANCE_DIFFS)
assert set(FAMILY_H_PLAYER_PERFORMANCE) == set(FAMILY_H1_ADR + FAMILY_H2_KAST + FAMILY_H3_KD + FAMILY_H4_APR)

FAMILY_I_ROSTER_STABILITY = [
    "recent_unique_players_10_maps_diff", "recent_unique_players_20_maps_diff",
    "core5_appearance_concentration_90d_diff", "core5_continuity_last_10_diff",
]
FAMILY_J_CONFIDENCE_EVIDENCE = [
    "roster_mean_player_history_mass_diff",
    "roster_size_min", "both_teams_have_5_inferred_players", "roster_min_player_history_mass",
    "roster_core_concentration_min", "roster_core_continuity_last10_min", "roster_form_players_min",
]
assert set(FAMILY_H_PLAYER_PERFORMANCE + FAMILY_I_ROSTER_STABILITY + FAMILY_J_CONFIDENCE_EVIDENCE) == \
    set(ROSTER_DIRECTIONAL_FEATURES) | set(ROSTER_SYMMETRIC_FEATURES)

GROUPED_FAMILIES_HIJ = {"H_player_performance": FAMILY_H_PLAYER_PERFORMANCE,
                         "I_roster_stability": FAMILY_I_ROSTER_STABILITY,
                         "J_confidence_evidence": FAMILY_J_CONFIDENCE_EVIDENCE}
GROUPED_FAMILIES_H_SUB = {"H1_adr": FAMILY_H1_ADR, "H2_kast": FAMILY_H2_KAST,
                           "H3_kd_balance": FAMILY_H3_KD, "H4_assists_per_round": FAMILY_H4_APR}


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
    if feature in FAMILY_H_PLAYER_PERFORMANCE:
        return "H_player_performance"
    if feature in FAMILY_I_ROSTER_STABILITY:
        return "I_roster_stability"
    if feature in FAMILY_J_CONFIDENCE_EVIDENCE:
        return "J_confidence_evidence"
    return "A_original_v1"


def main():
    cfg_v3 = yaml.safe_load(CONFIG_V3_PATH.read_text(encoding="utf-8"))
    target_col = cfg_v3["target"]
    roles_v3 = pc3.load_v3_roles(CONFIG_V3_PATH)
    roles_v4 = pc4.load_v4_roles(CONFIG_V4_PATH)
    assert roles_v4["target"] == target_col

    v3_df = pd.read_parquet(FEATURES_V3_PATH, engine="fastparquet")
    v4_df = pd.read_parquet(FEATURES_V4_PATH, engine="fastparquet")

    # ---- pre-flight V3 <-> V4 contract ----
    assert len(v3_df) == len(v4_df) == EXPECTED_SERIES_ROWS
    assert v3_df["match_id"].tolist() == v4_df["match_id"].tolist(), "match_id order differs V3 vs V4"
    assert v3_df[target_col].equals(v4_df[target_col]), "target differs V3 vs V4"
    assert v3_df["datetime"].equals(v4_df["datetime"]), "datetime differs V3 vs V4"
    for c in v3_df.columns:
        if pd.api.types.is_numeric_dtype(v3_df[c]):
            assert np.array_equal(v3_df[c].to_numpy(dtype=float), v4_df[c].to_numpy(dtype=float), equal_nan=True), c
        else:
            assert v3_df[c].equals(v4_df[c]), c
    new_cols = set(v4_df.columns) - set(v3_df.columns)
    assert new_cols == set(ROSTER_DIRECTIONAL_FEATURES) | set(ROSTER_SYMMETRIC_FEATURES), \
        f"expected exactly 21 new columns, got {sorted(new_cols)}"
    print(f"Pre-flight OK: V3/V4 share {EXPECTED_SERIES_ROWS} rows, identical order/target/datetime, "
          f"every V3 column preserved, exactly {len(new_cols)} new columns.")

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
          f"n_estimators={xgb_n_estimators} (no early stopping - identical for V3 and V4 arms)")

    def build_rf():
        return RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)

    def build_xgb():
        return XGBClassifier(n_estimators=xgb_n_estimators, **xgb_hp, **xgb_fixed)

    def v3_adapters():
        return (lambda df: pc2.build_augmented_training_raw(df, roles_v3),
                lambda aug: rf2.fit_preprocessing(aug, roles_v3),
                lambda df, params: rf2.transform(df, params, roles_v3),
                lambda aug: xgb2.fit_preprocessing(aug, roles_v3),
                lambda df, params: xgb2.transform(df, params, roles_v3))

    def v4_adapters():
        return (lambda df: pc2.build_augmented_training_raw(df, roles_v4),
                lambda aug: rf2.fit_preprocessing(aug, roles_v4),
                lambda df, params: rf2.transform(df, params, roles_v4),
                lambda aug: xgb2.fit_preprocessing(aug, roles_v4),
                lambda df, params: xgb2.transform(df, params, roles_v4))

    feature_sets = {"v3": {"df": v3_df, "adapters": v3_adapters()},
                     "v4": {"df": v4_df, "adapters": v4_adapters()}}

    fold_rows = []
    perm_importance_by_model = {"rf": [], "xgb": []}
    group_importance_full = {"rf": [], "xgb": []}
    group_importance_covrich = {"rf": [], "xgb": []}
    group_h_sub_full = {"rf": [], "xgb": []}
    group_h_sub_covrich = {"rf": [], "xgb": []}
    audit_entries = []

    # pooled OOF accumulators: [model][feature_set] -> lists across folds
    pooled = {m: {fs: {"y_true": [], "proba": [], "pred": [], "match_id": []}
                  for fs in ("v3", "v4")} for m in ("rf", "xgb")}
    coverage_lookup = {}   # match_id (str) -> roster_form_players_min, accumulated across folds

    for fold in range(1, N_FOLDS + 1):
        # coverage lookup for THIS fold, derived once from V4 (a pre-match feature, never the target)
        v4_fold_train_raw, v4_fold_val_raw, _, _ = slice_fold(cv_df, v4_df, fold)
        fold_coverage = dict(zip(v4_fold_val_raw["match_id"].astype(str),
                                  v4_fold_val_raw["roster_form_players_min"]))
        coverage_lookup.update(fold_coverage)

        for fs_name, fs in feature_sets.items():
            augment_fn, rf_fit, rf_transform, xgb_fit, xgb_transform = fs["adapters"]
            fold_train_raw, fold_val_raw, fold_train_ids, fold_val_ids = slice_fold(cv_df, fs["df"], fold)
            assert fold_train_ids.isdisjoint(fold_val_ids)
            val_match_ids = fold_val_raw["match_id"].astype(str).to_numpy()
            val_coverage = np.array([fold_coverage.get(mid, -1) for mid in val_match_ids])

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

                # ---- pooled OOF accumulation (no refit - reuse the fitted model) ----
                proba_val = result["model"].predict_proba(result["X_val"])[:, 1]
                pred_val = result["model"].predict(result["X_val"])
                pooled[model_name][fs_name]["y_true"].append(result["y_val"])
                pooled[model_name][fs_name]["proba"].append(proba_val)
                pooled[model_name][fs_name]["pred"].append(pred_val)
                pooled[model_name][fs_name]["match_id"].append(val_match_ids)

                # ---- coverage-rich / cold-start SUBGROUP fold metrics (descriptive) ----
                rich_mask = val_coverage >= 5
                cold_mask = val_coverage == 0
                for subgroup_name, mask in [("coverage_rich", rich_mask), ("cold_start", cold_mask)]:
                    if mask.sum() > 0:
                        sm = compute_metrics(result["y_val"][mask], proba_val[mask], pred_val[mask])
                        fold_rows.append({
                            "model": model_name, "feature_set": fs_name, "fold": fold,
                            "subgroup": subgroup_name, "row_type": "subgroup_fold",
                            "n_fold_val": int(mask.sum()),
                            "val_accuracy": sm["accuracy"], "val_roc_auc": sm["roc_auc"],
                            "val_log_loss": sm["log_loss"], "val_brier": sm["brier"], "val_f1": sm["f1"],
                        })

                if fs_name == "v4":
                    perm = permutation_importance(result["model"], result["X_val"], result["y_val"],
                                                    scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
                    perm_df = pd.DataFrame({"feature": result["feature_names"], "importance": perm.importances_mean})
                    perm_importance_by_model[model_name].append(perm_df)

                    grp_full = grouped_permutation_importance(
                        result["model"], result["X_val"], result["y_val"], result["feature_names"],
                        GROUPED_FAMILIES_HIJ, n_repeats=10, random_state=RANDOM_STATE)
                    for gname, stats in grp_full.items():
                        group_importance_full[model_name].append(
                            {"fold": fold, "family": gname, "mean_auc_decrease": stats["mean"]})

                    grp_hsub_full = grouped_permutation_importance(
                        result["model"], result["X_val"], result["y_val"], result["feature_names"],
                        GROUPED_FAMILIES_H_SUB, n_repeats=10, random_state=RANDOM_STATE)
                    for gname, stats in grp_hsub_full.items():
                        group_h_sub_full[model_name].append(
                            {"fold": fold, "family": gname, "mean_auc_decrease": stats["mean"]})

                    # coverage-rich grouped importance (descriptive) - same fitted model,
                    # scored on the coverage-rich slice of this fold's own validation set
                    if rich_mask.sum() >= 30:
                        X_rich = result["X_val"][rich_mask]
                        y_rich = result["y_val"][rich_mask]
                        grp_rich = grouped_permutation_importance(
                            result["model"], X_rich, y_rich, result["feature_names"],
                            GROUPED_FAMILIES_HIJ, n_repeats=10, random_state=RANDOM_STATE)
                        for gname, stats in grp_rich.items():
                            group_importance_covrich[model_name].append(
                                {"fold": fold, "family": gname, "mean_auc_decrease": stats["mean"]})

                        grp_hsub_rich = grouped_permutation_importance(
                            result["model"], X_rich, y_rich, result["feature_names"],
                            GROUPED_FAMILIES_H_SUB, n_repeats=10, random_state=RANDOM_STATE)
                        for gname, stats in grp_hsub_rich.items():
                            group_h_sub_covrich[model_name].append(
                                {"fold": fold, "family": gname, "mean_auc_decrease": stats["mean"]})

            print(f"  fold {fold} / {fs_name}: done")

    fold_df = pd.DataFrame(fold_rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)

    main_fold = fold_df[fold_df.get("row_type").isna()].drop(columns=["subgroup", "row_type"], errors="ignore")
    subgroup_fold = fold_df[fold_df.get("row_type") == "subgroup_fold"]

    agg_records = []
    for (model_name, fs_name), g in main_fold.groupby(["model", "feature_set"]):
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
        v3_f = main_fold[(main_fold.model == model_name) & (main_fold.feature_set == "v3")].set_index("fold")
        v4_f = main_fold[(main_fold.model == model_name) & (main_fold.feature_set == "v4")].set_index("fold")
        for fold in range(1, N_FOLDS + 1):
            delta_rows.append({
                "model": model_name, "fold": fold,
                "delta_log_loss": v4_f.loc[fold, "val_log_loss"] - v3_f.loc[fold, "val_log_loss"],
                "delta_roc_auc": v4_f.loc[fold, "val_roc_auc"] - v3_f.loc[fold, "val_roc_auc"],
                "delta_brier": v4_f.loc[fold, "val_brier"] - v3_f.loc[fold, "val_brier"],
                "delta_accuracy": v4_f.loc[fold, "val_accuracy"] - v3_f.loc[fold, "val_accuracy"],
                "delta_f1": v4_f.loc[fold, "val_f1"] - v3_f.loc[fold, "val_f1"],
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_agg = delta_df.groupby("model")[["delta_log_loss", "delta_roc_auc", "delta_brier",
                                            "delta_accuracy", "delta_f1"]].mean().reset_index()
    delta_agg["fold"] = "mean"

    combined = pd.concat([
        main_fold.assign(row_type="fold"), agg_df,
        delta_df.assign(row_type="paired_delta"), delta_agg.assign(row_type="paired_delta_mean"),
    ], ignore_index=True, sort=False)
    combined.to_csv(TABLES_DIR / "series_feature_v3_v4_cv_comparison.csv", index=False, encoding="utf-8")

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": audit_entries}, f, indent=2)

    # =======================================================================
    # Coverage diagnostic: subgroup fold-agg + paired deltas, and pooled OOF
    # (full / coverage_rich / cold_start), per model.
    # =======================================================================
    diag_rows = []
    for model_name in ["rf", "xgb"]:
        for subgroup in ["coverage_rich", "cold_start"]:
            sub = subgroup_fold[(subgroup_fold.model == model_name) & (subgroup_fold.subgroup == subgroup)]
            v3s = sub[sub.feature_set == "v3"].set_index("fold")
            v4s = sub[sub.feature_set == "v4"].set_index("fold")
            common_folds = sorted(set(v3s.index) & set(v4s.index))
            for fold in common_folds:
                diag_rows.append({
                    "model": model_name, "subgroup": subgroup, "row_type": "fold", "fold": fold,
                    "n": int(v4s.loc[fold, "n_fold_val"]),
                    "v3_accuracy": v3s.loc[fold, "val_accuracy"], "v4_accuracy": v4s.loc[fold, "val_accuracy"],
                    "delta_accuracy": v4s.loc[fold, "val_accuracy"] - v3s.loc[fold, "val_accuracy"],
                    "v3_roc_auc": v3s.loc[fold, "val_roc_auc"], "v4_roc_auc": v4s.loc[fold, "val_roc_auc"],
                    "delta_roc_auc": v4s.loc[fold, "val_roc_auc"] - v3s.loc[fold, "val_roc_auc"],
                    "v3_log_loss": v3s.loc[fold, "val_log_loss"], "v4_log_loss": v4s.loc[fold, "val_log_loss"],
                    "delta_log_loss": v4s.loc[fold, "val_log_loss"] - v3s.loc[fold, "val_log_loss"],
                    "v3_brier": v3s.loc[fold, "val_brier"], "v4_brier": v4s.loc[fold, "val_brier"],
                    "delta_brier": v4s.loc[fold, "val_brier"] - v3s.loc[fold, "val_brier"],
                    "v3_f1": v3s.loc[fold, "val_f1"], "v4_f1": v4s.loc[fold, "val_f1"],
                    "delta_f1": v4s.loc[fold, "val_f1"] - v3s.loc[fold, "val_f1"],
                })
            if common_folds:
                fdf = pd.DataFrame([r for r in diag_rows if r["subgroup"] == subgroup
                                     and r["model"] == model_name and r["row_type"] == "fold"])
                diag_rows.append({
                    "model": model_name, "subgroup": subgroup, "row_type": "fold_agg_mean", "fold": "mean",
                    "n": float(fdf["n"].mean()),
                    "v3_accuracy": fdf["v3_accuracy"].mean(), "v4_accuracy": fdf["v4_accuracy"].mean(),
                    "delta_accuracy": fdf["delta_accuracy"].mean(),
                    "v3_roc_auc": fdf["v3_roc_auc"].mean(), "v4_roc_auc": fdf["v4_roc_auc"].mean(),
                    "delta_roc_auc": fdf["delta_roc_auc"].mean(),
                    "v3_log_loss": fdf["v3_log_loss"].mean(), "v4_log_loss": fdf["v4_log_loss"].mean(),
                    "delta_log_loss": fdf["delta_log_loss"].mean(),
                    "v3_brier": fdf["v3_brier"].mean(), "v4_brier": fdf["v4_brier"].mean(),
                    "delta_brier": fdf["delta_brier"].mean(),
                    "v3_f1": fdf["v3_f1"].mean(), "v4_f1": fdf["v4_f1"].mean(),
                    "delta_f1": fdf["delta_f1"].mean(),
                })

    # ---- pooled OOF (full / coverage_rich / cold_start) ----
    pooled_records = {}
    for model_name in ["rf", "xgb"]:
        for fs_name in ["v3", "v4"]:
            y_true = np.concatenate(pooled[model_name][fs_name]["y_true"])
            proba = np.concatenate(pooled[model_name][fs_name]["proba"])
            pred = np.concatenate(pooled[model_name][fs_name]["pred"])
            mids = np.concatenate(pooled[model_name][fs_name]["match_id"])
            cov = np.array([coverage_lookup.get(m, -1) for m in mids])
            pooled_records[(model_name, fs_name)] = {"y_true": y_true, "proba": proba, "pred": pred, "cov": cov}

    pooled_rows = []
    for model_name in ["rf", "xgb"]:
        for subgroup, mask_fn in [("full", lambda c: np.ones_like(c, dtype=bool)),
                                    ("coverage_rich", lambda c: c >= 5),
                                    ("cold_start", lambda c: c == 0)]:
            v3r = pooled_records[(model_name, "v3")]
            v4r = pooled_records[(model_name, "v4")]
            m3 = mask_fn(v3r["cov"])
            m4 = mask_fn(v4r["cov"])
            n3, n4 = int(m3.sum()), int(m4.sum())
            if n3 == 0 or n4 == 0:
                continue
            reliable = True
            if subgroup == "cold_start" and min(n3, n4) < COLD_START_MIN_SAMPLE:
                reliable = False
            sm3 = compute_metrics(v3r["y_true"][m3], v3r["proba"][m3], v3r["pred"][m3])
            sm4 = compute_metrics(v4r["y_true"][m4], v4r["proba"][m4], v4r["pred"][m4])
            correct3 = int((v3r["pred"][m3] == v3r["y_true"][m3]).sum())
            correct4 = int((v4r["pred"][m4] == v4r["y_true"][m4]).sum())
            pooled_rows.append({
                "model": model_name, "subgroup": subgroup, "row_type": "pooled_oof",
                "n_v3": n3, "n_v4": n4, "reliable_sample_size": reliable,
                "v3_accuracy": sm3["accuracy"], "v4_accuracy": sm4["accuracy"],
                "delta_accuracy_pp": 100 * (sm4["accuracy"] - sm3["accuracy"]),
                "v3_roc_auc": sm3["roc_auc"], "v4_roc_auc": sm4["roc_auc"], "delta_roc_auc": sm4["roc_auc"] - sm3["roc_auc"],
                "v3_log_loss": sm3["log_loss"], "v4_log_loss": sm4["log_loss"], "delta_log_loss": sm4["log_loss"] - sm3["log_loss"],
                "v3_brier": sm3["brier"], "v4_brier": sm4["brier"], "delta_brier": sm4["brier"] - sm3["brier"],
                "v3_f1": sm3["f1"], "v4_f1": sm4["f1"], "delta_f1": sm4["f1"] - sm3["f1"],
                "v3_correct": correct3, "v4_correct": correct4, "delta_correct": correct4 - correct3,
            })
    pooled_df = pd.DataFrame(pooled_rows)

    diag_df = pd.concat([pd.DataFrame(diag_rows), pooled_df], ignore_index=True, sort=False)
    diag_df.to_csv(TABLES_DIR / "series_feature_v3_v4_coverage_diagnostic.csv", index=False, encoding="utf-8")

    # =======================================================================
    # Full augmented-TRAIN V4 refits, for stable impurity/gain/weight importance
    # =======================================================================
    full_train_df_v4 = v4_df[v4_df["match_id"].isin(full_train_ids)].reset_index(drop=True)
    assert len(full_train_df_v4) == 6619
    augmented_full_v4 = pc2.build_augmented_training_raw(full_train_df_v4, roles_v4)
    pc2.assert_augmented_symmetry(augmented_full_v4, roles_v4)

    rf_full_params = rf2.fit_preprocessing(augmented_full_v4, roles_v4)
    X_full_rf, rf_names = rf2.transform(augmented_full_v4, rf_full_params, roles_v4)
    y_full = augmented_full_v4[target_col].to_numpy(dtype=float)
    rf_full_model = build_rf()
    rf_full_model.fit(X_full_rf, y_full)

    xgb_full_params = xgb2.fit_preprocessing(augmented_full_v4, roles_v4)
    X_full_xgb, xgb_names = xgb2.transform(augmented_full_v4, xgb_full_params, roles_v4)
    xgb_full_model = build_xgb()
    xgb_full_model.fit(X_full_xgb, y_full)

    rf_importance_df = pd.DataFrame({"feature": rf_names, "impurity_importance": rf_full_model.feature_importances_})
    rf_perm_avg = pd.concat(perm_importance_by_model["rf"]).groupby("feature")["importance"] \
        .agg(["mean", "std"]).rename(columns={"mean": "fold_val_permutation_mean", "std": "fold_val_permutation_std"})
    rf_importance_df = rf_importance_df.merge(rf_perm_avg, on="feature", how="left")
    rf_importance_df["family"] = rf_importance_df["feature"].map(family_of)
    rf_importance_df = rf_importance_df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)
    rf_importance_df.to_csv(TABLES_DIR / "series_v4_feature_importance_rf.csv", index=False, encoding="utf-8")

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
    xgb_importance_df.to_csv(TABLES_DIR / "series_v4_feature_importance_xgb.csv", index=False, encoding="utf-8")

    # ---- grouped importance tables: full-data + coverage-rich, H/I/J + H1-H4 ----
    group_dfs = {}
    for model_name in ["rf", "xgb"]:
        parts = []
        for scope, store in [("full_data", group_importance_full), ("coverage_rich", group_importance_covrich)]:
            gdf = pd.DataFrame(store[model_name])
            if len(gdf) == 0:
                continue
            agg_g = gdf.groupby("family")["mean_auc_decrease"].agg(["mean", "std"]).reset_index()
            agg_g.columns = ["family", "mean_auc_decrease_across_folds", "std_auc_decrease_across_folds"]
            agg_g["scope"] = scope
            parts.append(agg_g)
        for scope, store in [("full_data", group_h_sub_full), ("coverage_rich", group_h_sub_covrich)]:
            gdf = pd.DataFrame(store[model_name])
            if len(gdf) == 0:
                continue
            agg_g = gdf.groupby("family")["mean_auc_decrease"].agg(["mean", "std"]).reset_index()
            agg_g.columns = ["family", "mean_auc_decrease_across_folds", "std_auc_decrease_across_folds"]
            agg_g["scope"] = scope
            parts.append(agg_g)
        gcombined = pd.concat(parts, ignore_index=True)
        gcombined.to_csv(TABLES_DIR / f"series_v4_group_permutation_importance_{model_name}.csv",
                         index=False, encoding="utf-8")
        group_dfs[model_name] = gcombined

    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    for model_name, fname, title in [("rf", "series_feature_v3_v4_rf_cv.png", "Random Forest V2 config"),
                                       ("xgb", "series_feature_v3_v4_xgb_cv.png", "XGBoost V2 config")]:
        sub = main_fold[main_fold.model == model_name]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, metric, label in zip(axes, ["val_log_loss", "val_roc_auc", "val_brier"],
                                      ["Log Loss (lower better)", "ROC-AUC (higher better)", "Brier (lower better)"]):
            for fs_name, marker in [("v3", "o"), ("v4", "s")]:
                s = sub[sub.feature_set == fs_name].sort_values("fold")
                ax.plot(s["fold"], s[metric], marker=marker, label=f"Feature set {fs_name.upper()}")
            ax.set_xlabel("fold")
            ax.set_ylabel(label)
            ax.set_xticks(range(1, N_FOLDS + 1))
            ax.legend()
        fig.suptitle(f"V3 vs V4 series features, TRAIN-only CV, frozen {title} (paired ablation)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close(fig)

    write_report(main_fold, agg_df, delta_df, delta_agg, diag_df, rf_importance_df, xgb_importance_df, group_dfs,
                 rf_selected, xgb_selected, rf_params, xgb_hp, xgb_fixed, xgb_n_estimators)

    print(f"\nWrote {combined.shape[0]} rows to reports/tables/series_feature_v3_v4_cv_comparison.csv")
    print(f"Wrote {diag_df.shape[0]} rows to reports/tables/series_feature_v3_v4_coverage_diagnostic.csv")
    print("Wrote reports/tables/series_v4_feature_importance_{rf,xgb}.csv")
    print("Wrote reports/tables/series_v4_group_permutation_importance_{rf,xgb}.csv")
    print("Wrote reports/figures/series_feature_v3_v4_{rf,xgb}_cv.png")
    print("Wrote reports/phase5c1_player_roster_cv_results.md")
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


def write_report(fold_df, agg_df, delta_df, delta_agg, diag_df, rf_importance_df, xgb_importance_df, group_dfs,
                  rf_selected, xgb_selected, rf_params, xgb_hp, xgb_fixed, xgb_n_estimators):
    def agg_row(model_name, fs_name):
        return agg_df[(agg_df.model == model_name) & (agg_df.feature_set == fs_name)].iloc[0]

    def delta_row(model_name):
        return delta_agg[delta_agg.model == model_name].iloc[0]

    md = []
    md.append("# Phase 5C.1 - Paired V3 vs V4 Player/Roster Feature Evaluation (TRAIN-only CV)\n")
    md.append("**Framing.** The 4 chronological CV folds below (`data/modeling/random_forest_cv_folds_v2.csv`) "
              "are the SAME folds RF V2's and XGB V2's frozen hyperparameters were originally selected against, "
              "before V3 or V4 existed. This is a paired development-set feature ablation under a fixed, "
              "previously-selected model configuration, not an independent estimate of future generalization. "
              "The main 1,419-match VALIDATION partition was never loaded here.\n")
    md.append("**Asymmetric interpretation.** If V4 improves, that IS evidence the player/roster information "
              "adds signal without retuning. If overall improvement is small but the coverage-rich subgroup "
              "(`roster_form_players_min >= 5`) improves substantially, the correct reading is that player "
              "features appear useful where historical coverage exists but incomplete source coverage dilutes "
              "the aggregate gain - NOT that the information has no value. If V4 does not improve anywhere, the "
              "only correct conclusion is that it did not improve under these frozen, previously-selected "
              "configurations; a later V4-specific tune could use the feature space differently.\n")
    md.append("**Regression parity.** V3-arm fold metrics here are required (by `scripts/validate_phase5c1.py`) "
              "to match Phase 5B.3's own saved V3-arm rows in `reports/tables/series_feature_v2_v3_cv_comparison.csv` "
              "within a strict numeric tolerance.\n")

    md.append("## Frozen configurations (loaded, never altered)\n")
    md.append(f"- **RF V2** (`{rf_selected['candidate_id']}`): `{rf_params}`.")
    md.append(f"- **XGB V2** (`{xgb_selected['selected_candidate_id']}`): `{xgb_hp}` + fixed `{xgb_fixed}`, "
              f"`n_estimators={xgb_n_estimators}`, no early stopping - identical for V3 and V4.\n")

    md.append("## Full-data results\n")
    for model_name, label in [("rf", "Random Forest (frozen RF V2 config)"), ("xgb", "XGBoost (frozen XGB V2 config)")]:
        v3a, v4a = agg_row(model_name, "v3"), agg_row(model_name, "v4")
        d = delta_row(model_name)
        dfold = delta_df[delta_df.model == model_name].sort_values("fold")
        md.append(f"### {label}\n")
        md.append("| feature set | mean CV log loss | mean CV ROC-AUC | mean CV Brier | mean CV accuracy | "
                  "mean CV F1 | mean train ROC-AUC | mean train-val AUC gap |")
        md.append("|---|---|---|---|---|---|---|---|")
        for fs_name, a in [("V3", v3a), ("V4", v4a)]:
            md.append(f"| {fs_name} | {a['val_log_loss_mean']:.4f}±{a['val_log_loss_std']:.4f} | "
                      f"{a['val_roc_auc_mean']:.4f}±{a['val_roc_auc_std']:.4f} | "
                      f"{a['val_brier_mean']:.4f}±{a['val_brier_std']:.4f} | {a['val_accuracy_mean']:.4f} | "
                      f"{a['val_f1_mean']:.4f} | {a['train_roc_auc_mean']:.4f} | {a['train_val_auc_gap_mean']:+.4f} |")
        md.append("")
        md.append("Paired fold-wise deltas (V4 - V3):\n")
        md.append("| fold | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |")
        md.append("|---|---|---|---|---|---|")
        for _, r in dfold.iterrows():
            md.append(f"| {int(r['fold'])} | {r['delta_log_loss']:+.4f} | {r['delta_roc_auc']:+.4f} | "
                      f"{r['delta_brier']:+.4f} | {r['delta_accuracy']:+.4f} | {r['delta_f1']:+.4f} |")
        md.append(f"| **mean** | **{d['delta_log_loss']:+.4f}** | **{d['delta_roc_auc']:+.4f}** | "
                  f"**{d['delta_brier']:+.4f}** | **{d['delta_accuracy']:+.4f}** | **{d['delta_f1']:+.4f}** |\n")
        n_ll = int((dfold["delta_log_loss"] < 0).sum())
        n_auc = int((dfold["delta_roc_auc"] > 0).sum())
        n_br = int((dfold["delta_brier"] < 0).sum())
        md.append(f"Log loss improved in **{n_ll}/{N_FOLDS}** folds; ROC-AUC improved in **{n_auc}/{N_FOLDS}**; "
                  f"Brier improved in **{n_br}/{N_FOLDS}**.\n")

        pooled = diag_df[(diag_df.model == model_name) & (diag_df.row_type == "pooled_oof")
                          & (diag_df.subgroup == "full")].iloc[0]
        md.append("**Pooled TRAIN-only out-of-fold result** (all 4 folds' validation predictions concatenated):\n")
        md.append(f"- n = {pooled['n_v3']:.0f} (V3) / {pooled['n_v4']:.0f} (V4) | "
                  f"Accuracy {pooled['v3_accuracy']:.4f} -> {pooled['v4_accuracy']:.4f} "
                  f"({pooled['delta_accuracy_pp']:+.2f} pp) | ROC-AUC {pooled['v3_roc_auc']:.4f} -> "
                  f"{pooled['v4_roc_auc']:.4f} ({pooled['delta_roc_auc']:+.4f}) | Log Loss "
                  f"{pooled['v3_log_loss']:.4f} -> {pooled['v4_log_loss']:.4f} ({pooled['delta_log_loss']:+.4f}) | "
                  f"Brier {pooled['v3_brier']:.4f} -> {pooled['v4_brier']:.4f} ({pooled['delta_brier']:+.4f}) | "
                  f"F1 {pooled['v3_f1']:.4f} -> {pooled['v4_f1']:.4f} ({pooled['delta_f1']:+.4f})")
        md.append(f"- Correct predictions: V3 = {pooled['v3_correct']:.0f}, V4 = {pooled['v4_correct']:.0f}, "
                  f"**V4 - V3 = {pooled['delta_correct']:+.0f} additional correct matches**.\n")

    md.append("## Coverage-stratified diagnostic (descriptive only)\n")
    md.append("Defined ENTIRELY from a pre-match feature (`roster_form_players_min`), never the target. "
              "Coverage-rich = both inferred rosters have >=5 players with usable prior history. Never used "
              "to tune, select, or modify V4.\n")
    for model_name in ["rf", "xgb"]:
        md.append(f"### {model_name.upper()}\n")
        for subgroup, label in [("coverage_rich", "Coverage-rich (roster_form_players_min >= 5)"),
                                  ("cold_start", "Cold-start (roster_form_players_min == 0)")]:
            agg = diag_df[(diag_df.model == model_name) & (diag_df.subgroup == subgroup)
                          & (diag_df.row_type == "fold_agg_mean")]
            pooled = diag_df[(diag_df.model == model_name) & (diag_df.subgroup == subgroup)
                             & (diag_df.row_type == "pooled_oof")]
            md.append(f"**{label}**\n")
            if len(agg):
                r = agg.iloc[0]
                md.append(f"- Fold-mean (n≈{r['n']:.0f}/fold): Δ log loss {r['delta_log_loss']:+.4f}, "
                          f"Δ ROC-AUC {r['delta_roc_auc']:+.4f}, Δ Brier {r['delta_brier']:+.4f}, "
                          f"Δ accuracy {r['delta_accuracy']:+.4f}, Δ F1 {r['delta_f1']:+.4f}")
            if len(pooled):
                p = pooled.iloc[0]
                reliability_note = "" if p["reliable_sample_size"] else " (small sample - interpret cautiously)"
                md.append(f"- Pooled OOF (n={p['n_v3']:.0f}/{p['n_v4']:.0f}){reliability_note}: "
                          f"Accuracy {p['v3_accuracy']:.4f} -> {p['v4_accuracy']:.4f} "
                          f"({p['delta_accuracy_pp']:+.2f} pp), ROC-AUC {p['v3_roc_auc']:.4f} -> "
                          f"{p['v4_roc_auc']:.4f} ({p['delta_roc_auc']:+.4f}), Log Loss {p['v3_log_loss']:.4f} -> "
                          f"{p['v4_log_loss']:.4f} ({p['delta_log_loss']:+.4f}), correct: {p['v3_correct']:.0f} -> "
                          f"{p['v4_correct']:.0f} ({p['delta_correct']:+.0f})")
            md.append("")

    md.append("## V4-only feature importance (descriptive, no feature selection performed)\n")
    md.append("**Correlated-feature caveat.** Several new player features are highly correlated (Phase 5C's own "
              "quality report found r=0.965 between core-5 concentration and continuity). Permutation importance "
              "may be *shared* across correlated features - a near-zero individual score does not prove a "
              "correlated feature or family carries no signal.\n")
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

    md.append("## Family-level GROUPED permutation importance: full-data vs. coverage-rich\n")
    md.append("H = player performance (ADR/KAST/KD-balance/assists-per-round), I = roster stability, "
              "J = confidence/evidence. Each family's columns are permuted JOINTLY per repeat (10 repeats, "
              "`random_state=42`); reported is the ROC-AUC decrease averaged across folds. Coverage-rich is "
              "computed on the same fitted fold model, scored on that fold's coverage-rich validation slice "
              "only (still descriptive, no retraining).\n")
    for model_name in ["rf", "xgb"]:
        gdf = group_dfs[model_name]
        md.append(f"**{model_name.upper()} - H/I/J:**\n")
        md.append("| family | scope | mean ROC-AUC decrease | std across folds |")
        md.append("|---|---|---|---|")
        for fam in ["H_player_performance", "I_roster_stability", "J_confidence_evidence"]:
            for scope in ["full_data", "coverage_rich"]:
                row = gdf[(gdf.family == fam) & (gdf.scope == scope)]
                if len(row):
                    r = row.iloc[0]
                    md.append(f"| {fam} | {scope} | {r['mean_auc_decrease_across_folds']:.4f} | "
                              f"{r['std_auc_decrease_across_folds']:.4f} |")
        md.append("")
        md.append(f"**{model_name.upper()} - H1-H4 subfamilies:**\n")
        md.append("| subfamily | scope | mean ROC-AUC decrease | std across folds |")
        md.append("|---|---|---|---|")
        for fam in ["H1_adr", "H2_kast", "H3_kd_balance", "H4_assists_per_round"]:
            for scope in ["full_data", "coverage_rich"]:
                row = gdf[(gdf.family == fam) & (gdf.scope == scope)]
                if len(row):
                    r = row.iloc[0]
                    md.append(f"| {fam} | {scope} | {r['mean_auc_decrease_across_folds']:.4f} | "
                              f"{r['std_auc_decrease_across_folds']:.4f} |")
        md.append("")
    md.append("Not used for feature selection, model changes, or tuning - descriptive only.\n")

    # ---- verdict ----
    rf_d, xgb_d = delta_row("rf"), delta_row("xgb")
    rf_dfold = delta_df[delta_df.model == "rf"]
    xgb_dfold = delta_df[delta_df.model == "xgb"]
    rf_full_verdict = interpret(rf_d["delta_log_loss"], rf_d["delta_roc_auc"], rf_d["delta_brier"],
                                 int((rf_dfold["delta_log_loss"] < 0).sum()), int((rf_dfold["delta_roc_auc"] > 0).sum()), N_FOLDS)
    xgb_full_verdict = interpret(xgb_d["delta_log_loss"], xgb_d["delta_roc_auc"], xgb_d["delta_brier"],
                                  int((xgb_dfold["delta_log_loss"] < 0).sum()), int((xgb_dfold["delta_roc_auc"] > 0).sum()), N_FOLDS)

    rich_deltas = diag_df[(diag_df.row_type == "fold_agg_mean") & (diag_df.subgroup == "coverage_rich")]
    rich_improves = bool(len(rich_deltas)) and (rich_deltas["delta_roc_auc"] > 0).all() and (rich_deltas["delta_log_loss"] < 0).all()
    full_helps = rf_full_verdict == "HELP" and xgb_full_verdict == "HELP"
    full_none = rf_full_verdict == "DO NOT HELP" and xgb_full_verdict == "DO NOT HELP"

    if full_helps:
        overall_verdict = "PLAYER FEATURES HELP CLEARLY"
        verdict_prose = ("Both models improved on all three primary metrics under the frozen, previously-selected "
                         "configurations, across the full TRAIN-only development set.")
    elif rich_improves and not full_helps:
        overall_verdict = "PLAYER FEATURES HELP BUT COVERAGE-LIMITED"
        verdict_prose = ("The full-data result is muted, but the coverage-rich subgroup - where both teams "
                         "actually have usable prior player history - shows a consistent improvement, "
                         "suggesting incomplete source coverage (~30% cold-start) dilutes the aggregate gain "
                         "rather than the features carrying no signal.")
    elif full_none:
        overall_verdict = "DO NOT HELP UNDER FROZEN CONFIGS"
        verdict_prose = ("Neither model improved under the frozen, previously-selected RF V2/XGB V2 "
                         "configurations. This does NOT mean the player/roster information has zero value - "
                         "only that it did not help THESE configurations, which were selected before V4 "
                         "existed. A later V4-specific tune could use the richer feature space differently.")
    else:
        overall_verdict = "MIXED"
        verdict_prose = ("RF and XGB disagree, or the signal is inconsistent across folds/subgroups. Reported "
                         "as-is rather than resolved into a stronger claim.")

    md.append("## Verdict\n")
    md.append(f"RF full-data: **{rf_full_verdict}**. XGB full-data: **{xgb_full_verdict}**.\n")
    md.append(f"**{overall_verdict}**\n")
    md.append(verdict_prose + "\n")
    md.append("- **MAIN VALIDATION = NOT USED**")
    md.append("- **TEST = SEALED**")
    md.append("- **COLOGNE = UNTOUCHED**\n")

    (REPORTS / "phase5c1_player_roster_cv_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
