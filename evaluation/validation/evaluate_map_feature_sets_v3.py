"""
Phase 6C, Stage B: TRAIN-only paired V2-rich vs V3-modern-map ablation.

Question answered: with RF's and XGB's FROZEN Phase 6B configurations held
fixed (map_random_forest_v1_selected_config.json's `random_007`,
map_xgboost_v1_selected_config.json's `random_013` with
`final_n_estimators=124` and no early stopping - selected BEFORE V3 existed,
never retuned here), do the 25 new modern-selected-map features
(map_features_v3_modern_map.parquet, Phase 6C) improve known-map prediction
relative to V2 rich (map_features_v2_rich.parquet, Phase 6A) alone?

Runs ONLY over `data/modeling/map_cv_folds_v1.csv` (TRAIN-only, reused
byte-identically). The 1,129-map main VALIDATION partition
(`data/modeling/map_split_v1.csv`) is NEVER opened by this script - not
avoided by convention, structurally absent (this script contains no read of
that file at all). TEST and Cologne are never opened either.

INTERPRETATION
------------------------------------------------------------------------
RF's/XGB's hyperparameters were selected via CV on V2-era features, before
V3 existed. If V3 improves under these frozen, previously-selected
configurations, that IS evidence the new features add signal without
retuning. If V3 does NOT improve, the correct conclusion is only that "V3
did not improve performance under the frozen, previously-selected model
configurations" - not that the new features carry no signal at all. No
feature is added, removed, or reweighted based on these results (Phase 6C is
the final feature-engineering experiment, brief sections 27/32).

Writes:
    reports/tables/map_feature_v2_v3_cv_comparison.csv
    reports/tables/map_v3_feature_importance_rf.csv
    reports/tables/map_v3_feature_importance_xgb.csv
    reports/tables/map_v3_group_permutation_importance.csv
    reports/figures/map_feature_v2_v3_rf_cv.png
    reports/figures/map_feature_v2_v3_xgb_cv.png
    reports/phase6c_modern_map_cv_results.md
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from _common import ROOT, REPORTS
from training.map_models.map_modeling_common import (
    N_FOLDS, compute_metrics, fold_frames, load_cv_manifest, load_features as load_v2_features,
    series_macro_metrics,
)
from feature_engineering.preprocessing.preprocessing_common_map_v2 import load_map_v2_roles, build_augmented_training_raw as build_aug_v2
import feature_engineering.preprocessing.preprocessing_random_forest_map_v2 as prep_rf_v2
import feature_engineering.preprocessing.preprocessing_xgboost_map_v2 as prep_xgb_v2
from feature_engineering.preprocessing.preprocessing_common_map_v3 import load_map_v3_roles
from feature_engineering.preprocessing.preprocessing_common_map_v2 import build_augmented_training_raw as build_aug_v3   # roles-driven, reusable
import feature_engineering.preprocessing.preprocessing_random_forest_map_v3 as prep_rf_v3
import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb_v3

TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"
V3_PATH = ROOT / "data" / "features" / "map_features_v3_modern_map.parquet"
V3_CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"
V2_CONFIG_PATH = ROOT / "config" / "features" / "map_features_v2_rich.yaml"
RF_SELECTED_PATH = ROOT / "data" / "modeling" / "map_random_forest_v1_selected_config.json"
XGB_SELECTED_PATH = ROOT / "data" / "modeling" / "map_xgboost_v1_selected_config.json"

RANDOM_STATE = 42
PERM_REPEATS = 10

# Family taxonomy for V3's 25 new features (brief section 31).
V3_FAMILIES = {
    "M": ["time_weighted_map_wr_diff", "time_weighted_map_margin_diff",
          "time_weighted_map_performance_residual_diff", "time_weighted_map_opponent_elo_diff",
          "map_recent_history_mass_diff", "map_recent_history_mass_min",
          "both_teams_have_recent_selected_map_history", "map_adjusted_history_mass_min"],
    "N": ["selected_map_elo_vs_overall_diff", "selected_map_elo_vs_pool_mean_diff",
          "selected_map_wr_vs_pool_mean_diff", "selected_map_rank_percentile_diff",
          "selected_map_in_both_recent_pools"],
    "O": ["roster_map_mean_kast_diff", "roster_map_bottom_kast_diff", "roster_map_mean_adr_diff",
          "roster_map_mean_kd_balance_diff", "roster_map_mean_history_mass_diff",
          "roster_map_players_with_history_diff", "roster_map_kast_specialization_diff",
          "roster_map_players_with_history_min", "roster_map_history_mass_min"],
    "P": ["current_core_map_continuity_diff", "current_core_map_history_mass_diff",
          "current_core_map_continuity_min"],
}
V3_FAMILY_LABELS = {
    "M": "recent/opponent-adjusted selected-map team features",
    "N": "map specialization (relative to overall/pool strength)",
    "O": "current-roster selected-map player performance",
    "P": "current-core selected-map continuity",
}
assert sum(len(v) for v in V3_FAMILIES.values()) == 25

COVERAGE_SUBGROUP_NAME = "high_evidence (both_recent_map_history & roster_map_players_with_history_min>=5)"


def grouped_permutation_importance(model, X, y, feature_names, groups, n_repeats=10, random_state=42):
    """Jointly permutes ALL columns of one family together (one shared row
    permutation per repeat), reused verbatim from
    evaluation/validation/evaluate_series_feature_sets_v3.py's implementation."""
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


class Arm:
    """One feature-set arm (V2 or V3): roles + preprocessing modules bundled
    so the fold loop can treat both arms identically."""

    def __init__(self, name, features_df, roles, prep_rf, prep_xgb, build_aug):
        self.name = name
        self.features_df = features_df
        self.roles = roles
        self.prep_rf = prep_rf
        self.prep_xgb = prep_xgb
        self.build_aug = build_aug


def evaluate_arm_on_fold(arm, cv, fold, rf_params, xgb_params, xgb_fixed, final_n_estimators):
    raw_tr, raw_va = fold_frames(cv, arm.features_df, fold)
    aug = arm.build_aug(raw_tr, arm.roles)
    y_aug = aug[arm.roles["target"]].to_numpy(dtype=float)
    y_val = raw_va[arm.roles["target"]].to_numpy(dtype=float)

    rf_prep = arm.prep_rf.fit_preprocessing(aug, arm.roles)
    X_rf_aug, rf_names = arm.prep_rf.transform(aug, rf_prep, arm.roles)
    X_rf_val, _ = arm.prep_rf.transform(raw_va, rf_prep, arm.roles)
    rf = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_rf_aug, y_aug)
    p_rf = rf.predict_proba(X_rf_val)[:, 1]

    xgb_prep = arm.prep_xgb.fit_preprocessing(aug, arm.roles)
    X_xgb_aug, xgb_names = arm.prep_xgb.transform(aug, xgb_prep, arm.roles)
    X_xgb_val, _ = arm.prep_xgb.transform(raw_va, xgb_prep, arm.roles)
    xgb = XGBClassifier(n_estimators=final_n_estimators, **xgb_params, **xgb_fixed)
    xgb.fit(X_xgb_aug, y_aug, verbose=False)
    p_xgb = xgb.predict_proba(X_xgb_val)[:, 1]

    return {
        "rf_model": rf, "xgb_model": xgb, "rf_names": rf_names, "xgb_names": xgb_names,
        "X_rf_val": X_rf_val, "X_xgb_val": X_xgb_val,
        "p_rf": p_rf, "p_xgb": p_xgb, "y_val": y_val,
        "match_id": raw_va["match_id"].to_numpy(), "game_id": raw_va["game_id"].to_numpy(),
        "raw_val": raw_va,
    }


def main():
    rf_sel = json.loads(RF_SELECTED_PATH.read_text(encoding="utf-8"))
    xgb_sel = json.loads(XGB_SELECTED_PATH.read_text(encoding="utf-8"))
    rf_params = dict(rf_sel["params"])
    if rf_params.get("max_depth") in ("None", None):
        rf_params["max_depth"] = None
    xgb_params = dict(xgb_sel["params"])
    xgb_fixed = xgb_sel["fixed_params"]
    final_n_estimators = int(xgb_sel["final_n_estimators"])
    print(f"Frozen RF:  {rf_sel['selected_candidate_id']} {rf_params}")
    print(f"Frozen XGB: {xgb_sel['selected_candidate_id']} {xgb_params} n_estimators={final_n_estimators}")
    print("Identical configuration used for BOTH the V2 and V3 arms.\n")

    v2_roles = load_map_v2_roles(V2_CONFIG_PATH)
    v3_roles = load_map_v3_roles(V3_CONFIG_PATH)
    v2_features = load_v2_features()
    v3_features = pd.read_parquet(V3_PATH, engine="fastparquet")
    assert len(v2_features) == len(v3_features)

    cv = load_cv_manifest(verify_against_split=False)   # STRUCTURAL: this script never opens map_split_v1.csv

    v2_arm = Arm("v2", v2_features, v2_roles, prep_rf_v2, prep_xgb_v2, build_aug_v2)
    v3_arm = Arm("v3", v3_features, v3_roles, prep_rf_v3, prep_xgb_v3, build_aug_v3)

    fold_rows, oof = {"v2": [], "v3": []}, {"v2": {"rf": [], "xgb": []}, "v3": {"rf": [], "xgb": []}}
    oof_meta = {"v2": [], "v3": []}
    last_result = {"v2": None, "v3": None}

    for fold in range(1, N_FOLDS + 1):
        for arm in (v2_arm, v3_arm):
            r = evaluate_arm_on_fold(arm, cv, fold, rf_params, xgb_params, xgb_fixed, final_n_estimators)
            last_result[arm.name] = r
            rf_m = compute_metrics(r["y_val"], r["p_rf"])
            xgb_m = compute_metrics(r["y_val"], r["p_xgb"])
            fold_rows[arm.name].append({"fold": fold, "model": "random_forest", **rf_m})
            fold_rows[arm.name].append({"fold": fold, "model": "xgboost", **xgb_m})
            oof[arm.name]["rf"].append(r["p_rf"])
            oof[arm.name]["xgb"].append(r["p_xgb"])
            oof_meta[arm.name].append(pd.DataFrame({"match_id": r["match_id"], "game_id": r["game_id"],
                                                      "fold": fold, "y_true": r["y_val"]}))
            print(f"fold {fold} [{arm.name}]: RF logloss={rf_m['log_loss']:.4f} acc={rf_m['accuracy']:.4f} | "
                  f"XGB logloss={xgb_m['log_loss']:.4f} acc={xgb_m['accuracy']:.4f}")

    fold_df = pd.concat([pd.DataFrame(fold_rows["v2"]).assign(arm="v2"),
                          pd.DataFrame(fold_rows["v3"]).assign(arm="v3")], ignore_index=True)

    # ---------------- paired fold-wise deltas ----------------
    pair_rows = []
    for fold in range(1, N_FOLDS + 1):
        for model in ["random_forest", "xgboost"]:
            v2r = fold_df[(fold_df.arm == "v2") & (fold_df.fold == fold) & (fold_df.model == model)].iloc[0]
            v3r = fold_df[(fold_df.arm == "v3") & (fold_df.fold == fold) & (fold_df.model == model)].iloc[0]
            pair_rows.append({
                "fold": fold, "model": model,
                "v2_log_loss": v2r["log_loss"], "v3_log_loss": v3r["log_loss"],
                "delta_log_loss": v3r["log_loss"] - v2r["log_loss"],     # negative = better
                "v2_brier": v2r["brier"], "v3_brier": v3r["brier"], "delta_brier": v3r["brier"] - v2r["brier"],
                "v2_roc_auc": v2r["roc_auc"], "v3_roc_auc": v3r["roc_auc"],
                "delta_roc_auc": v3r["roc_auc"] - v2r["roc_auc"],        # positive = better
                "v2_accuracy": v2r["accuracy"], "v3_accuracy": v3r["accuracy"],
                "delta_accuracy": v3r["accuracy"] - v2r["accuracy"],     # positive = better
                "v2_f1": v2r["f1"], "v3_f1": v3r["f1"], "delta_f1": v3r["f1"] - v2r["f1"],
            })
    pair_df = pd.DataFrame(pair_rows)

    mean_delta_rows = []
    for model in ["random_forest", "xgboost"]:
        sub = pair_df[pair_df.model == model]
        mean_delta_rows.append({"model": model, "row_type": "mean_fold_delta",
                                 "delta_log_loss": sub["delta_log_loss"].mean(),
                                 "delta_brier": sub["delta_brier"].mean(),
                                 "delta_roc_auc": sub["delta_roc_auc"].mean(),
                                 "delta_accuracy": sub["delta_accuracy"].mean(),
                                 "delta_f1": sub["delta_f1"].mean()})
    mean_delta_df = pd.DataFrame(mean_delta_rows)

    # ---------------- pooled TRAIN-only OOF metrics ----------------
    pooled_rows = []
    for arm_name in ["v2", "v3"]:
        meta = pd.concat(oof_meta[arm_name], ignore_index=True)
        y = meta["y_true"].to_numpy(dtype=float)
        for model in ["rf", "xgb"]:
            p = np.concatenate(oof[arm_name][model])
            m = compute_metrics(y, p, with_confusion=False)
            sm = series_macro_metrics(meta["match_id"].to_numpy(), y, p)
            pooled_rows.append({"arm": arm_name, "model": "random_forest" if model == "rf" else "xgboost",
                                 "row_type": "pooled_oof", **m, **sm,
                                 "n_correct": int(((p >= 0.5).astype(int) == y).sum())})
    pooled_df = pd.DataFrame(pooled_rows)

    additional_correct = {}
    for model_label, model_key in [("random_forest", "rf"), ("xgboost", "xgb")]:
        v2_meta = pd.concat(oof_meta["v2"], ignore_index=True)
        v3_meta = pd.concat(oof_meta["v3"], ignore_index=True)
        p_v2 = np.concatenate(oof["v2"][model_key])
        p_v3 = np.concatenate(oof["v3"][model_key])
        assert v2_meta["match_id"].tolist() == v3_meta["match_id"].tolist(), "OOF row order diverged V2 vs V3"
        assert v2_meta["game_id"].tolist() == v3_meta["game_id"].tolist()
        y = v2_meta["y_true"].to_numpy(dtype=float)
        v2_correct = (p_v2 >= 0.5).astype(int) == y
        v3_correct = (p_v3 >= 0.5).astype(int) == y
        additional_correct[model_label] = {
            "n": int(len(y)), "v2_correct": int(v2_correct.sum()), "v3_correct": int(v3_correct.sum()),
            "additional_correct": int(v3_correct.sum()) - int(v2_correct.sum()),
            "pp_gain": 100.0 * (int(v3_correct.sum()) - int(v2_correct.sum())) / len(y),
        }

    # ---------------- predefined high-evidence coverage subgroup (V3 only) ----------------
    cov_rows = []
    for model_label, model_key in [("random_forest", "rf"), ("xgboost", "xgb")]:
        v2_meta = pd.concat(oof_meta["v2"], ignore_index=True)
        v3_meta = pd.concat(oof_meta["v3"], ignore_index=True).merge(
            v3_features[["match_id", "game_id", "both_teams_have_recent_selected_map_history",
                          "roster_map_players_with_history_min"]],
            on=["match_id", "game_id"], how="left")
        mask = ((v3_meta["both_teams_have_recent_selected_map_history"] == 1)
                & (v3_meta["roster_map_players_with_history_min"] >= 5)).to_numpy()
        p_v2 = np.concatenate(oof["v2"][model_key])
        p_v3 = np.concatenate(oof["v3"][model_key])
        y = v2_meta["y_true"].to_numpy(dtype=float)
        m_v2 = compute_metrics(y[mask], p_v2[mask])
        m_v3 = compute_metrics(y[mask], p_v3[mask])
        cov_rows.append({"model": model_label, "subgroup": COVERAGE_SUBGROUP_NAME, "n": int(mask.sum()),
                          "v2_log_loss": m_v2["log_loss"], "v3_log_loss": m_v3["log_loss"],
                          "v2_roc_auc": m_v2["roc_auc"], "v3_roc_auc": m_v3["roc_auc"],
                          "v2_accuracy": m_v2["accuracy"], "v3_accuracy": m_v3["accuracy"]})
    cov_df = pd.DataFrame(cov_rows)

    # ---------------- V3-only importance (fold-validation), individual + grouped ----------------
    rf_imp_rows, xgb_imp_rows, group_rows = [], [], []
    for fold in range(1, N_FOLDS + 1):
        r = evaluate_arm_on_fold(v3_arm, cv, fold, rf_params, xgb_params, xgb_fixed, final_n_estimators)
        rf_perm = permutation_importance(r["rf_model"], r["X_rf_val"], r["y_val"], scoring="roc_auc",
                                          n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=1)
        for i, name in enumerate(r["rf_names"]):
            rf_imp_rows.append({"fold": fold, "feature": name,
                                 "impurity_importance": float(r["rf_model"].feature_importances_[i]),
                                 "permutation_importance": float(rf_perm.importances_mean[i])})
        booster = r["xgb_model"].get_booster()
        gain = booster.get_score(importance_type="gain")
        xgb_perm = permutation_importance(r["xgb_model"], r["X_xgb_val"], r["y_val"], scoring="roc_auc",
                                           n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=1)
        for i, name in enumerate(r["xgb_names"]):
            xgb_imp_rows.append({"fold": fold, "feature": name, "gain": float(gain.get(f"f{i}", 0.0)),
                                  "permutation_importance": float(xgb_perm.importances_mean[i])})
        for model_name, model, X, names in [("random_forest", r["rf_model"], r["X_rf_val"], r["rf_names"]),
                                             ("xgboost", r["xgb_model"], r["X_xgb_val"], r["xgb_names"])]:
            g = grouped_permutation_importance(model, X, r["y_val"], names, V3_FAMILIES, PERM_REPEATS,
                                                RANDOM_STATE)
            for fam, v in g.items():
                group_rows.append({"model": model_name, "fold": fold, "family": fam,
                                    "family_label": V3_FAMILY_LABELS[fam], "n_features": len(V3_FAMILIES[fam]),
                                    "auc_decrease_mean": v["mean"], "auc_decrease_std": v["std"]})

    rf_imp = pd.DataFrame(rf_imp_rows).groupby("feature", as_index=False).agg(
        impurity_importance_mean=("impurity_importance", "mean"),
        permutation_importance_mean=("permutation_importance", "mean"),
    ).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    xgb_imp = pd.DataFrame(xgb_imp_rows).groupby("feature", as_index=False).agg(
        gain_mean=("gain", "mean"),
        permutation_importance_mean=("permutation_importance", "mean"),
    ).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    group_df = pd.DataFrame(group_rows)
    group_agg = group_df.groupby(["model", "family", "family_label", "n_features"], as_index=False).agg(
        auc_decrease_mean=("auc_decrease_mean", "mean")).sort_values(
        ["model", "auc_decrease_mean"], ascending=[True, False]).reset_index(drop=True)

    # ---------------- write tables ----------------
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    pd.concat([fold_df, mean_delta_df.assign(fold=np.nan)], ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "map_feature_v2_v3_cv_comparison.csv", index=False, encoding="utf-8")
    pair_df.to_csv(TABLES_DIR / "map_feature_v2_v3_paired_deltas.csv", index=False, encoding="utf-8")
    pooled_df.to_csv(TABLES_DIR / "map_feature_v2_v3_pooled_oof.csv", index=False, encoding="utf-8")
    cov_df.to_csv(TABLES_DIR / "map_feature_v2_v3_coverage.csv", index=False, encoding="utf-8")
    rf_imp.to_csv(TABLES_DIR / "map_v3_feature_importance_rf.csv", index=False, encoding="utf-8")
    xgb_imp.to_csv(TABLES_DIR / "map_v3_feature_importance_xgb.csv", index=False, encoding="utf-8")
    pd.concat([group_df.assign(row_type="fold"), group_agg.assign(row_type="aggregate", fold=np.nan)],
               ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "map_v3_group_permutation_importance.csv", index=False, encoding="utf-8")

    # ---------------- figures ----------------
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    for model, stem in [("random_forest", "rf"), ("xgboost", "xgb")]:
        sub = fold_df[fold_df.model == model]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm_name, marker in [("v2", "o"), ("v3", "s")]:
            s = sub[sub.arm == arm_name].sort_values("fold")
            ax.plot(s["fold"], s["log_loss"], marker=marker, label=f"{arm_name} log loss")
        ax.set_xlabel("fold"), ax.set_ylabel("log loss")
        ax.set_title(f"Map V2 vs V3 - {model} TRAIN-only CV log loss by fold")
        ax.legend(), fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"map_feature_v2_v3_{stem}_cv.png", dpi=150)
        plt.close(fig)

    write_report(rf_sel, xgb_sel, pair_df, mean_delta_df, pooled_df, additional_correct, cov_df,
                 rf_imp, xgb_imp, group_agg)
    print("\nWrote reports/tables/map_feature_v2_v3_cv_comparison.csv")
    print("Wrote reports/tables/map_v3_feature_importance_{rf,xgb}.csv")
    print("Wrote reports/tables/map_v3_group_permutation_importance.csv")
    print("Wrote reports/figures/map_feature_v2_v3_{rf,xgb}_cv.png")
    print("Wrote reports/phase6c_modern_map_cv_results.md")


def write_report(rf_sel, xgb_sel, pair_df, mean_delta_df, pooled_df, additional_correct, cov_df,
                  rf_imp, xgb_imp, group_agg):
    md = []
    md.append("# Phase 6C, Stage B - Known-Map V2-rich vs V3-modern-map TRAIN-only Ablation\n")
    md.append("**MAP VALIDATION = NOT USED. TEST = SEALED. COLOGNE = UNTOUCHED.** This script never opens "
              "`data/modeling/map_split_v1.csv` - only the TRAIN-only `data/modeling/map_cv_folds_v1.csv` "
              "manifest.\n")
    md.append(f"Identical, frozen Phase 6B configurations used for BOTH arms: RF `{rf_sel['selected_candidate_id']}` "
              f"({', '.join(f'{k}={v}' for k, v in rf_sel['params'].items())}), XGB "
              f"`{xgb_sel['selected_candidate_id']}` ({', '.join(f'{k}={v}' for k, v in xgb_sel['params'].items())}, "
              f"`n_estimators={xgb_sel['final_n_estimators']}`, no early stopping). Neither model was retuned "
              "for V3.\n")

    md.append("## Paired fold-wise deltas (V3 - V2)\n")
    md.append("Negative `delta_log_loss`/`delta_brier` and positive `delta_roc_auc`/`delta_accuracy`/`delta_f1` "
              "favor V3.\n")
    md.append("| fold | model | V2 log loss | V3 log loss | Δ log loss | V2 ROC-AUC | V3 ROC-AUC | Δ ROC-AUC | "
              "Δ accuracy | Δ Brier |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in pair_df.iterrows():
        md.append(f"| {int(r['fold'])} | {r['model']} | {r['v2_log_loss']:.4f} | {r['v3_log_loss']:.4f} | "
                  f"{r['delta_log_loss']:+.4f} | {r['v2_roc_auc']:.4f} | {r['v3_roc_auc']:.4f} | "
                  f"{r['delta_roc_auc']:+.4f} | {r['delta_accuracy']:+.4f} | {r['delta_brier']:+.4f} |")
    md.append("")
    md.append("Mean fold deltas:\n")
    md.append("| model | Δ log loss | Δ ROC-AUC | Δ Brier | Δ accuracy | Δ F1 |")
    md.append("|---|---|---|---|---|---|")
    for _, r in mean_delta_df.iterrows():
        md.append(f"| {r['model']} | {r['delta_log_loss']:+.4f} | {r['delta_roc_auc']:+.4f} | "
                  f"{r['delta_brier']:+.4f} | {r['delta_accuracy']:+.4f} | {r['delta_f1']:+.4f} |")
    md.append("")

    md.append("## Pooled TRAIN-only OOF metrics\n")
    md.append("| arm | model | n | accuracy | ROC-AUC | log loss | Brier | series-macro log loss | "
              "series-macro accuracy |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in pooled_df.iterrows():
        md.append(f"| {r['arm']} | {r['model']} | {int(r['n'])} | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} | {r['series_macro_log_loss']:.4f} | "
                  f"{r['series_macro_accuracy']:.4f} |")
    md.append("")

    md.append("## Additional correct maps (pooled TRAIN-only OOF)\n")
    md.append("| model | n | V2 correct | V3 correct | additional correct | pp accuracy gain |")
    md.append("|---|---|---|---|---|---|")
    for model, d in additional_correct.items():
        md.append(f"| {model} | {d['n']} | {d['v2_correct']} | {d['v3_correct']} | "
                  f"{d['additional_correct']:+d} | {d['pp_gain']:+.3f} |")
    md.append("")

    md.append(f"## Coverage diagnostic: {COVERAGE_SUBGROUP_NAME}\n")
    md.append("Predefined using ONLY pre-match V3 evidence. Descriptive only - no subgroup-specific model is "
              "trained.\n")
    md.append("| model | n | V2 log loss | V3 log loss | V2 ROC-AUC | V3 ROC-AUC | V2 accuracy | V3 accuracy |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in cov_df.iterrows():
        md.append(f"| {r['model']} | {int(r['n'])} | {r['v2_log_loss']:.4f} | {r['v3_log_loss']:.4f} | "
                  f"{r['v2_roc_auc']:.4f} | {r['v3_roc_auc']:.4f} | {r['v2_accuracy']:.4f} | "
                  f"{r['v3_accuracy']:.4f} |")
    md.append("")

    md.append("## V3-only grouped permutation importance (TRAIN-only CV, fold-validation)\n")
    md.append("Each family's columns are permuted jointly. No feature is selected or removed based on this.\n")
    md.append("| model | family | label | n features | AUC decrease |")
    md.append("|---|---|---|---|---|")
    for _, r in group_agg.iterrows():
        md.append(f"| {r['model']} | {r['family']} | {r['family_label']} | {int(r['n_features'])} | "
                  f"{r['auc_decrease_mean']:+.4f} |")
    md.append("")

    md.append("## Strongest new individual features (TRAIN-only CV permutation importance)\n")
    md.append("**Random Forest** (top 10):\n")
    md.append("| rank | feature | permutation | impurity |")
    md.append("|---|---|---|---|")
    for i, (_, r) in enumerate(rf_imp.head(10).iterrows(), start=1):
        md.append(f"| {i} | {r['feature']} | {r['permutation_importance_mean']:+.4f} | "
                  f"{r['impurity_importance_mean']:.4f} |")
    md.append("\n**XGBoost** (top 10):\n")
    md.append("| rank | feature | permutation | gain |")
    md.append("|---|---|---|---|")
    for i, (_, r) in enumerate(xgb_imp.head(10).iterrows(), start=1):
        md.append(f"| {i} | {r['feature']} | {r['permutation_importance_mean']:+.4f} | {r['gain_mean']:.4f} |")
    md.append("")

    md.append("## Interpretation (asymmetric, per the framing established in every prior ablation)\n")
    rf_delta = mean_delta_df[mean_delta_df.model == "random_forest"].iloc[0]
    xgb_delta = mean_delta_df[mean_delta_df.model == "xgboost"].iloc[0]
    md.append(f"RF mean fold Δlog loss {rf_delta['delta_log_loss']:+.4f}, ΔROC-AUC {rf_delta['delta_roc_auc']:+.4f}. "
              f"XGB mean fold Δlog loss {xgb_delta['delta_log_loss']:+.4f}, ΔROC-AUC {xgb_delta['delta_roc_auc']:+.4f}. "
              "If V3 improves under these frozen, previously-selected configurations, that is evidence the new "
              "features add signal without retuning. If V3 does not improve, the correct conclusion is only that "
              "it did not improve under the frozen configurations used here - not that the new selected-map "
              "information carries no signal at all. No feature is added, removed or reweighted based on these "
              "results; this is the final feature-engineering experiment for the known-map task.\n")

    md.append("## Status\n")
    md.append("- **MAP VALIDATION = NOT USED**\n- **TEST = SEALED**\n- **COLOGNE = UNTOUCHED**\n"
              "- **NO POST-RESULT FEATURE CHANGES**\n- **SRC = UNCHANGED**\n")

    (REPORTS / "phases" / "phase6c_modern_map_cv_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
