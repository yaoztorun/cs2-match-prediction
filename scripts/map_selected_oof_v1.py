"""
Phase 6B, brief sections 23-27 and 35: everything that happens AFTER the RF and
XGBoost structural configurations are frozen but BEFORE the main map validation
partition is ever opened.

    1. Re-run each SELECTED configuration across the same four TRAIN-only outer
       folds in its FINAL deployment-style form (RF: selected config directly;
       XGB: the frozen final_n_estimators for every fold, no early stopping),
       producing out-of-fold predictions that match how the final models will
       actually be fit.
    2. Pooled TRAIN-only OOF metrics, map-level and series-macro.
    3. TRAIN-only CV feature importance for both selected models: RF impurity +
       fold-validation permutation; XGB gain + weight + fold-validation
       permutation; plus GROUPED (jointly-permuted) family importance, which is
       how the scientific question - does exact selected-map history add
       anything beyond overall ELO/form? - actually gets answered.
    4. The predefined 11-weight RF/XGB probability blend, selected from
       TRAIN-only OOF alone and frozen.
    5. The provisional development winner, again from TRAIN-only OOF alone.

The main VALIDATION partition is never opened here.

Writes:
    data/modeling/map_selected_models_oof_v1.parquet
    data/modeling/map_ensemble_v1_config.json
    reports/tables/map_model_oof_metrics_v1.csv
    reports/tables/map_rf_feature_importance_v1.csv
    reports/tables/map_xgb_feature_importance_v1.csv
    reports/tables/map_group_permutation_importance_v1.csv
    reports/phase6b_known_map_tuning_summary.md
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from _common import REPORTS
from map_feature_families import feature_family_map, FAMILY_LABELS
from map_modeling_common import (
    FoldCache, MODELING_DIR, N_FOLDS, RANDOM_STATE, assert_target_and_no_forbidden_columns,
    compute_metrics, load_cv_manifest, load_features, load_roles, series_macro_metrics,
)

TABLES_DIR = REPORTS / "tables"
RF_SELECTED_PATH = MODELING_DIR / "map_random_forest_v1_selected_config.json"
XGB_SELECTED_PATH = MODELING_DIR / "map_xgboost_v1_selected_config.json"
OOF_PATH = MODELING_DIR / "map_selected_models_oof_v1.parquet"
ENSEMBLE_PATH = MODELING_DIR / "map_ensemble_v1_config.json"

ENSEMBLE_WEIGHTS = [round(0.1 * i, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0 - fixed BEFORE any result
ENSEMBLE_LOG_LOSS_EPSILON = 0.001                            # fixed BEFORE any result
PERM_REPEATS = 5
GROUP_PERM_REPEATS = 10


def grouped_permutation_importance(model, X, y, feature_names, groups, n_repeats, random_state):
    """Jointly permutes ALL columns of one family together (one shared row
    permutation per repeat, not an independent permutation per column), so
    correlated features inside a family are broken together. Returns the
    ROC-AUC decrease per group. Same implementation the series phases used."""
    rng = np.random.RandomState(random_state)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for group_name, cols in groups.items():
        idxs = [name_to_idx[c] for c in cols if c in name_to_idx]
        if not idxs:
            continue
        decreases = []
        for _ in range(n_repeats):
            perm = rng.permutation(X.shape[0])
            X_perm = X.copy()
            X_perm[:, idxs] = X_perm[perm][:, idxs]
            decreases.append(baseline - roc_auc_score(y, model.predict_proba(X_perm)[:, 1]))
        out[group_name] = {"mean": float(np.mean(decreases)), "std": float(np.std(decreases))}
    return out


def select_ensemble_weight(y, p_rf, p_xgb):
    """PRIMARY lowest pooled TRAIN-only OOF log loss; ties within 0.001 broken by
    higher ROC-AUC -> lower Brier -> higher accuracy -> weight closest to 0.5 ->
    lower weight. Every rule fixed before any ensemble number was computed."""
    rows = []
    for w in ENSEMBLE_WEIGHTS:
        p = w * p_rf + (1 - w) * p_xgb
        rows.append({"weight_rf": w, **compute_metrics(y, p)})
    df = pd.DataFrame(rows)

    best = df["log_loss"].min()
    tied = df[df["log_loss"] <= best + ENSEMBLE_LOG_LOSS_EPSILON].copy()
    stage = "primary (lowest pooled OOF log loss, unique)"
    if len(tied) > 1:
        stage = "secondary (log-loss tie within epsilon, resolved by highest ROC-AUC)"
        tied = tied[tied["roc_auc"] == tied["roc_auc"].max()]
    if len(tied) > 1:
        stage = "tertiary (resolved by lowest Brier)"
        tied = tied[tied["brier"] == tied["brier"].min()]
    if len(tied) > 1:
        stage = "quaternary (resolved by highest accuracy)"
        tied = tied[tied["accuracy"] == tied["accuracy"].max()]
    if len(tied) > 1:
        stage = "quinary (resolved by weight closest to 0.5, then lower weight)"
        tied = tied.assign(_d=(tied["weight_rf"] - 0.5).abs()).sort_values(["_d", "weight_rf"])
    return float(tied.iloc[0]["weight_rf"]), stage, df


def provisional_winner(metrics_by_model):
    """Probability quality first: log loss -> ROC-AUC -> Brier -> accuracy.
    TRAIN-only OOF evidence. Explicitly NOT a final project model."""
    order = sorted(metrics_by_model.items(),
                    key=lambda kv: (kv[1]["log_loss"], -kv[1]["roc_auc"], kv[1]["brier"], -kv[1]["accuracy"]))
    return order[0][0], order


def main():
    roles = load_roles()
    features = load_features()
    assert_target_and_no_forbidden_columns(features, roles)
    cv = load_cv_manifest(verify_against_split=False)

    rf_sel = json.loads(RF_SELECTED_PATH.read_text(encoding="utf-8"))
    xgb_sel = json.loads(XGB_SELECTED_PATH.read_text(encoding="utf-8"))
    rf_params = dict(rf_sel["params"])
    if rf_params.get("max_depth") in ("None", None):
        rf_params["max_depth"] = None
    xgb_params = dict(xgb_sel["params"])
    final_n_estimators = int(xgb_sel["final_n_estimators"])
    print(f"RF  selected: {rf_sel['selected_candidate_id']} -> {rf_params}")
    print(f"XGB selected: {xgb_sel['selected_candidate_id']} -> {xgb_params} "
          f"| final_n_estimators={final_n_estimators} (frozen, no early stopping)")

    cache = FoldCache(cv, features, roles, build_rf=True, build_xgb=True)
    feature_names = cache.feature_names
    families = feature_family_map(roles, feature_names)

    oof_rows, rf_imp_rows, xgb_imp_rows, group_rows = [], [], [], []

    for fold in range(1, N_FOLDS + 1):
        e = cache[fold]
        print(f"\nfold {fold}: refitting both selected configurations in deployment form")

        rf = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(e["rf_X_aug"], e["y_aug"])
        p_rf = rf.predict_proba(e["rf_X_val"])[:, 1]

        xgb = XGBClassifier(n_estimators=final_n_estimators, **xgb_params, **xgb_sel["fixed_params"])
        xgb.fit(e["xgb_X_aug"], e["y_aug"], verbose=False)   # NO eval_set, NO early stopping
        p_xgb = xgb.predict_proba(e["xgb_X_val"])[:, 1]

        val = e["raw_val"]
        oof_rows.append(pd.DataFrame({
            "match_id": val["match_id"].to_numpy(), "game_id": val["game_id"].to_numpy(),
            "fold": fold, "map_name": val["map_name"].to_numpy(),
            "series_datetime": val["series_datetime"].to_numpy(),
            "y_true": e["y_val"], "p_rf": p_rf, "p_xgb": p_xgb,
        }))
        print(f"  RF  fold log loss {compute_metrics(e['y_val'], p_rf)['log_loss']:.4f} | "
              f"XGB fold log loss {compute_metrics(e['y_val'], p_xgb)['log_loss']:.4f}")

        # ---- individual feature importance, TRAIN-only CV ----
        rf_perm = permutation_importance(rf, e["rf_X_val"], e["y_val"], scoring="roc_auc",
                                          n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=1)
        for i, name in enumerate(feature_names):
            rf_imp_rows.append({"fold": fold, "feature": name, "family": families["by_feature"][name],
                                 "impurity_importance": float(rf.feature_importances_[i]),
                                 "permutation_importance": float(rf_perm.importances_mean[i]),
                                 "permutation_std": float(rf_perm.importances_std[i])})

        booster = xgb.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        xgb_perm = permutation_importance(xgb, e["xgb_X_val"], e["y_val"], scoring="roc_auc",
                                           n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=1)
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            xgb_imp_rows.append({"fold": fold, "feature": name, "family": families["by_feature"][name],
                                  "gain": float(gain.get(key, 0.0)), "weight": float(weight.get(key, 0.0)),
                                  "permutation_importance": float(xgb_perm.importances_mean[i]),
                                  "permutation_std": float(xgb_perm.importances_std[i])})

        # ---- grouped (jointly permuted) family importance ----
        for model_name, model, X in [("random_forest", rf, e["rf_X_val"]), ("xgboost", xgb, e["xgb_X_val"])]:
            g = grouped_permutation_importance(model, X, e["y_val"], feature_names, families["groups"],
                                                GROUP_PERM_REPEATS, RANDOM_STATE)
            for fam, v in g.items():
                group_rows.append({"model": model_name, "fold": fold, "family": fam,
                                    "family_label": FAMILY_LABELS[fam], "n_features": len(families["groups"][fam]),
                                    "auc_decrease_mean": v["mean"], "auc_decrease_std": v["std"]})

    # ---------------- OOF artifact ----------------
    oof = pd.concat(oof_rows, ignore_index=True)
    assert oof.duplicated(subset=["match_id", "game_id"]).sum() == 0, "an OOF map row appeared in two folds"
    assert oof["p_rf"].between(0, 1).all() and oof["p_xgb"].between(0, 1).all()
    oof.to_parquet(OOF_PATH, engine="fastparquet", index=False)
    print(f"\nWrote {OOF_PATH} ({len(oof)} pooled TRAIN-only out-of-fold map rows)")

    y = oof["y_true"].to_numpy(dtype=float)
    p_rf_all, p_xgb_all = oof["p_rf"].to_numpy(), oof["p_xgb"].to_numpy()

    # ---------------- ensemble weight, frozen from TRAIN-only OOF ----------------
    weight_rf, ens_stage, weight_table = select_ensemble_weight(y, p_rf_all, p_xgb_all)
    p_ens_all = weight_rf * p_rf_all + (1 - weight_rf) * p_xgb_all
    print(f"\nEnsemble weight selected: w_rf={weight_rf} via {ens_stage}")

    ENSEMBLE_PATH.write_text(json.dumps({
        "formula": "p_ensemble = w * p_rf + (1 - w) * p_xgb",
        "weight_rf": weight_rf, "weight_xgb": round(1 - weight_rf, 10),
        "candidate_weights": ENSEMBLE_WEIGHTS,
        "selection_stage": ens_stage,
        "selection_rule": ("PRIMARY lowest pooled TRAIN-only OOF log loss; within "
                            f"{ENSEMBLE_LOG_LOSS_EPSILON} log loss: higher ROC-AUC -> lower Brier -> higher "
                            "accuracy -> weight closest to 0.5 -> lower weight"),
        "log_loss_epsilon": ENSEMBLE_LOG_LOSS_EPSILON,
        "selected_from": "data/modeling/map_selected_models_oof_v1.parquet (TRAIN-only)",
        "main_validation_used_in_selection": False,
        "oof_metrics": {"rf": compute_metrics(y, p_rf_all), "xgb": compute_metrics(y, p_xgb_all),
                         "ensemble": compute_metrics(y, p_ens_all)},
    }, indent=2), encoding="utf-8")

    # ---------------- pooled OOF metrics table ----------------
    metric_rows = []
    for label, p in [("random_forest", p_rf_all), ("xgboost", p_xgb_all), ("ensemble", p_ens_all)]:
        m = compute_metrics(y, p, with_confusion=True)
        sm = series_macro_metrics(oof["match_id"].to_numpy(), y, p)
        metric_rows.append({"model": label, "population": "pooled_train_oof",
                             **{k: v for k, v in m.items() if k != "confusion_matrix"},
                             "confusion_matrix": json.dumps(m["confusion_matrix"]), **sm})
        for fold in range(1, N_FOLDS + 1):
            mask = (oof["fold"] == fold).to_numpy()
            metric_rows.append({"model": label, "population": f"fold_{fold}",
                                 **{k: v for k, v in compute_metrics(y[mask], p[mask]).items()}})
    oof_metrics = pd.DataFrame(metric_rows)
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    oof_metrics.to_csv(TABLES_DIR / "map_model_oof_metrics_v1.csv", index=False, encoding="utf-8")

    pooled = {r["model"]: r for _, r in oof_metrics[oof_metrics["population"] == "pooled_train_oof"].iterrows()}
    winner, ranking = provisional_winner({k: v for k, v in pooled.items()})
    print(f"Provisional development winner (TRAIN-only OOF): {winner}")

    # ---------------- importance tables ----------------
    rf_imp = pd.DataFrame(rf_imp_rows).groupby(["feature", "family"], as_index=False).agg(
        impurity_importance_mean=("impurity_importance", "mean"),
        permutation_importance_mean=("permutation_importance", "mean"),
        permutation_importance_std=("permutation_importance", "std"),
    ).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    rf_imp["rank_permutation"] = rf_imp.index + 1
    rf_imp.to_csv(TABLES_DIR / "map_rf_feature_importance_v1.csv", index=False, encoding="utf-8")

    xgb_imp = pd.DataFrame(xgb_imp_rows).groupby(["feature", "family"], as_index=False).agg(
        gain_mean=("gain", "mean"), weight_mean=("weight", "mean"),
        permutation_importance_mean=("permutation_importance", "mean"),
        permutation_importance_std=("permutation_importance", "std"),
    ).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    xgb_imp["rank_permutation"] = xgb_imp.index + 1
    xgb_imp.to_csv(TABLES_DIR / "map_xgb_feature_importance_v1.csv", index=False, encoding="utf-8")

    group_df = pd.DataFrame(group_rows)
    group_agg = group_df.groupby(["model", "family", "family_label", "n_features"], as_index=False).agg(
        auc_decrease_mean=("auc_decrease_mean", "mean"),
        auc_decrease_fold_std=("auc_decrease_mean", "std"),
    ).sort_values(["model", "auc_decrease_mean"], ascending=[True, False]).reset_index(drop=True)
    pd.concat([group_df.assign(row_type="fold"), group_agg.assign(row_type="aggregate", fold=np.nan)],
               ignore_index=True, sort=False).to_csv(
        TABLES_DIR / "map_group_permutation_importance_v1.csv", index=False, encoding="utf-8")

    write_summary(rf_sel, xgb_sel, weight_rf, ens_stage, weight_table, oof_metrics, winner, ranking,
                   rf_imp, xgb_imp, group_agg, families)

    print("Wrote reports/tables/map_model_oof_metrics_v1.csv")
    print("Wrote reports/tables/map_{rf,xgb}_feature_importance_v1.csv")
    print("Wrote reports/tables/map_group_permutation_importance_v1.csv")
    print("Wrote data/modeling/map_ensemble_v1_config.json")
    print("Wrote reports/phase6b_known_map_tuning_summary.md")


def write_summary(rf_sel, xgb_sel, weight_rf, ens_stage, weight_table, oof_metrics, winner, ranking,
                   rf_imp, xgb_imp, group_agg, families):
    baselines = pd.read_csv(TABLES_DIR / "map_baselines_cv_v1.csv")
    pooled_base = baselines[baselines["row_type"] == "pooled_oof"]
    pooled = oof_metrics[oof_metrics["population"] == "pooled_train_oof"]

    md = []
    md.append("# Phase 6B - Known-Map Tuning Summary (TRAIN-only)\n")
    md.append("Everything in this document was produced **before the main map validation partition was ever "
              "opened**. Nothing here is a generalization estimate: the four chronological folds are the same "
              "folds the configurations were selected against, so these are development numbers.\n")
    md.append("**Task: predict the winner of one specific, user-selected map (`team1_map_win`).** These figures "
              "are not comparable with the pre-veto series models' accuracies - different target, different "
              "task.\n")

    md.append("## TRAIN-CV reference baselines (never tuned)\n")
    md.append("| baseline | n | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|---|")
    for _, r in pooled_base.iterrows():
        md.append(f"| {r['baseline']} | {int(r['n'])} | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append("")

    md.append("## Selected configurations (frozen before validation)\n")
    md.append(f"**Random Forest** - `{rf_sel['selected_candidate_id']}`, via {rf_sel['selection_stage']}:\n")
    md.append("```\n" + ", ".join(f"{k}={v}" for k, v in rf_sel["params"].items()) + "\n```")
    md.append(f"CV mean log loss {rf_sel['cv_mean_log_loss']:.4f} ± {rf_sel['cv_std_log_loss']:.4f}, "
              f"CV mean ROC-AUC {rf_sel['cv_mean_roc_auc']:.4f}.\n")
    md.append(f"**XGBoost** - `{xgb_sel['selected_candidate_id']}`, via {xgb_sel['selection_stage']}:\n")
    md.append("```\n" + ", ".join(f"{k}={v}" for k, v in xgb_sel["params"].items()) + "\n```")
    md.append(f"`best_iteration` by fold {xgb_sel['best_iterations_by_fold']} -> "
              f"**final_n_estimators = {xgb_sel['final_n_estimators']}** "
              f"({xgb_sel['final_n_estimators_rule']}), frozen. CV mean log loss "
              f"{xgb_sel['cv_mean_log_loss']:.4f} ± {xgb_sel['cv_std_log_loss']:.4f}, CV mean ROC-AUC "
              f"{xgb_sel['cv_mean_roc_auc']:.4f}.\n")

    md.append("## Selected-config out-of-fold metrics (pooled, TRAIN-only)\n")
    md.append("Each selected configuration re-run across the same four folds in its **final deployment form** "
              "(XGBoost with the frozen tree count and no early stopping), so these predictions match how the "
              "final models are actually fitted.\n")
    md.append("| model | n | accuracy | precision | recall | F1 | ROC-AUC | log loss | Brier | series-macro "
              "log loss | series-macro Brier | series-macro accuracy |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in pooled.iterrows():
        md.append(f"| {r['model']} | {int(r['n'])} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                  f"{r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['log_loss']:.4f} | "
                  f"{r['brier']:.4f} | {r['series_macro_log_loss']:.4f} | {r['series_macro_brier']:.4f} | "
                  f"{r['series_macro_accuracy']:.4f} |")
    md.append("")
    md.append("Series-macro figures average each series' own per-map mean, then average those equally across "
              "match_ids, so a BO5 does not outweigh a BO1. Map-level metrics remain PRIMARY - the task is map "
              "prediction. No per-series ROC-AUC is computed: most series have too few maps for it to mean "
              "anything.\n")

    md.append("## Frozen RF/XGB probability ensemble\n")
    md.append(f"`p_ensemble = w * p_rf + (1 - w) * p_xgb` over the 11 predefined weights "
              f"{[f'{w:.1f}' for w in ENSEMBLE_WEIGHTS]}, selected on pooled TRAIN-only OOF log loss alone.\n")
    md.append("| w_rf | accuracy | ROC-AUC | log loss | Brier |")
    md.append("|---|---|---|---|---|")
    for _, r in weight_table.iterrows():
        mark = " **<-- selected**" if abs(r["weight_rf"] - weight_rf) < 1e-9 else ""
        md.append(f"| {r['weight_rf']:.1f}{mark} | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
                  f"{r['log_loss']:.4f} | {r['brier']:.4f} |")
    md.append(f"\n**Frozen: w_rf = {weight_rf}** via {ens_stage}. Saved to "
              "`data/modeling/map_ensemble_v1_config.json` before any validation data was opened.\n")

    md.append("## Provisional development winner\n")
    md.append("Probability quality first: log loss -> ROC-AUC -> Brier -> accuracy, on TRAIN-only OOF.\n")
    md.append("| rank | model | log loss | ROC-AUC | Brier | accuracy |")
    md.append("|---|---|---|---|---|---|")
    for i, (name, m) in enumerate(ranking, start=1):
        md.append(f"| {i} | {name} | {m['log_loss']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} | "
                  f"{m['accuracy']:.4f} |")
    md.append(f"\nProvisional winner: **{winner}**. This is NOT a final project model - the internal TEST "
              "partition remains the final unbiased internal evaluation and has not been opened.\n")

    md.append("## Feature-family grouped permutation importance (TRAIN-only CV)\n")
    md.append("Each family's columns are permuted **jointly** (one shared row permutation per repeat), so "
              "correlated features inside a family are broken together rather than one at a time. Values are the "
              "mean fold-validation ROC-AUC decrease, averaged over the four folds.\n")
    md.append("| family | label | n features | RF AUC decrease | XGB AUC decrease |")
    md.append("|---|---|---|---|---|")
    rf_g = group_agg[group_agg["model"] == "random_forest"].set_index("family")
    xg_g = group_agg[group_agg["model"] == "xgboost"].set_index("family")
    for fam in sorted(FAMILY_LABELS):
        if fam not in rf_g.index:
            continue
        md.append(f"| {fam} | {FAMILY_LABELS[fam]} | {int(rf_g.loc[fam, 'n_features'])} | "
                  f"{rf_g.loc[fam, 'auc_decrease_mean']:+.4f} | {xg_g.loc[fam, 'auc_decrease_mean']:+.4f} |")
    md.append("")
    k_rf = float(rf_g.loc["K", "auc_decrease_mean"])
    k_xgb = float(xg_g.loc["K", "auc_decrease_mean"])
    a_rf = float(rf_g.loc["A", "auc_decrease_mean"])
    a_xgb = float(xg_g.loc["A", "auc_decrease_mean"])
    md.append("### Does exact selected-map history add importance beyond overall ELO/form?\n")
    md.append(f"Family **K** (map-specific historical strength, {int(rf_g.loc['K', 'n_features'])} features) "
              f"scores {k_rf:+.4f} (RF) and {k_xgb:+.4f} (XGB) AUC decrease, against family **A** (the original "
              f"series-level ELO/form block) at {a_rf:+.4f} (RF) and {a_xgb:+.4f} (XGB). Read descriptively: a "
              "grouped permutation measures how much a model *relies on* a family given everything else it can "
              "see, not how much information that family contains in isolation - correlated families mask one "
              "another. No feature is removed in this phase.\n")

    md.append("## Top individual features by TRAIN-only CV permutation importance\n")
    for label, table, col in [("Random Forest", rf_imp, "impurity_importance_mean"),
                               ("XGBoost", xgb_imp, "gain_mean")]:
        md.append(f"**{label}** (top 12 by fold-validation permutation importance):\n")
        md.append(f"| rank | feature | family | permutation | {col} |")
        md.append("|---|---|---|---|---|")
        for _, r in table.head(12).iterrows():
            md.append(f"| {int(r['rank_permutation'])} | {r['feature']} | {r['family']} | "
                      f"{r['permutation_importance_mean']:+.4f} | {r[col]:.4f} |")
        md.append("")
    md.append("Full tables in `reports/tables/map_rf_feature_importance_v1.csv` and "
              "`map_xgb_feature_importance_v1.csv`. Diagnostic only - no feature selection is performed in "
              "Phase 6B.\n")

    md.append("## Status at this point\n")
    md.append("- RF configuration: **FROZEN**\n- XGBoost structural configuration and `final_n_estimators`: "
              "**FROZEN**\n- Ensemble weight: **FROZEN**\n- Main map VALIDATION: **not yet opened**\n"
              "- TEST: **SEALED**\n- Cologne: **UNTOUCHED**\n")

    (REPORTS / "phase6b_known_map_tuning_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
