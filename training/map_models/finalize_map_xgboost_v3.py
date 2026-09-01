"""
Phase 6D: finalize the selected V3 XGBoost configuration.

Three responsibilities, run only after training/map_models/map_xgboost_v3_final_tuning.py
has frozen a structural configuration and final_n_estimators:

  1. SELECTED-CONFIG OOF (brief sections 20-21, 29-30): rerun the frozen
     structural config + frozen final_n_estimators (IDENTICAL for every fold,
     no early stopping) across the four TRAIN-only outer folds, producing
     pooled out-of-fold predictions. This is TRAIN-only development
     performance under the SAME four folds the model was selected against -
     explicitly reported as "TRAIN-only selected-model OOF development
     performance", never as a generalization estimate (the sealed 1,427-map
     TEST is the next unbiased internal evaluation). Also: side-symmetry
     diagnostic, individual + grouped (A/B/E/H/K/M/N/O/P) permutation
     importance - interpretation only, no feature changes follow.
  2. THREE-WAY COMPARISON (brief section 22): A = V2 + frozen Phase 6B config
     (124 trees, read from Phase 6C's own frozen
     reports/tables/map_feature_v2_v3_pooled_oof.csv - not recomputed); B = V3
     + the SAME frozen Phase 6B config (124 trees, same source); C = V3 +
     Phase 6D's final tuned config (this script's own fresh OOF). "Feature
     gain" (A->B) and "development tuning gain" (B->C) are kept explicitly
     separate - never attributed to each other.
  3. FULL-TRAIN REFIT: the full-TRAIN map universe is explicitly
     RECONSTRUCTED from the union of every (match_id, game_id) appearing in
     any train/validation role across all four folds of
     data/modeling/map_cv_folds_v1.csv - data/modeling/map_split_v1.csv is
     never reopened. Asserts this union is exactly 7,762 rows before
     proceeding (STOPS otherwise). Fits V3 preprocessing fresh on the full
     augmented (15,524-observation) TRAIN, fits the frozen structural
     config + frozen tree count with no eval_set/early stopping, saves the
     model + metadata (deliberately NO validation/test metrics inside the
     metadata) + preprocessing + config, and verifies future-inference parity
     against feature_engineering/maps/rich_modern_map_feature_composer.py using a SYNTHETIC
     matchup (never real Cologne data).

Writes:
    data/modeling/map_xgboost_v3_final_oof.parquet
    data/modeling/map_xgboost_v3_final_preprocessing.json
    data/modeling/map_xgboost_v3_final_config.json
    models/map_xgboost_v3_final.json
    models/map_xgboost_v3_final_metadata.json
    reports/phase6d_final_xgboost_v3_results.md
    reports/tables/map_xgboost_v3_final_oof_metrics.csv
    reports/tables/map_xgboost_v3_final_feature_importance.csv
    reports/tables/map_xgboost_v3_final_group_importance.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


from _common import ROOT, REPORTS
from training.map_models.map_modeling_common import (
    MODELING_DIR, MODELS_DIR, N_FOLDS, RANDOM_STATE, compute_metrics, fold_frames,
    load_cv_manifest, package_versions, series_macro_metrics, sha256_file,
)
from feature_engineering.preprocessing.preprocessing_common_map_v3 import load_map_v3_roles
from feature_engineering.preprocessing.preprocessing_common_map_v2 import build_augmented_training_raw, assert_augmented_symmetry, mirror_raw_rows
import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb
from feature_engineering.preprocessing.preprocessing_random_forest_map_v3 import EXPECTED_TRANSFORMED_FEATURES
from feature_engineering.maps.map_feature_engine import MapStateStore
from feature_engineering.series.feature_engine import StateStore
from feature_engineering.form.team_form_engine import TeamFormStateStore
from feature_engineering.roster.player_roster_feature_engine import PlayerRosterStateStore
from feature_engineering.maps.modern_map_feature_engine import ModernMapStateStore
from feature_engineering.maps.rich_modern_map_feature_composer import (
    build_future_modern_rich_map_features, MODERN_RICH_MAP_DIRECTIONAL_FEATURES,
    MODERN_RICH_MAP_SYMMETRIC_FEATURES,
)
from feature_engineering.maps.map_stream_common import cologne_cutoff
from evaluation.validation.evaluate_map_feature_sets_v3 import V3_FAMILIES, V3_FAMILY_LABELS, grouped_permutation_importance

from training.map_models.map_xgboost_v3_final_tuning import XgbFoldCache   # noqa: E402

CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "map_features_v3_modern_map.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "map_cv_folds_v1.csv"
SELECTED_PATH = MODELING_DIR / "map_xgboost_v3_final_selected_config.json"
V2_V3_POOLED_OOF_PATH = REPORTS / "tables" / "map_feature_v2_v3_pooled_oof.csv"
TABLES_DIR = REPORTS / "tables"

SERIES_STATE_JSON = ROOT / "data" / "features" / "series_team_state_v1_full.json"
MAP_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_map_state_v1.json"
FORM_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_form_state_v1.json"
ROSTER_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_player_roster_state_v1.json"
MODERN_STATE_JSON = ROOT / "data" / "interim" / "pre_cologne_modern_map_state_v1.json"

EXPECTED_TRAIN_UNIQUE = 7762
EXPECTED_TRAIN_AUGMENTED = 15524
PERM_REPEATS = 10

# The 12 V2-era families (A-L), restated here rather than importing
# map_feature_families._RAW_FAMILIES (that module's own partition assertion is
# scoped to V2's 92 raw columns and would reject V3's extra 25) - combined
# below with the 4 Phase 6C families (M/N/O/P, imported unchanged from
# evaluate_map_feature_sets_v3) into one complete V3 taxonomy.
from feature_engineering.maps.map_feature_families import _RAW_FAMILIES as V2_RAW_FAMILIES, FAMILY_LABELS as V2_FAMILY_LABELS

V3_ALL_FAMILY_LABELS = dict(V2_FAMILY_LABELS)
V3_ALL_FAMILY_LABELS.update(V3_FAMILY_LABELS)
REPORT_FAMILIES = ["A", "B", "E", "H", "K", "M", "N", "O", "P"]   # brief section 30's requested subset


def v3_feature_family_map(roles, transformed_feature_names):
    raw_declared = [c for cols in V2_RAW_FAMILIES.values() for c in cols] + \
        [c for cols in V3_FAMILIES.values() for c in cols]
    assert len(raw_declared) == len(set(raw_declared))
    expected_raw = set(roles["directional"]) | set(roles["symmetric"])
    assert set(raw_declared) == expected_raw, (
        f"missing={sorted(expected_raw - set(raw_declared))} extra={sorted(set(raw_declared) - expected_raw)}")
    raw_to_family = {c: fam for fam, cols in {**V2_RAW_FAMILIES, **V3_FAMILIES}.items() for c in cols}
    groups = {k: [] for k in V3_ALL_FAMILY_LABELS}
    by_feature = {}
    for name in transformed_feature_names:
        fam = raw_to_family.get(name)
        if fam is None:
            assert name.startswith(("map_name_", "bestOf_BO", "tier_")), name
            fam = "L"
        by_feature[name] = fam
        groups[fam].append(name)
    return {"groups": groups, "by_feature": by_feature}


def reconstruct_full_train_universe():
    """Brief correction #1: derive the full-TRAIN map universe from the union
    of every (match_id, game_id) appearing in ANY train/validation role
    across all four CV folds - never from map_split_v1.csv."""
    cv = pd.read_csv(CV_FOLDS_PATH)
    union = cv[["match_id", "game_id"]].drop_duplicates()
    n_union = len(union)
    print(f"CV manifest train/validation-role union: {n_union} unique (match_id, game_id) pairs")
    if n_union != EXPECTED_TRAIN_UNIQUE:
        raise AssertionError(
            f"reconstructed TRAIN universe has {n_union} rows, expected {EXPECTED_TRAIN_UNIQUE}. STOPPING.")
    return union


def main():
    roles = load_map_v3_roles(CONFIG_PATH)
    target = roles["target"]
    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    params = dict(selected["params"])
    fixed_params = selected["fixed_params"]
    final_n_estimators = int(selected["final_n_estimators"])
    print(f"Frozen final V3 XGB: {selected['selected_candidate_id']} {params} "
          f"n_estimators={final_n_estimators} (selected via: {selected['selection_stage']})")

    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    cv = load_cv_manifest(verify_against_split=False)   # STRUCTURAL: never opens map_split_v1.csv

    # =====================================================================
    # 1. SELECTED-CONFIG OOF (TRAIN-only development performance)
    # =====================================================================
    cache = XgbFoldCache(cv, features, roles)
    assert cache[1]["X_outer_aug"].shape[1] == EXPECTED_TRANSFORMED_FEATURES == 131

    oof_rows, models_by_fold, prep_by_fold = [], {}, {}
    rf_imp_rows, group_rows = [], []   # "rf_imp_rows" name kept generic; XGB-only in Phase 6D
    for fold in range(1, N_FOLDS + 1):
        e = cache[fold]
        model = XGBClassifier(n_estimators=final_n_estimators, **params, **fixed_params)
        model.fit(e["X_outer_aug"], e["y_outer_aug"], verbose=False)   # NO eval_set, NO early stopping
        p_val = model.predict_proba(e["X_val"])[:, 1]
        models_by_fold[fold] = model

        raw_tr, raw_va = fold_frames(cv, features, fold)
        oof_rows.append(pd.DataFrame({
            "match_id": raw_va["match_id"].to_numpy(), "game_id": raw_va["game_id"].to_numpy(),
            "fold": fold, "map_name": raw_va["map_name"].to_numpy(),
            "y_true": e["y_val"], "p_xgb_v3_final": p_val,
        }))
        prep_by_fold[fold] = (raw_va, e)

        m = compute_metrics(e["y_val"], p_val)
        print(f"fold {fold}: OOF log loss {m['log_loss']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
              f"accuracy {m['accuracy']:.4f}")

        perm = permutation_importance(model, e["X_val"], e["y_val"], scoring="roc_auc",
                                       n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=1)
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        families = v3_feature_family_map(roles, cache.feature_names)
        for i, name in enumerate(cache.feature_names):
            rf_imp_rows.append({"fold": fold, "feature": name, "family": families["by_feature"][name],
                                 "gain": float(gain.get(f"f{i}", 0.0)),
                                 "permutation_importance": float(perm.importances_mean[i])})
        g = grouped_permutation_importance(model, e["X_val"], e["y_val"], cache.feature_names,
                                            families["groups"], PERM_REPEATS, RANDOM_STATE)
        for fam, v in g.items():
            group_rows.append({"fold": fold, "family": fam, "family_label": V3_ALL_FAMILY_LABELS[fam],
                                "n_features": len(families["groups"][fam]), "auc_decrease_mean": v["mean"]})

    oof = pd.concat(oof_rows, ignore_index=True)
    assert oof.duplicated(subset=["match_id", "game_id"]).sum() == 0
    assert oof["p_xgb_v3_final"].between(0, 1).all()
    oof.to_parquet(MODELING_DIR / "map_xgboost_v3_final_oof.parquet", engine="fastparquet", index=False)
    print(f"Wrote {MODELING_DIR / 'map_xgboost_v3_final_oof.parquet'} ({len(oof)} pooled TRAIN-only OOF rows)")

    y = oof["y_true"].to_numpy(dtype=float)
    p = oof["p_xgb_v3_final"].to_numpy()
    pooled_m = compute_metrics(y, p, with_confusion=True)
    pooled_sm = series_macro_metrics(oof["match_id"].to_numpy(), y, p)

    # ---- side-symmetry diagnostic (TRAIN-only, per fold model, brief section 29) ----
    sym_errors = []
    for fold in range(1, N_FOLDS + 1):
        raw_va, e = prep_by_fold[fold]
        # re-fit preprocessing exactly as the fold did (median/vocab only - deterministic, same as XgbFoldCache)
        raw_tr, _ = fold_frames(cv, features, fold)
        aug_tr = build_augmented_training_raw(raw_tr, roles)
        p_outer = prep_xgb.fit_preprocessing(aug_tr, roles)
        mirrored = mirror_raw_rows(raw_va, roles)
        X_mirrored, _ = prep_xgb.transform(mirrored, p_outer, roles)
        p_mirrored = models_by_fold[fold].predict_proba(X_mirrored)[:, 1]
        p_orig = models_by_fold[fold].predict_proba(e["X_val"])[:, 1]
        sym_errors.append(np.abs(p_orig - (1 - p_mirrored)))
    sym_errors = np.concatenate(sym_errors)
    symmetry_stats = {"mean": float(sym_errors.mean()), "median": float(np.median(sym_errors)),
                       "p95": float(np.percentile(sym_errors, 95)), "max": float(sym_errors.max())}
    print(f"Side-symmetry (TRAIN-only OOF, per-fold models): {symmetry_stats}")

    oof_metrics_df = pd.DataFrame([{
        "population": "pooled_train_oof_development", **{k: v for k, v in pooled_m.items()
                                                           if k != "confusion_matrix"},
        "confusion_matrix": json.dumps(pooled_m["confusion_matrix"]), **pooled_sm,
    }])
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    oof_metrics_df.to_csv(TABLES_DIR / "map_xgboost_v3_final_oof_metrics.csv", index=False, encoding="utf-8")

    imp_df = pd.DataFrame(rf_imp_rows).groupby(["feature", "family"], as_index=False).agg(
        gain_mean=("gain", "mean"), permutation_importance_mean=("permutation_importance", "mean"),
    ).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1
    imp_df.to_csv(TABLES_DIR / "map_xgboost_v3_final_feature_importance.csv", index=False, encoding="utf-8")

    group_df = pd.DataFrame(group_rows)
    group_agg = group_df.groupby(["family", "family_label", "n_features"], as_index=False).agg(
        auc_decrease_mean=("auc_decrease_mean", "mean")).sort_values("auc_decrease_mean", ascending=False)
    group_agg.to_csv(TABLES_DIR / "map_xgboost_v3_final_group_importance.csv", index=False, encoding="utf-8")

    # =====================================================================
    # 2. THREE-WAY COMPARISON (feature gain vs development tuning gain)
    # =====================================================================
    v2v3 = pd.read_csv(V2_V3_POOLED_OOF_PATH)
    arm_a = v2v3[(v2v3.arm == "v2") & (v2v3.model == "xgboost")].iloc[0]   # V2 + frozen 6B config (124 trees)
    arm_b = v2v3[(v2v3.arm == "v3") & (v2v3.model == "xgboost")].iloc[0]   # V3 + frozen 6B config (124 trees)
    arm_c = {"log_loss": pooled_m["log_loss"], "roc_auc": pooled_m["roc_auc"],
             "brier": pooled_m["brier"], "accuracy": pooled_m["accuracy"], "n_correct":
             int(((p >= 0.5).astype(int) == y).sum())}
    feature_gain = {"delta_log_loss": arm_b["log_loss"] - arm_a["log_loss"],
                     "delta_roc_auc": arm_b["roc_auc"] - arm_a["roc_auc"],
                     "delta_accuracy": arm_b["accuracy"] - arm_a["accuracy"]}
    tuning_gain = {"delta_log_loss": arm_c["log_loss"] - arm_b["log_loss"],
                   "delta_roc_auc": arm_c["roc_auc"] - arm_b["roc_auc"],
                   "delta_accuracy": arm_c["accuracy"] - arm_b["accuracy"]}
    print(f"Feature gain (A->B, V2->V3 under frozen 6B config): {feature_gain}")
    print(f"Development tuning gain (B->C, V3 old config -> V3 final tuned config): {tuning_gain}")

    # =====================================================================
    # 3. FULL-TRAIN REFIT (universe reconstructed from CV manifest, never map_split_v1.csv)
    # =====================================================================
    union = reconstruct_full_train_universe()
    train_raw = features.merge(union, on=["match_id", "game_id"], how="inner", validate="one_to_one")
    assert len(train_raw) == EXPECTED_TRAIN_UNIQUE, \
        f"joined TRAIN universe has {len(train_raw)} rows, expected {EXPECTED_TRAIN_UNIQUE}. STOPPING."
    train_raw = train_raw.sort_values(["series_datetime", "match_id", "game_id"]).reset_index(drop=True)

    aug_train = build_augmented_training_raw(train_raw, roles)
    assert len(aug_train) == EXPECTED_TRAIN_AUGMENTED == 2 * EXPECTED_TRAIN_UNIQUE, \
        f"augmented TRAIN has {len(aug_train)} rows, expected {EXPECTED_TRAIN_AUGMENTED}. STOPPING."
    assert abs(float(aug_train[target].mean()) - 0.5) < 1e-9
    assert_augmented_symmetry(aug_train, roles)
    print(f"Full-TRAIN refit universe: {len(train_raw):,} unique maps (reconstructed from the CV manifest, "
          f"NOT map_split_v1.csv) -> {len(aug_train):,} augmented observations")

    final_prep = prep_xgb.fit_preprocessing(aug_train, roles)
    X_full_aug, feature_names = prep_xgb.transform(aug_train, final_prep, roles)
    y_full_aug = aug_train[target].to_numpy(dtype=float)

    final_model = XGBClassifier(n_estimators=final_n_estimators, **params, **fixed_params)
    final_model.fit(X_full_aug, y_full_aug, verbose=False)   # NO eval_set, NO early stopping

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    final_model.save_model(str(MODELS_DIR / "map" / "map_xgboost_v3_final.json"))
    prep_xgb.save_preprocessing(final_prep, MODELING_DIR / "map_xgboost_v3_final_preprocessing.json")

    # ---- reload parity ----
    reloaded = XGBClassifier()
    reloaded.load_model(str(MODELS_DIR / "map" / "map_xgboost_v3_final.json"))
    X_train_orig, _ = prep_xgb.transform(train_raw, final_prep, roles)
    p1 = final_model.predict_proba(X_train_orig)[:, 1]
    p2 = reloaded.predict_proba(X_train_orig)[:, 1]
    reload_parity = float(np.max(np.abs(p1 - p2)))
    assert reload_parity < 1e-9, f"reload changed predictions (max abs diff {reload_parity})"
    print(f"Save/load parity: max|dP|={reload_parity:.2e}")

    final_config = {
        "prediction_task": "known_map", "feature_version": "map_features_v3_modern_map",
        "target": target, "params": params, "fixed_params": fixed_params,
        "final_n_estimators": final_n_estimators, "threshold": 0.5,
        "selected_candidate_id": selected["selected_candidate_id"],
        "selection_stage": selected["selection_stage"], "search_plan_hash": selected["search_plan_hash"],
    }
    (MODELING_DIR / "map_xgboost_v3_final_config.json").write_text(
        json.dumps(final_config, indent=2, default=str), encoding="utf-8")

    categorical_vocab = final_prep["categorical"]
    import hashlib
    metadata = {
        "prediction_task": "known_map",
        "feature_version": "map_features_v3_modern_map",
        "raw_feature_count": 120, "transformed_feature_count": 131,
        "target": target,
        "training_unique_map_rows": len(train_raw), "training_augmented_rows": len(aug_train),
        "training_split": "TRAIN only (reconstructed from data/modeling/map_cv_folds_v1.csv, "
                          "NOT map_split_v1.csv)",
        "threshold": 0.5,
        "categorical_vocabularies": categorical_vocab,
        "directional_features_hash": hashlib.sha256(
            json.dumps(roles["directional"]).encode()).hexdigest(),
        "symmetric_features_hash": hashlib.sha256(json.dumps(roles["symmetric"]).encode()).hexdigest(),
        "config_hash": sha256_file(CONFIG_PATH),
        "features_artifact_hash": sha256_file(FEATURES_PATH),
        "cv_manifest_hash": sha256_file(CV_FOLDS_PATH),
        "selected_hyperparameters": params, "fixed_params": fixed_params,
        "final_n_estimators": final_n_estimators,
        "final_n_estimators_rule": selected["final_n_estimators_rule"],
        "random_state": RANDOM_STATE,
        "package_versions": package_versions(),
        "search_plan_hash": selected["search_plan_hash"],
        "selected_candidate_id": selected["selected_candidate_id"],
        "preprocessing_artifact": "data/modeling/map_xgboost_v3_final_preprocessing.json",
        "reload_prediction_max_abs_diff": reload_parity,
        "calibration_applied": False, "predictions_symmetrized": False,
        "ensemble": "none - XGBoost only, Phase 6D does not build a new ensemble",
        "test_status": "SEALED - never loaded in Phase 6D",
        "cologne_status": "UNTOUCHED - never loaded in Phase 6D",
        "main_validation_status": "CONSUMED in Phase 6B - not reopened in Phase 6C or 6D",
        # deliberately NO validation/test metrics below this point (brief section 27)
    }
    (MODELS_DIR / "map" / "map_xgboost_v3_final_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # =====================================================================
    # future-inference parity (brief section 28) - SYNTHETIC matchup only
    # =====================================================================
    series_state = StateStore.from_json(SERIES_STATE_JSON)
    map_state = MapStateStore.from_json(MAP_STATE_JSON)
    form_state = TeamFormStateStore.from_json(FORM_STATE_JSON)
    roster_state = PlayerRosterStateStore.from_json(ROSTER_STATE_JSON)
    modern_state = ModernMapStateStore.from_json(MODERN_STATE_JSON)
    cologne_dt, _ = cologne_cutoff()
    candidate_teams = sorted(set(series_state.teams) & set(map_state.teams()) & set(form_state.teams)
                              & set(roster_state.teams))
    syn_t1, syn_t2 = candidate_teams[0], candidate_teams[1]
    synthetic = build_future_modern_rich_map_features(
        syn_t1, syn_t2, 3, "Mirage", cologne_dt, series_state, map_state, form_state, roster_state,
        modern_state, tier=None)
    assert len(synthetic) == 120
    row = pd.DataFrame([{**synthetic, "match_id": "synthetic", "game_id": "synthetic"}])
    X_syn, syn_names = prep_xgb.transform(row, final_prep, roles)
    assert X_syn.shape == (1, 131)
    p_syn = float(final_model.predict_proba(X_syn)[:, 1][0])
    assert np.isfinite(p_syn) and 0.0 <= p_syn <= 1.0
    print(f"Future-inference parity check (SYNTHETIC matchup, never real Cologne): "
          f"120 raw -> 131 transformed -> p={p_syn:.4f} (finite, in [0,1]) PASSED")

    write_report(selected, pooled_m, pooled_sm, symmetry_stats, arm_a, arm_b, arm_c, feature_gain, tuning_gain,
                 imp_df, group_agg, train_raw, aug_train, p_syn)
    print("\nWrote models/map_xgboost_v3_final.json + metadata")
    print("Wrote data/modeling/map_xgboost_v3_final_{preprocessing,config}.json")
    print("Wrote reports/tables/map_xgboost_v3_final_{oof_metrics,feature_importance,group_importance}.csv")
    print("Wrote reports/phase6d_final_xgboost_v3_results.md")


def write_report(selected, pooled_m, pooled_sm, symmetry_stats, arm_a, arm_b, arm_c, feature_gain, tuning_gain,
                  imp_df, group_agg, train_raw, aug_train, p_syn):
    md = []
    md.append("# Phase 6D - Final XGBoost V3 Results\n")
    md.append("## TRAIN-only selected-model OOF development performance\n")
    md.append("**This is TRAIN-only selected-model OOF development performance, not an unbiased generalization "
              "estimate** - the model was selected using these same four folds. The sealed 1,427-map TEST "
              "partition is the next unbiased internal evaluation and remains closed.\n")
    md.append(f"n={pooled_m['n']}, Accuracy={pooled_m['accuracy']:.4f}, Precision={pooled_m['precision']:.4f}, "
              f"Recall={pooled_m['recall']:.4f}, F1={pooled_m['f1']:.4f}, ROC-AUC={pooled_m['roc_auc']:.4f}, "
              f"Log Loss={pooled_m['log_loss']:.4f}, Brier={pooled_m['brier']:.4f}\n")
    md.append(f"Series-macro: Log Loss={pooled_sm['series_macro_log_loss']:.4f}, "
              f"Brier={pooled_sm['series_macro_brier']:.4f}, "
              f"Accuracy={pooled_sm['series_macro_accuracy']:.4f}\n")
    md.append(f"Side-symmetry (TRAIN-only, per-fold models): mean={symmetry_stats['mean']:.4f}, "
              f"median={symmetry_stats['median']:.4f}, p95={symmetry_stats['p95']:.4f}, "
              f"max={symmetry_stats['max']:.4f}\n")

    md.append("## Three-way comparison (feature gain vs development tuning gain)\n")
    md.append("| arm | description | log loss | ROC-AUC | accuracy |")
    md.append("|---|---|---|---|---|")
    md.append(f"| A | V2 + frozen Phase 6B config (124 trees) | {arm_a['log_loss']:.4f} | "
              f"{arm_a['roc_auc']:.4f} | {arm_a['accuracy']:.4f} |")
    md.append(f"| B | V3 + frozen Phase 6B config (124 trees) | {arm_b['log_loss']:.4f} | "
              f"{arm_b['roc_auc']:.4f} | {arm_b['accuracy']:.4f} |")
    md.append(f"| C | V3 + Phase 6D final tuned config | {arm_c['log_loss']:.4f} | "
              f"{arm_c['roc_auc']:.4f} | {arm_c['accuracy']:.4f} |")
    md.append(f"\n**Feature gain (A -> B)**: Δlog loss {feature_gain['delta_log_loss']:+.4f}, "
              f"ΔROC-AUC {feature_gain['delta_roc_auc']:+.4f}, Δaccuracy {feature_gain['delta_accuracy']:+.4f} pp\n")
    md.append(f"**Development tuning gain (B -> C)**: Δlog loss {tuning_gain['delta_log_loss']:+.4f}, "
              f"ΔROC-AUC {tuning_gain['delta_roc_auc']:+.4f}, "
              f"Δaccuracy {tuning_gain['delta_accuracy']:+.4f} pp - these are NOT the same quantity and are "
              "never attributed to each other.\n")

    md.append("## Grouped permutation importance (TRAIN-only CV, requested families)\n")
    md.append("| family | label | n features | AUC decrease |")
    md.append("|---|---|---|---|")
    for fam in REPORT_FAMILIES:
        r = group_agg[group_agg["family"] == fam]
        if len(r):
            r = r.iloc[0]
            md.append(f"| {fam} | {r['family_label']} | {int(r['n_features'])} | {r['auc_decrease_mean']:+.4f} |")
    md.append("\nInterpretation only - no feature changes follow.\n")

    md.append("## Top 10 individual features (permutation importance)\n")
    md.append("| rank | feature | family | permutation | gain |")
    md.append("|---|---|---|---|---|")
    for _, r in imp_df.head(10).iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['family']} | "
                  f"{r['permutation_importance_mean']:+.4f} | {r['gain_mean']:.4f} |")
    md.append("")

    md.append("## Full-TRAIN refit\n")
    md.append(f"{len(train_raw):,} unique TRAIN maps (universe reconstructed from "
              f"`data/modeling/map_cv_folds_v1.csv`, never `map_split_v1.csv`) -> {len(aug_train):,} augmented "
              "observations. Future-inference parity check (synthetic matchup, never real Cologne data): "
              f"p={p_syn:.4f}, finite, in [0,1] - PASSED.\n")

    md.append("## Status\n")
    md.append("- **FEATURES = FROZEN**\n- **PHASE-6B MAP VALIDATION = CONSUMED, NOT REUSED**\n"
              "- **TEST = SEALED**\n- **COLOGNE = UNTOUCHED**\n- **THRESHOLD = 0.5**\n- **NO CALIBRATION**\n"
              "- **NO NEW ENSEMBLE**\n- **NO POST-SELECTION RETUNING**\n- **SRC = UNCHANGED**\n")

    (REPORTS / "phases" / "phase6d_final_xgboost_v3_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
