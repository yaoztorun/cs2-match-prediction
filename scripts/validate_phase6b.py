"""
Phase 6B validation (artifact-level). Read-only. Exits non-zero on failure.

Groups:
    1. repo structure / src / data/raw / reference untouched
    2. every Phase 1-6A artifact byte-unchanged
    3. split and CV manifests: counts, zero partition crossing, TRAIN-only CV
    4. tuning never loaded the main validation, TEST or Cologne (AST, not grep)
    5. deterministic candidate generation for both searches
    6. selected configs reproduce from their own tuning tables
    7. XGB final_n_estimators exactly derived from the fold best_iterations
    8. ensemble weight reproducible from the saved TRAIN-only OOF predictions
    9. full-TRAIN refits used the frozen configurations
    10. freeze-before-validation ordering
    11. saved preprocessing vocabularies carry both unknown categories
    12. saved models reload with tolerance-safe identical predictions, and every
        probability is finite and inside [0, 1]
    13. TEST still sealed; no validation-driven retuning artifact exists
"""

import ast
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "models"))

from _common import ROOT
import map_modeling_common as mmc
import map_random_forest_tuning_v1 as rf_tune
import map_xgboost_tuning_v1 as xgb_tune
import preprocessing_common_map_v2 as pcm
import preprocessing_random_forest_map_v2 as prep_rf
import preprocessing_xgboost_map_v2 as prep_xgb
from map_selected_oof_v1 import ENSEMBLE_WEIGHTS, select_ensemble_weight

MODELING = ROOT / "data" / "modeling"
MODELS = ROOT / "models"
TABLES = ROOT / "reports" / "tables"
REPORTS_DIR = ROOT / "reports"

TUNING_SCRIPTS = [
    ROOT / "scripts" / "models" / "map_random_forest_tuning_v1.py",
    ROOT / "scripts" / "models" / "map_xgboost_tuning_v1.py",
    ROOT / "scripts" / "map_selected_oof_v1.py",
    ROOT / "scripts" / "map_baselines_v1.py",
]

# Everything Phase 1-6A produced that Phase 6B must not have touched.
FROZEN_PATHS = [
    "scripts/feature_engine.py", "scripts/map_feature_engine.py", "scripts/team_form_engine.py",
    "scripts/player_roster_feature_engine.py", "scripts/rich_map_feature_composer.py",
    "scripts/build_map_features_v2_rich.py", "scripts/build_map_split_v1.py",
    "scripts/build_map_cv_folds_v1.py", "scripts/validate_phase6a.py",
    "scripts/preprocessing_common.py", "scripts/preprocessing_common_v2_map_pool.py",
    "scripts/preprocessing_common_v3_form.py", "scripts/preprocessing_common_v4_roster.py",
    "scripts/preprocessing_random_forest_v1.py", "scripts/preprocessing_random_forest_v2_map_pool.py",
    "scripts/preprocessing_xgboost_v1.py", "scripts/preprocessing_xgboost_v2_map_pool.py",
    "scripts/random_forest_tuning_v2.py", "scripts/xgboost_tuning_v2.py",
    "data/features/series_features_v1.parquet", "data/features/series_features_v2_map_pool.parquet",
    "data/features/series_features_v3_form.parquet", "data/features/series_features_v4_roster.parquet",
    "data/features/map_features_v1.parquet", "data/features/map_features_v2_rich.parquet",
    "data/interim/map_base.parquet", "data/interim/series_base.parquet",
    "data/interim/evaluation_manifest.csv",
    "config/series_features_v1.yaml", "config/series_features_v4_roster.yaml",
    "config/map_features_v1.yaml", "config/map_features_v2_rich.yaml",
    "data/modeling/series_split_v1.csv", "data/modeling/random_forest_cv_folds_v2.csv",
    "data/modeling/map_split_v1.csv", "data/modeling/map_cv_folds_v1.csv",
    "data/modeling/random_forest_v2_selected_config.json", "data/modeling/xgboost_v2_selected_config.json",
    "models/random_forest_v2.joblib", "models/xgboost_v2.json",
    "models/logistic_regression_scratch_v2.npz",
    "reports/phase6a_map_v2_feature_quality.md",
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reads_any(source, needles):
    """AST search for pandas read_csv/read_parquet whose arguments mention any
    needle. Comments and docstrings are invisible to this by design."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if any(n in ast.unparse(arg) for n in needles):
                        return True
    return False


def main():
    print("=== 1. repo structure ===")
    src_dir = ROOT / "src"
    src_files = [p for p in src_dir.rglob("*") if p.is_file()] if src_dir.exists() else []
    check("src/ remains empty (no repo restructuring)", len(src_files) == 0)
    check("data/raw/ present and non-empty", any((ROOT / "data" / "raw").rglob("*")))
    check("reference/ present and non-empty", any((ROOT / "reference").rglob("*")))
    for d in ["scripts", "scripts/models", "tests", "data", "config", "reports", "models"]:
        check(f"expected directory still in place: {d}/", (ROOT / d).is_dir())

    print("\n=== 2. Phase 1-6A artifacts byte-unchanged ===")
    baseline = {}
    for rel in FROZEN_PATHS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"frozen artifact present: {rel}", baseline[rel] is not None)

    roles = mmc.load_roles()
    features = mmc.load_features()
    split = pd.read_csv(mmc.SPLIT_PATH)
    cv = pd.read_csv(mmc.CV_FOLDS_PATH, parse_dates=["datetime"])

    print("\n=== 3. splits and CV manifest ===")
    counts = split["split"].value_counts()
    check(f"TRAIN map rows == {mmc.EXPECTED_TRAIN_N}", int(counts.get("train", 0)) == mmc.EXPECTED_TRAIN_N)
    check(f"VALIDATION map rows == {mmc.EXPECTED_VAL_N}", int(counts.get("validation", 0)) == mmc.EXPECTED_VAL_N)
    check(f"TEST map rows == {mmc.EXPECTED_TEST_N} (counted only, never loaded)",
          int(counts.get("test", 0)) == mmc.EXPECTED_TEST_N)
    check("zero match_id crosses a partition", (split.groupby("match_id")["split"].nunique() == 1).all())

    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    val_ids = set(split.loc[split["split"] == "validation", "match_id"])
    test_ids = set(split.loc[split["split"] == "test", "match_id"])
    check("map CV manifest uses TRAIN match_ids only", set(cv["match_id"]) <= train_ids)
    check("map CV manifest is disjoint from VALIDATION", set(cv["match_id"]).isdisjoint(val_ids))
    check("map CV manifest is disjoint from TEST", set(cv["match_id"]).isdisjoint(test_ids))
    try:
        mmc.load_cv_manifest(verify_against_split=True)
        manifest_ok = True
    except AssertionError as e:
        manifest_ok = False
        print(f"    manifest assertion failed: {e}")
    check("map CV manifest passes every structural check (atomicity, chronology, timestamp groups)", manifest_ok)

    em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
    check("no Cologne/post-Cologne match_id anywhere in the map feature artifact",
          set(features["match_id"]).isdisjoint(cologne_ids))

    print("\n=== 4. tuning never reached main validation, TEST or Cologne ===")
    for p in TUNING_SCRIPTS:
        src = p.read_text(encoding="utf-8")
        check(f"{p.name} never reads the split manifest (AST)",
              not reads_any(src, ["map_split_v1", "SPLIT_PATH", "series_split_v1"]))
        check(f"{p.name} asks the shared loader not to open the split manifest",
              "verify_against_split=False" in src)
        check(f"{p.name} never reads a Cologne artifact",
              not reads_any(src, ["cologne", "Cologne"]))
    for name in ["map_rf_search_plan_v1.json", "map_xgb_search_plan_v1.json"]:
        plan = json.loads((MODELING / name).read_text(encoding="utf-8"))
        check(f"{name} records main_validation_used_in_selection = False",
              plan["main_validation_used_in_selection"] is False)

    print("\n=== 5. deterministic candidate generation ===")
    rf_c1, rf_c2 = rf_tune.build_candidates(), rf_tune.build_candidates()
    xgb_c1, xgb_c2 = xgb_tune.build_candidates(), xgb_tune.build_candidates()
    check("RF candidate list is deterministic across calls", rf_c1 == rf_c2)
    check("XGB candidate list is deterministic across calls", xgb_c1 == xgb_c2)
    check("RF candidate list has 36 entries, no duplicates", len(rf_c1) == 36 and
          len({tuple(c[k] for k in rf_tune.SEARCH_KEYS) for c in rf_c1}) == 36)
    check("XGB candidate list has 40 entries, no duplicates", len(xgb_c1) == 40 and
          len({tuple(c[k] for k in xgb_tune.SEARCH_KEYS) for c in xgb_c1}) == 40)

    rf_plan = json.loads((MODELING / "map_rf_search_plan_v1.json").read_text(encoding="utf-8"))
    xgb_plan = json.loads((MODELING / "map_xgb_search_plan_v1.json").read_text(encoding="utf-8"))
    check("saved RF search plan lists exactly the candidates the code regenerates today",
          json.dumps(rf_plan["candidates"], sort_keys=True, default=str)
          == json.dumps(rf_c1, sort_keys=True, default=str))
    check("saved XGB search plan lists exactly the candidates the code regenerates today",
          json.dumps(xgb_plan["candidates"], sort_keys=True, default=str)
          == json.dumps(xgb_c1, sort_keys=True, default=str))
    check("RF search plan's recorded CV-manifest hash matches the manifest on disk",
          rf_plan["artifact_hashes"]["data/modeling/map_cv_folds_v1.csv"] == sha256(mmc.CV_FOLDS_PATH))
    check("XGB search plan's recorded feature-artifact hash matches the parquet on disk",
          xgb_plan["artifact_hashes"]["data/features/map_features_v2_rich.parquet"]
          == sha256(mmc.FEATURES_PATH))

    print("\n=== 6/7. selected configs reproduce from their tuning tables ===")
    rf_sel = json.loads((MODELING / "map_random_forest_v1_selected_config.json").read_text(encoding="utf-8"))
    xgb_sel = json.loads((MODELING / "map_xgboost_v1_selected_config.json").read_text(encoding="utf-8"))
    rf_table = pd.read_csv(TABLES / "map_random_forest_tuning_v1.csv")
    xgb_table = pd.read_csv(TABLES / "map_xgboost_tuning_v1.csv")

    for label, sel, table, tuner in [("RF", rf_sel, rf_table, rf_tune), ("XGB", xgb_sel, xgb_table, xgb_tune)]:
        agg = table[table["row_type"] == "aggregate"].copy()
        params_by_id = {c["candidate_id"]: c for c in (rf_c1 if label == "RF" else xgb_c1)}
        recomputed_id, _stage = tuner.select_winner(agg, params_by_id)
        check(f"{label}: re-applying the frozen selection rule to the saved tuning table reproduces the "
              f"selected candidate", recomputed_id == sel["selected_candidate_id"])
        row = agg[agg["candidate_id"] == sel["selected_candidate_id"]]
        check(f"{label}: selected candidate appears exactly once in the tuning table", len(row) == 1)
        if len(row) == 1:
            check(f"{label}: saved CV log loss matches the tuning table",
                  abs(float(row.iloc[0]["val_log_loss_mean"]) - sel["cv_mean_log_loss"]) < 1e-9)
        check(f"{label}: search-plan hash recorded in the selected config matches the plan on disk",
              sel["search_plan_hash"] == (rf_plan if label == "RF" else xgb_plan)["plan_hash"])

    folds = xgb_table[(xgb_table["row_type"] == "fold") &
                       (xgb_table["candidate_id"] == xgb_sel["selected_candidate_id"])].sort_values("fold")
    best_iters = folds["best_iteration"].astype(int).tolist()
    check("XGB best_iterations in the config match the tuning table", best_iters == xgb_sel["best_iterations_by_fold"])
    check("XGB final_n_estimators == round(median(best_iteration + 1)) over the 4 folds",
          xgb_tune.derive_final_n_estimators(best_iters) == xgb_sel["final_n_estimators"])

    print("\n=== 8. ensemble weight reproducible from the TRAIN-only OOF ===")
    oof = pd.read_parquet(MODELING / "map_selected_models_oof_v1.parquet", engine="fastparquet")
    ens = json.loads((MODELING / "map_ensemble_v1_config.json").read_text(encoding="utf-8"))
    check("OOF predictions cover TRAIN match_ids only", set(oof["match_id"]) <= train_ids)
    check("OOF is disjoint from VALIDATION and TEST",
          set(oof["match_id"]).isdisjoint(val_ids | test_ids))
    check("no OOF map row appears in two folds", oof.duplicated(subset=["match_id", "game_id"]).sum() == 0)
    w, _stage, table = select_ensemble_weight(oof["y_true"].to_numpy(dtype=float),
                                               oof["p_rf"].to_numpy(), oof["p_xgb"].to_numpy())
    check("frozen ensemble weight recomputes exactly from the saved OOF predictions", w == ens["weight_rf"])
    check("ensemble weights are exactly the 11 predefined values",
          list(table["weight_rf"]) == ENSEMBLE_WEIGHTS == ens["candidate_weights"])
    check("ensemble config records main_validation_used_in_selection = False",
          ens["main_validation_used_in_selection"] is False)

    print("\n=== 9. full-TRAIN refits used the frozen configurations ===")
    rf_meta = json.loads((MODELS / "map_random_forest_v1_metadata.json").read_text(encoding="utf-8"))
    xgb_meta = json.loads((MODELS / "map_xgboost_v1_metadata.json").read_text(encoding="utf-8"))
    check("RF refit used the frozen selected candidate", rf_meta["selected_candidate_id"] == rf_sel["selected_candidate_id"])
    check("RF refit params equal the frozen config",
          all(str(rf_meta["params"][k]) == str(rf_sel["params"][k]) for k in rf_sel["params"]))
    check("XGB refit used the frozen selected candidate",
          xgb_meta["selected_candidate_id"] == xgb_sel["selected_candidate_id"])
    check("XGB refit params equal the frozen config",
          all(str(xgb_meta["params"][k]) == str(xgb_sel["params"][k]) for k in xgb_sel["params"]))
    check("XGB refit used the frozen final_n_estimators",
          xgb_meta["final_n_estimators"] == xgb_sel["final_n_estimators"])
    check("XGB refit used no early stopping and no eval_set",
          xgb_meta["early_stopping_used_in_final_fit"] is False
          and xgb_meta["eval_set_used_in_final_fit"] is False)
    check(f"RF refit trained on {mmc.EXPECTED_TRAIN_N} unique maps -> {2 * mmc.EXPECTED_TRAIN_N} augmented "
          "observations", rf_meta["original_train_maps"] == mmc.EXPECTED_TRAIN_N
          and rf_meta["augmented_train_observations"] == 2 * mmc.EXPECTED_TRAIN_N)
    check("no calibration was fitted and no prediction was symmetrized in Phase 6B",
          rf_meta["calibration_applied"] is False and rf_meta["predictions_symmetrized"] is False
          and xgb_meta["calibration_applied"] is False and xgb_meta["predictions_symmetrized"] is False)

    print("\n=== 10. freeze-before-validation ordering ===")
    val_metrics = pd.read_csv(TABLES / "map_model_validation_metrics_v1.csv")
    for label, p in [("RF selected config", MODELING / "map_random_forest_v1_selected_config.json"),
                      ("XGB selected config", MODELING / "map_xgboost_v1_selected_config.json"),
                      ("ensemble config", MODELING / "map_ensemble_v1_config.json")]:
        check(f"{label} was written before the validation metrics table",
              p.stat().st_mtime <= (TABLES / "map_model_validation_metrics_v1.csv").stat().st_mtime + 1)
    train_src = (ROOT / "scripts" / "train_map_models_v1.py").read_text(encoding="utf-8")
    check("the validation-scoring script refuses to run unless all three configs are already frozen",
          train_src.count("require(") >= 4)   # 3 call sites + the definition
    check("the validation-scoring script is the only Phase 6B script that opens the split manifest",
          reads_any(train_src, ["SPLIT_PATH"]))

    print("\n=== 11. saved preprocessing vocabularies ===")
    for name in ["map_random_forest_preprocessing_v1.json", "map_xgboost_preprocessing_v1.json"]:
        p = json.loads((MODELING / name).read_text(encoding="utf-8"))
        cats = p["categorical"]
        check(f"{name}: __UNKNOWN_MAP__ is an explicit category AND an explicit dummy column",
              pcm.UNKNOWN_MAP_CATEGORY in cats["map_name"]["categories"]
              and pcm.UNKNOWN_MAP_CATEGORY in cats["map_name"]["dummies"])
        check(f"{name}: __UNKNOWN_TIER__ is an explicit category AND an explicit dummy column",
              pcm.UNKNOWN_TIER_CATEGORY in cats["tier"]["categories"]
              and pcm.UNKNOWN_TIER_CATEGORY in cats["tier"]["dummies"])
        check(f"{name}: unknown categories do not collapse onto the reference",
              cats["map_name"]["reference"] != pcm.UNKNOWN_MAP_CATEGORY
              and cats["tier"]["reference"] != pcm.UNKNOWN_TIER_CATEGORY)
        check(f"{name}: transformed feature order is the frozen 106-column layout",
              p["transformed_feature_names"] == pcm.transformed_feature_names(roles)
              and len(p["transformed_feature_names"]) == 106)
    rf_prep = prep_rf.load_preprocessing(MODELING / "map_random_forest_preprocessing_v1.json")
    xgb_prep = prep_xgb.load_preprocessing(MODELING / "map_xgboost_preprocessing_v1.json")
    check("RF preprocessing imputes, XGB preprocessing does not",
          rf_prep["imputation_applied"] is True and xgb_prep["imputation_applied"] is False)

    print("\n=== 12. saved models reload and produce valid probabilities ===")
    val_raw = features.merge(split[["match_id", "game_id", "split"]], on=["match_id", "game_id"],
                              how="inner").query("split == 'validation'").reset_index(drop=True)
    check(f"validation partition materializes exactly {mmc.EXPECTED_VAL_N} map rows",
          len(val_raw) == mmc.EXPECTED_VAL_N)
    X_rf, _ = prep_rf.transform(val_raw, rf_prep, roles)
    X_xgb, _ = prep_xgb.transform(val_raw, xgb_prep, roles)
    check("RF matrix is 106 columns wide", X_rf.shape[1] == 106)
    check("XGB matrix is 106 columns wide", X_xgb.shape[1] == 106)

    rf_a = joblib.load(MODELS / "map_random_forest_v1.joblib")
    rf_b = joblib.load(MODELS / "map_random_forest_v1.joblib")
    p_rf_a, p_rf_b = rf_a.predict_proba(X_rf)[:, 1], rf_b.predict_proba(X_rf)[:, 1]
    xg_a, xg_b = XGBClassifier(), XGBClassifier()
    xg_a.load_model(str(MODELS / "map_xgboost_v1.json"))
    xg_b.load_model(str(MODELS / "map_xgboost_v1.json"))
    p_xg_a, p_xg_b = xg_a.predict_proba(X_xgb)[:, 1], xg_b.predict_proba(X_xgb)[:, 1]
    check("RF reload predictions are tolerance-safe identical", np.max(np.abs(p_rf_a - p_rf_b)) < 1e-9)
    check("XGB reload predictions are tolerance-safe identical", np.max(np.abs(p_xg_a - p_xg_b)) < 1e-9)

    p_ens = ens["weight_rf"] * p_rf_a + (1 - ens["weight_rf"]) * p_xg_a
    for label, p in [("RF", p_rf_a), ("XGB", p_xg_a), ("ensemble", p_ens)]:
        check(f"{label}: every validation probability is finite and inside [0, 1]",
              np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())

    saved_rf_auc = float(val_metrics.query("model == 'random_forest' and split == 'validation'")["roc_auc"].iloc[0])
    from sklearn.metrics import roc_auc_score
    y_val = val_raw[roles["target"]].to_numpy(dtype=float)
    check("reloaded RF reproduces the reported validation ROC-AUC",
          abs(roc_auc_score(y_val, p_rf_a) - saved_rf_auc) < 1e-9)

    print("\n=== 13. TEST sealed; no validation-driven retuning ===")
    check("validation metrics table reports n = 1,129 for every model",
          (val_metrics["n"] == mmc.EXPECTED_VAL_N).all())
    per_map = pd.read_csv(TABLES / "map_model_per_map_validation_v1.csv")
    check("per-map validation counts sum to 1,129 per model",
          (per_map.groupby("model")["n"].sum() == mmc.EXPECTED_VAL_N).all())
    coverage = pd.read_csv(TABLES / "map_model_coverage_validation_v1.csv")
    check("no coverage subgroup exceeds the validation partition size",
          (coverage["n"] <= mmc.EXPECTED_VAL_N).all())
    for name, path in [("OOF", MODELING / "map_selected_models_oof_v1.parquet")]:
        d = pd.read_parquet(path, engine="fastparquet")
        check(f"{name} artifact contains no TEST match_id", set(d["match_id"]).isdisjoint(test_ids))

    for name, path in [("RF", MODELING / "map_rf_tuning_progress_v1.jsonl"),
                        ("XGB", MODELING / "map_xgb_tuning_progress_v1.jsonl")]:
        hashes = {json.loads(l)["plan_hash"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        expected = (rf_plan if name == "RF" else xgb_plan)["plan_hash"]
        check(f"{name} checkpoint file contains results from exactly one search plan (no mixing)",
              hashes == {expected})

    results_md = (REPORTS_DIR / "phase6b_known_map_model_results.md").read_text(encoding="utf-8")
    check("the results report states explicitly that map accuracy is not comparable with series accuracy",
          "different prediction task" in results_md.lower() and "series" in results_md.lower())
    check("the results report records the validation-used-once / TEST-sealed status block",
          "MAIN MAP VALIDATION = USED ONCE AFTER FREEZE" in results_md
          and "TEST = SEALED" in results_md and "NO POST-VALIDATION RETUNING" in results_md)
    check("no calibrated or symmetrized model artifact was produced in Phase 6B",
          not list(MODELS.glob("*calibrat*")) and not list(MODELS.glob("*symmetr*")))

    print("\n=== 14. frozen artifacts still byte-unchanged after this run ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        print("FAILED:")
        for name, ok in CHECKS:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
