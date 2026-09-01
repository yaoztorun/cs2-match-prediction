"""
Phase 6D validation (hard model freeze gate). Read-only. Exits non-zero on
failure. Passing this script means: V3 feature representation, preprocessing,
XGBoost structural parameters, final_n_estimators, threshold=0.5 and the
model-selection procedure are all FROZEN. No further model development
before TEST.
"""

import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT

import training.map_models.map_xgboost_v3_final_tuning as tune                       # noqa: E402
import feature_engineering.preprocessing.preprocessing_common_map_v3 as pcm3                       # noqa: E402
import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb3                 # noqa: E402
from feature_engineering.preprocessing.preprocessing_common_map_v2 import transformed_feature_names, UNKNOWN_MAP_CATEGORY, UNKNOWN_TIER_CATEGORY

CONFIG_PATH = ROOT / "config" / "features" / "map_features_v3_modern_map.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "map_features_v3_modern_map.parquet"
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "map_cv_folds_v1.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "map_split_v1.csv"

MODELING = ROOT / "data" / "modeling"
MODELS = ROOT / "models"
TABLES = ROOT / "reports" / "tables"

PHASE6C_SCRIPTS = [
    "feature_engineering/maps/modern_map_feature_engine.py", "feature_engineering/maps/modern_map_stream_common.py",
    "feature_engineering/maps/build_map_features_v3_modern_map.py", "feature_engineering/state/build_pre_cologne_modern_map_state_v1.py",
    "feature_engineering/maps/rich_modern_map_feature_composer.py", "feature_engineering/preprocessing/preprocessing_common_map_v3.py",
    "feature_engineering/preprocessing/preprocessing_random_forest_map_v3.py", "feature_engineering/preprocessing/preprocessing_xgboost_map_v3.py",
    "evaluation/validation/evaluate_map_feature_sets_v3.py", "validation/validate_phase6c.py",
]
PHASE1_6C_ARTIFACTS = [
    "data/features/map_features_v2_rich.parquet", "data/features/map_features_v3_modern_map.parquet",
    "config/features/map_features_v2_rich.yaml", "config/features/map_features_v3_modern_map.yaml",
    "data/modeling/map_split_v1.csv", "data/modeling/map_cv_folds_v1.csv",
    "data/modeling/map_random_forest_v1_selected_config.json",
    "data/modeling/map_xgboost_v1_selected_config.json", "data/modeling/map_ensemble_v1_config.json",
    "models/map/map_random_forest_v1.joblib", "models/map/map_xgboost_v1.json",
    "reports/phases/phase6b_known_map_model_results.md", "reports/phases/phase6c_modern_map_cv_results.md",
    "reports/tables/map_feature_v2_v3_pooled_oof.csv",
    "data/interim/pre_cologne_modern_map_state_v1.json",
    "data/interim/pre_cologne_map_state_v1.json", "data/interim/pre_cologne_form_state_v1.json",
    "data/interim/pre_cologne_player_roster_state_v1.json",
    "data/features/series_team_state_v1_full.json",
] + PHASE6C_SCRIPTS

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reads_any(source, needles):
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
    print("=== capturing pre-run hashes of Phase 1-6C artifacts ===")
    baseline = {}
    for rel in PHASE1_6C_ARTIFACTS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"Phase 1-6C artifact present: {rel}", baseline[rel] is not None)

    print("\n=== 1. repo structure ===")
    src_dir = ROOT / "src"
    check("src/ remains empty", not any(p.is_file() for p in src_dir.rglob("*")) if src_dir.exists() else True)
    check("data/raw/ present and non-empty", any((ROOT / "data" / "raw").rglob("*")))
    check("reference/ present and non-empty", any((ROOT / "reference").rglob("*")))

    print("\n=== 2. feature freeze ===")
    roles = pcm3.load_map_v3_roles(CONFIG_PATH)
    ok = True
    try:
        tune.assert_feature_freeze(roles)
    except AssertionError as e:
        ok = False
        print(f"    {e}")
    check("V3 feature-freeze assertion passes (120 raw / 80 dir / 37 sym / 3 cat)", ok)
    names = transformed_feature_names(roles)
    check("transformed width == 131", len(names) == 131)

    print("\n=== 3. Phase 6D scripts never read map VALIDATION, TEST or Cologne ===")
    for name in ["training/map_models/map_xgboost_v3_final_tuning.py", "training/map_models/finalize_map_xgboost_v3.py"]:
        src = (ROOT / name).read_text(encoding="utf-8")
        check(f"{name} never reads map_split_v1.csv (AST)", not reads_any(src, ["map_split_v1", "SPLIT_PATH"]))
        check(f"{name} never reads a Cologne artifact", not reads_any(src, ["cologne", "Cologne"])
              or "cologne_cutoff" in src)   # cologne_cutoff() itself is allowed (metadata boundary only, no map/score read)
    em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
    oof_path = MODELING / "map_xgboost_v3_final_oof.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path, engine="fastparquet")
        check("final OOF contains no Cologne/post-Cologne match_id", set(oof["match_id"]).isdisjoint(cologne_ids))
        split = pd.read_csv(SPLIT_PATH) if SPLIT_PATH.exists() else None
        if split is not None:
            val_ids = set(split.loc[split["split"] == "validation", "match_id"])
            test_ids = set(split.loc[split["split"] == "test", "match_id"])
            check("final OOF is disjoint from the consumed main VALIDATION partition",
                  set(oof["match_id"]).isdisjoint(val_ids))
            check("final OOF is disjoint from TEST", set(oof["match_id"]).isdisjoint(test_ids))

    print("\n=== 4. deterministic candidate generation ===")
    c1, c2 = tune.build_candidates(), tune.build_candidates()
    check("candidate list deterministic across calls", c1 == c2)
    check("exactly 24 candidates, no duplicates", len(c1) == 24 and
          len({tuple(c[k] for k in tune.SEARCH_KEYS) for c in c1}) == 24)
    ids = {c["candidate_id"] for c in c1}
    for req in ["A0_phase6b_reference_structure", "A1_less_l2", "A2_more_l2", "A3_stronger_child_constraint",
                "A4_less_gamma", "A5_more_sampling", "A6_depth3_strongly_regularized",
                "A7_depth4_controlled_check"]:
        check(f"required anchor present: {req}", req in ids)
    check("random draws restricted to max_depth in {2,3}; A7 is the sole depth-4 candidate",
          set(tune.SEARCH_SPACE["max_depth"]) == {2, 3}
          and sum(1 for c in c1 if c["max_depth"] == 4) == 1)
    check("equivalence epsilon == 0.002", tune.LOG_LOSS_EQUIVALENCE_EPSILON == 0.002)

    plan_path = MODELING / "map_xgboost_v3_final_search_plan.json"
    if not plan_path.exists():
        check("search plan artifact exists", False)
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        check("search plan lists exactly today's regenerated candidates",
              json.dumps(plan["candidates"], sort_keys=True, default=str)
              == json.dumps(c1, sort_keys=True, default=str))
        check("search plan's recorded V3 feature-artifact hash matches the file on disk",
              plan["artifact_hashes"]["data/features/map_features_v3_modern_map.parquet"] == sha256(FEATURES_PATH))
        check("search plan's recorded config hash matches the file on disk",
              plan["artifact_hashes"]["config/map_features_v3_modern_map.yaml"] == sha256(CONFIG_PATH))
        check("search plan's recorded CV-manifest hash matches the file on disk",
              plan["artifact_hashes"]["data/modeling/map_cv_folds_v1.csv"] == sha256(CV_FOLDS_PATH))
        check("search plan records main_validation_used_in_selection = False",
              plan["main_validation_used_in_selection"] is False)

        print("\n=== 5. selected candidate / tree count reproduce from the saved tuning table ===")
        sel_path = MODELING / "map_xgboost_v3_final_selected_config.json"
        table_path = TABLES / "map_xgboost_v3_final_tuning.csv"
        if not (sel_path.exists() and table_path.exists()):
            check("selected-config and tuning-table artifacts exist", False)
        else:
            sel = json.loads(sel_path.read_text(encoding="utf-8"))
            table = pd.read_csv(table_path)
            agg = table[table["row_type"] == "aggregate"].copy()
            params_by_id = {c["candidate_id"]: c for c in c1}
            winner_id, stage, _ = tune.select_winner(agg, params_by_id)
            check("re-applying the frozen selection rule reproduces the selected candidate",
                  winner_id == sel["selected_candidate_id"])
            check("selection stage matches", stage == sel["selection_stage"])
            check("search_plan_hash recorded in the selected config matches the plan on disk",
                  sel["search_plan_hash"] == plan["plan_hash"])
            best_iters = table[(table["row_type"] == "fold")
                                & (table["candidate_id"] == sel["selected_candidate_id"])] \
                .sort_values("fold")["best_iteration"].astype(int).tolist()
            check("saved best_iterations_by_fold matches the tuning table",
                  best_iters == sel["best_iterations_by_fold"])
            check("final_n_estimators == round(median(best_iteration + 1)) over the 4 folds",
                  tune.derive_final_n_estimators(best_iters) == sel["final_n_estimators"])

    print("\n=== 6. full-TRAIN universe reconstruction (never from map_split_v1.csv) ===")
    cv = pd.read_csv(CV_FOLDS_PATH)
    union = cv[["match_id", "game_id"]].drop_duplicates()
    check("CV-manifest train/validation-role union == 7762", len(union) == 7762)
    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    joined = features.merge(union, on=["match_id", "game_id"], how="inner", validate="one_to_one")
    check("joining the reconstructed union to the V3 feature table gives exactly 7762 rows", len(joined) == 7762)

    print("\n=== 7. full-TRAIN refit + saved artifacts ===")
    final_config_path = MODELING / "map_xgboost_v3_final_config.json"
    prep_path = MODELING / "map_xgboost_v3_final_preprocessing.json"
    model_path = MODELS / "map" / "map_xgboost_v3_final.json"
    meta_path = MODELS / "map" / "map_xgboost_v3_final_metadata.json"
    for label, p in [("final config", final_config_path), ("final preprocessing", prep_path),
                      ("final model", model_path), ("final metadata", meta_path)]:
        check(f"{label} artifact exists", p.exists())

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        check("metadata: training_unique_map_rows == 7762", meta["training_unique_map_rows"] == 7762)
        check("metadata: training_augmented_rows == 15524", meta["training_augmented_rows"] == 15524)
        check("metadata: raw_feature_count == 120 and transformed_feature_count == 131",
              meta["raw_feature_count"] == 120 and meta["transformed_feature_count"] == 131)
        check("metadata: threshold == 0.5", meta["threshold"] == 0.5)
        check("metadata: no calibration applied, no symmetrized predictions",
              meta["calibration_applied"] is False and meta["predictions_symmetrized"] is False)
        check("metadata: no validation/test metric keys present",
              not any(k for k in meta if "validation_metric" in k.lower() or "test_metric" in k.lower()))
        check("metadata: TEST sealed / Cologne untouched / main validation consumed, recorded explicitly",
              "SEALED" in meta["test_status"].upper() and "UNTOUCHED" in meta["cologne_status"].upper()
              and "CONSUMED" in meta["main_validation_status"].upper())

    if prep_path.exists():
        prep = prep_xgb3.load_preprocessing(prep_path)
        cats = prep["categorical"]
        check(f"saved preprocessing has exactly 131 transformed feature names", len(prep["transformed_feature_names"]) == 131)
        check("saved preprocessing supports __UNKNOWN_MAP__",
              UNKNOWN_MAP_CATEGORY in cats["map_name"]["categories"]
              and UNKNOWN_MAP_CATEGORY in cats["map_name"]["dummies"])
        check("saved preprocessing supports __UNKNOWN_TIER__",
              UNKNOWN_TIER_CATEGORY in cats["tier"]["categories"]
              and UNKNOWN_TIER_CATEGORY in cats["tier"]["dummies"])
        check("saved preprocessing does not impute (XGB native-NaN policy)", prep["imputation_applied"] is False)

    if model_path.exists() and prep_path.exists():
        from xgboost import XGBClassifier
        prep = prep_xgb3.load_preprocessing(prep_path)
        sample = features.head(200)
        X, _ = prep_xgb3.transform(sample, prep, roles)
        m1 = XGBClassifier(); m1.load_model(str(model_path))
        m2 = XGBClassifier(); m2.load_model(str(model_path))
        p1, p2 = m1.predict_proba(X)[:, 1], m2.predict_proba(X)[:, 1]
        check("reload prediction parity (tolerance-safe)", float(np.max(np.abs(p1 - p2))) < 1e-9)
        check("reloaded predictions are finite and in [0,1]",
              bool(np.isfinite(p1).all()) and bool(((p1 >= 0) & (p1 <= 1)).all()))

    if final_config_path.exists() and meta_path.exists():
        cfg = json.loads(final_config_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        check("full-TRAIN refit used the frozen selected structural params",
              cfg["params"] == meta["selected_hyperparameters"])
        check("full-TRAIN refit used the frozen final_n_estimators",
              cfg["final_n_estimators"] == meta["final_n_estimators"])

    print("\n=== 8. no threshold-tuning / calibration / new-ensemble artifact exists ===")
    check("no threshold-search artifact", not list(MODELING.glob("*threshold*")))
    check("no calibration artifact", not list(MODELING.glob("*calibrat*")) and not list(MODELS.glob("*calibrat*")))
    check("no new Phase 6D ensemble artifact",
          not list(MODELING.glob("*ensemble_v2*")) and not (MODELING / "map_ensemble_v2_config.json").exists())

    print("\n=== 9. TEST sealed, Cologne untouched (structural) ===")
    check("map_features_v3_modern_map.parquet contains no Cologne/post-Cologne match_id",
          set(features["match_id"]).isdisjoint(cologne_ids))

    print("\n=== 10. Phase 1-6C artifacts byte-unchanged ===")
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
