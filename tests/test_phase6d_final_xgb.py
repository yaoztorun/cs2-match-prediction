"""
Tests for the Phase 6D final XGBoost V3 tuning/freeze stack (brief section 33).
"""

import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODELS_SCRIPTS_DIR = SCRIPTS_DIR / "models"
for d in (SCRIPTS_DIR, MODELS_SCRIPTS_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import map_xgboost_v3_final_tuning as tune                     # noqa: E402
import preprocessing_common_map_v3 as pcm3                     # noqa: E402
import preprocessing_xgboost_map_v3 as prep_xgb3                # noqa: E402
from preprocessing_random_forest_map_v3 import EXPECTED_TRANSFORMED_FEATURES  # noqa: E402
from preprocessing_common_map_v2 import UNKNOWN_MAP_CATEGORY, UNKNOWN_TIER_CATEGORY  # noqa: E402
from map_modeling_common import split_inner_early_stop, N_FOLDS, LOG_LOSS_EQUIVALENCE_EPSILON  # noqa: E402

ROOT = SCRIPTS_DIR.parent


@pytest.fixture(scope="module")
def roles():
    return pcm3.load_map_v3_roles(ROOT / "config" / "map_features_v3_modern_map.yaml")


# --- feature freeze (brief section 3) ---

def test_v3_config_is_exactly_120_raw_inputs(roles):
    assert len(roles["directional"]) == 80
    assert len(roles["symmetric"]) == 37
    assert len(roles["categorical"]) == 3
    assert len(roles["model_features"]) == 120
    assert roles["target"] == "team1_map_win"


def test_transformed_width_is_exactly_131(roles):
    from preprocessing_common_map_v2 import transformed_feature_names
    names = transformed_feature_names(roles)
    assert len(names) == 131 == EXPECTED_TRANSFORMED_FEATURES
    assert len(set(names)) == 131


def test_feature_freeze_assertion_passes_on_the_real_config(roles):
    tune.assert_feature_freeze(roles)   # raises on failure


def test_unknown_map_and_unknown_tier_supported():
    from preprocessing_common_map_v2 import transformed_feature_names
    names = transformed_feature_names(pcm3.load_map_v3_roles(ROOT / "config" / "map_features_v3_modern_map.yaml"))
    assert f"map_name_{UNKNOWN_MAP_CATEGORY}" in names
    assert f"tier_{UNKNOWN_TIER_CATEGORY}" in names


# --- deterministic candidate generation ---

def test_candidates_are_deterministic_24_no_duplicates():
    a, b = tune.build_candidates(), tune.build_candidates()
    assert a == b
    assert len(a) == 24
    keys = [tuple(c[k] for k in tune.SEARCH_KEYS) for c in a]
    assert len(set(keys)) == len(keys)
    ids = [c["candidate_id"] for c in a]
    assert len(set(ids)) == len(ids)


def test_required_anchors_present():
    ids = {c["candidate_id"] for c in tune.build_candidates()}
    for req in ["A0_phase6b_reference_structure", "A1_less_l2", "A2_more_l2", "A3_stronger_child_constraint",
                "A4_less_gamma", "A5_more_sampling", "A6_depth3_strongly_regularized",
                "A7_depth4_controlled_check"]:
        assert req in ids


def test_a0_carries_exact_phase6b_structural_params():
    a0 = next(c for c in tune.build_candidates() if c["candidate_id"] == "A0_phase6b_reference_structure")
    assert a0["learning_rate"] == 0.03 and a0["max_depth"] == 2 and a0["min_child_weight"] == 10
    assert a0["subsample"] == 0.75 and a0["colsample_bytree"] == 0.85 and a0["gamma"] == 5.0
    assert a0["reg_alpha"] == 0.01 and a0["reg_lambda"] == 10.0


def test_only_one_depth4_anchor_random_draws_restricted_to_2_3():
    cands = tune.build_candidates()
    depth4 = [c for c in cands if c["max_depth"] == 4]
    assert len(depth4) == 1 and depth4[0]["candidate_id"] == "A7_depth4_controlled_check"
    assert set(tune.SEARCH_SPACE["max_depth"]) == {2, 3}


def test_equivalence_epsilon_is_exactly_0_002():
    assert LOG_LOSS_EQUIVALENCE_EPSILON == 0.002


# --- inner early-stop split (reused from map_modeling_common, re-verified against real V3 data) ---

def test_inner_split_chronological_and_atomic_on_real_v3_data():
    from map_modeling_common import load_cv_manifest, fold_frames
    features = pd.read_parquet(ROOT / "data" / "features" / "map_features_v3_modern_map.parquet",
                                engine="fastparquet")
    cv = load_cv_manifest(verify_against_split=False)
    for fold in range(1, N_FOLDS + 1):
        raw_tr, raw_va = fold_frames(cv, features, fold)
        fit, es = split_inner_early_stop(raw_tr)
        assert fit["series_datetime"].max() < es["series_datetime"].min()
        assert set(fit["match_id"]).isdisjoint(set(es["match_id"]))
        assert es["series_datetime"].max() < raw_va["series_datetime"].min()


def test_this_script_never_reads_the_map_split_file():
    for name in ["models/map_xgboost_v3_final_tuning.py", "finalize_map_xgboost_v3.py"]:
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                f = node.func
                fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
                if fname in {"read_csv", "read_parquet"}:
                    for arg in list(node.args) + [kw.value for kw in node.keywords]:
                        assert "map_split_v1" not in ast.unparse(arg), f"{name} reads map_split_v1: {ast.unparse(arg)}"


def test_outer_validation_never_used_for_early_stopping():
    src = inspect.getsource(tune.evaluate_on_fold)
    tree = ast.parse(textwrap.dedent(src))
    fits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (f.attr if isinstance(f, ast.Attribute) else "") == "fit":
                fits.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords})
    with_eval = [f for f in fits if "eval_set" in f]
    assert len(with_eval) == 1
    assert "X_inner_es" in with_eval[0]["eval_set"] and "X_val" not in with_eval[0]["eval_set"]
    scoring_fit = [f for f in fits if "eval_set" not in f]
    assert len(scoring_fit) == 1
    assert "early_stopping" not in json.dumps(scoring_fit[0])


def _fit_call_kwargs(src, var_name):
    """AST (not substring) search: kwargs of every `<var_name>.fit(...)` call
    in `src`. Immune to comments/strings mentioning the same words."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var_name):
            out.append({kw.arg for kw in node.keywords})
    return out


def _ctor_call_kwargs(src, class_name):
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == class_name:
            out.append({kw.arg for kw in node.keywords})
    return out


def test_final_refit_has_no_eval_set_or_early_stopping():
    src = (SCRIPTS_DIR / "finalize_map_xgboost_v3.py").read_text(encoding="utf-8")
    fits = _fit_call_kwargs(src, "final_model")
    assert len(fits) == 1
    assert "eval_set" not in fits[0]
    # finalize_map_xgboost_v3.py never early-stops anything - no XGBClassifier
    # constructor call in this file passes early_stopping_rounds at all.
    ctors = _ctor_call_kwargs(src, "XGBClassifier")
    assert not any("early_stopping_rounds" in c for c in ctors)


# --- reproducibility against the saved real artifacts ---

def test_selected_candidate_reproduces_from_the_saved_tuning_table():
    sel_path = ROOT / "data" / "modeling" / "map_xgboost_v3_final_selected_config.json"
    table_path = ROOT / "reports" / "tables" / "map_xgboost_v3_final_tuning.csv"
    if not (sel_path.exists() and table_path.exists()):
        pytest.skip("tuning has not been run yet")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)
    agg = table[table["row_type"] == "aggregate"].copy()
    candidates = tune.build_candidates()
    params_by_id = {c["candidate_id"]: c for c in candidates}
    winner_id, stage, _ = tune.select_winner(agg, params_by_id)
    assert winner_id == sel["selected_candidate_id"]
    assert stage == sel["selection_stage"]


def test_final_tree_count_median_reproduces():
    sel_path = ROOT / "data" / "modeling" / "map_xgboost_v3_final_selected_config.json"
    if not sel_path.exists():
        pytest.skip("tuning has not been run yet")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    recomputed = tune.derive_final_n_estimators(sel["best_iterations_by_fold"])
    assert recomputed == sel["final_n_estimators"]


def test_final_oof_used_identical_tree_count_for_every_fold_no_early_stopping():
    oof_path = ROOT / "data" / "modeling" / "map_xgboost_v3_final_oof.parquet"
    sel_path = ROOT / "data" / "modeling" / "map_xgboost_v3_final_selected_config.json"
    if not (oof_path.exists() and sel_path.exists()):
        pytest.skip("finalize has not been run yet")
    src = (SCRIPTS_DIR / "finalize_map_xgboost_v3.py").read_text(encoding="utf-8")
    fits = _fit_call_kwargs(src, "model")
    assert len(fits) == 1 and "eval_set" not in fits[0]
    ctors = _ctor_call_kwargs(src, "XGBClassifier")
    assert not any("early_stopping_rounds" in c for c in ctors)
    oof_section = src.split("oof_rows, models_by_fold")[1].split("oof = pd.concat")[0]
    assert "n_estimators=final_n_estimators" in oof_section


def test_threshold_is_0_5_and_no_threshold_search_artifact():
    cfg = json.loads((ROOT / "data" / "modeling" / "map_xgboost_v3_final_config.json").read_text(encoding="utf-8"))
    assert cfg["threshold"] == 0.5
    assert not list((ROOT / "data" / "modeling").glob("*threshold*"))


def test_no_calibration_artifact_or_import():
    """AST-based: no actual import of a calibration tool. The metadata dict's
    own `"calibration_applied": False` documentation field is not a false
    positive here, unlike a naive substring search."""
    for name in ["models/map_xgboost_v3_final_tuning.py", "finalize_map_xgboost_v3.py"]:
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(a.name for a in node.names)
        bad = {m for m in imported if "calibrat" in m.lower() or "isotonic" in m.lower()}
        assert not bad, f"{name} imports a calibration tool: {bad}"
    meta_path = ROOT / "models" / "map_xgboost_v3_final_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["calibration_applied"] is False


# --- save/reload + future-composer inference parity ---

def test_saved_model_reload_prediction_parity():
    model_path = ROOT / "models" / "map_xgboost_v3_final.json"
    prep_path = ROOT / "data" / "modeling" / "map_xgboost_v3_final_preprocessing.json"
    if not (model_path.exists() and prep_path.exists()):
        pytest.skip("finalize has not been run yet")
    from xgboost import XGBClassifier
    roles_ = pcm3.load_map_v3_roles(ROOT / "config" / "map_features_v3_modern_map.yaml")
    params = prep_xgb3.load_preprocessing(prep_path)
    df = pd.read_parquet(ROOT / "data" / "features" / "map_features_v3_modern_map.parquet",
                          engine="fastparquet").head(100)
    X, _ = prep_xgb3.transform(df, params, roles_)
    m1 = XGBClassifier(); m1.load_model(str(model_path))
    m2 = XGBClassifier(); m2.load_model(str(model_path))
    p1, p2 = m1.predict_proba(X)[:, 1], m2.predict_proba(X)[:, 1]
    assert np.max(np.abs(p1 - p2)) < 1e-9
    assert np.isfinite(p1).all() and ((p1 >= 0) & (p1 <= 1)).all()


def test_future_composer_inference_parity_synthetic_only():
    model_path = ROOT / "models" / "map_xgboost_v3_final.json"
    prep_path = ROOT / "data" / "modeling" / "map_xgboost_v3_final_preprocessing.json"
    if not (model_path.exists() and prep_path.exists()):
        pytest.skip("finalize has not been run yet")
    from xgboost import XGBClassifier
    from feature_engine import StateStore
    from map_feature_engine import MapStateStore
    from team_form_engine import TeamFormStateStore
    from player_roster_feature_engine import PlayerRosterStateStore
    from modern_map_feature_engine import ModernMapStateStore
    from map_stream_common import cologne_cutoff
    from rich_modern_map_feature_composer import build_future_modern_rich_map_features

    series_state = StateStore.from_json(ROOT / "data" / "features" / "series_team_state_v1_full.json")
    map_state = MapStateStore.from_json(ROOT / "data" / "interim" / "pre_cologne_map_state_v1.json")
    form_state = TeamFormStateStore.from_json(ROOT / "data" / "interim" / "pre_cologne_form_state_v1.json")
    roster_state = PlayerRosterStateStore.from_json(
        ROOT / "data" / "interim" / "pre_cologne_player_roster_state_v1.json")
    modern_state = ModernMapStateStore.from_json(
        ROOT / "data" / "interim" / "pre_cologne_modern_map_state_v1.json")
    cologne_dt, _ = cologne_cutoff()
    teams = sorted(set(series_state.teams) & set(map_state.teams()) & set(form_state.teams)
                    & set(roster_state.teams))
    t1, t2 = teams[0], teams[1]
    synthetic = build_future_modern_rich_map_features(t1, t2, 3, "Mirage", cologne_dt, series_state, map_state,
                                                        form_state, roster_state, modern_state, tier=None)
    assert len(synthetic) == 120

    roles_ = pcm3.load_map_v3_roles(ROOT / "config" / "map_features_v3_modern_map.yaml")
    params = prep_xgb3.load_preprocessing(prep_path)
    row = pd.DataFrame([synthetic])
    X, names = prep_xgb3.transform(row, params, roles_)
    assert X.shape == (1, 131)

    model = XGBClassifier(); model.load_model(str(model_path))
    p = float(model.predict_proba(X)[:, 1][0])
    assert np.isfinite(p) and 0.0 <= p <= 1.0


def test_full_train_universe_reconstruction_matches_expected_counts():
    cv_path = ROOT / "data" / "modeling" / "map_cv_folds_v1.csv"
    cv = pd.read_csv(cv_path)
    union = cv[["match_id", "game_id"]].drop_duplicates()
    assert len(union) == 7762
    features = pd.read_parquet(ROOT / "data" / "features" / "map_features_v3_modern_map.parquet",
                                engine="fastparquet")
    joined = features.merge(union, on=["match_id", "game_id"], how="inner", validate="one_to_one")
    assert len(joined) == 7762
