"""
Tests for the Phase 6B known-map modeling stack (brief section 38).

Everything that can be tested on small synthetic fixtures is tested that way;
the handful of checks that are genuinely about the real frozen artifacts
(manifest atomicity, TEST/Cologne inaccessibility, saved-model reload parity)
read those artifacts read-only.
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

MODELS_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "training" / "map_models"

import training.map_models.map_modeling_common as mmc                      # noqa: E402
import training.map_models.map_random_forest_tuning_v1 as rf_tune          # noqa: E402
import training.map_models.map_xgboost_tuning_v1 as xgb_tune               # noqa: E402
import feature_engineering.preprocessing.preprocessing_common_map_v2 as pcm              # noqa: E402
import feature_engineering.preprocessing.preprocessing_random_forest_map_v2 as prep_rf   # noqa: E402
import feature_engineering.preprocessing.preprocessing_xgboost_map_v2 as prep_xgb        # noqa: E402
from feature_engineering.maps.map_feature_families import feature_family_map, FAMILY_LABELS   # noqa: E402
from training.map_models.map_selected_oof_v1 import ENSEMBLE_WEIGHTS, select_ensemble_weight   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def roles():
    return mmc.load_roles()


def synthetic_frame(n=40, seed=0, roles=None):
    """A small raw frame carrying every config-declared predictive column."""
    rng = np.random.RandomState(seed)
    data = {}
    for c in roles["directional"]:
        data[c] = rng.normal(size=n)
    for c in roles["symmetric_continuous"]:
        data[c] = rng.uniform(0, 10, size=n)
    for c in roles["symmetric_binary"]:
        data[c] = rng.randint(0, 2, size=n).astype(float)
    data["map_name"] = rng.choice(["Mirage", "Nuke", "Ancient"], size=n)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    data[roles["target"]] = rng.randint(0, 2, size=n)
    data["match_id"] = [f"M{i // 2}" for i in range(n)]
    data["game_id"] = [f"g{i}" for i in range(n)]
    data["series_datetime"] = pd.date_range("2024-01-01", periods=n // 2).repeat(2)[:n]
    return pd.DataFrame(data)


# --- config-driven loading -------------------------------------------------

def test_config_declares_exactly_95_predictive_inputs(roles):
    assert len(roles["directional"]) == 62
    assert len(roles["symmetric"]) == 30
    assert len(roles["categorical"]) == 3
    assert len(roles["model_features"]) == 95 == pcm.EXPECTED_RAW_PREDICTIVE_INPUTS
    assert len(roles["symmetric_binary"]) + len(roles["symmetric_continuous"]) == 30


def test_predictors_come_from_the_config_not_from_numeric_columns(roles):
    df = pd.read_parquet(ROOT / "data" / "features" / "map_features_v2_rich.parquet", engine="fastparquet")
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    # numeric columns include the target and numeric metadata; the model feature
    # list is strictly smaller and is never derived by selecting numeric columns
    assert set(roles["model_features"]) < numeric | {"map_name", "tier"}
    assert roles["target"] not in roles["model_features"]
    for c in ["match_id", "game_id", "team1_map_matches_before", "team2_map_matches_before"]:
        assert c not in roles["model_features"]


def test_forbidden_columns_never_enter_the_model_matrix(roles):
    df = pd.read_parquet(ROOT / "data" / "features" / "map_features_v2_rich.parquet", engine="fastparquet")
    mmc.assert_target_and_no_forbidden_columns(df, roles)
    for bad in mmc.FORBIDDEN_PREDICTORS:
        assert bad not in roles["model_features"]


# --- transformed schema ----------------------------------------------------

def test_transformed_feature_order_is_deterministic_and_106_wide(roles):
    names = pcm.transformed_feature_names(roles)
    assert len(names) == 106 == pcm.EXPECTED_TRANSFORMED_FEATURES
    assert names == pcm.transformed_feature_names(roles)      # stable across calls
    assert len(set(names)) == len(names)
    assert names[:62] == roles["directional"]                  # directional block first, in config order


def test_unknown_map_and_unknown_tier_have_their_own_dummy_columns(roles):
    names = pcm.transformed_feature_names(roles)
    assert "map_name___UNKNOWN_MAP__" in names
    assert "tier___UNKNOWN_TIER__" in names
    assert f"map_name_{pcm.MAP_REFERENCE}" not in names        # reference category has no dummy
    assert f"tier_{pcm.TIER_REFERENCE}" not in names


def test_unknown_categories_do_not_collapse_onto_the_reference(roles):
    df = synthetic_frame(4, seed=1, roles=roles)
    df.loc[:, "map_name"] = [pcm.UNKNOWN_MAP_CATEGORY, "Mirage", "Nuke", "SomeBrandNewMap"]
    df.loc[:, "tier"] = [pcm.UNKNOWN_TIER_CATEGORY, "tier1", "tier2", None]
    params = prep_rf.fit_preprocessing(df, roles)
    X, names = prep_rf.transform(df, params, roles)
    col = {n: X[:, i] for i, n in enumerate(names)}

    unk_map = col["map_name___UNKNOWN_MAP__"]
    unk_tier = col["tier___UNKNOWN_TIER__"]
    # rows 0 and 3 are unknown map; rows 0 and 3 are unknown tier
    assert unk_map.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert unk_tier.tolist() == [1.0, 0.0, 0.0, 1.0]
    # a reference-category row (Mirage / tier1) is all-zero across its dummies,
    # and the unknown row is NOT identical to it
    map_dummy_names = [n for n in names if n.startswith("map_name_")]
    ref_row = np.array([col[n][1] for n in map_dummy_names])
    unk_row = np.array([col[n][0] for n in map_dummy_names])
    assert ref_row.sum() == 0.0
    assert not np.array_equal(ref_row, unk_row)


def test_categoricals_are_never_nan_after_the_unknown_contract(roles):
    df = synthetic_frame(6, seed=2, roles=roles)
    df.loc[0, "map_name"] = None
    df.loc[1, "tier"] = None
    m, b, t = pcm.resolve_categoricals(df)
    assert m.notna().all() and b.notna().all() and t.notna().all()
    assert m.iloc[0] == pcm.UNKNOWN_MAP_CATEGORY
    assert t.iloc[1] == pcm.UNKNOWN_TIER_CATEGORY


# --- mirroring -------------------------------------------------------------

def test_mirroring_negates_directional_preserves_symmetric_flips_target(roles):
    df = synthetic_frame(20, seed=3, roles=roles)
    m = pcm.mirror_raw_rows(df, roles)
    for c in roles["directional"]:
        assert np.allclose(m[c].to_numpy(), -df[c].to_numpy())
    for c in roles["symmetric"]:
        assert np.array_equal(m[c].to_numpy(), df[c].to_numpy())
    for c in roles["categorical"]:
        assert (m[c].to_numpy() == df[c].to_numpy()).all()
    assert np.array_equal(m[roles["target"]].to_numpy(), 1 - df[roles["target"]].to_numpy())


def test_augmentation_doubles_observations_and_balances_the_target(roles):
    df = synthetic_frame(20, seed=4, roles=roles)
    aug = pcm.build_augmented_training_raw(df, roles)
    assert len(aug) == 2 * len(df)
    assert abs(aug[roles["target"]].mean() - 0.5) < 1e-12
    pcm.assert_augmented_symmetry(aug, roles)


def test_mirroring_is_training_side_only_in_every_harness(roles):
    """Neither fold cache ever mirrors a validation (or inner-early-stop) block."""
    src = (ROOT / "training" / "map_models" / "map_modeling_common.py").read_text(encoding="utf-8")
    xgb_src = (MODELS_SCRIPTS_DIR / "map_xgboost_tuning_v1.py").read_text(encoding="utf-8")
    train_src = (ROOT / "training" / "map_models" / "train_map_models_v1.py").read_text(encoding="utf-8")
    for text in (src, xgb_src, train_src):
        for line in text.splitlines():
            if "build_augmented_training_raw(" in line and "def " not in line:
                assert not any(tok in line for tok in ("raw_va", "inner_es", "val_raw", "X_val")), line


# --- missing-value handling ------------------------------------------------

def test_rf_imputes_with_fold_train_medians_only(roles):
    train = synthetic_frame(30, seed=5, roles=roles)
    other = synthetic_frame(10, seed=6, roles=roles)
    col = roles["directional"][0]
    train[col] = np.arange(30, dtype=float)
    other[col] = np.nan

    params = prep_rf.fit_preprocessing(train, roles)
    expected_median = float(np.median(np.arange(30, dtype=float)))
    assert params["train_medians"][col] == expected_median

    X, names = prep_rf.transform(other, params, roles)
    assert np.allclose(X[:, names.index(col)], expected_median)
    assert np.isfinite(X).all()


def test_rf_stops_rather_than_inventing_a_value_for_an_all_nan_feature(roles):
    train = synthetic_frame(20, seed=7, roles=roles)
    col = roles["directional"][3]
    train[col] = np.nan
    with pytest.raises(RuntimeError, match=col):
        prep_rf.fit_preprocessing(train, roles)


def test_xgb_preserves_nan_natively(roles):
    df = synthetic_frame(20, seed=8, roles=roles)
    col = roles["directional"][2]
    df.loc[0:4, col] = np.nan
    params = prep_xgb.fit_preprocessing(df, roles)
    assert params["imputation_applied"] is False
    X, names = prep_xgb.transform(df, params, roles)
    assert np.isnan(X[0:5, names.index(col)]).all()
    n_cat = len(pcm.MAP_DUMMIES) + len(pcm.BESTOF_DUMMIES) + len(pcm.TIER_DUMMIES)
    assert np.isfinite(X[:, -n_cat:]).all()


# --- fold structure --------------------------------------------------------

def test_map_cv_folds_keep_series_atomic_and_timestamp_groups_intact():
    cv = mmc.load_cv_manifest(verify_against_split=True)   # raises on any violation
    assert (cv.groupby(["fold", "match_id"])["role"].nunique() == 1).all()
    for fold in range(1, mmc.N_FOLDS + 1):
        f = cv[cv["fold"] == fold]
        tr, va = f[f.role == "train"], f[f.role == "validation"]
        assert tr["datetime"].max() < va["datetime"].min()
        assert set(tr["datetime"]).isdisjoint(set(va["datetime"]))


def test_inner_early_stop_split_is_chronological_and_series_atomic():
    roles = mmc.load_roles()
    features = mmc.load_features()
    cv = mmc.load_cv_manifest(verify_against_split=False)
    for fold in range(1, mmc.N_FOLDS + 1):
        raw_tr, raw_va = mmc.fold_frames(cv, features, fold)
        fit, es = mmc.split_inner_early_stop(raw_tr)
        assert len(fit) and len(es)
        assert fit["series_datetime"].max() < es["series_datetime"].min()
        assert set(fit["match_id"]).isdisjoint(set(es["match_id"]))
        assert set(fit["series_datetime"]).isdisjoint(set(es["series_datetime"]))
        # the OUTER fold validation is untouched by the inner split
        assert es["series_datetime"].max() < raw_va["series_datetime"].min()
        assert len(fit) + len(es) == len(raw_tr)


def _calls_in(func, name_filter):
    """AST (not substring) search: every Call node inside `func`'s source whose
    callee name matches, returned as (kwargs dict of unparsed values, unparsed
    positional args)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if fname == name_filter:
            out.append(({kw.arg: ast.unparse(kw.value) for kw in node.keywords},
                        [ast.unparse(a) for a in node.args]))
    return out


def test_outer_validation_is_never_used_for_xgb_early_stopping():
    """Exactly one fit() in the XGB tuner's per-fold routine passes an eval_set,
    and that eval_set is the INNER EARLY STOP block - never the outer fold
    validation block."""
    fits = _calls_in(xgb_tune.evaluate_on_fold, "fit")
    assert len(fits) == 2, fits
    with_eval = [f for f in fits if "eval_set" in f[0]]
    assert len(with_eval) == 1
    eval_src = with_eval[0][0]["eval_set"]
    assert "X_inner_es" in eval_src and "y_inner_es" in eval_src
    assert "X_val" not in eval_src and "y_val" not in eval_src


def test_xgb_refits_on_the_full_outer_fold_train_after_choosing_best_iteration():
    fits = _calls_in(xgb_tune.evaluate_on_fold, "fit")
    scoring_fit = [f for f in fits if "eval_set" not in f[0]]
    assert len(scoring_fit) == 1
    kwargs, args = scoring_fit[0]
    assert any("X_outer_aug" in a for a in args) and any("y_outer_aug" in a for a in args)

    ctors = _calls_in(xgb_tune.evaluate_on_fold, "XGBClassifier")
    assert len(ctors) == 2
    es_ctor = [c for c in ctors if "early_stopping_rounds" in c[0]]
    scoring_ctor = [c for c in ctors if "early_stopping_rounds" not in c[0]]
    assert len(es_ctor) == 1 and len(scoring_ctor) == 1
    assert scoring_ctor[0][0]["n_estimators"] == "effective_n_estimators"
    # the early-stopping model is explicitly discarded, never scored
    assert "del es_model" in inspect.getsource(xgb_tune.evaluate_on_fold)


# --- deterministic candidate generation ------------------------------------

def test_rf_candidates_are_deterministic_unique_and_36_strong():
    a, b = rf_tune.build_candidates(), rf_tune.build_candidates()
    assert a == b
    assert len(a) == 36 == len(rf_tune.ANCHOR_CONFIGS) + rf_tune.N_RANDOM_CANDIDATES
    keys = [tuple(c[k] for k in rf_tune.SEARCH_KEYS) for c in a]
    assert len(set(keys)) == len(keys)
    assert len({c["candidate_id"] for c in a}) == len(a)
    assert all(c["n_estimators"] == 400 for c in a)


def test_xgb_candidates_are_deterministic_unique_and_40_strong():
    a, b = xgb_tune.build_candidates(), xgb_tune.build_candidates()
    assert a == b
    assert len(a) == 40 == len(xgb_tune.ANCHOR_CONFIGS) + xgb_tune.N_RANDOM_CANDIDATES
    keys = [tuple(c[k] for k in xgb_tune.SEARCH_KEYS) for c in a]
    assert len(set(keys)) == len(keys)
    assert len({c["candidate_id"] for c in a}) == len(a)


def test_both_searches_include_the_previous_series_structure_as_an_anchor():
    assert any(c["candidate_id"] == "anchor_series_rf_v2_structure" for c in rf_tune.build_candidates())
    assert any(c["candidate_id"] == "anchor_series_xgb_v2_structure" for c in xgb_tune.build_candidates())


def test_selection_epsilon_is_exactly_0_002():
    assert mmc.LOG_LOSS_EQUIVALENCE_EPSILON == 0.002
    assert rf_tune.LOG_LOSS_EQUIVALENCE_EPSILON == 0.002
    assert xgb_tune.LOG_LOSS_EQUIVALENCE_EPSILON == 0.002


def test_log_loss_equivalence_epsilon_actually_groups_candidates():
    agg = pd.DataFrame([
        {"candidate_id": "a", "val_log_loss_mean": 0.6600, "val_log_loss_std": 0.01, "val_roc_auc_mean": 0.60,
         "val_brier_mean": 0.23, "val_accuracy_mean": 0.58},
        {"candidate_id": "b", "val_log_loss_mean": 0.6615, "val_log_loss_std": 0.01, "val_roc_auc_mean": 0.65,
         "val_brier_mean": 0.23, "val_accuracy_mean": 0.58},
        {"candidate_id": "c", "val_log_loss_mean": 0.6700, "val_log_loss_std": 0.01, "val_roc_auc_mean": 0.99,
         "val_brier_mean": 0.20, "val_accuracy_mean": 0.70},
    ])
    params = {c: {"max_depth": 5, "min_samples_leaf": 10, "min_samples_split": 10,
                   "max_features": "sqrt", "candidate_id": c} for c in "abc"}
    winner, stage = rf_tune.select_winner(agg, params)
    # b is within 0.002 of a and has the better AUC; c is outside the epsilon
    # despite a far better AUC, so it must not win
    assert winner == "b" and "secondary" in stage


def test_xgb_final_n_estimators_is_the_median_of_best_iteration_plus_one():
    assert xgb_tune.derive_final_n_estimators([9, 19, 29, 39]) == int(round(np.median([10, 20, 30, 40])))
    assert xgb_tune.derive_final_n_estimators([68, 361, 74, 172]) == 124


# --- ensemble --------------------------------------------------------------

def test_ensemble_weights_are_exactly_the_eleven_predefined_values():
    assert ENSEMBLE_WEIGHTS == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_ensemble_weight_selection_minimizes_log_loss_over_those_weights():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=500).astype(float)
    p_rf = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.05, 500), 0.01, 0.99)
    p_xgb = np.full(500, 0.5)                     # deliberately uninformative
    w, stage, table = select_ensemble_weight(y, p_rf, p_xgb)
    assert list(table["weight_rf"]) == ENSEMBLE_WEIGHTS
    assert table["log_loss"].idxmin() == table.index[table["weight_rf"] == w][0]
    assert w == 1.0                                # all weight to the informative model


def test_frozen_ensemble_weight_reproduces_from_the_saved_train_only_oof():
    cfg = json.loads((ROOT / "data" / "modeling" / "map_ensemble_v1_config.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(ROOT / "data" / "modeling" / "map_selected_models_oof_v1.parquet",
                           engine="fastparquet")
    w, _, _ = select_ensemble_weight(oof["y_true"].to_numpy(dtype=float),
                                      oof["p_rf"].to_numpy(), oof["p_xgb"].to_numpy())
    assert w == cfg["weight_rf"]
    assert cfg["main_validation_used_in_selection"] is False


def test_ensemble_was_selected_from_train_only_oof():
    """The OOF artifact it was selected from contains no validation/test map."""
    split = pd.read_csv(ROOT / "data" / "modeling" / "map_split_v1.csv")
    train_ids = set(split.loc[split["split"] == "train", "match_id"])
    oof = pd.read_parquet(ROOT / "data" / "modeling" / "map_selected_models_oof_v1.parquet",
                           engine="fastparquet")
    assert set(oof["match_id"]) <= train_ids


# --- development-time inaccessibility --------------------------------------

def _reads_any(source, needles):
    """AST search for pandas read_csv/read_parquet calls whose arguments mention
    any of `needles`. Comments and docstrings are invisible to this, which is
    the point - only real reads count."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in {"read_csv", "read_parquet"}:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if any(n in ast.unparse(arg) for n in needles):
                        return True
    return False


def test_test_partition_is_inaccessible_during_tuning():
    """No tuning/OOF/baseline script reads the split manifest, so neither TEST
    nor the main validation partition can be reached from them, and each one
    explicitly asks the shared loader not to open it either."""
    for name in ["map_random_forest_tuning_v1.py", "map_xgboost_tuning_v1.py",
                  "map_selected_oof_v1.py", "map_baselines_v1.py"]:
        src = (MODELS_SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert not _reads_any(src, ["map_split_v1", "SPLIT_PATH"]), name
        assert "verify_against_split=False" in src, name


def test_cologne_is_inaccessible_from_the_map_feature_artifact():
    df = pd.read_parquet(ROOT / "data" / "features" / "map_features_v2_rich.parquet", engine="fastparquet")
    em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
    cologne = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
    assert set(df["match_id"]).isdisjoint(cologne)


# --- feature families ------------------------------------------------------

def test_feature_families_partition_the_transformed_matrix_exactly(roles):
    names = pcm.transformed_feature_names(roles)
    fam = feature_family_map(roles, names)
    assert set(fam["by_feature"]) == set(names)
    assert sum(len(v) for v in fam["groups"].values()) == len(names)
    assert set(fam["groups"]) == set(FAMILY_LABELS)
    assert len(fam["groups"]["K"]) == 14      # the map-specific family
    assert len(fam["groups"]["L"]) == 14      # 9 map + 2 bestOf + 3 tier dummies


# --- saved artifacts -------------------------------------------------------

def test_saved_preprocessing_vocabularies_include_both_unknown_categories():
    for name in ["map_random_forest_preprocessing_v1.json", "map_xgboost_preprocessing_v1.json"]:
        p = json.loads((ROOT / "data" / "modeling" / name).read_text(encoding="utf-8"))
        cats = p["categorical"]
        assert pcm.UNKNOWN_MAP_CATEGORY in cats["map_name"]["categories"]
        assert pcm.UNKNOWN_MAP_CATEGORY in cats["map_name"]["dummies"]
        assert pcm.UNKNOWN_TIER_CATEGORY in cats["tier"]["categories"]
        assert pcm.UNKNOWN_TIER_CATEGORY in cats["tier"]["dummies"]
        assert len(p["transformed_feature_names"]) == 106


def test_saved_models_reload_with_identical_predictions(roles):
    import joblib
    from xgboost import XGBClassifier

    df = pd.read_parquet(ROOT / "data" / "features" / "map_features_v2_rich.parquet", engine="fastparquet")
    sample = df.head(200)

    rf_params = prep_rf.load_preprocessing(ROOT / "data" / "modeling" / "map_random_forest_preprocessing_v1.json")
    X_rf, _ = prep_rf.transform(sample, rf_params, roles)
    rf = joblib.load(ROOT / "models" / "map" / "map_random_forest_v1.joblib")
    p1 = rf.predict_proba(X_rf)[:, 1]
    p2 = joblib.load(ROOT / "models" / "map" / "map_random_forest_v1.joblib").predict_proba(X_rf)[:, 1]
    assert np.max(np.abs(p1 - p2)) < 1e-9
    assert np.isfinite(p1).all() and ((p1 >= 0) & (p1 <= 1)).all()

    xgb_params = prep_xgb.load_preprocessing(ROOT / "data" / "modeling" / "map_xgboost_preprocessing_v1.json")
    X_xgb, _ = prep_xgb.transform(sample, xgb_params, roles)
    m1 = XGBClassifier(); m1.load_model(str(ROOT / "models" / "map" / "map_xgboost_v1.json"))
    m2 = XGBClassifier(); m2.load_model(str(ROOT / "models" / "map" / "map_xgboost_v1.json"))
    q1, q2 = m1.predict_proba(X_xgb)[:, 1], m2.predict_proba(X_xgb)[:, 1]
    assert np.max(np.abs(q1 - q2)) < 1e-9
    assert np.isfinite(q1).all() and ((q1 >= 0) & (q1 <= 1)).all()


def test_selected_configs_reproduce_from_their_tuning_tables():
    for sel_name, table_name, keys in [
        ("map_random_forest_v1_selected_config.json", "map_random_forest_tuning_v1.csv", rf_tune.SEARCH_KEYS),
        ("map_xgboost_v1_selected_config.json", "map_xgboost_tuning_v1.csv", xgb_tune.SEARCH_KEYS),
    ]:
        sel = json.loads((ROOT / "data" / "modeling" / sel_name).read_text(encoding="utf-8"))
        table = pd.read_csv(ROOT / "reports" / "tables" / table_name)
        row = table[(table["row_type"] == "aggregate") &
                     (table["candidate_id"] == sel["selected_candidate_id"])]
        assert len(row) == 1
        row = row.iloc[0]
        for k in keys:
            saved, tabled = sel["params"][k], row[k]
            # the CSV round-trip widens integer columns to float wherever some
            # other candidate wrote None (e.g. max_depth), so compare numerically
            # when both sides are numeric and textually otherwise
            try:
                assert float(saved) == float(tabled), (sel_name, k, tabled, saved)
            except (TypeError, ValueError):
                assert str(saved) == str(tabled), (sel_name, k, tabled, saved)
        assert abs(float(row["val_log_loss_mean"]) - sel["cv_mean_log_loss"]) < 1e-9


def test_xgb_final_n_estimators_matches_the_saved_fold_best_iterations():
    sel = json.loads((ROOT / "data" / "modeling" / "map_xgboost_v1_selected_config.json")
                      .read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / "reports" / "tables" / "map_xgboost_tuning_v1.csv")
    folds = table[(table["row_type"] == "fold") &
                   (table["candidate_id"] == sel["selected_candidate_id"])].sort_values("fold")
    best_iters = folds["best_iteration"].astype(int).tolist()
    assert best_iters == sel["best_iterations_by_fold"]
    assert xgb_tune.derive_final_n_estimators(best_iters) == sel["final_n_estimators"]
