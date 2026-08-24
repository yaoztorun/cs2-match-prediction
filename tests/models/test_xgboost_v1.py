"""
Tests for the XGBoost V1 model wrapper and orchestration-level guarantees
(Phase 4C, spec section 18).

Probability comparisons use tight allclose tolerances rather than bitwise
identity: XGBoost is multithreaded (`n_jobs=-1`) and tree-library floating
point behavior can vary by environment even with `random_state=42`, so
requiring cross-platform bitwise equality would be a flaky (and wrong)
assertion.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.xgboost.xgboost_v1 import build_model, save_model, load_model, XGB_CONFIG
from feature_engineering.preprocessing.preprocessing_common import DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES
from feature_engineering.preprocessing.preprocessing_xgboost_v1 import build_augmented_training_raw

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
MODEL_SOURCE = ROOT / "training" / "xgboost" / "xgboost_v1.py"
TRAIN_SOURCE = ROOT / "training" / "xgboost" / "train_xgboost_v1.py"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418
PROBA_TOL = 1e-9


def _synthetic_train_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(scale=50, size=n) for c in DIRECTIONAL_DIFF_FEATURES}
    for c in SYMMETRIC_COUNT_FEATURES:
        data[c] = rng.integers(0, 100, size=n).astype(float)
    for c in BINARY_FEATURES:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    data["team1_series_win"] = rng.integers(0, 2, size=n).astype(float)
    return pd.DataFrame(data)


def _synthetic_X_y(n=200, n_features=6, seed=0, with_nan=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    if with_nan:
        X[::9, 1] = np.nan
    y = (rng.uniform(size=n) > 0.5).astype(int)
    return X, y


# --- split reuse ---

def test_real_split_manifest_has_exact_phase4_counts():
    assert SPLIT_PATH.exists(), "series_split_v1.csv must already exist - Phase 4C never regenerates it"
    counts = pd.read_csv(SPLIT_PATH)["split"].value_counts()
    assert counts["train"] == EXPECTED_TRAIN_N
    assert counts["validation"] == EXPECTED_VAL_N
    assert counts["test"] == EXPECTED_TEST_N


# --- mirroring: train only, exact 50/50 ---

def test_augmentation_doubles_only_train_with_exact_half_target_mean():
    train = _synthetic_train_df(n=200, seed=1)
    augmented = build_augmented_training_raw(train)
    assert len(augmented) == 2 * len(train)
    assert augmented["team1_series_win"].mean() == pytest.approx(0.5, abs=1e-12)


def test_validation_slice_is_not_mirrored_by_any_transform_path():
    """build_augmented_training_raw is the ONLY mirroring entry point used for
    training, and the orchestration script calls it exclusively on train_raw -
    verified by source inspection here so the guarantee is pinned, not assumed."""
    src = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert "build_augmented_training_raw(train_raw)" in src
    assert "build_augmented_training_raw(val_raw)" not in src
    assert "build_augmented_training_raw(test" not in src


# --- config / determinism ---

def test_xgb_config_random_state_is_42_and_matches_spec():
    assert XGB_CONFIG["random_state"] == 42
    assert XGB_CONFIG["objective"] == "binary:logistic"
    assert XGB_CONFIG["eval_metric"] == "logloss"
    assert XGB_CONFIG["n_estimators"] == 300
    assert XGB_CONFIG["learning_rate"] == 0.05
    assert XGB_CONFIG["max_depth"] == 4
    assert XGB_CONFIG["tree_method"] == "hist"


def test_two_fits_on_identical_data_give_matching_probabilities():
    X, y = _synthetic_X_y(n=300, n_features=8, seed=2)
    a, b = build_model(), build_model()
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1], atol=PROBA_TOL)


# --- native save/load round-trip through the CLASSIFIER interface ---

def test_saved_and_reloaded_model_reproduces_probabilities(tmp_path):
    X, y = _synthetic_X_y(n=250, n_features=6, seed=3)
    model = build_model()
    model.fit(X, y)
    before = model.predict_proba(X)[:, 1]

    path = tmp_path / "xgb_test.json"
    save_model(model, path)
    reloaded = load_model(path)

    # must still be a classifier exposing predict_proba (not a bare Booster)
    assert hasattr(reloaded, "predict_proba")
    after = reloaded.predict_proba(X)[:, 1]
    np.testing.assert_allclose(before, after, atol=PROBA_TOL)


# --- probability bounds ---

def test_probabilities_finite_and_in_unit_interval():
    X, y = _synthetic_X_y(n=200, n_features=5, seed=4)
    model = build_model()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_handles_nan_natively_without_imputation():
    X, y = _synthetic_X_y(n=200, n_features=5, seed=5, with_nan=True)
    assert np.isnan(X).any(), "fixture must contain NaN for this test to be meaningful"
    model = build_model()
    model.fit(X, y)  # must not raise
    proba = model.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()


# --- no early stopping / no eval_set anywhere ---

def test_no_early_stopping_or_eval_set_in_sources():
    for src_path in (MODEL_SOURCE, TRAIN_SOURCE):
        src = src_path.read_text(encoding="utf-8")
        code_lines = []
        in_doc = False
        for line in src.splitlines():
            if line.strip().startswith('"""'):
                in_doc = not in_doc
                continue
            if not in_doc and not line.strip().startswith("#"):
                code_lines.append(line)
        code = "\n".join(code_lines)
        assert "early_stopping_rounds" not in code, f"{src_path.name} must not use early stopping"
        assert "eval_set" not in code, f"{src_path.name} must not pass an eval_set"


def test_exactly_one_build_model_call_in_orchestration():
    """Guards against a hyperparameter sweep sneaking in: the training script
    must construct exactly one model."""
    src = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert src.count("build_model()") == 1
