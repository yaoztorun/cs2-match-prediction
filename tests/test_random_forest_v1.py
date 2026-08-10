"""
Tests for the Random Forest V1 model wrapper and orchestration-level
guarantees. Covers spec letters A, B, C, H, I, K from Phase 4B section 17.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.random_forest_v1 import build_model, save_model, load_model, RF_CONFIG
from preprocessing_common import DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES
from preprocessing_random_forest_v1 import build_augmented_training_raw

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"

EXPECTED_TRAIN_N = 6619
EXPECTED_VAL_N = 1419
EXPECTED_TEST_N = 1418


def _synthetic_train_df(n=100, seed=0):
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


def _synthetic_X_y(n=100, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (rng.uniform(size=n) > 0.5).astype(int)
    return X, y


# --- A: existing split manifest is reused exactly ---

def test_real_split_manifest_has_exact_phase4a_counts():
    assert SPLIT_PATH.exists(), "data/modeling/series_split_v1.csv must already exist from Phase 4A - not regenerated"
    split = pd.read_csv(SPLIT_PATH)
    counts = split["split"].value_counts()
    assert counts["train"] == EXPECTED_TRAIN_N
    assert counts["validation"] == EXPECTED_VAL_N
    assert counts["test"] == EXPECTED_TEST_N


# --- B: training augmentation doubles only TRAIN ---

def test_augmentation_doubles_only_the_given_train_slice():
    train_like = _synthetic_train_df(n=150, seed=1)
    augmented = build_augmented_training_raw(train_like)
    assert len(augmented) == 2 * len(train_like)
    # build_augmented_training_raw has no concept of "validation"/"test" at all -
    # it only ever operates on whatever single dataframe it's given, and the
    # RF orchestration script only ever calls it on train_raw (by construction,
    # verified by inspection of scripts/train_random_forest_v1.py).


# --- C: augmented target is exactly 50/50 ---

def test_augmented_target_is_exactly_50_50():
    train_like = _synthetic_train_df(n=150, seed=2)
    augmented = build_augmented_training_raw(train_like)
    assert augmented["team1_series_win"].mean() == pytest.approx(0.5)


# --- H: reloaded model produces identical probabilities ---

def test_reloaded_model_reproduces_identical_probabilities(tmp_path):
    X, y = _synthetic_X_y(n=200, n_features=6, seed=3)
    model = build_model()
    model.fit(X, y)
    proba_before = model.predict_proba(X)

    path = tmp_path / "rf_test.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    proba_after = reloaded.predict_proba(X)

    # Tolerance, not exact equality: RF_CONFIG uses n_jobs=-1, so
    # predict_proba averages 300 trees' outputs across threads, and
    # floating-point addition is not associative under thread-scheduling
    # non-determinism - this produces up to ~1 ULP of difference even
    # calling predict_proba twice on the SAME model object, not just across
    # a save/reload. 1e-9 is tight enough to catch a real reload bug while
    # tolerating this expected, harmless floating-point noise.
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-9)


# --- I: fixed random_state gives deterministic results ---

def test_fixed_random_state_is_deterministic():
    X, y = _synthetic_X_y(n=300, n_features=8, seed=4)
    model_a = build_model()
    model_a.fit(X, y)
    model_b = build_model()
    model_b.fit(X, y)

    assert RF_CONFIG["random_state"] == 42
    # feature_importances_ is a pure structural property of the fitted trees
    # (no cross-thread float summation involved) so it IS exactly reproducible.
    np.testing.assert_array_equal(model_a.feature_importances_, model_b.feature_importances_)
    # predict_proba averages across threads (n_jobs=-1) - same tolerance
    # reasoning as the reload test above.
    np.testing.assert_allclose(model_a.predict_proba(X), model_b.predict_proba(X), atol=1e-9)


# --- K: probabilities are in [0,1] ---

def test_probabilities_bounded_in_unit_interval():
    X, y = _synthetic_X_y(n=200, n_features=6, seed=5)
    model = build_model()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.isfinite(proba).all()
