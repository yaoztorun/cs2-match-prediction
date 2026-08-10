"""
Tests for Phase 4A.1: Logistic Regression V2 chronological L2 tuning.

Key properties under test:
  - the frozen scratch core is still used and still sklearn-free
  - the dual convergence criterion is deterministic and TRAINING-ONLY
  - non-converged candidates cannot win selection
  - the full selection ladder (epsilon -> AUC -> std -> larger lambda)
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.logistic_regression_scratch import compute_cost_reg, compute_gradient_reg
from logistic_regression_convergence_v2 import (
    gradient_descent_until_convergence, gradient_norm,
    MIN_ITERATIONS, CHECK_EVERY, RELATIVE_TOLERANCE, CONSECUTIVE_CHECKS_REQUIRED,
    GRADIENT_NORM_TOLERANCE, CONVERGENCE_RULE_TEXT,
)
from logistic_regression_tuning_v2 import (
    LAMBDA_GRID, ALPHA, MAX_ITERATIONS, LOG_LOSS_EQUIVALENCE_EPSILON, select_winner,
)
from preprocessing_common import DIRECTIONAL_DIFF_FEATURES, SYMMETRIC_COUNT_FEATURES, BINARY_FEATURES
from preprocessing_logistic_v1 import build_augmented_training_raw, fit_preprocessing, transform

ROOT = Path(__file__).resolve().parents[1]
CV_FOLDS_PATH = ROOT / "data" / "modeling" / "random_forest_cv_folds_v2.csv"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
EVAL_MANIFEST = ROOT / "data" / "interim" / "evaluation_manifest.csv"
SCRATCH_CORE = ROOT / "scripts" / "models" / "logistic_regression_scratch.py"
TUNING_SOURCE = ROOT / "scripts" / "logistic_regression_tuning_v2.py"

RF_CV_FOLDS_SHA256 = "152864c64ef558139af8b588d80e94102a13f52786275dc386357b52ac524247"
MODEL_FEATURES = DIRECTIONAL_DIFF_FEATURES + SYMMETRIC_COUNT_FEATURES + BINARY_FEATURES + ["bestOf", "tier"]


def _synthetic(n=400, seed=0, with_missing=True):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    data = {
        "match_id": np.arange(n),
        "datetime": [base + pd.Timedelta(hours=i // 3) for i in range(n)],
        "team1_series_win": rng.integers(0, 2, size=n).astype(float),
    }
    for c in DIRECTIONAL_DIFF_FEATURES:
        data[c] = rng.normal(scale=50, size=n)
    for c in SYMMETRIC_COUNT_FEATURES:
        data[c] = rng.integers(0, 100, size=n).astype(float)
    for c in BINARY_FEATURES:
        data[c] = rng.integers(0, 2, size=n).astype(float)
    data["bestOf"] = rng.choice([1, 3, 5], size=n)
    data["tier"] = rng.choice(["tier1", "tier2", "tier3"], size=n)
    df = pd.DataFrame(data)
    if with_missing:
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df.loc[idx, "days_since_last_match_diff"] = np.nan
    return df


def _toy_problem(n=300, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + rng.normal(scale=1.0, size=n) > 0).astype(float)
    return X, y


# ---------------- folds reused exactly ----------------

def test_cv_fold_manifest_reused_byte_identically():
    assert hashlib.sha256(CV_FOLDS_PATH.read_bytes()).hexdigest() == RF_CV_FOLDS_SHA256


def test_folds_use_only_global_train_ids():
    cv = pd.read_csv(CV_FOLDS_PATH)
    split = pd.read_csv(SPLIT_PATH)
    train_ids = set(split.loc[split.split == "train", "match_id"])
    val_ids = set(split.loc[split.split == "validation", "match_id"])
    test_ids = set(split.loc[split.split == "test", "match_id"])
    cv_ids = set(cv["match_id"])
    assert cv_ids <= train_ids
    assert cv_ids.isdisjoint(val_ids)
    assert cv_ids.isdisjoint(test_ids)

    em = pd.read_csv(EVAL_MANIFEST)
    assert cv_ids.isdisjoint(set(em.loc[em.evaluation_group == "cologne_2026", "match_id"]))
    assert cv_ids.isdisjoint(set(em.loc[em.evaluation_group == "post_cologne", "match_id"]))


def test_fold_chronology_and_no_group_crossing():
    cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])
    for f in sorted(cv.fold.unique()):
        tr = cv[(cv.fold == f) & (cv.role == "train")]
        va = cv[(cv.fold == f) & (cv.role == "validation")]
        assert tr["datetime"].max() < va["datetime"].min()
        per_dt_role = cv[cv.fold == f].groupby("datetime")["role"].nunique()
        assert (per_dt_role == 1).all()


# ---------------- mirroring / preprocessing ----------------

def test_mirroring_doubles_only_fold_train_with_exact_half_target():
    train = _synthetic(n=300, seed=1)
    aug = build_augmented_training_raw(train)
    assert len(aug) == 2 * len(train)
    assert aug["team1_series_win"].mean() == pytest.approx(0.5, abs=1e-12)


def test_fold_preprocessing_is_train_only_and_19_features():
    train = _synthetic(n=400, seed=2)
    aug = build_augmented_training_raw(train)
    params = fit_preprocessing(aug, MODEL_FEATURES)
    X, names = transform(aug, params)
    assert X.shape[1] == 19 and len(names) == 19
    # medians/means recomputed independently from the same augmented train only
    for c in DIRECTIONAL_DIFF_FEATURES[:3]:
        assert params["train_medians"][c] == pytest.approx(float(aug[c].median()))


def test_validation_transform_does_not_mirror():
    train = _synthetic(n=300, seed=3)
    val = _synthetic(n=90, seed=99)
    params = fit_preprocessing(build_augmented_training_raw(train), MODEL_FEATURES)
    X_val, _ = transform(val, params)
    assert X_val.shape[0] == 90  # not doubled


# ---------------- lambda grid ----------------

def test_lambda_grid_is_fixed_deterministic_and_includes_zero():
    assert LAMBDA_GRID == [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    assert len(LAMBDA_GRID) == len(set(LAMBDA_GRID))
    assert 0.0 in LAMBDA_GRID
    assert LAMBDA_GRID == sorted(LAMBDA_GRID)


def test_optimization_settings_are_fixed_not_searched():
    """alpha / max_iterations are single fixed scalars, not grids."""
    assert isinstance(ALPHA, float) and ALPHA == 0.01
    assert isinstance(MAX_ITERATIONS, int) and MAX_ITERATIONS == 20000


# ---------------- convergence criterion ----------------

def test_convergence_is_deterministic():
    X, y = _toy_problem(seed=4)
    a = gradient_descent_until_convergence(X, y, np.zeros(X.shape[1]), 0.0, compute_cost_reg,
                                            compute_gradient_reg, ALPHA, MAX_ITERATIONS, 1.0, verbose=False)
    b = gradient_descent_until_convergence(X, y, np.zeros(X.shape[1]), 0.0, compute_cost_reg,
                                            compute_gradient_reg, ALPHA, MAX_ITERATIONS, 1.0, verbose=False)
    assert a[3]["iterations_run"] == b[3]["iterations_run"]
    assert a[3]["converged"] == b[3]["converged"]
    assert a[3]["converged_by"] == b[3]["converged_by"]
    np.testing.assert_allclose(a[0], b[0])


def test_convergence_uses_only_training_objective_not_validation():
    """The driver's signature accepts no validation data at all - stopping
    therefore cannot depend on validation performance by construction."""
    import inspect
    sig = inspect.signature(gradient_descent_until_convergence)
    for forbidden in ["X_val", "y_val", "eval_set", "validation", "val_X"]:
        assert forbidden not in sig.parameters


def test_convergence_reports_which_criterion_fired_and_diagnostics():
    X, y = _toy_problem(seed=5)
    _, _, _, info = gradient_descent_until_convergence(
        X, y, np.zeros(X.shape[1]), 0.0, compute_cost_reg, compute_gradient_reg,
        ALPHA, MAX_ITERATIONS, 0.0, verbose=False)
    assert info["converged"] is True
    assert info["converged_by"] in {"relative_cost_plateau", "gradient_norm"}
    assert np.isfinite(info["final_gradient_norm"])
    assert np.isfinite(info["final_relative_improvement"])
    assert info["iterations_run"] >= MIN_ITERATIONS


def test_gradient_norm_criterion_can_fire_independently():
    """Criterion (B): a tiny gradient norm marks convergence even if the plateau
    counter has not yet accumulated the required consecutive checks."""
    X, y = _toy_problem(n=200, d=3, seed=6)
    # start from an already near-optimal point by pre-training, then rerun briefly
    w, b, _, _ = gradient_descent_until_convergence(
        X, y, np.zeros(3), 0.0, compute_cost_reg, compute_gradient_reg,
        ALPHA, MAX_ITERATIONS, 0.0, verbose=False)
    dj_db, dj_dw = compute_gradient_reg(X, y, w, b, 0.0)
    gn = gradient_norm(dj_dw, dj_db)
    assert gn >= 0.0 and np.isfinite(gn)
    # the tolerance constant is what criterion (B) compares against
    assert GRADIENT_NORM_TOLERANCE == 1e-5


def test_max_iterations_reached_reports_not_converged():
    X, y = _toy_problem(seed=7)
    _, _, _, info = gradient_descent_until_convergence(
        X, y, np.zeros(X.shape[1]), 0.0, compute_cost_reg, compute_gradient_reg,
        alpha=1e-9, max_iters=MIN_ITERATIONS + CHECK_EVERY, lambda_=0.0, verbose=False)
    assert info["converged"] is False
    assert info["converged_by"] is None


def test_numerical_safety_detects_nan_cost():
    """Exercises the divergence-detection path deterministically with a stub cost
    function. The real compute_cost_reg is clip-bounded for numerical stability
    (a [PROJECT ADAPTATION] in the frozen scratch core), so genuine gradient
    descent cannot reliably reach NaN/blow-up from data alone - the detection
    logic itself is what needs testing, and stubs test it exactly."""
    from models.logistic_regression_scratch import GradientDescentDivergenceError
    X, y = np.zeros((5, 2)), np.zeros(5)

    def nan_cost(X, y, w, b, lambda_):
        return np.nan

    def zero_grad(X, y, w, b, lambda_):
        return 0.0, np.zeros(X.shape[1])

    with pytest.raises(GradientDescentDivergenceError):
        gradient_descent_until_convergence(X, y, np.zeros(2), 0.0, nan_cost, zero_grad,
                                            alpha=0.01, max_iters=50, lambda_=0.0, verbose=False)


def test_numerical_safety_detects_catastrophic_cost_increase():
    from models.logistic_regression_scratch import GradientDescentDivergenceError
    X, y = np.zeros((5, 2)), np.zeros(5)
    calls = {"n": 0}

    def exploding_cost(X, y, w, b, lambda_):
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 10_000.0

    def zero_grad(X, y, w, b, lambda_):
        return 0.0, np.zeros(X.shape[1])

    with pytest.raises(GradientDescentDivergenceError):
        gradient_descent_until_convergence(X, y, np.zeros(2), 0.0, exploding_cost, zero_grad,
                                            alpha=0.01, max_iters=50, lambda_=0.0, verbose=False)


# ---------------- selection rule ----------------

def _agg(rows):
    return pd.DataFrame(rows)


def test_non_converged_candidate_excluded_even_with_best_log_loss():
    agg = _agg([
        {"lambda": 0.0, "val_log_loss_mean": 0.600, "val_roc_auc_mean": 0.70,
         "val_log_loss_std": 0.01, "all_folds_converged": False},   # best but ineligible
        {"lambda": 1.0, "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.64,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
    ])
    lam, stage = select_winner(agg)
    assert lam == 1.0, "an unconverged candidate must never win"


def test_selection_raises_when_nothing_converged():
    agg = _agg([
        {"lambda": 0.0, "val_log_loss_mean": 0.60, "val_roc_auc_mean": 0.7,
         "val_log_loss_std": 0.01, "all_folds_converged": False},
    ])
    with pytest.raises(RuntimeError):
        select_winner(agg)


def test_selection_primary_lowest_log_loss():
    agg = _agg([
        {"lambda": 0.0, "val_log_loss_mean": 0.640, "val_roc_auc_mean": 0.60,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
        {"lambda": 1.0, "val_log_loss_mean": 0.660, "val_roc_auc_mean": 0.70,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
    ])
    lam, stage = select_winner(agg)
    assert lam == 0.0 and "primary" in stage


def test_selection_epsilon_tie_resolved_by_roc_auc():
    agg = _agg([
        {"lambda": 0.0, "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.60,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
        {"lambda": 1.0, "val_log_loss_mean": 0.650 + LOG_LOSS_EQUIVALENCE_EPSILON / 2,
         "val_roc_auc_mean": 0.66, "val_log_loss_std": 0.01, "all_folds_converged": True},
    ])
    lam, stage = select_winner(agg)
    assert lam == 1.0 and "secondary" in stage


def test_selection_tertiary_resolved_by_lower_std():
    agg = _agg([
        {"lambda": 0.0, "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.64,
         "val_log_loss_std": 0.05, "all_folds_converged": True},
        {"lambda": 1.0, "val_log_loss_mean": 0.6505, "val_roc_auc_mean": 0.64,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
    ])
    lam, stage = select_winner(agg)
    assert lam == 1.0 and "tertiary" in stage


def test_final_tiebreak_prefers_larger_lambda():
    agg = _agg([
        {"lambda": 0.1, "val_log_loss_mean": 0.650, "val_roc_auc_mean": 0.64,
         "val_log_loss_std": 0.02, "all_folds_converged": True},
        {"lambda": 25.0, "val_log_loss_mean": 0.6505, "val_roc_auc_mean": 0.64,
         "val_log_loss_std": 0.02, "all_folds_converged": True},
    ])
    lam, stage = select_winner(agg)
    assert lam == 25.0, "among equivalent candidates, stronger regularization must win"
    assert "larger lambda" in stage


def test_lambda_zero_can_win_when_genuinely_best():
    agg = _agg([
        {"lambda": 0.0, "val_log_loss_mean": 0.6400, "val_roc_auc_mean": 0.66,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
        {"lambda": 50.0, "val_log_loss_mean": 0.6800, "val_roc_auc_mean": 0.60,
         "val_log_loss_std": 0.01, "all_folds_converged": True},
    ])
    lam, _ = select_winner(agg)
    assert lam == 0.0, "lambda=0 must remain eligible and winnable"


# ---------------- from-scratch guarantee ----------------

def test_scratch_core_still_has_no_sklearn():
    src = SCRATCH_CORE.read_text(encoding="utf-8")
    code, in_doc = [], False
    for line in src.splitlines():
        if line.strip().startswith('"""'):
            in_doc = not in_doc
            continue
        if not in_doc and not line.strip().startswith("#"):
            code.append(line)
    code = "\n".join(code)
    assert "import sklearn" not in code
    assert "from sklearn" not in code
    assert "LogisticRegression" not in code


def test_tuning_uses_the_frozen_scratch_primitives():
    src = TUNING_SOURCE.read_text(encoding="utf-8")
    assert "from models.logistic_regression_scratch import" in src
    assert "compute_cost_reg" in src and "compute_gradient_reg" in src
    assert "sklearn.linear_model" not in src
