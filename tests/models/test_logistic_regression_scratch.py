"""
Validates the scratch Logistic Regression implementation itself
(training/logistic_regression/logistic_regression_scratch.py), independent of the CS2
dataset. Covers spec items A-G for Phase 4A.
"""

from pathlib import Path

import numpy as np
import pytest

from training.logistic_regression.logistic_regression_scratch import (
    sigmoid, compute_cost, compute_gradient, gradient_descent,
    predict_proba, predict, compute_cost_reg, compute_gradient_reg,
    GradientDescentDivergenceError,
)

MODEL_SOURCE_PATH = Path(__file__).resolve().parents[2] / "training" / "logistic_regression" / "logistic_regression_scratch.py"


# --- A. Sigmoid ---

def test_sigmoid_zero_is_one_half():
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_vector_behavior():
    z = np.array([-1, 0, 1, 2])
    g = sigmoid(z)
    expected = np.array([0.26894142, 0.5, 0.73105858, 0.88079708])
    np.testing.assert_allclose(g, expected, atol=1e-8)
    # bounded and monotonic
    assert np.all(g > 0) and np.all(g < 1)
    assert np.all(np.diff(g) > 0)
    # symmetry: sigmoid(-z) == 1 - sigmoid(z)
    np.testing.assert_allclose(sigmoid(-z), 1 - sigmoid(z), atol=1e-12)


# --- B. Cost sanity ---

def _toy_dataset():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0], [0.5, -0.5], [-0.5, 0.5]])
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    return X, y


def test_cost_is_finite_on_small_dataset():
    X, y = _toy_dataset()
    w, b = np.zeros(X.shape[1]), 0.0
    cost = compute_cost(X, y, w, b)
    assert np.isfinite(cost)
    assert cost == pytest.approx(np.log(2), abs=1e-9)  # w=0,b=0 -> p=0.5 for all rows


def test_cost_finite_even_for_saturated_predictions():
    """Numerical-stability clipping must prevent log(0) from producing inf/nan
    when a large weight drives sigmoid all the way to 0 or 1."""
    X, y = _toy_dataset()
    w = np.array([1e6, 1e6])
    b = 0.0
    cost = compute_cost(X, y, w, b)
    assert np.isfinite(cost)


# --- C. Gradient numerical check (finite differences) ---

def test_gradient_matches_finite_difference_approximation():
    rng = np.random.default_rng(42)
    m, n = 20, 4
    X = rng.normal(size=(m, n))
    y = (rng.uniform(size=m) > 0.5).astype(float)
    w = rng.normal(size=n) * 0.1
    b = 0.3

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    eps = 1e-6
    numeric_dj_dw = np.zeros(n)
    for j in range(n):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += eps
        w_minus[j] -= eps
        numeric_dj_dw[j] = (compute_cost(X, y, w_plus, b) - compute_cost(X, y, w_minus, b)) / (2 * eps)

    numeric_dj_db = (compute_cost(X, y, w, b + eps) - compute_cost(X, y, w, b - eps)) / (2 * eps)

    np.testing.assert_allclose(dj_dw, numeric_dj_dw, atol=1e-5, rtol=1e-4)
    assert dj_db == pytest.approx(numeric_dj_db, abs=1e-5)


# --- D. Gradient descent ---

def test_gradient_descent_decreases_cost_and_learns_separable_data():
    rng = np.random.default_rng(1)
    m = 200
    x1 = rng.normal(loc=2, scale=0.5, size=m // 2)
    x2 = rng.normal(loc=-2, scale=0.5, size=m // 2)
    X = np.concatenate([x1, x2]).reshape(-1, 1)
    y = np.concatenate([np.ones(m // 2), np.zeros(m // 2)])

    w0, b0 = np.zeros(1), 0.0
    w, b, J_history, _ = gradient_descent(X, y, w0, b0, compute_cost, compute_gradient,
                                           alpha=0.1, num_iters=500, lambda_=0, verbose=False)

    assert all(np.isfinite(c) for c in J_history)
    assert J_history[-1] < J_history[0]

    preds = predict(X, w, b)
    accuracy = np.mean(preds == y.astype(bool))
    assert accuracy >= 0.95


def test_gradient_descent_raises_on_nan_cost():
    """Directly exercises the divergence-detection path with a stub cost
    function that returns NaN (real compute_cost is clip-bounded and won't
    reliably reach NaN from gradient descent dynamics alone, since the
    numerical-stability clip caps the worst-case per-example cost - the
    detection logic itself is tested deterministically here instead)."""
    X, y = np.zeros((5, 2)), np.zeros(5)

    def nan_cost(X, y, w, b, lambda_):
        return np.nan

    def zero_gradient(X, y, w, b, lambda_):
        return 0.0, np.zeros(X.shape[1])

    with pytest.raises(GradientDescentDivergenceError):
        gradient_descent(X, y, np.zeros(2), 0.0, nan_cost, zero_gradient,
                          alpha=0.01, num_iters=5, lambda_=0, verbose=False)


def test_gradient_descent_raises_on_catastrophic_cost_increase():
    X, y = np.zeros((5, 2)), np.zeros(5)
    calls = {"n": 0}

    def increasing_cost(X, y, w, b, lambda_):
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 1000.0

    def zero_gradient(X, y, w, b, lambda_):
        return 0.0, np.zeros(X.shape[1])

    with pytest.raises(GradientDescentDivergenceError):
        gradient_descent(X, y, np.zeros(2), 0.0, increasing_cost, zero_gradient,
                          alpha=0.01, num_iters=5, lambda_=0, verbose=False)


# --- E. Prediction threshold ---

def test_predict_default_threshold_is_point_five():
    X = np.array([[0.0], [0.0]])
    w = np.array([0.0])
    # b chosen so sigmoid(b) is just above/below 0.5
    p_low = predict(X, w, b=-0.0001, threshold=0.5)
    p_high = predict(X, w, b=0.0001, threshold=0.5)
    assert not p_low[0]
    assert p_high[0]


def test_predict_explicit_threshold_changes_result():
    X = np.array([[0.0]])
    w = np.array([0.0])
    b = 0.0  # predict_proba == 0.5 exactly
    assert predict(X, w, b, threshold=0.5)[0]        # 0.5 >= 0.5 -> True
    assert not predict(X, w, b, threshold=0.9)[0]    # 0.5 >= 0.9 -> False
    assert predict(X, w, b, threshold=0.1)[0]        # 0.5 >= 0.1 -> True


# --- F. Regularization ---

def test_lambda_zero_matches_unregularized():
    X, y = _toy_dataset()
    w = np.array([0.3, -0.2])
    b = 0.1

    cost_plain = compute_cost(X, y, w, b)
    cost_reg0 = compute_cost_reg(X, y, w, b, lambda_=0)
    assert cost_reg0 == pytest.approx(cost_plain)

    dj_db_plain, dj_dw_plain = compute_gradient(X, y, w, b)
    dj_db_reg0, dj_dw_reg0 = compute_gradient_reg(X, y, w, b, lambda_=0)
    assert dj_db_reg0 == pytest.approx(dj_db_plain)
    np.testing.assert_allclose(dj_dw_reg0, dj_dw_plain)


def test_positive_lambda_adds_expected_l2_penalty():
    X, y = _toy_dataset()
    w = np.array([0.3, -0.2])
    b = 0.1
    m = X.shape[0]
    lambda_ = 2.0

    cost_plain = compute_cost(X, y, w, b)
    cost_reg = compute_cost_reg(X, y, w, b, lambda_=lambda_)
    expected_penalty = (lambda_ / (2 * m)) * np.sum(w ** 2)
    assert cost_reg - cost_plain == pytest.approx(expected_penalty)

    dj_db_plain, dj_dw_plain = compute_gradient(X, y, w, b)
    dj_db_reg, dj_dw_reg = compute_gradient_reg(X, y, w, b, lambda_=lambda_)
    np.testing.assert_allclose(dj_dw_reg - dj_dw_plain, (lambda_ / m) * w)
    # bias must NOT be regularized
    assert dj_db_reg == pytest.approx(dj_db_plain)


# --- G. No sklearn LogisticRegression ---

def test_scratch_model_source_has_no_sklearn_import():
    """Checks for actual sklearn usage (import statements / attribute access),
    not the bare word - the module's own docstring legitimately documents
    that it avoids sklearn, which would otherwise trip a naive substring check."""
    source = MODEL_SOURCE_PATH.read_text(encoding="utf-8")
    # drop lines inside triple-quoted docstring blocks (module/function docs
    # legitimately mention "sklearn" to document that it's intentionally absent)
    in_docstring = False
    filtered = []
    for line in source.splitlines():
        if line.strip().startswith('"""'):
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            filtered.append(line)
    code = "\n".join(filtered)
    assert "import sklearn" not in code
    assert "from sklearn" not in code
    assert "sklearn." not in code
