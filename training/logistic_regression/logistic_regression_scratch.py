"""
Logistic Regression - implemented from scratch (Phase 4A, Model 1).

This file deliberately contains NO import of sklearn (or any other ML
estimator library) anywhere. It is checked by
tests/models/test_logistic_regression_scratch.py (test G) and
validation/validate_phase4a.py to guarantee this stays true.
"""

import numpy as np
import math

# ============================================================
# LAB-ADAPTED LOGISTIC REGRESSION CORE
# Adapted from:
# reference/Lab_2_Logistic_Regression_answer.ipynb
#
# Original lab functions:
# sigmoid            (cell 15, UNQ_C1)
# compute_cost       (cell 23, UNQ_C2)
# compute_gradient   (cell 32, UNQ_C3)
# gradient_descent   (cell 40)
# predict            (cell 48, UNQ_C4)
# compute_cost_reg   (cell 71, UNQ_C5)
# compute_gradient_reg (cell 78, UNQ_C6)
#
# The mathematics below (sigmoid, binary cross-entropy cost, gradient,
# batch gradient descent update rule, L2 regularization terms) are
# preserved exactly as taught in the lab. Additions/adaptations required to
# run this on the larger CS2 dataset are individually marked
# [PROJECT ADDITION] / [PROJECT ADAPTATION: ...] at the point they occur -
# nothing about the lab's math is silently changed.
# ============================================================


# [LAB-ADAPTED: sigmoid / UNQ_C1]
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g = 1 / (1 + np.exp(-z))
    return g


# [PROJECT ADAPTATION: numerical stability]
# The lab's small teaching datasets never push sigmoid() all the way to
# exactly 0.0 or 1.0, so compute_cost's log(F_wb)/log(1-F_wb) never sees
# log(0). On the larger, more separable CS2 feature set this can happen
# after enough gradient descent steps. _EPS clips the probability into
# (0, 1) before taking the log. This does NOT change the logistic
# regression objective (binary cross-entropy) - it only prevents a
# -inf/NaN numerical failure at the floating-point boundary; the true
# BCE value at F_wb=0 or 1 is already +inf/undefined for a misclassified
# point, so clipping merely caps that value instead of crashing.
_EPS = 1e-12


def _clip_prob(p):
    return np.clip(p, _EPS, 1 - _EPS)


# [LAB-ADAPTED: compute_cost / UNQ_C2]
def compute_cost(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder (matches the lab - regularization is
               added separately in compute_cost_reg)
    Returns:
      total_cost: (scalar)         cost
    """
    m, n = X.shape

    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)
    F_wb = _clip_prob(F_wb)  # [PROJECT ADAPTATION: numerical stability]
    total_cost = (-np.dot(y, np.log(F_wb)) - np.dot(1 - y, np.log(1 - F_wb))) / m

    return total_cost


# [LAB-ADAPTED: compute_gradient / UNQ_C3]
def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data
      y : (array_like Shape (m,)) actual value
      w : (array_like Shape (n,)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder (matches the lab).
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. b.
      dj_dw: (array_like Shape (n,)) The gradient of the cost w.r.t. w.
    """
    m, n = X.shape

    F_wb = sigmoid(np.dot(X, w) + b)
    Err = F_wb - y
    dj_dw = np.dot(Err, X) / m
    dj_db = np.sum(Err) / m

    return dj_db, dj_dw


# [PROJECT ADDITION: divergence detection]
class GradientDescentDivergenceError(RuntimeError):
    """Raised by gradient_descent when the cost becomes non-finite or
    increases catastrophically, so a broken run STOPs loudly instead of
    silently producing a fake-successful model."""


# [LAB-ADAPTED: gradient_descent]
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_,
                      verbose=True, divergence_factor=10.0):
    """
    Performs batch gradient descent to learn w and b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha.

    Args:
      X :    (array_like Shape (m, n))
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      gradient_function:               function to compute gradient
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant
      verbose : (bool)                [PROJECT ADDITION] print progress like the lab did (default True)
      divergence_factor : (float)     [PROJECT ADDITION] see GradientDescentDivergenceError below

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model
      b : (scalar)                Updated value of parameter of the model
      J_history : list of cost at each recorded iteration
      w_history : list of w snapshots at reporting intervals
    """
    m = len(X)

    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:  # prevent resource exhaustion, as in the lab
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

            # [PROJECT ADDITION: divergence detection] - not in the lab.
            # The lab's tiny, well-behaved datasets never diverge under the
            # settings it demonstrates; the CS2 run must not silently
            # continue (and report success) if it does.
            if not np.isfinite(cost):
                raise GradientDescentDivergenceError(
                    f"cost became non-finite ({cost}) at iteration {i} - STOPPING. "
                    "This indicates alpha is too large or the inputs are not properly scaled."
                )
            if i > 0 and cost > divergence_factor * J_history[0]:
                raise GradientDescentDivergenceError(
                    f"cost increased catastrophically to {cost} (> {divergence_factor}x the initial "
                    f"cost {J_history[0]}) at iteration {i} - STOPPING."
                )

        if verbose and (i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1)):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.6f}   ")

    return w_in, b_in, J_history, w_history


# [LAB-ADAPTED: predict / UNQ_C4, generalized - see predict() below]
# The lab's predict() hardcodes a 0.5 threshold with no parameter:
#     F_wb = sigmoid(np.dot(X,w) + b); p = (F_wb >= 0.5)
# predict_proba/predict below reproduce this exactly at the default
# threshold; predict_proba is a [PROJECT ADDITION] (the lab never
# separates the probability step from the thresholding step).

# [PROJECT ADDITION: predict_proba]
def predict_proba(X, w, b):
    """Predicted P(y=1) for each row of X, using the model's own sigmoid.
    Not present as a separate function in the lab - factored out of
    predict() so training/evaluation code can request probabilities
    (needed for ROC-AUC, log loss, Brier score, calibration) without
    duplicating the sigmoid(X@w+b) computation."""
    return sigmoid(np.dot(X, w) + b)


# [LAB-ADAPTED: predict / UNQ_C4]
# [PROJECT ADDITION: configurable `threshold` parameter, default 0.5 reproduces
#  the lab's hardcoded F_wb >= 0.5 behavior exactly]
def predict(X, w, b, threshold=0.5):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w, b.

    Args:
      X : (ndarray Shape (m, n))
      w : (array_like Shape (n,))      Parameters of the model
      b : (scalar, float)              Parameter of the model
      threshold : (float)              [PROJECT ADDITION] decision threshold;
                  defaults to 0.5, identical to the lab's hardcoded UNQ_C4 behavior.

    Returns:
      p: (ndarray (m,)) boolean predictions for X
    """
    F_wb = predict_proba(X, w, b)
    p = F_wb >= threshold
    return p


# [LAB-ADAPTED: compute_cost_reg / UNQ_C5]
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples, with L2 regularization.
    Args:
      X : (array_like Shape (m,n)) data
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : (scalar) Value of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar) cost
    """
    m, n = X.shape

    cost_without_reg = compute_cost(X, y, w, b)

    reg_cost = np.sum(np.square(w))

    # bias b is NOT regularized, matching the lab
    total_cost = cost_without_reg + (lambda_ / (2 * m)) * reg_cost

    return total_cost


# [LAB-ADAPTED: compute_gradient_reg / UNQ_C6]
def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for logistic regression with L2 regularization.

    Args:
      X : (ndarray Shape (m,n))   data
      y : (ndarray Shape (m,))    actual value
      w : (ndarray Shape (n,))    values of parameters of the model
      b : (scalar)                value of parameter of the model
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. b (unregularized, matches the lab).
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. w.
    """
    m, n = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    dj_dw = dj_dw + (lambda_ / m) * w
    # dj_db intentionally left unregularized - matches the lab exactly.

    return dj_db, dj_dw
