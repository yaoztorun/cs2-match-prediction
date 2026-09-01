"""
[PROJECT ADDITION - Phase 4A.1. No lab code is reimplemented in this file.]

Convergence-aware batch gradient-descent driver for the scratch Logistic
Regression.

WHY THIS FILE EXISTS INSTEAD OF EDITING THE SCRATCH CORE
--------------------------------------------------------
`training/logistic_regression/logistic_regression_scratch.py` is a FROZEN, lab-adapted
implementation. Its `gradient_descent` always runs exactly `num_iters`
iterations - it has divergence *detection* but no convergence *stopping*.
Phase 4A.1 needs a training-objective convergence criterion, which is a new
capability, not a bug fix, so the frozen file is left completely untouched.

This driver therefore CALLS the frozen lab primitives
(`compute_cost_reg`, `compute_gradient_reg`) and applies the identical
update rule the lab teaches:

    w <- w - alpha * dj_dw
    b <- b - alpha * dj_db

The mathematics remains the from-scratch lab implementation; only the loop
wrapper (stopping rule + diagnostics + progress logging) is new.

DUAL CONVERGENCE CRITERION (both defined BEFORE tuning, both TRAINING-ONLY)
--------------------------------------------------------------------------
A fit counts as converged if EITHER holds:

  (A) SUSTAINED RELATIVE-COST PLATEAU
      Every CHECK_EVERY iterations, compute the relative improvement of the
      training objective over the preceding window:
          rel = (J[i-CHECK_EVERY] - J[i]) / max(|J[i-CHECK_EVERY]|, 1e-12)
      Converged when rel < RELATIVE_TOLERANCE on
      CONSECUTIVE_CHECKS_REQUIRED consecutive checks.

  (B) SMALL REGULARIZED GRADIENT NORM
      ||grad||_2 = sqrt(||dj_dw||^2 + dj_db^2) < GRADIENT_NORM_TOLERANCE,
      using the SAME regularized gradient the optimizer is descending.

Criterion (B) exists so that a fit which has genuinely reached a flat,
small-gradient optimum is not declared "non-converged" purely because the
plateau tolerance in (A) is unnecessarily strict. Neither criterion looks at
validation data in any way - both are computed from the training objective
and its gradient only.

Nothing here consults fold-validation performance: doing so would silently
convert optimization stopping into model selection.
"""

import time

import numpy as np

from training.logistic_regression.logistic_regression_scratch import GradientDescentDivergenceError

# --- convergence constants: fixed BEFORE tuning, never adjusted afterwards ---
MIN_ITERATIONS = 1000
CHECK_EVERY = 100
RELATIVE_TOLERANCE = 1e-7
CONSECUTIVE_CHECKS_REQUIRED = 3
GRADIENT_NORM_TOLERANCE = 1e-5

CONVERGENCE_RULE_TEXT = (
    f"TRAINING-OBJECTIVE ONLY. min_iterations={MIN_ITERATIONS}, checked every {CHECK_EVERY} iterations. "
    f"Converged if EITHER (A) relative training-cost improvement over the preceding {CHECK_EVERY}-iteration "
    f"window is < {RELATIVE_TOLERANCE} on {CONSECUTIVE_CHECKS_REQUIRED} consecutive checks, OR "
    f"(B) the regularized gradient norm sqrt(||dj_dw||^2 + dj_db^2) < {GRADIENT_NORM_TOLERANCE}. "
    "Validation data is never consulted."
)


def gradient_norm(dj_dw, dj_db):
    return float(np.sqrt(np.sum(np.square(dj_dw)) + dj_db ** 2))


def gradient_descent_until_convergence(
    X, y, w_in, b_in, cost_function, gradient_function, alpha, max_iters, lambda_,
    divergence_factor=10.0, progress_label="", progress_every=2000, verbose=True,
):
    """Batch gradient descent with the dual training-only convergence criterion.

    Returns (w, b, J_history, info) where info contains:
        iterations_run, converged, converged_by ('relative_cost_plateau' |
        'gradient_norm' | None), initial_cost, final_cost,
        relative_cost_decrease, final_relative_improvement,
        final_gradient_norm, elapsed_seconds.
    """
    w = np.array(w_in, dtype=float).copy()
    b = float(b_in)

    J_history = []
    consecutive_flat_checks = 0
    converged = False
    converged_by = None
    last_rel = float("nan")
    grad_norm = float("nan")
    t0 = time.time()
    i = 0

    for i in range(max_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(X, y, w, b, lambda_)
        J_history.append(cost)

        # divergence semantics preserved from the frozen scratch core
        if not np.isfinite(cost):
            raise GradientDescentDivergenceError(
                f"cost became non-finite ({cost}) at iteration {i} - STOPPING. "
                f"alpha={alpha}, lambda={lambda_}."
            )
        if i > 0 and cost > divergence_factor * J_history[0]:
            raise GradientDescentDivergenceError(
                f"cost increased catastrophically to {cost} (> {divergence_factor}x the initial cost "
                f"{J_history[0]}) at iteration {i} - STOPPING. alpha={alpha}, lambda={lambda_}."
            )

        if verbose and progress_every and (i + 1) % progress_every == 0:
            print(f"      {progress_label} iter {i+1}/{max_iters} | J={cost:.8f} | "
                  f"{time.time()-t0:.1f}s", flush=True)

        # convergence is only assessed after MIN_ITERATIONS, every CHECK_EVERY steps
        if (i + 1) >= MIN_ITERATIONS and (i + 1) % CHECK_EVERY == 0:
            prev = J_history[-(CHECK_EVERY + 1)]
            last_rel = (prev - cost) / max(abs(prev), 1e-12)
            if last_rel < RELATIVE_TOLERANCE:
                consecutive_flat_checks += 1
            else:
                consecutive_flat_checks = 0

            grad_norm = gradient_norm(dj_dw, dj_db)

            if consecutive_flat_checks >= CONSECUTIVE_CHECKS_REQUIRED:
                converged, converged_by = True, "relative_cost_plateau"
            elif grad_norm < GRADIENT_NORM_TOLERANCE:
                converged, converged_by = True, "gradient_norm"

            if converged:
                break

    iterations_run = i + 1

    # final gradient norm at the returned iterate (recomputed so it is always defined,
    # including when max_iters is reached without a convergence check firing)
    dj_db_f, dj_dw_f = gradient_function(X, y, w, b, lambda_)
    final_grad_norm = gradient_norm(dj_dw_f, dj_db_f)

    initial_cost = float(J_history[0])
    final_cost = float(J_history[-1])
    info = {
        "iterations_run": int(iterations_run),
        "converged": bool(converged),
        "converged_by": converged_by,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "relative_cost_decrease": float((initial_cost - final_cost) / max(abs(initial_cost), 1e-12)),
        "final_relative_improvement": float(last_rel),
        "final_gradient_norm": float(final_grad_norm),
        "elapsed_seconds": float(time.time() - t0),
    }
    return w, b, J_history, info
