"""
Phase 4A orchestration: trains the scratch Logistic Regression V1 baseline
and evaluates it on TRAIN and VALIDATION only. The internal TEST partition
(data/modeling/series_split_v1.csv, split=="test") is never loaded for
scoring in this script. Cologne/post-Cologne rows are not part of
series_features_v1.parquet at all (Phase 3 guarantee) and are therefore
structurally unreachable here.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, brier_score_loss, confusion_matrix, roc_curve,
)

from _common import ROOT, REPORTS, INTERIM
from models.logistic_regression_scratch import (
    compute_cost, compute_gradient, gradient_descent, predict_proba, predict,
    GradientDescentDivergenceError,
)
from preprocessing_logistic_v1 import (
    build_augmented_training_raw, fit_preprocessing, transform, save_preprocessing,
    assert_augmented_symmetry, PREPROCESSING_VERSION,
)

CONFIG_PATH = ROOT / "config" / "series_features_v1.yaml"
FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR = REPORTS / "figures"
TABLES_DIR = REPORTS / "tables"

# lab cell 42 of reference/Lab_2_Logistic_Regression_answer.ipynb - the lab's
# OWN unregularized (lambda_=0) example - uses exactly these values:
#   iterations = 10000
#   alpha = 0.001
# 10000 iterations is lab-style. alpha=0.001 happens to match that exact lab
# cell too (verified by inspection, not assumed) - it is not merely a
# "conservative project choice" here, it is the lab's own unregularized
# baseline setting, reused as-is because it is a reasonable starting point
# for standardized inputs.
ALPHA = 0.001
NUM_ITERS = 10000
LAMBDA_ = 0.0
THRESHOLD = 0.5

ENGINE_VERSION = "1.0.0"


def majority_class_accuracy(y):
    p1 = np.mean(y)
    return max(p1, 1 - p1)


def compute_metrics(y_true, y_proba, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "majority_class_accuracy": float(majority_class_accuracy(y_true)),
    }


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_features = cfg["model_features"]
    target_col = cfg["target"]

    df = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    split = pd.read_csv(SPLIT_PATH)
    df = df.merge(split[["match_id", "split"]], on="match_id", how="inner")
    assert len(df) == len(split), "split file and feature parquet row counts diverged"

    train_raw = df[df["split"] == "train"].reset_index(drop=True)
    val_raw = df[df["split"] == "validation"].reset_index(drop=True)
    # TEST IS DELIBERATELY NEVER LOADED HERE. `test_raw` does not exist in this script.

    # ---------------- mirror BEFORE fitting preprocessing (corrected order) ----------------
    augmented_train_raw = build_augmented_training_raw(train_raw)
    assert len(augmented_train_raw) == 2 * len(train_raw)
    mirrored_target_mean = float(augmented_train_raw[target_col].mean())
    assert abs(mirrored_target_mean - 0.5) < 1e-9, \
        f"mirrored augmented training target mean must be exactly 0.5, got {mirrored_target_mean}"
    assert_augmented_symmetry(augmented_train_raw)
    print(f"train (unmirrored): {len(train_raw)} rows | augmented (train+mirror): {len(augmented_train_raw)} rows "
          f"| mirrored target mean: {mirrored_target_mean}")

    # ---------------- fit preprocessing on augmented train ONLY ----------------
    params = fit_preprocessing(augmented_train_raw, model_features)
    save_preprocessing(params, MODELING_DIR / "logistic_preprocessing_v1.json")

    X_train_aug, feature_names = transform(augmented_train_raw, params)
    y_train_aug = augmented_train_raw[target_col].to_numpy(dtype=float)

    X_train_orig, _ = transform(train_raw, params)
    y_train_orig = train_raw[target_col].to_numpy(dtype=float)

    X_val, _ = transform(val_raw, params)
    y_val = val_raw[target_col].to_numpy(dtype=float)

    n_features = X_train_aug.shape[1]
    print(f"transformed feature count: {n_features} -> {feature_names}")

    # ---------------- train (single baseline configuration, no tuning) ----------------
    w0, b0 = np.zeros(n_features), 0.0
    try:
        w, b, J_history, _ = gradient_descent(
            X_train_aug, y_train_aug, w0, b0, compute_cost, compute_gradient,
            alpha=ALPHA, num_iters=NUM_ITERS, lambda_=LAMBDA_, verbose=True,
        )
    except GradientDescentDivergenceError as e:
        print(f"\nCONVERGENCE PROBLEM - STOPPING: {e}")
        raise

    assert np.isfinite(w).all(), "learned weights are not all finite"
    assert np.isfinite(b), "learned bias is not finite"
    assert all(np.isfinite(c) for c in J_history), "cost history contains non-finite values"

    initial_cost, final_cost = J_history[0], J_history[-1]
    abs_decrease = initial_cost - final_cost
    rel_decrease = abs_decrease / initial_cost if initial_cost != 0 else float("nan")
    print(f"initial cost: {initial_cost:.6f}, final cost: {final_cost:.6f}, "
          f"abs decrease: {abs_decrease:.6f}, rel decrease: {100*rel_decrease:.2f}%")

    # ---------------- cost plot ----------------
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(J_history)), J_history, linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("training cost J(w,b)")
    ax.set_title("Logistic Regression V1 - training cost vs. iteration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logistic_regression_cost_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- evaluation: TRAIN (unmirrored) + VALIDATION only ----------------
    proba_train = predict_proba(X_train_orig, w, b)
    pred_train = predict(X_train_orig, w, b, threshold=THRESHOLD)
    train_metrics = compute_metrics(y_train_orig, proba_train, pred_train)

    proba_val = predict_proba(X_val, w, b)
    pred_val = predict(X_val, w, b, threshold=THRESHOLD)
    val_metrics = compute_metrics(y_val, proba_val, pred_val)

    assert (proba_train >= 0).all() and (proba_train <= 1).all()
    assert (proba_val >= 0).all() and (proba_val <= 1).all()

    print("TRAIN metrics:", json.dumps(train_metrics, indent=2))
    print("VALIDATION metrics:", json.dumps(val_metrics, indent=2))

    # ---------------- ROC curve (validation only) ----------------
    fpr, tpr, _ = roc_curve(y_val, proba_val)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"Logistic Regression V1 (AUC={val_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC curve - Logistic Regression V1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logistic_regression_roc_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- calibration diagnostic (validation only, manual binning) ----------------
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(proba_val, bins) - 1, 0, 9)
    cal_rows = []
    for i in range(10):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        cal_rows.append({
            "bin": f"[{bins[i]:.1f},{bins[i+1]:.1f})",
            "n": int(mask.sum()),
            "mean_predicted": float(proba_val[mask].mean()),
            "empirical_win_rate": float(y_val[mask].mean()),
        })
    cal_df = pd.DataFrame(cal_rows)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.scatter(cal_df["mean_predicted"], cal_df["empirical_win_rate"],
               s=np.clip(cal_df["n"], 10, None), alpha=0.8)
    ax.set_xlabel("mean predicted P(team1 wins)")
    ax.set_ylabel("empirical team1 win rate")
    ax.set_title("Validation calibration diagnostic (10 bins) - Logistic Regression V1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "logistic_regression_calibration_v1.png", dpi=150)
    plt.close(fig)

    # ---------------- coefficients table ----------------
    TABLES_DIR.mkdir(exist_ok=True, parents=True)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": w,
        "abs_coefficient": np.abs(w),
    })
    coef_df.to_csv(TABLES_DIR / "logistic_regression_coefficients_v1.csv", index=False, encoding="utf-8")
    coef_sorted = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    # ---------------- save model artifacts ----------------
    np.savez(
        MODELS_DIR / "logistic_regression_scratch_v1.npz",
        w=w, b=np.array([b]), feature_names=np.array(feature_names), J_history=np.array(J_history),
    )

    model_meta = {
        "model_type": "logistic_regression_from_scratch",
        "implementation_source": "adapted from course Lab 2 Logistic Regression",
        "engine_version": ENGINE_VERSION,
        "alpha": ALPHA,
        "num_iters": NUM_ITERS,
        "lambda_": LAMBDA_,
        "decision_threshold": THRESHOLD,
        "training_cutoff_datetime": str(train_raw["datetime"].max()),
        "training_rows_before_mirroring": int(len(train_raw)),
        "training_rows_after_mirroring": int(len(augmented_train_raw)),
        "mirrored_train_target_mean": mirrored_target_mean,
        "initial_cost": float(initial_cost),
        "final_cost": float(final_cost),
        "bias_b": float(b),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_artifact_path": "data/modeling/logistic_preprocessing_v1.json",
        "feature_config_path": "config/series_features_v1.yaml",
        "split_manifest_path": "data/modeling/series_split_v1.csv",
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_status": "SEALED - not evaluated in Phase 4A",
        "cologne_status": "UNTOUCHED - not present in series_features_v1.parquet",
    }
    (MODELS_DIR / "logistic_regression_scratch_v1.json").write_text(
        json.dumps(model_meta, indent=2), encoding="utf-8")

    write_report(train_raw, val_raw, augmented_train_raw, feature_names, params,
                 initial_cost, final_cost, abs_decrease, rel_decrease,
                 train_metrics, val_metrics, coef_sorted, cal_df, model_meta)

    print("\nWrote models/logistic_regression_scratch_v1.npz")
    print("Wrote models/logistic_regression_scratch_v1.json")
    print("Wrote data/modeling/logistic_preprocessing_v1.json")
    print("Wrote reports/figures/logistic_regression_{cost,roc,calibration}_v1.png")
    print("Wrote reports/tables/logistic_regression_coefficients_v1.csv")
    print("Wrote reports/phase4a_logistic_regression_v1.md")


def write_report(train_raw, val_raw, augmented_train_raw, feature_names, params,
                  initial_cost, final_cost, abs_decrease, rel_decrease,
                  train_metrics, val_metrics, coef_sorted, cal_df, model_meta):
    md = []
    md.append("# Phase 4A - Model 1: Logistic Regression From Scratch (V1 baseline)\n")

    md.append("## 1. Why Logistic Regression is Model 1\n")
    md.append("The project proposal specifies Model 1 as Logistic Regression, adapted from the course lab. "
              "It serves as the interpretable linear baseline that every later, more flexible model (Random "
              "Forest, XGBoost) must be compared against.\n")

    md.append("## 2. Implemented from scratch\n")
    md.append("The model is implemented entirely from scratch in `scripts/models/logistic_regression_scratch.py`, "
              "adapted from `reference/Lab_2_Logistic_Regression_answer.ipynb`. No `sklearn.linear_model."
              "LogisticRegression`, `statsmodels`, `scipy.optimize`, or any other ML estimator was used to fit "
              "this model - `sklearn.metrics` is used only after our own `predict_proba`/`predict` produced "
              "predictions.\n")

    md.append("## 3. Lab functions adapted\n")
    md.append("From `reference/Lab_2_Logistic_Regression_answer.ipynb`: `sigmoid` (cell 15, `UNQ_C1`), "
              "`compute_cost` (cell 23, `UNQ_C2`), `compute_gradient` (cell 32, `UNQ_C3`), `gradient_descent` "
              "(cell 40), `predict` (cell 48, `UNQ_C4`), `compute_cost_reg` (cell 71, `UNQ_C5`), "
              "`compute_gradient_reg` (cell 78, `UNQ_C6`) - all preserved with the lab's original mathematics "
              "(binary cross-entropy cost, `dj_dw = Err@X/m`, `dj_db = sum(Err)/m`, `w -= alpha*dj_dw`, "
              "`b -= alpha*dj_db`, L2 penalty `(lambda_/(2m))*sum(w^2)` added to cost and `(lambda_/m)*w` added "
              "to `dj_dw`, bias never regularized).\n")

    md.append("## 4. Project-specific modifications\n")
    md.append("Clearly marked `[PROJECT ADAPTATION]`/`[PROJECT ADDITION]` in the source: numerical-stability "
              "probability clipping in `compute_cost`/`compute_cost_reg` (prevents `log(0)` on the larger, more "
              "separable CS2 feature set - does not change the BCE objective); divergence detection inside "
              "`gradient_descent` (raises `GradientDescentDivergenceError` on non-finite or catastrophically "
              "increasing cost, so a broken run stops loudly); `predict_proba` (factored out of the lab's "
              "`predict`); a configurable `threshold` parameter on `predict` (defaults to the lab's hardcoded "
              "0.5); the entire `scripts/preprocessing_logistic_v1.py` module (imputation/standardization/"
              "reference encoding, mirrored augmentation); model serialization to `.npz`/`.json`.\n")

    md.append("## 5. Chronological split methodology\n")
    md.append(f"`scripts/build_series_split_v1.py` splits the 9,456-row Phase 3 development set by exact "
              f"`datetime` group into train/validation/test, choosing the group boundary closest to the "
              f"70%/85% cumulative-row-count marks: train={len(train_raw):,} "
              f"({train_raw['datetime'].min()} to {train_raw['datetime'].max()}), "
              f"validation={len(val_raw):,} ({val_raw['datetime'].min()} to {val_raw['datetime'].max()}). "
              "Full detail in `reports/phase4a_split_summary.md`.\n")

    md.append("## 6. Why validation/test are chronological\n")
    md.append("A random split would let the model be evaluated on matches that occurred *before* some of its "
              "training data chronologically - exactly the kind of leakage Phase 3's historical-feature engine "
              "was built to prevent. A chronological split mirrors genuine deployment: predicting future matches "
              "from past information only.\n")

    md.append("## 7. Why same-timestamp groups are not split\n")
    md.append("Phase 3's engine already treats every match sharing an exact timestamp as mutually invisible to "
              "each other (neither can see the other's result). If such a group were divided across partitions, "
              "a validation/test row could end up in the same simultaneous batch as a training row it was never "
              "meant to be distinguishable from at prediction time - keeping the whole group in one partition "
              "preserves that guarantee.\n")

    md.append("## 8. Train-only preprocessing\n")
    n_cont = len(params["continuous_standardized_features"])
    md.append(f"All imputation medians, means, and standard deviations for the {n_cont} continuous features, and "
              "the `bestOf`/`tier` reference-category encoding, are fit exclusively on the training partition "
              "(specifically the augmented train+mirror set - see Section 9) and saved to "
              "`data/modeling/logistic_preprocessing_v1.json`. Validation (and later test) are only ever "
              "*transformed* with these already-fitted values, never used to compute them.\n")

    md.append("## 9. Mirrored training augmentation and the Team1 orientation bias\n")
    md.append("Phase 1/2.5 established a persistent ~55% Team1 win-rate artifact from how `team1`/`team2` are "
              "assigned, unrelated to real skill. **Mirroring is applied to the raw (pre-preprocessing) training "
              "rows, before preprocessing is fit** - not by negating already-standardized values. This was a "
              "deliberate correction during planning: fitting standardization on the original (biased) training "
              "mean and then separately negating+re-standardizing a mirrored raw row would not give exact "
              "negatives whenever that mean is non-zero (`(-x-mean)/std != -(x-mean)/std`), silently "
              f"reintroducing the very bias mirroring exists to cancel. Instead: {len(train_raw):,} raw training "
              f"rows are mirrored (every directional diff feature negated, symmetric/context features unchanged, "
              f"target flipped) and concatenated with the originals into a "
              f"{len(augmented_train_raw):,}-row augmented set; preprocessing is fit on *that*. Verified: augmented "
              f"target mean = **{model_meta['mirrored_train_target_mean']}** (exactly 0.5), and every directional "
              "diff feature's augmented raw mean is ~0 by construction (`assert_augmented_symmetry`, tight "
              "tolerance for the fully-populated features, a documented looser tolerance for "
              "`days_since_last_match_diff` because NaN-negated-is-still-NaN means a missing value's mirrored pair "
              "shares one imputed value rather than becoming exact negatives - a small, bounded, expected "
              "deviation, not a bug). A dedicated future-inference symmetry test "
              "(`tests/test_preprocessing_logistic.py`) proves the *same fitted artifact* transforms a genuinely "
              "reversed future matchup consistently with this training-time scheme. Validation and test are never "
              "mirrored.\n")

    md.append("## 10. Gradient descent configuration\n")
    md.append(f"`alpha={model_meta['alpha']}`, `num_iters={model_meta['num_iters']}`, `lambda_={model_meta['lambda_']}` "
              "(unregularized V1 baseline) - a single, non-tuned configuration. `num_iters=10000` is lab-style. "
              "`alpha=0.001` is not just a conservative project guess: cell 42 of "
              "`reference/Lab_2_Logistic_Regression_answer.ipynb` - the lab's own **unregularized** "
              "(`lambda_=0`) example - literally uses `alpha=0.001, iterations=10000`, verified by direct "
              "inspection of that cell rather than assumed. (The lab's separate *regularized* example, cell 84, "
              "uses `alpha=0.01, lambda_=0.01` - a different configuration for a different, regularized run, not "
              "reused here since V1 is unregularized.) `w` initialized to zeros, `b=0`, batch gradient descent "
              "only, no other optimizer.\n")

    md.append("## 11. Cost convergence\n")
    md.append(f"Initial cost: **{initial_cost:.6f}**. Final cost: **{final_cost:.6f}**. Absolute decrease: "
              f"**{abs_decrease:.6f}**. Relative decrease: **{100*rel_decrease:.2f}%**. Cost history is finite "
              f"throughout (`reports/figures/logistic_regression_cost_v1.png`); no divergence was encountered "
              "at this configuration.\n")

    def metrics_table(m):
        cm = m["confusion_matrix"]
        lines = [
            f"- n = {m['n']:,}",
            f"- Accuracy: {m['accuracy']:.4f} (majority-class reference: {m['majority_class_accuracy']:.4f})",
            f"- Precision: {m['precision']:.4f}",
            f"- Recall: {m['recall']:.4f}",
            f"- F1: {m['f1']:.4f}",
            f"- ROC-AUC: {m['roc_auc']:.4f}",
            f"- Log loss: {m['log_loss']:.4f}",
            f"- Brier score: {m['brier_score']:.4f}",
            f"- Confusion matrix [[TN,FP],[FN,TP]]: {cm}",
        ]
        return "\n".join(lines)

    md.append("## 12. Train metrics (unmirrored original orientation)\n")
    md.append("Evaluated on the original, unmirrored training rows (not the 2x augmented matrix used to fit the "
              "model) so train and validation are directly comparable.\n")
    md.append(metrics_table(train_metrics) + "\n")

    md.append("## 13. Validation metrics\n")
    md.append(metrics_table(val_metrics) + "\n")

    md.append("## 14. Coefficient interpretation\n")
    md.append("`reports/tables/logistic_regression_coefficients_v1.csv` (feature, coefficient, abs_coefficient), "
              f"bias/intercept `b = {model_meta['bias_b']:.4f}`. Standardized-continuous "
              "features (the 10 diffs + `history_matches_min`/`history_matches_sum`) have coefficients in "
              "standardized units; `both_teams_have_*` are binary; `bestOf_BO3`/`bestOf_BO5`/`tier_tier2`/"
              "`tier_tier3` are one-hot relative to the BO1/tier1 reference category. Top features by absolute "
              "magnitude:\n")
    md.append("| feature | coefficient | abs_coefficient |")
    md.append("|---|---|---|")
    for _, r in coef_sorted.head(8).iterrows():
        md.append(f"| {r['feature']} | {r['coefficient']:.4f} | {r['abs_coefficient']:.4f} |")
    md.append("\nSign and relative direction only - **no causal claims**. For example, a positive `elo_diff` "
              "coefficient means higher Team1 historical ELO relative to Team2 is *associated with* higher "
              "predicted Team1 win probability, holding the model's other inputs fixed; it does not mean ELO "
              "*causes* wins.\n")
    md.append("**Observation**: the 5 symmetric confidence features (`history_matches_min`/`_sum`, "
              "`both_teams_have_*`) and the 4 one-hot `bestOf`/`tier` context columns all learned coefficients on "
              "the order of `1e-18` - effectively zero - while the bias `b` also stayed at essentially 0. This is "
              "an expected consequence of the mirrored augmentation, not a training failure: at initialization "
              "(`w=0,b=0`) every feature's error term is `sigmoid(0)-y = 0.5-y` for a row and `0.5-(1-y) = -(0.5-y)` "
              "for its mirror, so a symmetric feature (identical in both rows) receives exactly cancelling gradient "
              "contributions from each mirrored pair, while a directional feature (sign-flipped in the mirror) "
              "receives *reinforcing* contributions - and this near-cancellation is self-sustaining as long as the "
              "symmetric-feature weights and `b` stay close to 0. In other words, the augmentation doesn't just "
              "balance the target class - it structurally suppresses any coefficient on a feature that carries no "
              "side-relative signal, which is exactly the desired behavior for confidence/context features that "
              "were never meant to indicate *which side* wins.\n")

    md.append("## 15. Probability / calibration observations (validation, diagnostic only)\n")
    md.append("`reports/figures/logistic_regression_roc_v1.png` and `logistic_regression_calibration_v1.png`. "
              "No calibration correction (isotonic/Platt) is applied at this stage - the plot is diagnostic only. "
              "Per-bin mean predicted probability vs. empirical win rate:\n")
    md.append("| bin | n | mean predicted | empirical win rate |")
    md.append("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        md.append(f"| {r['bin']} | {int(r['n'])} | {r['mean_predicted']:.3f} | {r['empirical_win_rate']:.3f} |")
    md.append("")

    md.append("## 16. Limitations of this V1 baseline\n")
    md.append("- Purely linear decision boundary in the 19 transformed features - no interaction terms.")
    md.append("- No regularization tuning yet (`lambda_=0` fixed); no learning-rate or iteration-count tuning.")
    md.append("- Mirrored augmentation neutralizes the orientation bias's effect on training, but does not "
              "explain or fix its unknown root cause (Phase 2.5).")
    md.append("- Only the 17 Phase-3 series-level features - no map-level or player-level detail yet.")
    n_cold_start_train = int((train_raw["both_teams_have_history"] == 0).sum())
    md.append(f"- {n_cold_start_train:,} train rows have `both_teams_have_history==0` (at least one side is a "
              "cold-start team with no prior match) - these add noise the model cannot fully distinguish from a "
              "genuinely even matchup beyond the confidence flags.")
    md.append("- Decision threshold fixed at the lab's default 0.5, not tuned for any downstream use case.")
    md.append("- Single chronological train/validation split - no cross-validation.")
    md.append("- BO5 is a small sample in both train and validation - metrics on that subset are less reliable.\n")

    md.append("## Status\n")
    md.append("- **Internal test partition: SEALED** - not opened or scored in this phase.")
    md.append("- **Cologne 2026 / post-Cologne: UNTOUCHED** - structurally absent from `series_features_v1.parquet`.")
    md.append("- No hyperparameter tuning performed.")
    md.append("- Random Forest and XGBoost have not been trained.\n")

    (REPORTS / "phase4a_logistic_regression_v1.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
