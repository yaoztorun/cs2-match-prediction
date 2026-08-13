"""
Phase 7, Stage C (visualization layer).

Purely descriptive. Every figure below is generated strictly AFTER the
immutable TEST prediction artifact exists, and changes nothing about the
evaluation procedure, model, threshold, or metrics - see
config/phase7_test_evaluation_protocol.yaml's own `visualization` block for
the frozen no-op guarantee.

READ CONTRACT (same discipline as phase7_test_reports.py / phase7_test_bootstrap.py):
this script never reads data/features/map_features_v3_modern_map.parquet and
never reads data/modeling/map_split_v1.csv. It reads:
  (a) data/evaluation/map_test_predictions_v1.parquet directly, for every
      row-level figure (confusion matrix, ROC, probability distributions,
      calibration);
  (b) the tables scripts/phase7_test_reports.py and
      scripts/phase7_test_bootstrap.py already wrote to reports/tables/ -
      every one of those tables is itself derived SOLELY from the prediction
      artifact, so reading them is not a second TEST access, just a cached
      view of the same one-shot record;
  (c) already-frozen PRE-Phase-7 machine-readable artifacts
      (reports/tables/map_xgboost_v3_final_feature_importance.csv,
      map_xgboost_v3_final_group_importance.csv, map_development_vs_test_v1.csv)
      for the feature-importance and generalization panels - interpretation
      only, no TEST target/feature is read from them.

Must run AFTER phase7_test_reports.py and phase7_test_bootstrap.py.

No seaborn. Matplotlib only. 300 DPI minimum, PNG + PDF.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from _common import ROOT, REPORTS

EVAL_DIR = ROOT / "data" / "evaluation"
PRED_PATH = EVAL_DIR / "map_test_predictions_v1.parquet"
TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"

DPI = 300
SMALL_SAMPLE_N = 30
MAP_ORDER = ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "figure.dpi": 100, "savefig.dpi": DPI,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save_fig(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------------------
# B. confusion matrix
# ---------------------------------------------------------------------------

def fig_confusion_matrix(pred):
    y = pred["y_true"].to_numpy()
    p = pred["y_pred_xgb_v3_final"].to_numpy()
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    total = tn + fp + fn + tp
    mat = np.array([[tn, fp], [fn, tp]])
    pct = 100 * mat / total

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. Team1 Loss", "Pred. Team1 Win"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Team1 Loss", "Actual Team1 Win"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                     fontsize=13, color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_title("Final XGBoost — Internal TEST Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    save_fig(fig, "map_xgb_v3_test_confusion_matrix")


# ---------------------------------------------------------------------------
# C. model vs baseline comparison
# ---------------------------------------------------------------------------

def fig_accuracy_auc_comparison(metrics_df):
    order = ["baseline_constant_05", "baseline_overall_elo", "baseline_map_elo", "final_xgb_v3"]
    labels = ["Constant 0.5", "Overall ELO", "Map ELO", "Final XGBoost V3"]
    m = metrics_df.set_index("model").loc[order]
    x = np.arange(len(order)); width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, m["accuracy"], width, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + width / 2, m["roc_auc"], width, label="ROC-AUC", color="#DD8452")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("score (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Internal TEST — Accuracy & ROC-AUC vs. Baselines")
    ax.legend()
    save_fig(fig, "map_test_accuracy_auc_comparison")


def fig_probability_quality_comparison(metrics_df):
    order = ["baseline_constant_05", "baseline_overall_elo", "baseline_map_elo", "final_xgb_v3"]
    labels = ["Constant 0.5", "Overall ELO", "Map ELO", "Final XGBoost V3"]
    m = metrics_df.set_index("model").loc[order]
    x = np.arange(len(order)); width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, m["log_loss"], width, label="Log Loss", color="#55A868")
    b2 = ax.bar(x + width / 2, m["brier"], width, label="Brier", color="#C44E52")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.4f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("error (LOWER IS BETTER)")
    ax.set_title("Internal TEST — Probability Quality vs. Baselines  (lower is better)")
    ax.legend()
    save_fig(fig, "map_test_probability_quality_comparison")


# ---------------------------------------------------------------------------
# D. ROC comparison
# ---------------------------------------------------------------------------

def fig_roc(pred):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for col, label, color in [("p_xgb_v3_final", "Final XGBoost V3", "#4C72B0"),
                               ("p_overall_elo", "Overall ELO", "#DD8452"),
                               ("p_map_elo", "Map ELO", "#55A868")]:
        fpr, tpr, _ = roc_curve(pred["y_true"], pred[col])
        auc = roc_auc_score(pred["y_true"], pred[col])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Internal TEST ROC — Final Model vs. Baselines")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    save_fig(fig, "map_xgb_v3_test_roc")


# ---------------------------------------------------------------------------
# E. probability distributions
# ---------------------------------------------------------------------------

def fig_probability_distribution(pred):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(pred["p_xgb_v3_final"], bins=30, range=(0, 1), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("P(Team1 wins map)"); ax.set_ylabel("count")
    ax.set_title("Internal TEST — Predicted Probability Distribution (Final XGBoost V3)")
    save_fig(fig, "map_xgb_v3_test_probability_distribution")


def fig_probability_by_outcome(pred):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, 1, 26)
    p_win = pred.loc[pred["y_true"] == 1, "p_xgb_v3_final"]
    p_loss = pred.loc[pred["y_true"] == 0, "p_xgb_v3_final"]
    ax.hist(p_win, bins=bins, alpha=0.6, label=f"Actual Team1 win (n={len(p_win)})", color="#55A868",
            density=True)
    ax.hist(p_loss, bins=bins, alpha=0.6, label=f"Actual Team1 loss (n={len(p_loss)})", color="#C44E52",
            density=True)
    ax.set_xlabel("P(Team1 wins map)"); ax.set_ylabel("density")
    ax.set_title("Internal TEST — Predicted Probability by Actual Outcome")
    ax.legend()
    save_fig(fig, "map_xgb_v3_test_probability_by_outcome")


# ---------------------------------------------------------------------------
# F. calibration
# ---------------------------------------------------------------------------

def fig_calibration(cal_df):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    sizes = np.clip(cal_df["n"], 20, None) * 1.2
    ax.scatter(cal_df["mean_predicted_probability"], cal_df["empirical_team1_win_rate"], s=sizes,
               alpha=0.75, color="#4C72B0", edgecolor="black", linewidth=0.5)
    for _, r in cal_df.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_predicted_probability"], r["empirical_team1_win_rate"]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(Team1 wins)"); ax.set_ylabel("empirical Team1 win rate")
    ax.set_title("Internal TEST Calibration — 10 Fixed-Width Bins\n(bubble size ~ bin sample count)")
    ax.legend(loc="upper left")
    save_fig(fig, "map_xgb_v3_test_calibration")


# ---------------------------------------------------------------------------
# G/H/I. map-level bar charts
# ---------------------------------------------------------------------------

def _map_order(df):
    present = [m for m in MAP_ORDER if m in set(df["map_name"])]
    extra = [m for m in df["map_name"] if m not in present]
    return present + sorted(set(extra))


def fig_accuracy_by_map(per_map_df, overall_accuracy):
    order = _map_order(per_map_df)
    d = per_map_df.set_index("map_name").loc[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    bars = ax.barh(y, d["accuracy"], color="#4C72B0")
    for i, (m, r) in enumerate(d.iterrows()):
        label = f"{r['accuracy']:.3f} (n={int(r['n'])})" + (" — small sample" if r["small_sample"] else "")
        ax.annotate(label, (r["accuracy"], i), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#b03030" if r["small_sample"] else "black")
    ax.axvline(overall_accuracy, color="black", linestyle="--", linewidth=1.2,
               label=f"overall TEST accuracy ({overall_accuracy:.3f})")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy"); ax.set_xlim(0, 1)
    ax.set_title("Internal TEST Accuracy by Map (alphabetical order)")
    ax.legend(loc="lower right")
    save_fig(fig, "map_test_accuracy_by_map")


def fig_auc_by_map(per_map_df, overall_auc):
    d = per_map_df[per_map_df["both_classes_present"]].copy()
    order = [m for m in _map_order(per_map_df) if m in set(d["map_name"])]
    d = d.set_index("map_name").loc[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    ax.barh(y, d["roc_auc"], color="#DD8452")
    for i, (m, r) in enumerate(d.iterrows()):
        label = f"{r['roc_auc']:.3f} (n={int(r['n'])})" + (" — small sample" if r["small_sample"] else "")
        ax.annotate(label, (r["roc_auc"], i), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#b03030" if r["small_sample"] else "black")
    ax.axvline(overall_auc, color="black", linestyle="--", linewidth=1.2,
               label=f"overall TEST AUC ({overall_auc:.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.2, label="chance (0.5)")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Internal TEST ROC-AUC by Map (both classes present only)")
    ax.legend(loc="lower right")
    save_fig(fig, "map_test_auc_by_map")


def fig_logloss_by_map(per_map_df, overall_logloss):
    order = _map_order(per_map_df)
    d = per_map_df.set_index("map_name").loc[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    ax.barh(y, d["log_loss"], color="#55A868")
    for i, (m, r) in enumerate(d.iterrows()):
        label = f"{r['log_loss']:.3f} (n={int(r['n'])})" + (" — small sample" if r["small_sample"] else "")
        ax.annotate(label, (r["log_loss"], i), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#b03030" if r["small_sample"] else "black")
    ax.axvline(overall_logloss, color="black", linestyle="--", linewidth=1.2,
               label=f"overall XGB TEST log loss ({overall_logloss:.3f})")
    ax.axvline(0.6931, color="gray", linestyle=":", linewidth=1.2, label="constant-0.5 log loss (0.6931)")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("log loss (LOWER IS BETTER)")
    ax.set_title("Internal TEST Log Loss by Map")
    ax.legend(loc="lower right")
    save_fig(fig, "map_test_logloss_by_map")


# ---------------------------------------------------------------------------
# J. map performance matrix / heatmap
# ---------------------------------------------------------------------------

def fig_performance_matrix(per_map_df):
    order = _map_order(per_map_df)
    d = per_map_df.set_index("map_name").loc[order]
    metrics = ["n", "accuracy", "roc_auc", "log_loss", "brier"]
    higher_better = {"n": None, "accuracy": True, "roc_auc": True, "log_loss": False, "brier": False}

    # column-wise normalization to [0,1] for color, direction-aware; n uses its own scale, not "better/worse"
    norm = np.zeros((len(order), len(metrics)))
    for j, met in enumerate(metrics):
        col = d[met].to_numpy(dtype=float)
        valid = np.isfinite(col)
        if valid.sum() == 0:
            norm[:, j] = 0.5
            continue
        lo, hi = np.nanmin(col[valid]), np.nanmax(col[valid])
        rng = hi - lo if hi > lo else 1.0
        scaled = (col - lo) / rng
        if higher_better.get(met) is False:
            scaled = 1 - scaled
        scaled = np.where(valid, scaled, 0.5)
        norm[:, j] = scaled

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["n", "Accuracy", "ROC-AUC", "Log Loss\n(lower better)", "Brier\n(lower better)"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j, met in enumerate(metrics):
            val = d[met].iloc[i]
            text = f"{int(val)}" if met == "n" else (f"{val:.3f}" if np.isfinite(val) else "n/a")
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                     color="black")
    ax.set_title("Internal TEST — Per-Map Performance Matrix\n(color: green=better, red=worse; exact values printed)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="normalized performance (direction-corrected)")
    save_fig(fig, "map_test_performance_matrix")


# ---------------------------------------------------------------------------
# K/L/M. sample-size figures
# ---------------------------------------------------------------------------

def fig_sample_size_by_map(per_map_df):
    order = _map_order(per_map_df)
    d = per_map_df.set_index("map_name").loc[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(order))
    colors = ["#b03030" if n < SMALL_SAMPLE_N else "#4C72B0" for n in d["n"]]
    ax.barh(y, d["n"], color=colors)
    for i, n in enumerate(d["n"]):
        ax.annotate(str(int(n)), (n, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    ax.axvline(SMALL_SAMPLE_N, color="gray", linestyle=":", label=f"small-sample threshold (n={SMALL_SAMPLE_N})")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("TEST map rows (n)")
    ax.set_title("Internal TEST — Sample Size by Map")
    ax.legend(loc="lower right")
    save_fig(fig, "map_test_sample_size_by_map")


def fig_accuracy_vs_sample_size(per_map_df, overall_accuracy):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(per_map_df["n"], per_map_df["accuracy"], s=70, color="#4C72B0", edgecolor="black", zorder=3)
    for _, r in per_map_df.iterrows():
        ax.annotate(r["map_name"], (r["n"], r["accuracy"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    ax.axhline(overall_accuracy, color="black", linestyle="--", linewidth=1,
               label=f"overall TEST accuracy ({overall_accuracy:.3f})")
    ax.axvline(SMALL_SAMPLE_N, color="gray", linestyle=":", linewidth=1, label=f"n={SMALL_SAMPLE_N}")
    ax.set_xlabel("TEST map rows (n)"); ax.set_ylabel("accuracy")
    ax.set_title("Internal TEST — Accuracy vs. Sample Size, by Map")
    ax.legend()
    save_fig(fig, "map_test_accuracy_vs_sample_size")


def fig_auc_vs_sample_size(per_map_df):
    d = per_map_df[per_map_df["both_classes_present"]]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(d["n"], d["roc_auc"], s=70, color="#DD8452", edgecolor="black", zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["map_name"], (r["n"], r["roc_auc"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="AUC = 0.5 (chance)")
    ax.axvline(SMALL_SAMPLE_N, color="gray", linestyle=":", linewidth=1, label=f"n={SMALL_SAMPLE_N}")
    ax.set_xlabel("TEST map rows (n)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Internal TEST — ROC-AUC vs. Sample Size, by Map")
    ax.legend()
    save_fig(fig, "map_test_auc_vs_sample_size")


# ---------------------------------------------------------------------------
# N. evidence/coverage by map
# ---------------------------------------------------------------------------

def fig_evidence_by_map(pred):
    order = [m for m in MAP_ORDER if m in set(pred["map_name"])]
    rows = []
    for m in order:
        d = pred[pred["map_name"] == m]
        rows.append({
            "map_name": m, "n": len(d),
            "recent_map_history_pct": 100 * (d["both_teams_have_recent_selected_map_history"] == 1).mean(),
            "trusted_adjusted_pct": 100 * (d["map_adjusted_history_mass_min"] > 0).mean(),
            "roster_map_evidence_pct": 100 * (d["roster_map_players_with_history_min"] >= 3).mean(),
        })
    d = pd.DataFrame(rows).set_index("map_name")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(order)); width = 0.26
    ax.bar(x - width, d["recent_map_history_pct"], width, label="recent selected-map history", color="#4C72B0")
    ax.bar(x, d["trusted_adjusted_pct"], width, label="trusted opponent-adjusted evidence", color="#DD8452")
    ax.bar(x + width, d["roster_map_evidence_pct"], width, label="current-roster map evidence (≥3)",
           color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("% of TEST rows"); ax.set_ylim(0, 100)
    ax.set_title("Internal TEST — Evidence Coverage by Map")
    ax.legend()
    save_fig(fig, "map_test_evidence_by_map")


# ---------------------------------------------------------------------------
# O. development -> TEST generalization
# ---------------------------------------------------------------------------

def fig_generalization_summary(dev_test_df):
    dev = dev_test_df[dev_test_df["source"].str.contains("Phase 6D")].iloc[0]
    test = dev_test_df[dev_test_df["source"].str.contains("Phase 7")].iloc[0]
    metrics = ["accuracy", "roc_auc", "log_loss", "brier"]
    labels = ["Accuracy", "ROC-AUC", "Log Loss", "Brier"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, met, lab in zip(axes, metrics, labels):
        vals = [dev[met], test[met]]
        bars = ax.bar(["Development\nOOF", "Sealed\nTEST"], vals, color=["#4C72B0", "#C44E52"])
        for b in bars:
            ax.annotate(f"{b.get_height():.4f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(lab)
    fig.suptitle("Phase 6D Development OOF vs. Phase 7 Sealed TEST — Same Frozen Model", fontweight="bold")
    save_fig(fig, "map_xgb_generalization_summary")


# ---------------------------------------------------------------------------
# P. feature importance (frozen Phase 6D artifacts - interpretation only)
# ---------------------------------------------------------------------------

def fig_top_features(imp_df):
    d = imp_df.sort_values("permutation_importance_mean", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(d))
    ax.barh(y, d["permutation_importance_mean"], color="#4C72B0")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.annotate(f"  [{r['family']}]", (r["permutation_importance_mean"], i), va="center", fontsize=8,
                    color="#555555")
    ax.set_yticks(y); ax.set_yticklabels(d["feature"])
    ax.set_xlabel("TRAIN-only CV permutation importance (ROC-AUC decrease)")
    ax.set_title("Top 15 Individually Important Features — Final XGBoost V3\n(frozen Phase 6D result; "
                 "interpretation only)")
    save_fig(fig, "map_xgb_v3_top_features")


def fig_feature_family_importance(group_df):
    d = group_df.sort_values("auc_decrease_mean", ascending=False)
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in d["auc_decrease_mean"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(d))
    ax.barh(y, d["auc_decrease_mean"], color=colors)
    ax.set_yticks(y); ax.set_yticklabels([f"{f} — {lbl}" for f, lbl in zip(d["family"], d["family_label"])],
                                          fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("grouped permutation importance (ROC-AUC decrease)")
    ax.set_title("Feature-Family Grouped Importance — Final XGBoost V3\n(frozen Phase 6D result; "
                 "interpretation only)")
    save_fig(fig, "map_xgb_v3_feature_family_importance")


# ---------------------------------------------------------------------------
# Q. BO1/BO3/BO5
# ---------------------------------------------------------------------------

def fig_performance_by_bestof(bo_df):
    d = bo_df.sort_values("bestOf")
    labels = [f"BO{int(b)}" for b in d["bestOf"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, met, lab in zip(axes, ["accuracy", "roc_auc", "log_loss"], ["Accuracy", "ROC-AUC", "Log Loss"]):
        bars = ax.bar(labels, d[met].fillna(0), color="#4C72B0")
        for b, n, small in zip(bars, d["n"], d["small_sample"]):
            note = " (small n)" if small else ""
            ax.annotate(f"n={int(n)}{note}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        color="#b03030" if small else "black")
        ax.set_title(lab)
    fig.suptitle("Internal TEST Performance by Series Format (BO1/BO3/BO5)", fontweight="bold")
    save_fig(fig, "map_test_performance_by_bestof")


# ---------------------------------------------------------------------------
# R. coverage/evidence performance
# ---------------------------------------------------------------------------

def fig_performance_by_evidence(pred, cov_df):
    all_m = {"subgroup": "All TEST", "n": len(pred),
             "log_loss": float(np.mean(-(pred["y_true"] * np.log(np.clip(pred["p_xgb_v3_final"], 1e-15, 1))
                                          + (1 - pred["y_true"]) * np.log(np.clip(1 - pred["p_xgb_v3_final"], 1e-15, 1))))),
             "accuracy": float((pred["y_pred_xgb_v3_final"] == pred["y_true"]).mean())}
    label_map = {
        "A_recent_map_evidence": "Recent map history",
        "B_trusted_adjusted_evidence": "Trusted adjusted history",
        "C_roster_map_evidence_ge3": "Roster map evidence ≥3",
        "D_strong_roster_map_evidence_ge5": "Strong roster evidence ≥5",
        "H_high_evidence": "High evidence",
        "E_roster_map_cold_start": "Roster-map cold start",
        "F_team_map_cold_start": "Team-map cold start",
    }
    order = ["All TEST", "Recent map history", "Trusted adjusted history", "Roster map evidence ≥3",
             "Strong roster evidence ≥5", "High evidence", "Roster-map cold start", "Team-map cold start"]
    rows = [all_m]
    for _, r in cov_df.iterrows():
        if r["subgroup"] in label_map:
            rows.append({"subgroup": label_map[r["subgroup"]], "n": r["n"], "log_loss": r["log_loss"],
                          "accuracy": r["accuracy"]})
    d = pd.DataFrame(rows).set_index("subgroup").reindex(order).dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(d))
    ax.barh(y, d["log_loss"], color="#55A868")
    for i, (name, r) in enumerate(d.iterrows()):
        ax.annotate(f"{r['log_loss']:.3f} (n={int(r['n'])})", (r["log_loss"], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(d.index)
    ax.invert_yaxis()
    ax.set_xlabel("log loss (lower is better)")
    ax.set_title("Internal TEST — Log Loss by Evidence/Coverage Group")
    save_fig(fig, "map_test_performance_by_evidence")


# ---------------------------------------------------------------------------
# S. bootstrap uncertainty
# ---------------------------------------------------------------------------

def fig_bootstrap_ci(ci_df):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, metric, lab in zip(axes, ["accuracy", "roc_auc", "log_loss", "brier"],
                                 ["Accuracy", "ROC-AUC", "Log Loss", "Brier"]):
        r = ci_df[ci_df["metric"] == metric].iloc[0]
        ax.errorbar([0], [r["point_estimate"]],
                     yerr=[[r["point_estimate"] - r["ci_lower_95"]], [r["ci_upper_95"] - r["point_estimate"]]],
                     fmt="o", color="#4C72B0", capsize=8, markersize=10, linewidth=2)
        ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_title(f"{lab}\n{r['point_estimate']:.4f} [{r['ci_lower_95']:.4f}, {r['ci_upper_95']:.4f}]",
                     fontsize=10)
    fig.suptitle("Internal TEST — Final XGBoost V3, 95% Cluster-Bootstrap Intervals (N=2000, by match_id)",
                 fontweight="bold")
    save_fig(fig, "map_test_bootstrap_ci")


# ---------------------------------------------------------------------------
# T. poster summary (6-panel composite)
# ---------------------------------------------------------------------------

def fig_poster_summary(pred, cal_df, per_map_df, overall_accuracy, overall_logloss):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # (a) confusion matrix
    ax = fig.add_subplot(gs[0, 0])
    y = pred["y_true"].to_numpy(); p = pred["y_pred_xgb_v3_final"].to_numpy()
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    mat = np.array([[tn, fp], [fn, tp]])
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred L", "Pred W"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual L", "Actual W"], fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=11,
                     color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_title("(a) Confusion Matrix", fontsize=11, fontweight="bold")

    # (b) ROC
    ax = fig.add_subplot(gs[0, 1])
    for col, label in [("p_xgb_v3_final", "XGB"), ("p_overall_elo", "Overall ELO"), ("p_map_elo", "Map ELO")]:
        fpr, tpr, _ = roc_curve(pred["y_true"], pred[col])
        auc = roc_auc_score(pred["y_true"], pred[col])
        ax.plot(fpr, tpr, label=f"{label} ({auc:.3f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(b) ROC Curve", fontsize=11, fontweight="bold")

    # (c) probability distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(pred["p_xgb_v3_final"], bins=25, range=(0, 1), color="#4C72B0")
    ax.set_xlabel("P(Team1 wins)", fontsize=9); ax.set_ylabel("count", fontsize=9)
    ax.set_title("(c) Probability Distribution", fontsize=11, fontweight="bold")

    # (d) calibration
    ax = fig.add_subplot(gs[1, 0])
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.scatter(cal_df["mean_predicted_probability"], cal_df["empirical_team1_win_rate"],
               s=np.clip(cal_df["n"], 15, None), color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted", fontsize=9); ax.set_ylabel("empirical rate", fontsize=9)
    ax.set_title("(d) Calibration", fontsize=11, fontweight="bold")

    # (e) accuracy by map
    ax = fig.add_subplot(gs[1, 1])
    order = _map_order(per_map_df)
    d = per_map_df.set_index("map_name").loc[order]
    ypos = np.arange(len(order))
    ax.barh(ypos, d["accuracy"], color="#4C72B0")
    ax.axvline(overall_accuracy, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy", fontsize=9)
    ax.set_title("(e) Accuracy by Map", fontsize=11, fontweight="bold")

    # (f) log loss by map
    ax = fig.add_subplot(gs[1, 2])
    ax.barh(ypos, d["log_loss"], color="#55A868")
    ax.axvline(overall_logloss, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("log loss", fontsize=9)
    ax.set_title("(f) Log Loss by Map", fontsize=11, fontweight="bold")

    fig.suptitle("Final Known-Map XGBoost V3 — Internal TEST Summary", fontsize=16, fontweight="bold", y=1.02)
    save_fig(fig, "map_test_poster_summary", tight=False)


# ---------------------------------------------------------------------------

def main():
    if not PRED_PATH.exists():
        raise RuntimeError(f"{PRED_PATH} does not exist - run scripts/evaluate_phase7_test_once.py first.")
    pred = pd.read_parquet(PRED_PATH, engine="fastparquet")

    metrics_df = pd.read_csv(TABLES_DIR / "map_test_metrics_v1.csv")
    per_map_df = pd.read_csv(TABLES_DIR / "map_test_per_map_v1.csv")
    bo_df = pd.read_csv(TABLES_DIR / "map_test_bestof_v1.csv")
    cov_df = pd.read_csv(TABLES_DIR / "map_test_coverage_v1.csv")
    cal_df = pd.read_csv(TABLES_DIR / "map_test_calibration_bins_v1.csv")
    dev_test_df = pd.read_csv(TABLES_DIR / "map_development_vs_test_v1.csv")
    ci_df = pd.read_csv(TABLES_DIR / "map_test_bootstrap_ci_v1.csv")
    imp_df = pd.read_csv(TABLES_DIR / "map_xgboost_v3_final_feature_importance.csv")   # frozen Phase 6D
    group_df = pd.read_csv(TABLES_DIR / "map_xgboost_v3_final_group_importance.csv")   # frozen Phase 6D

    final_row = metrics_df[metrics_df["model"] == "final_xgb_v3"].iloc[0]
    overall_accuracy = float(final_row["accuracy"])
    overall_auc = float(final_row["roc_auc"])
    overall_logloss = float(final_row["log_loss"])

    print("Generating Phase 7 visualization layer (descriptive only, no procedure changes)...")
    fig_confusion_matrix(pred)
    fig_accuracy_auc_comparison(metrics_df)
    fig_probability_quality_comparison(metrics_df)
    fig_roc(pred)
    fig_probability_distribution(pred)
    fig_probability_by_outcome(pred)
    fig_calibration(cal_df)
    fig_accuracy_by_map(per_map_df, overall_accuracy)
    fig_auc_by_map(per_map_df, overall_auc)
    fig_logloss_by_map(per_map_df, overall_logloss)
    fig_performance_matrix(per_map_df)
    fig_sample_size_by_map(per_map_df)
    fig_accuracy_vs_sample_size(per_map_df, overall_accuracy)
    fig_auc_vs_sample_size(per_map_df)
    fig_evidence_by_map(pred)
    fig_generalization_summary(dev_test_df)
    fig_top_features(imp_df)
    fig_feature_family_importance(group_df)
    fig_performance_by_bestof(bo_df)
    fig_performance_by_evidence(pred, cov_df)
    fig_bootstrap_ci(ci_df)
    fig_poster_summary(pred, cal_df, per_map_df, overall_accuracy, overall_logloss)
    print("\nAll Phase 7 figures written to reports/figures/ (PNG + PDF, >=300 DPI).")


if __name__ == "__main__":
    main()
