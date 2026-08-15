"""
Phase 8E presentation figures. Strictly downstream of finalized local
artifacts (canonical actual results, actual-match predictions, milestone/
metric tables, frozen Phase 8D aggregates) - this module never opens the RF
model, never runs the tournament engine, never reads raw match data, and
never accesses the network (amendment #21). matplotlib only, no seaborn,
PNG + PDF at 300 DPI, mirroring phase8d_figures.py's palette exactly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from _common import ROOT
import phase8e_metrics as pm

DPI = 300
FIG_BG = "#ffffff"
BAR_COLOR = "#2b6cb0"
ACCENT_COLOR = "#c05621"
GRID_COLOR = "#d9d9d9"
plt.rcParams.update({
    "figure.facecolor": FIG_BG, "axes.facecolor": FIG_BG, "savefig.facecolor": FIG_BG,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "text.color": "#222222",
    "xtick.color": "#222222", "ytick.color": "#222222", "axes.grid": True,
    "grid.color": GRID_COLOR, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 100,
})

OUT_DIR = ROOT / "reports" / "figures" / "phase8e"


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def fig_01_championship_probability(m, out_dir):
    tp = pd.read_csv(pm.TEAM_PROB_PATH)
    sub = tp[tp.metric == "win_tournament"].sort_values("probability", ascending=False)
    colors = [ACCENT_COLOR if t == m["champion"] else BAR_COLOR for t in sub["canonical_model_name"]]
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(sub["display_name"], sub["probability"] * 100, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Pre-event championship probability (%)")
    ax.set_title("Cologne 2026 — Pre-Event Championship Probability\n(actual champion in orange)")
    return _save(fig, out_dir, "01_championship_probability_actual_champion")


def fig_02_playoff_probability(m, out_dir):
    tp = pd.read_csv(pm.TEAM_PROB_PATH)
    sub = tp[tp.metric == "reach_playoffs"].sort_values("probability", ascending=False)
    colors = [ACCENT_COLOR if t in m["playoff_teams"] else BAR_COLOR for t in sub["canonical_model_name"]]
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(sub["display_name"], sub["probability"] * 100, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Pre-event playoff-qualification probability (%)")
    ax.set_title("Cologne 2026 — Playoff Probability\n(actual playoff teams in orange)")
    return _save(fig, out_dir, "02_playoff_probability_actual_teams")


def fig_03_progression_matrix(m, out_dir):
    milestone_df = m["milestone_df"].set_index("team")
    cols = [("predicted_p_playoffs", "actual_reached_playoffs", "Playoffs"),
            ("predicted_p_semifinal", "actual_reached_semifinal", "Semifinal"),
            ("predicted_p_final", "actual_reached_final", "Final"),
            ("predicted_p_champion", "actual_champion", "Champion")]
    order = milestone_df.sort_values("predicted_p_playoffs", ascending=False).index
    mat = np.array([[milestone_df.loc[t, c[0]] for c in cols] for t in order])
    fig, ax = plt.subplots(figsize=(6, 11))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    for i, t in enumerate(order):
        for j, c in enumerate(cols):
            if milestone_df.loc[t, c[1]]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ACCENT_COLOR, linewidth=2.5))
    fig.colorbar(im, ax=ax, label="predicted probability", shrink=0.6)
    ax.set_title("Predicted vs Actual Progression\n(orange border = actually reached)")
    return _save(fig, out_dir, "03_predicted_vs_actual_progression")


def fig_04_top8_playoff_comparison(m, out_dir):
    comp = m["top_k_set_comparisons"]["playoffs_top8"]
    teams = sorted(set(comp["predicted"]) | set(comp["actual"]))
    tp = pd.read_csv(pm.TEAM_PROB_PATH)
    tp = tp[tp.metric == "reach_playoffs"].set_index("canonical_model_name")
    both = [t for t in teams if t in comp["predicted"] and t in comp["actual"]]
    pred_only = [t for t in teams if t in comp["predicted"] and t not in comp["actual"]]
    act_only = [t for t in teams if t in comp["actual"] and t not in comp["predicted"]]
    ordered = sorted(both, key=lambda t: -tp.loc[t, "probability"]) + \
        sorted(pred_only, key=lambda t: -tp.loc[t, "probability"]) + \
        sorted(act_only, key=lambda t: -tp.loc[t, "probability"])
    colors = {"both": "#2f855a", "predicted_only": BAR_COLOR, "actual_only": ACCENT_COLOR}
    bar_colors = [colors["both"] if t in both else colors["predicted_only"] if t in pred_only else colors["actual_only"]
                  for t in ordered]
    probs = [tp.loc[t, "probability"] * 100 for t in ordered]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(ordered))))
    ax.barh(ordered, probs, color=bar_colors)
    ax.invert_yaxis()
    ax.set_xlabel("Pre-event playoff-qualification probability (%)")
    ax.set_title(f"Top-8 Predicted vs Actual Playoff Teams "
                 f"(overlap {comp['overlap_count']}/8, Jaccard {comp['jaccard']:.2f})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["both"], label="predicted AND actual"),
                        Patch(color=colors["predicted_only"], label="predicted only"),
                        Patch(color=colors["actual_only"], label="actual only")], loc="lower right")
    return _save(fig, out_dir, "04_top8_playoff_prediction_vs_actual")


def fig_05_confusion_matrix(m, out_dir):
    pred_df = m["pred_df"]
    y_true = pred_df["actual_team_a_win"].values
    y_pred = (pred_df["probability_team_a"].values >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["team_b wins", "team_a wins"])
    ax.set_yticklabels(["team_b wins", "team_a wins"])
    ax.set_xlabel("Predicted (p>=0.5 -> team_a)")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#222222", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(f"Match-Level Confusion Matrix (n={len(pred_df)})")
    return _save(fig, out_dir, "05_match_confusion_matrix")


def fig_06_roc_curve(m, out_dir):
    pred_df = m["pred_df"]
    y_true = pred_df["actual_team_a_win"].values
    p = pred_df["probability_team_a"].values
    fpr, tpr, _ = roc_curve(y_true, p)
    auc = m["match_metrics"]["overall"]["roc_auc"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BAR_COLOR, linewidth=2, label=f"RF V2 @ Cologne (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, linestyle="--", linewidth=1.5, label="AUC=0.5 baseline")
    ax.set_xlabel("False positive rate (team_a wins)")
    ax.set_ylabel("True positive rate (team_a wins)")
    ax.set_title("Match-Level ROC Curve (106 actual series)")
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "06_match_roc_curve")


def fig_07_winner_probability_distribution(m, out_dir):
    pred_df = m["pred_df"].copy()
    pred_df["p_actual_winner"] = np.where(pred_df["actual_team_a_win"] == 1, pred_df["probability_team_a"],
                                           pred_df["probability_team_b"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_df["p_actual_winner"], bins=20, color=BAR_COLOR, edgecolor="white")
    ax.axvline(0.5, color=ACCENT_COLOR, linestyle="--", linewidth=1.5, label="p=0.5")
    ax.axvline(pred_df["p_actual_winner"].mean(), color="#2f855a", linestyle="-", linewidth=1.5,
               label=f"mean={pred_df['p_actual_winner'].mean():.3f}")
    ax.set_xlabel("Frozen probability assigned to the actual winner")
    ax.set_ylabel("Number of matches")
    ax.set_title("Distribution of P(actual winner) — 106 Actual Series")
    ax.legend()
    return _save(fig, out_dir, "07_actual_winner_probability_distribution")


def fig_08_metrics_by_stage(m, out_dir):
    by_stage = m["match_metrics"]["by_stage"]
    stages = ["stage_1", "stage_2", "stage_3", "playoffs"]
    labels = ["Stage 1", "Stage 2", "Stage 3", "Playoffs"]
    acc = [by_stage[s]["accuracy"] for s in stages]
    ll = [by_stage[s]["log_loss"] for s in stages]
    n = [by_stage[s]["n"] for s in stages]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stages))
    ax1.bar(x - 0.2, acc, width=0.4, color=BAR_COLOR, label="Accuracy")
    ax1.set_ylabel("Accuracy", color=BAR_COLOR)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, ll, width=0.4, color=ACCENT_COLOR, label="Log Loss")
    ax2.set_ylabel("Log Loss", color=ACCENT_COLOR)
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n(n={ni})" for l, ni in zip(labels, n)])
    ax1.set_title("Match Accuracy and Log Loss by Stage")
    return _save(fig, out_dir, "08_match_metrics_by_stage")


def fig_09_stage_advancement(m, out_dir):
    adv = m["stage_advancement_evaluation"]
    stages = ["stage_1", "stage_2", "stage_3"]
    labels = ["Stage 1\nP(advance)", "Stage 2\nP(advance|participates)", "Stage 3\nP(advance|participates)"]
    brier = [adv[s]["metrics"]["brier"] for s in stages]
    ll = [adv[s]["metrics"]["log_loss"] for s in stages]
    n = [adv[s]["metrics"]["n"] for s in stages]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stages))
    ax.bar(x - 0.2, brier, width=0.4, color=BAR_COLOR, label="Brier")
    ax.bar(x + 0.2, ll, width=0.4, color=ACCENT_COLOR, label="Log Loss")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={ni})" for l, ni in zip(labels, n)])
    ax.set_title("Stage-Advancement Prediction Quality vs Reality")
    ax.legend()
    return _save(fig, out_dir, "09_stage_advancement_vs_reality")


def fig_10_swiss_record_probabilities(m, out_dir):
    df = m["swiss_record_df"].dropna(subset=["realized_record_probability"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["realized_record_probability"], bins=15, color=BAR_COLOR, edgecolor="white")
    ax.axvline(df["realized_record_probability"].mean(), color=ACCENT_COLOR, linestyle="--", linewidth=1.5,
               label=f"mean={df['realized_record_probability'].mean():.3f}")
    ax.set_xlabel("P(realized terminal Swiss record | participated in stage)")
    ax.set_ylabel("Number of (team, stage) participations")
    ax.set_title("Realized Swiss-Record Probability — All Stage Participations")
    ax.legend()
    return _save(fig, out_dir, "10_swiss_realized_record_probability")


def fig_11_favorite_path_vs_reality(m, out_dir):
    structural = m["favorite_path_vs_reality"]["structural_set_comparisons"]
    labels = ["Stage 1\nadvancers", "Stage 2\nadvancers", "Playoff\nteams", "Semi-\nfinalists", "Finalists", "Champion"]
    keys = ["stage_1_advancers", "stage_2_advancers", "stage_3_advancers_playoff_teams",
            "semifinalists", "finalists", "champion"]
    overlap = [structural[k]["overlap_count"] for k in keys]
    total = [len(structural[k]["actual"]) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(keys))
    ax.bar(x, total, color=GRID_COLOR, label="actual set size")
    ax.bar(x, overlap, color=BAR_COLOR, label="matched by favorite-wins path")
    for i, (o, t) in enumerate(zip(overlap, total)):
        ax.text(i, t + 0.15, f"{o}/{t}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Deterministic Favorite-Wins Path vs Reality (structural milestones)")
    ax.legend()
    return _save(fig, out_dir, "11_favorite_path_vs_actual_progression")


def fig_12_summary_card(m, out_dir, summary):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle("IEM Cologne Major 2026 — Simulation vs Reality", fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    champ = m["champion_analysis"]
    ax.text(0.5, 0.7, "Actual Champion", ha="center", fontsize=11, color="#555555")
    ax.text(0.5, 0.45, champ["actual_champion"], ha="center", fontsize=15, fontweight="bold", color=ACCENT_COLOR)
    ax.text(0.5, 0.15, f"pre-event p={champ['championship_probability']:.3f}, rank {champ['championship_rank']}/32",
            ha="center", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    mm = m["match_metrics"]["overall"]
    ax.text(0.5, 0.85, "Match-Level (n=106)", ha="center", fontsize=11, color="#555555")
    ax.text(0.5, 0.55, f"Accuracy {mm['accuracy']:.3f}", ha="center", fontsize=12)
    ax.text(0.5, 0.30, f"AUC {mm['roc_auc']:.3f}", ha="center", fontsize=12)
    ax.text(0.5, 0.05, f"Log Loss {mm['log_loss']:.3f}", ha="center", fontsize=12)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    topk = m["top_k_set_comparisons"]["playoffs_top8"]
    ax.text(0.5, 0.7, "Playoff Overlap", ha="center", fontsize=11, color="#555555")
    ax.text(0.5, 0.4, f"{topk['overlap_count']}/8", ha="center", fontsize=20, fontweight="bold", color=BAR_COLOR)
    ax.text(0.5, 0.1, f"Jaccard {topk['jaccard']:.2f}", ha="center", fontsize=10)

    ax = fig.add_subplot(gs[1, :2])
    stages = ["stage_1", "stage_2", "stage_3", "playoffs"]
    acc = [m["match_metrics"]["by_stage"][s]["accuracy"] for s in stages]
    ax.bar(["Stage 1", "Stage 2", "Stage 3", "Playoffs"], acc, color=BAR_COLOR)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by Stage")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    fav = m["favorite_path_vs_reality"]["structural_set_comparisons"]["champion"]
    ax.text(0.5, 0.7, "Favorite-Path Champion", ha="center", fontsize=11, color="#555555")
    ax.text(0.5, 0.4, fav["predicted"][0], ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.1, "correct" if fav["overlap_count"] == 1 else "did not win", ha="center", fontsize=10,
            color="#2f855a" if fav["overlap_count"] == 1 else ACCENT_COLOR)

    return _save(fig, out_dir, "12_simulation_vs_reality_summary_card")


def build_all():
    m = pm.compute_all_metrics()
    import json
    summary = json.loads((ROOT / "data" / "evaluation" / "cologne_2026_simulation_vs_reality_summary_v1.json")
                          .read_text(encoding="utf-8"))
    out_dir = OUT_DIR
    figs = [
        fig_01_championship_probability(m, out_dir), fig_02_playoff_probability(m, out_dir),
        fig_03_progression_matrix(m, out_dir), fig_04_top8_playoff_comparison(m, out_dir),
        fig_05_confusion_matrix(m, out_dir), fig_06_roc_curve(m, out_dir),
        fig_07_winner_probability_distribution(m, out_dir), fig_08_metrics_by_stage(m, out_dir),
        fig_09_stage_advancement(m, out_dir), fig_10_swiss_record_probabilities(m, out_dir),
        fig_11_favorite_path_vs_reality(m, out_dir), fig_12_summary_card(m, out_dir, summary),
    ]
    all_paths = [p for group in figs for p in group]
    print(f"wrote {len(all_paths)} figure files under {out_dir}")
    return all_paths


if __name__ == "__main__":
    build_all()
