"""
Phase 8D presentation figures. Strictly downstream of the frozen aggregate
artifacts - this module never re-runs the simulation, never touches the
probability matrix or the model, and never changes anything based on how a
figure looks. matplotlib only (no seaborn). PNG + PDF, >=300 DPI.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _team_probabilities(agg_dir):
    return pd.read_csv(agg_dir / "cologne_2026_pre_event_team_probabilities_v1.csv")


def _swiss_records(agg_dir):
    return pd.read_csv(agg_dir / "cologne_2026_pre_event_swiss_record_distributions_v1.csv")


def _playoff_seeds(agg_dir):
    return pd.read_csv(agg_dir / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv")


def _favorite_path(agg_dir):
    return json.loads((agg_dir / "cologne_2026_pre_event_favorite_path_v1.json").read_text(encoding="utf-8"))


def _metric_series(tp, metric, sort_desc=True):
    sub = tp[tp["metric"] == metric].copy()
    sub = sub.sort_values("probability", ascending=not sort_desc)
    return sub


def fig_championship_probability_all(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    sub = _metric_series(tp, "win_tournament")
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(sub["display_name"], sub["probability"] * 100, color=BAR_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Championship probability (%)")
    ax.set_title("IEM Cologne Major 2026 — Pre-Event Championship Probability (All 32 Teams)")
    return _save(fig, out_dir, "01_championship_probability_all_teams")


def fig_championship_probability_top16(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    sub = _metric_series(tp, "win_tournament").head(16)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sub["display_name"], sub["probability"] * 100, color=BAR_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Championship probability (%)")
    ax.set_title("Top 16 — Championship Probability")
    return _save(fig, out_dir, "02_championship_probability_top16")


def fig_playoff_qualification_probability(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    sub = _metric_series(tp, "reach_playoffs")
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(sub["display_name"], sub["probability"] * 100, color=BAR_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Playoff qualification probability (%)")
    ax.set_title("Playoff Qualification Probability (All 32 Teams)")
    return _save(fig, out_dir, "03_playoff_qualification_probability")


def fig_final_probability(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    sub = _metric_series(tp, "reach_final")
    sub = sub[sub["probability"] > 0]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sub))))
    ax.barh(sub["display_name"], sub["probability"] * 100, color=ACCENT_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Grand Final appearance probability (%)")
    ax.set_title("Grand Final Appearance Probability")
    return _save(fig, out_dir, "04_grand_final_probability")


def fig_stage_progression_matrix(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    metrics = ["participate_stage_1", "participate_stage_2", "participate_stage_3",
               "reach_playoffs", "reach_semifinal", "reach_final", "win_tournament"]
    labels = ["Stage 1", "Stage 2", "Stage 3", "Playoffs", "Semifinal", "Final", "Champion"]
    pivot = tp[tp["metric"].isin(metrics)].pivot(index="display_name", columns="metric", values="probability")
    pivot = pivot[metrics]
    order = pivot.sort_values("win_tournament", ascending=False).index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Stage Progression Probability Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Probability")
    return _save(fig, out_dir, "05_stage_progression_matrix")


def fig_advancement_by_starting_stage(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    metric_by_stage = {"stage_1": "advance_from_stage_1", "stage_2": "advance_from_stage_2",
                        "stage_3": "advance_from_stage_3"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, (stage, metric) in zip(axes, metric_by_stage.items()):
        sub = tp[(tp["metric"] == metric) & (tp["starting_stage"] == stage)].sort_values("probability", ascending=False)
        ax.bar(range(len(sub)), sub["probability"] * 100, color=BAR_COLOR)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["display_name"], rotation=90, fontsize=6)
        ax.set_title(f"{stage.replace('_', ' ').title()} entrants")
        ax.set_ylabel("Advance probability (%)")
    fig.suptitle("Swiss Advancement Probability, Conditional on Stage Participation, by Starting Stage",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "06_advancement_by_starting_stage")


def fig_30_probability(agg_dir, out_dir):
    sr = _swiss_records(agg_dir)
    sub = sr[sr["record"] == "3-0"].sort_values("conditional_on_participating_probability", ascending=False)
    sub = sub[sub["conditional_on_participating_probability"] > 0]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(sub))))
    ax.barh(sub["display_name"] + " (" + sub["stage"].str.replace("stage_", "S") + ")",
            sub["conditional_on_participating_probability"] * 100, color=BAR_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("P(3-0) | participated in stage (%)")
    ax.set_title("3-0 Sweep Probability, by Swiss-Stage Participant")
    return _save(fig, out_dir, "07_probability_3_0")


def fig_03_probability(agg_dir, out_dir):
    sr = _swiss_records(agg_dir)
    sub = sr[sr["record"] == "0-3"].sort_values("conditional_on_participating_probability", ascending=False)
    sub = sub[sub["conditional_on_participating_probability"] > 0]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(sub))))
    ax.barh(sub["display_name"] + " (" + sub["stage"].str.replace("stage_", "S") + ")",
            sub["conditional_on_participating_probability"] * 100, color=ACCENT_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("P(0-3) | participated in stage (%)")
    ax.set_title("0-3 Elimination Probability, by Swiss-Stage Participant")
    return _save(fig, out_dir, "08_probability_0_3")


def fig_playoff_seed_heatmap(agg_dir, out_dir):
    ps = _playoff_seeds(agg_dir)
    pivot = ps.pivot(index="display_name", columns="playoff_seed",
                      values="conditional_on_reaching_playoffs_probability")
    order = ps.groupby("display_name")["count"].sum().sort_values(ascending=False).index
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(1, 9))
    ax.set_xlabel("Playoff seed")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Playoff Seed Distribution | Reaches Playoffs")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Conditional probability")
    return _save(fig, out_dir, "09_playoff_seed_heatmap")


def fig_favorite_bracket(agg_dir, out_dir):
    fav = _favorite_path(agg_dir)
    playoff_matches = [m for m in fav["matches"] if m["stage"] == "playoffs"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_title(f"Model-Favorite Deterministic Bracket — Champion: {fav['champion']}", fontweight="bold")
    y = 0.95
    round_labels = {1: "Quarterfinal", 2: "Semifinal", 3: "Grand Final"}
    for m in playoff_matches:
        label = (f"{round_labels.get(m['round'], m['round'])}: {m['team_a']} vs {m['team_b']}  "
                 f"(Bo{m['best_of']}, P(a)={m['probability_team_a']:.2f})  ->  {m['predicted_winner']}")
        ax.text(0.02, y, label, fontsize=9, family="monospace", transform=ax.transAxes)
        y -= 0.09
    return _save(fig, out_dir, "10_model_favorite_bracket")


def fig_monte_carlo_funnel_top_teams(agg_dir, out_dir, top_n=8):
    tp = _team_probabilities(agg_dir)
    champs = _metric_series(tp, "win_tournament").head(top_n)
    metrics = ["participate_stage_1", "participate_stage_2", "participate_stage_3",
               "reach_playoffs", "reach_semifinal", "reach_final", "win_tournament"]
    labels = ["S1", "S2", "S3", "Playoffs", "SF", "Final", "Champion"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in champs.iterrows():
        team = row["display_name"]
        series = tp[(tp["display_name"] == team) & (tp["metric"].isin(metrics))].set_index("metric")
        y = [series.loc[m, "probability"] * 100 if m in series.index else np.nan for m in metrics]
        ax.plot(labels, y, marker="o", label=team)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"Monte Carlo Tournament Funnel — Top {top_n} Teams by Championship Probability")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_dir, "11_monte_carlo_funnel_top_teams")


def fig_presentation_summary(agg_dir, out_dir):
    tp = _team_probabilities(agg_dir)
    champs = _metric_series(tp, "win_tournament").head(8)
    playoffs = _metric_series(tp, "reach_playoffs").head(8)
    fav = _favorite_path(agg_dir)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(champs["display_name"], champs["probability"] * 100, color=BAR_COLOR)
    ax1.invert_yaxis()
    ax1.set_title("Top 8 — Championship Probability")
    ax1.set_xlabel("%")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(playoffs["display_name"], playoffs["probability"] * 100, color=ACCENT_COLOR)
    ax2.invert_yaxis()
    ax2.set_title("Top 8 — Playoff Qualification Probability")
    ax2.set_xlabel("%")

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    text = (f"IEM Cologne Major 2026 — Pre-Event Simulation Summary\n\n"
            f"Model: series_random_forest_v2 (RF V2, frozen)   |   Prediction mode: pre_veto\n"
            f"Simulations: 50,000 complete Majors   |   Model-favorite path champion: {fav['champion']}\n\n"
            f"COLOGNE RESULTS = UNOPENED.  This is what the frozen model believed BEFORE the event.")
    ax3.text(0.02, 0.6, text, fontsize=11, family="monospace", transform=ax3.transAxes, va="top")

    fig.suptitle("Phase 8D — Pre-Event Cologne Simulation", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "12_presentation_summary")


FIGURE_FUNCTIONS = [
    fig_championship_probability_all, fig_championship_probability_top16,
    fig_playoff_qualification_probability, fig_final_probability, fig_stage_progression_matrix,
    fig_advancement_by_starting_stage, fig_30_probability, fig_03_probability,
    fig_playoff_seed_heatmap, fig_favorite_bracket, fig_monte_carlo_funnel_top_teams,
    fig_presentation_summary,
]


def generate_all_figures(agg_dir, out_dir):
    agg_dir, out_dir = Path(agg_dir), Path(out_dir)
    written = []
    for fn in FIGURE_FUNCTIONS:
        written.extend(fn(agg_dir, out_dir))
    return written
