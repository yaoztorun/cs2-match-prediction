# Presentation-only figures, re-plotted from frozen report/table numbers.
# Light academic theme to match the slide design. No model is loaded, no data
# is re-evaluated: every number is transcribed from the frozen reports listed
# in reports/presentation/presentation_sources.md.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0F2A43"
BLUE = "#1D6FB8"
CYAN = "#4DA3C7"
LIGHTBLUE = "#A8C8E4"
GRAY = "#5B6B7C"
GRID = "#D8E2EC"
ORANGE = "#D9772A"
GREEN = "#3E8E6B"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "text.color": NAVY,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# 1. Pre-veto tuned model comparison (validation, 1,419 series)
#    Source: reports/model_comparison_tuned_v1features.md
# ---------------------------------------------------------------------------
models = ["Logistic Regression\n(LR V2)", "Random Forest\n(RF V2)", "XGBoost\n(XGB V2)"]
acc = [0.6131, 0.6068, 0.6117]
auc = [0.6412, 0.6566, 0.6504]
logloss = [0.6581, 0.6514, 0.6542]
brier = [0.2329, 0.2298, 0.2311]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
x = np.arange(3)
w = 0.34

ax = axes[0]
b1 = ax.bar(x - w / 2, acc, w, color=LIGHTBLUE, label="Accuracy")
b2 = ax.bar(x + w / 2, auc, w, color=BLUE, label="ROC-AUC")
baseline_line = ax.axhline(0.5532, color=GRAY, ls="--", lw=1.4)
ax.set_ylim(0.50, 0.72)
ax.set_xticks(x, models, fontsize=12)
ax.set_title("Discrimination  (higher = better)", fontsize=13.5, fontweight="bold", pad=10)
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.3f", fontsize=11, padding=2)
ax.legend([b1, b2, baseline_line],
          ["Accuracy", "ROC-AUC", "majority baseline (accuracy 0.553)"],
          fontsize=10.5, frameon=False, loc="upper left")
ax.margins(x=0.02)

ax = axes[1]
b1 = ax.bar(x - w / 2, logloss, w, color=CYAN, label="Log Loss")
ax.set_ylim(0.60, 0.70)
ax.set_xticks(x, models, fontsize=12)
ax.set_title("Probability quality  (lower = better)", fontsize=13.5, fontweight="bold", pad=10)
ax.bar_label(b1, fmt="%.3f", fontsize=11, padding=2)
ax2 = ax.twinx()
b2 = ax2.bar(x + w / 2, brier, w, color=NAVY, label="Brier")
ax2.set_ylim(0.20, 0.26)
ax2.bar_label(b2, fmt="%.3f", fontsize=11, padding=2)
ax2.grid(False)
ax2.tick_params(colors=NAVY)
handles = [b1, b2]
ax.legend(handles, ["Log Loss (left axis)", "Brier (right axis)"], fontsize=11, frameon=False, loc="upper right")

fig.suptitle(
    "Tuned pre-veto models — held-out chronological validation (1,419 series)",
    fontsize=15, fontweight="bold", y=1.04,
)
save(fig, "preveto_model_comparison.png")

# ---------------------------------------------------------------------------
# 2. Pre-event championship probabilities, top 8 + Falcons highlight
#    Source: reports/phase8d_cologne_pre_event_simulation.md section L
# ---------------------------------------------------------------------------
teams = ["Team Vitality", "Team Spirit", "Natus Vincere", "Team Falcons",
         "MOUZ", "FURIA", "The MongolZ", "Aurora Gaming"]
probs = [29.70, 18.95, 11.42, 8.93, 7.75, 3.72, 3.50, 2.72]
colors = [BLUE] * 8
colors[3] = ORANGE

fig, ax = plt.subplots(figsize=(11.8, 4.9))
y = np.arange(len(teams))[::-1]
bars = ax.barh(y, probs, color=colors, height=0.62)
ax.set_yticks(y, teams, fontsize=13)
ax.set_xlabel("Pre-event championship probability (%)  —  50,000 simulated Majors", fontsize=12)
ax.set_xlim(0, 34)
for yi, p in zip(y, probs):
    ax.text(p + 0.5, yi, f"{p:.1f}%", va="center", fontsize=12.5,
            fontweight="bold" if p == 8.93 else "normal",
            color=ORANGE if p == 8.93 else NAVY)
ax.axvline(3.125, color=GRAY, ls="--", lw=1.4)
ax.text(12.3, 1.2, "dashed line: uniform 32-team reference (1/32 = 3.1%)",
        fontsize=11, color=GRAY, va="center")
ax.text(12.3, y[4], "actual champion — rank 4 of 32", fontsize=12.5, color=ORANGE,
        va="center", fontweight="bold")
ax.set_title("Frozen pre-event forecast vs. what actually happened",
             fontsize=15, fontweight="bold", pad=12)
ax.grid(axis="y", visible=False)
save(fig, "cologne_champion_top8.png")

# ---------------------------------------------------------------------------
# 3. Cologne external metrics vs uninformed baseline
#    Source: reports/phase8e_cologne_simulation_vs_reality.md section F
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))
metrics = ["Log Loss", "Brier"]
model_v = [0.6316, 0.2208]
base_v = [0.6931, 0.2500]

ax = axes[0]
x = np.arange(2)
w = 0.32
b1 = ax.bar(x - w / 2, model_v, w, color=BLUE, label="Frozen RF V2")
b2 = ax.bar(x + w / 2, base_v, w, color=LIGHTBLUE, label="Constant p = 0.5")
ax.set_xticks(x, metrics, fontsize=13)
ax.set_title("Probability quality  (lower = better)", fontsize=13, fontweight="bold", pad=10)
ax.bar_label(b1, fmt="%.3f", fontsize=11.5, padding=2)
ax.bar_label(b2, fmt="%.3f", fontsize=11.5, padding=2)
ax.set_ylim(0, 0.80)
ax.legend(fontsize=11, frameon=False)

ax = axes[1]
metrics2 = ["Accuracy", "ROC-AUC"]
model_v2 = [0.642, 0.697]  # 0.6415 / 0.6968 rounded for display
base_v2 = [0.5283, 0.5000]
b1 = ax.bar(x - w / 2, model_v2, w, color=BLUE, label="Frozen RF V2")
b2 = ax.bar(x + w / 2, base_v2, w, color=LIGHTBLUE, label="Uninformed baseline")
ax.set_xticks(x, metrics2, fontsize=13)
ax.set_title("Discrimination  (higher = better)", fontsize=13, fontweight="bold", pad=10)
ax.bar_label(b1, fmt="%.3f", fontsize=11.5, padding=2)
ax.bar_label(b2, fmt="%.3f", fontsize=11.5, padding=2)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=11, frameon=False)

fig.suptitle("IEM Cologne Major 2026 — 106 official matches, frozen pre-event model",
             fontsize=14.5, fontweight="bold", y=1.05)
save(fig, "cologne_match_metrics.png")

# ---------------------------------------------------------------------------
# 4. Known-map model vs baselines on the sealed TEST (1,427 maps)
#    Source: reports/phase7_internal_test_results.md section 2
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.6, 3.7))
names = ["Constant p = 0.5", "Map-only ELO", "Overall ELO", "XGB V3 (final)"]
lls = [0.6931, 0.6786, 0.6549, 0.6521]
aucs = [0.5000, 0.5863, 0.6477, 0.6489]
cols = [LIGHTBLUE, LIGHTBLUE, CYAN, BLUE]
y = np.arange(len(names))
bars = ax.barh(y, lls, color=cols, height=0.58)
ax.set_yticks(y, names, fontsize=13)
ax.invert_yaxis()
ax.set_xlim(0.60, 0.70)
ax.set_xlabel("Log Loss on sealed TEST maps (lower = better)", fontsize=12)
for yi, (ll, au) in enumerate(zip(lls, aucs)):
    ax.text(ll + 0.0008, yi, f"LL {ll:.3f}   ·   AUC {au:.3f}", va="center", fontsize=12,
            fontweight="bold" if yi == 3 else "normal")
ax.set_title("Known-map model vs. baselines — sealed internal TEST (1,427 maps)",
             fontsize=14.5, fontweight="bold", pad=12)
ax.grid(axis="y", visible=False)
save(fig, "knownmap_test_vs_baselines.png")

print("all presentation figures written")
