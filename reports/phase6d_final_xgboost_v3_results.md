# Phase 6D - Final XGBoost V3 Results

## TRAIN-only selected-model OOF development performance

**This is TRAIN-only selected-model OOF development performance, not an unbiased generalization estimate** - the model was selected using these same four folds. The sealed 1,427-map TEST partition is the next unbiased internal evaluation and remains closed.

n=6541, Accuracy=0.5901, Precision=0.6005, Recall=0.6835, F1=0.6393, ROC-AUC=0.6200, Log Loss=0.6682, Brier=0.2379

Series-macro: Log Loss=0.6631, Brier=0.2354, Accuracy=0.6028

Side-symmetry (TRAIN-only, per-fold models): mean=0.0109, median=0.0088, p95=0.0279, max=0.0778

## Three-way comparison (feature gain vs development tuning gain)

| arm | description | log loss | ROC-AUC | accuracy |
|---|---|---|---|---|
| A | V2 + frozen Phase 6B config (124 trees) | 0.6695 | 0.6164 | 0.5843 |
| B | V3 + frozen Phase 6B config (124 trees) | 0.6688 | 0.6183 | 0.5880 |
| C | V3 + Phase 6D final tuned config | 0.6682 | 0.6200 | 0.5901 |

**Feature gain (A -> B)**: Δlog loss -0.0007, ΔROC-AUC +0.0019, Δaccuracy +0.0037 pp

**Development tuning gain (B -> C)**: Δlog loss -0.0006, ΔROC-AUC +0.0017, Δaccuracy +0.0021 pp - these are NOT the same quantity and are never attributed to each other.

## Grouped permutation importance (TRAIN-only CV, requested families)

| family | label | n features | AUC decrease |
|---|---|---|---|
| A | original series V1 (ELO / win rate / activity) | 15 | +0.0078 |
| B | map-pool depth and order statistics | 14 | +0.0096 |
| E | opponent-strength / residual form | 5 | +0.0078 |
| H | player performance | 10 | +0.0097 |
| K | map-specific historical strength (selected map) | 14 | +0.0006 |
| M | recent/opponent-adjusted selected-map team features | 8 | +0.0042 |
| N | map specialization (relative to overall/pool strength) | 5 | -0.0004 |
| O | current-roster selected-map player performance | 9 | +0.0017 |
| P | current-core selected-map continuity | 3 | +0.0007 |

Interpretation only - no feature changes follow.

## Top 10 individual features (permutation importance)

| rank | feature | family | permutation | gain |
|---|---|---|---|---|
| 1 | elo_diff | A | +0.0063 | 47.1657 |
| 2 | roster_mean_kast_diff | H | +0.0047 | 20.1629 |
| 3 | avg_opponent_elo_last_10_diff | E | +0.0035 | 21.5706 |
| 4 | time_weighted_map_opponent_elo_diff | M | +0.0032 | 20.2244 |
| 5 | map_pool_best_elo_diff | B | +0.0029 | 29.8008 |
| 6 | map_pool_total_matches_diff | B | +0.0025 | 29.8494 |
| 7 | avg_opponent_elo_last_5_diff | E | +0.0015 | 14.0562 |
| 8 | roster_map_mean_history_mass_diff | O | +0.0011 | 15.7592 |
| 9 | roster_top_kd_balance_diff | H | +0.0011 | 11.6761 |
| 10 | time_weighted_series_margin_diff | F | +0.0009 | 20.8535 |

## Full-TRAIN refit

7,762 unique TRAIN maps (universe reconstructed from `data/modeling/map_cv_folds_v1.csv`, never `map_split_v1.csv`) -> 15,524 augmented observations. Future-inference parity check (synthetic matchup, never real Cologne data): p=0.5062, finite, in [0,1] - PASSED.

## Status

- **FEATURES = FROZEN**
- **PHASE-6B MAP VALIDATION = CONSUMED, NOT REUSED**
- **TEST = SEALED**
- **COLOGNE = UNTOUCHED**
- **THRESHOLD = 0.5**
- **NO CALIBRATION**
- **NO NEW ENSEMBLE**
- **NO POST-SELECTION RETUNING**
- **SRC = UNCHANGED**
