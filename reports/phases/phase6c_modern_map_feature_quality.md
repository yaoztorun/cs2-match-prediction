# Phase 6C - Modern Selected-Map Feature Quality (descriptive)

**Scope discipline.** Every number below is computed on the **global TRAIN partition** (`data/modeling/map_split_v1.csv`), the same discipline every prior phase's quality report used. Validation is not summarized, the test partition is not opened, and Cologne is not read. **No feature-vs-target association is reported anywhere.**

TRAIN map rows: **7,762** of 10,318.

## 1. Coverage - team-level recent selected-map history

| quantity | n | pct_of_train |
| --- | --- | --- |
| both teams have recent selected-map history (both_teams_have_recent_selected_map_history==1) | 6502 | 83.77 |
| both teams have trusted-opponent-adjusted recent map history (map_adjusted_history_mass_min > 0) | 6490 | 83.61 |
| selected map present in BOTH teams' recent map-pool (selected_map_in_both_recent_pools==1) | 6351 | 81.82 |

## 2. Coverage - current-roster selected-map performance

| quantity | n | pct_of_train |
| --- | --- | --- |
| roster_map_players_with_history_min >= 1 (at least one evidenced player per side) | 6572 | 84.67 |
| roster_map_players_with_history_min >= 3 | 6382 | 82.22 |
| roster_map_players_with_history_min == 0 (cold start - the NaN gate) | 1190 | 15.33 |
| current_core_map_continuity_min > 0 (at least some prior-core overlap on this map, both sides) | 6340 | 81.68 |

## 3. Missingness (TRAIN)

Every NaN below is a documented cold-start contract, never a fabricated value - the four roster-map performance diffs are NaN exactly when `roster_map_players_with_history_min == 0`. No other new feature carries any NaN.

| feature | n_missing | pct_missing |
| --- | --- | --- |
| roster_map_mean_kast_diff | 1190 | 15.33 |
| roster_map_bottom_kast_diff | 1190 | 15.33 |
| roster_map_mean_adr_diff | 1190 | 15.33 |
| roster_map_mean_kd_balance_diff | 1190 | 15.33 |

## 4. History-mass distributions (TRAIN)

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| map_recent_history_mass_min | 3.6183 | 3.2193 | 0.0000 | 2.8290 | 18.3128 |
| map_adjusted_history_mass_min | 3.5877 | 3.2056 | 0.0000 | 2.8099 | 18.3128 |
| roster_map_history_mass_min | 3.4787 | 3.0393 | 0.0000 | 2.8163 | 18.0303 |
| current_core_map_continuity_min | 0.6899 | 0.3511 | 0.0000 | 0.8325 | 1.0000 |

## 5. Selected-map recency and specialization distributions (TRAIN)

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| time_weighted_map_wr_diff | 0.0077 | 0.1336 | -0.4554 | 0.0000 | 0.5658 |
| time_weighted_map_performance_residual_diff | 0.0064 | 0.3217 | -1.2532 | 0.0000 | 1.2591 |
| time_weighted_map_opponent_elo_diff | 10.5885 | 60.5761 | -253.5338 | 7.1650 | 504.8722 |
| selected_map_elo_vs_overall_diff | -15.9630 | 92.0413 | -494.8777 | -13.7216 | 346.2232 |
| selected_map_elo_vs_pool_mean_diff | 0.4625 | 55.2668 | -231.4760 | 0.0000 | 242.0137 |
| selected_map_wr_vs_pool_mean_diff | -0.0021 | 0.1274 | -0.4310 | 0.0000 | 0.4605 |
| selected_map_rank_percentile_diff | 0.0035 | 0.4660 | -1.0000 | 0.0000 | 1.0000 |
| roster_map_kast_specialization_diff | 0.1155 | 5.0559 | -26.1735 | 0.0000 | 39.1058 |
| current_core_map_continuity_diff | 0.0322 | 0.3691 | -1.0000 | 0.0000 | 1.0000 |

## 6. Correlation among the new features (TRAIN, descriptive only)

| feature | time_weighted_map_wr_diff | selected_map_elo_vs_overall_diff | selected_map_elo_vs_pool_mean_diff | roster_map_mean_kast_diff | roster_map_kast_specialization_diff | current_core_map_continuity_diff |
| --- | --- | --- | --- | --- | --- | --- |
| time_weighted_map_wr_diff | 1.00 | 0.26 | 0.81 | 0.59 | 0.52 | 0.05 |
| selected_map_elo_vs_overall_diff | 0.26 | 1.00 | 0.56 | 0.09 | 0.26 | -0.10 |
| selected_map_elo_vs_pool_mean_diff | 0.81 | 0.56 | 1.00 | 0.41 | 0.44 | 0.03 |
| roster_map_mean_kast_diff | 0.59 | 0.09 | 0.41 | 1.00 | 0.85 | 0.06 |
| roster_map_kast_specialization_diff | 0.52 | 0.26 | 0.44 | 0.85 | 1.00 | 0.02 |
| current_core_map_continuity_diff | 0.05 | -0.10 | 0.03 | 0.06 | 0.02 | 1.00 |

Descriptive only - no feature is added, removed or reweighted based on this table.

## 7. Map-by-map coverage (TRAIN)

| map_name | rows | both_recent_history_pct | both_pool_membership_pct | roster_map_coverage_pct |
| --- | --- | --- | --- | --- |
| Ancient | 1417 | 86.45 | 84.40 | 86.73 |
| Mirage | 1258 | 86.17 | 84.26 | 86.57 |
| Nuke | 1131 | 85.85 | 83.73 | 86.91 |
| Anubis | 1088 | 84.83 | 83.46 | 85.66 |
| Inferno | 981 | 85.12 | 83.59 | 86.65 |
| Dust2 | 793 | 81.84 | 80.08 | 81.84 |
| Vertigo | 532 | 79.32 | 78.01 | 81.77 |
| Overpass | 339 | 71.09 | 64.60 | 74.34 |
| Train | 223 | 68.16 | 67.71 | 68.61 |

## 8. What this report does not claim

Nothing here says any feature is useful. Coverage, spread and completeness are properties of the data; predictive value is assessed separately, TRAIN-only, in `scripts/evaluate_map_feature_sets_v3.py` (Stage B) - never here.
