# Phase 5A - Map Feature Quality (descriptive)

**Scope discipline.** Every number below is computed on the **global TRAIN partition** (`data/modeling/series_split_v1.csv`). Validation is not summarized, the test partition is not opened, and Cologne 2026 is not read. **No feature-vs-target association is reported** - no correlations, no rankings, no "promising feature" claims. Those are model-selection decisions; making them here would spend evidence that later phases need and would quietly turn a descriptive report into feature selection on data the model has not earned.

TRAIN series: **6,619** of 9,456. TRAIN map rows: **7,762** of 10,318.

## 1. Completeness

- Pre-veto pool features with any missing value: **0 of 30** columns.

- Map-specific features with any missing value: **1 of 14** columns.


| feature | n_missing | pct |
| --- | --- | --- |
| days_since_map_played_diff | 1260 | 16.23 |

These are the documented `days_since_*` cold-start NaNs - a team that has never played the map has no "days since" value, and a sentinel number would be a lie. They are genuinely missing, not corrupt.


- All pre-veto pool features finite: **True**.

## 2. Map-history coverage and cold start

| quantity | n | pct of TRAIN series |
| --- | --- | --- |
| series with an empty recent pool for at least one side | 1726 | 26.08 |
| series with a completely empty union pool (both sides cold) | 763 | 11.53 |
| series where both teams have >= 3 recent maps | 4464 | 67.44 |
| series where both teams have >= 5 experienced maps | 2159 | 32.62 |
| series with zero shared recent maps | 1803 | 27.24 |

The cold-start share is a direct consequence of map coverage starting about nine months after series coverage; it is concentrated in the earliest part of the timeline, which the chronological split places entirely inside TRAIN.

### Recent-pool-size distribution (TRAIN series, orientation-independent measures)

| statistic | map_pool_size_min | union_map_count | shared_recent_map_count |
| --- | --- | --- | --- |
| min | 0.00 | 0.00 | 0.00 |
| median | 6.00 | 7.00 | 5.00 |
| mean | 4.05 | 6.03 | 3.69 |
| max | 8.00 | 8.00 | 8.00 |

## 3. Map-specific coverage (TRAIN map rows)

| quantity | n | pct |
| --- | --- | --- |
| rows where both teams have prior history on this map | 6502 | 83.77 |
| rows where both teams have >= 5 prior maps here | 4191 | 53.99 |
| rows where both teams have >= 10 prior maps here | 2730 | 35.17 |
| rows where at least one side is cold on this map | 1260 | 16.23 |

## 4. Distribution summaries

Descriptive only - spread and centring, to confirm nothing is degenerate or absurdly scaled.

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| map_pool_size_diff | 0.379 | 2.607 | -8.000 | 0.000 | 8.000 |
| map_pool_total_matches_diff | 9.578 | 59.714 | -265.000 | 1.000 | 265.000 |
| map_pool_experienced_maps_diff | 0.460 | 2.724 | -8.000 | 0.000 | 8.000 |
| map_pool_mean_elo_diff | 4.152 | 27.956 | -165.726 | 0.000 | 180.202 |
| map_pool_best_elo_diff | 8.041 | 48.989 | -226.614 | 0.736 | 226.885 |
| map_pool_second_best_elo_diff | 6.445 | 37.547 | -189.918 | 0.000 | 240.307 |
| map_pool_third_best_elo_diff | 5.280 | 33.303 | -192.942 | 0.000 | 248.331 |
| map_pool_worst_elo_diff | 0.788 | 39.848 | -193.081 | 0.000 | 190.363 |
| map_pool_mean_smoothed_wr_diff | 0.008 | 0.071 | -0.261 | 0.000 | 0.255 |
| map_pool_best_smoothed_wr_diff | 0.013 | 0.107 | -0.378 | 0.000 | 0.446 |
| map_pool_second_best_smoothed_wr_diff | 0.011 | 0.091 | -0.389 | 0.000 | 0.381 |
| map_pool_third_best_smoothed_wr_diff | 0.010 | 0.086 | -0.324 | 0.000 | 0.350 |
| map_pool_worst_smoothed_wr_diff | 0.003 | 0.113 | -0.400 | 0.000 | 0.401 |
| map_pool_mean_normalized_margin_diff | 0.009 | 0.143 | -0.924 | 0.000 | 1.039 |
| map_matchup_mean_elo_advantage | 3.605 | 24.601 | -143.079 | 0.000 | 177.324 |
| map_matchup_median_elo_advantage | 3.018 | 26.225 | -126.992 | 0.000 | 195.879 |
| map_matchup_midrange_elo_advantage | 4.260 | 29.142 | -150.773 | 0.000 | 175.301 |
| map_matchup_positive_advantage_balance | 0.302 | 2.752 | -8.000 | 0.000 | 8.000 |
| map_matchup_mean_smoothed_wr_advantage | 0.006 | 0.060 | -0.208 | 0.000 | 0.244 |
| map_matchup_median_smoothed_wr_advantage | 0.006 | 0.070 | -0.292 | 0.000 | 0.298 |
| map_pool_size_min | 4.045 | 2.780 | 0.000 | 6.000 | 8.000 |
| map_pool_total_matches_min | 41.437 | 55.774 | 0.000 | 16.000 | 275.000 |
| both_teams_have_map_pool_history | 0.739 | 0.439 | 0.000 | 1.000 | 1.000 |
| both_teams_have_3_recent_maps | 0.674 | 0.469 | 0.000 | 1.000 | 1.000 |
| both_teams_have_5_experienced_maps | 0.326 | 0.469 | 0.000 | 0.000 | 1.000 |
| union_map_count | 6.033 | 2.526 | 0.000 | 7.000 | 8.000 |
| shared_recent_map_count | 3.690 | 2.651 | 0.000 | 5.000 | 8.000 |
| shared_experienced_map_count | 2.044 | 2.381 | 0.000 | 0.000 | 8.000 |
| map_matchup_shared_coverage | 0.512 | 0.358 | 0.000 | 0.625 | 1.000 |
| map_matchup_elo_advantage_range | 117.150 | 79.496 | 0.000 | 112.631 | 426.022 |

**These directional means are NOT zero, and that is expected.** `map_pool_size_diff` averages 0.379 and `map_pool_total_matches_diff` averages 9.578; the Team1 side has the larger recent pool in 36.0% of TRAIN series. This is the same Team1 orientation artifact documented in Phase 2 (`reports/orientation_analysis.md`) - the export tends to list the better-established team first, so a feature that measures establishment inherits the tilt. It is a property of the raw rows, not a defect in the features: the features are antisymmetric *by construction* (swapping the sides negates each of them exactly, proven in `TestG_SideSwapSymmetry`), and mirrored augmentation neutralizes the offset at training time by adding the swapped copy of every row. No de-biasing is applied here, because these parquets store raw pre-mirroring rows - mirroring is a training-time step that must happen after the train/validation split, never inside feature engineering.


### Map-specific features

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| map_elo_diff | 6.142 | 65.257 | -273.585 | 0.736 | 320.240 |
| map_smoothed_win_rate_diff | 0.010 | 0.149 | -0.550 | 0.000 | 0.636 |
| map_win_rate_last_5_diff | 0.016 | 0.366 | -1.000 | 0.000 | 1.000 |
| map_win_rate_last_10_diff | 0.016 | 0.327 | -1.000 | 0.000 | 1.000 |
| map_normalized_margin_all_diff | 0.016 | 0.213 | -1.059 | 0.005 | 1.765 |
| map_normalized_margin_last_5_diff | 0.016 | 0.251 | -1.059 | 0.000 | 1.765 |
| map_normalized_margin_last_10_diff | 0.015 | 0.227 | -1.059 | 0.001 | 1.765 |
| map_matches_before_diff | 1.755 | 12.496 | -63.000 | 1.000 | 76.000 |
| days_since_map_played_diff | -2.988 | 52.679 | -552.125 | 0.028 | 616.708 |
| map_matches_before_min | 9.309 | 10.666 | 0.000 | 5.000 | 63.000 |
| map_matches_before_sum | 27.207 | 24.238 | 0.000 | 20.000 | 129.000 |
| both_teams_have_map_history | 0.838 | 0.369 | 0.000 | 1.000 | 1.000 |
| both_teams_have_5_map_matches | 0.540 | 0.498 | 0.000 | 1.000 | 1.000 |
| both_teams_have_10_map_matches | 0.352 | 0.478 | 0.000 | 0.000 | 1.000 |

## 5. Per-map coverage

Full detail in `reports/tables/map_feature_coverage_v1.csv`.

| map_name | rows_all | rows_train | teams_seen | team1_cold_start | team2_cold_start | both_have_history | pct_both_have_history |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ancient | 1806 | 1417 | 287 | 123 | 165 | 1571 | 86.99 |
| Anubis | 1200 | 1088 | 233 | 89 | 144 | 1010 | 84.17 |
| Dust2 | 1296 | 793 | 236 | 107 | 128 | 1102 | 85.03 |
| Inferno | 1315 | 981 | 215 | 97 | 119 | 1141 | 86.77 |
| Mirage | 1741 | 1258 | 269 | 104 | 166 | 1513 | 86.90 |
| Nuke | 1507 | 1131 | 244 | 102 | 140 | 1300 | 86.26 |
| Overpass | 588 | 339 | 169 | 79 | 96 | 447 | 76.02 |
| Train | 333 | 223 | 115 | 48 | 67 | 242 | 72.67 |
| Vertigo | 532 | 532 | 143 | 55 | 86 | 422 | 79.32 |

## 6. Teams with map history in the frozen pre-Cologne snapshot

- Team-map states: **1,911**
- Distinct teams: **382**
- Distinct maps: **9**
- Median maps played per team-map state: **3**
- Team-map states with >= 5 maps: **836** (43.7%)

- History entries recorded against an untrusted opponent: **113** - real own-team evidence that a rebuild from the training table would have discarded.


## 7. What this report does not claim

Nothing here says any feature is useful. Coverage, spread and completeness are properties of the data; predictive value is a property of a model that has not been fitted yet. The next phase must establish that separately, and only against TRAIN and validation.
