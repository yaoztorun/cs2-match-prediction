# Phase 6A - Known-Map Feature Quality (descriptive)

**Scope discipline.** Every number below is computed on the **global TRAIN partition** (`data/modeling/map_split_v1.csv`, itself derived from `series_split_v1.csv` so no match_id crosses a partition), the same discipline every prior phase's quality report used. Validation is not summarized, the test partition is not opened, and Cologne is not read. **No feature-vs-target association is reported anywhere.**

TRAIN map rows: **7,762** of 10,318.

## 1. Row count and map representation

- Total rows: **10,318** (identical to `map_features_v1.parquet`)
- Distinct matches: **4,952**
- Distinct maps represented: **9**

### Rows per map (imbalance)

| map_name | rows | pct_of_total |
| --- | --- | --- |
| Ancient | 1806 | 17.50 |
| Mirage | 1741 | 16.87 |
| Nuke | 1507 | 14.61 |
| Inferno | 1315 | 12.74 |
| Dust2 | 1296 | 12.56 |
| Anubis | 1200 | 11.63 |
| Overpass | 588 | 5.70 |
| Vertigo | 532 | 5.16 |
| Train | 333 | 3.23 |

Max/min row-count ratio across maps: **5.42x** (Ancient=1806 vs Train=333).

## 2. Coverage (TRAIN)

| quantity | n | pct_of_train |
| --- | --- | --- |
| both teams have prior history on this map (both_teams_have_map_history==1) | 6502 | 83.77 |
| both teams have >=5 prior matches on this map | 4191 | 53.99 |
| both teams have >=10 prior matches on this map | 2730 | 35.17 |
| both_teams_have_5_inferred_players (V4 roster) | 6623 | 85.33 |
| roster_form_players_min >= 5 (V4 player-performance evidence, both sides) | 6484 | 83.54 |
| roster_form_players_min == 0 (>=1 side has no usable player history) | 705 | 9.08 |

## 3. NaN rates on the new/inherited features (TRAIN)

Every NaN below is a documented cold-start contract, never a fabricated value - the two `days_since_*` features are NaN exactly when the corresponding side has zero prior history, and the ten roster-performance diffs are NaN exactly when `roster_form_players_min == 0`.

| feature | n_missing | pct_missing |
| --- | --- | --- |
| days_since_map_played_diff | 1260 | 16.23 |
| roster_mean_adr_diff | 705 | 9.08 |
| roster_top_adr_diff | 705 | 9.08 |
| roster_bottom_adr_diff | 705 | 9.08 |
| roster_mean_kast_diff | 705 | 9.08 |
| roster_top_kast_diff | 705 | 9.08 |
| roster_bottom_kast_diff | 705 | 9.08 |
| roster_mean_kd_balance_diff | 705 | 9.08 |
| roster_top_kd_balance_diff | 705 | 9.08 |
| roster_bottom_kd_balance_diff | 705 | 9.08 |
| roster_mean_assists_per_round_diff | 705 | 9.08 |
| days_since_last_match_diff | 132 | 1.70 |

## 4. Feature-family counts

| family | directional | symmetric |
| --- | --- | --- |
| map-specific (Phase 5A) | 9 | 5 |
| V2 map-pool depth (Phase 5A) | 14 | 0 |
| V2 same-map matchup (Phase 5A) | 6 | 0 |
| V2 map-pool confidence (Phase 5A) | 0 | 10 |
| V3 opponent-strength/residual form (Phase 5B.2) | 5 | 0 |
| V3 time-decayed form (Phase 5B.2) | 3 | 0 |
| V3 form confidence (Phase 5B.2) | 0 | 4 |
| V4 player performance (Phase 5C) | 10 | 0 |
| V4 roster stability (Phase 5C) | 4 | 0 |
| V4 roster evidence/confidence (Phase 5C) | 1 | 6 |
| inherited Phase 3 V1 series-level | 10 | 5 |

Total: **62 directional + 30 symmetric + 3 categorical context = 95 predictive inputs**.

## 5. Map-specific evidence by map (TRAIN)

| map_name | rows | both_history_pct | both5_pct | both10_pct |
| --- | --- | --- | --- | --- |
| Ancient | 1417 | 86.45 | 59.00 | 41.85 |
| Mirage | 1258 | 86.17 | 59.22 | 39.83 |
| Nuke | 1131 | 85.85 | 59.68 | 41.29 |
| Anubis | 1088 | 84.83 | 55.24 | 36.76 |
| Inferno | 981 | 85.12 | 57.90 | 38.94 |
| Dust2 | 793 | 81.84 | 51.45 | 32.91 |
| Vertigo | 532 | 79.32 | 42.48 | 19.17 |
| Overpass | 339 | 71.09 | 26.25 | 5.31 |
| Train | 223 | 68.16 | 19.28 | 2.69 |

## 6. Distribution summaries (TRAIN, new/inherited numeric features - sample)

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| map_elo_diff | 6.1417 | 65.2567 | -273.5851 | 0.7363 | 320.2399 |
| map_pool_best_elo_diff | 8.7017 | 53.2589 | -181.4249 | 7.1135 | 226.8852 |
| avg_opponent_elo_last_5_diff | 12.3971 | 61.9421 | -359.3695 | 9.1225 | 343.6727 |
| performance_residual_all_diff | 0.0200 | 0.1348 | -0.6764 | 0.0105 | 0.8810 |
| roster_mean_adr_diff | 0.3010 | 3.8952 | -20.7522 | 0.2799 | 38.5303 |
| roster_mean_kast_diff | 0.3188 | 3.6285 | -18.5103 | 0.2553 | 35.8607 |
| core5_continuity_last_10_diff | 0.0258 | 0.2775 | -1.0000 | 0.0000 | 1.0000 |
| roster_mean_player_history_mass_diff | 2.2884 | 13.4734 | -50.1622 | 1.3976 | 51.2072 |

## 7. What this report does not claim

Nothing here says any feature is useful. Coverage, spread and completeness are properties of the data; predictive value is a property of a model that has not been fitted yet - no model is trained in Phase 6A.
