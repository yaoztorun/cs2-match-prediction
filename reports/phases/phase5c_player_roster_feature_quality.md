# Phase 5C - Player / Roster Feature Quality (descriptive)

**Scope discipline.** Every number below is computed on the **global TRAIN partition** (`data/modeling/series_split_v1.csv`), the same discipline `scripts/phase5a_reports.py` and `phase5b2_reports.py` established. Validation is not summarized, the test partition is not opened, and Cologne is not read. **No feature-vs-target association is reported** - no correlations with the target, no rankings, no predictive-performance metric, no tuning of the 90/60-day constants.

TRAIN series: **6,619** of 9,456.

## 1. Player identity and stat coverage (development stream)

| quantity | n |
| --- | --- |
| distinct persistent player_ids | 1214 |
| distinct maps contributing player observations | 10405 |
| distinct matches contributing player observations | 4997 |
| player-slot observations after cleaning | 101097 |
| observations with a usable box score | 99770 |
| observations with an id but no usable box score | 1327 |
| maps excluded (same player on both sides) | 5 |
| duplicate player-slot groups collapsed | 14 |
| duplicate player-slot groups excluded (conflicting stats) | 0 |
| map rows whose map timestamp differs from the authoritative series start | 0 |

Usable-box-score rate among retained player observations: **98.69%**.

## 2. Series-level player coverage

| quantity | n | pct_of_v4 |
| --- | --- | --- |
| V4 series with their own player rows | 4941 | 52.25 |
| V4 series with NO player rows of their own | 4515 | 47.75 |

A series without player rows of its own still receives roster features - they describe the two teams' PRIOR history, not the target match.

## 3. Inferred-roster completeness and cold start (TRAIN)

| quantity | n | pct_of_train |
| --- | --- | --- |
| both inferred rosters contain five players | 4307 | 65.07 |
| roster_size_min == 0 (>=1 side has no inferred roster at all) | 1927 | 29.11 |
| roster_form_players_min == 0 (>=1 side has no usable player history) -> NaN performance | 1931 | 29.17 |
| roster_form_players_min >= 5 (both sides fully evidenced) | 4207 | 63.56 |
| roster_min_player_history_mass == 0 (>=1 inferred player with no evidence) | 2051 | 30.99 |

**`roster_size_min` and `roster_form_players_min` are different quantities**: the gap between the two rows above is exactly the population of series where a five-player lineup could be inferred but some of those players have no usable prior box score.

## 4. Player-form distributions (TRAIN, per-side inferred-roster aggregates)

Descriptive only - to confirm nothing is degenerate or absurdly scaled. `adr` and `kast` are the source's own round-normalized rates (KAST on 0-100); `kd_balance` is bounded in [-1, 1].

| quantity | n_defined | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- | --- |
| roster mean ADR | 5347 | 73.2536 | 2.9477 | 47.1500 | 73.3532 | 95.0000 |
| roster top ADR | 5347 | 81.7769 | 5.6502 | 51.7000 | 80.9645 | 122.2000 |
| roster bottom ADR | 5347 | 65.1561 | 4.7861 | 27.5000 | 65.8778 | 90.4667 |
| roster mean KAST | 5347 | 71.9235 | 2.9619 | 40.6500 | 72.0729 | 93.7500 |
| roster mean KD-balance | 5347 | -0.0113 | 0.0513 | -0.3864 | -0.0091 | 0.3028 |
| roster mean assists/round | 5347 | 0.2269 | 0.0215 | 0.0000 | 0.2270 | 0.4688 |
| roster mean player history mass | 6619 | 17.6118 | 14.5483 | 0.0000 | 16.1955 | 63.9536 |
| roster min player history mass | 6619 | 14.9767 | 13.6841 | 0.0000 | 12.0544 | 63.1730 |

## 5. Roster-stability distributions (TRAIN, team1 side)

| quantity | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| unique players in last 10 maps | 4.4209 | 2.1965 | 0.0000 | 5.0000 | 11.0000 |
| unique players in last 20 maps | 4.7093 | 2.4640 | 0.0000 | 5.0000 | 15.0000 |
| core-5 appearance concentration (90d) | 0.7845 | 0.3860 | 0.0000 | 1.0000 | 1.0000 |
| core-5 continuity over last 10 maps | 0.7737 | 0.3865 | 0.0000 | 1.0000 | 1.0000 |
| inferred roster size | 3.9875 | 1.9735 | 0.0000 | 5.0000 | 5.0000 |

A perfectly settled five-player roster scores 1.0 on both concentration and continuity; frequent stand-ins or turnover push both down.

## 6. Observed player mobility (team changes)

In the strictly pre-Cologne state: **628** of 1,187 tracked players (52.9%) were observed playing for more than one canonical team - i.e. transfers are directly measurable, which is exactly why global player form is tracked separately from team membership.

## 7. Missingness of the new features (TRAIN)

| feature | n_missing | pct_missing |
| --- | --- | --- |
| roster_mean_adr_diff | 1931 | 29.17 |
| roster_top_adr_diff | 1931 | 29.17 |
| roster_bottom_adr_diff | 1931 | 29.17 |
| roster_mean_kast_diff | 1931 | 29.17 |
| roster_top_kast_diff | 1931 | 29.17 |
| roster_bottom_kast_diff | 1931 | 29.17 |
| roster_mean_kd_balance_diff | 1931 | 29.17 |
| roster_top_kd_balance_diff | 1931 | 29.17 |
| roster_bottom_kd_balance_diff | 1931 | 29.17 |
| roster_mean_assists_per_round_diff | 1931 | 29.17 |

Exactly the 10 performance diffs carry NaN, and only where `roster_form_players_min == 0` - the documented cold-start contract, asserted at build time and re-checked by `scripts/validate_phase5c.py`. The remaining 11 features are always defined.

## 8. Distribution summaries of the new features (TRAIN)

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| roster_mean_adr_diff | 0.2856 | 4.1865 | -30.5300 | 0.3005 | 38.5303 |
| roster_top_adr_diff | 0.1236 | 8.0965 | -49.0465 | 0.1499 | 63.3913 |
| roster_bottom_adr_diff | 0.3861 | 6.4873 | -34.6467 | 0.2839 | 51.6000 |
| roster_mean_kast_diff | 0.3126 | 4.0212 | -29.2500 | 0.2989 | 35.8607 |
| roster_top_kast_diff | 0.1579 | 4.6263 | -29.5224 | 0.1856 | 38.4427 |
| roster_bottom_kast_diff | 0.4513 | 4.8577 | -33.3500 | 0.3628 | 46.4275 |
| roster_mean_kd_balance_diff | 0.0059 | 0.0699 | -0.4423 | 0.0040 | 0.8724 |
| roster_top_kd_balance_diff | 0.0047 | 0.0910 | -0.6239 | 0.0043 | 0.8724 |
| roster_bottom_kd_balance_diff | 0.0062 | 0.0911 | -0.5345 | 0.0054 | 0.8724 |
| roster_mean_assists_per_round_diff | 0.0018 | 0.0294 | -0.4118 | 0.0019 | 0.1988 |
| recent_unique_players_10_maps_diff | 0.1423 | 2.3799 | -11.0000 | 0.0000 | 10.0000 |
| recent_unique_players_20_maps_diff | 0.1357 | 2.7542 | -12.0000 | 0.0000 | 12.0000 |
| core5_appearance_concentration_90d_diff | 0.0353 | 0.3977 | -1.0000 | 0.0000 | 1.0000 |
| core5_continuity_last_10_diff | 0.0406 | 0.3997 | -1.0000 | 0.0000 | 1.0000 |
| roster_mean_player_history_mass_diff | 2.4101 | 13.3142 | -58.1626 | 0.1632 | 53.8947 |
| roster_size_min | 3.4448 | 2.2566 | 0.0000 | 5.0000 | 5.0000 |
| both_teams_have_5_inferred_players | 0.6507 | 0.4768 | 0.0000 | 1.0000 | 1.0000 |
| roster_min_player_history_mass | 9.2206 | 10.5584 | 0.0000 | 5.3771 | 51.4607 |
| roster_core_concentration_min | 0.6718 | 0.4350 | 0.0000 | 0.9262 | 1.0000 |
| roster_core_continuity_last10_min | 0.6510 | 0.4305 | 0.0000 | 0.9000 | 1.0000 |
| roster_form_players_min | 3.4230 | 2.2520 | 0.0000 | 5.0000 | 5.0000 |

## 9. Feature-feature redundancy (descriptive, NOT target correlation)

Correlation among the 15 new directional features only, to describe overlap without ever touching the target:

Top 12 |corr| pairs among the new directional features:

| feature A | feature B | r |
|---|---|---|
| core5_appearance_concentration_90d_diff | core5_continuity_last_10_diff | +0.965 |
| recent_unique_players_10_maps_diff | recent_unique_players_20_maps_diff | +0.923 |
| roster_mean_adr_diff | roster_mean_kd_balance_diff | +0.871 |
| roster_mean_kast_diff | roster_mean_kd_balance_diff | +0.871 |
| roster_mean_kast_diff | roster_top_kast_diff | +0.870 |
| roster_mean_kast_diff | roster_bottom_kast_diff | +0.849 |
| roster_mean_kd_balance_diff | roster_top_kd_balance_diff | +0.811 |
| roster_mean_kd_balance_diff | roster_bottom_kd_balance_diff | +0.765 |
| roster_mean_adr_diff | roster_mean_kast_diff | +0.763 |
| roster_top_kast_diff | roster_mean_kd_balance_diff | +0.745 |
| roster_bottom_kast_diff | roster_mean_kd_balance_diff | +0.743 |
| roster_top_kast_diff | roster_top_kd_balance_diff | +0.735 |

## 10. What this report does not claim

Nothing here says any feature is useful. Coverage, spread and completeness are properties of the data; predictive value is a property of a model that has not been fitted yet. A separate paired V3-vs-V4 ablation will measure that, and V4 will not be modified in response to it.
