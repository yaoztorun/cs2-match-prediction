# Phase 5B.2 - Team Form Feature Quality (descriptive)

**Scope discipline.** Every number below is computed on the **global TRAIN partition** (`data/modeling/series_split_v1.csv`), the same discipline `scripts/phase5a_reports.py` established. Validation is not summarized, the test partition is not opened, and Cologne is not read. **No feature-vs-target association is reported** - no correlations, no rankings, no "promising feature" claims, no half-life tuning discussion.

TRAIN series: **6,619** of 9,456.

## 1. Missingness

New features with any missing value: **0 of 12** columns (all 12 new features are always defined, by cold-start construction).

## 2. Coverage and cold start (TRAIN series)

| quantity | n | pct_of_train |
| --- | --- | --- |
| series with opponent_adjusted_history_min == 0 (>=1 side has zero trusted opponent history) | 473 | 7.15 |
| series where both teams have >= 5 trusted opponent-adjusted matches | 4967 | 75.04 |
| series where both teams have >= 10 trusted opponent-adjusted matches | 4262 | 64.39 |
| series where time_weighted_history_mass_min == 0 (>=1 side has zero own eligible history) | 463 | 7.00 |

## 3. Time-weight behavior (illustrative, not data-derived)

`weight = 0.5 ** (age_days / 60)` - the fixed 60-day half-life, shown for representative ages:

| age_days | weight |
| --- | --- |
| 0 | 1.0000 |
| 30 | 0.7071 |
| 60 | 0.5000 |
| 90 | 0.3536 |
| 120 | 0.2500 |
| 180 | 0.1250 |
| 365 | 0.0147 |

## 4. Distribution summaries (TRAIN series)

Descriptive only - spread and centring, to confirm nothing is degenerate or absurdly scaled.

| feature | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| avg_opponent_elo_last_5_diff | 11.8231 | 56.1551 | -359.3695 | 7.2595 | 343.6727 |
| avg_opponent_elo_last_10_diff | 12.0834 | 49.5628 | -275.9418 | 7.6713 | 280.8113 |
| performance_residual_last_5_diff | 0.0059 | 0.2856 | -1.1498 | 0.0000 | 1.0514 |
| performance_residual_last_10_diff | 0.0159 | 0.2358 | -1.0281 | 0.0029 | 1.0000 |
| performance_residual_all_diff | 0.0226 | 0.1815 | -1.0281 | 0.0097 | 1.0000 |
| time_weighted_win_rate_diff | 0.0245 | 0.2125 | -1.0000 | 0.0082 | 1.0000 |
| time_weighted_performance_residual_diff | 0.0176 | 0.1949 | -1.0281 | 0.0039 | 1.0000 |
| time_weighted_series_margin_diff | 0.0450 | 0.3817 | -2.0000 | 0.0219 | 2.0000 |
| opponent_adjusted_history_min | 34.7823 | 37.2797 | 0.0000 | 21.0000 | 181.0000 |
| both_teams_have_5_adjusted_matches | 0.7504 | 0.4328 | 0.0000 | 1.0000 | 1.0000 |
| both_teams_have_10_adjusted_matches | 0.6439 | 0.4789 | 0.0000 | 1.0000 | 1.0000 |
| time_weighted_history_mass_min | 9.8046 | 7.3550 | 0.0000 | 9.1098 | 36.3450 |

## 5. Feature-feature redundancy (descriptive, NOT target correlation)

Correlation among the 8 new directional features only, to describe overlap without ever touching the target:

Top 10 |corr| pairs among the new directional features:

| feature A | feature B | r |
|---|---|---|
| time_weighted_win_rate_diff | time_weighted_series_margin_diff | +0.929 |
| performance_residual_all_diff | time_weighted_performance_residual_diff | +0.898 |
| time_weighted_win_rate_diff | time_weighted_performance_residual_diff | +0.894 |
| performance_residual_all_diff | time_weighted_win_rate_diff | +0.890 |
| performance_residual_last_10_diff | time_weighted_performance_residual_diff | +0.851 |
| time_weighted_performance_residual_diff | time_weighted_series_margin_diff | +0.829 |
| performance_residual_all_diff | time_weighted_series_margin_diff | +0.829 |
| avg_opponent_elo_last_5_diff | avg_opponent_elo_last_10_diff | +0.814 |
| performance_residual_last_5_diff | performance_residual_last_10_diff | +0.768 |
| performance_residual_last_10_diff | performance_residual_all_diff | +0.759 |

## 6. What this report does not claim

Nothing here says any feature is useful. Coverage, spread and completeness are properties of the data; predictive value is a property of a model that has not been fitted yet.
