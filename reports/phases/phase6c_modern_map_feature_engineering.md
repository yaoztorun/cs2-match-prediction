# Phase 6C - Modern Selected-Map Strength + Current-Roster Map Specialization: Feature Engineering

## Motivation

Phase 6B's feature-family grouped permutation importance showed the known-map model's exact selected-map historical family (K) contributing *less* than the broad strength/form families (B: map-pool depth, E: opponent-adjusted form, H: player performance). The model's map-specific signal was mostly stale organization-level history rather than something reflecting recent form, opponent quality, or the current roster's specific record on the map. Phase 6C adds 25 new features (18 directional + 7 symmetric/confidence) that are recent, opponent-adjusted, relative-to-general-strength, and current-roster-aware.

## New state: `scripts/modern_map_feature_engine.py`

Three flat, order-independent ledgers (no mutated cumulative scalar, unlike ELO):

- `TeamSelectedMapState (team, map)` - selected-map results with a STATIC pre-series overall-ELO pair per entry (read once from the frozen Phase 5B.2 audit artifact `series_team_form_states_v1.parquet`, never recomputed by replay).
- `PlayerSelectedMapState (player_id, map)` - ADR/KAST/KD-balance per selected map, keyed by player so it follows a transferred player to their new team.
- `TeamSelectedMapRosterState (team, map)` - historical appearances for current-core continuity only.

Every query filters strictly `series_dt < as_of` at read time, so it is provably safe to build these three ledgers via a single, order-agnostic pass over the full history (see the engine module's docstring for the proof). The one genuinely order-dependent read this phase needs - raw per-team `map_elo` for section 13's map-vs-overall/map-vs-pool differences - is taken from a properly `series_datetime`-batched replay of the frozen, unmodified `map_feature_engine.MapStateStore`.

## Formulas (fixed by design, none tuned)

- `MAP_FORM_HALF_LIFE_DAYS = 90.0`; weight `0.5 ** (age_days / 90)`.
- `weighted_map_wr = (Σw·outcome + 2) / (Σw + 4)` (Beta(2,2) smoothing).
- `weighted_map_margin`, `weighted_map_performance_residual`, `weighted_map_opponent_elo`: weighted means over the team's own eligible history.
- **Trusted-opponent gating** (Phase 6C approval correction #3): `weighted_map_performance_residual` and `weighted_map_opponent_elo` use ONLY entries where `opponent_identity_trusted == True`; `weighted_map_wr`/`weighted_map_margin` use ALL of the team's own eligible history. A new confidence feature, `map_adjusted_history_mass_min`, exposes the trusted-only evidence mass so the model can distinguish "genuinely average performance" from "no trustworthy opponent-adjusted evidence".
- **Map specialization is DIFFERENCES, never ratios** (approval correction #1): `selected_map_elo_vs_overall = map_elo - overall_team_elo`, `selected_map_elo_vs_pool_mean = map_elo - recent_pool_mean_elo`, `selected_map_wr_vs_pool_mean = weighted_map_wr - recent_pool_mean_smoothed_wr`. Cold-start neutral is `0.0` in every case - well-defined even when the pool is empty (both sides then default to `1500.0`, so the difference is `0.0`), never a fabricated ratio fallback.
- `selected_map_rank_percentile`: the map's ELO rank within the team's own recent map pool, `1.0` = best. Cold start `0.5` when the map is absent from the pool or the pool itself is empty, paired with `selected_map_in_both_recent_pools`.
- `player_map_kast_specialization = time_weighted_player_selected_map_kast - time_weighted_player_global_kast` (difference, per approval correction #1), computed per inferred roster player and defaulting to `0.0` for a player missing either side of the comparison, then averaged over the (up to 5) inferred roster players - so the roster-level feature is always finite, never NaN.
- `current_core_map_continuity`: 90-day time-weighted mean overlap fraction between the CURRENT inferred roster and each historical selected-map lineup for that team.

## NaN contract

NaN is permitted only on `roster_map_mean_kast_diff`, `roster_map_bottom_kast_diff`, `roster_map_mean_adr_diff`, `roster_map_mean_kd_balance_diff`, and only exactly where `roster_map_players_with_history_min == 0` - mirroring Phase 5C's `ROSTER_PERFORMANCE_DIFFS` contract precisely, enforced by a hard assertion in the build script (exact pattern match, not "no worse than"). Every other new feature - including every specialization difference - is a documented finite neutral value.

## Full pre-series-ELO join parity (approval correction #4)

Before any opponent-adjusted map residual is computed, the build script asserts, exhaustively over all 10,318 `map_features_v2_rich.parquet` rows (not a sample), that `team1_pre_series_elo - team2_pre_series_elo` equals that file's own inherited `elo_diff` within `1e-6`. **Result: PASSED, max abs diff = 0.0** (see `data/interim/map_features_v3_build_summary.json`). 143 of the 10,461 raw map-stream rows have no match in the frozen `series_team_form_states_v1.parquet` audit artifact (69 match_ids, mostly matches touching an identity-ineligible team, some entirely absent from `series_base.parquet`); none of these ever reach `map_features_v2_rich.parquet` - confirmed empirically (0 of the 30 both-eligible orphan rows are present in it) - so this has zero effect on the parity check or on the final V3 dataset. For these 143 rows, `team1/2_pre_series_elo` is filled with a harmless `ELO_INITIAL` placeholder purely so the arithmetic does not crash; `opponent_identity_trusted` is additionally ANDed with a `pre_series_elo_known` flag so the placeholder can never enter a trusted-population aggregate, while each eligible side's own win/margin fact still updates (matching `map_feature_engine`'s identity policy exactly).

## Map-order audit (brief section 19)

Checked over the real data:

- Every one of the 3,942 multi-map matches shares **one identical raw `datetime`** across all of its maps - there is no independent per-map timestamp in this dataset at all.
- `game_id` is monotonic and contiguous within a match only **42.6%** of the time - not a reliable ordinal even taken at face value.
- Team1's map-win rate on the *last* `game_id` of a series (54.6%) is close to the overall map win rate (53.9%) - reported descriptively only, and **not used as evidence either way**, since series length is itself a function of the result (a red herring either way it points).

**Conclusion: "Map order cannot be independently verified from the current dataset."** Per the Phase 6C approval corrections, `map_slot` is **not added**, regardless of what the contiguity/correlation sub-checks individually showed - the decisive fact is the complete absence of an independent per-map timestamp.

## Final schema

95 (V2 rich) + 25 (Phase 6C: 18 directional + 7 symmetric) = **120 predictive inputs** (80 directional + 37 symmetric/confidence + 3 categorical context, no `map_slot`). Transformed width after deterministic one-hot encoding: 131 columns (106 previous + 25 new numeric columns; the categorical vocabulary itself is unchanged from V2).
