# Phase 5B.2 - Leakage-Safe Opponent-Adjusted and Recency-Aware Team Form (Design)

**No model is trained in this phase.** No validation, test or Cologne metric is computed anywhere. The deliverable is a reusable team-form state engine plus one extended feature dataset built with it.

## 1. Why this phase exists

V1's form features (`win_rate_last_5/10`, `avg_series_margin_last_5/10`) treat "5 wins vs weak opponents" the same as "5 wins vs elite opponents" - they carry no strength-of-schedule or over/under-performance-vs-expectation signal. Phase 5B.2 adds that signal: average pre-match ELO of recent opponents, ELO-expectation performance residuals, and a fixed 60-day exponential time-decay weighting.

## 2. Why a new, independent state store

`scripts/feature_engine.py` (frozen, never modified) computes `elo_expected`/`elo_update` as pure functions, but its `HistoryEntry`/`TeamState` never records the OPPONENT's pre-match ELO or a performance residual - that data does not exist in the frozen engine's state. `scripts/team_form_engine.py` therefore replays the SAME series stream (same rows, same eligibility gating, same two-phase per-timestamp batching, same pure `elo_expected`/`elo_update` calls, imported unchanged) through its OWN state store, additionally recording each match's opponent ELO and performance residual.

**This is not merely asserted to reproduce Phase 3 - it is exhaustively verified.** Before `series_features_v3_form.parquet` is written, `scripts/build_series_features_v3_form.py` compares the new engine's independently-computed pre-match `team1_elo_before - team2_elo_before` against V1/V2's own `elo_diff` column for ALL 9,456 rows (not a sample). Observed result: **max absolute difference 0.0** across all 9,456 rows - bit-for-bit identical. Re-verified independently in `scripts/validate_phase5b2.py` from the audit parquet.

## 3. Pre-match ELO only

`expected_win_prob = elo_expected(own_elo_before, opponent_elo_before)` is always computed BEFORE `elo_update` is called for that match - never recomputed from post-match ratings. `performance_residual = actual_result - expected_win_prob`: an upset win against a stronger opponent gives a large positive residual, an expected win against a weak opponent gives a small positive residual, a loss as a strong favorite gives a large negative residual, and an expected loss against a much stronger opponent gives a smaller-magnitude negative residual. Directly tested for all four scenarios in `tests/test_team_form_engine.py`.

## 4. Trusted-opponent gating - the two populations are NOT interchangeable

Following the Phase 3 / Phase 5A identity policy: an eligible team's own history (result, normalized margin, activity) may still update from a match against an identity-ineligible opponent - that team's own result is a real fact - flagged `opponent_identity_trusted=False`. But opponent-adjusted information depends on a reliable, persistent opponent identity and ELO trajectory, which an untrusted opponent does not have. Therefore:

- **TRUSTED population** (`opponent_identity_trusted == True` only): `avg_opponent_elo_last_5/10`, `performance_residual_last_5/10/all`, `time_weighted_performance_residual`, and the confidence flags `opponent_adjusted_history_min`/`both_teams_have_5_adjusted_matches`/`both_teams_have_10_adjusted_matches` (which count ONLY trusted opponent-adjusted observations).
- **ALL-eligible population** (trusted or not): `time_weighted_win_rate`, `time_weighted_normalized_series_margin`, and their confidence companion `time_weighted_history_mass` - these describe a team's own result/margin, not who they played, so they do not depend on persistent opponent identity.

## 5. Recency weighting

Fixed engineering constant `FORM_HALF_LIFE_DAYS = 60` - never tuned against any metric, never compared against 30/90-day alternatives. `weight = 0.5 ** (age_days / 60)`. Applied to `time_weighted_win_rate`, `time_weighted_performance_residual` (trusted population), and `time_weighted_normalized_series_margin`. `time_weighted_history_mass = sum(weight_i)` over the ALL-eligible population is the confidence companion for the first and third of these; `time_weighted_history_mass_min = min(mass_team1, mass_team2)` is the symmetric confidence feature actually stored.

## 6. Series margin

`normalized_series_margin(score_for, score_against) = (score_for - score_against) / (score_for + score_against)`, 0.0 when the denominator is 0 - the same normalization convention `scripts/map_feature_engine.py` uses for map margins (a local copy, not an import, to keep this module's only dependency on the Phase 3 engine), applied here to series map-count scores so it scales consistently across BO1/BO3/BO5. Computed only from each historical match's own final score - never from the target series.

## 7. Cold start

| quantity | cold start | rationale |
|---|---|---|
| avg_opponent_elo_last_5/10 | 1500.0 (ELO_INITIAL) | neutral opponent assumption, no evidence |
| performance_residual_last_5/10/all | 0.0 | neutral - no evidence of over/under-performance |
| time_weighted_win_rate | 0.5 | neutral |
| time_weighted_performance_residual | 0.0 | neutral |
| time_weighted_normalized_series_margin | 0.0 | neutral |
| time_weighted_history_mass | 0.0 | zero effective evidence - a true absence, not "neutral" |
| confidence flags | 0 | explicit, so a neutral value is never mistaken for evidence |

No opponent history is ever fabricated for a cold-start team.

## 8. Exact-timestamp leakage protection

`process_form_stream` uses the identical two-phase per-timestamp-group protocol as `feature_engine.process_chronological_stream`: Phase A emits features for every eligible-pair row in a timestamp group from the state as it was BEFORE the group; Phase B applies every row's result only after the whole group has been read. Proved on real data in `scripts/validate_phase5b2.py` (a real shared-timestamp group is rebuilt from a pre-group state snapshot and every match in the group is confirmed to see that snapshot, not any other match's result from the same group) and on synthetic fixtures in `tests/test_team_form_engine.py`.

## 9. Feature inventory

8 new directional (Team1-Team2 diffs) + 4 new symmetric/confidence = **12 new features**, appended to V2's 47 to give V3's 59 predictive features (38 directional + 19 symmetric + 2 categorical context).

Directional: `avg_opponent_elo_last_5_diff`, `avg_opponent_elo_last_10_diff`, `performance_residual_last_5_diff`, `performance_residual_last_10_diff`, `performance_residual_all_diff`, `time_weighted_win_rate_diff`, `time_weighted_performance_residual_diff`, `time_weighted_series_margin_diff`

Symmetric/confidence: `opponent_adjusted_history_min`, `both_teams_have_5_adjusted_matches`, `both_teams_have_10_adjusted_matches`, `time_weighted_history_mass_min`

## 10. What Phase 5B.2 deliberately does NOT do

- No model is trained; no validation, test or Cologne metric is computed.
- No half-life tuning (30/60/90-day comparison) - 60 days is a fixed a priori constant.
- No feature selection, and no feature-vs-target association is reported anywhere.
- No post-Cologne deployment snapshot.
- Nothing under `data/raw/`, `reference/` or `src/` is touched; no Phase 1-5B.1 artifact is modified; the test partition and main validation partition are never loaded.
