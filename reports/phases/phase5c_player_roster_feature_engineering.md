# Phase 5C - Leakage-Safe Player Form and Roster Stability (Design)

**No model is trained in this phase.** No validation, test or Cologne metric is computed anywhere. The deliverable is a reusable player/roster state engine plus one extended feature dataset built with it.

## 1. Why player features may add signal

Team ELO, map pool and team-level form all describe a team as an indivisible unit. A CS2 team is five individuals whose recent individual output varies, who change employers, and who are sometimes replaced by stand-ins. Two teams with identical ELO can differ sharply in the current form of their star, the form of their weakest player, and whether they are even fielding a settled lineup. Phase 5C tries to make that visible pre-match.

## 2. Two distinct state types (never conflated)

- **Global player performance**, keyed by persistent `player_id`. Individual form FOLLOWS THE PLAYER across team changes - a transfer must not pretend a strong player has no history. This state never asks which team the player was on.
- **Team roster / appearance**, keyed by canonical team identity. Which players have recently REPRESENTED THAT TEAM. Team membership does NOT transfer with the player: a transferred player enters the new team's inferred roster only after actually appearing for it. Both halves of that distinction are pinned by explicit tests.

## 3. Persistent player identity

The schema audit found `player_id` to be by far the most trustworthy identifier in this dataset: 1,214 distinct ids in the development stream, nullable `Int64` with no `0`/`-1` sentinels, and **zero** id-to-name conflicts in either direction. By contrast `team_id` is not an organisation key at all (92% of players change it), so every team-keyed structure here uses `team{1,2}_canonical`. Player NAME columns are never read: 1,316 slots carry an id with a blank name, so the id is the strictly broader signal.

## 4. Historical roster inference (and why the target lineup is forbidden)

For team X at cutoff T: take that team's appearances in the half-open window `[T - 90d, T)`, weight each by `0.5 ** (age_days / 60)`, and take the top 5 players by (mass, then most recent appearance, then lowest `player_id`). Fewer than five available -> use what exists; players are never fabricated.

The target series' actual lineup is forbidden as a predictor for two independent reasons, both pointing the same way:

1. **Application contract.** The planned single-page app asks a user to choose two TEAMS, not ten player ids. A training-time feature depending on the real lineup could never be reproduced at inference time.
2. **This repo's own standing ruling.** `reports/data_audit.md` open question 2 and `reports/leakage_analysis.md` record that it is *unknown* whether the lineup columns are a pre-announced starting five or a post-hoc box-score roster including substitutions, and rule that they must be **treated as leaky by default until the collection method is confirmed**. Inferring from prior appearances only is what keeps Phase 5C compliant with that ruling.

The actual lineup is used ONLY in Phase B, after the series' own feature row has been emitted, to update state for LATER series. A dedicated test replaces the target series' five players with five completely different players and asserts that not one emitted feature moves.

## 5. Authoritative series datetime

The cutoff is the authoritative series start taken from the canonical series source (`series_base.parquet`, the same column V1/V2/V3 use), joined onto every map/player observation of that `match_id`. It is deliberately **not** `groupby(match_id).datetime.min()` over map rows - map timestamps are provenance only. In this export 0 map rows carry a map timestamp differing from the authoritative series start; a test constructs a series whose maps carry deliberately different map timestamps and proves the emitted pre-series features are unchanged.

## 6. Exact-timestamp and same-series isolation

```
for each authoritative series_datetime batch:
    PHASE A (read):  emit ONE pre-series vector per requested series,
                     from the state as it was BEFORE the batch
    PHASE B (write): only then apply every player observation in the batch
```

Map 1's box score therefore cannot reach Map 2 of the same series, a series cannot see its own maps, and two series sharing an instant cannot see each other. Input row order within a batch is irrelevant to the output.

## 7. Player-statistic normalization

- `kd_balance = (kills - deaths) / max(kills + deaths, 1)` - bounded in [-1, 1], preferred over a raw K/D ratio because it cannot explode when deaths are small.
- `assists_per_round = assists / max(rounds, 1)`, where `rounds = score1_game + score2_game`. Included only because the audit proved that denominator clean (0 nulls, median 21, and a kills-per-round distribution tightly concentrated at the theoretical ~6.6).
- `adr` and `kast` are **already round-normalized rates** in the source and are used as-is - never divided again. `kast` is on a 0-100 scale.
- `kddiff` is **never read**: the audit found it is 100% collinear with `kills - deaths` (0 mismatches in 102,243 slots).

## 8. Time weighting

Player form uses `weight = 0.5 ** (age_days / 60)` over **all** of a player's strictly-prior maps (the 90-day window applies to roster inference only). `player_history_mass = sum(weights)` is carried alongside every form statistic so the model can tell a well-evidenced estimate from a thin one. Both constants are fixed engineering choices for this phase and are **not tuned** against any metric.

## 9. Malformed source rows

The audit found two structural defects that would silently corrupt a per-player state. Both are handled at load time and counted:

- **Same player on BOTH sides of one map** (5 maps): that player would receive a win and a loss for one map and both teams would record them as a member -> the **entire map** is excluded from performance and appearance updates.
- **Same player in more than one slot on one side** (14 groups): collapsed to a single observation when the duplicated slots agree (14), and the player's entry excluded from that map when they disagree (0).

Independently of the loader, the store enforces two structural invariants by construction, so duplication can never inflate history mass even if a future data source presents it: **at most one `PlayerMapEntry` per (game_id, player_id)** and **at most one `AppearanceEntry` per (game_id, team, player_id)**.

## 10. Identity policy for the two states

Following the Phase 3 / 5A / 5B.2 principle ("a team's own result is a real fact") applied at the correct key granularity: **player performance** is player-keyed and updates whenever that player's box score is usable, regardless of either team's identity eligibility - the player's own performance is a real fact about the player. **Team appearance** is team-keyed and updates only when that canonical team is identity-eligible, exactly like every other team-keyed state in this project.

Note that canonical team identity deliberately spans roster turnover (`KEEP_AS_SINGLE_TEAM` in the Phase 2.5 policy, whose stated rationale is that full roster turnover is expected for real organisations over a multi-year dataset). That is what makes turnover a measurable signal here rather than a hidden identity split.

## 11. Cold start

| quantity | cold start | rationale |
|---|---|---|
| roster performance aggregates | **NaN** | genuinely missing; never a sentinel and never a population-wide mean, which would import information from outside the strictly-prior window |
| player_history_mass | 0.0 | a true absence of evidence, not "neutral" |
| roster_size, stability counts/ratios, flags | 0 | |

Aggregates are computed over the inferred roster players that have at least one usable prior observation, and are NaN only when NO inferred roster player has any. The build asserts the exact invariant that the ten performance diffs are NaN **exactly** where `roster_form_players_min == 0`. Downstream this is handled the same way `days_since_last_match_diff` already is: preserved natively by XGBoost, train-fold-only median imputation for Random Forest.

`roster_size_min` and `roster_form_players_min` are deliberately different quantities: a team can have five INFERRED players while only some of them have usable prior box scores.

## 12. Feature inventory

**15 directional + 6 symmetric = 21 new features**, appended to V3's 57 predictive columns for a V4 total of 80.

Deliberately compact and CS2-meaningful rather than exhaustive: lineup quality (mean), star form (top), weak-link form (bottom) for ADR/KAST/KD-balance; support contribution (assists per round); roster churn and continuity; and the evidence behind all of it.

Directional: `roster_mean_adr_diff`, `roster_top_adr_diff`, `roster_bottom_adr_diff`, `roster_mean_kast_diff`, `roster_top_kast_diff`, `roster_bottom_kast_diff`, `roster_mean_kd_balance_diff`, `roster_top_kd_balance_diff`, `roster_bottom_kd_balance_diff`, `roster_mean_assists_per_round_diff`, `recent_unique_players_10_maps_diff`, `recent_unique_players_20_maps_diff`, `core5_appearance_concentration_90d_diff`, `core5_continuity_last_10_diff`, `roster_mean_player_history_mass_diff`

Symmetric/confidence: `roster_size_min`, `both_teams_have_5_inferred_players`, `roster_min_player_history_mass`, `roster_core_concentration_min`, `roster_core_continuity_last10_min`, `roster_form_players_min`

## 13. Future application contract

`build_future_player_roster_features(store, team1, team2, as_of_datetime)` is pure and read-only and takes **no target, no score and no lineup argument**, so the target series' five players cannot enter even by accident (asserted by signature inspection). It is the exact function the offline builder calls, so training and inference cannot silently diverge. The state is deliberately keyed by persistent player id and canonical team name only, so a future provider (e.g. GRID) could update it without any feature definition changing. No such integration is done now.

## 14. Pre-Cologne state

`scripts/build_pre_cologne_player_roster_state_v1.py` replays the canonical stream strictly before the Cologne cutoff into a fresh store and independently re-derives, from the store itself, that no history entry or appearance is at/after the cutoff and that no Cologne `match_id` appears anywhere. Written as a flat scalar parquet plus the full reloadable JSON. No post-Cologne deployment state is built.

## 15. Limitations (read before interpreting any later ablation)

- **Player data covers only about half the series universe.** `map_base` carries player rows for 4,941 of the 9,456 series in V4 (4,515 have none of their own), and map coverage is strongly tier-skewed (~74% tier1 vs ~36% tier2 and ~26% tier3). **2,880 of 9,456 V4 rows (30.5%) have no usable prior player history on at least one side** and therefore carry NaN performance features. This is the binding constraint on how much these features can possibly contribute, and it is a property of the source data, not of the engine.
- The inferred roster is a *prediction* of the lineup, not the lineup. When a team fields an unexpected stand-in, the features describe who was expected to play.
- Individual statistics are opponent-unadjusted: ADR earned against weak opposition counts the same as ADR against elite opposition. The opponent-adjustment machinery built in Phase 5B.2 operates at team level only.
- `ex-<Org>` canonical variants register as separate teams, so a rebrand looks like a brand-new team with no roster history.
- No post-Cologne deployment state; no GRID or other live provider integration.

## 16. What Phase 5C deliberately does NOT do

- No model is trained or tuned; no validation, test or Cologne metric is computed.
- No tuning of the 90-day roster window or the 60-day half-life.
- No feature selection, and no feature-vs-target association is reported anywhere.
- No use of the target series' lineup, player ids, player names or box score.
- Nothing under `data/raw/`, `reference/` or `src/` is touched, and no Phase 1-5B.3 artifact is modified.
