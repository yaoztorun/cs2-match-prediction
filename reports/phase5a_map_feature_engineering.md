# Phase 5A - Leakage-Safe Map History and Map-Pool Feature Engineering

**No model is trained in this phase.** No validation, test or Cologne metric is computed anywhere. The deliverable is a reusable map state engine plus two feature datasets built with it.

## 1. Why this phase exists

Phase 4 tuned three very different algorithms on the same 17 series features and they converged into a narrow band (validation ROC-AUC 0.6412 / 0.6566 / 0.6504 for LR V2, RF V2 and XGB V2). When three model families with different inductive biases agree that closely, the binding constraint is usually the information in the features rather than the flexibility of the model. Phase 5A adds genuine CS2 domain knowledge - what maps each team actually plays and how well - to test that hypothesis in a later phase.

## 2. The two prediction tasks are NOT the same task

Everything in this phase follows from one distinction, which is why the two datasets exist separately rather than as one table with a flag:

|  | Pre-veto series | Known map |
| --- | --- | --- |
| **Task** | predict the series winner **before the veto** | predict the winner of **one specific map** |
| **Dataset** | `series_features_v2_map_pool.parquet` | `map_features_v1.parquet` |
| **`prediction_task`** | `pre_veto_series` | `known_map` |
| **Maps of the target match** | **FORBIDDEN** - unknown at prediction time | **LEGITIMATE INPUT** - the map is given |
| **`map_name` column** | absent (enforced) | present, a real feature |
| **Map knowledge enters as** | summaries of the teams' own PRIOR maps | both teams' record ON THIS MAP + pool context |
| **Rows** | 9,456 series | 10,318 maps |

The `prediction_task` identifier lives in the YAML config as metadata. It is deliberately **not** a column in either parquet: it describes the task, and feeding it to a model would be meaningless (it is constant within a dataset).

## 3. The prediction cutoff: `series_datetime` vs `map_datetime`

These are different concepts and the engine keeps them apart everywhere:

- **`series_datetime`** - the prediction cutoff. Everything knowable about a series, *including every map that will be played in it*, is frozen at this instant.
- **`map_datetime`** - when an individual map was actually played. It may be later than the cutoff, and a future provider may expose a genuine per-map start time.

All batching, all history filtering and all feature emission key on `series_datetime`. `map_datetime` is carried for provenance and **never** decides what a map may see. In the current export the two coincide for every row (`map_datetime_differs_from_series` = 0), but that is a property of this export, not an assumption of the pipeline - `TestD2_DifferentMapTimestampsSameSeriesCutoff` constructs a BO3 whose three maps carry deliberately different `map_datetime` values (12:05, 13:20, 14:45) under one cutoff and proves all three still receive the identical pre-series state, and that reversing those map times changes no feature at all.

## 4. Series-atomic, two-phase processing

A `match_id` is an **atomic prediction/update unit**:

```
for each exact series_datetime batch:
    PHASE A (read):  emit features for ALL maps of ALL series in the batch
                     from the state as it was BEFORE the batch
    PHASE B (write): only then apply every completed map result
```

Consequences, each proved by a synthetic test rather than argued:

- Map 1's result cannot reach Map 2 or Map 3 of the same series - the state is never updated inside a series.
- Two different series sharing one timestamp cannot see each other, in either input order. This is not theoretical here: **706 timestamps in this dataset carry more than one distinct match, covering 3,348 map rows.**
- The same map appearing twice in one series is still isolated.
- Input row order is irrelevant to the output.

## 5. Reuse of the Phase 3 engine

`map_feature_engine.py` imports `elo_expected`, `elo_update`, `ELO_INITIAL`, `ELO_K` and `_beta_smoothed_win_rate` from `feature_engine.py` rather than reimplementing them. Map ELO is therefore the same rating mathematics with the same **K = 32** and the same 1500 cold start as series ELO, and map win rates use the same Beta(2,2) smoothing. **No K tuning and no time-decay tuning happens in Phase 5A** - `MAP_POOL_LOOKBACK_DAYS = 180` is a fixed design constant, chosen a priori and never selected against any performance metric.

## 6. Feature definitions

### 6.1 Normalized round margin

`(rounds_for - rounds_against) / (rounds_for + rounds_against)`, and exactly `0.0` when the denominator is zero. Normalizing by the total makes the signal invariant to CS scoring-format changes (MR15 -> MR12), so a 16-8 win and an 8-4 win score identically. Examples: 13-10 -> 3/23 = 0.130, 13-5 -> 8/18 = 0.444.

### 6.2 Beta(2,2) smoothed map win rate

`(wins + 2) / (matches + 4)`. A team that has won its only map on Nuke gets 0.60, not 1.00. This matters far more per-map than per-series: map samples are thin by construction, and an unsmoothed rate would hand the model a pile of spurious 0.0/1.0 certainties.

### 6.3 Recent map pool

The maps a team played in the half-open window `[T - 180d, T)`. Half-open by construction: a map at exactly `T` is the current series and is excluded; a map at exactly the lower bound is included. A map is **experienced** for a team when it has at least 5 prior maps on it.

## 7. Two DISTINCT map-pool families (do not describe them interchangeably)

### (A) Pool-depth / order-statistic features - `map_pool_*`

Each team's own recent pool is summarized independently (size, total matches, experienced-map count, mean/best/2nd/3rd/worst map ELO and smoothed win rate, mean margin), and only then subtracted. **The two sides' k-th-best entries may refer to completely different map identities.** `map_pool_best_elo_diff` says "Team1's strongest map is stronger than Team2's strongest map" - it says nothing about whether those are the same map, and it is not a head-to-head comparison. This family describes the *shape and depth* of each pool.

### (B) Same-map matchup features - `map_matchup_*`

Computed across `union(pool1, pool2)`; for each map identity the two teams are compared **on that same map** (neutral cold-start defaults where one side lacks history), and the resulting per-map advantages are then summarized. These *are* genuine head-to-head, per-map comparisons.

### Why family (B) uses mean / median / midrange and not k-th-best

Mirrored augmentation requires every directional feature to negate exactly under a Team1<->Team2 swap. Swapping negates the per-map advantage list, and the k-th **largest** of a negated list is minus the k-th **smallest** of the original - so a "2nd best advantage" feature is antisymmetric only when *paired* with "2nd worst advantage", which the mirroring contract cannot express. `mean`, `median` and `midrange = (max+min)/2` each satisfy `f(-x) = -f(x)` individually. The spread that the order statistics would have carried is preserved by the swap-**invariant** `map_matchup_elo_advantage_range`, which is therefore declared symmetric, not directional. Family (A) keeps its order statistics precisely because each side's statistic is computed over its own pool *before* subtraction, so the difference negates cleanly. `TestG.test_order_statistics_would_have_broken_symmetry` pins this down.

## 8. Feature inventory

| dataset | family | n | example |
| --- | --- | --- | --- |
| `map_features_v1` | map-specific directional | 9 | `map_elo_diff` |
| `map_features_v1` | map-specific symmetric | 5 | `both_teams_have_5_map_matches` |
| `map_features_v1` | Phase 3 series features (joined) | 15 | `elo_diff` |
| `map_features_v1` | categorical context | 3 | `map_name` |
| `series_features_v2_map_pool` | (A) pool depth, directional | 14 | `map_pool_best_elo_diff` |
| `series_features_v2_map_pool` | (B) same-map matchup, directional | 6 | `map_matchup_mean_elo_advantage` |
| `series_features_v2_map_pool` | map-pool symmetric / confidence | 10 | `shared_experienced_map_count` |
| `series_features_v2_map_pool` | inherited Phase 3 V1 features | 17 | `elo_diff` |

The three symmetric map-coverage confidence features are `shared_recent_map_count` (`|pool1 n pool2|`), `shared_experienced_map_count` (map identities where **both** teams have >= 5 prior maps) and `map_matchup_shared_coverage` (`|pool1 n pool2| / |pool1 u pool2|`, **0.0 when the union is empty**). They let a model discount a matchup advantage that rests on thin or non-overlapping evidence.

## 9. Cold-start contract

Identical at training time and at inference time, so a cold-start row is not a special case the deployment path has to reinvent:

| quantity | cold start |
| --- | --- |
| map ELO | 1500.0 (= `ELO_INITIAL`) |
| smoothed map win rate | 0.5 (Beta(2,2), zero observations) |
| rolling win rates | 0.5 |
| normalized margins | 0.0 |
| match counts / pool sizes / flags | 0 |
| `days_since_map_played` | **NaN - genuinely missing, never a sentinel number** |
| k-th-best slot, pool shallower than k | the neutral value (1500 / 0.5), never an extreme |
| empty `union(pool1, pool2)` | every `map_matchup_*` advantage 0.0; coverage 0.0 |
| unseen map identity | reserved category `__UNKNOWN_MAP__` |

Using a neutral rather than an extreme fallback for a shallow pool is deliberate: an extreme sentinel would masquerade as a real strength reading. The thin-evidence signal is carried by the confidence features instead, where a model can weigh it explicitly.

## 10. Identity policy - two different eligibility questions

The policy is inherited unchanged from Phase 2.5/3 (`team_identity_policy.csv`); Phase 5A makes no new identity decisions. Two rules that are easy to conflate:

1. **A supervised feature row** requires **both** canonical identities to be trusted.
2. **State updates** are finer-grained. An eligible team's *own* map history (win/loss, margin, recency, pool membership) updates even when the opponent's identity is not trustworthy - that team's own result is a real fact - and the entry is flagged `opponent_identity_trusted=False`. But **map ELO is pair-dependent** and moves both ratings, so it updates *only* when both identities are trusted.

This is measurable, not just asserted: the pre-Cologne snapshot contains **113 history entries** recorded against an untrusted opponent, from 113 such maps. Rebuilding the snapshot by replaying `map_features_v1` (which contains only both-eligible rows) would have silently destroyed every one of them - which is exactly why `build_pre_cologne_map_state_v1.py` replays the **canonical map stream** instead.

## 11. Dataset construction

### 11.1 `map_features_v1.parquet`

| step | rows |
| --- | --- |
| `map_base.parquet` rows | 10674 |
| labeled `development` (Cologne + post-Cologne dropped) | 10461 |
| minus rows touching an identity-ineligible team | 10348 |
| minus maps whose match is absent from `series_features_v1` | 10318 |

Final: **10,318 rows** over 4,952 matches and 9 map identities. The target `team1_map_win` was **re-derived from `score1_game > score2_game`** rather than trusted: 10,461 rows checked, 0 ties, 0 null scores, 0 disagreements. Target positive rate 0.5386.

The 17 Phase 3 series features are joined by `match_id`. This is legitimate: they are computed strictly before the same series cutoff, so a map row sees exactly the series-level knowledge that existed before its series began.

### 11.2 `series_features_v2_map_pool.parquet`

**9,456 rows - the same match_id universe, the same targets and the same ordering as `series_features_v1`, with every V1 feature column preserved value-for-value** (asserted in the builder and re-checked independently by the validator).

Critically, 4,504 of these 9,456 series have **no map rows of their own**, and they still receive full pool features - because pool features describe the two teams' *prior* map history, not the target match. Driving emission off the map stream would have silently dropped them, so the engine takes an explicit series-request frame instead (`process_combined_stream`). A series request contributes nothing to state; only completed maps do.

## 12. Forbidden columns

Enforced in code at build time and re-checked by `validate_phase5a.py`:

- **Pre-veto series V2**: no `map_name`, `map_id`, `game_id`, no list/count/order of the maps selected in the target match, no `score1_game`/`score2_game`, no per-map result of the target series, no player statistic from it. Any of these would leak the veto.
- **Map V1**: no `score1_game`/`score2_game` (the current map's own score *is* the target), no `map_id`, no player boxscore column, nothing derived from the current or any later map.
- **Both**: `evaluation_group` is experimental bookkeeping and never a model input; no player-level statistics anywhere in Phase 5A.

## 13. Frozen pre-Cologne map state

| property | value |
| --- | --- |
| cutoff source | `evaluation_manifest.csv` `cologne_2026` group (tournament name only cross-checked) |
| cutoff rule | `series_datetime < cologne_first_datetime` (strict) |
| Cologne first datetime | 2026-06-02 13:30:00 |
| max source `series_datetime` | 2026-05-30 19:30:00 |
| maps replayed | 10,397 |
| team-map states | 1,911 |
| distinct teams / maps | 382 / 9 |
| Cologne match_ids in any history | 0 (asserted independently) |
| post-Cologne deployment snapshot | **not built** - out of scope for Phase 5A |

The snapshot is written twice: a flat scalar `.parquet` summary (one row per team-map, fastparquet-safe) and a `.json` carrying the full re-loadable state. A round-trip test proves a reloaded store reproduces identical feature vectors.

## 14. Future application

The functions that generate training rows are the *same* functions a live pre-match call would use - `build_future_map_matchup_features(store, team1, team2, best_of, map_name, as_of_datetime, tier)` and `build_future_series_map_pool_features(store, team1, team2, best_of, as_of_datetime, tier)`. Both are pure and read-only, neither takes a target or a current score, and the series builder takes **no map argument at all** (asserted by signature inspection, so pre-veto contamination cannot creep in via a later edit). Two equivalence tests confirm that an offline stream and a live call produce identical numbers, which is what stops training and deployment from silently diverging.

## 15. Test coverage

`tests/test_map_feature_engine.py` - 73 synthetic-fixture tests, no dependency on the real dataset. Groups A-N cover Beta smoothing, map ELO (including rating conservation), strict chronology, same-series isolation, differing map timestamps under one cutoff, same-timestamp isolation across series, target reconstruction, side-swap symmetry, cold start, the 180-day window boundaries, normalized-margin arithmetic, the identity policy, the future-application contract, absence of player statistics, and snapshot round-trip - plus the stream contract and the combined driver.

## 16. What Phase 5A deliberately does NOT do

- No model is trained; no validation, test or Cologne metric is computed.
- No player-level features.
- No time-decay or ELO-K tuning; every constant is fixed a priori.
- No new identity decisions.
- No feature selection, and no feature-vs-target association is reported anywhere - including in the quality report, which is descriptive only.
- No post-Cologne deployment snapshot.
- Nothing under `data/raw/`, `reference/` or `src/` is touched, no Phase 4 artifact is modified, and the test partition stays sealed.

## 17. Open questions carried into the next phase

1. **Map-pool features may be partly redundant with series ELO.** A team with a strong pool is usually just a strong team. Whether family (A) adds anything beyond `elo_diff` is an empirical question that must be answered by a model comparison, not asserted here.
2. **Map rows begin 2023-10-25, about nine months after series rows.** 957 of 9,456 series (10.1%) therefore have a completely empty recent pool. Whether to model these or restrict the map-pool comparison to the covered era is a Phase 5B decision.
3. **The 180-day window is unvalidated by construction.** It was fixed a priori to keep this phase honest. If it is ever tuned, that must happen inside chronological CV on TRAIN only.
4. **`map_features_v1` rows are not independent** - up to 5 maps share a series and a pre-series state. Any future map-level model must account for this in its splitting and in its error bars; treating 10,318 maps as 10,318 independent observations would overstate confidence.
