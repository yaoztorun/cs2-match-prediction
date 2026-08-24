# Preprocessing / Temporal Safety Rules

Static documentation (not code-generated) for Phase 3+ feature engineering. These rules govern how `data/interim/series_base.parquet` and `data/interim/map_base.parquet` may be used. Nothing in this document has been implemented yet — no ELO, no rolling statistics, no models exist as of Phase 2.

## Temporal safety rules

1. **Historical features for a match at time t may use only matches with `datetime < t`.** Strictly earlier, not `<=`.
2. **Matches sharing the same `datetime` must not leak into one another.** Same-timestamp matches (e.g. concurrent bracket matches) are not "before" each other and must be treated as mutually invisible when building historical features.
3. **Same-series map isolation**: when building features for the *future pre-series map-prediction task*, no map from a series (`match_id`) may be used as historical input for another map in that *same* series. A series' own maps are not "history" for each other.
4. **Same-series, same historical cutoff (added in this phase)**: for the future pre-series map-prediction task, every map belonging to the same `match_id` must use the **same** historical cutoff — the point immediately before the series starts (i.e. the series' own `datetime`). Earlier maps within that series (e.g. map 1's result) may **not** be used to build features for a later map in the same series (e.g. map 2). This is stricter than "just don't use future maps" — it also forbids using a series' *own earlier* maps as if they were independent history, since in the pre-series prediction task the whole series is being predicted before it starts.
5. **IEM Cologne Major 2026 remains untouched as an external case-study period.** It is audited for coverage only (Phase 1 `data_audit.md`, Phase 2 `evaluation_manifest.csv`'s `cologne_2026` group) and must never be used for feature selection, hyperparameter tuning, model selection, or training during the historical backtest (see "Evaluation manifest" below).
6. **Current-match player/map/score statistics are never direct predictors.** Any column describing the outcome of the match/map being predicted (scores, box-score stats, `team1_series_win`/`team1_map_win` themselves) may only enter a feature as a *historical aggregate* computed under rules 1-4 above — never as the current row's own value. See `reports/leakage_analysis.md` (Phase 1) for the full Task A (pre-veto series prediction) vs. Task B (future map-specific prediction) column classification this builds on.

## `map_base.parquet` retained columns — why they're there

`map_base` deliberately keeps `datetime`, `bestOf`, `map_id`, `score1_game`, `score2_game`, player names, persistent `player_id`s, and the raw per-player statistics (kills/deaths/assists/adr/kast/kddiff), even though several of these are post-map outcomes. **`score1_game`/`score2_game` and the player statistics must never be used as a direct predictor for their own row** — they are the outcome of the map they belong to. They are retained because they are exactly what's needed to construct strictly historical features (rules 1-4) for *other, later* matches — e.g. a player's rolling ADR going into a future match is computed from these same columns on their prior maps. Dropping them from `map_base` would have made all future historical feature engineering impossible; keeping them is safe only because rules 1-4 are followed when they're consumed.

`bestOf` on map rows is nullable and was never used as a rejection criterion when building `map_base` (unlike `series_base`, where a blank `bestOf` causes the whole series row to be rejected) — a map's own validity does not depend on its series having a known format.

## `team1_id` / `team2_id` — audit/orientation metadata only

Both canonical tables keep `team1_id`/`team2_id` from the raw data. Per Phase 1 and the Phase 2 team-identity analysis, these are **per-match-appearance surrogate keys, not persistent team identities** (every value appears in exactly one match). They are retained purely as audit/orientation metadata — e.g. to reproduce the Phase 1/2 analyses, or to re-verify `team1_id < team2_id` — and must **never** be used as a team-identity join key or as an ML strength/feature input. Use `team1_canonical`/`team2_canonical` (from `data/interim/team_aliases.csv`) for any identity-dependent logic, subject to the caveats in `reports/team_identity_analysis.md` (an *initial candidate key*, not a fully resolved one — `unresolved` names need manual review before being trusted).

## Evaluation manifest — experimental metadata, never a feature

`data/interim/evaluation_manifest.csv` (`match_id`, `evaluation_group` ∈ `{development, cologne_2026, post_cologne}`) is **experimental/backtest bookkeeping only**. It must never be joined into a feature table as a model input. Its purpose:
- `cologne_2026`: the 107 IEM Cologne Major 2026 matches — reserved as an external case study, never used for feature selection, hyperparameter tuning, model selection, or training.
- `post_cologne`: the 32 matches with a later `datetime` than Cologne's last match, excluding Cologne itself — the only genuine forward-holdout material this dataset currently has (and there is very little of it, per the Phase 1 audit).
- `development`: everything else (9,784 matches) — the pool for historical feature construction, training, and internal backtesting.

## Dataset-wide / future-aware statistics

Any statistic computed across the *entire* dataset regardless of time (e.g. `reports/orientation_analysis.md`'s per-team overall win rate, used there purely to diagnose the team1/team2 ordering artifact) is diagnostic only and must never become a model input. A real "team strength" feature must be rebuilt from scratch using only strictly-prior matches under rules 1-4 above.

## Cross-reference

See `reports/leakage_analysis.md` (Phase 1) for the column-by-column known-before-match / leaky / ambiguous classification, `reports/team_identity_analysis.md` and `reports/orientation_analysis.md` (Phase 2) for the evidence behind the team-identity and side-ordering caveats above, and `reports/phase2_summary.md` for what has and hasn't been built so far.
