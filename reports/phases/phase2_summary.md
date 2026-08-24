# Phase 2 Summary

Built canonical, conservatively-cleaned series-level and map-level datasets on top of the Phase 1 audit. No models trained, no ELO, no rolling features — that's Phase 3+.

## What was built

| Script | Output |
|---|---|
| `scripts/team_identity_analysis.py` | `reports/team_identity_analysis.md`, `data/interim/team_aliases.csv` |
| `scripts/build_canonical_datasets.py` | `data/interim/series_base.parquet`, `map_base.parquet`, `rejected_series_rows.csv`, `rejected_map_rows.csv`, `evaluation_manifest.csv` |
| `scripts/orientation_analysis.py` | `reports/orientation_analysis.md` |
| (static) | `reports/preprocessing_rules.md` |
| `scripts/validate_phase2.py` | 24/24 assertions passing (see below) |

All scripts are re-runnable from scratch and are read-only against `data/raw/` — verified both by an internal content-hash check in `build_canonical_datasets.py` and by `git status` showing zero changes under `data/raw/`.

## Headline counts

- **series_base**: 9,801 retained / 122 rejected (of 9,923 raw series rows) — rejected: 116 `missing_bestOf`, 5 `tie`, 1 `missing_score` (no overlap between reasons).
- **map_base**: 10,674 retained / 79 rejected (of 10,753 raw map rows) — rejected: 39 `blank_map_row`, 39 `missing_map_name`, 1 `tie`.
- **evaluation_manifest**: 9,923 matches — 9,784 `development`, 107 `cologne_2026`, 32 `post_cologne`.
- **team_aliases**: 792 distinct raw team names — 765 `exact`, 2 `normalized` (the single verified `Magic`/`magic` merge), 0 `manual_alias`, 25 `unresolved` (flagged for manual review, not auto-merged).

## Team identity conclusion

`team_id` is confirmed unusable as a persistent identity (per-match-appearance surrogate, 0 reuse across matches — Phase 1 finding, re-confirmed here). Normalized `team_name` is a workable *initial candidate key*, not a fully resolved one: the "Super DraculaN Season 1" tournament flagged as suspicious in Phase 1 is now resolved (100% roster overlap across every colliding team_id there); major Tier-1 orgs show strong roster/multi-year continuity; but 25 names remain `unresolved` (generic/placeholder lower-tier names, or names with a same-tournament/near-in-time zero-roster-overlap team_id pair) and must go through manual review before any ELO/rolling/head-to-head feature trusts them. `team1_id`/`team2_id` are retained in both canonical tables strictly as audit/orientation metadata — never a feature or join key.

## Orientation analysis conclusion

The reconstructed team1 win rate (55.1%, matching Phase 1) stays flat (~54-58%) across year, tier, and best-of format — the signature of a structural/listing-order artifact in how `team_id`s were assigned upstream, not genuine skill information or random noise. A diagnostic (never-a-feature) dataset-wide team strength proxy shows team1 is the stronger side in 57.4% of comparable matches, directionally consistent with that hypothesis. Recommendation for later phases: difference features and/or deterministic side-swapped/mirrored training augmentation, documented but not implemented here.

## Validation

`scripts/validate_phase2.py`: **24/24 checks passed** — one row per retained `match_id`/`game_id`, binary targets only, `team1_win` absent from both output tables, valid `tier`/`bestOf` values, every map traceable to a known `match_id`, every rejected row has a non-blank reason, rejected+retained counts reconcile exactly to the raw totals, evaluation manifest covers the full match_id universe and is absent from both feature tables, and `data/raw/` is untouched.

## Environment note

`pyarrow`'s native DLL is blocked by this machine's Application Control policy (`ImportError: ... An Application Control policy has blocked this file`). `fastparquet` was installed and verified (round-tripped nullable `Int64` correctly) and is used as the parquet engine instead; both `series_base.parquet` and `map_base.parquet` were written and validated with it.

## Explicitly NOT done in this phase

- No ELO ratings.
- No rolling/lagged historical features.
- No model training, no model selection, no hyperparameter tuning.
- No use of `cologne_2026`/`post_cologne` matches for anything beyond coverage auditing and manifest bookkeeping.
- No resolution of the 25 `unresolved` team names — flagged for manual review, not decided here.

Phase 3 (feature engineering) has not been started.
