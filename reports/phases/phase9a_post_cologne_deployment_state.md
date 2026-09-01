# Phase 9A — Post-Cologne Deployment History + State Snapshot

Data/state lifecycle work only. No model was fit, retrained, tuned, or calibrated anywhere in
this phase. RF V2 and known-map XGB V3 are referenced only by hash, never loaded.

Manifest: `data/deployment/deployment_history_manifest_v1.parquet`
Consumption audit: `data/deployment/deployment_state_consumption_audit_v1.csv`
Receipt (commit marker): `data/deployment/deployment_state_receipt_v1.json`

## A. Historical replay vs deployment

Two permanently separate timelines now exist. **Historical replay** — pre-Cologne state + RF V2
+ the frozen Phase 8D Monte Carlo simulation + the frozen Phase 8E simulation-vs-reality
evaluation — never changes; Phase 9A re-verifies its full hash record (39 tracked items: the 15
Phase 8E items, the 4 non-series pre-Cologne states, the known-map XGB V3 model, and the sealed
research split/CV-fold/Phase-7-TEST/model-selection-config surface) both before and after this
phase's build, in a single run, and additionally cross-checks the 15 overlapping items against
the values Phase 8E itself already recorded. **Deployment** — legitimate historical matches
(including official Cologne 2026) through the latest locally available data — feeds five brand
new `deployment_post_cologne` state snapshots, built fresh from a deployment-history manifest,
through the same unmodified state engines. Nothing here overwrites a historical-replay artifact.

## B. Deployment-history contract

`data/deployment/deployment_history_manifest_v1.parquet`: one row per raw series `match_id`
(9,923, the full raw universe), `history_status ∈ {included, excluded_showmatch,
excluded_existing_reject}`. Partition: **included 9,800 / excluded_showmatch 1 /
excluded_existing_reject 122** — reconciles exactly to 9,923. The 122 rejects are the pre-
existing Phase 2 structural rejections (116 `missing_bestOf`, 5 `tie`, 1 `missing_score`),
carried forward with their original reason (never re-admitted). Terminology follows the
amendment: this manifest answers *"is this legitimate historical information"* — it does **not**
imply every state engine can consume every included row (see section C/E–I).

## C. Cologne inclusion / showmatch exclusion

Cologne inclusion is a **positive whitelist**, not "cologne_2026 minus showmatch": every one of
the 107 `cologne_2026`-tagged manifest rows is checked against the frozen Phase 8E canonical
106-match actual-results artifact (`cologne_2026_actual_series_results_v1.parquet`). Any row
that resolved to neither "official Cologne match" nor "the frozen showmatch exclusion" would
have been a hard build STOP (an unexplained-row error) — zero such rows occurred.
**Official Cologne series included: 106/106. Showmatch excluded: 1
(`match_id 10094318`, Team Germany vs Team Poland, `non_tournament_showmatch`, per the frozen
Phase 8E reconciliation artifact — never re-derived from tournament name or BO arithmetic).**

## D. Deployment cutoff

`deployment_cutoff = max(datetime)` over `history_status == included` = **2026-06-28 20:00:00**
(naive; the source dataset's timezone convention remains undocumented — same limitation carried
forward from Phase 8D.1/8E, not re-litigated here). Latest included match: `match_id 7303015`
("Super DraculaN Season 1"). This is the **latest locally available historical state through
2026-06-28** — it is explicitly *not* claimed to represent live/current August 2026 data.

## Post-Cologne rows (explicit)

32 raw `post_cologne` rows, all 32 included (legitimate). Distribution:

| Tournament | n |
|---|---|
| Super DraculaN Season 1 | 30 |
| CCT Europe 2026 Series #4 | 1 |
| European Pro League Series 7 | 1 |

| Tier | n |
|---|---|
| tier3 | 31 |
| tier2 | 1 |

## E. Series-state rebuild

`data/features/series_team_state_v1_deployment_post_cologne.json` (+ `.parquet` summary). Fresh
`feature_engine.StateStore` (never resumes the Mode-A development store or patches
`pre_cologne_team_state_v1_full.json`), fed via `build_series_features_v1.build_stream_rows` on
the manifest-gated pool. **106/106 official Cologne matches eligible (≥1 side identity-eligible)
and consumed. 9,800/9,800 included rows reachable.** 775 teams (up from 772 pre-Cologne).

## F. Map-state rebuild

`data/interim/map_state_v1_deployment_post_cologne.json` (+ `.parquet`). Fresh
`map_feature_engine.MapStateStore` via `map_stream_common.load_map_stream`. **99/106 official
Cologne matches eligible and consumed** — 7 legitimately excluded (`eligible_for_state=False`):
those 7 match_ids have **zero surviving `map_base.parquet` rows**, a Phase-2-level structural
fact, independent of team identity, confirmed directly (not invented, not forced to 106).
1,946 team-map states.

## G. Form-state rebuild

`data/interim/form_state_v1_deployment_post_cologne.json` (+ `.parquet`). Fresh
`team_form_engine.TeamFormStateStore`. `process_form_stream` only populates its native
excluded-rows/reason output when `emit_features=True`; since state-building uses
`emit_features=False` (matching the existing Mode B convention exactly), the consumption audit
here is derived directly from the stream's own `team1_eligible`/`team2_eligible` columns, not
from an engine-provided reason string. **106/106 official Cologne matches eligible and
consumed.** 775 teams, 9,800 matches processed.

## H. Player/roster-state rebuild

`data/interim/player_roster_state_v1_deployment_post_cologne.json` (+ `.parquet`). Fresh
`player_roster_feature_engine.PlayerRosterStateStore`. This engine has **no exclusion-tracking
mechanism at all** (confirmed by direct engine inspection — an ineligible series request is
silently skipped with zero trace), so eligibility/consumption is derived directly from the
stream's own `team_eligible`/`has_usable_stats` columns. Roster data is joined from
`map_base.parquet`, so the same 7 match_ids that lack map rows structurally lack player rows
too. **99/106 official Cologne matches eligible and consumed** (identical 7 exclusions as
section F — a consistency cross-check, not a coincidence, since both engines share the same
upstream map-row dependency). 1,189 players, 348 teams. No player data was invented or imputed
for the 7 unreachable matches.

## I. Modern-map-state rebuild

`data/interim/modern_map_state_v1_deployment_post_cologne.json`. Fresh
`modern_map_feature_engine.ModernMapStateStore`, fed by both the map-level and player-level
streams (same map_base.parquet dependency as F/H). **99/106 official Cologne matches eligible
and consumed** — same 7 exclusions. 1,946 team-map, 7,540 player-map, 1,838 team-map-roster
ledgers. This engine has **zero map-name allowlist of any kind** (confirmed by direct code
inspection: any `map_name` string is accepted generically) — see section N for what this means
for the Active Duty limitation.

## J. Representative state changes (pre-Cologne vs deployment)

Descriptive only — no causal attribution of an exact ELO delta to Cologne alone versus the 32
additional post-Cologne matches; both are present in the deployment state simultaneously.

| Team | State | Matches | ELO | Last-10 win rate | Last match datetime |
|---|---|---|---|---|---|
| Team Falcons | pre-Cologne | 192 | 1925.1 | 0.80 | 2026-05-24 |
| | deployment | 199 | 1990.8 | 0.80 | 2026-06-21 (Grand Final) |
| Team Vitality | pre-Cologne | 191 | 2075.6 | 0.80 | 2026-05-16 |
| | deployment | 196 | 2045.3 | 0.60 | 2026-06-19 (lost the semifinal) |
| Team Spirit | pre-Cologne | 200 | 1982.6 | 0.90 | 2026-05-17 |
| | deployment | 208 | 2011.7 | 0.90 | 2026-06-20 |
| FURIA Esports | pre-Cologne | 230 | 1816.5 | 0.40 | 2026-05-15 |
| | deployment | 236 | 1874.3 | 0.70 | 2026-06-21 (runner-up) |
| THUNDERdOWNUNDER | pre-Cologne | **absent (true cold start)** | — | — | — |
| | deployment | 4 | 1492.8 | 0.25 (1/4) | 2026-06-04 (Stage 1 elimination) |

All five deployment rows include both the 106 official Cologne matches and any applicable
post-June-21 matches through the deployment cutoff (Team Falcons' and FURIA's `matches`/`elo`
already reflect the full Cologne run through the Grand Final on 2026-06-21; none of these five
teams appear in the 32 post-Cologne rows, so their deployment numbers are Cologne-only deltas
specifically — this was checked, not assumed).

## K. Cold-start transition

Per Phase 8D, THUNDERdOWNUNDER was a true cold-start team (zero history). Independently
re-confirmed here: **absent entirely from `pre_cologne_team_state_v1_full.json`'s team dict.**
After the deployment rebuild it has real history: **4 series matches, 1 win, ELO 1492.8**,
eliminated from Stage 1 on 2026-06-04 (its 4 recorded matches all fall within the Stage-1 window
— consistent with an early elimination, not a fabricated/forced result). No alias or identity
fallback was introduced to produce this — it is the same `team_identity_policy.csv` entry the
frozen Phase 8B roster already used. This confirms the deployment state genuinely advanced
beyond the frozen pre-event snapshot, for the exact team the historical replay could say nothing
about.

## L. Versioning / hashes

`historical_replay_state: pre_cologne` and `deployment_state: deployment_post_cologne` are
recorded explicitly in every new state file's `meta` block and in the receipt — never conflated.
The receipt hashes: the historical-replay record (39 items, before/after within this run),
deployment build inputs (data: `evaluation_manifest.csv`, `series_base.parquet`,
`map_base.parquet`, `rejected_series_rows.csv`, `team_identity_policy.csv`,
`series_team_form_states_v1.parquet`, the Phase 8E reconciliation and canonical-results
artifacts; code: all 9 engine/stream modules plus every new Phase 9A builder script), and every
new deployment artifact (manifest, consumption audit, 5 state files + summaries).

## M. Validation

Transactional: preflight (abort if a valid receipt already exists) → build the manifest → build
all five states **twice, independently, into two separate staging directories** → require
byte-identical hashes between them (a real determinism proof, not "overwrite and re-hash") →
promote → `validate_phase9a.py` → receipt written **last**. Every builder starts from a **fresh,
empty `StateStore`** on every invocation — none load or patch the pre-Cologne snapshot or the
Mode-A development state. `pytest tests/test_phase9a_post_cologne_deployment_state.py` and the
full repository suite pass; `scripts/validate_phase9a.py` reports all checks passed.

## N. Limitations

- **Timezone**: still naive/undocumented (Phase 8D.1/8E's finding, unchanged).
- **Active Duty map pool**: deployment history ends 2026-06-28, predating any later Active Duty
  change (e.g. a hypothetical Cache re-addition). The resulting map/modern-map states contain no
  legitimate post-change map experience. `modern_map_feature_engine.py` has no map-name
  allowlist mechanism at all — it would accept a new map name generically if one appeared in the
  data, but none has, because the local data doesn't extend past 2026-06-28. This is **not**
  fixed in Phase 9A; a later application-inference phase must define unsupported/new-map
  fallback behavior explicitly rather than silently exposing an unsupported map as if it had
  normal historical support.
- **Engine-specific coverage gaps are expected, not bugs**: 7 of the 106 official Cologne series
  have zero surviving `map_base.parquet` rows (Phase-2-level, pre-existing, independent of
  Cologne). This legitimately caps map/roster/modern-map coverage at 99/106 while series/form
  reach 106/106 — the consumption audit records this with an explicit reason for every affected
  match_id, and the build would have hard-STOPped had any of those 7 shown up as "eligible" for
  an engine that structurally cannot see them.
- **Not current data**: this snapshot is the latest locally available historical state through
  2026-06-28, not a claim about real-world state today.

---

```
POST-COLOGNE DEPLOYMENT STATE = CREATED
HISTORICAL COLOGNE REPLAY = UNCHANGED
MODELS = UNCHANGED
NO RETRAINING
NO API YET
NO PWA YET
```
