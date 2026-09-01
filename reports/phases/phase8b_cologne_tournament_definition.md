# Phase 8B — IEM Cologne Major 2026 Pre-Event Tournament Definition (source audit)

Frozen artifact: `config/tournaments/iem_cologne_major_2026_pre_event.yaml`
Frozen YAML SHA-256: `e481ca4dc3ab5bdf63636ad53eeeba8d3677305b643ecb98b7391e6419383ba3`
Source manifest: `data/tournaments/iem_cologne_major_2026_sources.json`
Validator: `scripts/validate_phase8b.py`

This phase freezes the *structure* of IEM Cologne Major 2026 exactly as it
was knowable before the tournament's first match (2026-06-02T13:30:00, the
repository's established Cologne cutoff). No Swiss engine, match simulator,
or model is implemented or called here. No match winner, score, standing,
qualifier, champion, map result, player statistic, or result-derived seed
calculation appears anywhere in this file or its supporting artifacts.

---

## A. Historical Valve rulebook revision (amendment #1)

The Major Supplemental Rulebook is a living GitHub document
(`ValveSoftware/counter-strike_rules_and_regs`) that may have changed after
Cologne. Rather than trust today's `main`, the exact pre-cutoff revision was
pinned:

| Field | Value |
|---|---|
| Repository | `ValveSoftware/counter-strike_rules_and_regs` |
| File | `major-supplemental-rulebook.md` |
| Commit SHA | `a22f91da503d3afc00045fdad339bc6a3cd19c98` |
| Commit date | `2025-08-05T21:40:53Z` |
| Commit URL | https://github.com/ValveSoftware/counter-strike_rules_and_regs/commit/a22f91da503d3afc00045fdad339bc6a3cd19c98 |
| Content SHA-256 | `6bcbef90a81eb20cc1be0e728a2a4b32ded06c63874b907c8b2d3f799150f3b0` |

**Selection procedure:** queried the GitHub commits API for
`major-supplemental-rulebook.md` with `until=2026-06-02T13:30:00Z`. The most
recent of the 5 returned commits is the one above; no later pre-cutoff
commit exists, so today's `main` was never used as a stand-in.

**Sections used** (verbatim/derived, frozen as reference metadata — no
Swiss/veto engine implemented): Invitations / Regional Standing (VRS);
Major Format (stage structure); Swiss Bracket (advancement/elimination,
rematch avoidance, Round 2/3 rule, Round 4/5 15-row priority table); Initial
Swiss Match-ups (1v9...8v16); Single Elimination Bracket (playoff bracket
seeding, generic best-of-three); Map Pick-Ban (Bo1 7-step and Bo3 9-step
veto procedures); Seeding (Pre-event Seeding, Stage Seeding, Mid-stage Seed
Calculation, Difficulty Score / Buchholz, with the rulebook's own worked
example).

**Important caveat:** this pinned revision does **not** distinguish a Grand
Final format from other single-elimination matches — it states all playoff
matches are best-of-three, full stop. The Cologne Grand Final Bo5 format is
therefore **not** derived from this document; see section D below for its
independent, event-specific sourcing.

---

## B. Invitation VRS vs. seeding VRS (amendment #2)

These are two distinct Valve Regional Standings snapshots, deliberately
never collapsed:

| Field | Value | Primary source |
|---|---|---|
| `invitation_vrs_date` | 2026-04-06 | `ValveSoftware/counter-strike_regional_standings`, `invitation/2026/standings_global_2026_04_06.md` |
| `seeding_vrs_date` | 2026-05-04 | `ValveSoftware/counter-strike_regional_standings`, `invitation/2026/standings_global_2026_05_04.md` |

Both files were independently confirmed to exist as separate, distinctly
dated primary files in Valve's own GitHub repository (directory listing via
`gh api repos/ValveSoftware/counter-strike_regional_standings/contents/invitation/2026`).
The 2026-04-06 file determined which 32 rosters were invited and to which
stage; Liquipedia's Cologne overview page explicitly states seeding uses
the 2026-05-04 snapshot ("All teams will be seeded solely based on Valve's
Global Standings within their opening stage," linking directly to that
file). The 2026-05-04 file's global rank order was fetched directly and
used to **independently cross-validate every seed table** below — full
32/32 agreement, described in section C.

---

## C. Team identity resolution (32/32 confirmed, amendment #6)

No edit-distance or general fuzzy matching was used anywhere. Every one of
the 32 teams resolves via one of: exact match against
`data/interim/team_identity_policy.csv`, deterministic case normalization,
or an explicit manually-recorded one-to-one mapping with a stated,
principled justification (never "closest string").

| # | Display name (tournament) | Canonical model name | Method | Stage | Seed | VRS rank (2026-05-04) |
|---|---|---|---|---|---|---|
| 1 | Team Vitality | Team Vitality | exact | Stage 3 | 1 | 1 |
| 2 | Natus Vincere | Natus Vincere | exact | Stage 3 | 2 | 2 |
| 3 | Team Falcons | Team Falcons | exact | Stage 3 | 3 | 3 |
| 4 | The MongolZ | The Mongolz | case_normalization | Stage 3 | 4 | 7 |
| 5 | PARIVISION | PARIVISION | exact | Stage 3 | 5 | 8 |
| 6 | Aurora Gaming | Aurora Gaming | exact | Stage 3 | 6 | 9 |
| 7 | FURIA | FURIA Esports | manual_roster_disambiguation | Stage 3 | 7 | 10 |
| 8 | MOUZ | MOUZ | exact | Stage 3 | 8 | 11 |
| 9 | FUT Esports | FUT Esports | exact | Stage 2 | 1 | 4 |
| 10 | Team Spirit | Team Spirit | exact | Stage 2 | 2 | 5 |
| 11 | Astralis | Astralis | exact | Stage 2 | 3 | 6 |
| 12 | G2 Esports | G2 Esports | exact | Stage 2 | 4 | 12 |
| 13 | Legacy | Legacy | exact | Stage 2 | 5 | 14 |
| 14 | paiN Gaming | paiN Gaming | exact | Stage 2 | 6 | 17 |
| 15 | Monte | Monte | exact | Stage 2 | 7 | 18 |
| 16 | 9z Team | 9z Team | exact | Stage 2 | 8 | 19 |
| 17 | GamerLegion | GamerLegion | exact | Stage 1 | 1 | 13 |
| 18 | B8 | B8 Esports | manual_suffix_expansion | Stage 1 | 2 | 16 |
| 19 | HEROIC | Heroic | case_normalization | Stage 1 | 3 | 20 |
| 20 | BetBoom Team | BetBoom Team | exact | Stage 1 | 4 | 21 |
| 21 | BIG | BIG | exact | Stage 1 | 5 | 23 |
| 22 | M80 | M80 | exact | Stage 1 | 6 | 24 |
| 23 | MIBR | MIBR | exact | Stage 1 | 7 | 27 |
| 24 | SINNERS Esports | Sinners Esports | case_normalization | Stage 1 | 8 | 30 |
| 25 | NRG | NRG Esports | manual_suffix_expansion | Stage 1 | 9 | 31 |
| 26 | TYLOO | TYLOO | exact | Stage 1 | 10 | 34 |
| 27 | Sharks Esports | Sharks Esports | exact | Stage 1 | 11 | 37 |
| 28 | Gaimin Gladiators | Gaimin Gladiators | exact | Stage 1 | 12 | 40 |
| 29 | Team Liquid | Team Liquid | exact | Stage 1 | 13 | 47 |
| 30 | Lynn Vision Gaming | Lynn Vision Gaming | exact | Stage 1 | 14 | 49 |
| 31 | THUNDERdOWNUNDER | THUNDERdOWNUNDER | exact | Stage 1 | 15 | 56 |
| 32 | FlyQuest | FlyQuest | exact | Stage 1 | 16 | 74 |

**Formerly ambiguous families, now resolved with a principled, non-fuzzy
justification:**

- **FURIA** — the policy file has 3 FURIA-family candidates (`FURIA
  Academy` / `FURIA Esports` / `FURIA Esports Female`, explicitly flagged
  "do not merge"). The tournament roster (yuurih, KSCERATO, FalleN,
  molodoy, YEKINDAR) is FURIA's known premier male CS2 lineup, which
  uniquely identifies `FURIA Esports` — resolved by roster identity, not
  string distance.
- **B8** and **NRG** — each has exactly one candidate in the policy file
  (`B8 Esports`, `NRG Esports`); the tournament display name simply omits
  the generic "Esports" suffix. Single unambiguous candidate, recorded as
  an explicit manual mapping.
- **Team Falcons, Aurora Gaming, Team Spirit, G2 Esports, FUT Esports, 9z
  Team, BetBoom Team** — each is an **exact** string match to the tournament
  display name (no disambiguation needed at all; the previously-suspected
  ambiguity did not materialize once the real participant list was
  fetched).

Three teams (Aurora Gaming, Team Spirit, Heroic) carry a pre-existing
Phase 2.5 policy-file caveat (`KEEP_AS_SINGLE_TEAM` — a possible same-org
identity collision noted in `reports/unresolved_team_review.md`). This does
not block resolution for Phase 8B (a structural, not predictive, freeze)
and is carried forward as a footnote in the YAML, exactly as it already
exists in the policy file.

**Participant/stage-assignment sourcing (amendment #4):** the 32-team,
per-stage roster was independently fetched from Liquipedia's raw wikitext
Participants section (not through a summarizing intermediary, to avoid
name-hallucination risk — see section G) and cross-checked against the
originally supplied stage-entry hypothesis: **32/32 agreement, no
conflicts.**

---

## D. Match format overrides (amendments #7, #8)

| Stage/round | Format | Source type | Source |
|---|---|---|---|
| Stage 1 / Stage 2 | Bo1 default; Bo3 iff winner advances or loser is eliminated | generic Valve rule | pinned rulebook |
| **Stage 3** | **ALL matches Bo3** | **Cologne event override** | HLTV.org 2026-02-23; csgo.com 2026-02-23; strafe.com 2026-05-01; escharts.com 2026-05-26 (4 independent dated pre-event sources) |
| Quarterfinal / Semifinal | Bo3 | generic Valve rule | pinned rulebook |
| **Grand Final** | **Bo5** | **Cologne/Major-era event override** | escharts.com, published 2026-05-26 (before cutoff): *"Single-Elimination bracket. All matches (excl. Grand Final) are Bo3. The Grand Final is Bo5."* Corroborated by Liquipedia's identical wording and by precedent (Bo5 grand finals were introduced the immediately preceding Major, StarLadder Budapest Major 2025, and continued at Cologne). |

Per amendment #7, the Grand Final Bo5 format was **not** derived from the
generic Valve playoff rule (which the pinned rulebook states as uniformly
best-of-three with no Grand Final exception), and a reliable, dated
pre-event source was found — so this fact is resolved, not left
unresolved.

Per amendment #8, the Stage-3 all-Bo3 override references dated pre-event
reporting specifically about the format change, never inferred from played
results — 4 independently dated sources corroborate it (earliest:
2026-02-23, over three months before the event).

---

## E. Stage-1 seed table and the 8/8 opening-pairing validation (amendment #3)

**Seed derivation and independent cross-validation.** Liquipedia's
per-team `tiebreaker=` field (used in its Swiss standings template) runs in
ascending worst-to-best order for a 16-team stage — i.e. `seed = 17 -
tiebreaker`. This directionality is not self-evident from the field alone
(the pairing-validity check below is symmetric under the field's two
possible directions), so it was independently confirmed by reproducing the
**exact same 16-team order** from Valve's own primary
`standings_global_2026_05_04.md` rank column — 16/16 agreement for Stage 1,
and 8/8 agreement for both Stage 2 and Stage 3 direct entrants (see
section C's rank column).

**Stage 1 seed table (1–16):**

| Seed | Team | VRS rank |
|---|---|---|
| 1 | GamerLegion | 13 |
| 2 | B8 Esports | 16 |
| 3 | Heroic | 20 |
| 4 | BetBoom Team | 21 |
| 5 | BIG | 23 |
| 6 | M80 | 24 |
| 7 | MIBR | 27 |
| 8 | Sinners Esports | 30 |
| 9 | NRG Esports | 31 |
| 10 | TYLOO | 34 |
| 11 | Sharks Esports | 37 |
| 12 | Gaimin Gladiators | 40 |
| 13 | Team Liquid | 47 |
| 14 | Lynn Vision Gaming | 49 |
| 15 | THUNDERdOWNUNDER | 56 |
| 16 | FlyQuest | 74 |

**8/8 opening-pairing validation** — applying the generic `1v9, 2v10, ...,
8v16` rule to this seed table against the **published Round 1 schedule**
(pairing fields only — no score, map, or winner data extracted from the
same page, per the explicit same-page caveat):

| Seed | Team | Opp. seed | Expected opponent | Published opening opponent | Match |
|---|---|---|---|---|---|
| 1 | GamerLegion | 9 | NRG Esports | NRG Esports | ✅ |
| 2 | B8 Esports | 10 | TYLOO | TYLOO | ✅ |
| 3 | Heroic | 11 | Sharks Esports | Sharks Esports | ✅ |
| 4 | BetBoom Team | 12 | Gaimin Gladiators | Gaimin Gladiators | ✅ |
| 5 | BIG | 13 | Team Liquid | Team Liquid | ✅ |
| 6 | M80 | 14 | Lynn Vision Gaming | Lynn Vision Gaming | ✅ |
| 7 | MIBR | 15 | THUNDERdOWNUNDER | THUNDERdOWNUNDER | ✅ |
| 8 | Sinners Esports | 16 | FlyQuest | FlyQuest | ✅ |

**Result: 8/8 exact agreement.** No seed was adjusted to fit a pairing;
seeds were derived first (from VRS rank + the independently-resolved
tiebreaker direction), then validated against the published schedule.

**Stage 2 / Stage 3 direct-entrant seed tables (1–8 only).** Seeds 9–16 in
both stages belong to Swiss advancers from the prior stage and are
determined by in-stage results (Mid-stage Seed Calculation: record, then
Difficulty Score, then initial seed) — **not knowable pre-event**, and are
therefore intentionally left unassigned in the frozen YAML.

| Stage 2 seed | Team | VRS rank | | Stage 3 seed | Team | VRS rank |
|---|---|---|---|---|---|---|
| 1 | FUT Esports | 4 | | 1 | Team Vitality | 1 |
| 2 | Team Spirit | 5 | | 2 | Natus Vincere | 2 |
| 3 | Astralis | 6 | | 3 | Team Falcons | 3 |
| 4 | G2 Esports | 12 | | 4 | The Mongolz | 7 |
| 5 | Legacy | 14 | | 5 | PARIVISION | 8 |
| 6 | paiN Gaming | 17 | | 6 | Aurora Gaming | 9 |
| 7 | Monte | 18 | | 7 | FURIA Esports | 10 |
| 8 | 9z Team | 19 | | 8 | MOUZ | 11 |

---

## F. Map pool (amendment #5)

7/7 agreement against the originally supplied hypothesis: **Ancient,
Anubis, Dust2, Inferno, Mirage, Nuke, Overpass.** Confirmed via escharts.com
(2026-05-26) and corroborated by Liquipedia's overview page and multiple
independently searched outlets. This is the standard CS2 Active Duty group
in effect at the time (Anubis replaced Train ahead of this Major), not a
Cologne-specific selection. Recorded as metadata only — **not** a
prediction input for the pre-veto series simulation.

---

## G. Data-quality note: raw wikitext vs. summarized fetches

An early fetch of Liquipedia's Stage 1 page through a summarizing
intermediary produced a **hallucinated** seed table (e.g. "FaZe Clan (FQ)"
at seed 1 and "Gambit Legends (gl)" at seed 16) that directly contradicted
an already-confirmed fact (FaZe did not qualify for this Major). All
subsequent Liquipedia reads in this phase used direct `curl` fetches of raw
wikitext (`action=raw`), read and parsed by hand, specifically to eliminate
this risk. Every fact in the frozen YAML that originates from Liquipedia
traces to a raw-wikitext read, never a summarized one.

---

## H. Source manifest

`data/tournaments/iem_cologne_major_2026_sources.json` contains one record
per sourced fact (15 records covering 13 distinct fact_ids, several with
more than one corroborating dated source): `valve_major_rulebook`,
`cologne_invitation_vrs`, `cologne_seeding_vrs`,
`cologne_stage_entry_assignments`, `cologne_stage1_seed_table`,
`cologne_stage1_opening_matchups`, `cologne_stage2_seed_table`,
`cologne_stage3_seed_table`, `cologne_stage3_bo3_override` (×3 sources),
`cologne_grand_final_bo5_override`, `cologne_map_pool`,
`cologne_identity_furia`, `cologne_identity_b8_nrg_suffix`. Every record
has `known_before_cutoff: true`. No Cologne result data appears anywhere in
the manifest.

---

## I. BO5 veto procedure (unresolved, non-blocking)

No Cologne-specific or generic BO5 map-veto procedure was found in any
sourced document (the pinned Valve rulebook only defines Bo1 and Bo3 veto
sequences). Recorded in the YAML as `status: not_implemented,
source_required: true`. Per the approved plan, this does **not** block the
tournament-definition freeze or the pre-veto series simulation, which never
consumes veto or map data as a prediction input.

---

## J. Prediction-engine contract

```yaml
prediction_engine: pre_veto_series      # corrected per user instruction — NOT known_map_preferred
historical_prediction_mode: pre_veto
model_id: series_random_forest_v2
model_artifact: models/random_forest_v2.joblib
maps_used_as_prediction_input: false
state_policy: frozen_pre_event
state_source: data/features/pre_cologne_team_state_v1_full.json
state_updates_during_simulation: false
actual_results_available_to_predictor: false
```

This matches the Phase 8A audit's recommended MODE A application candidate
(Random Forest V2, series-V1 features, strict pre-Cologne state) exactly.

---

## K. Validation

`scripts/validate_phase8b.py` checks: historical rulebook revision recorded
and dated before cutoff; `invitation_vrs_date`/`seeding_vrs_date` recorded
as distinct fields with a justified note; participant/seed counts (32
total, 16/8/8 per group, no duplicates); all 32 identities
`resolution_status: confirmed` via a non-fuzzy method only; all 8 Stage-1
pairings reproduce the 1v9...8v16 rule exactly against the published
schedule; Swiss advancement constants (3 wins / 3 losses); map pool 7
uniques; Stage-3 Bo3 and Grand-Final Bo5 each carry a `cologne_event_override`
source type with a non-trivial dated source note; prediction-engine
contract fields match section J exactly; source manifest fully populated
and `known_before_cutoff: true` throughout; a static token scan across every
new Phase 8B file for forbidden result-shaped fields; and byte-level
unchanged checks on every pre-existing artifact this phase reads
(`team_identity_policy.csv`, the Phase 8A report).

`pytest` (new Phase 8B tests + full existing suite) and
`python scripts/validate_phase8b.py` both pass — see the accompanying
summary message for exact counts.

---

## Summary (return-block format)

**TOURNAMENT**
IEM Cologne Major 2026, Cologne Germany, 2026-06-02 to 2026-06-21, 32 teams,
4-stage structure (Stage 1 → Stage 2 → Stage 3 → Playoffs), $1,250,000
prize pool context only (not modeled).

**SEEDING**
`invitation_vrs_date = 2026-04-06`, `seeding_vrs_date = 2026-05-04` (kept
distinct, both independently confirmed as separate primary Valve files).
Stage 1 full 1–16 seed table derived and 8/8-validated against the
published opening pairings. Stage 2/3 direct-entrant seeds (1–8) derived
and cross-validated; advancer seeds (9–16) intentionally left unassigned —
not knowable pre-event.

**SWISS RULES**
3 wins advance / 3 losses eliminate; rematch avoidance within a stage;
initial pairing 1v9...8v16; Round 2/3 highest-vs-lowest-available;
Round 4+ frozen 15-row priority table; Mid-stage Seed Calculation (record →
Difficulty Score/Buchholz → initial seed), all frozen as reference rules
only, no engine implemented.

**FORMATS**
Stage 1/2: Bo1 default, Bo3 at advancement/elimination stakes (generic).
Stage 3: **all Bo3** (Cologne override, 4 dated pre-event sources).
Playoffs: QF/SF Bo3 (generic), **Grand Final Bo5** (Cologne/Major-era
override, dated pre-event source). Bracket A 1v8/4v5, Bracket B 2v7/3v6, no
reseeding. Bo1/Bo3 veto procedures frozen verbatim; Bo5 veto
`not_implemented/source_required` (non-blocking).

**PREDICTION CONTRACT**
`prediction_engine: pre_veto_series`, `model_id: series_random_forest_v2`,
`maps_used_as_prediction_input: false`, `state_policy: frozen_pre_event`,
no state updates during simulation, no actual results available to the
predictor.

**VALIDATION**
`scripts/validate_phase8b.py` — all checks pass. `pytest` — full suite
passes. YAML SHA-256: `e481ca4dc3ab5bdf63636ad53eeeba8d3677305b643ecb98b7391e6419383ba3`.

**COLOGNE RESULTS = UNOPENED**
**TOURNAMENT DEFINITION = FROZEN PRE-EVENT INFORMATION ONLY**
**MODEL = UNCHANGED**
**PHASE 7 = UNCHANGED**
**NO SIMULATION YET**
**NO PWA YET**
