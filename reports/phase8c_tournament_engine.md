# Phase 8C — Pure Major Tournament Engine

Engine: `scripts/tournament_engine.py`
Tests: `tests/test_phase8c_tournament_engine.py` (61 tests)
Validator: `scripts/validate_phase8c.py` (48 checks)
Demo: `scripts/run_phase8c_synthetic_demo.py` → `data/tournaments/phase8c_synthetic_trace.json`

This phase implements the Swiss + playoff mechanics frozen in
`config/tournaments/iem_cologne_major_2026_pre_event.yaml` (Phase 8B,
SHA-256 `e481ca4dc3ab5bdf63636ad53eeeba8d3677305b643ecb98b7391e6419383ba3`,
unchanged) as pure, deterministic Python. The engine has no knowledge of
machine learning or of Cologne's real teams: match winners are supplied
entirely through an injected `OutcomeProvider`; the engine only ever asks
"who won this match?" Every result in this report comes from synthetic
teams (`S1_*`/`S2_*`/`S3_*`) or hand-built letter fixtures — no real
Cologne prediction of any kind occurs anywhere in Phase 8C.

## Engine architecture

Mirrors the repository's existing "engine" convention
(`team_form_engine.py`, `map_feature_engine.py`): plain dataclasses hold
state (`TeamStageState`, mutable), module-level pure functions operate on
that state. `MatchSpec` and `MatchResolution` are **frozen** dataclasses —
a deliberate departure from the repo's usual `@dataclass` convention,
justified by the round-start-atomicity requirement: once a round's
pairings are generated, nothing about them (including the pre-match
record/Difficulty-Score snapshot) may ever be mutated again, and Python
enforces that at the type level rather than by convention alone.

`load_frozen_rules()` reads only `swiss_rules` / `match_formats` /
`playoff_bracket` / `metadata` from the frozen YAML and verifies its
SHA-256 against the frozen constant — it **never** reads the YAML's
`participants` key, anywhere in this module. This makes "accidentally
running a real Cologne team through an outcome provider" structurally
impossible in this phase, not just a documented policy — enforced by a
static test/validator check (`'participants'` does not appear in
`tournament_engine.py`'s source at all).

**Outcome-provider interface**: `OutcomeProvider.resolve_match(match) ->
MatchResolution`. Three synthetic providers ship in this module:
`ScriptedOutcomeProvider` (winner keyed by `match_id`), `HigherSeedWinsProvider`
(fully deterministic — the better seed always wins), and
`SeededRandomOutcomeProvider` (owns one `np.random.default_rng(seed)`
instance, advanced once per call — used only for stress testing). Every
provider's output passes through `_finalize_resolution()`, which validates
`winner ∈ {team_a, team_b}` and derives `loser` automatically.

**Round-start atomicity** (`run_swiss_round`): seed order and Difficulty
Score are recomputed from the complete post-previous-round state; all of a
round's matchups are generated and frozen from that one snapshot; only then
are outcomes resolved and results applied. No pairing is ever generated
after another same-round result is known.

## Pairing rule implementation

**Round 1**: direct `1v9, 2v10, ..., 8v16` on `initial_stage_seed`
(`round1_pairing_ids`).

**Round 2/3** (`constrained_pool_pairing`): exhaustive backtracking over a
same-record pool (max 8 teams, ≤105 perfect matchings — trivial to
enumerate in full). Minimizes total rematches first; among arrangements
achieving that minimum, prefers, at every step, the highest remaining seed
against the worst available opponent, moving toward better opponents only
when required.

**Round 4/5** (`priority_table_pairing`): the frozen 15-row priority table
applied to a six-team pool ranked 1–6 by pool-relative `current_seed`. The
first row with zero rematches is used; if none of the 15 achieves zero, the
earliest row achieving the true minimum across all 15 is used instead.

**Round dispatch is by round number, not pool size** (per amendment #2):
`EXPECTED_GROUPS_BY_ROUND` is an *assertion*, checked before any pairing
runs —

| Round | Expected active record pools |
|---|---|
| 1 | `(0,0): 16` |
| 2 | `(1,0): 8`, `(0,1): 8` |
| 3 | `(2,0): 4`, `(1,1): 8`, `(0,2): 4` |
| 4 | `(2,1): 6`, `(1,2): 6` |
| 5 | `(2,2): 6` |

A shape that doesn't match raises a descriptive `ValueError` naming the
round and the expected-vs-actual groups — the algorithm is never inferred
from an unexpected size.

### Rematch fallback (the Valve rulebook's under-specified "if possible" clause)

Documented engine convention: (1) search for a zero-rematch arrangement;
(2) if one exists, it is always used, no exceptions; (3) only if exhaustive
search proves **zero** fully rematch-free arrangement exists does the
engine fall back to the minimum-rematch arrangement, chosen deterministically,
with `rematch_fallback_used`/`rematch_count`/`priority_pattern_used`
recorded on every affected `MatchSpec` so the fallback is always visible in
the trace, never silent. Proven by targeted unit tests that construct
genuinely impossible (K4-edge-decomposition-style) histories for both a
4-team Round-2/3 pool and a 6-team Round-4/5 pool — both correctly report
`rematch_fallback_used=True, rematch_count=1`.

## Difficulty Score

`difficulty_score(team_id, states)` sums, over every stage-local opponent
already faced, `opponent.wins - opponent.losses` **using each opponent's
current record** — never cached. Test proof: with A's opponents fixed at
{B, C}, `difficulty_score("A", ...)` recomputes from 0 → 1 → 4 purely from
B/C's records changing, with zero new matches played by A
(`test_difficulty_score_recomputes_when_opponent_record_changes_later`).

The pre-round value used for a specific pairing decision is snapshotted
once, at pairing time, into that match's frozen `difficulty_a_before`/
`difficulty_b_before` fields — proven never to drift even after many
further rounds change the live team's Difficulty Score
(`test_match_spec_difficulty_before_snapshot_never_changes_after_later_rounds`),
and structurally guaranteed by `MatchSpec` being a frozen dataclass
(attempting to mutate it raises `dataclasses.FrozenInstanceError`).

## Seeding

One canonical ordering function, `compute_seed_order`, used for both
mid-stage pairing and final stage order — sort key `(-wins, losses,
-difficulty_score, initial_stage_seed)`. Verified: equal record + higher
Difficulty Score → better seed; equal record + equal Difficulty Score →
lower initial seed wins the tie; among advancers, `3-0 > 3-1 > 3-2`; among
eliminated, `2-3 > 1-3 > 0-3` — all via the identical key, no special-casing.

## Format selection

`select_format(stage, wins, losses, rules)`: Stage 3 always returns
`bo3`. Stage 1/2 returns `bo3` iff `wins+1 == advancement_wins` or
`losses+1 == elimination_losses`, else the frozen default (`bo1`). Per
amendment #4, `_build_swiss_matches` asserts `team_a.record ==
team_b.record` **before** calling `select_format` — a mismatched pairing
raises immediately rather than silently deriving a format from one side
only.

## Stage transitions

`next_stage_entrants`: direct entrants keep seeds 1–8 as given; the 8
Swiss advancers — already ordered by `finalize_stage`'s explicit
record → Difficulty Score → initial-seed order, never by qualification
order or match-execution order — become seeds 9–16 in that order. Record,
opponent history, and status all reset to a fresh `TeamStageState` per
stage; nothing carries over.

## Playoff bracket

QFs `[1v8, 4v5, 2v7, 3v6]` (Bracket A = QF1/QF2, Bracket B = QF3/QF4, per
the frozen YAML's own `bracket_a`/`bracket_b` lists — parsed, not
hardcoded). SF1 = winner(QF1) vs winner(QF2); SF2 = winner(QF3) vs
winner(QF4); Final = winner(SF1) vs winner(SF2). QF/SF Bo3, Final Bo5, no
reseeding, no third-place match — exactly 7 matches (4+2+1).

**Bracket lineage vs. team_a/team_b orientation** (amendment #5): the
frozen `team_a = better seed` convention still applies to every playoff
match, but `source_a`/`source_b`/`bracket_side` travel *with* the team they
describe, so reorientation never loses lineage. Proven with a scripted
upset (`test_playoff_bracket_lineage_survives_team_a_team_b_reorientation`):
seed 5 upsets seed 4 in QF2; in the resulting semifinal, seed 1 (from QF1)
is `team_a` and the upset winner, seed 5 (from QF2), is `team_b` — and
`source_a == "winner_playoff_qf_01"`, `source_b == "winner_playoff_qf_02"`
correctly regardless of which one ended up `team_a`.

## Synthetic deterministic run

`scripts/run_phase8c_synthetic_demo.py` runs one full Major over the
synthetic 32-team fixture (`S1_01..S1_16`, `S2_01..S2_08`, `S3_01..S3_08`)
via `HigherSeedWinsProvider`, writing `data/tournaments/phase8c_synthetic_trace.json`.

- Champion: `S3_01` (the best overall seed, as expected under a
  higher-seed-always-wins provider).
- Canonical trace SHA-256 (via `canonical_json_bytes` —
  `json.dumps(..., sort_keys=True, separators=(",", ":"))`, no timestamps/UUIDs):
  **`3277cf70ad02c7e27c534a1adf4018a36c4b0dd4602f73bbcd3a9ac7dc4ed7e7`**
- Run twice: identical matchups, identical records, identical stage
  transitions, identical playoff bracket, identical champion, identical
  canonical hash. The on-disk trace file is itself byte-identical across
  regenerations.
- Every `team_id` in the trace matches `S[123]_\d\d`; zero real Cologne
  participant `display_name`/`canonical_model_name` strings appear anywhere
  in the trace (checked against the frozen YAML's `participants` section —
  the one narrow, validation-only exception to "never reads participants,"
  confined entirely to tests/validator, never to `tournament_engine.py` or
  the demo script).

## Stress test

`run_stress_test(n_runs=500, base_seed=42)`: 500 full synthetic Majors
through `SeededRandomOutcomeProvider` (a distinct seed per run), with every
engine invariant enforced internally on every run.

| Metric | Value |
|---|---|
| Runs | 500 |
| Total Swiss stages | 1,500 |
| Total matches | 53,000 |
| No-rematch pairings | 49,500 |
| Rematch-fallback events (natural) | 0 |
| Max rematches in any fallback pairing | 0 |
| **Invariant failures** | **0** |
| Max rounds per stage | 5 |
| Elapsed | <1s |

A 50/50 random provider never naturally forced an unavoidable rematch in
this batch — expected, given the small pool sizes involved — which is why
the targeted fallback unit tests (constructed impossible-zero-rematch
histories, see "Rematch fallback" above) exist specifically to exercise
that code path independent of the stress test.

## Invariants

Enforced inside `run_swiss_round`/`finalize_stage` (raising `ValueError` on
violation, never silently continuing) and separately exercised by
standalone unit tests calling `validate_round_invariants`/
`validate_stage_completion` directly:

- every matchup pairs two different teams; no team appears twice in a round
- only teams active before the round are paired; every active team appears
  exactly once
- paired teams share an identical pre-match record
- total wins-delta == total losses-delta == matches completed, each round
- exactly 16 unique entrants, exactly 8 advanced, exactly 8 eliminated,
  exactly 33 matches, every team played between 3 and 5 matches, final
  order contains all 16 exactly once

## Validation

- `pytest tests/test_phase8c_tournament_engine.py`: **61/61 passed**
  (per-rule unit tests before integration tests: Round 1 exact; Round 2/3
  incl. dead-end avoidance and forced fallback; Round 4/5 incl.
  row-1-invalid→row-2-invalid→row-3-selected and forced fallback; dynamic
  Difficulty Score; seed tie-breaks; the full Stage 1/2 format table;
  Stage 3 always-Bo3; cross-stage rematch independence; stage-transition
  seeding; playoff bracket/format/lineage; canonical serialization;
  double-run determinism; full-stage 33-match/8-8 invariants; zero-real-team
  trace check; 500-run stress test).
- Full repository `pytest`: **467/467 passed** (406 pre-Phase-8C + 61 new,
  zero regressions).
- `python scripts/validate_phase8c.py`: **48/48 checks passed** (frozen-hash,
  Phase 1–8B artifacts byte-unchanged, `src/` unchanged, no-ML-import,
  no-participants-access, per-rule spot checks, full-stage/playoff
  integration checks, determinism, zero-real-team-in-trace, 500-run stress
  test).

---

**PURE TOURNAMENT ENGINE = IMPLEMENTED**
**ML MODEL = NOT CONNECTED**
**COLOGNE RESULTS = UNOPENED**
**REAL COLOGNE SIMULATION = NOT RUN**
**PHASE 7 = UNCHANGED**
**PHASE 8B YAML = UNCHANGED**
**NO API**
**NO PWA**
