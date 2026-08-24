# Phase 8E — IEM Cologne Major 2026: Simulation vs Reality

Evaluation-only phase. No retraining, recalibration, feature change, or state update happens
anywhere below. Cologne results were opened only after the Phase 8D pre-event simulation and
Phase 8D.1 provenance audit were permanently frozen (see immutable-hash re-verification in
section T).

Protocol: `config/phase8e_cologne_simulation_vs_reality_protocol.yaml`
Gate-2 replay artifact: `data/evaluation/cologne_2026_actual_engine_replay_v1.json`
Receipt: `data/evaluation/cologne_2026_simulation_vs_reality_receipt_v1.json`

## A. Evaluation protocol

Frozen *before* any Cologne result was read (`scripts/build_phase8e_protocol.py`). It
predeclares reconciliation rules, the strict actual-outcome-provider match key, the binary
evaluation contract, the frozen top-K tie-break rule (probability → pre-event/VRS seed →
canonical name), the no-model-development policy, and the transactional (stage → validate →
promote → receipt-last) lifecycle. All 22 amendments the user mandated on top of the original
approved plan are encoded as named policy blocks inside the protocol YAML, not left as
convention.

## B. Dataset reconciliation

107 `cologne_2026`-tagged rows in `series_base.parquet`. Official Major structure requires
Stage 1 (33) + Stage 2 (33) + Stage 3 (33) + Playoffs (7) = **106**.

The excluded row: **Team Germany vs Team Poland, 2026-06-21 15:30, BO1** (Germany won 13–3).
`reconciliation_status = non_tournament_showmatch`. Evidence: neither team appears anywhere in
the frozen Phase 8B 32-team roster (mechanically verified: exactly `{Team Germany, Team
Poland}` are the only two raw team names outside the 32-team set, and all 32 roster teams
appear in the raw data); HLTV's own match page is titled *"Team Germany vs. Team Poland at
Showmatch CS"*; press coverage independently describes it as a national all-star exhibition
played the same day as the Grand Final. The BO1-count arithmetic noticed during planning
(expected Stage-1+Stage-2 BO1 = 40, observed 41) was a diagnostic lead only, never treated as
exclusion evidence on its own, per the user's explicit instruction.

The showmatch row is preserved in `data/evaluation/cologne_2026_result_reconciliation_v1.csv`
with `included_in_official_event=False` and full provenance fields — never deleted, and the
raw dataset itself was never modified.

The remaining 106 rows were independently cross-validated three ways before being trusted as
the canonical event table: (1) date clustering alone produces exactly 33/33/33/7; (2) Stage-1
Round-1 pairings match the already-frozen Phase 8B pre-event validation table 8/8; (3) Liquipedia
raw wikitext (`action=raw`, all four stage/playoff pages) matches the dataset's team pairings
and per-round grouping exactly — including reproducing `EXPECTED_GROUPS_BY_ROUND`'s exact
match-count shape (8/8/8/6/3) in every one of the three Swiss stages, derived purely from
chronological replay of real scores, never from asking the engine "where would this fit."

One source-quality note surfaced during this cross-check (see section R).

## C. Actual tournament reconstruction

`scripts/reconcile_cologne_actual_results.py` builds
`data/evaluation/cologne_2026_actual_series_results_v1.parquet` (106 rows): stage, Swiss
round_number/record_group (derived by chronological per-stage replay of real scores) or
playoff round_number (1=QF/2=SF/3=Final), both teams, best_of, scores, and the winner —
**always** derived from `score_team_1 > score_team_2`, never the historically-broken
`team1_series_win` field. No ties; every winner is one of the two paired teams (both asserted).

## D. Engine replay validation (Gate 2 — user-approved)

`scripts/phase8e_actual_outcome_provider.py`'s `ActualOutcomeProvider` (outside
`tournament_engine.py`, which is never modified) fed the real winners through the untouched
Phase 8C engine using the strict key (Swiss: stage/round_number/record_group/unordered team
pair/best_of; Playoffs: stage/round_number/unordered team pair/best_of — zero-or-duplicate
match ⇒ hard STOP, never "pick the first candidate").

**Result: 106/106 matches reproduced exactly** — Stage 1 33/33, Stage 2 33/33, Stage 3 33/33,
Playoffs 7/7. All stage transitions, all 8 playoff seeds, all 4 quarterfinals, both
semifinals, the Grand Final, and the champion (Team Falcons) were reproduced. 106/106 actual
rows consumed exactly once; unused=duplicate=ambiguous=missing=0. Frozen in
`data/evaluation/cologne_2026_actual_engine_replay_v1.json`.

## E. Frozen prediction lookup

`scripts/build_phase8e_predictions.py` deliberately imports only `pandas`,
`phase8e_actual_outcome_provider`, `phase8e_common`, and `_common` — **no** `joblib`, RF V2,
`feature_engine`, RF preprocessing, or pre-Cologne state anywhere in this code path (mirrors
the frozen Phase 8D `build_matrix_lookup` key exactly: `(team_a, team_b, best_of)` against
`canonical_model_name`). **106/106 exact lookups succeeded** — no orientation reversal,
approximate matching, symmetrization, or probability averaging. Output:
`data/evaluation/cologne_2026_actual_match_predictions_v1.parquet`.

## F. Match-level external performance

Binary contract: `y_true = 1` iff actual winner == engine-oriented `team_a` (the better current
tournament seed at pairing time — a representation convention only, unrelated to CT/T side);
`p = probability_team_a`; predicted winner is `team_a` iff `p ≥ 0.5`.

| Metric | Cologne (n=106) | Constant p=0.5 baseline |
|---|---|---|
| Log Loss | **0.6316** | 0.6931 |
| Brier | **0.2208** | 0.2500 |
| ROC-AUC | **0.6968** | 0.5000 |
| Accuracy | 0.6415 | 0.5283 (= observed team_a-win prevalence, not a tuned classifier) |
| Precision | 0.6324 | — |
| Recall | 0.7679 | — |
| F1 | 0.6935 | — |

No threshold optimization, no calibration, no model change of any kind.

### vs. development validation (context only, not new model-selection evidence)

| | Cologne | Phase 4 development validation | external-event metric difference |
|---|---|---|---|
| Log Loss | 0.6316 | 0.6514 | −0.0198 |
| Brier | 0.2208 | 0.2298 | −0.0090 |
| ROC-AUC | 0.6968 | 0.6566 | +0.0402 |
| Accuracy | 0.6415 | 0.6068 | +0.0347 |

Every Cologne metric lands on the favorable side of its development-validation counterpart.
This is reported strictly as an **external-event metric difference** — not "improvement,"
not new tuning evidence, and the earlier model-selection decision is not reinterpreted. It is
one 106-match external event; the prior chronological validation set is larger and covers a
different opponent/BO/tournament-composition distribution, so the two are not directly
comparable in a statistical-significance sense.

## G. Performance by stage

| Stage | n | Accuracy | Log Loss | Brier | ROC-AUC |
|---|---|---|---|---|---|
| Stage 1 | 33 | 0.697 | 0.612 | 0.211 | 0.743 |
| Stage 2 | 33 | 0.576 | 0.648 | 0.229 | 0.673 |
| Stage 3 | 33 | 0.667 | 0.628 | 0.218 | 0.707 |
| Playoffs | 7 | 0.571 | 0.665 | 0.237 | 0.583 |

## H. Performance by best-of

| BO | n | Accuracy | Log Loss | Brier | ROC-AUC |
|---|---|---|---|---|---|
| BO1 | 40 | 0.600 | 0.653 | 0.231 | 0.643 |
| BO3 | 65 | 0.662 | 0.620 | 0.215 | 0.723 |
| BO5 | 1 | 1.000 | 0.539 | 0.174 | N/A | 

**BO5: INSUFFICIENT SAMPLE FOR GENERAL INFERENCE** — the single Grand Final result is not
interpreted as evidence of BO5 model quality in either direction.

## I. Actual-winner probability analysis

Distribution of the frozen probability assigned to whichever team actually won, across all
106 matches: mean **0.547**, median 0.553, Q1 0.433, Q3 0.638, min 0.218, max 0.794.

- **Highest-confidence correct prediction**: Team Spirit over 9z Team (Stage 2, Round 3, BO3),
  p(actual winner) = 0.794.
- **Highest-confidence incorrect prediction / biggest model upset**: Team Vitality vs 9z Team
  (Stage 3, Round 2, BO3) — the model favored Vitality at ~0.782, but **9z Team won**;
  p(actual winner) = 0.218. Both selection criteria (lowest p(actual winner), and the incorrect
  prediction the model was most confident about) picked the same match, as expected.

## J. Realized tournament path log score

`sum_log_probability_actual_winners = -66.953`; `mean_negative_log_probability = 0.6316`,
which equals the overall match Log Loss exactly (cross-checked against `sklearn.log_loss` to
1e-9 tolerance). Called the **conditional realized-path log probability** — conditional on the
deterministic pairing rules induced by preceding outcomes, not an unconditional probability of
"the entire Major." The raw path probability (`exp(sum_log_probability)`) is not foregrounded
anywhere (it would be astronomically small and uninformative).

## K. Swiss terminal-record probabilities vs reality

Across all 48 (team, stage) participations, the frozen probability of the *actually realized*
terminal Swiss record (3-0/3-1/3-2/2-3/1-3/0-3, conditional on participating in that stage):
mean **0.210**, median 0.217, min 0.067, max 0.425. Not treated as 48 independent multiclass
samples — Swiss paths within a stage are mutually coupled (an opponent's record determines
who a team can face next).

## L. Playoff seed probabilities vs reality

| Team | Actual seed | P(actual seed \| reaches playoffs) | Most probable seed | Modal matched? |
|---|---|---|---|---|
| Team Spirit | 1 | 0.193 | 1 | ✓ |
| FURIA Esports | 2 | 0.130 | 8 | ✗ |
| Aurora Gaming | 3 | 0.110 | 8 | ✗ |
| Team Vitality | 4 | 0.122 | 1 | ✗ |
| Team Falcons | 5 | 0.129 | 2 | ✗ |
| BetBoom Team | 6 | 0.137 | 8 | ✗ |
| 9z Team | 7 | 0.169 | 8 | ✗ |
| G2 Esports | 8 | 0.166 | 8 | ✓ |

2/8 modal-seed matches — expected, since each team's seed distribution is spread across 8
possibilities and the modal seed carries well under 50% probability for most teams; P(reach
playoffs) is reported separately per team in the milestone table, never multiplied into these
conditional seed probabilities.

## M. Tournament milestone evaluation

Brier / Log Loss across all 32 teams (unconditional binary reach-X indicators):

| Milestone | n | Brier | Log Loss |
|---|---|---|---|
| Reach playoffs | 32 | 0.134 | 0.399 |
| Reach semifinal | 32 | 0.085 | 0.254 |
| Reach final | 32 | 0.059 | 0.187 |

Top-K set comparisons (frozen tie-break: probability → pre-event/VRS seed → canonical name):

| Set | Overlap | Precision | Recall | Jaccard |
|---|---|---|---|---|
| Top-8 playoffs vs actual 8 | 5/8 | 0.625 | 0.625 | 0.455 |
| Top-4 semifinal vs actual 4 | 2/4 | 0.500 | 0.500 | 0.333 |
| Top-2 final vs actual 2 | 0/2 | 0.000 | 0.000 | 0.000 |
| Top-1 champion vs actual champion | 0/1 | 0.000 | 0.000 | 0.000 |

## N. Actual champion analysis

**Team Falcons** — pre-event championship probability **0.0893**, rank **4/32**. Cumulative
championship probability of the three teams ranked above them: 0.601. Inside top 5 and top 8
by pre-event probability; **not** inside top 1 or top 3. Multiclass champion log score
`-log(P(champion))` = 2.416. Falcons' other frozen milestone probabilities: reach playoffs
0.686, reach semifinal 0.377, reach final 0.194 — a real but clearly-secondary contender, not a
long-shot outsider and not the favorite either.

## O. Deterministic favorite-wins path vs reality

The frozen favorite-wins path's champion was **Team Vitality** (re-replayed via the same
untouched engine + frozen matrix + `FavoriteWinsProvider`; champion matched the frozen
`favorite_path.json` exactly, confirming reproducibility). Team Vitality did **not** win.

Structural milestone comparison (favorite path vs actual):

| Milestone | Overlap |
|---|---|
| Stage-1 advancer set | 6/8 |
| Stage-2 advancer set | 4/8 |
| Stage-3 advancer / playoff set | 4/8 |
| Semifinalist set | 2/4 |
| Finalist set | 0/2 |
| Champion | 0/1 |

Individual match predictions are compared **only** where the exact same (stage, round,
unordered matchup) occurred in both the favorite path and reality: 12 shared matchups, 6/12
(50%) correct. This is kept strictly separate from actual-match model accuracy (section F) —
no downstream favorite-path matchup that never actually happened is scored as a wrong
prediction, per the user's explicit instruction.

**What the Monte Carlo captured that the single deterministic bracket did not**: the
deterministic path is one point estimate (always the >=0.5 favorite), which is why the actual
champion (rank 4, p=0.089) is invisible to it entirely — a single bracket can only ever "predict"
its own greedy favorite as champion. The Monte Carlo distribution, by contrast, explicitly
assigned Team Falcons a real, non-trivial 8.9% share and ranked it inside the top 5 of 32 teams
pre-event — the outcome that actually happened was a plausible, moderately-likely draw from the
distribution the simulation produced, not something outside its support.

## P. Development validation vs external Cologne event

Restated from section F: Cologne's external-event metrics (Log Loss 0.6316, Brier 0.2208,
AUC 0.6968, Accuracy 0.6415) are directionally favorable relative to the frozen Phase 4
development validation (Log Loss 0.6514, Brier 0.2298, AUC 0.6566, Accuracy 0.6068), reported
strictly as external-event differences, not as new evidence for model selection or as proof of
generalization from a single 106-match event.

## Q. Timestamp provenance limitation (carried forward from Phase 8D.1, updated)

Repository timestamps remain naive; the source dataset's timezone semantics remain
undocumented (confirmed again — no timezone-handling code exists anywhere in this repository).
Phase 8E provided **additional independent corroboration** for the UTC+3 wall-clock
interpretation: Liquipedia's raw wikitext shows the Stage 1 opening matches scheduled at
"12:30 CEST," while the dataset records the same matches as "13:30" naive — the identical
one-hour CEST-to-naive-dataset offset Phase 8D.1 found for the tournament's very first match,
now independently reproduced by a second, unrelated real match. This further supports the
UTC+3 wall-clock interpretation. **UTC+3 is still not proven** — the source dataset does not
document its timezone convention anywhere, so the interpretation remains inferred, not
established fact. Independent of this ambiguity: the strict pre-Cologne state ended 63–66 hours
before the event's true first-match instant under either plausible interpretation; no Cologne
state leakage occurred; a possible ≤0.125-day offset in `days_since_last_match` remains a
bounded, unresolved provenance limitation, not something Phase 8E attempted to correct.

## R. Other limitations

**Source-quality note (rendered-page summarization vs raw structured data)**: an early
Liquipedia fetch, rendered as prose by a summarizing intermediary, produced an internally
contradictory statement (claiming Spirit beat Falcons in the semifinal while also stating
Falcons won the Grand Final — impossible). The dataset's own score fields (Spirit 1–2 Falcons)
were correct, independently corroborated by press coverage of the champion and by a second,
raw-wikitext (`action=raw`) fetch of the same page, which was internally consistent. Exact
score/bracket verification throughout this phase relied on **raw structured/wikitext source
data, the dataset's own score fields, and independent press corroboration — never on
summarized rendered-page prose**. This is a note about the rendered/summarized-interpretation
path specifically, not a claim that Liquipedia's underlying data is unreliable.

**Single-event statistical discipline**: this is one real tournament. No claim here should be
read as "the model is proven to generalize" or "the simulation is statistically validated."
Point estimates are reported without inferential confidence intervals — an ordinary IID
bootstrap would understate uncertainty given that the same teams recur across multiple matches
within the 106-match sample (a dependency structure, not independent draws).

**BO5 sample size**: n=1, explicitly excluded from any general-inference claim (section H).

## S. Presentation conclusions

- **How good was the model on actual matches?** 64.2% accuracy, 0.697 AUC, 0.632 Log Loss
  across 106 real series — modestly favorable relative to its own pre-Cologne chronological
  validation, though from a single external event.
- **How good were its probabilities?** Better-calibrated-looking than the constant-0.5
  baseline on every primary metric (Log Loss 0.632 vs 0.693, Brier 0.221 vs 0.250); the mean
  probability assigned to actual winners (0.547) sits only modestly above chance, consistent
  with a genuinely competitive, high-variance Major field rather than a lopsided one.
- **How many actual playoff teams were highly ranked pre-event?** 5 of the 8 real playoff
  teams were in the model's own pre-event top 8 (Jaccard 0.455).
  It correctly favored the eventual champion far more than a coin flip would (8.9% in a 32-team field, rank 4/32) even though it was not the top pick.
- **Was the actual champion a major upset or a plausible contender?** Plausible contender, not
  a long-shot: top-5 of 32 pre-event, real playoffs/semifinal/final probabilities all
  meaningfully above the field baseline, cumulative probability of the three teams ranked above
  it was 60% — not overwhelming favorites either.
- **How close was the deterministic favorite path to reality?** Not very, on the marquee
  outcome — 0/2 finalists, 0/1 champion — but reasonably close earlier in the bracket (6/8
  Stage-1 advancers). This is exactly the shape of degradation expected from a single
  greedy-favorite path in a field with meaningful upset probability at every stage.
- **What did Monte Carlo capture that a single deterministic bracket did not?** A real,
  quantified, non-zero probability for the team that actually won — the deterministic path
  structurally cannot express "4th-most-likely champion, plausible enough to happen," only
  ever "the favorite."

This is one event. None of the above is claimed as statistical proof the system generalizes.

## T. Validation / immutability

- All 15 immutable pre-event artifacts (Phase 8D protocol/matrix/receipts/aggregates, Phase 8B
  tournament YAML, Phase 8C engine, RF V2 model, strict pre-Cologne state, both Phase 8D.1
  files) re-verified byte-unchanged against the Phase 8E protocol's frozen hash baseline —
  0 mismatches.
- `tests/test_phase8e_cologne_simulation_vs_reality.py` and the full repository `pytest` suite
  pass; `scripts/validate_phase8e.py` reports all checks passed.
- `data/evaluation/cologne_2026_simulation_vs_reality_receipt_v1.json` is the commit marker,
  written last, referencing the frozen Phase 8D simulation receipt.

---

```
PHASE 8D PRE-EVENT SIMULATION = UNCHANGED
COLOGNE RESULTS = OPENED ONLY AFTER PHASE 8D FREEZE
MODEL = UNCHANGED
NO POST-EVENT RETUNING
NO COLOGNE TRAINING INGESTION YET
PHASE 7 = UNCHANGED
PHASE 8B = UNCHANGED
PHASE 8C = UNCHANGED
NO API YET
NO PWA YET
```
