# Phase 8D — Frozen Pre-Event IEM Cologne Major 2026 Simulation

**COLOGNE RESULTS = UNOPENED.** This report answers exactly one question:
*what would the frozen prediction system have believed immediately before
IEM Cologne Major 2026 began?* Every number below comes from the frozen
RF V2 series model, the frozen strict pre-Cologne state, and 50,000 Monte
Carlo simulations of the frozen tournament engine. No actual Cologne match
result, score, standing, qualifier, bracket, or champion appears anywhere
in this document or in any artifact it references.

Simulation receipt: `data/evaluation/cologne_2026_pre_event_simulation_receipt_v1.json`
Protocol: `config/phase8d_cologne_pre_event_simulation_protocol.yaml`
(hash `ec0d57aa98fa63521ec04d7fa4d3c6fa2a4e24a371a42a0c1bed51359e50a7a3`)

---

## A. Frozen input contract

| Input | Value |
|---|---|
| Model | `series_random_forest_v2` (`models/random_forest_v2.joblib`, `sklearn.ensemble.RandomForestClassifier`, unmodified) |
| Preprocessing | `data/modeling/random_forest_preprocessing_v2.json` (17 raw → 19 transformed features) |
| Selected config | `data/modeling/random_forest_v2_selected_config.json` (hyperparameters only, unmodified) |
| Strict state | `data/features/pre_cologne_team_state_v1_full.json` (max history `2026-05-30T19:30:00`, strictly pre-cutoff) |
| Prediction cutoff | `2026-06-02T13:30:00`, identical for every one of the 2,976 predictions - never advanced |
| Tier | `tier1` (see rationale below) |
| Prediction mode | `pre_veto` — maps never used as a prediction input |
| Tournament definition | `config/tournaments/iem_cologne_major_2026_pre_event.yaml` (SHA-256 `e481ca4dc3ab5bdf63636ad53eeeba8d3677305b643ecb98b7391e6419383ba3`, unchanged) |
| Engine | `scripts/tournament_engine.py` (Phase 8C, unmodified) |

Every artifact that can materially change inference is hashed in the
protocol and both receipts: the model, preprocessing JSON, selected config,
`config/series_features_v1.yaml`, `scripts/feature_engine.py`,
`scripts/preprocessing_random_forest_v1.py`, the strict state file, the
predictor adapter itself, the tournament YAML, and `tournament_engine.py` -
10 pipeline hashes in total, plus Python/numpy/pandas/scikit-learn/joblib
versions.

**Tier rationale.** Only `tier1`/`tier2`/`tier3` are known preprocessing
categories (no "unknown" category exists - any other value silently
collapses to the `tier1` reference). Resolved to `tier1` on two grounds
requiring no Cologne-result inspection: the repository's own tier scheme is
unambiguous from *other* tournaments' names alone (tier1 contains top-tier
LAN/Major-caliber events; tier2 contains clearly lower-caliber events), and
IEM Cologne Major 2026 is a Valve-sponsored Major Championship - the
highest tier of competition that exists by definition.

**A determinism fix applied before finalizing.** `RandomForestClassifier.predict_proba`
parallelizes cross-tree averaging when `n_jobs=-1` (as the model was
trained); testing surfaced that parallel float summation order is not
guaranteed identical across repeated calls (a 1-ULP jitter). The adapter
now forces `model.n_jobs = 1` immediately after loading - a pure
inference-time execution setting, changing nothing about what the model
learned or predicts in distribution, only guaranteeing that repeated
identical queries are byte-identical. The frozen artifacts below reflect
this corrected, fully deterministic adapter (the first, pre-fix generation
was discarded and regenerated before any result was reported).

---

## B. Model adapter

`scripts/pre_veto_series_predictor.py` implements the Phase 8A contract
(`predict_series_unknown_maps`) for the first time: `feature_engine.build_features`
→ `preprocessing_random_forest_v1.transform` → `RandomForestClassifier.predict_proba`,
with zero retraining, refitting, calibration, threshold tuning, feature
changes, averaging, or ensembling anywhere.

**Model contract verification** (amendment #2, all passed before any real
prediction was generated): `model.classes_ == [0, 1]` with class `1`
meaning `team_a`/`team1` wins; raw feature count 17; transformed feature
count 19; transformed feature order exactly matches the frozen
preprocessing contract; `model.n_features_in_ == 19`; no missing or
unexpected transformed feature.

## E. BO1/BO3/BO5 inference validation

All three formats transform and score cleanly with the exact expected
one-hot pattern (inference-contract validation only, no outcome
comparison):

| bestOf | `bestOf_BO3` | `bestOf_BO5` |
|---|---|---|
| 1 (reference) | 0 | 0 |
| 3 | 1 | 0 |
| 5 | 0 | 1 |

`tier1` was proven to produce the frozen reference-category representation
(`tier_tier2=0, tier_tier3=0`) - no silent category fallback.

**BO5 data-sparsity caveat.** RF V2's preprocessing and model structurally
support `bestOf=5` (proven above), but BO5 was substantially less common in
the historical training data than BO1/BO3. This is recorded as a
limitation only - the Grand Final still uses the frozen RF V2 with
`bestOf=5` unmodified; no retraining, blending, or separate calibration was
applied.

---

## C. Strict pre-event state / D. Team coverage

31 of the 32 real Cologne teams are present in the strict pre-Cologne
state; **THUNDERdOWNUNDER** (Stage 1, seed 15) has zero recorded prior
history and is scored entirely through `feature_engine`'s pre-existing,
already-defined cold-start default (`elo_before=1500.0`, win-rates default
to 0.5, `total_matches_before=0`, `days_since_last_match` NaN → median-imputed) -
no new fallback was invented, and this did not require stopping.

| Team | Stage | Prior matches | ELO | Last match |
|---|---|---:|---:|---|
| Team Vitality | 3 | 191 | 2075.6 | 2026-05-16 |
| Natus Vincere | 3 | 191 | 1923.7 | 2026-05-17 |
| Team Falcons | 3 | 192 | 1925.1 | 2026-05-24 |
| MOUZ | 3 | 208 | 1864.9 | 2026-05-24 |
| FURIA | 3 | 230 | 1816.5 | 2026-05-15 |
| The MongolZ | 3 | 226 | 1794.4 | 2026-05-22 |
| Aurora Gaming | 3 | 261 | 1763.0 | 2026-05-15 |
| PARIVISION | 3 | 234 | 1715.2 | 2026-05-22 |
| Team Spirit | 2 | 200 | 1982.6 | 2026-05-17 |
| G2 Esports | 2 | 213 | 1788.5 | 2026-05-15 |
| **THUNDERdOWNUNDER** | 1 | **0 (cold start)** | 1500.0 (default) | n/a |

(Full 32-row coverage table available via `pre_veto_series_predictor.audit_team_coverage`.)

---

## F. Probability matrix

`data/evaluation/cologne_2026_pre_event_matchup_probabilities_v1.parquet` -
**2,976 rows** (32 teams × 31 ordered opponents × 3 formats), unique key
`(team_a, team_b, best_of)`, every probability finite and in `[0,1]`,
`probability_team_b == 1 - probability_team_a` to strict tolerance for
every row, no result/winner column present. No probability was clipped,
smoothed, or otherwise altered - validation only asserts the `[0,1]`/
complementarity contract (no exact 0 or exact 1 occurred in this run).

## G. Orientation diagnostic

Diagnostic only - never used to average, correct, or symmetrize the matrix
or the simulation. Over the 1,488 unordered pair/format combinations:

| Statistic | Value |
|---|---|
| Mean | 0.00935 |
| Median | 0.00780 |
| p95 | 0.02333 |
| Max | 0.04314 |

Small, expected RF orientation sensitivity (roughly 1 percentage point on
average). The simulation always uses the tournament engine's own
deterministic `team_a`/`team_b` orientation - never an averaged or
corrected value.

---

## H. Monte Carlo design

50,000 complete Majors, `BASE_SEED=42`, each simulation's outcomes drawn
from `np.random.default_rng(np.random.SeedSequence([42, simulation_index]))`
- independently reproducible per index, no shared/global RNG state. The
`CachedProbabilityOutcomeProvider` consumes ONLY the frozen matrix; RF V2 is
never called during simulation. Expected matches per tournament: 106
(33×3 Swiss + 7 playoff). **Actual total matches: 5,300,000** (50,000×106,
exact).

**Rematch-fallback realism audit** (amendment #6, report only - Phase 8C's
fallback behavior was never altered based on this observation): across
4,950,000 real Swiss pairing events (50,000×99), **zero** required the
minimum-rematch fallback - a fully rematch-free arrangement existed in
every single pairing decision for the real 16/8/8 Cologne bracket
structure across all 50,000 simulated stages.

---

## I. Stage participation / advancement

Participation is structural, not simulated, for direct entrants
(Stage-2/Stage-3 direct entrants always show `participate_stage_1=0` etc. -
this is *not* "failed to reach Stage 1," they never entered it by design).
Conditional advancement probabilities (`advance_from_stage_N`), computed
with the correct stage-participation denominator (never the global 50,000):

**Highest Stage-1 → Stage-2 advancement probability:** BetBoom Team
(79.4%, 39,714/50,000). **Highest Stage-2 → Stage-3 advancement
probability:** Team Spirit (90.7%, 45,354/50,000).

## J. Swiss record probabilities

Full terminal-record distributions (3-0/3-1/3-2/2-3/1-3/0-3), both
unconditional (÷50,000) and conditional on stage participation, are in
`data/evaluation/cologne_2026_pre_event_swiss_record_distributions_v1.csv`.

## K. Playoff probabilities

**Highest playoff-qualification probability:** Team Vitality (88.0%,
43,997/50,000). Full 1–8 seed distributions (both unconditional and
conditional on reaching playoffs) are in
`data/evaluation/cologne_2026_pre_event_playoff_seed_distributions_v1.csv`.

## L. Championship probabilities

| Rank | Team | Championship count | Probability | MC SE |
|---|---|---:|---:|---:|
| 1 | Team Vitality | 14,848 | 29.70% | ±0.20% |
| 2 | Team Spirit | 9,476 | 18.95% | ±0.18% |
| 3 | Natus Vincere | 5,710 | 11.42% | ±0.14% |
| 4 | Team Falcons | 4,465 | 8.93% | ±0.13% |
| 5 | MOUZ | 3,877 | 7.75% | ±0.12% |
| 6 | FURIA | 1,861 | 3.72% | ±0.08% |
| 7 | The MongolZ | 1,752 | 3.50% | ±0.08% |
| 8 | Aurora Gaming | 1,358 | 2.72% | ±0.07% |

`sum(championship_count) == 50,000` exactly (verified in section Q).

## M. Deterministic favorite-wins path

Rule: `if P(team_a) >= 0.5: team_a wins; else team_b wins` (exact `p=0.5`
resolves to `team_a`). **Model-favorite champion: Team Vitality** - the
same team as the Monte Carlo championship favorite, though this is a
coincidence of this run, not a guarantee (greedy per-match favorites do not
maximize a joint tournament-path probability, which is why this path is
never called "the most likely bracket"). Full 106-match trace in
`data/evaluation/cologne_2026_pre_event_favorite_path_v1.json`.

---

## N. Monte Carlo sampling error

Every reported probability carries `numerator_count`, `denominator_count`,
`mc_standard_error = sqrt(p(1-p)/N)`, and a 95% Monte Carlo interval,
labeled explicitly as **Monte Carlo sampling uncertainty**, never model
confidence. At `N=50,000`, the maximum possible unconditional SE
(`p=0.5`) is `0.002236` (~0.22 percentage points) - matches every
unconditional row in the output tables exactly.

## O. Reproducibility

A 1,000-simulation subset (and, separately, every one of the 4 fixed
sample indices individually) was rerun twice; canonical serialization
(`json.dumps(..., sort_keys=True, separators=(",", ":"))`) and SHA-256
hash were identical both times - proven in
`tests/test_phase8d_cologne_pre_event_simulation.py`
(`test_reproducibility_subset_identical_across_two_runs`,
`test_single_simulation_index_reproducible`). Fixed sample traces (chosen
in the protocol *before* execution, never after viewing outcomes):
simulation indices **0, 1, 42, 999**, saved in full in
`data/evaluation/cologne_2026_pre_event_sample_traces_v1.json` with each
simulation's derived `SeedSequence` entropy and canonical trace hash -
simulated outcomes only, no real Cologne result.

## P. Immutability / receipt

`data/evaluation/cologne_2026_pre_event_simulation_receipt_v1.json` is the
commit marker of a transactional staging→validate→promote pipeline
(`scripts/run_phase8d_pipeline.py`): every artifact was generated into a
staging directory, fully validated (accounting identities, matrix
contract, receipt hashes), and only then promoted atomically in a fixed
order with the receipt written and promoted **last**. `created_before_results_opened: true`,
`cologne_results_status: "UNOPENED"`. The receipt records SHA-256 for the
model, preprocessing, config, strict state, `feature_engine.py`,
`preprocessing_random_forest_v1.py`, `series_features_v1.yaml`, the
predictor adapter, the tournament YAML, `tournament_engine.py`, the
protocol, the probability matrix, every aggregate output file, the
favorite-wins path, and the sample traces - plus Python/numpy/pandas/
scikit-learn/joblib versions. A preflight check refuses to run again over
an existing completed receipt, and refuses to silently continue from a
partially-promoted (crashed) prior run.

## Q. Validation

- `pytest tests/test_phase8d_cologne_pre_event_simulation.py`: **58/58
  passed.** Full repository `pytest`: **525/525 passed** (467 pre-existing +
  58 new, zero regressions).
- `python scripts/validate_phase8d.py`: **104/104 checks passed**, including
  independently recomputing every stored probability from its own integer
  counts (never trusting the stored float), verifying every accounting
  identity (champion sum = 50,000; playoff/semifinal/final/per-stage-advance
  sums = 8N/4N/2N/8N respectively; total matches = 5,300,000), confirming
  zero network imports and zero forbidden-result-path reads across every
  new Phase 8D source file, confirming the figures module imports no
  ML/model-calling code, confirming every receipt-recorded hash matches the
  file on disk, and confirming all 20 Phase 1–8C artifacts this phase
  depends on are byte-unchanged.

---

**PRE-EVENT COLOGNE SIMULATION = FROZEN**
**COLOGNE RESULTS = STILL UNOPENED**
**RF V2 = UNCHANGED**
**MODEL STATE = FROZEN PRE-EVENT**
**NO MAP INPUT USED**
**PHASE 7 = UNCHANGED**
**PHASE 8B = UNCHANGED**
**PHASE 8C ENGINE = UNCHANGED**
**NO POST-SIMULATION RETUNING**
**NO API**
**NO PWA**
