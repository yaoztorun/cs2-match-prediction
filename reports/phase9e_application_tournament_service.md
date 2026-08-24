# Phase 9E — Major Simulation Application Service V1

## A. Tournament-service architecture

`scripts/application_tournament_service.py` sits ABOVE the frozen Phase 8C tournament engine
(`tournament_engine.py`, byte-for-byte unchanged — verified: its SHA-256 still equals the value
Phase 8D itself pinned, `012bd58e7792f7cce1e888d8f233fab274d798d4c3728f5b237f041fb73dd665`) and
wraps three more already-frozen subsystems: Phase 8C (engine mechanics), Phase 8D (historical
Cologne probability matrix + 50,000-simulation Monte Carlo), and Phase 8E (simulation-vs-reality
evaluation). `scripts/application_tournament_router.py` is a thin FastAPI `APIRouter` — all
business logic lives in the service module, the router only builds typed request/response models
and calls into it. `application_api.py` gets a minimal, additive diff: `app.include_router
(major_router)`, an extended `ERROR_STATUS_MAP`, and the startup contract split into independent
subsystem flags (section C). No code moved into `src/`, and `tournament_engine.py` itself was
never touched.

Every tournament prediction — historical or interactive — is RF V2 pre-veto only
(`application_inference.predict_series_unknown_maps`). XGB V3 is never imported, never called: an
AST-based static guard (`test_service_module_never_imports_xgb_or_known_map_prediction`, plus
`scripts/validate_phase9e.py` section 10) verifies the service module contains no `xgboost` import
and no reference to `predict_map`/`predict_series_known_maps`.

## B. Historical vs interactive contract

Two permanent modes, never conflated:

- **Historical** (`get_historical_cologne_pre_event` / `get_historical_cologne_results`) — pure
  file-backed views over already-frozen Phase 8D/8E artifacts. No RF call, no matrix build, no
  engine run, no Monte Carlo for a normal GET (verified both by an AST source-scan of the two
  reader functions and by a latency assertion: the pre-event endpoint responds in ~124 ms cold /
  ~0.003 ms cached — many orders of magnitude faster than the ~58 s a real matrix build takes).
  Historical participant/team identifiers are the frozen Phase 8B YAML's own
  `canonical_model_name` values, read via the existing `phase8d_common.build_cologne_entrants()` —
  **never** Phase 9B's deployment identity policy (`ai.resolve_team`). This was checked directly,
  not assumed: `test_historical_favorite_path_parity_hard_gate` asserts all 32 historical
  identifiers are a subset of the frozen matrix's own team keys before running anything.
- **Interactive** (`validate_tournament_participants` / `build_tournament_probability_matrix` /
  `predict_tournament_path` / `simulate_tournament`) — uses Phase 9B's `deployment_post_cologne_v1`
  identity resolution and builds a **fresh** 2,976-row RF matrix on demand (never at API startup,
  never persisted to a research artifact).

Drift in `team_identity_policy.csv` can never change whether a frozen historical endpoint serves —
the two identity contracts are structurally independent code paths with no shared state.

## C. Ruleset registry

`config/application_tournament_rulesets_v1.yaml` registers exactly one ruleset,
`iem_cologne_major_2026_format_v1` — application/version **metadata only**. The frozen
`tournament_engine.load_frozen_rules()` remains the sole executable rules authority; no second
tournament-rules engine was implemented. `verify_ruleset_matches_engine_rules()` cross-checks
every descriptive field (stage sizes, BO-per-stage, advancement/elimination thresholds, the two
Cologne-specific overrides) against a freshly-loaded `te.TournamentRules` and against the pinned
`source_yaml_sha256`/`tournament_engine_py_sha256` — both hashes were independently re-verified
against the live files on disk before the registry was written, and matched Phase 8D's own
recorded values exactly (confirming the engine and frozen YAML are unchanged since Phase 8D). This
check runs at API startup (`tournament_engine_ready` subsystem) and in the validator.

## D. Participant validation

`validate_tournament_participants` requires exactly 16 + 8 + 8 = 32 entries, each `{team, seed}`.
Every team is resolved via `ai.resolve_team` (Phase 9B's exact-match-only identity policy — never
fuzzy). Validation requires: per-stage seed sets exactly `{1..16}`/`{1..8}`/`{1..8}` (no gaps, no
duplicates — `invalid_seed`/`missing_seed`), and all 32 resolved canonical names pairwise unique
across all three stages combined (`duplicate_team`). An unresolvable team raises `unknown_team`
(404, reusing Phase 9B's own error code); a malformed count raises `invalid_participant_count`.
Cold-start identity-eligible teams remain permissible, exactly matching the existing frozen RF
behavior — no new eligibility rule was invented for tournament participants.

## E. Frozen matchup matrix

`build_tournament_probability_matrix` iterates all `32×31=992` ordered pairs × `{1,3,5}` = 2,976
rows, calling `ai.predict_series_unknown_maps` once per row — `probability_team_b` is always
`1 - probability_team_a` by construction, never independently computed. Validation mirrors Phase
8D's own matrix validator: exact row count, key uniqueness, `[0,1]`, complement identity, every
pair×BO present — never clipped.

**Matrix identity != tournament scenario identity** (amendment #2): the matrix depends only on the
32-team **set** (plus context/tier/prediction_datetime/RF pipeline fingerprint) — never on
seed/stage assignment, since it always contains every ordered pair regardless of bracket seeding.
Seed/stage assignment and manual overrides determine the deterministic-path/Monte-Carlo *result*,
which is therefore never cached by matrix key alone — Phase 9E keeps simulation results fully
stateless and recomputed (section H).

**Matrix hash semantics, frozen precisely** (amendment #3): `probability_matrix_hash` for a
freshly-built application matrix is a canonical-content SHA-256 (`_canonical_matrix_content_hash`)
over only semantically relevant fields (team_a, team_b, best_of, probability_team_a/b, model_id,
context_id, tier, prediction_datetime), sorted-keys/compact-separator JSON — never a pandas row
index or display ordering. For historical Cologne, Phase 8D's own receipt was inspected first: it
records only the **file-level** SHA-256 of the parquet (`matrix_sha256`/`probability_matrix` in
`cologne_2026_pre_event_probability_receipt_v1.json`/`..._simulation_receipt_v1.json`), never a
canonical-content hash. The historical pre-event response therefore exposes **two distinct,
explicitly labeled fields**: `artifact_file_sha256` (the frozen, authoritative Phase 8D value) and
`application_matrix_content_hash` (a *separately derived* canonical-content hash, computed at
serve time with the same function used for interactive matrices, for like-for-like comparison —
never presented as something Phase 8D itself recorded). Both endpoints/paths return this
distinction with an explicit `matrix_hash_semantics_note` string.

## F. Deterministic path

`_run_deterministic_path` builds an `_OverrideAwareFavoriteProvider` (`p_a >= 0.5 -> team_a`,
deliberately `>=` and distinct from Phase 9B's strict `>` + `favored_team=None` display-tie
convention — the tournament engine has no concept of an unresolved match, so it must always
produce a definite winner). A real bug was caught and fixed here during implementation: an early
version of this provider added a `selection_source` key to the engine's `provider_metadata`, which
changed `canonical_trace_hash` even with zero overrides and broke the hard parity gate below.
Fixed by keeping `provider_metadata` byte-identical to Phase 8D's own `FavoriteWinsProvider`
(`{"model_id": "series_random_forest_v2", "prediction_mode": "pre_veto"}`, no extra keys) and
deriving `selection_source` (model vs user) separately at response-projection time from the
override index — the engine-level trace content is never touched by anything transport-related.

**Hard gate — historical favorite-path parity**: using the real 32 Cologne teams/seeds (frozen
Phase 8B YAML, loaded via the existing `phase8d_common.build_cologne_entrants()`), the frozen
Phase 8D probability matrix (loaded directly from the parquet, never rebuilt), and zero overrides,
`_run_deterministic_path`'s `canonical_trace_hash` was verified to equal the frozen Phase 8D
favorite-path hash **exactly**:
`6d96855f4c3f08ec99229bdffe2ab6d7c8285a32db20281973db5f5abe58ed35` (champion: Team Vitality). This
is a single shared code path — the same `_run_deterministic_path` function serves both the live
`/major/path` endpoint (fed a freshly-built matrix) and this parity test (fed the frozen matrix) —
so the parity guarantee is a property of code-sharing, not a separately maintained reimplementation.

## G. Manual overrides / Pick'Em

**Semantic identity** (amendment #4): Swiss overrides are identified by
`(stage, round_number, record_group, frozenset({team_1, team_2}))`; playoff overrides by
`(playoff_round ∈ {quarterfinal, semifinal, grand_final}, frozenset({team_1, team_2}))` — never an
ordinal match number, so an override survives rebracketing caused by an earlier pick. `best_of` is
accepted as an optional field but is **not** part of the matching key (documented design choice —
it is deterministically implied by stage/record already, so treating it as part of identity would
only ever make an override stricter without adding real information; see section S).

**Duplicate/contradictory contract** (amendment #5), frozen and tested: the same identity + same
winner → `duplicate_override`; the same identity + a different winner → `contradictory_override`;
a winner not in the declared pair → `override_team_mismatch`; an unresolvable team → `unknown_team`
(reused from Phase 9B); a malformed stage/round/record_group shape → `invalid_override`. All are
rejected **before** any tournament runs — dictionary overwrite order never silently decides a
winner. A conservative bound of 106 overrides (the maximum possible matches in one complete path)
is enforced (amendment #24).

**Stateless replay**: no server-side mutable tournament session exists. Each `/major/path` /
`/major/simulate` request supplies the full participant set + override list and is deterministically
replayed from Stage 1. `override_usage` reports `overrides_supplied`, `overrides_used`,
`overrides_not_reached` (an override is `not_reached` **only** when its exact semantic matchup
never occurred — never approximated by round/slot proximity — amendment #12), and
`invalid_overrides` (always empty in a 200 response, since malformed overrides are hard-rejected at
validation time rather than silently excluded — kept in the schema for completeness).

## H. Monte Carlo design

Bernoulli sampling from the frozen matrix (`rng.random() < p_a`), **never** `p > 0.5 -> winner`
inside Monte Carlo — verified directly (`test_monte_carlo_bernoulli_not_deterministic_threshold`):
a near-50/50 matchup shows both possible outcomes across repeated draws, not one deterministic
winner every time. A matchup-specific override is forced only when the exact semantic identity
occurs in that simulation; otherwise sampling proceeds normally from the same frozen matrix.
`simulation_conditioned_on_manual_overrides` is returned explicitly.

**Override accounting** (amendment #11) returns three distinct counts per override —
`simulations_matchup_reached`, `simulations_override_applied`, `simulations_not_reached`
(`reached + not_reached == total_simulations` by construction) — plus `application_rate =
applied/total` and `conditional_application_rate = applied/reached` (which is exactly `1.0` for a
valid forced override whenever reached, verified directly).

**Aggregation** mirrors Phase 8D's own `Aggregator` shape (per-team stage participation/advancement,
Swiss terminal-record counts, playoff-seed counts, reach-playoffs/semifinal/final, champion counts)
— integer counts are the primary source of truth; every reported probability is literally
`numerator_count / denominator_count`, with the **correct** conditional denominator per metric
(amendment #18): `advance_from_stage_N` divides by that stage's participation count, not N;
`playoff_seed_distribution[k]` divides by `reach_playoffs`'s own count, not N; `swiss_record_
distribution` divides by the team's stage-participation count. Verified directly:
`test_swiss_record_distribution_denominators` confirms every record bucket for a team/stage shares
one common denominator, and `test_playoff_seed_distribution_conditional_on_reaching_playoffs`
confirms the seed-distribution denominator equals `reach_playoffs`'s numerator exactly.

## I. RNG reproducibility

Identical to Phase 8D's own scheme: `np.random.SeedSequence([base_seed, simulation_index])` →
`np.random.default_rng(seed_seq)`, one fresh `Generator` per simulation. `seed` defaults to 42,
always echoed back in the response.

**Hard gate — sample-trace Monte Carlo parity**: simulation indices **0, 1, 42, 999** were run
against the frozen Phase 8D matrix with `base_seed=42` and zero overrides; each resulting champion
and `canonical_trace_hash` matched `cologne_2026_pre_event_sample_traces_v1.json` **exactly**
(4/4) — proving the application simulator did not alter Monte Carlo semantics at all.

**Chunking independence** (amendment #8), the critical process-pool correctness property: since
each simulation's RNG depends only on `(base_seed, simulation_index)` — never on worker ID or
chunk boundary — splitting a fixed range of simulation indices into 1 batch vs. 4 batches of 50
must merge to an identical aggregate. Verified directly, not assumed:
`test_monte_carlo_chunking_independence` and `scripts/validate_phase9e.py` section 8 both confirm
`champion_counts` and the full per-team aggregate dict are byte-identical between a single
200-simulation batch and four merged 50-simulation batches for the same index range.
`test_monte_carlo_reproducible_same_seed` and `test_concurrent_same_seed_simulations_identical`
further confirm the same `(participants, seed, n)` always reproduces identically, including under
concurrent load.

## J. Stage/Swiss aggregates

`test_simulation_accounting_conservation` and `_verify_aggregate_conservation` (raised internally
as `missing_state_support` — an unexpected-internal-failure signal — if violated, never silently
tolerated) assert, for N simulations: `champion_sum == N`, `playoff_sum == 8N`,
`semifinal_sum == 4N`, `final_sum == 2N`, and per Swiss stage: `advancing == 8N`,
`eliminated == 8N`, and the full terminal-record total (all six buckets: 3-0/3-1/3-2/2-3/1-3/0-3)
`== 16N` (amendment #19) — every one of these held exactly in a 400-simulation live run.

## K. Playoff/championship distributions

`champion_ranking` is the primary integer-count championship distribution
(`numerator_count/denominator_count/probability/mc_standard_error`, denominator always N). Per-team
`playoff_seed_distribution[1..8]` is explicitly conditional on `reach_playoffs` (never on N) — the
distinction `P(reach playoffs)` vs. `P(seed=k | reaches playoffs)` is preserved by construction
(two separate fields, two separate denominators), matching amendment #21 of the original spec.

## L. Historical Cologne API

`GET /api/v1/major/historical/cologne-2026` returns (from Phase 8D artifacts only): tournament
metadata, the 32-team participant/seeding structure, the pre-event favorite and its championship
probability, the full championship-probability ranking (integer counts), stage-advancement and
playoff-qualification probabilities, Swiss/playoff-seed distributions, the frozen favorite-wins
path (with its own `canonical_trace_hash`), `artifact_file_sha256`/`application_matrix_content_hash`
(section E), `historical: true`, `immutable: true`.

`GET /api/v1/major/historical/cologne-2026/results` returns (from Phase 8E artifacts only): actual
champion (Team Falcons), its pre-event probability (0.0893) and rank (4/32), match-level AUC/Log
Loss/Brier vs. the p=0.5 baseline, playoff/semifinal/finalist overlap, the favorite-path-vs-reality
comparison, and the reconciliation counts — `original_cologne_tagged_rows=107`,
`official_major_matches=106`, `excluded_non_tournament_rows=1`, with the excluded row's detail
explicitly labeled `reconciliation_status: "non_tournament_showmatch"` (Team Germany vs Team
Poland, 2026-06-21) — never presented as an official tournament match.

## M. Historical adapter parity

Both hard gates (sections F and I) passed exactly on first correct implementation attempt (after
the `provider_metadata` fix described in section F). `scripts/validate_phase9e.py` re-verifies both
independently, plus `verify_historical_cologne_contract()` transitively re-hashes every Phase 8D
artifact actually served (matrix, summary, team probabilities, Swiss/playoff distributions,
favorite path, sample traces, plus the frozen YAML and engine) against the Phase 8D receipt's own
recorded hashes, and every Phase 8E artifact actually served (summary, metrics detail,
reconciliation table) against the Phase 8E receipt's own recorded hashes (amendment #13) — never
just the receipt JSON's own hash. If any mismatch is found, `historical_cologne_ready` is `false`
and historical routes return `503 service_unavailable` — the historical contract is never silently
regenerated.

## N. Caching

**Matrix cache** (amendment #1/#7): a bounded (`MAX_MATRIX_CACHE_ENTRIES=8`), deterministic LRU
(`collections.OrderedDict`), guarded by a `threading.RLock` around all metadata operations
(check/insert/evict), with per-key single-flight synchronization (a `threading.Lock` per cache key)
so concurrent first-requests for the *same* key don't each pay for 2,976 RF calls. The expensive
build itself runs **outside** the global lock — the pattern is exactly lock → check → unlock →
build → lock → re-check → insert/evict → unlock, as required. Cached values are
`types.MappingProxyType`-wrapped (read-only; `test_cached_matrix_lookup_is_read_only` confirms a
`TypeError` on mutation attempt). Verified directly: 16 concurrent threads building the identical
32-team/tier1 matrix all received the identical content-hash matrix
(`test_matrix_cache_thread_safety_no_corruption`); a different tier produces a different hash
(`test_matrix_cache_different_key_different_matrix`); repeatedly building distinct 32-team subsets
past the bound keeps the cache size `<= 8` (`test_matrix_cache_bounded_size`).

**Historical payload cache** (amendment #14): after the first successful hash-verified read, the
parsed historical pre-event/results payloads are cached in-process (guarded by a plain
`threading.Lock`) — safe because Phase 8D/8E artifacts are frozen and never change during the
process lifetime. This drops repeat-request latency from ~124 ms to ~0.003 ms.

**No Monte Carlo result cache** (amendment #21): deliberately absent in V1. A simulation request is
fully reproducible from `(participants, overrides, seed, n)`, so recomputation is preferred over
adding a second cache surface — simpler, and aligned with the stateless Pick'Em design.

## O. Concurrency

Endpoints are plain `def` functions (Starlette dispatches them to its worker thread pool, matching
Phase 9D's existing convention). `test_concurrent_deterministic_path_calls_are_stable` (8 concurrent
threads) and `test_concurrent_same_seed_simulations_identical` (4 concurrent threads) both confirm
identical results with no cross-contamination. A live check confirmed an ordinary `/predict/series`
-equivalent direct call completed in **20.8 ms** while a 20,000-simulation Major run was executing
concurrently in a background thread (process-pool execution keeps the prediction path unblocked) —
demonstrating, at local scale, that a large simulation does not starve ordinary prediction traffic;
this is not a claim of production-scale concurrency guarantees.

**Process-pool lifecycle** (amendment #6): one lazily-created, `threading.Lock`-guarded
`ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1))` singleton, explicitly
`shutdown(wait=True)` from the FastAPI `lifespan` shutdown phase — no orphan worker processes
survive `TestClient` context-manager exit or application shutdown. Workers receive **only** plain
data (canonical participant lists, the plain `{(a,b,bo): p_a}` matrix dict, override identities,
`base_seed`, an index range) — never the RF/XGB model, never a `StateStore`, never `application_
inference` state; each worker calls only `te.load_frozen_rules()` (hash-gated, reads solely the
frozen YAML's structure keys, zero ML/state dependencies — the same guarantee Phase 8C's own tests
already establish for that function) and the pure-Python `tournament_engine`. **Worker failures are
all-or-nothing** (amendment #20): if any chunk's future raises, the exception propagates
immediately, no partial aggregate is merged, cached, or persisted.

## P. Performance

Measured independently, matrix construction separated from simulation (amendment #6 of the second
plan):

| Phase | Time |
|---|---|
| A. Participant validation (32 teams) | ~6.4 s *(dominated by cold RF/XGB model load in a fresh process — warm-process validation is sub-millisecond identity resolution × 32)* |
| B. Cold matrix construction (2,976 RF calls) | **58.1 s** (~19.5 ms/call) |
| C. Cached matrix retrieval | **5.0 ms** |
| D. Deterministic path (matrix cached) | **49.4 ms** (106 engine matches, zero RF calls) |
| E. Monte Carlo, 1,000 sims (synchronous, below threshold) | 4.82 s (~4.8 ms/sim) |
| E. Monte Carlo, 5,000 sims (process pool) | 6.35 s (~1.27 ms/sim) |
| E. Monte Carlo, 10,000 sims (process pool) | 12.51 s (~1.25 ms/sim) |
| E. Monte Carlo, 50,000 sims (process pool) | 63.0 s (~1.26 ms/sim) |

`PROCESS_POOL_MIN_SIMULATIONS = 2,000` was frozen after this benchmark: below it, per-simulation
cost is dominated by process-pool submission/pickling overhead relative to a single ~106-match
tournament's cost, so synchronous execution is faster; at/above it, 4-way parallelism yields a
consistent ~3.6× per-simulation speedup (4.8 ms → 1.25–1.3 ms) that comfortably outweighs pool
overhead. `test_monte_carlo_chunking_independence` proves this threshold choice affects **only**
performance, never output.

Response sizes (aggregate-only, no per-simulation traces returned, per amendment #29):
deterministic path **31.8 KB**; simulate at 1k/5k/10k/50k sims: **154 KB / 157 KB / 158 KB / 161 KB**
(size grows only with the number of distinct teams/records/seeds observed across more simulations,
not linearly with N — confirms aggregates, not per-simulation traces, are being serialized).
Historical pre-event page: **446 KB** cold (~124 ms), **~0 ms** cached; historical results page:
**1.7 KB** (~5.8 ms).

## Q. API / OpenAPI

Routes, all under `/api/v1/major`: `GET /rulesets`, `GET /historical/cologne-2026`,
`GET /historical/cologne-2026/results`, `POST /path`, `POST /simulate`. `/openapi.json` remains
valid and includes all five (verified both in the test suite and the validator). Typed Pydantic
contracts (amendment #17) cover the new stable high-level shapes — `TournamentParticipant(s)`,
`ManualOverride`, `MajorPathRequest`/`MajorSimulateRequest`, `MatchOutcome`, `StageMatches`,
`PlayoffMatches`, `OverrideUsageReport`/`OverrideDiagnostic`, `TeamAggregate`,
`ChampionshipProbability`, `MonteCarloMetadata`, and the response envelopes
`TournamentPathResponse`/`TournamentSimulationResponse` — none of the new `/major/path` or
`/major/simulate` contracts are untyped `Dict[str, Any]` blobs, matching the additive requirement.
Deep historical artifact payloads (`result` inside the two historical GET responses) remain
controlled dict passthrough, as explicitly permitted, since re-declaring the already-versioned
Phase 8D/8E schemas field-by-field would be pure duplication risk for no benefit.

A real typing bug was caught during implementation and fixed: `champion_ranking` was initially
typed as `List[ProbabilityStat]`, which has no `team` field — FastAPI's response-model validation
silently stripped the team name from every entry. Fixed by adding a `ChampionshipProbability
(ProbabilityStat)` subclass with the `team` field. This is exactly the class of transport-layer
bug that typed response models are supposed to catch loudly (a schema mismatch) rather than
silently drop data — caught here by directly inspecting a live HTTP response, not assumed correct
from code review alone.

## R. Validation

- `pytest tests/test_phase9e_application_tournament_service.py` — 59/59 passed.
- `scripts/validate_phase9e.py` — all checks pass, including both hard parity gates, aggregate
  conservation, RNG/chunking-independence, override semantics, the no-XGB static guard, state
  immutability, and the full Phase 8C/9B/9C/9D regression gate re-run as real subprocesses.
- Phase 8C: `pytest tests/test_phase8c_tournament_engine.py` (61/61, unchanged) and
  `scripts/validate_phase8c.py` (48/48) both still pass — the engine and its canonical
  `HigherSeedWinsProvider` regression trace are confirmed byte-identical to Phase 8C.
- Phase 9D: full `pytest tests/test_phase9d_application_api.py` (96/96 — 85 original + 11
  parametrized instances added by the now-extended `ERROR_STATUS_MAP`) and
  `scripts/validate_phase9d.py` (61/61) both pass — existing `/predict/series`, `/predict/map`,
  discovery endpoints, error status semantics, request-ID behavior, and readiness semantics for
  the original core service are all unchanged. Two Phase 9D-owned artifacts legitimately changed as
  a direct, in-scope consequence of this phase's approved plan and were regenerated accordingly
  (see section S): the `application_api_receipt_v1.json` commit-marker receipt (since
  `application_api.py` itself gained the new router/subsystem-readiness code) and two small,
  additive test-only edits — `tests/test_phase9d_application_api.py`'s and
  `scripts/validate_phase9d.py`'s own hardcoded `ERROR_STATUS_MAP`-equality assertions were widened
  to include the new Phase 9E codes (the original 13 Phase 9D codes are asserted unchanged in both
  places).
- Phase 9B/9C: `scripts/validate_phase9b.py` (32/32) and `scripts/validate_phase9c.py` (33/33)
  both re-verified clean, untouched by this phase.

## S. Limitations

- `best_of` on a manual override is accepted but not part of the matchup-matching key (section G) —
  a deliberate simplification, since it is fully implied by stage/record already; a future revision
  could add it as a stricter optional identity component if a real product need arises.
- Only one ruleset is registered (`iem_cologne_major_2026_format_v1`). A future Major with a
  different format would require a new registered ruleset id — the registry pattern supports this,
  but no second ruleset was authored in V1.
- `PROCESS_POOL_MAX_WORKERS = min(4, os.cpu_count() or 1)` is a local-development-appropriate bound,
  not a production capacity guarantee; the concurrency/performance numbers in sections O/P describe
  local behavior only, not production scalability.
- Section R notes two Phase-9D-owned artifacts (the API receipt, and two hardcoded
  `ERROR_STATUS_MAP` equality assertions) that were legitimately updated as a direct consequence of
  this phase's approved, additive `application_api.py` extension — the previous Phase 9D receipt was
  archived (not deleted) at `data/deployment/pre_phase9e_archive/` before regeneration, for audit
  continuity.
- No tournament/simulator-level result cache exists (by design, section N); no PWA/frontend work
  was started, per explicit scope.

```
MAJOR SIMULATION SERVICE V1 = IMPLEMENTED
HISTORICAL COLOGNE = IMMUTABLE
CUSTOM MAJOR SIMULATION = ENABLED
MONTE CARLO = BERNOULLI FROM FROZEN RF PROBABILITIES
MANUAL PICK'EM OVERRIDES = ENABLED
RF V2 = UNCHANGED
XGB V3 = NOT USED FOR TOURNAMENT SIMULATION
PHASE 8C ENGINE = UNCHANGED
NO RETRAINING
NO PWA YET
```
