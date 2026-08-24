# Phase 9B — Application Inference Core

Reusable Python inference layer only. No model was fit, retrained, tuned, calibrated,
ensembled, or symmetrized. No API, frontend, or explanation engine.

Module: `scripts/application_inference.py`
Registries: `config/application_inference_contexts_v1.yaml`, `config/application_map_registry_v1.yaml`

## A. Architecture

One module, a small public surface (`list_inference_contexts`, `get_context_metadata`,
`list_supported_teams`, `list_supported_maps`, `predict_series_unknown_maps`, `predict_map`,
`predict_series_known_maps`, `compose_series_probability`), one structured exception
(`ApplicationInferenceError`, `error_code`/`message`/`detail`), and one `InferenceContext`
dataclass holding a loaded-once `PredictorContext` (RF), 4 other state stores, the loaded XGB
model/preprocessing/roles, the identity policy, and the map registry. `get_context(context_id)`
loads once and caches per process (`_CONTEXT_CACHE`); the RF/XGB model objects and their
preprocessing artifacts are identical across contexts and are loaded exactly once and shared
(`_SHARED_XGB`) — only the 5 state stores differ per context. The unknown-map path reuses the
**unmodified** `pre_veto_series_predictor.predict_series_unknown_maps` function directly (a
second `PredictorContext` is constructed pointing at the deployment state file; the frozen file
itself was never edited). The known-map path is new: `rich_modern_map_feature_composer` →
`preprocessing_xgboost_map_v3.transform` → `XGBClassifier.predict_proba`, exactly mirroring the
RF wrapper's shape. `compose_series_probability` is one generic DP, used by both `predict_map`
(indirectly, via the caller) and `predict_series_known_maps`.

## B. Versioned inference contexts

| | `historical_cologne_pre_event` | `deployment_post_cologne_v1` |
|---|---|---|
| Classification | historical_replay | deployment |
| State cutoff | 2026-06-02T13:30:00 (frozen Phase 8D cutoff) | 2026-06-28T20:00:00 (Phase 9A receipt) |
| Series state | `pre_cologne_team_state_v1_full.json` | `series_team_state_v1_deployment_post_cologne.json` |
| Prediction datetime | locked | defaults to cutoff; later = explicit hypothetical mode |
| Freshness label | `frozen_historical_snapshot` | `deployment_post_cologne_v1` (never "live"/"current") |

`list_inference_contexts()` reads only `config/application_inference_contexts_v1.yaml` — no
filesystem "latest" scanning anywhere. The registry hashes the **complete executable prediction
contract**, not just a model file: 7 RF-pipeline items (model, preprocessing, selected config,
feature schema YAML, `feature_engine.py`, `preprocessing_random_forest_v1.py`, the predictor
adapter itself) and 15 XGB-pipeline items (model, metadata, preprocessing, selected config,
feature schema YAML, both composer modules, both preprocessing-vocabulary modules, and all 5
state-engine modules) plus 5 state-file hashes per context and the Phase 9A deployment receipt
hash for the deployment context.

## C. Historical parity

**2,976/2,976 exact matches** (tolerance 1e-9) between
`application_inference.predict_series_unknown_maps("historical_cologne_pre_event", ...)` and the
frozen `cologne_2026_pre_event_matchup_probabilities_v1.parquet` matrix, run for real (not
asserted), in both the test suite and the validator. Parity is exact by construction — the
wrapper calls the identical `pre_veto_series_predictor.predict_series_unknown_maps` function
Phase 8D used, against the identical pre-Cologne state file, with `model.n_jobs=1` preserved.

## D. Deployment-state contract

`prediction_datetime` **cannot move backward** past the deployment cutoff — an earlier value
raises `prediction_datetime_before_state_contract` (would risk future-information leakage from
history the snapshot legitimately contains up to 2026-06-28). Default: the cutoff itself
(`staleness_days = 0`). A later value is accepted only as
`hypothetical_future_from_stale_snapshot`, with `state_data_through`,
`requested_prediction_datetime`, `staleness_days`, `state_is_live: false`, and an explicit
warning — never a silent implication that newer history exists. The historical context remains
fully locked; any override raises `historical_context_datetime_locked`. Staleness is always
computed against the state cutoff, never the machine's current date.

## E. Team identity resolution

Exact match only against `team_identity_policy.csv`'s `team_name`/`canonical_team_name` columns
(no fuzzy/edit-distance/embedding matching, no new aliases). A team is application-resolvable
**only if** `identity_feature_eligible` is true for its canonical identity — intentionally
ineligible identities, unresolved collisions (`MANUAL_REVIEW`), and unsafe org merges
(`KEEP_AS_SINGLE_TEAM` treated per the frozen policy) are never re-admitted; this path is
exercised directly against the real policy data in tests. `list_supported_teams(context_id)`
returns 777 identity-eligible teams for the deployment context, each with
`identity_eligible`/`history_available`/`history_match_count`/`cold_start` reported separately —
a policy-known, eligible team with zero history stays in the list rather than being removed. The
policy's 792 raw names have zero duplicates, so real `ambiguous_team` collisions cannot currently
occur; the code path is implemented and tested against a synthetic collision rather than forced
on real data.

## F. Tier / BO validation

Tiers: exactly `tier1`/`tier2`/`tier3`; `None` defaults to `tier1` with `tier_source:
"application_default"`, an explicit value gets `tier_source: "user_supplied"`; anything else
raises `invalid_tier`. Best-of: exactly `1`/`3`/`5` as a typed `int` (not `bool`, not a string
like `"bo3"`); anything else raises `invalid_best_of`. No implicit string normalization exists in
the core.

## G. Unknown-map predictor

`predict_series_unknown_maps(context_id, team_a, team_b, best_of, prediction_datetime=None,
tier=None)`. `team_a`/`team_b` map 1:1 onto the frozen model's `team1`/`team2` — the user's input
order, never tournament-seed-reoriented, never averaged with the reverse orientation.
`favored_team` is `None` with `prediction_is_tied: true` at exactly `p == 0.5` (a display
convention, not a change to Phase 8D's own `p >= 0.5 -> team_a` deterministic-favorite-path tie
rule, which remains untouched). `team_a_history`/`team_b_history` report
`identity_known`/`available`/`cold_start`/`matches` — descriptive only, never behavior-changing.

## H. Known-map predictor

`predict_map(context_id, team_a, team_b, map_name, best_of, prediction_datetime=None,
tier=None)`: composer → `preprocessing_xgboost_map_v3.transform` → `XGBClassifier.predict_proba`.
The XGB contract was verified as strictly as RF's (`_verify_xgb_contract`, run once at context
load): raw feature count 120, transformed count 131, `classes_ == [0, 1]`,
`n_features_in_ == 131`, every expected map/BO/tier dummy column present. `state_support` metadata
surfaces the composer's **own existing** confidence flags
(`both_teams_have_history`/`_map_history`/`_5_adjusted_matches`/`_5_inferred_players`/
`_map_pool_history`/`_recent_selected_map_history`) as
`overall_state_available`/`map_state_available`/`form_state_available`/
`player_roster_state_available`/`modern_map_state_available`/`player_map_state_available`, and
`fallbacks_used` lists exactly which of those were false for the given matchup — no new fallback
semantics were invented.

## I. Supported-map contract (three layers)

**A. Model support** — the frozen XGB V3 preprocessing vocabulary, exactly 9 categories:
Ancient, Anubis, Dust2, Inferno, Mirage, Nuke, Overpass, Train, Vertigo. **Cache is not in this
list and is explicitly rejected** (`unsupported_map`) rather than silently routed to the frozen
pipeline's own `__UNKNOWN_MAP__` training-time fallback bucket — the application layer adds this
gate itself; the underlying composer would accept any string, but that is irrelevant to the
application contract. **B. State support** — computed per call from the composer's own
confidence flags (section H); missing state does not equal unsupported, it's a descriptive
cold-start fact. **C. Competitive/UI selectability** — kept explicitly separate:
`cologne_2026_competitive_pool` records Phase 8B's frozen 7-map historical tournament fact
(Ancient, Anubis, Dust2, Inferno, Mirage, Nuke, Overpass — Train and Vertigo were not part of that
pool), never presented as "the current Active Duty pool." The deployment context's data ends
2026-06-28 and cannot know of any later real-world pool change.

## J. Ordered-map series composition

`compose_series_probability(map_probabilities, best_of)` — one generic DP over `(maps_played,
wins_a)`, terminating the instant either side reaches `(best_of+1)//2` wins; no hardcoded BO1/3/5
formulas. Validates its own inputs independently (`invalid_best_of`, `invalid_map_count`,
`invalid_probability`) rather than trusting a caller. Verified: all-`0.5` → `0.5`, all-`1.0` →
`1.0`, all-`0.0` → `0.0` for BO1/3/5; matches the closed-form race-to-N formula
(`Σ C(need-1+j,j) p^need (1-p)^j`) to 1e-9 for several `p`; probability conservation
(`P(A)+P(B)==1`) verified on 30 random asymmetric cases; an explicit asymmetric-ordered-probability
case checked by hand. `predict_series_known_maps` requires exactly `{1,3,5}`-matching
`ordered_maps` length (`invalid_map_count`) and rejects any repeated map
(`duplicate_map`) before ever calling `predict_map`.

## K. Data freshness semantics

Every deployment prediction returns `data_freshness: {state_data_through, state_is_live: false,
requested_prediction_datetime, staleness_days, snapshot_id}`, plus `mode`/`warning` when
`staleness_days > 0`. Historical predictions return the same shape with `staleness_days` fixed at
`0`. Staleness is never compared against the machine's real current date inside the core — only
against the context's own frozen `state_cutoff`.

## L. Error contract

One exception class, fixed `error_code`s: `unknown_team`, `ambiguous_team`, `same_team`,
`invalid_best_of`, `invalid_tier`, `unsupported_map`, `invalid_map_count`, `duplicate_map`,
`invalid_probability` (compose_series_probability's own numeric-input guard), `unknown_context`,
`prediction_datetime_before_state_contract`, `historical_context_datetime_locked`,
`missing_state_support` (XGB contract-drift guard). No raw stack trace is expected validation
behavior; every code above is exercised by a dedicated test.

## M. Determinism

`model.n_jobs = 1` is preserved for **every** context, including deployment (not just the
historical one Phase 8D originally set it for) — verified directly (`ctx.rf_context.model.n_jobs
== 1`) and by repeated-call equality tests for both RF and XGB. State stores are never mutated by
inference (`feature_engine.build_features`/the composer's `build_future_*` functions are already
read-only by design) — verified by hashing team-level state before/after repeated predictions
across all three prediction modes.

## N. Performance

Measured on this machine, single process, cold-then-warm:

| | Time |
|---|---|
| Historical context cold load | 7.21 s |
| Deployment context cold load (RF+XGB models freshly loaded/shared) | 3.56 s |
| Warm unknown-map (RF) prediction | 21.0 ms |
| Warm known-map (XGB) prediction | 99.9 ms |
| Warm known-maps series, BO3 (3 XGB calls + DP) | 293.6 ms |
| Warm known-maps series, BO5 (5 XGB calls + DP) | 505.8 ms |

Measurement only — no behavior was changed for optimization; XGB per-map latency dominates
`predict_series_known_maps` linearly with `best_of`, as expected from calling `predict_map`
`best_of` times.

## O. Validation

`pytest tests/test_phase9b_application_inference.py` — 65 tests, including the full 2,976-row
historical parity check, 18 known-map wrapper-parity cases (9 maps × 2 contexts) plus a
cold-start-team case, all DP/error/determinism/immutability/JSON-serialization checks, and the
THUNDERdOWNUNDER historical-cold-start-vs-deployment-history lifecycle test. Full repository
suite passes. `scripts/validate_phase9b.py` independently re-runs the full parity check and the
known-map wrapper-parity check (not just re-reading test results) and reports all checks passed.

## P. Limitations

- **Not live/current**: the deployment context reflects local historical data only through
  **June 28, 2026** — it is never presented as reflecting real-world state as of the current
  date, and no code path compares the state cutoff to the machine's clock.
- **9 model-supported maps ≠ any real-world Active Duty pool**: Train and Vertigo are
  model-supported (present in training data) but were not part of Cologne's 7-map competitive
  pool; Cache is supported by neither the frozen model nor the 2026 pool, and is rejected
  explicitly rather than silently routed through the model's own unknown-category fallback.
- **`ambiguous_team` is currently unreachable on real data** (zero duplicate raw names in the
  identity policy today) — implemented and tested against a synthetic case, not real data, stated
  plainly rather than hidden.
- **No explanation engine, no API, no PWA** — out of scope for this phase by design.

---

```
APPLICATION INFERENCE CORE = IMPLEMENTED
HISTORICAL REPLAY PARITY = VERIFIED
DEPLOYMENT STATE = VERSIONED
RF V2 = UNCHANGED
XGB V3 = UNCHANGED
NO RETRAINING
NO EXPLANATION ENGINE YET
NO API YET
NO PWA YET
```

Deployment state reflects local historical data only through June 28, 2026.
