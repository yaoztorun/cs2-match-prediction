# Phase 9D — Versioned Application Prediction API

## A. Architecture

`scripts/application_api.py` is a thin, read-only FastAPI + Pydantic + Uvicorn HTTP transport
over the already-frozen Phase 9B prediction core (`application_inference.py`) and Phase 9C
explanation core (`application_explanations.py`). It contains no model, no state, no feature
engineering, and no probability composition logic of its own — every number it returns comes
from calling directly into `ai.predict_series_unknown_maps` / `ai.predict_map` /
`ai.predict_series_known_maps` / `ae.explain_series_unknown_maps` / `ae.explain_map` /
`ae.explain_series_known_maps`, unmodified. `scripts/run_application_api.py` is a one-line
Uvicorn entrypoint with no logic of its own. All endpoints live under `/api/v1`, configured by
`config/application_api_v1.yaml` (CORS, default context, contract identifiers). No code moved
into `src/`.

The API is a pure function of the Phase 9B/9C contract: it makes **zero state writes** (verified
by both an AST/source-text scan for write-mode file operations and a state-hash-unchanged test
after a full fixture battery — sections K and P below) and never accepts a model/state/config/
context filesystem path from a client — the only "location" a caller can name is a registered
`context_id` string, validated against the Phase 9B context registry, never a path.

## B. Version / startup contract

The default context is **always** `deployment_post_cologne_v1` — `config/application_api_v1.yaml`
pins this, and `application_api.py` asserts `DEFAULT_CONTEXT_ID == ai.DEPLOYMENT_CONTEXT_ID` at
import time so the API can never silently default to the historical replay context.

At process startup (a FastAPI `lifespan` handler, not the deprecated `@app.on_event`), the API
re-verifies the FULL executable Phase 9B/9C contract before accepting traffic:

1. Recomputes `RF_PIPELINE` / `XGB_PIPELINE` / `DEPLOYMENT_STATE` file hashes and compares them
   against `config/application_inference_contexts_v1.yaml` (the same check
   `scripts/validate_phase9b.py` section 2 performs, run again here proactively).
2. Compares the Phase 9A deployment receipt hash referenced by the registry.
3. Recomputes the hashes `data/deployment/application_explanation_receipt_v1.json` recorded at
   Phase 9C commit time (`application_explanations.py`, `application_inference.py`, the feature
   groups registry, the context registry, the RF/XGB model+preprocessing files) and compares them.
4. Calls `ai.get_context(DEPLOYMENT_CONTEXT_ID)`, which exercises `pre_veto_series_predictor
   .verify_model_contract` (RF) and `application_inference._verify_xgb_contract` (XGB) and warms
   the process-level context cache.

If **any** of these checks fails, the API does **not** attempt to rebuild, retrain, or re-derive
anything — it marks itself not-ready and every prediction/metadata endpoint returns
`503 service_unavailable` with the specific failing check(s) in `error.detail`. `/health/live`
still returns `200` (the process is up), but `/health/ready` and every endpoint that depends on a
warmed, contract-verified context refuse to serve stale/drifted data. This was verified directly:
`scripts/validate_phase9d.py` section 2 asserts the startup check passes cleanly against the real
repository state, and `tests/test_phase9d_application_api.py::test_readiness_unavailable_blocks_
prediction_endpoints` simulates a not-ready state and asserts `503` on a prediction call.

## C. Health / readiness

- `GET /api/v1/health/live` — always `200 {"status": "live"}` once the process is up. No contract
  checks, no context access.
- `GET /api/v1/health/ready` — `200` with `default_context_id` and the startup check timestamp
  once section B's verification passed; `503 service_unavailable` with the failing-check detail
  otherwise. The two endpoints have deliberately distinct meanings, per the Phase 9D spec.

## D. Context metadata

`GET /api/v1/contexts` and `GET /api/v1/contexts/{context_id}` read **only**
`ai.list_inference_contexts()` / `ai.get_context_metadata()` — themselves reading only
`config/application_inference_contexts_v1.yaml` — never a filesystem "latest" scan. An unknown
`context_id` raises `ApplicationInferenceError("unknown_context", ...)`, mapped to `404`.

`GET /api/v1/meta` returns safe public metadata only: `api_version`, `prediction_contract`,
`explanation_version`, `default_context_id`, `available_context_ids`, the two model IDs, the
deployment state's `state_cutoff` (labeled `deployment_state_data_through`), and
`state_is_live: false` — always `false`, never inferred from wall-clock time.

## E. Team / map discovery

`GET /api/v1/teams?context_id=&q=&limit=` wraps `ai.list_supported_teams` and applies a
deterministic, case-insensitive **substring** filter on `canonical_name`, then a stable
alphabetical sort and a limit (default 50, max 200, `config/application_api_v1.yaml`). This
filter exists **only** for UI discovery — it does not touch `resolve_team`'s exact-match-only
semantics. Verified directly: `test_teams_search_never_alters_strict_resolution_semantics`
confirms that a name which *would* substring-match in `/teams` (`"Vitality"` vs. the canonical
`"Team Vitality"`) still gets rejected as `unknown_team` (404) by the actual prediction endpoints.

`GET /api/v1/maps?context_id=` wraps `ai.list_supported_maps`. `Cache` never appears (it is not a
named XGB V3 category); `Train`/`Vertigo` appear as `model_supported: true` but
`cologne_2026_competitive_pool: false` — the response is never presented as the current Active
Duty pool, matching the Phase 9B map-registry contract exactly (no new semantics invented at the
HTTP layer).

## F. Series prediction API

`POST /api/v1/predict/series` has two mutually exclusive `mode` values, enforced by a Pydantic
`model_validator`:

- `pre_veto` — RF V2 only. `ordered_maps` **must be null**; supplying it is a
  `422 schema_validation_error` before any model is touched.
- `known_maps` — XGB V3 + the existing generic best-of-N DP composer only. `ordered_maps` is
  **required**, and its length/uniqueness are validated by the SAME core functions
  (`invalid_map_count`, `duplicate_map`) Phase 9B already tests — the HTTP layer adds no new
  validation logic for these, only forwards to the core.

The two modes are never blended: `pre_veto` never touches XGB, `known_maps` never touches RF.

## G. Single-map API

`POST /api/v1/predict/map` is a thin wrapper around `ai.predict_map` / `ae.explain_map` — no
series composition, one map, one call.

## H. Explanation transport

`include_explanation` (default `true`) controls whether an explanation is computed at all — when
`false`, the API calls the plain `predict_*` functions and never touches
`application_explanations.py`, avoiding any unnecessary Saabas/TreeSHAP computation.
`explanation_detail` (`"summary"` default, or `"full"`) controls its shape:

- **`full`** is the complete Phase 9C payload, unmodified, with only a `detail_level: "full"` tag
  added.
- **`summary`** is a **deterministic projection** of the already-computed full explanation,
  implemented by `application_api._summarize_explanation_block` — a pure function over the full
  dict. It keeps `base_value`, `attribution_method`, `attribution_output_space`,
  `human_readable_summary`, `reconstruction_check`, `input_provenance` (RF)/`state_support` (XGB),
  and every `grouped_factors`/`top_positive_factors`/`top_negative_factors` entry trimmed down to
  `{factor_group, direction, signed_contribution, absolute_importance, rank,
  attribution_output_space}` — it drops `feature_contributions` (the full 19- or 131-row array)
  and every group's `supporting_features` list (the same low-level arrays, grouped). **It is never
  recomputed from the model** — `test_summary_is_deterministic_projection_of_full` and
  `test_full_matches_direct_core_explanation_exactly` verify this directly.

For `known_maps` series explanations, `application_api._split_known_series_full` reconstructs the
same `{prediction, explanation}` envelope shape from `ae.explain_series_known_maps`'s single
merged dict: `prediction.ordered_maps[i].probability_team_a/b` come **directly** from
`map_level_explanations[i]["prediction"]` (the same per-map XGB prediction dict `ai.predict_map`
itself produces) — never a separately recomputed value — and `explanation.series_composition`
(reach probability + `series_composition_leverage` per map slot) is carried through unmodified.
The two layers stay explicitly separate, matching Phase 9C's own "never conflated" design.

## I. Error contract

Every non-2xx response is `{"error": {"code", "message", "detail"}, "request_id"}`. The frozen
`error_code -> HTTP status` policy (`application_api.ERROR_STATUS_MAP`):

| error_code | status |
|---|---|
| `unknown_context`, `unknown_team` | 404 |
| `invalid_best_of`, `invalid_tier`, `same_team`, `unsupported_map`, `invalid_map_count`, `duplicate_map`, `invalid_probability`, `historical_context_datetime_locked`, `prediction_datetime_before_state_contract` | 422 |
| `ambiguous_team` | 409 |
| `missing_state_support` (internal contract-drift signal) | 500 |
| unmapped `ApplicationInferenceError` codes | 500 (default) |

**Design note — why `context_id` is not a Pydantic `Literal`.** Enforcing registered context IDs
via a Pydantic `Literal` would make an unknown `context_id` surface as a generic `422
schema_validation_error`, but the frozen policy above requires `unknown_context` to be a `404`.
Instead `context_id` is a plain non-empty string at the schema level, and every code path resolves
it by calling into `application_inference` (`get_context` / `get_context_metadata` /
`list_supported_teams` / `list_supported_maps` / `predict_*`), which raises
`ApplicationInferenceError("unknown_context", ...)` for anything outside the registry — caught by
the handler above and correctly mapped to 404. "Context IDs must be registry values" is still
enforced, just by the business-logic layer that already owns that error code, not by a generic
schema check that would produce the wrong status.

Two additional codes exist purely at the HTTP layer, for concerns
`ApplicationInferenceError` has no vocabulary for:

- `schema_validation_error` (422) — the request body/query did not match the Pydantic contract at
  all (wrong type, e.g. `best_of: true`; an extra field; `ordered_maps` supplied with
  `mode=pre_veto`). `StrictInt` is used for `best_of` specifically so a `bool` is rejected outright
  — pydantic v2's strict-int validation explicitly excludes `bool` (it is a `bool`-is-not-an-`int`
  check, not merely a range check) — verified directly by
  `test_best_of_bool_rejected_over_http`.
- `service_unavailable` (503) — startup contract verification failed or has not completed yet.

No raw traceback is ever returned: a genuinely unexpected exception is caught by
`_handle_unexpected_exception` (registered exception handler) **and** by a `try/except` inside the
request-ID middleware itself — the latter exists because Starlette's `BaseHTTPMiddleware` has a
documented quirk where an exception already converted to a response by a registered handler is
still re-raised through `call_next()`; this was caught empirically (a real test failure during
development, not a theoretical concern) and fixed by adding the second backstop. Both paths return
`500 internal_error` with a generic message; the real exception is logged server-side only
(`logger.exception`), never serialized to the client — verified by
`test_genuinely_unexpected_exception_is_500_no_traceback`.

## J. Freshness / datetime semantics

`prediction_datetime` is passed through to the core **unmodified** — no HTTP-layer timezone
conversion, exactly as specified. Historical context: omitting it or supplying exactly the locked
cutoff (`2026-06-02T13:30:00`) succeeds; anything else raises `historical_context_datetime_locked`
(422). Deployment context: omitting it defaults to the state cutoff; a value strictly before the
cutoff raises `prediction_datetime_before_state_contract` (422, to prevent look-ahead risk); a
value after the cutoff succeeds but is tagged
`data_freshness.mode = "hypothetical_future_from_stale_snapshot"` with a `staleness_days` figure —
never described as fresher data than the snapshot actually contains. `state_is_live` is `false` in
every response, unconditionally.

## K. Concurrency / state safety

Endpoints are defined as plain `def` (synchronous) functions, so FastAPI/Starlette dispatches each
one to a worker thread from its `anyio` threadpool automatically — CPU-bound work (RF/XGB
`predict_proba`, the Saabas tree walk, TreeSHAP) never blocks the async event loop.

**No new locks were added.** The reasoning, verified rather than assumed: (1) the process-level
context cache (`ai._CONTEXT_CACHE` / `ai._SHARED_XGB`) is populated exactly once, at startup,
before the server begins accepting traffic — there is no window where two requests could race to
populate it. (2) Every per-request code path downstream (`fe.build_features`,
`rmmc.build_future_modern_rich_map_features`, `predict_proba`, the Saabas walk, TreeSHAP) only
*reads* from the cached, frozen `StateStore`/model objects and allocates fresh local arrays/dicts
per call — this is the same read-only contract Phase 9A/9B already established and parity-tested
across thousands of calls. (3) This was verified empirically, not assumed: 40 concurrent requests
(mixed with/without explanation) against the same cached deployment context all returned the exact
same probability as a direct, single-threaded core call, and the cached context object's identity
was confirmed unchanged before/after (`ai._CONTEXT_CACHE[...] is ctx_before`) — i.e. no per-request
reload occurred. Both `tests/test_phase9d_application_api.py` and `scripts/validate_phase9d.py`
run this check independently.

## L. Prediction parity (hard gate)

HTTP prediction probability was verified to **exactly equal** (`==`, not within tolerance) the
direct Phase 9B Python call across:

- `pre_veto`: historical + deployment contexts, BO1/BO3/BO5, plus the THUNDERdOWNUNDER
  historical-context cold-start case.
- `known_maps` single-map: all 9 model-supported maps, both historical and deployment contexts.
- `known_maps` series: BO1/BO3/BO5, full `prediction` dict compared field-for-field
  (`ordered_maps`, `series_probability_team_a/b`, `favored_team`, `prediction_is_tied`, etc.).

All pass exactly, both in `tests/test_phase9d_application_api.py` and independently in
`scripts/validate_phase9d.py` section 7.

## M. Explanation parity (hard gate)

`explanation_detail="full"` was verified to **exactly equal** (after stripping only the
API-added `detail_level` transport tag, and for the single-map case the `state_support` key which
`ae.explain_map` returns as a sibling rather than nested) the direct Phase 9C
`ae.explain_series_unknown_maps` / `ae.explain_map` output — for both the RF (Saabas,
probability-space) and XGB (TreeSHAP, log-odds-space) attribution paths. `series_composition`
(reach probability + leverage) for `known_maps` series explanations matches
`ae.explain_series_known_maps` exactly. Verified in both the test suite and
`scripts/validate_phase9d.py` section 8.

## N. OpenAPI

`/openapi.json` loads and declares every public endpoint (`/api/v1/health/live`,
`/api/v1/health/ready`, `/api/v1/meta`, `/api/v1/contexts`, `/api/v1/contexts/{context_id}`,
`/api/v1/teams`, `/api/v1/maps`, `/api/v1/predict/series`, `/api/v1/predict/map`). Request models
(`SeriesPredictionRequest`, `MapPredictionRequest`) and the response envelope
(`PredictionEnvelope`, `ResponseMetadata`) are fully typed Pydantic models with field-level
descriptions covering the `pre_veto` vs `known_maps` distinction, the non-causal nature of
explanations, the stale-snapshot freshness mode, and the summary/full explanation distinction.
`prediction`/`explanation` payload bodies are typed as open dictionaries (`Dict[str, Any]`) rather
than fully re-declared field-by-field Pydantic models — a deliberate choice discussed in section Q.
No internal filesystem path is present anywhere in the generated spec (checked directly).

## O. Performance (descriptive only — no behavior changes made for benchmarking)

Cold start (fresh Python process: import FastAPI/pandas/sklearn/xgboost, run the full startup
contract verification, warm the deployment context — RF+XGB models, all 5 state stores): **~7.0
seconds**. First `/health/ready` call after that: **~7 ms**.

Core (direct Python, no HTTP) vs. warm HTTP (in-process `TestClient`, includes ASGI routing +
Pydantic validation/serialization) median latency, 20 iterations each (10 for the two BO5/BO3
known-series cases given their cost):

| Call | Core (ms) | HTTP (ms) |
|---|---|---|
| RF pre-veto explanation (BO3) | 31.3 | 35.2 (incl. request/response overhead) |
| XGB single-map explanation | 63.9 | 68.1 |
| Known-series explanation (BO3, 3 XGB calls) | 191.4 | 201.3 |
| Known-series explanation (BO5, 5 XGB calls) | 316.2 | 331.0 |

Additional warm HTTP-only latencies: `/api/v1/meta` **16.6 ms**; `/api/v1/teams?q=` **6.3 ms**;
pre-veto without explanation **21.7 ms** vs. with explanation **35.2 ms**; single-map without
explanation **62.5 ms** vs. with explanation **68.1 ms**; BO3 known-series without explanation
**185.3 ms** vs. with **201.3 ms**; BO5 known-series without explanation **297.0 ms** vs. with
**331.0 ms**. The known-map/known-series numbers are dominated by `rich_modern_map_feature_
composer.build_future_modern_rich_map_features`'s feature composition (already the case in Phase
9B/9C — the HTTP layer adds a roughly constant 4–15 ms of routing/validation/serialization
overhead on top, not a multiplicative cost).

## P. Validation

- `pytest tests/test_phase9d_application_api.py` — 85/85 passed.
- Full repository `pytest` — **759 passed** (674 Phase-9C baseline + 85 new Phase 9D tests), zero
  regressions.
- `scripts/validate_phase9b.py` — 32/32 (re-run as a real subprocess from `validate_phase9d.py`,
  not re-implemented).
- `scripts/validate_phase9c.py` — 33/33 (same).
- `scripts/validate_phase9d.py` — independently re-verifies: receipt hashes vs. disk, the startup
  contract, liveness/readiness, strict-typing rejection (bool `best_of`, extra fields), the frozen
  error-status policy end-to-end, prediction parity, explanation parity, OpenAPI coverage, zero
  state mutation across a full fixture battery, concurrency determinism, absence of any
  fitting/training import or write-mode file operation in `application_api.py`, absence of any
  path/file-style request field, and that the Phase 9B/9C validators still pass.

## Q. Limitations

- `prediction`/`explanation` response bodies are typed as open `Dict[str, Any]` in the OpenAPI
  schema rather than fully re-declared Pydantic models field-by-field. This is deliberate: those
  dict shapes are the already-versioned, already-parity-tested Phase 9B/9C contracts; re-declaring
  ~150 fields in Pydantic here would risk exactly the kind of transformation/drift bug the hard
  parity gates (sections L/M) exist to catch, for no behavioral benefit. Request models and the
  enclosing envelope/error schemas ARE fully typed.
- No tournament/Monte-Carlo simulator routes, no historical Cologne replay routes beyond the
  prediction/explanation parity checks above, and no Next.js/React/Tailwind/PWA work — all
  explicitly out of scope for Phase 9D.
- No authentication layer exists, matching the current state of the rest of the repository; none
  was added.
- The deployment context's underlying application state is data **through 2026-06-28 only** —
  the API never claims to be live or current, and `state_is_live` is unconditionally `false` in
  every response.

```
APPLICATION API V1 = IMPLEMENTED
PREDICTION PARITY = VERIFIED
EXPLANATION PARITY = VERIFIED
DEPLOYMENT DATA THROUGH 2026-06-28
RF V2 = UNCHANGED
XGB V3 = UNCHANGED
NO RETRAINING
NO TOURNAMENT API YET
NO PWA YET
```
