"""
Phase 9D versioned application prediction API. A thin, read-only HTTP
transport over the ALREADY-FROZEN Phase 9B prediction core
(application_inference.py) and Phase 9C explanation core
(application_explanations.py) - it never recomputes a probability or an
attribution differently than those modules already do, and it makes zero
state writes.

Everything lives under /api/v1. The default context is ALWAYS
`deployment_post_cologne_v1` (config/application_api_v1.yaml) - the API
never silently defaults to the historical replay context.

Error contract: every non-2xx response is `{error: {code, message, detail},
request_id}`. `ai.ApplicationInferenceError.error_code` values are mapped to
HTTP status via the frozen ERROR_STATUS_MAP below. Two additional API-layer
codes exist for concerns the inference core has no vocabulary for:
`schema_validation_error` (422 - the request body/query did not match the
Pydantic contract, e.g. a bool where an int best_of was required) and
`service_unavailable` (503 - startup contract verification failed or has not
completed; the API never auto-rebuilds a drifted state/model).

Context IDs are deliberately NOT enforced via a Pydantic Literal: doing so
would make an unknown context_id surface as a generic 422
schema_validation_error, but the frozen status policy requires
unknown_context to be a 404. Instead every code path resolves context_id by
calling into application_inference (get_context / get_context_metadata /
list_supported_teams / list_supported_maps / predict_*), which raises
ApplicationInferenceError("unknown_context", ...) for anything not in the
Phase 9B registry - caught by the handler below and mapped to 404.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from _common import ROOT
import application_inference as ai
import application_explanations as ae
import application_tournament_service as ats
import build_application_registries as bar
from application_tournament_router import router as major_router

DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"

logger = logging.getLogger("application_api")

_API_CONFIG = yaml.safe_load((CONFIG / "application_api_v1.yaml").read_text(encoding="utf-8"))

API_VERSION = _API_CONFIG["api_version"]
PREDICTION_CONTRACT = _API_CONFIG["prediction_contract"]
EXPLANATION_VERSION = _API_CONFIG["explanation_version"]
DEFAULT_CONTEXT_ID = _API_CONFIG["default_context_id"]
assert DEFAULT_CONTEXT_ID == ai.DEPLOYMENT_CONTEXT_ID, "default_context_id must be the deployment context, never historical"

# Frozen error_code -> HTTP status policy.
ERROR_STATUS_MAP = {
    "unknown_context": 404,
    "unknown_team": 404,
    "invalid_best_of": 422,
    "invalid_tier": 422,
    "same_team": 422,
    "unsupported_map": 422,
    "invalid_map_count": 422,
    "duplicate_map": 422,
    "invalid_probability": 422,
    "historical_context_datetime_locked": 422,
    "prediction_datetime_before_state_contract": 422,
    "ambiguous_team": 409,
    "missing_state_support": 500,  # internal contract-drift signal, not a client input error
    # Phase 9E tournament-specific codes (additive - Phase 9D codes above are unchanged)
    "unknown_ruleset": 404,
    "invalid_participant_count": 422,
    "duplicate_team": 422,
    "invalid_seed": 422,
    "missing_seed": 422,
    "invalid_override": 422,
    "override_team_mismatch": 422,
    "duplicate_override": 422,
    "contradictory_override": 422,
    "invalid_simulation_count": 422,
    "probability_matrix_incomplete": 500,  # internal contract-drift signal, not a client input error
}
DEFAULT_ERROR_STATUS = 500

_STARTUP_STATE: Dict[str, Any] = {
    "ready": False, "checked_at": None, "detail": {},
    "subsystems": {"prediction_ready": False, "explanation_ready": False, "tournament_engine_ready": False,
                    "historical_cologne_ready": False},
    "subsystem_detail": {},
}


# ---------------------------------------------------------------------------
# Startup contract verification (never auto-rebuilds a drifted artifact)
# ---------------------------------------------------------------------------

def _verify_prediction_and_explanation_contract():
    """Unchanged Phase 9D behavior, split into two independently-reported
    booleans (amendment #22/#3 of the Phase 9E plan) so a tournament/
    historical subsystem failure can never affect prediction_ready or
    explanation_ready, and vice versa."""
    detail: Dict[str, Any] = {}
    prediction_ok = True
    explanation_ok = True
    try:
        registry = yaml.safe_load((CONFIG / "application_inference_contexts_v1.yaml").read_text(encoding="utf-8"))
        fresh_rf = bar.hash_group(bar.RF_PIPELINE)
        fresh_xgb = bar.hash_group(bar.XGB_PIPELINE)
        fresh_state = bar.hash_group(bar.DEPLOYMENT_STATE)
        rf_ok = fresh_rf == registry["rf_unknown_map_pipeline"]
        xgb_ok = fresh_xgb == registry["xgb_known_map_pipeline"]
        state_ok = fresh_state == registry["contexts"][DEFAULT_CONTEXT_ID]["state_hashes"]
        receipt_ok = (bar.sha256_file(bar.DEPLOYMENT_RECEIPT) ==
                      registry["contexts"][DEFAULT_CONTEXT_ID]["phase9a_deployment_receipt_hash"])
        detail["rf_pipeline_hashes_match"] = rf_ok
        detail["xgb_pipeline_hashes_match"] = xgb_ok
        detail["deployment_state_hashes_match"] = state_ok
        detail["deployment_receipt_hash_matches"] = receipt_ok
        prediction_ok = rf_ok and xgb_ok and state_ok and receipt_ok

        ctx = ai.get_context(DEFAULT_CONTEXT_ID)  # exercises RF + XGB model-contract verification, warms the cache
        detail["default_context_loaded"] = True
        detail["default_context_state_cutoff"] = str(ctx.state_cutoff)
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure here means "not ready"
        prediction_ok = False
        detail["prediction_exception"] = f"{type(e).__name__}: {e}"

    try:
        exp_receipt = json.loads((DATA / "deployment" / "application_explanation_receipt_v1.json")
                                  .read_text(encoding="utf-8"))
        exp_hashes = exp_receipt["hashes"]
        explanation_ok = (
            exp_hashes["application_explanations_py"] == bar.sha256_file(SCRIPTS / "application_explanations.py")
            and exp_hashes["application_inference_py"] == bar.sha256_file(SCRIPTS / "application_inference.py")
            and exp_hashes["feature_groups_registry"] == bar.sha256_file(
                CONFIG / "application_explanation_feature_groups_v1.yaml")
            and exp_hashes["phase9b_context_registry"] == bar.sha256_file(
                CONFIG / "application_inference_contexts_v1.yaml")
            and exp_hashes["rf_v2_model"] == bar.sha256_file(bar.RF_PIPELINE["rf_model"])
            and exp_hashes["rf_v2_preprocessing"] == bar.sha256_file(bar.RF_PIPELINE["rf_preprocessing"])
            and exp_hashes["xgb_v3_model"] == bar.sha256_file(bar.XGB_PIPELINE["xgb_model"])
            and exp_hashes["xgb_v3_preprocessing"] == bar.sha256_file(bar.XGB_PIPELINE["xgb_preprocessing"])
        )
        detail["explanation_receipt_hashes_match"] = explanation_ok
    except Exception as e:  # noqa: BLE001
        explanation_ok = False
        detail["explanation_exception"] = f"{type(e).__name__}: {e}"

    return prediction_ok, explanation_ok, detail


@asynccontextmanager
async def lifespan(app: FastAPI):
    prediction_ok, explanation_ok, pred_detail = _verify_prediction_and_explanation_contract()
    engine_ok, engine_detail = ats.verify_ruleset_matches_engine_rules()
    historical_ok, historical_detail = ats.verify_historical_cologne_contract()

    _STARTUP_STATE["subsystems"] = {
        "prediction_ready": prediction_ok, "explanation_ready": explanation_ok,
        "tournament_engine_ready": engine_ok, "historical_cologne_ready": historical_ok,
    }
    _STARTUP_STATE["subsystem_detail"] = {
        "prediction": pred_detail, "tournament_engine": engine_detail, "historical_cologne": historical_detail,
    }
    # Amendment #3/#22: /health/ready and the /predict/* gate stay governed by the ORIGINAL Phase 9D
    # core contract only (prediction + explanation) - a tournament/historical subsystem failure never
    # takes the base prediction service down. Each /major/* route enforces its own required subsystem.
    core_ready = prediction_ok and explanation_ok
    _STARTUP_STATE["ready"] = core_ready
    _STARTUP_STATE["checked_at"] = pd.Timestamp.now("UTC").isoformat()
    _STARTUP_STATE["detail"] = pred_detail
    if core_ready:
        logger.info("startup contract verification passed; default context warmed")
    else:
        logger.error("startup contract verification FAILED: %s", pred_detail)
    if not engine_ok:
        logger.error("tournament engine subsystem verification FAILED: %s", engine_detail)
    if not historical_ok:
        logger.error("historical Cologne subsystem verification FAILED: %s", historical_detail)
    yield
    ats.shutdown_process_pool()  # amendment #6: deterministic lifecycle, no orphan worker processes


app = FastAPI(
    title="CS2 Match Prediction Application API",
    version=API_VERSION,
    description=(
        "Read-only HTTP API over the frozen Phase 9B prediction core and Phase 9C explanation core. "
        "Two prediction modes exist and are never blended: `pre_veto` (RF V2, unknown maps) and "
        "`known_maps` (XGB V3 + a generic best-of-N dynamic program, exact map order known). "
        "Explanations are deterministic, model-grounded feature attributions - NOT causal claims. "
        "The default context (`deployment_post_cologne_v1`) is the latest locally available "
        "application state (data through 2026-06-28), never a live/current claim."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_API_CONFIG["cors"]["allow_origins"],
    allow_credentials=_API_CONFIG["cors"]["allow_credentials"],
    allow_methods=_API_CONFIG["cors"]["allow_methods"],
    allow_headers=_API_CONFIG["cors"]["allow_headers"],
)


@app.middleware("http")
async def _request_id_and_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Starlette's BaseHTTPMiddleware re-raises an exception through
        # call_next() even after a registered @app.exception_handler already
        # built a response for it - caught here as a second, authoritative
        # backstop so a genuinely unexpected error still returns the
        # structured 500 envelope instead of propagating past this ASGI app.
        logger.exception("unhandled exception on %s", request.url.path)
        response = _error_response(request, 500, "internal_error", "an internal error occurred", {})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    context_id = getattr(request.state, "context_id", None)
    mode = getattr(request.state, "mode", None)
    logger.info("endpoint=%s method=%s status=%s latency_ms=%.2f context_id=%s mode=%s request_id=%s",
                request.url.path, request.method, response.status_code, elapsed_ms, context_id, mode, request_id)
    return response


class ServiceUnavailableError(Exception):
    def __init__(self, detail):
        self.detail = detail


def _require_ready():
    if not _STARTUP_STATE["ready"]:
        raise ServiceUnavailableError(_STARTUP_STATE["detail"])


# ---------------------------------------------------------------------------
# Error envelope + handlers (single structured shape for every non-2xx)
# ---------------------------------------------------------------------------

def _error_response(request: Request, status_code: int, code: str, message: str, detail=None):
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "detail": detail or {}}, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ai.ApplicationInferenceError)
async def _handle_application_inference_error(request: Request, exc: ai.ApplicationInferenceError):
    status_code = ERROR_STATUS_MAP.get(exc.error_code, DEFAULT_ERROR_STATUS)
    return _error_response(request, status_code, exc.error_code, exc.message, ai._to_jsonable(exc.detail))


def _sanitize_validation_errors(errors):
    """pydantic's RequestValidationError.errors() embeds raw exception
    objects in ctx.error for value_error-type failures (e.g. our
    model_validator ValueErrors) - not JSON-serializable as-is."""
    def sanitize(v):
        if isinstance(v, BaseException):
            return str(v)
        if isinstance(v, dict):
            return {k: sanitize(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [sanitize(vv) for vv in v]
        try:
            json.dumps(v)
            return v
        except TypeError:
            return str(v)
    return [sanitize(e) for e in errors]


@app.exception_handler(RequestValidationError)
async def _handle_request_validation_error(request: Request, exc: RequestValidationError):
    return _error_response(request, 422, "schema_validation_error", "request failed schema validation",
                            {"errors": _sanitize_validation_errors(exc.errors())})


@app.exception_handler(ServiceUnavailableError)
async def _handle_service_unavailable(request: Request, exc: ServiceUnavailableError):
    return _error_response(request, 503, "service_unavailable",
                            "startup contract verification failed or has not completed", exc.detail)


@app.exception_handler(Exception)
async def _handle_unexpected_exception(request: Request, exc: Exception):
    logger.exception("unhandled exception on %s", request.url.path)
    return _error_response(request, 500, "internal_error", "an internal error occurred", {})


# ---------------------------------------------------------------------------
# Response envelope schemas
# ---------------------------------------------------------------------------

class ResponseMetadata(BaseModel):
    context_id: str
    state_data_through: str
    state_is_live: bool
    prediction_contract: str
    explanation_version: str


class PredictionEnvelope(BaseModel):
    request_id: str
    api_version: str
    prediction: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None
    metadata: ResponseMetadata


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SeriesPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_id: str = Field(min_length=1, description="Registered inference context id (see GET /api/v1/contexts).")
    mode: Literal["pre_veto", "known_maps"] = Field(
        description="'pre_veto' = RF V2 only, map order unknown, ordered_maps must be null. "
                    "'known_maps' = XGB V3 + dynamic-program series composition, exact ordered_maps required.")
    team_a: str = Field(min_length=1)
    team_b: str = Field(min_length=1)
    best_of: StrictInt = Field(description="Must be 1, 3, or 5. Strictly typed - a bool is never accepted.")
    tier: Optional[Literal["tier1", "tier2", "tier3"]] = None
    ordered_maps: Optional[List[str]] = None
    prediction_datetime: Optional[str] = Field(
        default=None,
        description="ISO-8601 datetime, passed through unmodified (no HTTP-layer timezone conversion). "
                    "Historical context: omit, or supply exactly the locked cutoff. Deployment context: "
                    "omit to default to the state cutoff; a value after the cutoff is accepted only as a "
                    "hypothetical stale-snapshot projection.")
    include_explanation: StrictBool = True
    explanation_detail: Literal["summary", "full"] = Field(
        default="summary",
        description="'summary' (default) omits the full 19/131-feature low-level arrays. 'full' returns the "
                    "complete Phase 9C attribution payload.")

    @model_validator(mode="after")
    def _mode_ordered_maps_contract(self):
        if self.mode == "pre_veto" and self.ordered_maps is not None:
            raise ValueError("ordered_maps must be null when mode='pre_veto'")
        if self.mode == "known_maps" and not self.ordered_maps:
            raise ValueError("ordered_maps is required and non-empty when mode='known_maps'")
        return self


class MapPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_id: str = Field(min_length=1)
    team_a: str = Field(min_length=1)
    team_b: str = Field(min_length=1)
    map_name: str = Field(min_length=1)
    best_of: StrictInt
    tier: Optional[Literal["tier1", "tier2", "tier3"]] = None
    prediction_datetime: Optional[str] = None
    include_explanation: StrictBool = True
    explanation_detail: Literal["summary", "full"] = "summary"


# ---------------------------------------------------------------------------
# Explanation detail projection (transport-only; deterministic function of
# the already-computed Phase 9C payload - never recomputed differently)
# ---------------------------------------------------------------------------

_GROUP_SUMMARY_KEYS = ("factor_group", "direction", "signed_contribution", "absolute_importance", "rank",
                        "attribution_output_space")
_PASSTHROUGH_EXTRA_KEYS = ("input_provenance", "state_support")


def _trim_group(g):
    return {k: g[k] for k in _GROUP_SUMMARY_KEYS if k in g}


def _summarize_explanation_block(explanation):
    out = {
        "explanation_type": explanation["explanation_type"],
        "causal": explanation["causal"],
        "attribution_method": explanation["attribution_method"],
        "attribution_output_space": explanation["attribution_output_space"],
        "base_value": explanation["base_value"],
        "grouped_factors": [_trim_group(g) for g in explanation["grouped_factors"]],
        "team_a_factors": [_trim_group(g) for g in explanation["team_a_factors"]],
        "team_b_factors": [_trim_group(g) for g in explanation["team_b_factors"]],
        "neutral_factors": [_trim_group(g) for g in explanation["neutral_factors"]],
        "top_positive_factors": [_trim_group(g) for g in explanation["top_positive_factors"]],
        "top_negative_factors": [_trim_group(g) for g in explanation["top_negative_factors"]],
        "human_readable_summary": explanation["human_readable_summary"],
        "reconstruction_check": explanation["reconstruction_check"],
        "detail_level": "summary",
    }
    for k in _PASSTHROUGH_EXTRA_KEYS:
        if k in explanation:
            out[k] = explanation[k]
    return out


def _finalize_explanation_detail(explanation, detail_level):
    if detail_level == "full":
        return dict(explanation, detail_level="full")
    return _summarize_explanation_block(explanation)


def _split_known_series_full(full, detail_level):
    """Splits application_explanations.explain_series_known_maps's single
    merged dict into the same {prediction, explanation} shape every other
    endpoint uses. `prediction` is reconstructed field-for-field from the
    SAME underlying per-map prediction dicts application_inference.predict_map
    already produced inside `full` - never a separately recomputed value, so
    it is numerically identical to a direct predict_series_known_maps() call
    by construction."""
    ordered_maps_out = []
    map_level_out = []
    for me in full["map_level_explanations"]:
        p = me["prediction"]
        ordered_maps_out.append({"map_number": me["map_number"], "map_name": me["map_name"],
                                  "probability_team_a": p["probability_team_a"],
                                  "probability_team_b": p["probability_team_b"]})
        exp = dict(me["explanation"])
        exp["state_support"] = me["state_support"]
        map_level_out.append({"map_number": me["map_number"], "map_name": me["map_name"],
                               "probability_team_a": p["probability_team_a"],
                               "probability_team_b": p["probability_team_b"],
                               "explanation": _finalize_explanation_detail(exp, detail_level)})
    prediction = {
        "prediction_mode": full["prediction_mode"], "team_a": full["team_a"], "team_b": full["team_b"],
        "best_of": full["best_of"], "ordered_maps": ordered_maps_out,
        "series_probability_team_a": full["series_probability_team_a"],
        "series_probability_team_b": full["series_probability_team_b"],
        "favored_team": full["favored_team"], "prediction_is_tied": full["prediction_is_tied"],
        "model_id": "map_xgboost_v3_final", "composition_method": full["composition_method"],
        "context_id": full["context_id"], "state_cutoff": full["state_cutoff"],
    }
    explanation = {
        "map_level_explanations": map_level_out,
        "series_composition": full["series_composition"],
        "note": full["note"],
    }
    return prediction, explanation


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/v1/health/live", summary="Liveness - process is up")
def health_live():
    return {"status": "live"}


@app.get("/api/v1/health/ready", summary="Readiness - contract verified and default context can serve")
def health_ready(request: Request):
    """`status`/the 503 gate below are governed EXCLUSIVELY by the original
    Phase 9D core contract (prediction_ready and explanation_ready) -
    unchanged from Phase 9D. `subsystems` is purely additive (Phase 9E):
    tournament_engine_ready/historical_cologne_ready are reported here for
    visibility, but a failure in either never marks the base service
    not-ready - each /api/v1/major/* route enforces its own required
    subsystem independently via its own 503."""
    if not _STARTUP_STATE["ready"]:
        raise ServiceUnavailableError(_STARTUP_STATE["detail"])
    return {"status": "ready", "default_context_id": DEFAULT_CONTEXT_ID, "checked_at": _STARTUP_STATE["checked_at"],
            "subsystems": _STARTUP_STATE["subsystems"]}


# ---------------------------------------------------------------------------
# Metadata / registry discovery
# ---------------------------------------------------------------------------

@app.get("/api/v1/meta", summary="Safe public metadata")
def get_meta(request: Request):
    _require_ready()
    ctx_meta = ai.get_context_metadata(DEFAULT_CONTEXT_ID)
    explanation_meta = ae.get_explanation_metadata()
    return {
        "request_id": request.state.request_id,
        "api_version": API_VERSION,
        "prediction_contract": PREDICTION_CONTRACT,
        "explanation_version": EXPLANATION_VERSION,
        "default_context_id": DEFAULT_CONTEXT_ID,
        "available_context_ids": sorted(c["context_id"] for c in ai.list_inference_contexts()),
        "model_ids": {"series_unknown_map": "series_random_forest_v2", "known_map": "map_xgboost_v3_final"},
        "deployment_state_data_through": ctx_meta["state_cutoff"],
        "state_is_live": False,
        "explanation_causal": explanation_meta["causal"],
    }


@app.get("/api/v1/contexts", summary="List all registered inference contexts")
def get_contexts(request: Request):
    return {"request_id": request.state.request_id, "contexts": ai.list_inference_contexts()}


@app.get("/api/v1/contexts/{context_id}", summary="Metadata for one registered inference context")
def get_context_by_id(context_id: str, request: Request):
    request.state.context_id = context_id
    meta = ai.get_context_metadata(context_id)
    return {"request_id": request.state.request_id, **meta}


@app.get("/api/v1/teams", summary="Deterministic substring team search (UI discovery only)")
def get_teams(request: Request, context_id: str = Query(..., min_length=1),
              q: Optional[str] = Query(None, description="Case-insensitive substring filter on canonical_name."),
              limit: Optional[int] = Query(None, ge=1, le=200)):
    _require_ready()
    request.state.context_id = context_id
    cfg = _API_CONFIG["teams_search"]
    eff_limit = limit if limit is not None else cfg["default_limit"]
    teams = ai.list_supported_teams(context_id)
    if q:
        needle = q.strip().lower()
        teams = [t for t in teams if needle in t["canonical_name"].lower()]
    teams = sorted(teams, key=lambda t: t["canonical_name"])[:eff_limit]
    return {"request_id": request.state.request_id, "context_id": context_id, "count": len(teams), "teams": teams}


@app.get("/api/v1/maps", summary="Structured map support metadata (never the current Active Duty pool)")
def get_maps(request: Request, context_id: str = Query(..., min_length=1)):
    _require_ready()
    request.state.context_id = context_id
    maps = ai.list_supported_maps(context_id)
    return {"request_id": request.state.request_id, "context_id": context_id, "maps": maps}


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/predict/series",
    response_model=PredictionEnvelope,
    summary="Series-level prediction: pre_veto (RF V2) or known_maps (XGB V3 + DP), never blended",
)
def predict_series(payload: SeriesPredictionRequest, request: Request):
    _require_ready()
    request.state.context_id = payload.context_id
    request.state.mode = payload.mode

    if payload.mode == "pre_veto":
        if payload.include_explanation:
            full = ae.explain_series_unknown_maps(payload.context_id, payload.team_a, payload.team_b,
                                                    payload.best_of, prediction_datetime=payload.prediction_datetime,
                                                    tier=payload.tier)
            prediction = full["prediction"]
            explanation = _finalize_explanation_detail(full["explanation"], payload.explanation_detail)
        else:
            prediction = ai.predict_series_unknown_maps(payload.context_id, payload.team_a, payload.team_b,
                                                          payload.best_of,
                                                          prediction_datetime=payload.prediction_datetime,
                                                          tier=payload.tier)
            explanation = None
    else:
        if payload.include_explanation:
            full = ae.explain_series_known_maps(payload.context_id, payload.team_a, payload.team_b, payload.best_of,
                                                 payload.ordered_maps, prediction_datetime=payload.prediction_datetime,
                                                 tier=payload.tier)
            prediction, explanation = _split_known_series_full(full, payload.explanation_detail)
        else:
            prediction = ai.predict_series_known_maps(payload.context_id, payload.team_a, payload.team_b,
                                                        payload.best_of, payload.ordered_maps,
                                                        prediction_datetime=payload.prediction_datetime,
                                                        tier=payload.tier)
            explanation = None

    metadata = ResponseMetadata(context_id=payload.context_id, state_data_through=prediction["state_cutoff"],
                                 state_is_live=False, prediction_contract=PREDICTION_CONTRACT,
                                 explanation_version=EXPLANATION_VERSION)
    return PredictionEnvelope(request_id=request.state.request_id, api_version=API_VERSION,
                               prediction=prediction, explanation=explanation, metadata=metadata)


@app.post(
    "/api/v1/predict/map",
    response_model=PredictionEnvelope,
    summary="Single-map prediction (XGB V3 known-map)",
)
def predict_map(payload: MapPredictionRequest, request: Request):
    _require_ready()
    request.state.context_id = payload.context_id
    request.state.mode = "known_map"

    if payload.include_explanation:
        full = ae.explain_map(payload.context_id, payload.team_a, payload.team_b, payload.map_name,
                               payload.best_of, prediction_datetime=payload.prediction_datetime, tier=payload.tier)
        prediction = full["prediction"]
        exp = dict(full["explanation"])
        exp["state_support"] = full["state_support"]
        explanation = _finalize_explanation_detail(exp, payload.explanation_detail)
    else:
        prediction = ai.predict_map(payload.context_id, payload.team_a, payload.team_b, payload.map_name,
                                     payload.best_of, prediction_datetime=payload.prediction_datetime,
                                     tier=payload.tier)
        explanation = None

    metadata = ResponseMetadata(context_id=payload.context_id, state_data_through=prediction["state_cutoff"],
                                 state_is_live=False, prediction_contract=PREDICTION_CONTRACT,
                                 explanation_version=EXPLANATION_VERSION)
    return PredictionEnvelope(request_id=request.state.request_id, api_version=API_VERSION,
                               prediction=prediction, explanation=explanation, metadata=metadata)


# ---------------------------------------------------------------------------
# Phase 9E: Major tournament service routes (additive - see
# application_tournament_router.py / application_tournament_service.py)
# ---------------------------------------------------------------------------

app.include_router(major_router)
