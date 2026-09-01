"""
Phase 9B application inference core. Reusable Python layer the future
API/PWA will call. Wraps the ALREADY-FROZEN RF V2 (unknown-map) and XGB V3
(known-map) pipelines - no retraining, no feature changes, no calibration,
no ensembling, no probability symmetrization.

Two explicit, never-conflated inference contexts (config/
application_inference_contexts_v1.yaml is the sole source of truth - no
filesystem "latest" scanning): `historical_cologne_pre_event` (locked to the
frozen Phase 8D cutoff, for historical parity/replay) and
`deployment_post_cologne_v1` (the latest locally available application
state, cutoff 2026-06-28T20:00:00 - NOT live, NOT current).

Public surface: list_inference_contexts, get_context_metadata,
list_supported_teams, list_supported_maps, predict_series_unknown_maps,
predict_map, predict_series_known_maps, compose_series_probability.
Internal StateStore/model objects are never returned to callers.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from _common import ROOT
import feature_engineering.series.feature_engine as fe
import feature_engineering.maps.map_feature_engine as mfe
import feature_engineering.form.team_form_engine as tfe
import feature_engineering.roster.player_roster_feature_engine as prfe
import feature_engineering.maps.modern_map_feature_engine as mmfe
import feature_engineering.maps.rich_modern_map_feature_composer as rmmc
import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb
import feature_engineering.preprocessing.preprocessing_random_forest_v1 as prep_rf
from feature_engineering.preprocessing.preprocessing_common_map_v3 import load_map_v3_roles
import tournament.simulation.pre_veto_series_predictor as pvp

CONFIG = ROOT / "config"
DATA = ROOT / "data"
CONTEXTS_REGISTRY_PATH = CONFIG / "application" / "application_inference_contexts_v1.yaml"
MAP_REGISTRY_PATH = CONFIG / "application" / "application_map_registry_v1.yaml"
IDENTITY_POLICY_PATH = DATA / "interim" / "team_identity_policy.csv"
XGB_ROLES_CONFIG_PATH = CONFIG / "features" / "map_features_v3_modern_map.yaml"

HISTORICAL_CONTEXT_ID = "historical_cologne_pre_event"
DEPLOYMENT_CONTEXT_ID = "deployment_post_cologne_v1"


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------

class ApplicationInferenceError(Exception):
    """The one structured error type this module raises. `error_code` is one
    of a fixed, documented set (see reports/phase9b_application_inference_core.md
    section L) that a future API layer maps to HTTP responses. Never a raw
    stack trace as expected validation behavior."""
    def __init__(self, error_code, message, detail=None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.detail = detail or {}

    def to_dict(self):
        return {"error_code": self.error_code, "message": self.message, "detail": self.detail}


def _err(code, message, **detail):
    raise ApplicationInferenceError(code, message, detail=detail)


# ---------------------------------------------------------------------------
# JSON-safety
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Team identity resolution (amendment #2: respects Phase 2.5 eligibility)
# ---------------------------------------------------------------------------

@dataclass
class _IdentityPolicy:
    name_to_canonical: dict          # raw team_name/canonical_team_name -> {canonical names}
    canonical_eligible: dict         # canonical_team_name -> bool
    canonical_decision: dict         # canonical_team_name -> phase25_decision (or "" if n/a)


def _load_identity_policy():
    df = pd.read_csv(IDENTITY_POLICY_PATH)
    name_to_canonical = {}
    for _, row in df.iterrows():
        name_to_canonical.setdefault(row["team_name"].strip(), set()).add(row["canonical_team_name"])
        name_to_canonical.setdefault(row["canonical_team_name"].strip(), set()).add(row["canonical_team_name"])
    canonical_eligible = df.groupby("canonical_team_name")["identity_feature_eligible"].min().astype(bool).to_dict()
    canonical_decision = df.set_index("canonical_team_name")["phase25_decision"].to_dict()
    return _IdentityPolicy(name_to_canonical=name_to_canonical, canonical_eligible=canonical_eligible,
                            canonical_decision=canonical_decision)


def resolve_team(name, policy: "_IdentityPolicy"):
    """Exact match ONLY (against raw team_name or canonical_team_name, trimmed) -
    no fuzzy/edit-distance/embedding matching, no new aliases invented. A team
    is application-resolvable only if the frozen Phase 2.5 policy marks it
    identity_feature_eligible - intentionally-ineligible identities,
    unresolved collisions, and unsafe org merges are never re-admitted."""
    if not isinstance(name, str) or not name.strip():
        _err("unknown_team", f"team name is empty or not a string: {name!r}", requested_name=name)
    key = name.strip()
    candidates = policy.name_to_canonical.get(key)
    if not candidates:
        _err("unknown_team", f"{key!r} does not match any known team identity (Phase 2.5 policy)",
             requested_name=name)
    if len(candidates) > 1:
        _err("ambiguous_team", f"{key!r} resolves to more than one canonical identity: {sorted(candidates)}",
             requested_name=name, candidates=sorted(candidates))
    canonical = next(iter(candidates))
    if not policy.canonical_eligible.get(canonical, False):
        _err("unknown_team", f"{canonical!r} exists in the identity policy but is not "
             f"identity_feature_eligible (phase25_decision={policy.canonical_decision.get(canonical)!r}) - "
             f"not application-resolvable for prediction", requested_name=name, canonical_name=canonical,
             phase25_decision=policy.canonical_decision.get(canonical))
    return canonical


# ---------------------------------------------------------------------------
# Map registry (amendment #4: three separate layers - model / state / competitive)
# ---------------------------------------------------------------------------

def _load_map_registry():
    return yaml.safe_load(MAP_REGISTRY_PATH.read_text(encoding="utf-8"))


def validate_map_name(map_name, map_registry):
    supported = {m["canonical_name"] for m in map_registry["model_supported_maps"]}
    if map_name not in supported:
        _err("unsupported_map", f"{map_name!r} is not a named category in the frozen XGB V3 preprocessing "
             f"vocabulary - not fabricated, not routed to __UNKNOWN_MAP__", requested_map=map_name,
             model_supported_maps=sorted(supported))
    return map_name


# ---------------------------------------------------------------------------
# Tier / best-of contracts
# ---------------------------------------------------------------------------

VALID_TIERS = ("tier1", "tier2", "tier3")
VALID_BEST_OF = (1, 3, 5)


def validate_tier(tier):
    if tier is None:
        return "tier1", "application_default"
    if tier not in VALID_TIERS:
        _err("invalid_tier", f"tier must be one of {VALID_TIERS} or None, got {tier!r}", requested_tier=tier)
    return tier, "user_supplied"


def validate_best_of(best_of):
    if not isinstance(best_of, int) or isinstance(best_of, bool) or best_of not in VALID_BEST_OF:
        _err("invalid_best_of", f"best_of must be a typed int in {VALID_BEST_OF}, got {best_of!r}",
             requested_best_of=best_of)
    return best_of


# ---------------------------------------------------------------------------
# Inference context
# ---------------------------------------------------------------------------

@dataclass
class InferenceContext:
    context_id: str
    classification: str
    state_cutoff: pd.Timestamp
    prediction_datetime_policy: str
    rf_context: object                    # pre_veto_series_predictor.PredictorContext
    map_state: object
    form_state: object
    roster_state: object
    modern_map_state: object
    xgb_model: object
    xgb_preprocessing: dict
    xgb_roles: dict
    identity_policy: _IdentityPolicy
    map_registry: dict


_CONTEXT_CACHE = {}
_SHARED_XGB = {}   # amendment #17: RF/XGB model+preprocessing are identical across contexts


def _registry():
    return yaml.safe_load(CONTEXTS_REGISTRY_PATH.read_text(encoding="utf-8"))


def _load_shared_xgb():
    if "model" not in _SHARED_XGB:
        model = XGBClassifier()
        model.load_model(str(ROOT / "models" / "map" / "map_xgboost_v3_final.json"))
        preprocessing = prep_xgb.load_preprocessing(DATA / "modeling" / "map_xgboost_v3_final_preprocessing.json")
        roles = load_map_v3_roles(XGB_ROLES_CONFIG_PATH)
        _verify_xgb_contract(model, preprocessing)
        _SHARED_XGB["model"], _SHARED_XGB["preprocessing"], _SHARED_XGB["roles"] = model, preprocessing, roles
    return _SHARED_XGB["model"], _SHARED_XGB["preprocessing"], _SHARED_XGB["roles"]


def _verify_xgb_contract(model, preprocessing):
    """Amendment #6: verify the XGB inference contract as strictly as RF's
    verify_model_contract - never assume correctness merely because Phase 6D
    used the model."""
    raw_names = preprocessing["original_model_feature_names"]
    trans_names = preprocessing["transformed_feature_names"]
    if len(raw_names) != 120:
        _err("missing_state_support", f"XGB V3 raw feature count drifted: expected 120, got {len(raw_names)}")
    if len(trans_names) != 131:
        _err("missing_state_support", f"XGB V3 transformed feature count drifted: expected 131, got {len(trans_names)}")
    if list(model.classes_) != [0, 1]:
        _err("missing_state_support", f"XGB V3 classes_ drifted: expected [0, 1], got {list(model.classes_)}")
    if getattr(model, "n_features_in_", None) != 131:
        _err("missing_state_support", f"XGB V3 n_features_in_ drifted: expected 131, got "
             f"{getattr(model, 'n_features_in_', None)}")
    expected_map_dummies = [f"map_name_{c}" for c in
                             ["Ancient", "Anubis", "Dust2", "Inferno", "Nuke", "Overpass", "Train", "Vertigo",
                              "__UNKNOWN_MAP__"]]
    for col in expected_map_dummies + ["bestOf_BO3", "bestOf_BO5", "tier_tier2", "tier_tier3",
                                        "tier___UNKNOWN_TIER__"]:
        if col not in trans_names:
            _err("missing_state_support", f"expected transformed column {col!r} not found in XGB V3 "
                 f"preprocessing contract")
    return True


def get_context(context_id):
    if context_id in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[context_id]

    registry = _registry()
    contexts = registry["contexts"]
    if context_id not in contexts:
        _err("unknown_context", f"unknown context_id {context_id!r} - must be one of {sorted(contexts)}",
             requested_context_id=context_id)
    entry = contexts[context_id]

    if context_id == HISTORICAL_CONTEXT_ID:
        rf_context = pvp.load_predictor_context()  # already points at pre_cologne_team_state_v1_full.json
        state_dir_map = DATA / "interim" / "pre_cologne_map_state_v1.json"
        state_dir_form = DATA / "interim" / "pre_cologne_form_state_v1.json"
        state_dir_roster = DATA / "interim" / "pre_cologne_player_roster_state_v1.json"
        state_dir_modern = DATA / "interim" / "pre_cologne_modern_map_state_v1.json"
    elif context_id == DEPLOYMENT_CONTEXT_ID:
        import joblib
        rf_model = joblib.load(pvp.MODEL_PATH)
        rf_model.n_jobs = 1  # amendment #9: preserve Phase 8D determinism for EVERY context
        rf_preprocessing = pvp.prep_rf.load_preprocessing(pvp.PREPROCESSING_PATH)
        rf_config = __import__("json").loads(pvp.CONFIG_PATH.read_text(encoding="utf-8"))
        rf_store = fe.StateStore.from_json(DATA / "features" / "series_team_state_v1_deployment_post_cologne.json")
        rf_contract = pvp.verify_model_contract(rf_model, rf_preprocessing)
        rf_context = pvp.PredictorContext(model=rf_model, preprocessing=rf_preprocessing, config=rf_config,
                                           store=rf_store, tier=pvp.FROZEN_TIER, model_contract=rf_contract)
        state_dir_map = DATA / "interim" / "map_state_v1_deployment_post_cologne.json"
        state_dir_form = DATA / "interim" / "form_state_v1_deployment_post_cologne.json"
        state_dir_roster = DATA / "interim" / "player_roster_state_v1_deployment_post_cologne.json"
        state_dir_modern = DATA / "interim" / "modern_map_state_v1_deployment_post_cologne.json"
    else:
        _err("unknown_context", f"unhandled context_id {context_id!r}", requested_context_id=context_id)

    xgb_model, xgb_preprocessing, xgb_roles = _load_shared_xgb()

    ctx = InferenceContext(
        context_id=context_id, classification=entry["classification"],
        state_cutoff=pd.Timestamp(entry["state_cutoff"]),
        prediction_datetime_policy=entry["prediction_datetime_policy"],
        rf_context=rf_context,
        map_state=mfe.MapStateStore.from_json(state_dir_map),
        form_state=tfe.TeamFormStateStore.from_json(state_dir_form),
        roster_state=prfe.PlayerRosterStateStore.from_json(state_dir_roster),
        modern_map_state=mmfe.ModernMapStateStore.from_json(state_dir_modern),
        xgb_model=xgb_model, xgb_preprocessing=xgb_preprocessing, xgb_roles=xgb_roles,
        identity_policy=_load_identity_policy(), map_registry=_load_map_registry(),
    )
    _CONTEXT_CACHE[context_id] = ctx
    return ctx


def _require_valid_context_id(context_id):
    registry = _registry()
    if context_id not in registry["contexts"]:
        _err("unknown_context", f"unknown context_id {context_id!r} - must be one of "
             f"{sorted(registry['contexts'])}", requested_context_id=context_id)


# ---------------------------------------------------------------------------
# Public: registry / listing
# ---------------------------------------------------------------------------

def list_inference_contexts():
    registry = _registry()
    return _to_jsonable([{"context_id": cid, **{k: v for k, v in entry.items() if k != "state_hashes"}}
                          for cid, entry in registry["contexts"].items()])


def get_context_metadata(context_id):
    _require_valid_context_id(context_id)
    registry = _registry()
    return _to_jsonable({"context_id": context_id, **registry["contexts"][context_id]})


def list_supported_teams(context_id):
    """Amendment #2/#15: structured metadata distinguishing identity_known from
    history_available/cold_start - a policy-known, identity-eligible team with
    no history is NOT removed from this list."""
    ctx = get_context(context_id)
    policy = ctx.identity_policy
    out = []
    for canonical, eligible in sorted(policy.canonical_eligible.items()):
        if not eligible:
            continue
        state = ctx.rf_context.store.get(canonical)
        n = len(state.history) if state is not None else 0
        out.append({
            "canonical_name": canonical, "identity_eligible": True,
            "history_available": n > 0, "history_match_count": n, "cold_start": n == 0,
        })
    return _to_jsonable(out)


def list_supported_maps(context_id):
    ctx = get_context(context_id)
    out = []
    for m in ctx.map_registry["model_supported_maps"]:
        name = m["canonical_name"]
        hist_ctx = get_context(HISTORICAL_CONTEXT_ID) if context_id != HISTORICAL_CONTEXT_ID else ctx
        historical_observed = any(name == mn for (_t, mn) in hist_ctx.map_state.states.keys())
        deployment_observed = any(name == mn for (_t, mn) in ctx.map_state.states.keys())
        out.append({
            "map_name": name, "model_supported": True,
            "historical_context_available": bool(historical_observed),
            "deployment_context_observed": bool(deployment_observed),
            "cologne_2026_competitive_pool": bool(m["cologne_2026_competitive_pool"]),
            "competitive_pool_status": "context_dependent - see cologne_2026_competitive_pool for the "
                                        "one frozen historical fact known; current/future UI selectability "
                                        "is not resolved in Phase 9B",
        })
    return _to_jsonable(out)


# ---------------------------------------------------------------------------
# Prediction-datetime policy (amendments #3/#6/#16)
# ---------------------------------------------------------------------------

def _resolve_prediction_datetime(ctx: InferenceContext, prediction_datetime):
    if ctx.context_id == HISTORICAL_CONTEXT_ID:
        if prediction_datetime is not None and pd.Timestamp(prediction_datetime) != ctx.state_cutoff:
            _err("historical_context_datetime_locked",
                 f"historical_cologne_pre_event is locked to {ctx.state_cutoff} - cannot override",
                 requested_prediction_datetime=str(prediction_datetime), state_cutoff=str(ctx.state_cutoff))
        return ctx.state_cutoff, {"state_data_through": str(ctx.state_cutoff), "state_is_live": False,
                                   "requested_prediction_datetime": str(ctx.state_cutoff),
                                   "staleness_days": 0.0, "snapshot_id": ctx.context_id}

    # deployment context
    if prediction_datetime is None:
        effective = ctx.state_cutoff
    else:
        effective = pd.Timestamp(prediction_datetime)
        if effective < ctx.state_cutoff:
            _err("prediction_datetime_before_state_contract",
                 f"prediction_datetime {effective} is before the deployment state cutoff "
                 f"{ctx.state_cutoff} - would risk future-information leakage from history the "
                 f"snapshot legitimately contains up to the cutoff", requested_prediction_datetime=str(effective),
                 state_cutoff=str(ctx.state_cutoff))
    staleness_days = (effective - ctx.state_cutoff).total_seconds() / 86400.0
    freshness = {
        "state_data_through": str(ctx.state_cutoff), "state_is_live": False,
        "requested_prediction_datetime": str(effective), "staleness_days": staleness_days,
        "snapshot_id": ctx.context_id,
    }
    if staleness_days > 0:
        freshness["mode"] = "hypothetical_future_from_stale_snapshot"
        freshness["warning"] = (f"prediction_datetime is {staleness_days:.2f} day(s) after the deployment "
                                 f"state's actual data cutoff ({ctx.state_cutoff}). No newer match history "
                                 f"exists in this snapshot - this is a hypothetical projection, not a claim "
                                 f"of fresher data.")
    return effective, freshness


# ---------------------------------------------------------------------------
# Unknown-map (RF V2) prediction
# ---------------------------------------------------------------------------

def _team_history_metadata(store, canonical_name):
    state = store.get(canonical_name)
    n = len(state.history) if state is not None else 0
    return {"identity_known": True, "available": n > 0, "cold_start": n == 0, "matches": n}


def _prepare_rf_prediction(ctx: InferenceContext, team_a, team_b, best_of, prediction_datetime, tier):
    """Amendment #5: the ONE place the RF feature vector is built. Both
    `predict_series_unknown_maps` and (in application_explanations.py)
    `explain_series_unknown_maps` call this exactly once per request and then
    consume the same prepared object - the vector is never regenerated for
    explanation. Replicates pre_veto_series_predictor.predict_series_unknown_maps's
    own 3-step pipeline (fe.build_features -> prep_rf.transform -> predict_proba)
    using the identical frozen building-block functions, so behavior is
    unchanged - it just stops going through that function as an opaque black
    box so the intermediate X can be reused."""
    best_of = validate_best_of(best_of)
    tier_value, tier_source = validate_tier(tier)
    canon_a = resolve_team(team_a, ctx.identity_policy)
    canon_b = resolve_team(team_b, ctx.identity_policy)
    if canon_a == canon_b:
        _err("same_team", f"team_a and team_b resolve to the same team ({canon_a!r})",
             team_a=team_a, team_b=team_b, canonical=canon_a)
    effective_dt, freshness = _resolve_prediction_datetime(ctx, prediction_datetime)

    raw = fe.build_features(ctx.rf_context.store, canon_a, canon_b, effective_dt, best_of, tier=tier_value)
    df = pd.DataFrame([{k: raw[k] for k in ctx.rf_context.preprocessing["original_model_feature_names"]}])
    X, names = prep_rf.transform(df, ctx.rf_context.preprocessing)
    if names != ctx.rf_context.preprocessing["transformed_feature_names"]:
        _err("missing_state_support", "transformed RF feature order drifted from the frozen preprocessing contract")

    return {
        "canon_a": canon_a, "canon_b": canon_b, "best_of": best_of, "tier_value": tier_value,
        "tier_source": tier_source, "effective_dt": effective_dt, "freshness": freshness,
        "raw": raw, "X": X, "transformed_names": names,
    }


def _predict_rf_from_prepared(ctx: InferenceContext, context_id, prepared):
    proba = ctx.rf_context.model.predict_proba(prepared["X"])[:, 1]
    p_a = float(proba[0])
    canon_a, canon_b = prepared["canon_a"], prepared["canon_b"]
    favored_team, is_tied = (None, True) if p_a == 0.5 else ((canon_a if p_a > 0.5 else canon_b), False)

    out = {
        "prediction_mode": "pre_veto", "team_a": canon_a, "team_b": canon_b,
        "best_of": prepared["best_of"], "tier": prepared["tier_value"], "tier_source": prepared["tier_source"],
        "probability_team_a": p_a, "probability_team_b": 1.0 - p_a,
        "favored_team": favored_team, "prediction_is_tied": is_tied,
        "model_id": pvp.MODEL_ID, "context_id": context_id, "state_cutoff": str(ctx.state_cutoff),
        "prediction_datetime": str(prepared["effective_dt"]), "data_freshness": prepared["freshness"],
        "team_a_history": _team_history_metadata(ctx.rf_context.store, canon_a),
        "team_b_history": _team_history_metadata(ctx.rf_context.store, canon_b),
    }
    return _to_jsonable(out)


def predict_series_unknown_maps(context_id, team_a, team_b, best_of, prediction_datetime=None, tier=None):
    ctx = get_context(context_id)
    prepared = _prepare_rf_prediction(ctx, team_a, team_b, best_of, prediction_datetime, tier)
    return _predict_rf_from_prepared(ctx, context_id, prepared)


# ---------------------------------------------------------------------------
# Known-map (XGB V3) prediction
# ---------------------------------------------------------------------------

def _map_state_support_metadata(ctx: InferenceContext, canon_a, canon_b, raw_features):
    """Amendment #10: surfaces the composer's OWN existing confidence flags -
    never invents new fallback semantics."""
    return {
        "overall_state_available": bool(raw_features.get("both_teams_have_history")),
        "map_state_available": bool(raw_features.get("both_teams_have_map_history")),
        "form_state_available": bool(raw_features.get("both_teams_have_5_adjusted_matches")),
        "player_roster_state_available": bool(raw_features.get("both_teams_have_5_inferred_players")),
        "modern_map_state_available": bool(raw_features.get("both_teams_have_map_pool_history")),
        "player_map_state_available": bool(raw_features.get("both_teams_have_recent_selected_map_history")),
        "fallbacks_used": [k for k in (
            "both_teams_have_history", "both_teams_have_map_history", "both_teams_have_5_adjusted_matches",
            "both_teams_have_5_inferred_players", "both_teams_have_map_pool_history",
            "both_teams_have_recent_selected_map_history",
        ) if not raw_features.get(k)],
    }


def _prepare_xgb_prediction(ctx: InferenceContext, team_a, team_b, map_name, best_of, prediction_datetime, tier):
    """Amendment #5: the ONE place the XGB feature vector is built - both
    `predict_map` and `explain_map` consume the same prepared object."""
    best_of = validate_best_of(best_of)
    tier_value, tier_source = validate_tier(tier)
    map_name = validate_map_name(map_name, ctx.map_registry)
    canon_a = resolve_team(team_a, ctx.identity_policy)
    canon_b = resolve_team(team_b, ctx.identity_policy)
    if canon_a == canon_b:
        _err("same_team", f"team_a and team_b resolve to the same team ({canon_a!r})",
             team_a=team_a, team_b=team_b, canonical=canon_a)
    effective_dt, freshness = _resolve_prediction_datetime(ctx, prediction_datetime)

    raw = rmmc.build_future_modern_rich_map_features(
        canon_a, canon_b, best_of, map_name, effective_dt,
        ctx.rf_context.store, ctx.map_state, ctx.form_state, ctx.roster_state, ctx.modern_map_state,
        tier=tier_value)
    df = pd.DataFrame([{k: raw[k] for k in ctx.xgb_preprocessing["original_model_feature_names"]}])
    X, names = prep_xgb.transform(df, ctx.xgb_preprocessing, ctx.xgb_roles)
    if names != ctx.xgb_preprocessing["transformed_feature_names"]:
        _err("missing_state_support", "transformed feature order drifted from the frozen XGB V3 "
             "preprocessing contract")

    return {
        "canon_a": canon_a, "canon_b": canon_b, "map_name": map_name, "best_of": best_of,
        "tier_value": tier_value, "tier_source": tier_source, "effective_dt": effective_dt,
        "freshness": freshness, "raw": raw, "X": X, "transformed_names": names,
    }


def _predict_xgb_from_prepared(ctx: InferenceContext, context_id, prepared):
    proba = ctx.xgb_model.predict_proba(prepared["X"])[:, 1]
    p_a = float(proba[0])
    canon_a, canon_b = prepared["canon_a"], prepared["canon_b"]
    favored_team, is_tied = (None, True) if p_a == 0.5 else ((canon_a if p_a > 0.5 else canon_b), False)

    out = {
        "prediction_mode": "known_map", "team_a": canon_a, "team_b": canon_b, "map_name": prepared["map_name"],
        "best_of": prepared["best_of"], "tier": prepared["tier_value"], "tier_source": prepared["tier_source"],
        "probability_team_a": p_a, "probability_team_b": 1.0 - p_a,
        "favored_team": favored_team, "prediction_is_tied": is_tied,
        "model_id": "map_xgboost_v3_final", "context_id": context_id, "state_cutoff": str(ctx.state_cutoff),
        "prediction_datetime": str(prepared["effective_dt"]), "data_freshness": prepared["freshness"],
        "state_support": _map_state_support_metadata(ctx, canon_a, canon_b, prepared["raw"]),
    }
    return _to_jsonable(out)


def predict_map(context_id, team_a, team_b, map_name, best_of, prediction_datetime=None, tier=None):
    ctx = get_context(context_id)
    prepared = _prepare_xgb_prediction(ctx, team_a, team_b, map_name, best_of, prediction_datetime, tier)
    return _predict_xgb_from_prepared(ctx, context_id, prepared)


# ---------------------------------------------------------------------------
# Generic best-of-N series composition DP
# ---------------------------------------------------------------------------

def compose_series_probability(map_probabilities, best_of):
    """P(Team A wins the series) via one generic DP over (maps_played, wins_a),
    terminating the instant either side reaches ceil(best_of/2) wins. No
    hardcoded BO1/3/5 formulas. Validates its own inputs independently - never
    relies solely on a caller to have validated first (amendment #13)."""
    if best_of not in VALID_BEST_OF:
        _err("invalid_best_of", f"best_of must be one of {VALID_BEST_OF}, got {best_of!r}",
             requested_best_of=best_of)
    if len(map_probabilities) != best_of:
        _err("invalid_map_count", f"compose_series_probability requires exactly {best_of} map "
             f"probabilities for best_of={best_of}, got {len(map_probabilities)}",
             requested_best_of=best_of, n_probabilities=len(map_probabilities))
    for i, p in enumerate(map_probabilities):
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not (0.0 <= p <= 1.0):
            _err("invalid_probability", f"map_probabilities[{i}]={p!r} is not a finite value in [0, 1]",
                 index=i, value=p)

    wins_needed = (best_of + 1) // 2
    # dp[(maps_played, wins_a)] = P(reach this state with series still alive OR just won)
    dp = {(0, 0): 1.0}
    p_series_a = 0.0
    for map_idx in range(best_of):
        p = map_probabilities[map_idx]
        next_dp = {}
        for (played, wins_a), prob in dp.items():
            if played != map_idx:
                continue
            losses_a = played - wins_a
            wins_b = losses_a
            if wins_a >= wins_needed or wins_b >= wins_needed:
                continue  # series already decided - should not happen given pruning below
            # team A wins this map
            new_wins_a = wins_a + 1
            p_branch = prob * p
            if new_wins_a >= wins_needed:
                p_series_a += p_branch
            else:
                next_dp[(played + 1, new_wins_a)] = next_dp.get((played + 1, new_wins_a), 0.0) + p_branch
            # team A loses this map
            p_branch = prob * (1.0 - p)
            new_wins_b = (played - wins_a) + 1
            if new_wins_b >= wins_needed:
                pass  # team B has won the series - contributes 0 to p_series_a
            else:
                next_dp[(played + 1, wins_a)] = next_dp.get((played + 1, wins_a), 0.0) + p_branch
        dp = next_dp
    return p_series_a


# ---------------------------------------------------------------------------
# Known-map series composition
# ---------------------------------------------------------------------------

_MAP_COUNT_FOR_BO = {1: 1, 3: 3, 5: 5}


def predict_series_known_maps(context_id, team_a, team_b, best_of, ordered_maps, prediction_datetime=None,
                               tier=None):
    ctx = get_context(context_id)
    best_of = validate_best_of(best_of)
    canon_a = resolve_team(team_a, ctx.identity_policy)
    canon_b = resolve_team(team_b, ctx.identity_policy)
    if canon_a == canon_b:
        _err("same_team", f"team_a and team_b resolve to the same team ({canon_a!r})",
             team_a=team_a, team_b=team_b, canonical=canon_a)
    expected_n = _MAP_COUNT_FOR_BO[best_of]
    if len(ordered_maps) != expected_n:
        _err("invalid_map_count", f"best_of={best_of} requires exactly {expected_n} ordered_maps, "
             f"got {len(ordered_maps)}", best_of=best_of, expected=expected_n, got=len(ordered_maps))
    if len(set(ordered_maps)) != len(ordered_maps):
        _err("duplicate_map", f"ordered_maps contains a duplicate: {ordered_maps}", ordered_maps=ordered_maps)

    map_results = []
    for i, map_name in enumerate(ordered_maps, start=1):
        r = predict_map(context_id, canon_a, canon_b, map_name, best_of, prediction_datetime=prediction_datetime,
                         tier=tier)
        map_results.append({"map_number": i, "map_name": map_name,
                             "probability_team_a": r["probability_team_a"],
                             "probability_team_b": r["probability_team_b"]})

    series_p_a = compose_series_probability([m["probability_team_a"] for m in map_results], best_of)
    favored_team, is_tied = (None, True) if series_p_a == 0.5 else \
        ((canon_a if series_p_a > 0.5 else canon_b), False)

    out = {
        "prediction_mode": "known_maps", "team_a": canon_a, "team_b": canon_b, "best_of": best_of,
        "ordered_maps": map_results,
        "series_probability_team_a": series_p_a, "series_probability_team_b": 1.0 - series_p_a,
        "favored_team": favored_team, "prediction_is_tied": is_tied,
        "model_id": "map_xgboost_v3_final", "composition_method": "ordered_map_dynamic_program",
        "context_id": context_id, "state_cutoff": str(ctx.state_cutoff),
    }
    return _to_jsonable(out)
