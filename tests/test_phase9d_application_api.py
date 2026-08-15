"""
Phase 9D tests: the versioned application prediction API. A thin read-only
HTTP transport over the frozen Phase 9B/9C cores - these tests exist to
prove the HTTP layer performs ZERO transformation of prediction or
explanation values, maps every ApplicationInferenceError to the frozen HTTP
status policy, never mutates state, and never accepts a file-path input.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from fastapi.testclient import TestClient

from _common import ROOT
import application_inference as ai
import application_explanations as ae
import application_api as api

CONFIG = ROOT / "config"
DEPLOY = "deployment_post_cologne_v1"
HIST = "historical_cologne_pre_event"


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def fixtures():
    return yaml.safe_load((CONFIG / "application_api_fixtures_v1.yaml").read_text(encoding="utf-8"))


# --- 1. health / readiness ---

def test_health_live_always_200(client):
    r = client.get("/api/v1/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "live"


def test_health_ready_200_after_startup(client):
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["default_context_id"] == DEPLOY


def test_readiness_unavailable_blocks_prediction_endpoints(client, monkeypatch):
    monkeypatch.setitem(api._STARTUP_STATE, "ready", False)
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                      "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                      "best_of": 3})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "service_unavailable"
    monkeypatch.setitem(api._STARTUP_STATE, "ready", True)


# --- 2. metadata ---

def test_meta_shape_and_defaults(client):
    r = client.get("/api/v1/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["default_context_id"] == DEPLOY
    assert body["state_is_live"] is False
    assert body["api_version"] == api.API_VERSION
    assert body["prediction_contract"] == "phase9b"
    assert body["explanation_version"] == "application_explanations_v1"
    assert set(body["available_context_ids"]) == {DEPLOY, HIST}
    assert body["deployment_state_data_through"] == "2026-06-28T20:00:00"


def test_default_context_is_never_historical():
    assert api.DEFAULT_CONTEXT_ID == ai.DEPLOYMENT_CONTEXT_ID != ai.HISTORICAL_CONTEXT_ID


# --- 3. context discovery ---

def test_list_contexts_matches_core(client):
    r = client.get("/api/v1/contexts")
    assert r.status_code == 200
    ids = {c["context_id"] for c in r.json()["contexts"]}
    assert ids == {DEPLOY, HIST}


def test_get_context_by_id(client):
    r = client.get(f"/api/v1/contexts/{DEPLOY}")
    assert r.status_code == 200
    assert r.json()["state_cutoff"] == "2026-06-28T20:00:00"


def test_get_unknown_context_is_404(client):
    r = client.get("/api/v1/contexts/not_a_real_context")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_context"


# --- 4. team / map discovery ---

def test_teams_substring_search_case_insensitive(client):
    r = client.get("/api/v1/teams", params={"context_id": DEPLOY, "q": "vitality"})
    assert r.status_code == 200
    names = [t["canonical_name"] for t in r.json()["teams"]]
    assert "Team Vitality" in names
    r2 = client.get("/api/v1/teams", params={"context_id": DEPLOY, "q": "VITALITY"})
    assert r2.json()["teams"] == r.json()["teams"]


def test_teams_search_never_alters_strict_resolution_semantics(client):
    """A fuzzy/substring hit in discovery must NOT make prediction accept a
    non-exact name - resolve_team stays exact-match-only."""
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                      "team_a": "Vitality", "team_b": "Team Falcons", "best_of": 3})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_team"


def test_teams_limit_respected(client):
    r = client.get("/api/v1/teams", params={"context_id": DEPLOY, "limit": 3})
    assert r.status_code == 200
    assert len(r.json()["teams"]) <= 3


def test_maps_cache_not_model_supported(client):
    r = client.get("/api/v1/maps", params={"context_id": DEPLOY})
    assert r.status_code == 200
    maps = {m["map_name"]: m for m in r.json()["maps"]}
    assert "Cache" not in maps
    assert len(maps) == 9


def test_maps_never_presented_as_active_duty_pool(client):
    r = client.get("/api/v1/maps", params={"context_id": DEPLOY})
    for m in r.json()["maps"]:
        assert "competitive_pool_status" in m


# --- 5. request schema strict typing ---

def test_best_of_bool_rejected():
    with pytest.raises(Exception):
        api.SeriesPredictionRequest(context_id=DEPLOY, mode="pre_veto", team_a="A", team_b="B", best_of=True)


def test_best_of_bool_rejected_over_http(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": True})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "schema_validation_error"


def test_best_of_string_rejected_over_http(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": "3"})
    assert r.status_code == 422


def test_mode_invalid_literal_rejected(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "blend", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3})
    assert r.status_code == 422


def test_extra_field_rejected(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3, "unexpected_field": 1})
    assert r.status_code == 422


def test_pre_veto_rejects_ordered_maps(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "ordered_maps": ["Mirage", "Inferno", "Nuke"]})
    assert r.status_code == 422


def test_known_maps_requires_ordered_maps(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "known_maps", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3})
    assert r.status_code == 422


# --- 6. prediction modes: pre_veto ---

def test_pre_veto_only_uses_rf(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3, "include_explanation": True})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["model_id"] == "series_random_forest_v2"
    assert body["explanation"]["attribution_method"] == "saabas_path_decomposition"


def test_pre_veto_bo1_bo3_bo5_all_succeed(client):
    for bo in (1, 3, 5):
        r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": bo, "include_explanation": False})
        assert r.status_code == 200, (bo, r.json())


def test_pre_veto_historical_cold_start(client):
    r = client.post("/api/v1/predict/series", json={"context_id": HIST, "mode": "pre_veto",
                                                      "team_a": "THUNDERdOWNUNDER", "team_b": "MOUZ", "best_of": 3,
                                                      "include_explanation": False})
    assert r.status_code == 200
    assert r.json()["prediction"]["team_a_history"]["cold_start"] is True


# --- 7. prediction modes: known_maps / predict-map ---

def test_known_maps_never_blends_rf(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "known_maps", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "ordered_maps": ["Mirage", "Inferno", "Nuke"],
                                                      "include_explanation": False})
    assert r.status_code == 200
    assert r.json()["prediction"]["model_id"] == "map_xgboost_v3_final"
    assert r.json()["prediction"]["composition_method"] == "ordered_map_dynamic_program"


def test_known_maps_wrong_count_rejected(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "known_maps", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "ordered_maps": ["Mirage", "Inferno"]})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_map_count"


def test_known_maps_duplicate_map_rejected(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "known_maps", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "ordered_maps": ["Mirage", "Mirage", "Nuke"]})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "duplicate_map"


def test_predict_map_unsupported_cache_rejected(client):
    r = client.post("/api/v1/predict/map", json={"context_id": DEPLOY, "team_a": "Team Vitality",
                                                   "team_b": "Team Falcons", "map_name": "Cache", "best_of": 3})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_map"


def test_predict_map_partial_state_surfaces_fallbacks(client):
    r = client.post("/api/v1/predict/map", json={"context_id": DEPLOY, "team_a": "THUNDERdOWNUNDER",
                                                   "team_b": "MOUZ", "map_name": "Mirage", "best_of": 3,
                                                   "include_explanation": True})
    assert r.status_code == 200
    assert "fallbacks_used" in r.json()["explanation"]["state_support"]


# --- 8. team resolution errors ---

def test_unknown_team_is_404(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                      "team_a": "Not A Real Team", "team_b": "Team Vitality",
                                                      "best_of": 3})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_team"


def test_same_team_is_422(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                      "team_a": "Team Vitality", "team_b": "Team Vitality",
                                                      "best_of": 3})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "same_team"


def test_ambiguous_team_maps_to_409(client, monkeypatch):
    policy = ai._IdentityPolicy(name_to_canonical={"X": {"Team One", "Team Two"}},
                                 canonical_eligible={"Team One": True, "Team Two": True}, canonical_decision={})
    monkeypatch.setattr(ai, "resolve_team", lambda name, p: ai._err("ambiguous_team", "ambiguous", requested_name=name))
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "X",
                                                      "team_b": "Team Vitality", "best_of": 3})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "ambiguous_team"


# --- 9. datetime lock / rejection / stale-future ---

def test_historical_context_datetime_override_rejected(client):
    r = client.post("/api/v1/predict/series", json={"context_id": HIST, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "prediction_datetime": "2020-01-01T00:00:00"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "historical_context_datetime_locked"


def test_historical_context_datetime_equal_to_cutoff_accepted(client):
    r = client.post("/api/v1/predict/series", json={"context_id": HIST, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "prediction_datetime": "2026-06-02T13:30:00",
                                                      "include_explanation": False})
    assert r.status_code == 200


def test_deployment_datetime_before_cutoff_rejected(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "prediction_datetime": "2020-01-01T00:00:00"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "prediction_datetime_before_state_contract"


def test_deployment_datetime_after_cutoff_is_hypothetical_stale(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "prediction_datetime": "2026-07-15T00:00:00",
                                                      "include_explanation": False})
    assert r.status_code == 200
    freshness = r.json()["prediction"]["data_freshness"]
    assert freshness["mode"] == "hypothetical_future_from_stale_snapshot"
    assert freshness["state_is_live"] is False


# --- 10. structured error envelope / HTTP status mapping ---

def test_error_envelope_shape(client):
    r = client.post("/api/v1/predict/series", json={"context_id": "nope", "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3})
    body = r.json()
    assert set(body.keys()) == {"error", "request_id"}
    assert set(body["error"].keys()) == {"code", "message", "detail"}


@pytest.mark.parametrize("error_code,status", list(api.ERROR_STATUS_MAP.items()))
def test_error_status_map_is_exactly_the_frozen_policy(error_code, status):
    # The original 13 Phase 9D codes below are UNCHANGED (Phase 9E only adds
    # new tournament-specific codes additively - see application_api.py's
    # ERROR_STATUS_MAP and Phase 9E amendment #25 "Phase 9D routes must
    # remain byte/semantically stable").
    expected = {
        "unknown_context": 404, "unknown_team": 404, "invalid_best_of": 422, "invalid_tier": 422,
        "same_team": 422, "unsupported_map": 422, "invalid_map_count": 422, "duplicate_map": 422,
        "invalid_probability": 422, "historical_context_datetime_locked": 422,
        "prediction_datetime_before_state_contract": 422, "ambiguous_team": 409, "missing_state_support": 500,
        # Phase 9E additive tournament codes:
        "unknown_ruleset": 404, "invalid_participant_count": 422, "duplicate_team": 422, "invalid_seed": 422,
        "missing_seed": 422, "invalid_override": 422, "override_team_mismatch": 422, "duplicate_override": 422,
        "contradictory_override": 422, "invalid_simulation_count": 422, "probability_matrix_incomplete": 500,
    }
    assert expected[error_code] == status


def test_missing_state_support_maps_to_500(client, monkeypatch):
    def boom(*a, **k):
        raise ai.ApplicationInferenceError("missing_state_support", "simulated internal contract drift")
    monkeypatch.setattr(ai, "predict_series_unknown_maps", boom)
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3, "include_explanation": False})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "missing_state_support"


def test_genuinely_unexpected_exception_is_500_no_traceback(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom - should never leak to the client")
    monkeypatch.setattr(ai, "predict_series_unknown_maps", boom)
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3, "include_explanation": False})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "boom" not in json.dumps(r.json())
    assert "Traceback" not in json.dumps(r.json())


# --- 11. request IDs ---

def test_request_id_present_in_body_and_header(client):
    r = client.get("/api/v1/health/live")
    r2 = client.get("/api/v1/meta")
    assert "x-request-id" in r2.headers
    assert r2.json()["request_id"] == r2.headers["x-request-id"]


def test_request_ids_are_unique(client):
    ids = {client.get("/api/v1/meta").json()["request_id"] for _ in range(5)}
    assert len(ids) == 5


# --- 12. explanation detail: summary vs full ---

def test_summary_omits_low_level_feature_arrays(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "explanation_detail": "summary"})
    exp = r.json()["explanation"]
    assert "feature_contributions" not in exp
    for g in exp["grouped_factors"]:
        assert "supporting_features" not in g
    assert exp["detail_level"] == "summary"


def test_full_matches_direct_core_explanation_exactly(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3,
                                                      "explanation_detail": "full"})
    direct = ae.explain_series_unknown_maps(DEPLOY, "Team Vitality", "Team Falcons", 3)
    api_exp = dict(r.json()["explanation"])
    api_exp.pop("detail_level")
    assert api_exp == direct["explanation"]


def test_summary_is_deterministic_projection_of_full(client):
    r_full = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                           "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                           "best_of": 3, "explanation_detail": "full"})
    r_summary = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                              "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                              "best_of": 3, "explanation_detail": "summary"})
    full = r_full.json()["explanation"]
    projected = api._summarize_explanation_block(dict(full, detail_level=None))
    assert projected["grouped_factors"] == r_summary.json()["explanation"]["grouped_factors"]
    assert projected["human_readable_summary"] == r_summary.json()["explanation"]["human_readable_summary"]


def test_include_explanation_false_omits_explanation(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3, "include_explanation": False})
    assert r.json()["explanation"] is None


# --- 13. hard gate: prediction parity (HTTP vs direct core, exact match) ---

@pytest.mark.parametrize("context_id", [HIST, DEPLOY])
@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_pre_veto_prediction_parity(client, context_id, best_of):
    prediction_datetime = "2026-06-02T13:30:00" if context_id == HIST else None
    payload = {"context_id": context_id, "mode": "pre_veto", "team_a": "Team Vitality", "team_b": "Team Falcons",
               "best_of": best_of, "include_explanation": False}
    if prediction_datetime:
        payload["prediction_datetime"] = prediction_datetime
    r = client.post("/api/v1/predict/series", json=payload)
    direct = ai.predict_series_unknown_maps(context_id, "Team Vitality", "Team Falcons", best_of,
                                             prediction_datetime=prediction_datetime)
    assert r.json()["prediction"]["probability_team_a"] == direct["probability_team_a"]


def test_pre_veto_cold_start_team_prediction_parity(client):
    r = client.post("/api/v1/predict/series", json={"context_id": HIST, "mode": "pre_veto",
                                                      "team_a": "THUNDERdOWNUNDER", "team_b": "MOUZ", "best_of": 3,
                                                      "include_explanation": False})
    direct = ai.predict_series_unknown_maps(HIST, "THUNDERdOWNUNDER", "MOUZ", 3)
    assert r.json()["prediction"]["probability_team_a"] == direct["probability_team_a"]


@pytest.mark.parametrize("context_id", [HIST, DEPLOY])
def test_known_map_prediction_parity_all_9_maps(client, context_id):
    ctx = ai.get_context(context_id)
    n_ok = 0
    maps = [m["canonical_name"] for m in ctx.map_registry["model_supported_maps"]]
    for map_name in maps:
        prediction_datetime = "2026-06-02T13:30:00" if context_id == HIST else None
        payload = {"context_id": context_id, "team_a": "Team Vitality", "team_b": "Team Falcons",
                   "map_name": map_name, "best_of": 3, "include_explanation": False}
        if prediction_datetime:
            payload["prediction_datetime"] = prediction_datetime
        r = client.post("/api/v1/predict/map", json=payload)
        direct = ai.predict_map(context_id, "Team Vitality", "Team Falcons", map_name, 3,
                                 prediction_datetime=prediction_datetime)
        n_ok += int(r.json()["prediction"]["probability_team_a"] == direct["probability_team_a"])
    assert n_ok == len(maps) == 9


@pytest.mark.parametrize("best_of,maps", [(1, ["Mirage"]), (3, ["Mirage", "Inferno", "Nuke"]),
                                           (5, ["Mirage", "Inferno", "Nuke", "Ancient", "Overpass"])])
def test_known_series_prediction_parity(client, best_of, maps):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "known_maps",
                                                      "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                      "best_of": best_of, "ordered_maps": maps,
                                                      "include_explanation": False})
    direct = ai.predict_series_known_maps(DEPLOY, "Team Vitality", "Team Falcons", best_of, maps)
    assert r.json()["prediction"] == direct


# --- 14. hard gate: explanation parity (HTTP full vs direct core, exact match) ---

def test_known_map_explanation_full_parity(client):
    r = client.post("/api/v1/predict/map", json={"context_id": DEPLOY, "team_a": "Team Vitality",
                                                   "team_b": "Team Falcons", "map_name": "Mirage", "best_of": 3,
                                                   "explanation_detail": "full"})
    direct = ae.explain_map(DEPLOY, "Team Vitality", "Team Falcons", "Mirage", 3)
    api_exp = dict(r.json()["explanation"])
    api_exp.pop("detail_level")
    api_exp.pop("state_support")
    assert api_exp == direct["explanation"]
    assert r.json()["explanation"]["state_support"] == direct["state_support"]


def test_known_series_explanation_leverage_reach_parity(client):
    maps = ["Mirage", "Inferno", "Nuke"]
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "known_maps",
                                                      "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                      "best_of": 3, "ordered_maps": maps,
                                                      "explanation_detail": "full"})
    direct = ae.explain_series_known_maps(DEPLOY, "Team Vitality", "Team Falcons", 3, maps)
    assert r.json()["explanation"]["series_composition"] == direct["series_composition"]
    for i, me in enumerate(r.json()["explanation"]["map_level_explanations"]):
        assert me["probability_team_a"] == direct["map_level_explanations"][i]["prediction"]["probability_team_a"]


# --- 15. JSON serialization / OpenAPI ---

def test_predictions_are_json_serializable(client):
    r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto", "team_a": "Team Vitality",
                                                      "team_b": "Team Falcons", "best_of": 3})
    json.dumps(r.json())  # raises on failure


def test_openapi_loads_and_covers_endpoints(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec["paths"]
    for p in ("/api/v1/health/live", "/api/v1/health/ready", "/api/v1/meta", "/api/v1/contexts",
              "/api/v1/contexts/{context_id}", "/api/v1/teams", "/api/v1/maps", "/api/v1/predict/series",
              "/api/v1/predict/map"):
        assert p in paths, p


def test_openapi_no_internal_filesystem_paths_leaked(client):
    spec = client.get("/openapi.json").json()
    text = json.dumps(spec)
    assert "C:\\" not in text and "/scripts/" not in text and str(ROOT) not in text


# --- 16. CORS ---

def test_cors_preflight_allows_configured_origin(client):
    r = client.options("/api/v1/health/live", headers={"Origin": "http://localhost:3000",
                                                         "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_config_is_not_wildcard_with_credentials():
    cfg = yaml.safe_load((CONFIG / "application_api_v1.yaml").read_text(encoding="utf-8"))["cors"]
    assert not (cfg["allow_origins"] == ["*"] and cfg["allow_credentials"] is True)


# --- 17. concurrency / state safety ---

def test_concurrent_requests_are_deterministic_and_isolated(client):
    direct = ai.predict_series_unknown_maps(DEPLOY, "Team Vitality", "Team Falcons", 3)["probability_team_a"]
    ctx_before = ai._CONTEXT_CACHE[DEPLOY]

    def call(i):
        r = client.post("/api/v1/predict/series", json={"context_id": DEPLOY, "mode": "pre_veto",
                                                          "team_a": "Team Vitality", "team_b": "Team Falcons",
                                                          "best_of": 3, "include_explanation": bool(i % 2)})
        return r.json()["prediction"]["probability_team_a"]

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(call, range(40)))
    assert all(v == direct for v in results)
    assert ai._CONTEXT_CACHE[DEPLOY] is ctx_before  # no reload per request


def test_default_context_warmed_at_startup_not_first_request():
    assert DEPLOY in ai._CONTEXT_CACHE


# --- 18. zero state mutation / no path input / no write operations ---

def test_application_api_never_writes_files():
    src = (ROOT / "scripts" / "application_api.py").read_text(encoding="utf-8")
    for forbidden in ("open(", ".to_parquet(", ".to_csv(", ".write_text(", ".write_bytes(", "os.remove", "shutil."):
        assert forbidden not in src, forbidden


def test_request_schemas_have_no_path_style_fields():
    for model in (api.SeriesPredictionRequest, api.MapPredictionRequest):
        for name in model.model_fields:
            assert "path" not in name.lower() and "file" not in name.lower()


def test_no_state_hash_changes_after_a_full_request_battery(client, fixtures):
    import build_application_registries as bar
    before = bar.hash_group(bar.DEPLOYMENT_STATE)
    for fx in fixtures["fixtures"]["pre_veto"][:2]:
        client.post("/api/v1/predict/series", json={"context_id": fx["context_id"], "mode": "pre_veto",
                                                      "team_a": fx["team_a"], "team_b": fx["team_b"],
                                                      "best_of": fx["best_of"], "include_explanation": False})
    after = bar.hash_group(bar.DEPLOYMENT_STATE)
    assert before == after


# --- 19. fixture manifest replay ---

def test_declared_pre_veto_fixtures_all_succeed(client, fixtures):
    for fx in fixtures["fixtures"]["pre_veto"]:
        payload = {"context_id": fx["context_id"], "mode": "pre_veto", "team_a": fx["team_a"], "team_b": fx["team_b"],
                   "best_of": fx["best_of"], "include_explanation": False}
        if fx["context_id"] == HIST:
            payload["prediction_datetime"] = "2026-06-02T13:30:00"
        r = client.post("/api/v1/predict/series", json=payload)
        assert r.status_code == 200, (fx["id"], r.json())


def test_declared_known_map_fixtures_all_succeed(client, fixtures):
    for fx in fixtures["fixtures"]["known_map"]:
        r = client.post("/api/v1/predict/map", json={"context_id": fx["context_id"], "team_a": fx["team_a"],
                                                       "team_b": fx["team_b"], "map_name": fx["map_name"],
                                                       "best_of": fx["best_of"], "include_explanation": True})
        assert r.status_code == 200, (fx["id"], r.json())


def test_declared_known_series_fixtures_all_succeed(client, fixtures):
    for fx in fixtures["fixtures"]["known_series"]:
        r = client.post("/api/v1/predict/series", json={"context_id": fx["context_id"], "mode": "known_maps",
                                                          "team_a": fx["team_a"], "team_b": fx["team_b"],
                                                          "best_of": fx["best_of"], "ordered_maps": fx["ordered_maps"],
                                                          "include_explanation": True})
        assert r.status_code == 200, (fx["id"], r.json())


def test_declared_error_fixtures_produce_expected_code_and_status(client, fixtures):
    for fx in fixtures["fixtures"]["expected_errors"]:
        req = dict(fx["request"])
        endpoint = "/api/v1/predict/map" if "map_name" in req else "/api/v1/predict/series"
        r = client.post(endpoint, json=req)
        assert r.status_code == fx["expected_http_status"], (fx["id"], r.status_code, r.json())
        assert r.json()["error"]["code"] == fx["expected_error_code"], (fx["id"], r.json())
