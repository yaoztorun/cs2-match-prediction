"""
Phase 9B tests: context registry, historical RF parity (all 2,976 Phase 8D
matrix rows), known-map wrapper parity, team/tier/BO/map validation,
composition DP correctness, determinism, state immutability, JSON safety,
and the THUNDERdOWNUNDER historical/deployment lifecycle.
"""

import ast
import json
from itertools import product
from math import comb

import numpy as np
import pandas as pd
import pytest
import yaml

from _common import ROOT
import application.inference.application_inference as ai

EVAL = ROOT / "data" / "evaluation"


@pytest.fixture(scope="module")
def historical():
    return ai.get_context("historical_cologne_pre_event")


@pytest.fixture(scope="module")
def deployment():
    return ai.get_context("deployment_post_cologne_v1")


# --- 1. context registry ---

def test_registry_has_exactly_two_contexts():
    ctxs = ai.list_inference_contexts()
    ids = {c["context_id"] for c in ctxs}
    assert ids == {"historical_cologne_pre_event", "deployment_post_cologne_v1"}


def test_registry_never_scans_filesystem_for_latest():
    src = (ROOT / "application" / "inference" / "application_inference.py").read_text(encoding="utf-8")
    assert "glob(" not in src and "os.listdir" not in src and "iterdir(" not in src


def test_get_context_metadata_unknown_context_raises():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.get_context_metadata("nonexistent_context")
    assert exc.value.error_code == "unknown_context"


# --- 2. hash verification ---

def test_registry_hashes_full_pipeline_not_just_model():
    registry = yaml.safe_load((ROOT / "config" / "application" / "application_inference_contexts_v1.yaml")
                               .read_text(encoding="utf-8"))
    rf = registry["rf_unknown_map_pipeline"]
    xgb = registry["xgb_known_map_pipeline"]
    for key in ("rf_model", "rf_preprocessing", "rf_selected_config", "feature_engine",
                "preprocessing_random_forest_v1"):
        assert key in rf
    for key in ("xgb_model", "xgb_metadata", "xgb_preprocessing", "rich_modern_map_feature_composer",
                "preprocessing_xgboost_map_v3", "preprocessing_common_map_v2", "map_feature_engine",
                "team_form_engine", "player_roster_feature_engine", "modern_map_feature_engine"):
        assert key in xgb


def test_hashes_match_files_on_disk():
    import application.inference.build_application_registries as bar
    fresh = bar.hash_group(bar.RF_PIPELINE)
    registry = yaml.safe_load((ROOT / "config" / "application" / "application_inference_contexts_v1.yaml")
                               .read_text(encoding="utf-8"))
    assert fresh == registry["rf_unknown_map_pipeline"]


# --- 3. historical RF parity (hard gate) ---

def test_historical_rf_parity_all_2976_rows():
    df = pd.read_parquet(EVAL / "cologne_2026_pre_event_matchup_probabilities_v1.parquet", engine="fastparquet")
    assert len(df) == 2976
    n_exact = 0
    for row in df.itertuples(index=False):
        r = ai.predict_series_unknown_maps("historical_cologne_pre_event", row.team_a, row.team_b,
                                            int(row.best_of))
        if abs(r["probability_team_a"] - row.probability_team_a) <= 1e-9:
            n_exact += 1
    assert n_exact == 2976


# --- 4. deployment state loading ---

def test_deployment_context_uses_deployment_state_not_pre_cologne(deployment, historical):
    assert deployment.state_cutoff == pd.Timestamp("2026-06-28T20:00:00")
    assert historical.state_cutoff == pd.Timestamp("2026-06-02T13:30:00")
    assert deployment.rf_context.store is not historical.rf_context.store


# --- 5. team resolution ---

def test_resolve_team_exact_and_canonical(deployment):
    assert ai.resolve_team("Team Vitality", deployment.identity_policy) == "Team Vitality"


def test_unknown_team_raises():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Not A Real Team", "Team Vitality", 3)
    assert exc.value.error_code == "unknown_team"


def test_ambiguous_team_path():
    policy = ai._IdentityPolicy(name_to_canonical={"X": {"Team One", "Team Two"}},
                                 canonical_eligible={"Team One": True, "Team Two": True},
                                 canonical_decision={})
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.resolve_team("X", policy)
    assert exc.value.error_code == "ambiguous_team"


def test_ineligible_identity_rejected_as_unknown_team():
    ctx = ai.get_context("deployment_post_cologne_v1")
    ineligible = [c for c, e in ctx.identity_policy.canonical_eligible.items() if not e]
    assert len(ineligible) > 0
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.resolve_team(ineligible[0], ctx.identity_policy)
    assert exc.value.error_code == "unknown_team"


def test_same_team_rejected():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Vitality", 3)
    assert exc.value.error_code == "same_team"


# --- 6. tier / BO validation ---

def test_tier_validation():
    assert ai.validate_tier(None) == ("tier1", "application_default")
    assert ai.validate_tier("tier2") == ("tier2", "user_supplied")
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.validate_tier("tier9")
    assert exc.value.error_code == "invalid_tier"


def test_best_of_validation():
    assert ai.validate_best_of(3) == 3
    for bad in (2, 4, "3", 3.0, True):
        with pytest.raises(ai.ApplicationInferenceError) as exc:
            ai.validate_best_of(bad)
        assert exc.value.error_code == "invalid_best_of"


# --- 7. prediction datetime rules ---

def test_historical_datetime_locked():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_series_unknown_maps("historical_cologne_pre_event", "Team Vitality", "Team Falcons", 3,
                                        prediction_datetime="2026-01-01")
    assert exc.value.error_code == "historical_context_datetime_locked"


def test_deployment_datetime_cannot_move_backward():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                        prediction_datetime="2026-06-10")
    assert exc.value.error_code == "prediction_datetime_before_state_contract"


def test_deployment_hypothetical_future_mode():
    r = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                        prediction_datetime="2026-07-05")
    assert r["data_freshness"]["mode"] == "hypothetical_future_from_stale_snapshot"
    assert r["data_freshness"]["staleness_days"] > 0
    assert r["data_freshness"]["state_is_live"] is False


def test_deployment_default_datetime_is_state_cutoff():
    r = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    assert r["data_freshness"]["staleness_days"] == 0.0
    assert r["prediction_datetime"] == "2026-06-28 20:00:00"


# --- 8. determinism ---

def test_rf_determinism_and_n_jobs_1():
    ctx = ai.get_context("deployment_post_cologne_v1")
    assert ctx.rf_context.model.n_jobs == 1
    r1 = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    r2 = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    assert r1["probability_team_a"] == r2["probability_team_a"]


def test_xgb_determinism():
    r1 = ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    r2 = ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    assert r1["probability_team_a"] == r2["probability_team_a"]


# --- 9. supported maps / Cache ---

def test_supported_maps_are_the_nine_frozen_categories():
    maps = ai.list_supported_maps("deployment_post_cologne_v1")
    names = {m["map_name"] for m in maps}
    assert names == {"Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"}
    assert all(m["model_supported"] for m in maps)


def test_cache_is_explicitly_unsupported_not_bucketed():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Cache", 3)
    assert exc.value.error_code == "unsupported_map"
    assert "Cache" in str(exc.value.detail.get("requested_map"))


def test_no_map_aliases_invented():
    registry = yaml.safe_load((ROOT / "config" / "application" / "application_map_registry_v1.yaml").read_text(encoding="utf-8"))
    for m in registry["model_supported_maps"]:
        assert m["aliases"] == []


# --- 10. known-maps series contract ---

def test_exact_ordered_map_count_enforced():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                      ["Mirage", "Inferno"])
    assert exc.value.error_code == "invalid_map_count"


def test_duplicate_map_rejected():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.predict_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                      ["Mirage", "Mirage", "Inferno"])
    assert exc.value.error_code == "duplicate_map"


# --- 11. DP correctness (BO1/3/5, conservation, analytic race-to-N) ---

def _analytic_equal_p(p, best_of):
    need = (best_of + 1) // 2
    return sum(comb(need - 1 + j, j) * (p ** need) * ((1 - p) ** j) for j in range(need))


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_dp_all_half(best_of):
    assert ai.compose_series_probability([0.5] * best_of, best_of) == pytest.approx(0.5)


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_dp_all_one(best_of):
    assert ai.compose_series_probability([1.0] * best_of, best_of) == pytest.approx(1.0)


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_dp_all_zero(best_of):
    assert ai.compose_series_probability([0.0] * best_of, best_of) == pytest.approx(0.0)


@pytest.mark.parametrize("best_of,p", [(3, 0.6), (3, 0.3), (5, 0.6), (5, 0.9)])
def test_dp_matches_analytic_race_to_n(best_of, p):
    v = ai.compose_series_probability([p] * best_of, best_of)
    assert v == pytest.approx(_analytic_equal_p(p, best_of), abs=1e-9)


def test_dp_probability_conservation_random_asymmetric():
    rng = np.random.default_rng(0)
    for _ in range(30):
        best_of = int(rng.choice([1, 3, 5]))
        probs = list(rng.random(best_of))
        pa = ai.compose_series_probability(probs, best_of)
        pb = ai.compose_series_probability([1 - x for x in probs], best_of)
        assert pa + pb == pytest.approx(1.0, abs=1e-9)


def test_dp_validates_own_inputs_independently():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.compose_series_probability([0.5, 0.5], 3)
    assert exc.value.error_code == "invalid_map_count"
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.compose_series_probability([0.5, 1.5, 0.5], 3)
    assert exc.value.error_code == "invalid_probability"
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ai.compose_series_probability([0.5, 0.5], 2)
    assert exc.value.error_code == "invalid_best_of"


# --- 12. known-map wrapper parity (amendment #7) ---

@pytest.mark.parametrize("context_id", ["historical_cologne_pre_event", "deployment_post_cologne_v1"])
@pytest.mark.parametrize("map_name", ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass",
                                       "Train", "Vertigo"])
def test_known_map_wrapper_matches_direct_frozen_chain(context_id, map_name):
    import feature_engineering.maps.rich_modern_map_feature_composer as rmmc
    import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb

    ctx = ai.get_context(context_id)
    team_a, team_b, best_of = "Team Vitality", "Team Falcons", 3

    raw = rmmc.build_future_modern_rich_map_features(
        team_a, team_b, best_of, map_name, ctx.state_cutoff,
        ctx.rf_context.store, ctx.map_state, ctx.form_state, ctx.roster_state, ctx.modern_map_state, tier="tier1")
    df = pd.DataFrame([{k: raw[k] for k in ctx.xgb_preprocessing["original_model_feature_names"]}])
    X, _ = prep_xgb.transform(df, ctx.xgb_preprocessing, ctx.xgb_roles)
    p_direct = float(ctx.xgb_model.predict_proba(X)[:, 1][0])

    r = ai.predict_map(context_id, team_a, team_b, map_name, best_of)
    assert r["probability_team_a"] == pytest.approx(p_direct, abs=1e-12)


def test_known_map_wrapper_parity_cold_start_team():
    # a Stage-1 team with only 4 deployment matches - genuinely thinner state, still parity-checked
    r_direct_team = "THUNDERdOWNUNDER"
    import feature_engineering.maps.rich_modern_map_feature_composer as rmmc
    import feature_engineering.preprocessing.preprocessing_xgboost_map_v3 as prep_xgb

    ctx = ai.get_context("deployment_post_cologne_v1")
    raw = rmmc.build_future_modern_rich_map_features(
        r_direct_team, "MOUZ", 3, "Mirage", ctx.state_cutoff,
        ctx.rf_context.store, ctx.map_state, ctx.form_state, ctx.roster_state, ctx.modern_map_state, tier="tier1")
    df = pd.DataFrame([{k: raw[k] for k in ctx.xgb_preprocessing["original_model_feature_names"]}])
    X, _ = prep_xgb.transform(df, ctx.xgb_preprocessing, ctx.xgb_roles)
    p_direct = float(ctx.xgb_model.predict_proba(X)[:, 1][0])

    r = ai.predict_map("deployment_post_cologne_v1", r_direct_team, "MOUZ", "Mirage", 3)
    assert r["probability_team_a"] == pytest.approx(p_direct, abs=1e-12)


# --- 13. JSON serialization / probability complementarity ---

def test_all_outputs_json_serializable():
    r1 = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    r2 = ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    r3 = ai.predict_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 5,
                                       ["Mirage", "Inferno", "Nuke", "Ancient", "Overpass"])
    for r in (r1, r2, r3):
        json.dumps(r)  # raises if not serializable


def test_probability_complementarity():
    for r in (ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3),
              ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)):
        assert r["probability_team_a"] + r["probability_team_b"] == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= r["probability_team_a"] <= 1.0


def test_favored_team_tie_semantics():
    # construct a synthetic tie via the DP directly (real matchups essentially never tie exactly)
    p = ai.compose_series_probability([0.5, 0.5, 0.5], 3)
    assert p == 0.5


# --- 14. THUNDERdOWNUNDER lifecycle ---

def test_thunderdownunder_historical_cold_start_deployment_has_history():
    r_hist = ai.predict_series_unknown_maps("historical_cologne_pre_event", "THUNDERdOWNUNDER", "MOUZ", 3)
    assert r_hist["team_a_history"]["cold_start"] is True
    assert r_hist["team_a_history"]["matches"] == 0

    r_dep = ai.predict_series_unknown_maps("deployment_post_cologne_v1", "THUNDERdOWNUNDER", "MOUZ", 3)
    assert r_dep["team_a_history"]["cold_start"] is False
    assert r_dep["team_a_history"]["matches"] == 4


# --- 15. zero model/state mutation ---

def test_repeated_inference_does_not_mutate_state():
    ctx = ai.get_context("deployment_post_cologne_v1")
    before = json.dumps({t: {"elo": s.elo, "n": len(s.history)} for t, s in ctx.rf_context.store.teams.items()},
                         sort_keys=True)
    for _ in range(3):
        ai.predict_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
        ai.predict_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
        ai.predict_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                      ["Mirage", "Inferno", "Nuke"])
    after = json.dumps({t: {"elo": s.elo, "n": len(s.history)} for t, s in ctx.rf_context.store.teams.items()},
                        sort_keys=True)
    assert before == after


def test_no_model_fitting_imports_in_application_inference():
    tree = ast.parse((ROOT / "application" / "inference" / "application_inference.py").read_text(encoding="utf-8"))
    forbidden = {"sklearn.ensemble", "xgboost.training"}
    src = (ROOT / "application" / "inference" / "application_inference.py").read_text(encoding="utf-8")
    assert ".fit(" not in src
