"""
Phase 9C tests: explanation registry, feature-group coverage, RF/XGB
attribution parity with prediction, additivity reconstruction, sign
semantics, deterministic ranking, JSON serialization, historical/deployment
isolation, THUNDERdOWNUNDER lifecycle, DP reach/leverage invariants, and a
final re-confirmation that Phase 9B parity survived the refactor.
"""

import json
from itertools import product
from math import comb

import numpy as np
import pytest
import yaml

from _common import ROOT
import application.inference.application_inference as ai
import application.explanations.application_explanations as ae

CONFIG = ROOT / "config"


@pytest.fixture(scope="module")
def feature_groups():
    return yaml.safe_load((CONFIG / "application" / "application_explanation_feature_groups_v1.yaml").read_text(encoding="utf-8"))


# --- 1. explanation registry ---

def test_explanation_registry_fields():
    reg = ae.get_explanation_metadata()
    assert reg["causal"] is False
    assert reg["explanation_type"] == "model_feature_attribution"
    assert reg["models"]["series_random_forest_v2"]["attribution_method"] == "saabas_path_decomposition"
    assert reg["models"]["series_random_forest_v2"]["attribution_output_space"] == "probability"
    assert reg["models"]["map_xgboost_v3_final"]["attribution_method"] == "xgboost_native_treeshap"
    assert reg["models"]["map_xgboost_v3_final"]["attribution_output_space"] == "log_odds"


# --- 2. feature-group coverage ---

def test_rf_feature_group_full_coverage(feature_groups):
    ctx = ai.get_context("deployment_post_cologne_v1")
    expected = set(ctx.rf_context.preprocessing["transformed_feature_names"])
    mapped = {f["transformed_feature"] for f in feature_groups["rf_v2"]["features"]}
    assert mapped == expected
    assert len(mapped) == 19


def test_xgb_feature_group_full_coverage(feature_groups):
    ctx = ai.get_context("deployment_post_cologne_v1")
    expected = set(ctx.xgb_preprocessing["transformed_feature_names"])
    mapped = {f["transformed_feature"] for f in feature_groups["map_xgboost_v3_final"]["features"]}
    assert mapped == expected
    assert len(mapped) == 131


def test_no_duplicate_feature_mapping(feature_groups):
    for model_key in ("rf_v2", "map_xgboost_v3_final"):
        names = [f["transformed_feature"] for f in feature_groups[model_key]["features"]]
        assert len(names) == len(set(names))


def test_every_feature_maps_to_exactly_one_group(feature_groups):
    for model_key in ("rf_v2", "map_xgboost_v3_final"):
        for f in feature_groups[model_key]["features"]:
            assert f["factor_group"], f"{f['transformed_feature']} has no factor_group"


def test_rf_does_not_fabricate_unsupported_groups(feature_groups):
    groups_used = {f["factor_group"] for f in feature_groups["rf_v2"]["features"]}
    forbidden = {"opponent_strength", "map_pool", "selected_map_strength", "map_experience", "player_strength",
                 "roster_stability", "roster_map_familiarity"}
    assert groups_used.isdisjoint(forbidden)


# --- 3. RF explanation parity with prediction + additivity ---

def test_rf_explanation_matches_prediction():
    r = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    assert r["explanation"]["reconstruction_check"]["passed"] is True
    assert abs(r["explanation"]["reconstruction_check"]["reconstructed_probability_team_a"] -
               r["prediction"]["probability_team_a"]) < 1e-6


def test_rf_additivity_broad_fixture_set():
    """Amendment #3: broad empirical validation, not a handful of manual examples."""
    import joblib
    import pandas as pd
    import feature_engineering.series.feature_engine as fe
    import feature_engineering.preprocessing.preprocessing_random_forest_v1 as prep_rf
    import random

    ctx = ai.get_context("deployment_post_cologne_v1")
    store = ctx.rf_context.store
    prep = ctx.rf_context.preprocessing
    teams = list(store.teams.keys())
    random.seed(1)
    max_err = 0.0
    for _ in range(200):
        t1, t2 = random.sample(teams, 2)
        dt = ctx.state_cutoff
        bo = random.choice([1, 3, 5])
        tier = random.choice(["tier1", "tier2", "tier3"])
        raw = fe.build_features(store, t1, t2, dt, bo, tier=tier)
        df = pd.DataFrame([{k: raw[k] for k in prep["original_model_feature_names"]}])
        X, _ = prep_rf.transform(df, prep)
        base, contrib = ae._rf_saabas_contributions(ctx.rf_context.model, X[0])
        recon = base + contrib.sum()
        actual = ctx.rf_context.model.predict_proba(X)[0, 1]
        max_err = max(max_err, abs(recon - actual))
    assert max_err < 1e-9, f"max reconstruction error {max_err} exceeds tolerance"


def test_rf_class_semantics_verified():
    ctx = ai.get_context("deployment_post_cologne_v1")
    m = ctx.rf_context.model
    assert list(m.classes_) == [0.0, 1.0]
    assert all(list(est.classes_) == [0.0, 1.0] for est in m.estimators_)


# --- 4. XGB explanation parity with prediction + additivity ---

def test_xgb_explanation_matches_prediction():
    r = ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    check = r["explanation"]["reconstruction_check"]
    assert abs(check["sigmoid_reconstructed"] - r["prediction"]["probability_team_a"]) < \
        ae.XGB_RECONSTRUCTION_TOLERANCE


def test_xgb_tree_range_matches_predict_proba():
    """Amendment #8: proves TreeSHAP uses exactly the same trees as predict_proba."""
    ctx = ai.get_context("deployment_post_cologne_v1")
    booster = ctx.xgb_model.get_booster()
    assert booster.attributes() == {}
    assert getattr(ctx.xgb_model, "best_iteration", None) is None


def test_xgb_feature_order_proven():
    """Amendment #9."""
    ctx = ai.get_context("deployment_post_cologne_v1")
    prepared = ai._prepare_xgb_prediction(ctx, "Team Vitality", "Team Falcons", "Mirage", 3, None, None)
    assert prepared["transformed_names"] == ctx.xgb_preprocessing["transformed_feature_names"]
    base, contribs = ae._xgb_treeshap_contributions(ctx.xgb_model, prepared["X"][0], prepared["transformed_names"])
    assert len(contribs) == 131


def test_xgb_max_reconstruction_error_recorded():
    ctx = ai.get_context("deployment_post_cologne_v1")
    max_err = 0.0
    for map_name in [m["canonical_name"] for m in ctx.map_registry["model_supported_maps"]]:
        r = ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", map_name, 3)
        chk = r["explanation"]["reconstruction_check"]
        max_err = max(max_err, abs(chk["sigmoid_reconstructed"] - r["prediction"]["probability_team_a"]))
    assert max_err < ae.XGB_RECONSTRUCTION_TOLERANCE


# --- 5. sign semantics / grouped contribution sum ---

def test_grouped_contribution_sum_equals_total():
    r = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    total_feature = sum(f["contribution"] for f in r["explanation"]["feature_contributions"])
    total_group = sum(g["signed_contribution"] for g in r["explanation"]["grouped_factors"])
    assert abs(total_feature - total_group) < 1e-9


def test_direction_sign_semantics():
    r = ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    for g in r["explanation"]["grouped_factors"]:
        if g["direction"] == "team_a":
            assert g["signed_contribution"] > 0
        elif g["direction"] == "team_b":
            assert g["signed_contribution"] < 0
        else:
            assert abs(g["signed_contribution"]) <= ae.DIRECTION_EPSILON["log_odds"]


def test_team_a_team_b_factor_partition():
    r = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    ex = r["explanation"]
    total = len(ex["team_a_factors"]) + len(ex["team_b_factors"]) + len(ex["neutral_factors"])
    assert total == len(ex["grouped_factors"])


# --- 6. deterministic ranking ---

def test_deterministic_ranking_repeated_calls():
    r1 = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    r2 = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    ranks1 = [(g["factor_group"], g["rank"]) for g in r1["explanation"]["grouped_factors"]]
    ranks2 = [(g["factor_group"], g["rank"]) for g in r2["explanation"]["grouped_factors"]]
    assert ranks1 == ranks2
    assert r1["explanation"]["human_readable_summary"] == r2["explanation"]["human_readable_summary"]


# --- 7. JSON serialization ---

def test_all_explanation_outputs_json_serializable():
    r1 = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    r2 = ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    r3 = ae.explain_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                       ["Mirage", "Inferno", "Nuke"])
    for r in (r1, r2, r3):
        json.dumps(r)


# --- 8. historical / deployment isolation ---

def test_historical_deployment_may_differ_but_correct_state_loaded():
    r_hist = ae.explain_series_unknown_maps("historical_cologne_pre_event", "Team Vitality", "Team Falcons", 3)
    r_dep = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
    assert r_hist["prediction"]["state_cutoff"] != r_dep["prediction"]["state_cutoff"]
    assert r_hist["prediction"]["context_id"] == "historical_cologne_pre_event"
    assert r_dep["prediction"]["context_id"] == "deployment_post_cologne_v1"


def test_historical_datetime_locked_for_explanations():
    with pytest.raises(ai.ApplicationInferenceError) as exc:
        ae.explain_series_unknown_maps("historical_cologne_pre_event", "Team Vitality", "Team Falcons", 3,
                                        prediction_datetime="2026-01-01")
    assert exc.value.error_code == "historical_context_datetime_locked"


# --- 9. THUNDERdOWNUNDER lifecycle ---

def test_thunderdownunder_explanation_lifecycle():
    r_hist = ae.explain_series_unknown_maps("historical_cologne_pre_event", "THUNDERdOWNUNDER", "MOUZ", 3)
    assert r_hist["explanation"]["input_provenance"]["team_a_cold_start"] is True
    assert len(r_hist["explanation"]["input_provenance"]["notes"]) == 1

    r_dep = ae.explain_series_unknown_maps("deployment_post_cologne_v1", "THUNDERdOWNUNDER", "MOUZ", 3)
    assert r_dep["explanation"]["input_provenance"]["team_a_cold_start"] is False
    assert r_dep["prediction"]["team_a_history"]["matches"] == 4


# --- 10. missing-state/fallback metadata kept separate from attributions ---

def test_state_support_kept_separate_from_factors():
    r = ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
    assert "state_support" in r
    assert "state_support" not in r["explanation"]
    for g in r["explanation"]["grouped_factors"]:
        assert "fallbacks_used" not in g


# --- 11. DP map-reach probabilities ---

def test_reach_probability_bo1():
    assert ae._reach_probabilities([0.6], 1) == [1.0]


def test_reach_probability_bo3_analytic():
    p1, p2, p3 = 0.6, 0.4, 0.5
    reach = ae._reach_probabilities([p1, p2, p3], 3)
    assert reach[0] == pytest.approx(1.0)
    assert reach[1] == pytest.approx(1.0)
    assert reach[2] == pytest.approx(p1 * (1 - p2) + (1 - p1) * p2)


def test_reach_probability_bo5_structural():
    reach = ae._reach_probabilities([0.6, 0.5, 0.4, 0.55, 0.5], 5)
    assert reach[0] == pytest.approx(1.0)
    assert reach[1] == pytest.approx(1.0)
    assert all(0.0 <= r <= 1.0 for r in reach)
    # monotonic non-increasing
    for i in range(len(reach) - 1):
        assert reach[i + 1] <= reach[i] + 1e-12


def test_reach_probability_monotonic_random():
    rng = np.random.default_rng(2)
    for _ in range(20):
        bo = int(rng.choice([1, 3, 5]))
        probs = list(rng.random(bo))
        reach = ae._reach_probabilities(probs, bo)
        assert reach[0] == pytest.approx(1.0)
        assert all(0.0 <= r <= 1.0 + 1e-12 for r in reach)
        for i in range(len(reach) - 1):
            assert reach[i + 1] <= reach[i] + 1e-9


# --- 12. DP series-composition leverage ---

def test_leverage_independent_of_original_p_i():
    """Amendment #18: leverage_i must not depend on the original value of p_i."""
    others = [0.4, 0.6]
    lev_a = ae._series_composition_leverage([0.1] + others, 3)[0]
    lev_b = ae._series_composition_leverage([0.9] + others, 3)[0]
    assert lev_a == pytest.approx(lev_b, abs=1e-12)


def test_leverage_equals_reach_for_last_map():
    """Mathematical identity: for the final map slot, forcing p=1 vs p=0 changes the
    outcome ONLY in the branch where that map is reached, and swings it fully -
    so leverage_last == reach_last, always."""
    probs = [0.6, 0.4, 0.55, 0.3, 0.7]
    reach = ae._reach_probabilities(probs, 5)
    leverage = ae._series_composition_leverage(probs, 5)
    assert leverage[-1] == pytest.approx(reach[-1], abs=1e-9)


def test_leverage_not_confused_with_feature_attribution():
    r = ae.explain_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                      ["Mirage", "Inferno", "Nuke"])
    assert "series_composition" in r and "map_level_explanations" in r
    for sc in r["series_composition"]:
        assert set(sc) == {"map_number", "map_name", "probability_team_a", "probability_map_is_reached",
                           "series_composition_leverage"}
    # map-level explanations carry the real XGB attribution, never summed into series_composition
    for me in r["map_level_explanations"]:
        assert "explanation" in me and "feature_contributions" in me["explanation"]


# --- 13. BO1/BO3/BO5 explanations ---

@pytest.mark.parametrize("best_of,maps", [(1, ["Mirage"]), (3, ["Mirage", "Inferno", "Nuke"]),
                                          (5, ["Mirage", "Inferno", "Nuke", "Ancient", "Overpass"])])
def test_known_series_explanation_bo(best_of, maps):
    r = ae.explain_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", best_of, maps)
    assert len(r["map_level_explanations"]) == best_of
    assert len(r["series_composition"]) == best_of
    assert 0.0 <= r["series_probability_team_a"] <= 1.0


# --- 14. Phase 9B parity/determinism/immutability still intact (real commands) ---

def test_phase9b_pytest_suite_passes():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "pytest",
                              "tests/application/test_phase9b_application_inference.py", "-q"],
                             cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout[-3000:]


def test_phase9b_validator_passes():
    import subprocess
    import sys
    import os
    result = subprocess.run([sys.executable, "-m", "validation.validate_phase9b"], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8",
                             env={"PYTHONIOENCODING": "utf-8", **os.environ})
    assert result.returncode == 0, result.stdout[-3000:]


# --- 15. zero model/state mutation ---

def test_repeated_explanations_do_not_mutate_state():
    ctx = ai.get_context("deployment_post_cologne_v1")
    before = json.dumps({t: {"elo": s.elo, "n": len(s.history)} for t, s in ctx.rf_context.store.teams.items()},
                         sort_keys=True)
    for _ in range(3):
        ae.explain_series_unknown_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3)
        ae.explain_map("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", "Mirage", 3)
        ae.explain_series_known_maps("deployment_post_cologne_v1", "Team Vitality", "Team Falcons", 3,
                                      ["Mirage", "Inferno", "Nuke"])
    after = json.dumps({t: {"elo": s.elo, "n": len(s.history)} for t, s in ctx.rf_context.store.teams.items()},
                        sort_keys=True)
    assert before == after
