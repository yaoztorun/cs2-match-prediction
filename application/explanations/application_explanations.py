"""
Phase 9C model-grounded explanation core. Deterministic, non-causal,
model-grounded explanations for the Phase 9B prediction contract. Never
changes what a prediction returns - every explanation reuses the EXACT SAME
prepared feature vector Phase 9B's own prediction path builds (via
`application_inference._prepare_rf_prediction` / `_prepare_xgb_prediction`,
generated exactly once per request - amendment #5), and asserts additivity
reconstruction against the authoritative prediction before returning
anything.

ATTRIBUTION METHODS (see reports/phase9c_application_explanation_core.md
section B for the full audit):
  RF V2  -> Saabas-style tree PATH decomposition (Saabas 2014), implemented
            directly from sklearn's own exposed tree internals. NOT SHAP,
            NOT Shapley values - path/tree-structure dependent, lacks SHAP's
            symmetry/consistency guarantees. "Exact" means it exactly
            reconstructs THIS forest's own predict_proba output (verified
            empirically), not game-theoretic exactness.
  XGB V3 -> native TreeSHAP via `Booster.predict(..., pred_contribs=True)` -
            genuine, exact Shapley values, built into xgboost's C++ core,
            zero new dependencies.

explanation_type = "model_feature_attribution", causal = false, always.
"""

import numpy as np
import pandas as pd
import yaml
from xgboost import DMatrix

from _common import ROOT
import application.inference.application_inference as ai

CONFIG = ROOT / "config"
FEATURE_GROUPS_PATH = CONFIG / "application" / "application_explanation_feature_groups_v1.yaml"
EXPLANATIONS_REGISTRY_PATH = CONFIG / "application" / "application_explanations_v1.yaml"

# Amendment #14: small deterministic epsilon per attribution output space, used
# ONLY for display-direction classification - the underlying signed
# contribution used for additivity/reconstruction is never altered.
DIRECTION_EPSILON = {"probability": 1e-9, "log_odds": 1e-9}

# Amendment #16: reconstruction tolerances, frozen empirically (see report section B),
# never weakened after seeing a failing case.
RF_RECONSTRUCTION_TOLERANCE = 1e-6     # observed max error 8.88e-16 over 500 real+cold-start vectors
XGB_RECONSTRUCTION_TOLERANCE = 1e-3    # native TreeSHAP is float32-internal; observed margin-space error ~1e-4

_FEATURE_GROUPS_CACHE = None


def _feature_groups():
    global _FEATURE_GROUPS_CACHE
    if _FEATURE_GROUPS_CACHE is None:
        _FEATURE_GROUPS_CACHE = yaml.safe_load(FEATURE_GROUPS_PATH.read_text(encoding="utf-8"))
    return _FEATURE_GROUPS_CACHE


def get_explanation_metadata():
    return ai._to_jsonable(yaml.safe_load(EXPLANATIONS_REGISTRY_PATH.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# RF V2: Saabas-style tree path decomposition (19-dim TRANSFORMED input -
# amendment #1: the forest never sees the 17 raw features directly)
# ---------------------------------------------------------------------------

def _rf_node_p1(tree, node):
    v = tree.value[node, 0, :]
    s = v.sum()
    return float(v[1] / s) if s > 0 else 0.5


def _rf_saabas_contributions(model, x_row_float64):
    """Returns (base_value, contributions[19]) in PROBABILITY space (RF tree
    leaves already store class probabilities). Requires float32 casting for
    split comparisons: sklearn's Cython tree code internally compares
    against DTYPE=float32 even though `tree_.threshold` is exposed as
    float64 in the Python API - this was verified empirically (a genuine
    ~1e-3 reconstruction error was observed and traced to exactly this
    before the float32 cast was added; NOT a theoretical concern)."""
    x32 = x_row_float64.astype(np.float32)
    n_feat = model.n_features_in_
    total_contrib = np.zeros(n_feat, dtype=np.float64)
    total_bias = 0.0
    for est in model.estimators_:
        t = est.tree_
        node = 0
        current = _rf_node_p1(t, 0)
        total_bias += current
        while t.children_left[node] != t.children_right[node]:
            feat = t.feature[node]
            thresh = t.threshold[node]
            child = t.children_left[node] if x32[feat] <= thresh else t.children_right[node]
            child_p = _rf_node_p1(t, child)
            total_contrib[feat] += (child_p - current)
            current = child_p
            node = child
    n_trees = len(model.estimators_)
    return total_bias / n_trees, total_contrib / n_trees


# ---------------------------------------------------------------------------
# XGB V3: native TreeSHAP
# ---------------------------------------------------------------------------

def _xgb_treeshap_contributions(model, X_row, transformed_names):
    """Returns (base_value, contributions[131]) in LOG-ODDS (raw margin)
    space. Amendment #8: uses the low-level Booster.predict with no
    iteration_range restriction - verified (see report section B) that the
    frozen model has no best_iteration/best_ntree_limit/early-stopping
    metadata at all (booster.num_boosted_rounds()==118==metadata
    final_n_estimators, attributes()=={}), so this uses exactly the same
    118 trees XGBClassifier.predict_proba does; there is nothing to
    restrict. Amendment #9: feature order is asserted explicitly against the
    frozen preprocessing contract's transformed_feature_names, not implied
    by array length alone."""
    booster = model.get_booster()
    dm = DMatrix(X_row.reshape(1, -1), feature_names=list(transformed_names))
    if list(dm.feature_names) != list(transformed_names):
        raise ai.ApplicationInferenceError("missing_state_support",
            "DMatrix feature_names do not match the frozen XGB V3 transformed_feature_names order")
    contribs = booster.predict(dm, pred_contribs=True)  # (1, n_features + 1); last column = bias
    if contribs.shape[1] != len(transformed_names) + 1:
        raise ai.ApplicationInferenceError("missing_state_support",
            f"TreeSHAP returned {contribs.shape[1]} columns, expected {len(transformed_names) + 1} "
            f"(131 features + 1 bias)")
    return float(contribs[0, -1]), contribs[0, :-1].astype(np.float64)


# ---------------------------------------------------------------------------
# Grouped contribution aggregation (shared by both models)
# ---------------------------------------------------------------------------

def _classify_direction(contribution, epsilon):
    if abs(contribution) <= epsilon:
        return "neutral"
    return "team_a" if contribution > 0 else "team_b"


def _aggregate_groups(feature_rows, output_space):
    eps = DIRECTION_EPSILON[output_space]
    groups = {}
    for row in feature_rows:
        g = row["factor_group"]
        bucket = groups.setdefault(g, {"factor_group": g, "signed_contribution": 0.0, "supporting_features": []})
        bucket["signed_contribution"] += row["contribution"]
        bucket["supporting_features"].append(row)
    out = []
    for g, data in groups.items():
        c = data["signed_contribution"]
        data["direction"] = _classify_direction(c, eps)
        data["absolute_importance"] = abs(c)
        data["attribution_output_space"] = output_space
        # deterministic supporting-feature order within a group
        data["supporting_features"].sort(key=lambda r: (-abs(r["contribution"]), r["transformed_feature"]))
        out.append(data)
    out.sort(key=lambda d: (-d["absolute_importance"], d["factor_group"]))  # deterministic tie-break
    for rank, d in enumerate(out, start=1):
        d["rank"] = rank
    return out


_GROUP_DISPLAY_LABELS = {
    "overall_strength": "Overall team strength", "recent_performance": "Recent performance",
    "recent_form": "Recent form", "historical_experience": "Historical experience",
    "event_context": "Series format/tier context", "opponent_strength": "Opponent strength",
    "map_pool": "Map-pool depth", "selected_map_strength": "Selected-map strength",
    "map_experience": "Map experience", "player_strength": "Player performance",
    "roster_stability": "Roster stability", "roster_map_familiarity": "Roster map familiarity",
}

_RANK_WORDS = {1: "the largest", 2: "the second-largest", 3: "the third-largest"}


def _group_summary_sentence(group):
    label = _GROUP_DISPLAY_LABELS.get(group["factor_group"], group["factor_group"])
    if group["direction"] == "neutral":
        return f"{label} features had a negligible net effect on the model's prediction."
    team_word = "Team A" if group["direction"] == "team_a" else "Team B"
    rank_word = _RANK_WORDS.get(group["rank"], "a")
    return f"{label} features were {rank_word} factor pushing the model's prediction toward {team_word}."


def _top_positive_negative(grouped, n=3):
    positive = sorted([g for g in grouped if g["signed_contribution"] > 0],
                       key=lambda g: (-g["signed_contribution"], g["factor_group"]))[:n]
    negative = sorted([g for g in grouped if g["signed_contribution"] < 0],
                       key=lambda g: (g["signed_contribution"], g["factor_group"]))[:n]
    return positive, negative


# ---------------------------------------------------------------------------
# Unknown-map (RF V2) explanation
# ---------------------------------------------------------------------------

def _rf_input_provenance(prepared):
    """Amendment #21: distinguishes WHERE a raw input value came from (a real
    history entry vs. a cold-start default) from the model's contribution -
    the fallback itself is never described as having "caused" a contribution."""
    raw = prepared["raw"]
    n1 = raw.get("team1_total_matches_before", 0)
    n2 = raw.get("team2_total_matches_before", 0)
    notes = []
    for side, n in (("team_a", n1), ("team_b", n2)):
        if n == 0:
            notes.append(f"{side}: cold start - elo_before used the ELO_INITIAL default, win-rate features "
                         f"used their 0.5 defaults, match counts are 0, and days_since_last_match was NaN "
                         f"before preprocessing (median-imputed in the transformed model input). These are "
                         f"INPUT DEFAULTS, not model contributions in themselves.")
    return {"team_a_cold_start": n1 == 0, "team_b_cold_start": n2 == 0, "notes": notes}


def explain_series_unknown_maps(context_id, team_a, team_b, best_of, prediction_datetime=None, tier=None):
    ctx = ai.get_context(context_id)
    prepared = ai._prepare_rf_prediction(ctx, team_a, team_b, best_of, prediction_datetime, tier)
    prediction = ai._predict_rf_from_prepared(ctx, context_id, prepared)

    base, contributions = _rf_saabas_contributions(ctx.rf_context.model, prepared["X"][0])
    recon = base + contributions.sum()
    if abs(recon - prediction["probability_team_a"]) > RF_RECONSTRUCTION_TOLERANCE:
        raise ai.ApplicationInferenceError("missing_state_support",
            f"RF explanation additivity check failed: reconstructed={recon} vs predicted="
            f"{prediction['probability_team_a']} (tolerance={RF_RECONSTRUCTION_TOLERANCE})")

    fg_by_name = {f["transformed_feature"]: f for f in _feature_groups()["rf_v2"]["features"]}
    feature_rows = []
    for name, contrib in zip(prepared["transformed_names"], contributions):
        meta = fg_by_name[name]
        feature_rows.append({
            "transformed_feature": name, "raw_feature": meta["raw_feature"],
            "raw_value": prepared["raw"].get(meta["raw_feature"]),
            "transformed_value": None, "contribution": float(contrib), "factor_group": meta["factor_group"],
            "display_label": meta["display_label"],
        })
    # attach transformed values (positional, same order as prepared["X"])
    for row, val in zip(feature_rows, prepared["X"][0]):
        row["transformed_value"] = float(val)

    grouped = _aggregate_groups(feature_rows, "probability")
    top_positive, top_negative = _top_positive_negative(grouped)

    explanation = {
        "explanation_type": "model_feature_attribution", "causal": False,
        "attribution_method": "saabas_path_decomposition", "attribution_output_space": "probability",
        "base_value": float(base), "feature_contributions": feature_rows, "grouped_factors": grouped,
        "team_a_factors": [g for g in grouped if g["direction"] == "team_a"],
        "team_b_factors": [g for g in grouped if g["direction"] == "team_b"],
        "neutral_factors": [g for g in grouped if g["direction"] == "neutral"],
        "top_positive_factors": top_positive, "top_negative_factors": top_negative,
        "human_readable_summary": [_group_summary_sentence(g) for g in (top_positive + top_negative)],
        "input_provenance": _rf_input_provenance(prepared),
        "reconstruction_check": {"reconstructed_probability_team_a": float(recon),
                                 "predicted_probability_team_a": prediction["probability_team_a"],
                                 "tolerance": RF_RECONSTRUCTION_TOLERANCE, "passed": True},
    }
    return ai._to_jsonable({"prediction": prediction, "explanation": explanation})


# ---------------------------------------------------------------------------
# Known-map (XGB V3) explanation
# ---------------------------------------------------------------------------

def explain_map(context_id, team_a, team_b, map_name, best_of, prediction_datetime=None, tier=None):
    ctx = ai.get_context(context_id)
    prepared = ai._prepare_xgb_prediction(ctx, team_a, team_b, map_name, best_of, prediction_datetime, tier)
    prediction = ai._predict_xgb_from_prepared(ctx, context_id, prepared)

    base, contributions = _xgb_treeshap_contributions(ctx.xgb_model, prepared["X"][0], prepared["transformed_names"])
    margin = base + contributions.sum()
    recon_p = float(1.0 / (1.0 + np.exp(-margin)))
    if abs(recon_p - prediction["probability_team_a"]) > XGB_RECONSTRUCTION_TOLERANCE:
        raise ai.ApplicationInferenceError("missing_state_support",
            f"XGB explanation additivity check failed: sigmoid(reconstructed_margin)={recon_p} vs predicted="
            f"{prediction['probability_team_a']} (tolerance={XGB_RECONSTRUCTION_TOLERANCE})")

    fg_by_name = {f["transformed_feature"]: f for f in _feature_groups()["map_xgboost_v3_final"]["features"]}
    feature_rows = []
    for name, contrib, val in zip(prepared["transformed_names"], contributions, prepared["X"][0]):
        meta = fg_by_name[name]
        feature_rows.append({
            "transformed_feature": name, "transformed_value": float(val), "contribution": float(contrib),
            "factor_group": meta["factor_group"], "family": meta["family"], "family_label": meta["family_label"],
        })

    grouped = _aggregate_groups(feature_rows, "log_odds")
    top_positive, top_negative = _top_positive_negative(grouped)

    explanation = {
        "explanation_type": "model_feature_attribution", "causal": False,
        "attribution_method": "xgboost_native_treeshap", "attribution_output_space": "log_odds",
        "base_value": float(base), "feature_contributions": feature_rows, "grouped_factors": grouped,
        "team_a_factors": [g for g in grouped if g["direction"] == "team_a"],
        "team_b_factors": [g for g in grouped if g["direction"] == "team_b"],
        "neutral_factors": [g for g in grouped if g["direction"] == "neutral"],
        "top_positive_factors": top_positive, "top_negative_factors": top_negative,
        "human_readable_summary": [_group_summary_sentence(g) for g in (top_positive + top_negative)],
        "reconstruction_check": {"reconstructed_margin": float(margin), "sigmoid_reconstructed": recon_p,
                                 "predicted_probability_team_a": prediction["probability_team_a"],
                                 "tolerance": XGB_RECONSTRUCTION_TOLERANCE, "passed": True},
    }
    # amendment #20: state_support/fallbacks_used carried through SEPARATELY, never converted into factors
    out = {"prediction": prediction, "explanation": explanation, "state_support": prediction["state_support"]}
    return ai._to_jsonable(out)


# ---------------------------------------------------------------------------
# Known-maps series: TWO explicit layers (amendment #19 - never conflated)
# ---------------------------------------------------------------------------

def _reach_probabilities(map_probabilities, best_of):
    """probability_map_is_reached per slot (0-indexed). Verified against
    analytical closed forms for BO1/BO3/BO5 (amendment #17)."""
    wins_needed = (best_of + 1) // 2
    dp = {(0, 0): 1.0}
    reach = []
    for i in range(best_of):
        reach.append(sum(v for (played, _wa), v in dp.items() if played == i))
        p = map_probabilities[i]
        next_dp = {}
        for (played, wins_a), prob in dp.items():
            if played != i:
                continue
            wins_b = played - wins_a
            new_a = wins_a + 1
            if new_a < wins_needed:
                next_dp[(played + 1, new_a)] = next_dp.get((played + 1, new_a), 0.0) + prob * p
            new_b = wins_b + 1
            if new_b < wins_needed:
                next_dp[(played + 1, wins_a)] = next_dp.get((played + 1, wins_a), 0.0) + prob * (1.0 - p)
        dp = next_dp
    return reach


def _series_composition_leverage(map_probabilities, best_of):
    """series_composition_leverage_i = P_series(p_i=1) - P_series(p_i=0), all
    other map probabilities held fixed (amendment #18). NOT SHAP, NOT model
    feature attribution, NOT a causal effect - a property of the DP
    composer only."""
    leverages = []
    for i in range(best_of):
        p1 = list(map_probabilities); p1[i] = 1.0
        p0 = list(map_probabilities); p0[i] = 0.0
        leverages.append(ai.compose_series_probability(p1, best_of) - ai.compose_series_probability(p0, best_of))
    return leverages


def explain_series_known_maps(context_id, team_a, team_b, best_of, ordered_maps, prediction_datetime=None,
                               tier=None):
    ctx = ai.get_context(context_id)
    best_of = ai.validate_best_of(best_of)
    canon_a = ai.resolve_team(team_a, ctx.identity_policy)
    canon_b = ai.resolve_team(team_b, ctx.identity_policy)
    if canon_a == canon_b:
        ai._err("same_team", f"team_a and team_b resolve to the same team ({canon_a!r})",
                team_a=team_a, team_b=team_b, canonical=canon_a)
    expected_n = ai._MAP_COUNT_FOR_BO[best_of]
    if len(ordered_maps) != expected_n:
        ai._err("invalid_map_count", f"best_of={best_of} requires exactly {expected_n} ordered_maps, "
                f"got {len(ordered_maps)}", best_of=best_of, expected=expected_n, got=len(ordered_maps))
    if len(set(ordered_maps)) != len(ordered_maps):
        ai._err("duplicate_map", f"ordered_maps contains a duplicate: {ordered_maps}", ordered_maps=ordered_maps)

    # ---- layer A: real per-map XGB attribution (no series-level summing) ----
    map_explanations = []
    for i, map_name in enumerate(ordered_maps, start=1):
        me = explain_map(context_id, canon_a, canon_b, map_name, best_of,
                          prediction_datetime=prediction_datetime, tier=tier)
        map_explanations.append({"map_number": i, "map_name": map_name, **me})

    map_probabilities = [me["prediction"]["probability_team_a"] for me in map_explanations]
    series_p_a = ai.compose_series_probability(map_probabilities, best_of)

    # ---- layer B: series-composition mathematics only, kept explicitly separate ----
    reach = _reach_probabilities(map_probabilities, best_of)
    leverage = _series_composition_leverage(map_probabilities, best_of)

    series_composition = []
    for i, map_name in enumerate(ordered_maps):
        series_composition.append({
            "map_number": i + 1, "map_name": map_name, "probability_team_a": map_probabilities[i],
            "probability_map_is_reached": reach[i], "series_composition_leverage": leverage[i],
        })

    favored_team, is_tied = (None, True) if series_p_a == 0.5 else \
        ((canon_a if series_p_a > 0.5 else canon_b), False)

    out = {
        "prediction_mode": "known_maps", "team_a": canon_a, "team_b": canon_b, "best_of": best_of,
        "series_probability_team_a": series_p_a, "series_probability_team_b": 1.0 - series_p_a,
        "favored_team": favored_team, "prediction_is_tied": is_tied,
        "composition_method": "ordered_map_dynamic_program", "context_id": context_id,
        "state_cutoff": str(ctx.state_cutoff),
        "map_level_explanations": map_explanations,
        "series_composition": series_composition,
        "note": "map_level_explanations (XGB TreeSHAP, log-odds space, per map) and series_composition "
                "(reach probability / composition leverage, DP probability space) are two mathematically "
                "distinct layers and are never summed together.",
    }
    return ai._to_jsonable(out)
