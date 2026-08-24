"""
Pre-veto series-prediction adapter (Phase 8D). A thin wrapper around the
already-frozen Phase 3/4B components - it adds NO new modeling logic:

    feature_engine.build_features  ->  preprocessing_random_forest_v1.transform
    ->  RandomForestClassifier.predict_proba

No retraining, refitting, calibration, threshold tuning, feature changes,
probability averaging, or ensembling happens anywhere in this file. This
module also never reads data/raw, data/interim, or any Cologne-result
artifact - only the explicitly frozen allowlisted inputs below.

Implements the frozen contract from reports/phase8a_application_prediction_contracts.md
section F (`predict_series_unknown_maps`) for real, for the first time.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from _common import ROOT
import feature_engineering.series.feature_engine as fe
import feature_engineering.preprocessing.preprocessing_random_forest_v1 as prep_rf

MODEL_ID = "series_random_forest_v2"
PREDICTION_MODE = "pre_veto"
FROZEN_TIER = "tier1"

MODEL_PATH = ROOT / "models" / "series" / "random_forest_v2.joblib"
PREPROCESSING_PATH = ROOT / "data" / "modeling" / "random_forest_preprocessing_v2.json"
CONFIG_PATH = ROOT / "data" / "modeling" / "random_forest_v2_selected_config.json"
STATE_PATH = ROOT / "data" / "features" / "pre_cologne_team_state_v1_full.json"
SERIES_FEATURES_YAML_PATH = ROOT / "config" / "features" / "series_features_v1.yaml"
FEATURE_ENGINE_PATH = ROOT / "feature_engineering" / "series" / "feature_engine.py"
PREPROCESSING_MODULE_PATH = ROOT / "feature_engineering" / "preprocessing" / "preprocessing_random_forest_v1.py"
PREDICTOR_PATH = ROOT / "tournament" / "simulation" / "pre_veto_series_predictor.py"
TOURNAMENT_YAML_PATH = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"
TOURNAMENT_ENGINE_PATH = ROOT / "tournament" / "engine" / "tournament_engine.py"

# Every artifact whose bytes can materially change RF V2 inference -
# amendment #1's "complete prediction pipeline" hash list.
PIPELINE_HASH_INPUTS = {
    "model": MODEL_PATH,
    "preprocessing": PREPROCESSING_PATH,
    "selected_config": CONFIG_PATH,
    "series_features_v1_yaml": SERIES_FEATURES_YAML_PATH,
    "feature_engine": FEATURE_ENGINE_PATH,
    "preprocessing_random_forest_v1": PREPROCESSING_MODULE_PATH,
    "strict_pre_cologne_state": STATE_PATH,
    "predictor_adapter": PREDICTOR_PATH,
    "tournament_yaml": TOURNAMENT_YAML_PATH,
    "tournament_engine": TOURNAMENT_ENGINE_PATH,
}

EXPECTED_TRANSFORMED_FEATURE_NAMES = [
    "elo_diff", "overall_win_rate_diff", "win_rate_last_5_diff", "win_rate_last_10_diff",
    "format_win_rate_diff", "avg_series_margin_last_5_diff", "avg_series_margin_last_10_diff",
    "matches_last_30_days_diff", "days_since_last_match_diff", "total_matches_before_diff",
    "history_matches_min", "history_matches_sum", "both_teams_have_history",
    "both_teams_have_5_matches", "both_teams_have_10_matches",
    "bestOf_BO3", "bestOf_BO5", "tier_tier2", "tier_tier3",
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_pipeline_inputs():
    return {name: sha256_file(p) for name, p in PIPELINE_HASH_INPUTS.items()}


def environment_versions():
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__}


@dataclass
class PredictorContext:
    model: object
    preprocessing: dict
    config: dict
    store: "fe.StateStore"
    tier: str
    model_contract: dict  # verify_model_contract()'s report, captured at load time


def verify_model_contract(model, preprocessing):
    """Amendment #2: do not assume predict_proba(X)[:, 1] is correct merely
    by precedent - verify it against this exact frozen model artifact."""
    report = {}
    classes = list(model.classes_)
    report["classes_"] = classes
    report["classes_are_0_1_with_class1_meaning_team_a_wins"] = (classes == [0.0, 1.0] or classes == [0, 1])
    report["raw_feature_count"] = len(preprocessing["original_model_feature_names"])
    report["raw_feature_count_is_17"] = report["raw_feature_count"] == 17
    transformed_names = preprocessing["transformed_feature_names"]
    report["transformed_feature_count"] = len(transformed_names)
    report["transformed_feature_count_is_19"] = report["transformed_feature_count"] == 19
    report["transformed_names_match_expected_order"] = (transformed_names == EXPECTED_TRANSFORMED_FEATURE_NAMES)
    report["n_features_in_"] = int(model.n_features_in_)
    report["n_features_in_matches_transformed_count"] = (model.n_features_in_ == len(transformed_names))
    report["no_missing_or_unexpected_transformed_feature"] = (
        set(transformed_names) == set(EXPECTED_TRANSFORMED_FEATURE_NAMES))
    report["all_checks_pass"] = all([
        report["classes_are_0_1_with_class1_meaning_team_a_wins"],
        report["raw_feature_count_is_17"], report["transformed_feature_count_is_19"],
        report["transformed_names_match_expected_order"], report["n_features_in_matches_transformed_count"],
        report["no_missing_or_unexpected_transformed_feature"],
    ])
    if not report["all_checks_pass"]:
        raise ValueError(f"RF V2 model/feature contract verification FAILED: {report}")
    return report


def load_predictor_context(tier=FROZEN_TIER):
    model = joblib.load(MODEL_PATH)
    # Force single-threaded inference: RandomForestClassifier.predict_proba's
    # cross-tree averaging is parallelized when n_jobs=-1 (as trained), and
    # parallel float summation order is not guaranteed deterministic across
    # calls (observed: 1-ULP jitter). This changes NOTHING about what the
    # model learned or predicts in distribution - only forces a fixed
    # sequential summation order so repeated identical queries are
    # byte-identical, not just "close." Not retraining, not calibration.
    model.n_jobs = 1
    preprocessing = prep_rf.load_preprocessing(PREPROCESSING_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    store = fe.StateStore.from_json(STATE_PATH)
    contract = verify_model_contract(model, preprocessing)
    return PredictorContext(model=model, preprocessing=preprocessing, config=config,
                             store=store, tier=tier, model_contract=contract)


def predict_series_unknown_maps(context: PredictorContext, team_a, team_b, best_of,
                                 prediction_datetime, tier=None):
    """Frozen contract (reports/phase8a_application_prediction_contracts.md sec. F).
    team_a/team_b map 1:1 onto feature_engine's team1/team2 - no reorientation.
    Read-only: never mutates context.store."""
    if tier is not None and tier != context.tier:
        raise ValueError(f"tier={tier!r} was passed but context.tier={context.tier!r} is the frozen value - "
                          f"refusing a silent per-call tier override")
    tier = context.tier

    raw = fe.build_features(context.store, team_a, team_b, prediction_datetime, best_of, tier=tier)
    df = pd.DataFrame([{k: raw[k] for k in context.preprocessing["original_model_feature_names"]}])
    X, names = prep_rf.transform(df, context.preprocessing)
    if names != context.preprocessing["transformed_feature_names"]:
        raise ValueError("transformed feature order drifted from the frozen preprocessing contract")
    proba = context.model.predict_proba(X)[:, 1]
    probability_team_a = float(proba[0])
    return {
        "team_a": team_a, "team_b": team_b, "best_of": int(best_of),
        "prediction_datetime": str(pd.Timestamp(prediction_datetime)),
        "tier": tier,
        "probability_team_a": probability_team_a,
        "probability_team_b": 1.0 - probability_team_a,
        "model_id": MODEL_ID,
        "prediction_mode": PREDICTION_MODE,
    }


def validate_bo_support(context, team_a, team_b, prediction_datetime):
    """Amendment #2/#6: prove BO1/BO3/BO5 all transform and score cleanly,
    and save the exact transformed one-hot pattern for each - inference-
    contract validation only, no outcome comparison."""
    expected_pattern = {
        1: {"bestOf_BO3": 0.0, "bestOf_BO5": 0.0},
        3: {"bestOf_BO3": 1.0, "bestOf_BO5": 0.0},
        5: {"bestOf_BO3": 0.0, "bestOf_BO5": 1.0},
    }
    results = {}
    for bo in (1, 3, 5):
        raw = fe.build_features(context.store, team_a, team_b, prediction_datetime, bo, tier=context.tier)
        df = pd.DataFrame([{k: raw[k] for k in context.preprocessing["original_model_feature_names"]}])
        X, names = prep_rf.transform(df, context.preprocessing)
        row = dict(zip(names, X[0].tolist()))
        pattern_ok = all(row[k] == v for k, v in expected_pattern[bo].items())
        proba = float(context.model.predict_proba(X)[:, 1][0])
        results[bo] = {
            "transformed_one_hot_pattern": {k: row[k] for k in ("bestOf_BO3", "bestOf_BO5")},
            "pattern_matches_expected": pattern_ok,
            "probability_team_a": proba,
            "finite": bool(np.isfinite(proba)),
            "in_unit_interval": bool(0.0 <= proba <= 1.0),
        }
        if not (pattern_ok and np.isfinite(proba) and 0.0 <= proba <= 1.0):
            raise ValueError(f"BO{bo} inference-contract validation FAILED: {results[bo]}")
    return results


def verify_tier_representation(context, team_a, team_b, prediction_datetime, best_of=3):
    """Amendment #2: prove tier='tier1' produces the frozen reference-category
    representation (both tier dummy columns 0) - no silent category fallback."""
    raw = fe.build_features(context.store, team_a, team_b, prediction_datetime, best_of, tier=context.tier)
    df = pd.DataFrame([{k: raw[k] for k in context.preprocessing["original_model_feature_names"]}])
    X, names = prep_rf.transform(df, context.preprocessing)
    row = dict(zip(names, X[0].tolist()))
    is_reference = (row["tier_tier2"] == 0.0 and row["tier_tier3"] == 0.0)
    if not is_reference:
        raise ValueError(f"tier={context.tier!r} did NOT produce the reference-category representation: "
                          f"tier_tier2={row['tier_tier2']}, tier_tier3={row['tier_tier3']}")
    return {"tier": context.tier, "tier_tier2": row["tier_tier2"], "tier_tier3": row["tier_tier3"],
            "is_reference_category": is_reference}


def audit_team_coverage(context, teams):
    """teams: list of dicts with display_name/canonical_model_name/starting_stage
    (from the frozen tournament YAML's participants). Read-only, descriptive
    only - never alters model behavior based on coverage."""
    rows = []
    for t in teams:
        canonical = t["canonical_model_name"]
        state = context.store.get(canonical)
        present = state is not None
        if present and state.history:
            last_dt = max(h.dt for h in state.history)
            days_since = None  # filled by caller using the frozen cutoff, kept tz-naive here
        else:
            last_dt = None
        rows.append({
            "display_name": t["display_name"],
            "canonical_model_name": canonical,
            "starting_stage": t.get("starting_stage"),
            "identity_confirmed": t.get("resolution_status") == "confirmed",
            "present_in_pre_cologne_state": present,
            "total_matches_before": len(state.history) if present else 0,
            "elo": state.elo if present else fe.ELO_INITIAL,
            "last_match_datetime": str(last_dt) if last_dt is not None else None,
            "cold_start": (not present) or len(state.history) == 0,
        })
    return pd.DataFrame(rows)
