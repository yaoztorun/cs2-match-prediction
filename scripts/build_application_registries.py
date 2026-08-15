"""
Builds config/application_inference_contexts_v1.yaml: the explicit,
non-filesystem-scanning version registry for Phase 9B's two inference
contexts. Records the COMPLETE executable prediction contract for each
context - every file that can materially change the resulting inference
vector/probability - not just the model file (amendment #1).
"""

import hashlib

import yaml

from _common import ROOT
import pre_veto_series_predictor as pvp

DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"
MODELS = ROOT / "models"
MODELING = DATA / "modeling"

RF_PIPELINE = {
    "rf_model": MODELS / "random_forest_v2.joblib",
    "rf_preprocessing": MODELING / "random_forest_preprocessing_v2.json",
    "rf_selected_config": MODELING / "random_forest_v2_selected_config.json",
    "series_feature_schema_yaml": CONFIG / "series_features_v1.yaml",
    "feature_engine": SCRIPTS / "feature_engine.py",
    "preprocessing_random_forest_v1": SCRIPTS / "preprocessing_random_forest_v1.py",
    "predictor_adapter": SCRIPTS / "pre_veto_series_predictor.py",
}

XGB_PIPELINE = {
    "xgb_model": MODELS / "map_xgboost_v3_final.json",
    "xgb_metadata": MODELS / "map_xgboost_v3_final_metadata.json",
    "xgb_preprocessing": MODELING / "map_xgboost_v3_final_preprocessing.json",
    "xgb_selected_config": MODELING / "map_xgboost_v3_final_config.json",
    "map_v3_feature_schema_yaml": CONFIG / "map_features_v3_modern_map.yaml",
    "rich_modern_map_feature_composer": SCRIPTS / "rich_modern_map_feature_composer.py",
    "rich_map_feature_composer": SCRIPTS / "rich_map_feature_composer.py",
    "preprocessing_xgboost_map_v3": SCRIPTS / "preprocessing_xgboost_map_v3.py",
    "preprocessing_common_map_v2": SCRIPTS / "preprocessing_common_map_v2.py",
    "preprocessing_common_map_v3": SCRIPTS / "preprocessing_common_map_v3.py",
    "feature_engine": SCRIPTS / "feature_engine.py",
    "map_feature_engine": SCRIPTS / "map_feature_engine.py",
    "team_form_engine": SCRIPTS / "team_form_engine.py",
    "player_roster_feature_engine": SCRIPTS / "player_roster_feature_engine.py",
    "modern_map_feature_engine": SCRIPTS / "modern_map_feature_engine.py",
}

HISTORICAL_STATE = {
    "series_state": DATA / "features" / "pre_cologne_team_state_v1_full.json",
    "map_state": DATA / "interim" / "pre_cologne_map_state_v1.json",
    "form_state": DATA / "interim" / "pre_cologne_form_state_v1.json",
    "roster_state": DATA / "interim" / "pre_cologne_player_roster_state_v1.json",
    "modern_map_state": DATA / "interim" / "pre_cologne_modern_map_state_v1.json",
}

DEPLOYMENT_STATE = {
    "series_state": DATA / "features" / "series_team_state_v1_deployment_post_cologne.json",
    "map_state": DATA / "interim" / "map_state_v1_deployment_post_cologne.json",
    "form_state": DATA / "interim" / "form_state_v1_deployment_post_cologne.json",
    "roster_state": DATA / "interim" / "player_roster_state_v1_deployment_post_cologne.json",
    "modern_map_state": DATA / "interim" / "modern_map_state_v1_deployment_post_cologne.json",
}

DEPLOYMENT_RECEIPT = DATA / "deployment" / "deployment_state_receipt_v1.json"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_group(paths_dict):
    return {name: sha256_file(p) for name, p in paths_dict.items()}


def build_contexts_registry():
    return {
        "registry_version": "application_inference_contexts_v1",
        "note": "Explicit version registry. list_inference_contexts() reads ONLY this file - "
                "no filesystem 'latest' scanning. Each context's hashes make a prediction "
                "reproducible from versioned artifacts, not just an identified model file.",
        "rf_unknown_map_pipeline": hash_group(RF_PIPELINE),
        "xgb_known_map_pipeline": hash_group(XGB_PIPELINE),
        "contexts": {
            "historical_cologne_pre_event": {
                "purpose": "Reproduce exactly what the frozen system knew immediately before "
                           "IEM Cologne Major 2026 began (Phase 8D parity).",
                "classification": "historical_replay",
                "state_cutoff": "2026-06-02T13:30:00",
                "state_cutoff_source": "the frozen Cologne cutoff used in Phase 8D "
                                        "(map_stream_common.cologne_cutoff(), see Phase 8D.1)",
                "prediction_datetime_policy": "locked - any override is rejected "
                                               "(historical_context_datetime_locked)",
                "freshness_label": "frozen_historical_snapshot",
                "state_hashes": hash_group(HISTORICAL_STATE),
            },
            "deployment_post_cologne_v1": {
                "purpose": "Latest locally available application prediction state - includes "
                           "official Cologne 2026 results and legitimate post-Cologne matches.",
                "classification": "deployment",
                "state_cutoff": "2026-06-28T20:00:00",
                "state_cutoff_source": "data/deployment/deployment_state_receipt_v1.json "
                                        "(Phase 9A, cutoffs.deployment_history_cutoff)",
                "prediction_datetime_policy": "defaults to state_cutoff; earlier is rejected "
                                               "(prediction_datetime_before_state_contract); "
                                               "later is accepted only as "
                                               "hypothetical_future_from_stale_snapshot",
                "freshness_label": "deployment_post_cologne_v1 (NOT live, NOT current - "
                                    "latest locally available historical state through "
                                    "2026-06-28)",
                "state_hashes": hash_group(DEPLOYMENT_STATE),
                "phase9a_deployment_receipt_hash": sha256_file(DEPLOYMENT_RECEIPT),
            },
        },
    }


def main():
    registry = build_contexts_registry()
    out_path = CONFIG / "application_inference_contexts_v1.yaml"
    out_path.write_text(yaml.safe_dump(registry, sort_keys=True, default_flow_style=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return registry


if __name__ == "__main__":
    main()
