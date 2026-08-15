"""
Shared Phase 9A helpers. Two SEPARATE hash surfaces, kept deliberately
distinct per the approved amendments:

  HISTORICAL_REPLAY_IMMUTABLE_RECORD - everything from Phase 4/5/6/7/8B/8C/
  8D/8D.1/8E that must NEVER change. Re-verified byte-unchanged before and
  after every Phase 9A build.

  DEPLOYMENT_BUILD_INPUTS - every real data/code file that can materially
  change a deployment state's resulting hash (amendment #6). Hashed INTO
  the deployment receipt as provenance, not as an immutability check (these
  files are expected to exist and be read, not to be frozen artifacts of
  a prior phase).

Naming follows amendment #1: `pre_cologne` (historical replay) vs
`deployment_post_cologne` (deployment state) - never "Mode C".
"""

from _common import ROOT
import pre_veto_series_predictor as pvp

sha256_file = pvp.sha256_file

DATA = ROOT / "data"
EVAL = DATA / "evaluation"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
MODELING = DATA / "modeling"
MODELS = ROOT / "models"
DEPLOY = DATA / "deployment"
SCRIPTS = ROOT / "scripts"

# ---------------------------------------------------------------------------
# Historical replay: the pre-Cologne / Phase 8D-8E immutable record.
# Superset of phase8e_common.IMMUTABLE_PRE_EVENT_HASH_INPUTS (kept
# independent rather than imported, so phase8e_common.py itself is never
# touched by Phase 9A) plus the 4 non-series pre-Cologne state files, the
# known-map XGB V3 model, and the sealed research split/protocol/config
# surface named in amendment #16.
# ---------------------------------------------------------------------------
HISTORICAL_REPLAY_IMMUTABLE_RECORD = {
    # Phase 8D / 8B / 8C / 8D.1 (mirrors phase8e_common.py's list exactly)
    "phase8d_protocol": ROOT / "config" / "phase8d_cologne_pre_event_simulation_protocol.yaml",
    "phase8d_probability_matrix": EVAL / "cologne_2026_pre_event_matchup_probabilities_v1.parquet",
    "phase8d_probability_receipt": EVAL / "cologne_2026_pre_event_probability_receipt_v1.json",
    "phase8d_simulation_receipt": EVAL / "cologne_2026_pre_event_simulation_receipt_v1.json",
    "phase8d_team_probabilities": EVAL / "cologne_2026_pre_event_team_probabilities_v1.csv",
    "phase8d_swiss_record_distributions": EVAL / "cologne_2026_pre_event_swiss_record_distributions_v1.csv",
    "phase8d_playoff_seed_distributions": EVAL / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv",
    "phase8d_matchup_frequencies": EVAL / "cologne_2026_pre_event_matchup_frequencies_v1.csv",
    "phase8d_favorite_path": EVAL / "cologne_2026_pre_event_favorite_path_v1.json",
    "phase8b_tournament_yaml": ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml",
    "phase8c_tournament_engine": SCRIPTS / "tournament_engine.py",
    "rf_v2_model": MODELS / "random_forest_v2.joblib",
    "strict_pre_cologne_series_state": FEATURES / "pre_cologne_team_state_v1_full.json",
    "phase8d1_provenance_report": ROOT / "reports" / "phase8d1_cutoff_timestamp_provenance.md",
    "phase8d1_provenance_json": EVAL / "cologne_2026_pre_event_cutoff_provenance_v1.json",
    # Phase 8E
    "phase8e_protocol": ROOT / "config" / "phase8e_cologne_simulation_vs_reality_protocol.yaml",
    "phase8e_reconciliation": EVAL / "cologne_2026_result_reconciliation_v1.csv",
    "phase8e_canonical_actual_results": EVAL / "cologne_2026_actual_series_results_v1.parquet",
    "phase8e_actual_match_predictions": EVAL / "cologne_2026_actual_match_predictions_v1.parquet",
    "phase8e_milestone_comparison": EVAL / "cologne_2026_actual_milestone_comparison_v1.csv",
    "phase8e_summary": EVAL / "cologne_2026_simulation_vs_reality_summary_v1.json",
    "phase8e_receipt": EVAL / "cologne_2026_simulation_vs_reality_receipt_v1.json",
    # the other 4 pre-Cologne state snapshots (series' own is listed above)
    "strict_pre_cologne_map_state": INTERIM / "pre_cologne_map_state_v1.json",
    "strict_pre_cologne_form_state": INTERIM / "pre_cologne_form_state_v1.json",
    "strict_pre_cologne_roster_state": INTERIM / "pre_cologne_player_roster_state_v1.json",
    "strict_pre_cologne_modern_map_state": INTERIM / "pre_cologne_modern_map_state_v1.json",
    # known-map XGB V3 (the other frozen model family, amendment #16/#18)
    "known_map_xgb_v3_model": MODELS / "map_xgboost_v3_final.json",
    "known_map_xgb_v3_metadata": MODELS / "map_xgboost_v3_final_metadata.json",
    # sealed research splits/CV folds (amendment #16)
    "series_split_v1": MODELING / "series_split_v1.csv",
    "map_split_v1": MODELING / "map_split_v1.csv",
    "map_cv_folds_v1": MODELING / "map_cv_folds_v1.csv",
    "random_forest_cv_folds_v2": MODELING / "random_forest_cv_folds_v2.csv",
    # sealed Phase 7 TEST artifacts (amendment #16)
    "phase7_test_protocol": EVAL / "phase7_test_protocol_v1.json",
    "phase7_test_open_receipt": EVAL / "phase7_test_open_receipt_v1.json",
    "map_test_predictions": EVAL / "map_test_predictions_v1.parquet",
    # frozen model-selection configs (amendment #16)
    "random_forest_v2_selected_config": MODELING / "random_forest_v2_selected_config.json",
    "xgboost_v2_selected_config": MODELING / "xgboost_v2_selected_config.json",
    "logistic_regression_v2_selected_config": MODELING / "logistic_regression_v2_selected_config.json",
    "map_random_forest_v1_selected_config": MODELING / "map_random_forest_v1_selected_config.json",
    "map_xgboost_v1_selected_config": MODELING / "map_xgboost_v1_selected_config.json",
    "map_xgboost_v3_final_selected_config": MODELING / "map_xgboost_v3_final_selected_config.json",
}


def hash_historical_replay_record():
    return {name: sha256_file(p) for name, p in HISTORICAL_REPLAY_IMMUTABLE_RECORD.items()}


def verify_historical_replay_record(expected_hashes):
    actual = hash_historical_replay_record()
    return [(name, expected, actual.get(name)) for name, expected in expected_hashes.items()
            if actual.get(name) != expected]


# ---------------------------------------------------------------------------
# Deployment build inputs (amendment #6) - hashed for provenance, not for an
# immutability check (these are legitimately read, not frozen).
# ---------------------------------------------------------------------------
DEPLOYMENT_BUILD_DATA_INPUTS = {
    "evaluation_manifest": INTERIM / "evaluation_manifest.csv",
    "series_base": INTERIM / "series_base.parquet",
    "map_base": INTERIM / "map_base.parquet",
    "rejected_series_rows": INTERIM / "rejected_series_rows.csv",
    "team_identity_policy": INTERIM / "team_identity_policy.csv",
    "series_team_form_states_v1": FEATURES / "series_team_form_states_v1.parquet",
    "phase8e_reconciliation": EVAL / "cologne_2026_result_reconciliation_v1.csv",
    "phase8e_canonical_actual_results": EVAL / "cologne_2026_actual_series_results_v1.parquet",
}

DEPLOYMENT_BUILD_CODE_INPUTS = {
    "feature_engine": SCRIPTS / "feature_engine.py",
    "build_series_features_v1": SCRIPTS / "build_series_features_v1.py",
    "map_feature_engine": SCRIPTS / "map_feature_engine.py",
    "map_stream_common": SCRIPTS / "map_stream_common.py",
    "team_form_engine": SCRIPTS / "team_form_engine.py",
    "team_form_stream_common": SCRIPTS / "team_form_stream_common.py",
    "player_roster_feature_engine": SCRIPTS / "player_roster_feature_engine.py",
    "player_roster_stream_common": SCRIPTS / "player_roster_stream_common.py",
    "modern_map_feature_engine": SCRIPTS / "modern_map_feature_engine.py",
    "modern_map_stream_common": SCRIPTS / "modern_map_stream_common.py",
    "phase9a_common": SCRIPTS / "phase9a_common.py",
    "build_deployment_history_manifest": SCRIPTS / "build_deployment_history_manifest.py",
    "build_deployment_series_state": SCRIPTS / "build_deployment_series_state.py",
    "build_deployment_map_state": SCRIPTS / "build_deployment_map_state.py",
    "build_deployment_form_state": SCRIPTS / "build_deployment_form_state.py",
    "build_deployment_roster_state": SCRIPTS / "build_deployment_roster_state.py",
    "build_deployment_modern_map_state": SCRIPTS / "build_deployment_modern_map_state.py",
}


def hash_deployment_build_inputs():
    return {
        "data": {name: sha256_file(p) for name, p in DEPLOYMENT_BUILD_DATA_INPUTS.items()},
        "code": {name: sha256_file(p) for name, p in DEPLOYMENT_BUILD_CODE_INPUTS.items()},
    }


NEW_DEPLOYMENT_ARTIFACTS = {
    "deployment_history_manifest": DEPLOY / "deployment_history_manifest_v1.parquet",
    "deployment_state_consumption_audit": DEPLOY / "deployment_state_consumption_audit_v1.csv",
    "series_state": FEATURES / "series_team_state_v1_deployment_post_cologne.json",
    "map_state": INTERIM / "map_state_v1_deployment_post_cologne.json",
    "form_state": INTERIM / "form_state_v1_deployment_post_cologne.json",
    "roster_state": INTERIM / "player_roster_state_v1_deployment_post_cologne.json",
    "modern_map_state": INTERIM / "modern_map_state_v1_deployment_post_cologne.json",
}
