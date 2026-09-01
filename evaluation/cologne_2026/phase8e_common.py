"""
Shared Phase 8E helpers: the immutable pre-event record (every frozen Phase
8D/8C/8B/8D.1 artifact that must never change once Phase 8E starts opening
actual Cologne results), and the sha256 machinery to verify it. Every other
Phase 8E module imports IMMUTABLE_PRE_EVENT_HASH_INPUTS from here rather than
re-listing paths.
"""

from _common import ROOT
import tournament.simulation.pre_veto_series_predictor as pvp  # reuse sha256_file, not redefine it

sha256_file = pvp.sha256_file

EVALUATION_DIR = ROOT / "data" / "evaluation"
TOURNAMENTS_DIR = ROOT / "data" / "tournaments"
FIGURES_DIR = ROOT / "reports" / "figures" / "phase8e"

# The 14-item immutable pre-event record named in the approved Phase 8E plan.
# Phase 8D.1's "provenance artifact" bullet is two files (report + json); both
# are hashed under their own keys so nothing is bundled/hidden.
IMMUTABLE_PRE_EVENT_HASH_INPUTS = {
    "phase8d_protocol": ROOT / "config" / "evaluation" / "phase8d_cologne_pre_event_simulation_protocol.yaml",
    "phase8d_probability_matrix": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_matchup_probabilities_v1.parquet",
    "phase8d_probability_receipt": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_probability_receipt_v1.json",
    "phase8d_simulation_receipt": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_simulation_receipt_v1.json",
    "phase8d_team_probabilities": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_team_probabilities_v1.csv",
    "phase8d_swiss_record_distributions": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_swiss_record_distributions_v1.csv",
    "phase8d_playoff_seed_distributions": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv",
    "phase8d_matchup_frequencies": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_matchup_frequencies_v1.csv",
    "phase8d_favorite_path": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_favorite_path_v1.json",
    "phase8b_tournament_yaml": ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml",
    "phase8c_tournament_engine": ROOT / "tournament" / "engine" / "tournament_engine.py",
    "rf_v2_model": ROOT / "models" / "series" / "random_forest_v2.joblib",
    "strict_pre_cologne_state": ROOT / "data" / "features" / "pre_cologne_team_state_v1_full.json",
    "phase8d1_provenance_report": ROOT / "reports" / "phases" / "phase8d1_cutoff_timestamp_provenance.md",
    "phase8d1_provenance_json": ROOT / "data" / "evaluation" / "cologne_2026_pre_event_cutoff_provenance_v1.json",
}


def hash_immutable_pre_event_record():
    return {name: sha256_file(p) for name, p in IMMUTABLE_PRE_EVENT_HASH_INPUTS.items()}


def verify_immutable_pre_event_record(expected_hashes):
    """Returns a list of (name, expected, actual) tuples for any mismatch. Empty list == unchanged."""
    actual = hash_immutable_pre_event_record()
    mismatches = []
    for name, expected in expected_hashes.items():
        got = actual.get(name)
        if got != expected:
            mismatches.append((name, expected, got))
    return mismatches
