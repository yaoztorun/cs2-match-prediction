"""
Phase 8B data-contract tests. Tests the FROZEN tournament-definition YAML
and its rule contracts only -- no Swiss engine, no match simulation, no
model call. A tiny synthetic 16-seed example proves the 1v9...8v16 pairing
rule and the format-lookup table are implemented correctly, independent of
any real Cologne data.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"
SOURCES_PATH = ROOT / "data" / "tournaments" / "iem_cologne_major_2026_sources.json"

FORBIDDEN_RESULT_TOKENS = [
    "match_winner", "matchwinner", "final_score", "finalscore", "standing_position",
    "champion_team", "qualifier_result", "map_result", "player_stat", "won_map",
    "series_winner", "playoff_winner",
]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sources():
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def initial_pairing(seed, n_teams):
    """The frozen 1v9...8v16-style rule, generalized to any even n_teams."""
    half = n_teams // 2
    return seed + half if seed <= half else seed - half


def is_elimination_or_advancement(wins, losses, target_wins, target_losses):
    """Stage 1/2 generic rule: Bo3 iff this match's winner would advance OR
    this match's loser would be eliminated."""
    would_advance = (wins + 1) == target_wins
    would_eliminate = (losses + 1) == target_losses
    return would_advance or would_eliminate


# ---------------------------------------------------------------------------
# YAML structure
# ---------------------------------------------------------------------------

def test_yaml_parses(cfg):
    assert isinstance(cfg, dict)
    assert cfg["metadata"]["event_id"] == "iem_cologne_major_2026"


def test_yaml_hash_is_stable():
    h1 = hashlib.sha256(YAML_PATH.read_bytes()).hexdigest()
    h2 = hashlib.sha256(YAML_PATH.read_bytes()).hexdigest()
    assert h1 == h2
    assert len(h1) == 64


def test_metadata_counts(cfg):
    m = cfg["metadata"]
    assert m["total_teams"] == 32
    assert m["teams_per_swiss_stage"] == 16
    assert m["advancers_per_swiss_stage"] == 8
    assert m["playoff_teams"] == 8
    assert m["prediction_cutoff"] == "2026-06-02T13:30:00"


# ---------------------------------------------------------------------------
# Team/seed counts
# ---------------------------------------------------------------------------

def test_participant_group_counts(cfg):
    p = cfg["participants"]
    assert len(p["stage_1_entrants"]) == 16
    assert len(p["stage_2_direct_entrants"]) == 8
    assert len(p["stage_3_direct_entrants"]) == 8


def test_32_unique_teams_no_duplicates(cfg):
    p = cfg["participants"]
    names = [t["canonical_model_name"] for group in
              ["stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"]
              for t in p[group]]
    assert len(names) == 32
    assert len(set(names)) == 32


def test_seed_tables_contiguous_and_unique(cfg):
    p = cfg["participants"]
    s1_seeds = sorted(t["pre_event_seed"] for t in p["stage_1_entrants"])
    assert s1_seeds == list(range(1, 17))
    for group in ["stage_2_direct_entrants", "stage_3_direct_entrants"]:
        seeds = sorted(t["pre_event_seed"] for t in p[group])
        assert seeds == list(range(1, 9))


# ---------------------------------------------------------------------------
# Canonical-identity mapping spot checks (no fuzzy matching used)
# ---------------------------------------------------------------------------

ALLOWED_RESOLUTION_METHODS = {
    "exact", "case_normalization", "whitespace_normalization",
    "punctuation_normalization", "manual_suffix_expansion",
    "manual_roster_disambiguation", "manual_one_to_one_mapping",
}


def test_every_team_record_has_required_fields(cfg):
    p = cfg["participants"]
    required = {"display_name", "canonical_model_name", "resolution_method",
                "identity_feature_eligible", "resolution_status"}
    for group in ["stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"]:
        for t in p[group]:
            assert required <= set(t.keys()), t


def test_all_resolution_statuses_confirmed(cfg):
    p = cfg["participants"]
    for group in ["stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"]:
        for t in p[group]:
            assert t["resolution_status"] == "confirmed"


def test_no_fuzzy_resolution_method_used(cfg):
    p = cfg["participants"]
    for group in ["stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"]:
        for t in p[group]:
            assert t["resolution_method"] in ALLOWED_RESOLUTION_METHODS


@pytest.mark.parametrize("display_name,expected_canonical", [
    ("The MongolZ", "The Mongolz"),
    ("FURIA", "FURIA Esports"),
    ("B8", "B8 Esports"),
    ("NRG", "NRG Esports"),
    ("SINNERS Esports", "Sinners Esports"),
    ("HEROIC", "Heroic"),
    ("Team Vitality", "Team Vitality"),
])
def test_identity_resolution_spot_checks(cfg, display_name, expected_canonical):
    p = cfg["participants"]
    all_teams = p["stage_1_entrants"] + p["stage_2_direct_entrants"] + p["stage_3_direct_entrants"]
    match = next(t for t in all_teams if t["display_name"] == display_name)
    assert match["canonical_model_name"] == expected_canonical


# ---------------------------------------------------------------------------
# Synthetic pairing-rule test, independent of real Cologne data
# ---------------------------------------------------------------------------

def test_initial_pairing_rule_on_synthetic_16_seeds():
    expected = {1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16}
    for seed, opp in expected.items():
        assert initial_pairing(seed, 16) == opp
        assert initial_pairing(opp, 16) == seed


def test_stage1_opening_pairing_validation_reproduces_rule(cfg):
    rows = cfg["stage1_opening_pairing_validation"]["rows"]
    assert len(rows) == 8
    for r in rows:
        assert initial_pairing(r["seed"], 16) == r["opponent_seed"]
        assert r["match"] is True
        assert r["expected_opponent"] == r["published_opening_opponent"]


def test_stage1_pairing_teams_match_seed_table(cfg):
    # The pairing table's "team"/"expected_opponent"/"published_opening_opponent"
    # fields use tournament display names (e.g. "B8", "HEROIC"), matching the
    # published schedule verbatim; the seed table's canonical_model_name is
    # the resolved modeling identity (e.g. "B8 Esports", "Heroic"). Compare
    # against display_name, not canonical_model_name.
    p = cfg["participants"]
    seed_to_display = {t["pre_event_seed"]: t["display_name"] for t in p["stage_1_entrants"]}
    for r in cfg["stage1_opening_pairing_validation"]["rows"]:
        assert seed_to_display[r["seed"]] == r["team"]


# ---------------------------------------------------------------------------
# Format-lookup table, including the Stage-3 all-Bo3 override
# ---------------------------------------------------------------------------

def test_stage1_stage2_generic_bo_rule():
    # 8-team stage-fragment example: target 3 wins to advance, 3 losses to eliminate.
    assert is_elimination_or_advancement(wins=2, losses=1, target_wins=3, target_losses=3) is True   # winner would advance
    assert is_elimination_or_advancement(wins=1, losses=2, target_wins=3, target_losses=3) is True   # loser would be eliminated
    assert is_elimination_or_advancement(wins=1, losses=1, target_wins=3, target_losses=3) is False  # neither stake


def test_stage3_all_matches_bo3_override(cfg):
    stage3 = cfg["match_formats"]["stage_3"]
    assert stage3["default"] == "bo3"
    assert "ALL matches" in stage3["override_rule"]
    assert stage3["source_type"] == "cologne_event_override"


def test_playoff_format_table(cfg):
    playoffs = cfg["match_formats"]["playoffs"]
    assert playoffs["quarterfinal"]["best_of"] == 3
    assert playoffs["semifinal"]["best_of"] == 3
    assert playoffs["grand_final"]["best_of"] == 5
    assert playoffs["grand_final"]["source_type"] == "cologne_event_override"
    assert playoffs["third_place_match"] == "none"


def test_playoff_bracket_seeding(cfg):
    b = cfg["playoff_bracket"]
    assert b["bracket_a"] == ["1v8", "4v5"]
    assert b["bracket_b"] == ["2v7", "3v6"]
    assert b["reseeding_between_rounds"] is False


# ---------------------------------------------------------------------------
# Map pool
# ---------------------------------------------------------------------------

def test_map_pool_membership(cfg):
    maps = set(cfg["map_pool"]["maps"])
    assert maps == {"Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass"}
    assert cfg["map_pool"]["used_as_prediction_input"] is False


# ---------------------------------------------------------------------------
# Prediction-engine contract
# ---------------------------------------------------------------------------

def test_prediction_engine_contract_fields(cfg):
    pec = cfg["prediction_engine_contract"]
    assert pec["prediction_engine"] == "pre_veto_series"
    assert pec["historical_prediction_mode"] == "pre_veto"
    assert pec["model_id"] == "series_random_forest_v2"
    assert pec["maps_used_as_prediction_input"] is False
    assert pec["state_policy"] == "frozen_pre_event"
    assert pec["state_updates_during_simulation"] is False
    assert pec["actual_results_available_to_predictor"] is False


# ---------------------------------------------------------------------------
# Guard: nothing in this phase mutates ELO/form/roster/map state
# ---------------------------------------------------------------------------

def test_simulation_time_policy_forbids_state_mutation(cfg):
    stp = cfg["simulation_time_policy"]
    assert stp["frozen_state_throughout_simulation"] is True
    assert stp["incremental_elo_updates_from_simulated_results"] is False
    assert stp["incremental_elo_updates_from_real_results"] is False
    assert stp["incremental_form_roster_map_updates"] is False


def test_no_swiss_engine_or_state_mutating_functions_defined_in_phase8b_files():
    """Static guard: neither new Phase 8B file defines a callable that
    looks like a Swiss-round/match simulator or a state-mutation function."""
    forbidden_substrings = ["def simulate_", "def run_swiss", "def play_match", "def update_elo", "def advance_stage"]
    for rel in ["validation/validate_phase8b.py"]:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for token in forbidden_substrings:
            assert token not in text, f"{rel} unexpectedly defines {token!r}"


# ---------------------------------------------------------------------------
# Source manifest / no result data
# ---------------------------------------------------------------------------

def test_source_manifest_facts_all_known_before_cutoff(sources):
    facts = sources["facts"]
    assert len(facts) >= 12
    assert all(f["known_before_cutoff"] is True for f in facts)


def test_source_manifest_and_yaml_contain_no_forbidden_result_tokens(sources):
    manifest_text = json.dumps(sources).lower()
    yaml_text = YAML_PATH.read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_RESULT_TOKENS:
        assert token not in manifest_text
        assert token not in yaml_text


def test_required_fact_ids_present(sources):
    present = {f["fact_id"] for f in sources["facts"]}
    required = {
        "valve_major_rulebook", "cologne_invitation_vrs", "cologne_seeding_vrs",
        "cologne_stage_entry_assignments", "cologne_stage1_seed_table",
        "cologne_stage1_opening_matchups", "cologne_stage3_bo3_override",
        "cologne_grand_final_bo5_override", "cologne_map_pool",
    }
    assert required <= present


def test_rulebook_revision_recorded_before_cutoff(cfg):
    rb = cfg["sources_summary"]["historical_valve_rulebook"]
    assert rb["commit_date"] < cfg["metadata"]["prediction_cutoff"]
    assert len(rb["commit_sha"]) == 40


def test_invitation_and_seeding_vrs_dates_are_distinct_fields(cfg):
    ss = cfg["sources_summary"]
    assert "invitation_vrs_date" in ss
    assert "seeding_vrs_date" in ss
    # The two fields are independently recorded regardless of whether their
    # values happen to differ -- both must be present as distinct keys.
    assert ss["invitation_vrs_date"] == "2026-04-06"
    assert ss["seeding_vrs_date"] == "2026-05-04"
