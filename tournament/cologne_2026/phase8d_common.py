"""
Shared Phase 8D helpers: loading the real 32 Cologne teams from the frozen
Phase 8B tournament YAML. This is the ONE place `participants` is read for
simulation purposes (unlike tournament_engine.py, which never reads it at
all - see Phase 8C). Every other Phase 8D module imports from here rather
than re-parsing the YAML.
"""

import yaml

from _common import ROOT
import tournament.engine.tournament_engine as te

TOURNAMENT_YAML_PATH = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"


def load_tournament_yaml():
    raw = TOURNAMENT_YAML_PATH.read_bytes()
    import hashlib
    actual = hashlib.sha256(raw).hexdigest()
    if actual != te.FROZEN_YAML_SHA256:
        raise ValueError(f"frozen tournament YAML hash mismatch: expected {te.FROZEN_YAML_SHA256}, got {actual}")
    return yaml.safe_load(raw)


def load_cologne_teams():
    """Returns a flat list of 32 dicts (one per real Cologne team), each with
    display_name, canonical_model_name, starting_stage, pre_event_seed,
    resolution_status, identity_feature_eligible."""
    cfg = load_tournament_yaml()
    p = cfg["participants"]
    teams = []
    for group, stage_label in [("stage_1_entrants", "stage_1"),
                                ("stage_2_direct_entrants", "stage_2"),
                                ("stage_3_direct_entrants", "stage_3")]:
        for t in p[group]:
            teams.append({
                "display_name": t["display_name"],
                "canonical_model_name": t["canonical_model_name"],
                "starting_stage": stage_label,
                "pre_event_seed": t["pre_event_seed"],
                "resolution_status": t["resolution_status"],
                "identity_feature_eligible": t["identity_feature_eligible"],
            })
    if len(teams) != 32:
        raise ValueError(f"expected 32 Cologne teams, got {len(teams)}")
    return teams


def build_cologne_entrants():
    """Returns (stage1_entrants, stage2_direct_entrants, stage3_direct_entrants)
    as tournament_engine.TeamEntry lists, team_id == canonical_model_name."""
    teams = load_cologne_teams()
    by_stage = {"stage_1": [], "stage_2": [], "stage_3": []}
    for t in teams:
        by_stage[t["starting_stage"]].append(t)
    entrants = {}
    for stage, group in by_stage.items():
        ordered = sorted(group, key=lambda t: t["pre_event_seed"])
        entrants[stage] = [te.TeamEntry(team_id=t["canonical_model_name"], display_name=t["display_name"],
                                         initial_stage_seed=t["pre_event_seed"]) for t in ordered]
    return entrants["stage_1"], entrants["stage_2"], entrants["stage_3"]
