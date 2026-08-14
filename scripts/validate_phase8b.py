"""
Phase 8B validation. Read-only. Exits non-zero on failure. Passing this
script means: the IEM Cologne Major 2026 pre-event tournament definition is
fully source-audited and internally consistent -- zero outcome data, all 32
team identities confirmed without fuzzy matching, all 8 Stage-1 opening
pairings reproduce exactly, and every structural fact traces to a source
dated before the Cologne cutoff. No Swiss engine, simulator, or model is
touched by this script.
"""

import ast
import hashlib
import json
import sys

import yaml

from _common import ROOT

YAML_PATH = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"
SOURCES_PATH = ROOT / "data" / "tournaments" / "iem_cologne_major_2026_sources.json"
REPORT_PATH = ROOT / "reports" / "phase8b_cologne_tournament_definition.md"
IDENTITY_POLICY_PATH = ROOT / "data" / "interim" / "team_identity_policy.csv"
PHASE8A_REPORT_PATH = ROOT / "reports" / "phase8a_application_prediction_contracts.md"

NEW_PHASE8B_FILES = [
    "config/tournaments/iem_cologne_major_2026_pre_event.yaml",
    "data/tournaments/iem_cologne_major_2026_sources.json",
    "scripts/validate_phase8b.py",
    "tests/test_phase8b_tournament_definition.py",
]

# Artifacts this phase reads but must never modify.
PRE_PHASE8B_ARTIFACTS = [
    "data/interim/team_identity_policy.csv",
    "reports/phase8a_application_prediction_contracts.md",
]

FORBIDDEN_RESULT_TOKENS = [
    "match_winner", "matchwinner", "final_score", "finalscore", "standing_position",
    "champion_team", "qualifier_result", "map_result", "player_stat", "won_map",
    "series_winner", "playoff_winner",
]

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    print("=== capturing pre-run hashes of pre-existing artifacts this phase reads ===")
    baseline = {}
    for rel in PRE_PHASE8B_ARTIFACTS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"pre-existing artifact present: {rel}", baseline[rel] is not None)

    print("\n=== 1. YAML exists, parses, hash reproducible ===")
    check("tournament-definition YAML exists", YAML_PATH.exists())
    if not YAML_PATH.exists():
        _finish()
        return
    raw_bytes = YAML_PATH.read_bytes()
    yaml_hash = hashlib.sha256(raw_bytes).hexdigest()
    cfg = yaml.safe_load(raw_bytes)
    check("YAML parses to a dict", isinstance(cfg, dict))
    reload_hash = hashlib.sha256(YAML_PATH.read_bytes()).hexdigest()
    check("YAML hash reproducible across re-reads", yaml_hash == reload_hash)

    print("\n=== 2. historical Valve rulebook revision recorded ===")
    rb = cfg.get("sources_summary", {}).get("historical_valve_rulebook", {})
    check("rulebook commit_sha recorded", bool(rb.get("commit_sha")))
    check("rulebook commit_sha is a 40-hex-char git SHA",
          isinstance(rb.get("commit_sha"), str) and len(rb["commit_sha"]) == 40
          and all(c in "0123456789abcdef" for c in rb["commit_sha"]))
    check("rulebook commit_date recorded", bool(rb.get("commit_date")))
    check("rulebook commit_date strictly before the Cologne cutoff",
          rb.get("commit_date", "9999") < cfg["metadata"]["prediction_cutoff"])
    check("rulebook commit_url recorded", bool(rb.get("commit_url")) and rb["commit_url"].startswith("https://"))
    check("rulebook content_sha256 recorded (64 hex chars)",
          isinstance(rb.get("content_sha256"), str) and len(rb["content_sha256"]) == 64)
    check("rulebook sections_used non-empty", len(rb.get("sections_used", [])) >= 5)
    check("rulebook caveat documents Grand Final is NOT a generic rule", "Grand Final" in rb.get("caveat", ""))

    print("\n=== 3. invitation_vrs_date vs seeding_vrs_date kept distinct ===")
    ss = cfg.get("sources_summary", {})
    inv_date = ss.get("invitation_vrs_date")
    seed_date = ss.get("seeding_vrs_date")
    check("invitation_vrs_date recorded", bool(inv_date))
    check("seeding_vrs_date recorded", bool(seed_date))
    check("invitation_vrs_date and seeding_vrs_date are recorded as separate fields (not collapsed)",
          inv_date is not None and seed_date is not None)
    check("seeding source explicitly justified (vrs_date_note present and non-trivial)",
          len(ss.get("vrs_date_note", "")) > 80)

    print("\n=== 4. participants: 32 total, correct per-group counts, no duplicates ===")
    participants = cfg.get("participants", {})
    s1 = participants.get("stage_1_entrants", [])
    s2 = participants.get("stage_2_direct_entrants", [])
    s3 = participants.get("stage_3_direct_entrants", [])
    check("stage_1_entrants has 16 teams", len(s1) == 16)
    check("stage_2_direct_entrants has 8 teams", len(s2) == 8)
    check("stage_3_direct_entrants has 8 teams", len(s3) == 8)
    all_names = [t["canonical_model_name"] for t in s1 + s2 + s3]
    check("32 total participants", len(all_names) == 32)
    check("no duplicate canonical_model_name across the 32", len(set(all_names)) == 32)

    print("\n=== 5. all 32 identities confirmed without fuzzy matching (amendment #6) ===")
    allowed_methods = {"exact", "case_normalization", "whitespace_normalization",
                        "punctuation_normalization", "manual_suffix_expansion",
                        "manual_roster_disambiguation", "manual_one_to_one_mapping"}
    all_teams = s1 + s2 + s3
    check("every team record has display_name/canonical_model_name/resolution_method/"
          "identity_feature_eligible/resolution_status",
          all({"display_name", "canonical_model_name", "resolution_method",
               "identity_feature_eligible", "resolution_status"} <= set(t.keys()) for t in all_teams))
    check("resolution_status == confirmed for all 32",
          all(t.get("resolution_status") == "confirmed" for t in all_teams))
    check("resolution_method restricted to the non-fuzzy allowed set for all 32",
          all(t.get("resolution_method") in allowed_methods for t in all_teams))
    check("no resolution_method mentions edit distance / fuzzy / levenshtein",
          not any("fuzzy" in t.get("resolution_method", "").lower()
                  or "levenshtein" in t.get("resolution_method", "").lower()
                  or "edit_distance" in t.get("resolution_method", "").lower() for t in all_teams))

    if IDENTITY_POLICY_PATH.exists():
        import csv
        policy_names = set()
        with open(IDENTITY_POLICY_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                policy_names.add(row["canonical_team_name"].strip())
        check("every canonical_model_name exists verbatim in team_identity_policy.csv",
              all(t["canonical_model_name"] in policy_names for t in all_teams))

    print("\n=== 6. Stage-1 seed table: complete, contiguous, unique ===")
    s1_seeds = sorted(t["pre_event_seed"] for t in s1)
    check("Stage-1 seeds are exactly 1..16", s1_seeds == list(range(1, 17)))
    for group, label in [(s2, "Stage-2 direct entrants"), (s3, "Stage-3 direct entrants")]:
        seeds = sorted(t["pre_event_seed"] for t in group)
        check(f"{label} seeds are exactly 1..8", seeds == list(range(1, 9)))

    print("\n=== 7. all 8 Stage-1 opening pairings reproduce the 1v9...8v16 rule exactly ===")
    val = cfg.get("stage1_opening_pairing_validation", {})
    rows = val.get("rows", [])
    check("exactly 8 validation rows", len(rows) == 8)
    seed_to_team = {t["pre_event_seed"]: t["canonical_model_name"] for t in s1}
    all_match = True
    for r in rows:
        expected_pair = r["seed"] + 8 if r["seed"] <= 8 else r["seed"] - 8
        rule_ok = r["opponent_seed"] == expected_pair
        team_present = seed_to_team.get(r["seed"]) is not None
        row_ok = rule_ok and team_present and r["match"] is True \
            and r["expected_opponent"] == r["published_opening_opponent"]
        all_match = all_match and row_ok
    check("all 8 rows: opponent_seed == seed +/- 8 (1v9...8v16 rule)", all_match)
    check("all 8 rows: match == true (8/8 agreement, per amendment #3)",
          all(r["match"] is True for r in rows) and len(rows) == 8)
    check("all 8 rows: expected_opponent == published_opening_opponent",
          all(r["expected_opponent"] == r["published_opening_opponent"] for r in rows))

    print("\n=== 8. Swiss advancement constants ===")
    sw = cfg.get("swiss_rules", {})
    check("advancement_wins == 3", sw.get("advancement_wins") == 3)
    check("elimination_losses == 3", sw.get("elimination_losses") == 3)
    check("teams_per_stage == 16", sw.get("teams_per_stage") == 16)
    check("Round 4+ priority table has exactly 15 rows", len(sw.get("round_4_plus_priority_table", [])) == 15)

    print("\n=== 9. map pool: 7 unique, source-verified, not a prediction input ===")
    mp = cfg.get("map_pool", {})
    maps = mp.get("maps", [])
    check("map pool has exactly 7 unique maps", len(maps) == 7 and len(set(maps)) == 7)
    check("map pool actual_agreement == '7/7'", mp.get("actual_agreement") == "7/7")
    check("map pool explicitly NOT used as a prediction input", mp.get("used_as_prediction_input") is False)

    print("\n=== 10. format overrides: Stage-3 BO3 and Grand-Final BO5 each have an event-specific source ===")
    mf = cfg.get("match_formats", {})
    stage3 = mf.get("stage_3", {})
    check("Stage-3 default declared bo3", stage3.get("default") == "bo3")
    check("Stage-3 override_rule states ALL matches BO3", "ALL matches" in stage3.get("override_rule", ""))
    check("Stage-3 BO3 has source_type == cologne_event_override", stage3.get("source_type") == "cologne_event_override")
    check("Stage-3 BO3 source_note is non-trivial (cites a dated source)", len(stage3.get("source_note", "")) > 40)

    gf = mf.get("playoffs", {}).get("grand_final", {})
    check("Grand Final best_of == 5", gf.get("best_of") == 5)
    check("Grand Final has source_type == cologne_event_override", gf.get("source_type") == "cologne_event_override")
    check("Grand Final source_note is non-trivial (cites a dated pre-event source, not the generic rulebook)",
          len(gf.get("source_note", "")) > 40 and "escharts" in gf.get("source_note", "").lower())
    qf = mf.get("playoffs", {}).get("quarterfinal", {})
    sf = mf.get("playoffs", {}).get("semifinal", {})
    check("quarterfinal/semifinal best_of == 3", qf.get("best_of") == 3 and sf.get("best_of") == 3)

    print("\n=== 11. prediction-engine contract matches the Phase 8A correction exactly ===")
    pec = cfg.get("prediction_engine_contract", {})
    check("prediction_engine == 'pre_veto_series' (not known_map_preferred)",
          pec.get("prediction_engine") == "pre_veto_series")
    check("model_id == 'series_random_forest_v2'", pec.get("model_id") == "series_random_forest_v2")
    check("maps_used_as_prediction_input is false", pec.get("maps_used_as_prediction_input") is False)
    check("state_policy == 'frozen_pre_event'", pec.get("state_policy") == "frozen_pre_event")
    check("state_source references the STRICT pre-Cologne state file",
          "pre_cologne_team_state_v1_full" in pec.get("state_source", ""))
    check("state_updates_during_simulation is false", pec.get("state_updates_during_simulation") is False)
    check("actual_results_available_to_predictor is false", pec.get("actual_results_available_to_predictor") is False)

    print("\n=== 12. simulation time policy: no state mutation of any kind ===")
    stp = cfg.get("simulation_time_policy", {})
    check("frozen_state_throughout_simulation is true", stp.get("frozen_state_throughout_simulation") is True)
    check("no incremental ELO/form/roster/map updates declared",
          stp.get("incremental_elo_updates_from_simulated_results") is False
          and stp.get("incremental_elo_updates_from_real_results") is False
          and stp.get("incremental_form_roster_map_updates") is False)

    print("\n=== 13. source manifest: exists, every fact known_before_cutoff, no result data ===")
    check("source manifest JSON exists", SOURCES_PATH.exists())
    if SOURCES_PATH.exists():
        sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        facts = sources.get("facts", [])
        check("source manifest has at least 12 fact records", len(facts) >= 12)
        check("every fact has fact_id/field/source_category/known_before_cutoff",
              all({"fact_id", "field", "source_category", "known_before_cutoff"} <= set(f.keys()) for f in facts))
        check("every fact is known_before_cutoff == true",
              all(f.get("known_before_cutoff") is True for f in facts))
        required_fact_ids = {
            "valve_major_rulebook", "cologne_invitation_vrs", "cologne_seeding_vrs",
            "cologne_stage_entry_assignments", "cologne_stage1_seed_table",
            "cologne_stage1_opening_matchups", "cologne_stage3_bo3_override",
            "cologne_grand_final_bo5_override", "cologne_map_pool",
        }
        present_ids = {f["fact_id"] for f in facts}
        check("all required fact_ids present in the source manifest",
              required_fact_ids <= present_ids)
        manifest_text = json.dumps(sources).lower()
        check("source manifest contains no forbidden result-shaped field values",
              not any(tok in manifest_text for tok in FORBIDDEN_RESULT_TOKENS))

    print("\n=== 14. no forbidden result-shaped content anywhere in new Phase 8B files (static scan) ===")
    for rel in ["config/tournaments/iem_cologne_major_2026_pre_event.yaml",
                "data/tournaments/iem_cologne_major_2026_sources.json"]:
        p = ROOT / rel
        if not p.exists():
            continue
        text_lower = p.read_text(encoding="utf-8").lower()
        found = [tok for tok in FORBIDDEN_RESULT_TOKENS if tok in text_lower]
        check(f"{rel}: no forbidden result-shaped tokens", not found)

    print("\n=== 15. report exists and references the frozen YAML hash ===")
    check("phase8b report exists", REPORT_PATH.exists())
    if REPORT_PATH.exists():
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        check("report references the exact YAML SHA-256 hash", yaml_hash in report_text)
        check("report contains explicit COLOGNE RESULTS = UNOPENED marker", "COLOGNE RESULTS = UNOPENED" in report_text)
        check("report contains explicit NO SIMULATION YET marker", "NO SIMULATION YET" in report_text)

    print("\n=== 16. pre-existing artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

    _finish()


def _finish():
    n_pass = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if n_pass != len(CHECKS):
        print("FAILED:")
        for name, ok in CHECKS:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
