"""
Phase 8C validation (pure tournament-engine gate). Read-only. Exits
non-zero on failure. Passing this script means: the Swiss + playoff
mechanics engine correctly implements every frozen Phase 8B rule, is fully
deterministic, survives 500 synthetic stress runs with zero invariant
failures, and has never imported ML code or touched a real Cologne
participant or result.
"""

import ast
import hashlib
import sys

from _common import ROOT
import tournament_engine as te

# Phase 1-8B artifacts this phase reads (directly or transitively) and must
# never modify. Baseline extended from validate_phase8b.py's own
# PRE_PHASE8B_ARTIFACTS, plus Phase 8B's own outputs.
PRE_PHASE8C_ARTIFACTS = [
    "data/interim/team_identity_policy.csv",
    "reports/phase8a_application_prediction_contracts.md",
    "config/tournaments/iem_cologne_major_2026_pre_event.yaml",
    "data/tournaments/iem_cologne_major_2026_sources.json",
    "reports/phase8b_cologne_tournament_definition.md",
    "scripts/validate_phase8b.py",
    "tests/test_phase8b_tournament_definition.py",
]

FORBIDDEN_IMPORT_NAMES = {
    "joblib", "xgboost", "sklearn", "feature_engine", "map_feature_engine",
    "modern_map_feature_engine", "player_roster_feature_engine", "team_form_engine",
    "preprocessing_common_map_v2", "preprocessing_common_map_v3", "preprocessing_random_forest_map_v3",
    "preprocessing_xgboost_map_v3", "map_stream_common", "map_modeling_common",
}

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _code_only(source):
    """Strip the leading module docstring before scanning for forbidden
    substrings, so prose explaining a guarantee doesn't trip its own check."""
    tree = ast.parse(source)
    start = tree.body[0].end_lineno if (tree.body and isinstance(tree.body[0], ast.Expr)) else 0
    return "\n".join(source.splitlines()[start:])


def main():
    print("=== capturing pre-run hashes of Phase 1-8B artifacts ===")
    baseline = {}
    for rel in PRE_PHASE8C_ARTIFACTS:
        p = ROOT / rel
        baseline[rel] = sha256(p) if p.exists() else None
        check(f"Phase 1-8B artifact present: {rel}", baseline[rel] is not None)

    print("\n=== 1. frozen YAML hash unchanged ===")
    yaml_bytes = (ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml").read_bytes()
    check("frozen tournament YAML hash == e481ca4d...",
          hashlib.sha256(yaml_bytes).hexdigest() == te.FROZEN_YAML_SHA256)

    print("\n=== 2. src/ unchanged (still empty) ===")
    src_dir = ROOT / "src"
    check("src/ remains empty", not any(p.is_file() for p in src_dir.rglob("*")) if src_dir.exists() else True)

    print("\n=== 3. no ML imports, no Cologne-result access (AST/static) ===")
    engine_source = (ROOT / "scripts" / "tournament_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(engine_source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    check("tournament_engine.py imports no ML/feature/preprocessing module",
          not (imported & FORBIDDEN_IMPORT_NAMES))
    code_only = _code_only(engine_source)
    for token in ["evaluation_manifest", "series_base", "map_test_predictions", "cologne_2026"]:
        check(f"tournament_engine.py code never references {token!r}", token not in code_only)
    check("tournament_engine.py never reads the YAML's participants section",
          '"participants"' not in engine_source and "['participants']" not in engine_source)

    demo_path = ROOT / "scripts" / "run_phase8c_synthetic_demo.py"
    if demo_path.exists():
        demo_source = demo_path.read_text(encoding="utf-8")
        check("run_phase8c_synthetic_demo.py never reads participants",
              '"participants"' not in demo_source and "['participants']" not in demo_source)

    print("\n=== 4. Round 1 pairing exact ===")
    entrants = [te.TeamEntry(f"T{i}", f"T{i}", i) for i in range(1, 17)]
    pairs = te.round1_pairing_ids(entrants)
    check("1v9...8v16 exact", pairs == [("T1", "T9"), ("T2", "T10"), ("T3", "T11"), ("T4", "T12"),
                                         ("T5", "T13"), ("T6", "T14"), ("T7", "T15"), ("T8", "T16")])

    print("\n=== 5. Round 2/3 constrained pairing: deterministic dead-end avoidance ===")
    pool4 = [te.TeamStageState(f"T{i}", f"T{i}", i, current_seed=i) for i in range(1, 5)]
    hist = {"T2": {"T4"}, "T4": {"T2"}}
    dead_end_pairs, fb, cnt = te.constrained_pool_pairing(pool4, hist)
    check("dead-end scenario resolves with zero rematches", fb is False and cnt == 0)
    check("dead-end scenario never returns the forbidden rematch",
          ("T2", "T4") not in dead_end_pairs and ("T4", "T2") not in dead_end_pairs)

    print("\n=== 6. Round 4/5 priority-table selection exact ===")
    rules = te.load_frozen_rules()
    pool6 = [te.TeamStageState(f"T{i}", f"T{i}", i, current_seed=i) for i in range(1, 7)]
    pairs6, row, fb6, cnt6 = te.priority_table_pairing(pool6, {}, rules.round4_plus_priority_table)
    check("row 1 selected with no history", row == 1 and fb6 is False and cnt6 == 0)
    check("row 1 pairs == 1v6,2v5,3v4", set(pairs6) == {("T1", "T6"), ("T2", "T5"), ("T3", "T4")})

    print("\n=== 7. dynamic Difficulty Score ===")
    states = {tid: te.TeamStageState(tid, tid, i + 1, current_seed=i + 1)
              for i, tid in enumerate(["A", "B", "C"])}
    states["A"].opponents = ["B", "C"]
    states["B"].wins, states["B"].losses = 1, 1
    states["C"].wins, states["C"].losses = 1, 1
    d1 = te.difficulty_score("A", states)
    states["B"].wins, states["B"].losses = 2, 1
    d2 = te.difficulty_score("A", states)
    check("Difficulty Score recomputes when an opponent's record changes with no new match for A",
          d1 == 0 and d2 == 1 and d1 != d2)

    print("\n=== 8. mid-stage seed order (tie-break chain) ===")
    states2 = {"A": te.TeamStageState("A", "A", 5, current_seed=5, wins=1, losses=0),
               "B": te.TeamStageState("B", "B", 1, current_seed=1, wins=1, losses=0)}
    ordered = te.compute_seed_order(list(states2.values()))
    check("equal record & difficulty -> lower initial seed wins the tie",
          [s.team_id for s in ordered] == ["B", "A"])

    print("\n=== 9. rematch policy is stage-local ===")
    pool4b = [te.TeamStageState(f"X{i}", f"X{i}", i, current_seed=i) for i in range(1, 5)]
    pairs_fresh, fb_fresh, cnt_fresh = te.constrained_pool_pairing(pool4b, {})
    check("a fresh (new-stage) history never restricts pairing",
          ("X1", "X4") in pairs_fresh and fb_fresh is False and cnt_fresh == 0)

    print("\n=== 10. format selection: shared-record assertion ===")
    bad_states = {"A": te.TeamStageState("A", "A", 1, current_seed=1, wins=2, losses=0),
                  "B": te.TeamStageState("B", "B", 2, current_seed=2, wins=1, losses=1)}
    format_assertion_raised = False
    try:
        te._build_swiss_matches([{"id_a": "A", "id_b": "B"}], bad_states, te.STAGE_1, 3, rules)
    except ValueError:
        format_assertion_raised = True
    check("mismatched pre-match records raise before format is derived", format_assertion_raised)

    print("\n=== 11. full synthetic stage: Swiss termination + 33 matches / 8-8 split ===")
    stage_result = te.run_swiss_stage(entrants, te.STAGE_1, rules, te.HigherSeedWinsProvider())
    check("33 matches", len(stage_result.trace) == 33)
    check("8 advanced / 8 eliminated", len(stage_result.advancers) == 8 and len(stage_result.eliminated) == 8)
    round_counts = {}
    for e in stage_result.trace:
        round_counts[e.match.round_number] = round_counts.get(e.match.round_number, 0) + 1
    check("per-round match counts == {1:8,2:8,3:8,4:6,5:3}", round_counts == {1: 8, 2: 8, 3: 8, 4: 6, 5: 3})

    print("\n=== 12. stage-transition seeds 9-16 == final Stage-1 order ===")
    direct = [te.TeamEntry(f"D{i}", f"D{i}", i) for i in range(1, 9)]
    stage2_entrants = te.next_stage_entrants(direct, stage_result)
    check("advancers become seeds 9-16 in final order",
          [e.team_id for e in stage2_entrants[8:]] == [t.team_id for t in stage_result.advancers]
          and [e.initial_stage_seed for e in stage2_entrants[8:]] == list(range(9, 17)))

    print("\n=== 13. playoff bracket / formats ===")
    seeds8 = [te.TeamEntry(f"P{i}", f"P{i}", i) for i in range(1, 9)]
    playoff_result = te.run_playoffs(seeds8, rules, te.HigherSeedWinsProvider())
    check("exactly 7 playoff matches (4 QF + 2 SF + 1 Final)", len(playoff_result.trace) == 7)
    by_id = {e.match.match_id: e.match for e in playoff_result.trace}
    check("QF/SF are Bo3, Final is Bo5",
          all(by_id[f"playoff_qf_{i:02d}"].best_of == 3 for i in range(1, 5))
          and by_id["playoff_sf_01"].best_of == 3 and by_id["playoff_sf_02"].best_of == 3
          and by_id["playoff_final_01"].best_of == 5)
    check("bracket A = seed1v8 + seed4v5, bracket B = seed2v7 + seed3v6",
          {by_id["playoff_qf_01"].seed_a, by_id["playoff_qf_01"].seed_b} == {1, 8}
          and {by_id["playoff_qf_02"].seed_a, by_id["playoff_qf_02"].seed_b} == {4, 5}
          and {by_id["playoff_qf_03"].seed_a, by_id["playoff_qf_03"].seed_b} == {2, 7}
          and {by_id["playoff_qf_04"].seed_a, by_id["playoff_qf_04"].seed_b} == {3, 6})

    print("\n=== 14. deterministic trace (double run, canonical hash) ===")
    s1, s2, s3 = te.build_synthetic_fixture()
    r1 = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    r2 = te.run_major_tournament(s1, s2, s3, rules, te.HigherSeedWinsProvider())
    h1, h2 = te.trace_hash(r1.to_dict()), te.trace_hash(r2.to_dict())
    check("two runs of HigherSeedWinsProvider produce an identical canonical trace hash", h1 == h2)
    check("champion identical across both runs", r1.champion == r2.champion)

    print("\n=== 15. synthetic trace contains zero real Cologne teams ===")
    import json
    import yaml as _yaml
    real_cfg = _yaml.safe_load((ROOT / "config" / "tournaments" /
                                 "iem_cologne_major_2026_pre_event.yaml").read_bytes())
    real_names = set()
    for group in ("stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"):
        for t in real_cfg["participants"][group]:
            real_names.add(t["display_name"])
            real_names.add(t["canonical_model_name"])
    trace_text = json.dumps(r1.to_dict())
    leaked = [n for n in real_names if n in trace_text]
    check("no real Cologne participant name appears in the synthetic trace", not leaked)
    all_synthetic = all(e.match.team_a.split("_")[0] in ("S1", "S2", "S3")
                         and e.match.team_b.split("_")[0] in ("S1", "S2", "S3") for e in r1.full_trace())
    check("every team_id in the trace is synthetic (S1_*/S2_*/S3_*)", all_synthetic)

    trace_path = ROOT / "data" / "tournaments" / "phase8c_synthetic_trace.json"
    if trace_path.exists():
        on_disk = json.loads(trace_path.read_text(encoding="utf-8"))
        leaked_disk = [n for n in real_names if n in json.dumps(on_disk)]
        check("on-disk synthetic trace file also contains zero real Cologne team names", not leaked_disk)

    print("\n=== 16. synthetic stress test: 500 runs, 0 invariant failures ===")
    summary = te.run_stress_test(n_runs=500, base_seed=42, rules=rules)
    check("500 runs completed", summary.n_runs == 500)
    check("0 invariant failures", summary.invariant_failures == 0)
    check("max_rounds_per_stage == 5", summary.max_rounds_per_stage == 5)
    check("total_matches == 500 * (33*3 + 7)", summary.total_matches == 500 * (33 * 3 + 7))
    print(f"    stress summary: {summary.to_dict()}")

    print("\n=== 17. Phase 1-8B artifacts byte-unchanged ===")
    for rel, expected in baseline.items():
        p = ROOT / rel
        check(f"unchanged: {rel}", p.exists() and sha256(p) == expected)

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
