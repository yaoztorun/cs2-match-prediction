"""
Phase 9E validation. Read-only. Exits non-zero on failure. Passing this
script means: the tournament service receipt's hashes match disk, the
ruleset registry agrees with the frozen engine rules, the historical
Phase 8D/8E receipts verify transitively, the historical favorite-path and
sample-trace parity hard gates hold, the probability-matrix contract holds,
Monte Carlo accounting/RNG-determinism/override semantics hold, the API
schema/routes/OpenAPI are intact, state is never mutated, XGB V3 is never
used for tournament simulation, no model training exists anywhere in the
service, and the Phase 9B/9C/9D validators (and Phase 8C's own test suite)
still pass unchanged.
"""

import ast
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from _common import ROOT
import application.inference.application_inference as ai
import application.tournament.application_tournament_service as ats
import tournament.cologne_2026.phase8d_common as phase8d_common
import tournament.engine.tournament_engine as te

CONFIG = ROOT / "config"
TESTS = ROOT / "tests"
DEPLOY_DIR = ROOT / "data" / "deployment"
RULESET = ats.DEFAULT_RULESET_ID
FROZEN_FAVORITE_PATH_HASH = "6d96855f4c3f08ec99229bdffe2ab6d7c8285a32db20281973db5f5abe58ed35"

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def sha256_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_reads_forbidden(path, forbidden_substrings):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(f in alias.name for f in forbidden_substrings):
                    hits.append(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(f in node.module for f in forbidden_substrings):
                hits.append(node.module)
    return hits


def _real_participants():
    teams = phase8d_common.load_cologne_teams()
    by_stage = {"stage1": [], "stage2_direct": [], "stage3_direct": []}
    label_map = {"stage_1": "stage1", "stage_2": "stage2_direct", "stage_3": "stage3_direct"}
    for t in teams:
        by_stage[label_map[t["starting_stage"]]].append({"team": t["canonical_model_name"], "seed": t["pre_event_seed"]})
    return by_stage


def main():
    print("\n=== 1. tournament service receipt / version hashes ===")
    receipt_path = DEPLOY_DIR / "application_tournament_service_receipt_v1.json"
    check("application_tournament_service_receipt_v1.json exists", receipt_path.exists())
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        check("receipt committed=True", receipt.get("committed") is True)
        check("service_version = application_tournament_service_v1",
              receipt["service_version"] == "application_tournament_service_v1")
        check("default_prediction_context = deployment_post_cologne_v1",
              receipt["default_prediction_context"] == "deployment_post_cologne_v1")
        check("historical_cologne_contract = frozen_phase8d_phase8e",
              receipt["historical_cologne_contract"] == "frozen_phase8d_phase8e")
        h = receipt["hashes"]
        fresh = {
            "application_tournament_service_py": sha256_file(ROOT / "application" / "tournament" / "application_tournament_service.py"),
            "application_tournament_router_py": sha256_file(ROOT / "application" / "tournament" / "application_tournament_router.py"),
            "ruleset_registry": sha256_file(CONFIG / "application" / "application_tournament_rulesets_v1.yaml"),
            "fixture_manifest": sha256_file(CONFIG / "application" / "application_tournament_fixtures_v1.yaml"),
            "tournament_engine_py": sha256_file(ROOT / "tournament" / "engine" / "tournament_engine.py"),
            "phase9b_context_registry": sha256_file(CONFIG / "application" / "application_inference_contexts_v1.yaml"),
            "phase9d_api_receipt": sha256_file(DEPLOY_DIR / "application_api_receipt_v1.json"),
            "phase8d_simulation_receipt": sha256_file(ROOT / "data" / "evaluation" /
                                                        "cologne_2026_pre_event_simulation_receipt_v1.json"),
            "phase8e_evaluation_receipt": sha256_file(ROOT / "data" / "evaluation" /
                                                        "cologne_2026_simulation_vs_reality_receipt_v1.json"),
            "test_phase9e_py": sha256_file(TESTS / "application" / "test_phase9e_application_tournament_service.py"),
        }
        for k, v in fresh.items():
            check(f"receipt hash matches disk: {k}", h.get(k) == v)

    print("\n=== 2. ruleset registry agrees with the frozen engine rules ===")
    ok, checks_detail = ats.verify_ruleset_matches_engine_rules(RULESET)
    check(f"registry matches te.load_frozen_rules() exactly ({checks_detail})", ok)

    print("\n=== 3. engine hash unchanged since Phase 8D ===")
    check("tournament_engine.py hash == Phase 8D-recorded value",
          sha256_file(ROOT / "tournament" / "engine" / "tournament_engine.py") == "012bd58e7792f7cce1e888d8f233fab274d798d4c3728f5b237f041fb73dd665")

    print("\n=== 4. historical Phase 8D/8E receipts verify transitively ===")
    hist_ok, hist_detail = ats.verify_historical_cologne_contract()
    check(f"historical Cologne contract verifies cleanly ({hist_detail})", hist_ok)

    print("\n=== 5. hard gate: historical favorite-path parity ===")
    entrants = phase8d_common.build_cologne_entrants()
    matrix_df = pd.read_parquet(ats._PHASE8D_MATRIX, engine="fastparquet")
    lookup = {(r.team_a, r.team_b, r.best_of): float(r.probability_team_a) for r in matrix_df.itertuples(index=False)}
    rules = te.load_frozen_rules()
    path = ats._run_deterministic_path(lookup, rules, entrants, [])
    check(f"favorite path champion == Team Vitality (got {path['champion']})", path["champion"] == "Team Vitality")
    check("favorite path canonical_trace_hash == frozen Phase 8D hash",
          path["canonical_trace_hash"] == FROZEN_FAVORITE_PATH_HASH)

    print("\n=== 6. hard gate: sample-trace Monte Carlo parity (indices 0, 1, 42, 999) ===")
    entrants_dict = {"stage1": [e.to_dict() for e in entrants[0]], "stage2_direct": [e.to_dict() for e in entrants[1]],
                      "stage3_direct": [e.to_dict() for e in entrants[2]]}
    samples = json.loads(ats._PHASE8D_SAMPLE_TRACES.read_text(encoding="utf-8"))
    n_ok = 0
    for s in samples:
        idx = s["simulation_index"]
        stage1 = [te.TeamEntry(**e) for e in entrants_dict["stage1"]]
        stage2_direct = [te.TeamEntry(**e) for e in entrants_dict["stage2_direct"]]
        stage3_direct = [te.TeamEntry(**e) for e in entrants_dict["stage3_direct"]]
        seed_seq = np.random.SeedSequence([42, idx])
        rng = np.random.default_rng(seed_seq)
        provider = ats._MonteCarloOverrideAwareProvider(lookup, rng, {}, {})
        result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
        n_ok += int(result.champion == s["champion"] and te.trace_hash(result.to_dict()) == s["canonical_trace_hash"])
    check(f"4/4 sample-trace simulations match the frozen Phase 8D traces exactly (n_ok={n_ok})", n_ok == 4)

    print("\n=== 7. probability matrix contract + accounting (interactive) ===")
    participants = _real_participants()
    canon = ats.validate_tournament_participants(RULESET, participants, "deployment_post_cologne_v1")
    matrix = ats.build_tournament_probability_matrix("deployment_post_cologne_v1", canon["all_canonical_teams"],
                                                       "tier1", None)
    check("interactive matrix has exactly 2,976 rows", len(matrix.lookup) == 32 * 31 * 3)
    check("interactive matrix probabilities all finite in [0,1]",
          all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in matrix.lookup.values()))

    n = 500
    sim = ats.simulate_tournament(RULESET, "deployment_post_cologne_v1", "tier1", None, participants, n, seed=42)
    champ_sum = sum(r["numerator_count"] for r in sim["champion_ranking"])
    playoff_sum = sum(t["reach_playoffs"]["numerator_count"] for t in sim["teams"])
    semifinal_sum = sum(t["reach_semifinal"]["numerator_count"] for t in sim["teams"])
    final_sum = sum(t["reach_final"]["numerator_count"] for t in sim["teams"])
    check(f"champion_sum == n ({champ_sum} == {n})", champ_sum == n)
    check(f"playoff_sum == 8n ({playoff_sum} == {8*n})", playoff_sum == 8 * n)
    check(f"semifinal_sum == 4n ({semifinal_sum} == {4*n})", semifinal_sum == 4 * n)
    check(f"final_sum == 2n ({final_sum} == {2*n})", final_sum == 2 * n)

    print("\n=== 8. RNG determinism + chunking independence ===")
    sim2 = ats.simulate_tournament(RULESET, "deployment_post_cologne_v1", "tier1", None, participants, n, seed=42)
    check("same (participants, seed, n) -> identical champion_ranking", sim["champion_ranking"] == sim2["champion_ranking"])

    entrants_raw = {"stage1": [e.to_dict() for e in canon["entrants"][0]],
                     "stage2_direct": [e.to_dict() for e in canon["entrants"][1]],
                     "stage3_direct": [e.to_dict() for e in canon["entrants"][2]]}
    lookup_plain = dict(matrix.lookup)
    base_payload = {"entrants": entrants_raw, "matrix_lookup": lookup_plain, "overrides": [], "base_seed": 42,
                     "all_team_ids": canon["all_canonical_teams"]}
    single = ats._run_monte_carlo_batch(dict(base_payload, start_index=0, count=200))
    partials = [ats._run_monte_carlo_batch(dict(base_payload, start_index=s, count=50)) for s in (0, 50, 100, 150)]
    merged = ats._merge_partial_aggregates(partials)
    check("single-batch vs 4-chunk merged aggregate identical (champion_counts)",
          single["champion_counts"] == merged["champion_counts"])
    check("single-batch vs 4-chunk merged aggregate identical (team dict)", single["team"] == merged["team"])

    print("\n=== 9. manual override semantics ===")
    path_baseline = ats.predict_tournament_path(RULESET, "deployment_post_cologne_v1", "tier1", None, participants)
    m0 = path_baseline["stage_1"]["matches"][0]
    loser = m0["team_b"] if m0["winner"] == m0["team_a"] else m0["team_a"]
    valid_ov = {"stage": "stage_1", "round_number": 1, "record_group": m0["record_group"],
                "team_1": m0["team_a"], "team_2": m0["team_b"], "winner": loser}
    overridden = ats.predict_tournament_path(RULESET, "deployment_post_cologne_v1", "tier1", None, participants,
                                              manual_overrides=[valid_ov])
    check("valid override forces the declared winner", overridden["stage_1"]["matches"][0]["winner"] == loser)
    check("valid override reports overrides_used=1", overridden["override_usage"]["overrides_used"] == 1)

    not_reached_ov = {"stage": "stage_1", "round_number": 5, "record_group": "2-2",
                       "team_1": m0["team_a"], "team_2": m0["team_b"], "winner": m0["team_a"]}
    nr = ats.predict_tournament_path(RULESET, "deployment_post_cologne_v1", "tier1", None, participants,
                                      manual_overrides=[not_reached_ov])
    check("implausible-record override reports not_reached=1", nr["override_usage"]["overrides_not_reached"] == 1)

    try:
        ats.predict_tournament_path(RULESET, "deployment_post_cologne_v1", "tier1", None, participants,
                                     manual_overrides=[valid_ov, dict(valid_ov)])
        dup_ok = False
    except ai.ApplicationInferenceError as e:
        dup_ok = e.error_code == "duplicate_override"
    check("duplicate override rejected as duplicate_override", dup_ok)

    try:
        ats.predict_tournament_path(RULESET, "deployment_post_cologne_v1", "tier1", None, participants,
                                     manual_overrides=[valid_ov, dict(valid_ov, winner=m0["team_a"])])
        contra_ok = False
    except ai.ApplicationInferenceError as e:
        contra_ok = e.error_code == "contradictory_override"
    check("contradictory override rejected as contradictory_override", contra_ok)

    print("\n=== 10. no XGB usage / no fitting / no write operations ===")
    hits = module_reads_forbidden(ROOT / "application" / "tournament" / "application_tournament_service.py",
                                   ["xgboost", "sklearn.model_selection", "hyperopt", "optuna"])
    check("application_tournament_service.py imports no XGB/tuning module (AST)", len(hits) == 0)
    src = (ROOT / "application" / "tournament" / "application_tournament_service.py").read_text(encoding="utf-8")
    check("application_tournament_service.py never calls .fit(", ".fit(" not in src)
    check("application_tournament_service.py never calls predict_map", "predict_map(" not in src)
    check("application_tournament_service.py never calls predict_series_known_maps",
          "predict_series_known_maps(" not in src)
    for forbidden in (".to_parquet(", ".to_csv(", ".write_bytes(", "os.remove", "shutil."):
        check(f"application_tournament_service.py never calls {forbidden!r}", forbidden not in src)

    print("\n=== 11. API schema / routes / OpenAPI (subprocess-isolated TestClient) ===")
    import application.api.application_api as api
    with TestClient(api.app) as client:
        spec = client.get("/openapi.json").json()
        for p in ("/api/v1/major/rulesets", "/api/v1/major/historical/cologne-2026",
                  "/api/v1/major/historical/cologne-2026/results", "/api/v1/major/path", "/api/v1/major/simulate"):
            check(f"OpenAPI includes {p}", p in spec["paths"])
        r = client.get("/api/v1/health/ready")
        check("health/ready reports all 4 subsystems", set(r.json()["subsystems"]) == {
            "prediction_ready", "explanation_ready", "tournament_engine_ready", "historical_cologne_ready"})
        r = client.get("/api/v1/major/historical/cologne-2026")
        check("historical pre-event endpoint returns 200", r.status_code == 200)
        r = client.get("/api/v1/major/historical/cologne-2026/results")
        check("historical results endpoint returns 200", r.status_code == 200)

    print("\n=== 12. state immutability over a request battery ===")
    import application.inference.build_application_registries as bar
    before = bar.hash_group(bar.DEPLOYMENT_STATE)
    ats.simulate_tournament(RULESET, "deployment_post_cologne_v1", "tier1", None, participants, 50, seed=1)
    after = bar.hash_group(bar.DEPLOYMENT_STATE)
    check("deployment state file hashes unchanged after a Major simulation", before == after)

    print("\n=== 13. Phase 8C / 9B / 9C / 9D regression gate - REAL subprocess commands ===")
    import os
    env = {"PYTHONIOENCODING": "utf-8", **os.environ}
    r1 = subprocess.run([sys.executable, "-m", "pytest", "tests/tournament/test_phase8c_tournament_engine.py", "-q"],
                         cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    check("Phase 8C tournament engine tests still pass unchanged", r1.returncode == 0)
    r2 = subprocess.run([sys.executable, "-m", "validation.validate_phase8c"], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", env=env)
    check("validation/validate_phase8c.py still passes", r2.returncode == 0)
    r3 = subprocess.run([sys.executable, "-m", "pytest", "tests/application/test_phase9d_application_api.py", "-q"],
                         cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    check("full Phase 9D pytest suite still passes (transport-identical)", r3.returncode == 0)
    r4 = subprocess.run([sys.executable, "-m", "validation.validate_phase9b"], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", env=env)
    check("validation/validate_phase9b.py still passes", r4.returncode == 0)
    r5 = subprocess.run([sys.executable, "-m", "validation.validate_phase9c"], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", env=env)
    check("validation/validate_phase9c.py still passes", r5.returncode == 0)
    r6 = subprocess.run([sys.executable, "-m", "validation.validate_phase9d"], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", env=env)
    check("validation/validate_phase9d.py still passes", r6.returncode == 0)
    for label, res in (("8C pytest", r1), ("validate_phase8c", r2), ("9D pytest", r3), ("validate_phase9b", r4),
                        ("validate_phase9c", r5), ("validate_phase9d", r6)):
        if res.returncode != 0:
            print(f"  --- {label} tail ---\n{res.stdout[-2000:]}")

    print("\n=== 14. report exists with required markers ===")
    report_path = ROOT / "reports" / "phases" / "phase9e_application_tournament_service.md"
    check("phase9e report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for marker in ["MAJOR SIMULATION SERVICE V1 = IMPLEMENTED", "HISTORICAL COLOGNE = IMMUTABLE",
                       "CUSTOM MAJOR SIMULATION = ENABLED", "RF V2 = UNCHANGED", "PHASE 8C ENGINE = UNCHANGED",
                       "XGB V3 = NOT USED FOR TOURNAMENT SIMULATION"]:
            check(f"report contains marker: {marker!r}", marker in text)

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
