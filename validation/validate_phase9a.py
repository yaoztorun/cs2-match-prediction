"""
Phase 9A validation. Read-only. Exits non-zero on failure. Passing this
script means: the deployment-history manifest partitions correctly, the
positive 106-ID Cologne whitelist reconciles with zero unexplained rows,
every state engine's consumption audit satisfies eligible==consumed with no
unexplained gaps, no state builder imports a training/model-fitting module,
and the full historical-replay record (Phase 4-8E) has not drifted from the
values Phase 8E itself already recorded as ground truth.
"""

import ast
import json
import sys

import numpy as np
import pandas as pd
import yaml

from _common import ROOT
import feature_engineering.state.phase9a_common as p9a

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


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


def main():
    print("\n=== 1. deployment-history manifest partition ===")
    manifest = pd.read_parquet(p9a.NEW_DEPLOYMENT_ARTIFACTS["deployment_history_manifest"], engine="fastparquet")
    check("manifest has exactly 9,923 rows", len(manifest) == 9923)
    counts = manifest["history_status"].value_counts()
    check("included = 9,800", counts.get("included", 0) == 9800)
    check("excluded_showmatch = 1", counts.get("excluded_showmatch", 0) == 1)
    check("excluded_existing_reject = 122", counts.get("excluded_existing_reject", 0) == 122)
    check("no ERROR_unexplained_cologne_row status present",
          not (manifest["history_status"] == "ERROR_unexplained_cologne_row").any())

    print("\n=== 2. positive 106-ID Cologne whitelist ===")
    canonical = pd.read_parquet(ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet",
                                 engine="fastparquet")
    official_ids = set(canonical["source_match_id"].astype(int))
    check("106 official Cologne match_ids in the canonical actual-results artifact", len(official_ids) == 106)
    em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
    cologne_group_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    manifest_cologne = manifest[manifest["match_id"].isin(cologne_group_ids)]
    check("all 107 cologne_2026-tagged rows resolved (106 included + 1 showmatch)",
          len(manifest_cologne) == 107
          and (manifest_cologne["history_status"] == "included").sum() == 106
          and (manifest_cologne["history_status"] == "excluded_showmatch").sum() == 1)
    included_cologne_ids = set(manifest_cologne.loc[manifest_cologne["history_status"] == "included", "match_id"])
    check("manifest's included Cologne set equals the Phase 8E official whitelist exactly",
          included_cologne_ids == official_ids)

    print("\n=== 3. deployment cutoff ===")
    included = manifest[manifest["history_status"] == "included"]
    cutoff = included["datetime"].max()
    check("deployment cutoff = 2026-06-28 20:00:00", str(cutoff) == "2026-06-28 20:00:00")
    post_cologne_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    post_cologne_rows = manifest[manifest["match_id"].isin(post_cologne_ids)]
    check("post_cologne raw rows = 32", len(post_cologne_rows) == 32)
    check("post_cologne included = 32", (post_cologne_rows["history_status"] == "included").sum() == 32)

    print("\n=== 4. consumption audit: eligible == consumed, no unexplained gaps ===")
    audit = pd.read_csv(p9a.NEW_DEPLOYMENT_ARTIFACTS["deployment_state_consumption_audit"])
    check("audit covers exactly 5 state_types",
          set(audit["state_type"].unique()) == {"series", "map", "form", "roster", "modern_map"})
    gap = audit[audit["eligible_for_state"] & ~audit["consumed_by_state"]]
    check("zero eligible-but-not-consumed rows across all 5 state types", len(gap) == 0)
    for state_type in ("series", "form"):
        sub = audit[(audit["state_type"] == state_type) & (audit["match_id"].isin(official_ids))]
        check(f"{state_type}: all 106 official Cologne matches eligible and consumed",
              len(sub) == 106 and sub["eligible_for_state"].all() and sub["consumed_by_state"].all())
    for state_type in ("map", "roster", "modern_map"):
        sub = audit[(audit["state_type"] == state_type) & (audit["match_id"].isin(official_ids))]
        n_eligible = int(sub["eligible_for_state"].sum())
        check(f"{state_type}: 99/106 official Cologne matches eligible (7 legitimately lack map_base rows)",
              len(sub) == 106 and n_eligible == 99)
        check(f"{state_type}: every eligible official Cologne match was consumed",
              sub.loc[sub["eligible_for_state"], "consumed_by_state"].all())

    print("\n=== 5. deployment state artifacts exist and are internally consistent ===")
    for key in ("series_state", "map_state", "form_state", "roster_state", "modern_map_state"):
        path = p9a.NEW_DEPLOYMENT_ARTIFACTS[key]
        check(f"{key} exists", path.exists())
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            check(f"{key} meta declares historical_replay_state=pre_cologne / deployment_state=deployment_post_cologne",
                  payload.get("meta", {}).get("historical_replay_state") == "pre_cologne"
                  and payload.get("meta", {}).get("deployment_state") == "deployment_post_cologne")

    print("\n=== 6. no model-fitting / training imports in any Phase 9A builder ===")
    forbidden = ["joblib", "sklearn.ensemble", "sklearn.linear_model", "xgboost", "pre_veto_series_predictor"]
    builder_files = ["feature_engineering/state/build_deployment_history_manifest.py", "feature_engineering/state/build_deployment_series_state.py",
                      "feature_engineering/state/build_deployment_map_state.py", "feature_engineering/state/build_deployment_form_state.py",
                      "feature_engineering/state/build_deployment_roster_state.py", "feature_engineering/state/build_deployment_modern_map_state.py"]
    for f in builder_files:
        hits = module_reads_forbidden(ROOT / f, forbidden)
        check(f"{f} imports no model-fitting module (AST)", len(hits) == 0)

    print("\n=== 7. historical replay record: cross-checked against Phase 8E's own recorded baseline ===")
    phase8e_protocol = yaml.safe_load((ROOT / "config" / "evaluation" / "phase8e_cologne_simulation_vs_reality_protocol.yaml")
                                       .read_bytes())
    phase8e_baseline = phase8e_protocol["immutable_pre_event_record"]["hashes"]
    name_map = {
        "phase8d_protocol": "phase8d_protocol", "phase8d_probability_matrix": "phase8d_probability_matrix",
        "phase8d_probability_receipt": "phase8d_probability_receipt",
        "phase8d_simulation_receipt": "phase8d_simulation_receipt",
        "phase8d_team_probabilities": "phase8d_team_probabilities",
        "phase8d_swiss_record_distributions": "phase8d_swiss_record_distributions",
        "phase8d_playoff_seed_distributions": "phase8d_playoff_seed_distributions",
        "phase8d_matchup_frequencies": "phase8d_matchup_frequencies",
        "phase8d_favorite_path": "phase8d_favorite_path", "phase8b_tournament_yaml": "phase8b_tournament_yaml",
        "phase8c_tournament_engine": "phase8c_tournament_engine", "rf_v2_model": "rf_v2_model",
        "strict_pre_cologne_series_state": "strict_pre_cologne_state",
        "phase8d1_provenance_report": "phase8d1_provenance_report",
        "phase8d1_provenance_json": "phase8d1_provenance_json",
    }
    current = p9a.hash_historical_replay_record()
    n_cross_checked = 0
    for p9a_name, phase8e_name in name_map.items():
        if phase8e_name in phase8e_baseline:
            check(f"unchanged since Phase 8E: {p9a_name}", current[p9a_name] == phase8e_baseline[phase8e_name])
            n_cross_checked += 1
    check("cross-checked all 15 overlapping Phase 8E immutable-record items", n_cross_checked == 15)
    check(f"historical replay record has {len(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD)} total tracked items "
          f"(>= 15 cross-checked + 4 pre-Cologne states + known-map XGB V3 + sealed splits/protocols/configs)",
          len(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD) >= 35)

    receipt_path = p9a.DEPLOY / "deployment_state_receipt_v1.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        check("receipt records a within-run before/after historical-replay hash check that passed",
              receipt.get("historical_replay_unchanged_within_run") is True)

    print("\n=== 8. report exists with required markers ===")
    report_path = ROOT / "reports" / "phases" / "phase9a_post_cologne_deployment_state.md"
    check("phase9a report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for marker in ["POST-COLOGNE DEPLOYMENT STATE = CREATED", "HISTORICAL COLOGNE REPLAY = UNCHANGED",
                       "NO RETRAINING", "latest locally available historical state"]:
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
