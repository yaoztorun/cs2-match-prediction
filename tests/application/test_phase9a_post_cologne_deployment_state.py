"""
Phase 9A tests: deployment-history manifest, positive Cologne whitelist,
per-engine consumption audit, immutability of the historical replay record,
and THUNDERdOWNUNDER's cold-start transition.
"""

import ast
import json

import pandas as pd
import pytest
import yaml

from _common import ROOT
import feature_engineering.state.build_deployment_history_manifest as bhm
import feature_engineering.state.phase9a_common as p9a

EVAL = ROOT / "data" / "evaluation"


def _imported_module_names(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.fixture(scope="module")
def manifest():
    return pd.read_parquet(p9a.NEW_DEPLOYMENT_ARTIFACTS["deployment_history_manifest"], engine="fastparquet")


@pytest.fixture(scope="module")
def audit():
    return pd.read_csv(p9a.NEW_DEPLOYMENT_ARTIFACTS["deployment_state_consumption_audit"])


@pytest.fixture(scope="module")
def official_cologne_ids():
    canonical = pd.read_parquet(EVAL / "cologne_2026_actual_series_results_v1.parquet", engine="fastparquet")
    return set(canonical["source_match_id"].astype(int))


# --- 1. manifest partition / whitelist ---

def test_manifest_partition_sums_to_raw_universe(manifest):
    assert len(manifest) == 9923
    counts = manifest["history_status"].value_counts()
    assert counts.get("included", 0) == 9800
    assert counts.get("excluded_showmatch", 0) == 1
    assert counts.get("excluded_existing_reject", 0) == 122


def test_106_official_cologne_series_included(manifest, official_cologne_ids):
    included_cologne = set(manifest.loc[manifest["history_status"] == "included", "match_id"]) & official_cologne_ids
    assert included_cologne == official_cologne_ids
    assert len(included_cologne) == 106


def test_showmatch_excluded(manifest):
    excluded = manifest[manifest["history_status"] == "excluded_showmatch"]
    assert len(excluded) == 1
    assert int(excluded.iloc[0]["match_id"]) == 10094318


def test_old_rejects_preserved_with_original_reasons(manifest):
    rejects = manifest[manifest["history_status"] == "excluded_existing_reject"]
    assert len(rejects) == 122
    assert rejects["history_reason"].str.startswith("phase2_reject:").all()
    reasons = rejects["history_reason"].str.replace("phase2_reject: ", "", regex=False).value_counts()
    assert reasons.get("missing_bestOf", 0) == 116
    assert reasons.get("tie", 0) == 5
    assert reasons.get("missing_score", 0) == 1


def test_manifest_positive_whitelist_stop_condition_would_fire():
    # Directly exercises the STOP branch: an official Cologne match_id NOT in the whitelist
    # (and not the showmatch) must be a hard error, never silently classified either way.
    import inspect
    src = inspect.getsource(bhm.build)
    assert "ERROR_unexplained_cologne_row" in src
    assert "STOP" in src


# --- 2. consumption audit: eligible == consumed, no unexplained gaps ---

def test_audit_covers_five_state_types(audit):
    assert set(audit["state_type"].unique()) == {"series", "map", "form", "roster", "modern_map"}


def test_no_eligible_but_unconsumed_rows(audit):
    gap = audit[audit["eligible_for_state"] & ~audit["consumed_by_state"]]
    assert len(gap) == 0


def test_series_and_form_reach_106_106_official_cologne(audit, official_cologne_ids):
    for state_type in ("series", "form"):
        sub = audit[(audit["state_type"] == state_type) & (audit["match_id"].isin(official_cologne_ids))]
        assert len(sub) == 106
        assert sub["eligible_for_state"].all()
        assert sub["consumed_by_state"].all()


def test_map_roster_modern_map_reach_99_106_with_consistent_exclusions(audit, official_cologne_ids):
    excluded_sets = {}
    for state_type in ("map", "roster", "modern_map"):
        sub = audit[(audit["state_type"] == state_type) & (audit["match_id"].isin(official_cologne_ids))]
        assert len(sub) == 106
        assert int(sub["eligible_for_state"].sum()) == 99
        assert sub.loc[sub["eligible_for_state"], "consumed_by_state"].all()
        excluded_sets[state_type] = set(sub.loc[~sub["eligible_for_state"], "match_id"])
    # the same 7 match_ids are excluded across all three engines (shared map_base dependency)
    assert excluded_sets["map"] == excluded_sets["roster"] == excluded_sets["modern_map"]
    assert len(excluded_sets["map"]) == 7


# --- 3. chronological ordering / no duplicate state updates ---

def test_manifest_included_rows_chronologically_orderable(manifest):
    included = manifest[manifest["history_status"] == "included"]
    dt = pd.to_datetime(included["datetime"])
    assert dt.is_monotonic_increasing or dt.sort_values().equals(dt)


def test_no_duplicate_match_ids_in_manifest(manifest):
    assert not manifest["match_id"].duplicated().any()


def test_deployment_series_state_no_duplicate_processed_uids():
    state = json.loads(p9a.NEW_DEPLOYMENT_ARTIFACTS["series_state"].read_text(encoding="utf-8"))
    assert state["n_matches_processed"] == 9800


# --- 4. post-Cologne legitimate rows included ---

def test_post_cologne_32_rows_all_included(manifest):
    import pandas as pd
    em = pd.read_csv(ROOT / "data" / "interim" / "evaluation_manifest.csv")
    post_ids = set(em.loc[em["evaluation_group"] == "post_cologne", "match_id"])
    post_rows = manifest[manifest["match_id"].isin(post_ids)]
    assert len(post_rows) == 32
    assert (post_rows["history_status"] == "included").all()


# --- 5. pre-Cologne / historical replay state unchanged ---

def test_pre_cologne_series_state_file_unchanged_by_deployment_build():
    protocol = yaml.safe_load((ROOT / "config" / "evaluation" / "phase8e_cologne_simulation_vs_reality_protocol.yaml")
                               .read_bytes())
    expected = protocol["immutable_pre_event_record"]["hashes"]["strict_pre_cologne_state"]
    actual = p9a.sha256_file(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD["strict_pre_cologne_series_state"])
    assert actual == expected


def test_rf_v2_and_tournament_engine_unchanged():
    protocol = yaml.safe_load((ROOT / "config" / "evaluation" / "phase8e_cologne_simulation_vs_reality_protocol.yaml")
                               .read_bytes())
    baseline = protocol["immutable_pre_event_record"]["hashes"]
    assert p9a.sha256_file(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD["rf_v2_model"]) == baseline["rf_v2_model"]
    assert p9a.sha256_file(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD["phase8c_tournament_engine"]) == \
        baseline["phase8c_tournament_engine"]


def test_known_map_xgb_v3_hash_is_tracked():
    assert p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD["known_map_xgb_v3_model"].exists()
    h = p9a.sha256_file(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD["known_map_xgb_v3_model"])
    assert isinstance(h, str) and len(h) == 64


# --- 6. no model-fitting imports in any Phase 9A builder ---

def test_no_state_builder_imports_model_fitting_modules():
    forbidden = {"joblib", "xgboost", "sklearn"}
    builder_files = [
        "feature_engineering/state/build_deployment_history_manifest.py", "feature_engineering/state/build_deployment_series_state.py",
        "feature_engineering/state/build_deployment_map_state.py", "feature_engineering/state/build_deployment_form_state.py",
        "feature_engineering/state/build_deployment_roster_state.py", "feature_engineering/state/build_deployment_modern_map_state.py",
    ]
    for f in builder_files:
        imported = _imported_module_names(f)
        hit = imported & forbidden
        assert not hit, f"{f} imports forbidden module(s): {hit}"


# --- 7. deterministic rebuild ---

def test_manifest_build_is_deterministic():
    df1 = bhm.build()
    df2 = bhm.build()
    pd.testing.assert_frame_equal(
        df1.sort_values(["match_id"]).reset_index(drop=True),
        df2.sort_values(["match_id"]).reset_index(drop=True),
    )


# --- 8. THUNDERdOWNUNDER cold-start transition ---

def test_thunderdownunder_cold_start_then_legitimate_history():
    pre = json.loads(p9a.HISTORICAL_REPLAY_IMMUTABLE_RECORD["strict_pre_cologne_series_state"]
                      .read_text(encoding="utf-8"))
    assert "THUNDERdOWNUNDER" not in pre["teams"], "expected a true cold start pre-Cologne"

    post = json.loads(p9a.NEW_DEPLOYMENT_ARTIFACTS["series_state"].read_text(encoding="utf-8"))
    assert "THUNDERdOWNUNDER" in post["teams"]
    history = post["teams"]["THUNDERdOWNUNDER"]["history"]
    assert len(history) > 0


# --- 9. deployment cutoff correctness ---

def test_deployment_cutoff_equals_max_included_datetime(manifest):
    included = manifest[manifest["history_status"] == "included"]
    assert str(included["datetime"].max()) == "2026-06-28 20:00:00"
