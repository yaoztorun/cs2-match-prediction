"""
Phase 9A deployment map state: `deployment_post_cologne`, a FRESH rebuild
through the unmodified `map_feature_engine.MapStateStore` /
`process_combined_stream` engine, gated by the deployment-history manifest.
Mirrors `feature_engineering/state/build_pre_cologne_map_state_v1.py`.

Unlike the series/form engines, `process_combined_stream` exposes no native
excluded-rows/reason output (confirmed by direct engine inspection), so the
consumption audit here is derived directly from the same
`team1_eligible`/`team2_eligible` columns the engine itself reads - not by
guessing at internal logic. A genuinely separate, EXPECTED source of
non-eligibility exists for this engine: some legitimate series have zero
surviving `map_base.parquet` rows (a Phase-2, upstream, structural fact,
independent of identity eligibility) - those are recorded with their own
reason and are NOT treated as an error.
"""

import pandas as pd

from _common import INTERIM, ROOT, raw_file_hashes
from feature_engineering.maps.map_feature_engine import MAP_ENGINE_VERSION, MAP_POOL_LOOKBACK_DAYS, MapStateStore, process_combined_stream
from feature_engineering.maps.map_stream_common import load_map_stream

DEPLOY = ROOT / "data" / "deployment"
MANIFEST_PATH = DEPLOY / "deployment_history_manifest_v1.parquet"
CANONICAL_COLOGNE_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"


def build(output_path=None):
    hashes_before = raw_file_hashes()
    output_path = output_path or (INTERIM / "map_state_v1_deployment_post_cologne.json")

    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    included_ids = set(manifest.loc[manifest["history_status"] == "included", "match_id"].astype(int))
    official_cologne_ids = set(pd.read_parquet(CANONICAL_COLOGNE_PATH, engine="fastparquet")
                                ["source_match_id"].astype(int))
    deployment_cutoff = manifest.loc[manifest["history_status"] == "included", "datetime"].max()

    stream, info = load_map_stream(evaluation_groups=("development", "cologne_2026", "post_cologne"))
    before_n = len(stream)
    stream = stream[stream["match_id"].isin(included_ids)].copy()
    print(f"deployment map stream: {before_n} rows -> {len(stream)} after manifest anti-join")

    # every official Cologne match_id present in the manifest as "included" - but some may
    # legitimately have zero surviving map_base rows (structural, checked below, not an error)
    matches_with_map_rows = set(stream["match_id"].astype(int))
    cologne_without_map_rows = official_cologne_ids - matches_with_map_rows

    store = MapStateStore()
    process_combined_stream(store, stream, series_requests=None, emit_map_features=False)

    seen_matches = {int(h.match_id) for st in store.states.values() for h in st.history}
    reachable_cologne = official_cologne_ids & (matches_with_map_rows | seen_matches)
    if reachable_cologne != (official_cologne_ids - cologne_without_map_rows):
        raise ValueError("STOP: unexpected mismatch between map-row availability and store contents "
                          "for official Cologne matches")

    # ---- consumption audit (amendment #3/#4) at match_id granularity ----
    both_elig_by_match = stream.groupby("match_id").apply(
        lambda g: bool((g["team1_eligible"] & g["team2_eligible"]).any())).to_dict()
    any_elig_by_match = stream.groupby("match_id").apply(
        lambda g: bool((g["team1_eligible"] | g["team2_eligible"]).any())).to_dict()

    audit_rows = []
    for mid in included_ids:
        mid = int(mid)
        if mid not in matches_with_map_rows:
            eligible, consumed = False, False
            reason = "no surviving map_base.parquet rows for this match_id (Phase 2 map-level rejection - " \
                     "structural, independent of team identity)"
        else:
            eligible = any_elig_by_match.get(mid, False)
            consumed = mid in seen_matches
            if eligible and not both_elig_by_match.get(mid, False):
                reason = "at least one side identity-eligible (own map history updated); map ELO not " \
                          "updated for the ineligible side (needs both)"
            elif eligible:
                reason = "both sides identity-eligible: full map-history + map-ELO update"
            else:
                reason = "neither side identity-eligible"
        audit_rows.append({"state_type": "map", "match_id": mid, "deployment_history_status": "included",
                            "eligible_for_state": eligible, "consumed_by_state": consumed,
                            "eligibility_reason": reason})
        if eligible and not consumed:
            raise ValueError(f"STOP: match_id {mid} is eligible for the map state but was not consumed.")

    n_official_with_map_rows = len(official_cologne_ids) - len(cologne_without_map_rows)
    print(f"official Cologne matches with surviving map_base rows: {n_official_with_map_rows}/106 "
          f"({len(cologne_without_map_rows)} legitimately excluded: no map rows survived Phase 2)")

    untrusted_entries = sum(1 for st in store.states.values() for h in st.history if not h.opponent_identity_trusted)
    max_history_dt = max((h.series_dt for st in store.states.values() for h in st.history), default=None)
    if max_history_dt is not None and str(max_history_dt) > str(deployment_cutoff):
        raise ValueError(f"STOP: a map history entry ({max_history_dt}) is later than the deployment cutoff")

    summary = store.snapshot_summary_df().sort_values(["canonical_team_name", "map_name"]).reset_index(drop=True)
    parquet_path = output_path.with_suffix(".parquet")
    summary.to_parquet(parquet_path, engine="fastparquet", index=False)

    meta = {
        "historical_replay_state": "pre_cologne", "deployment_state": "deployment_post_cologne",
        "map_engine_version": MAP_ENGINE_VERSION, "snapshot_id": "map_state_v1_deployment_post_cologne",
        "state_source": "deployment_history_manifest_v1.parquet + canonical map stream (fresh rebuild)",
        "not_rebuilt_from": "map_features_v1.parquet (would drop untrusted-opponent history)",
        "deployment_history_cutoff": str(deployment_cutoff),
        "map_pool_lookback_days": MAP_POOL_LOOKBACK_DAYS,
        "source_maps_replayed": int(len(stream)), "source_matches_replayed": int(stream["match_id"].nunique()),
        "official_cologne_matches_with_map_rows": n_official_with_map_rows,
        "official_cologne_matches_without_map_rows": sorted(int(x) for x in cologne_without_map_rows),
        "team_map_states": int(len(store.states)),
        "entries_from_untrusted_opponent_matches": untrusted_entries,
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(output_path, meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the deployment build"
    print(f"Wrote {output_path} ({len(store.states)} team-map states)")
    print(f"Wrote {parquet_path} ({len(summary)} rows)")
    return store, pd.DataFrame(audit_rows), meta


if __name__ == "__main__":
    build()
