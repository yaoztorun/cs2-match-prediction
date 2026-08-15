"""
Phase 9A deployment form state: `deployment_post_cologne`, a FRESH rebuild
through the unmodified `team_form_engine.TeamFormStateStore` /
`process_form_stream` engine, gated by the deployment-history manifest.
Mirrors `scripts/build_pre_cologne_form_state_v1.py`.
"""

import pandas as pd

from _common import INTERIM, ROOT, raw_file_hashes
from team_form_engine import FORM_ENGINE_VERSION, TeamFormStateStore, process_form_stream
from team_form_stream_common import load_series_form_stream

DEPLOY = ROOT / "data" / "deployment"
MANIFEST_PATH = DEPLOY / "deployment_history_manifest_v1.parquet"
CANONICAL_COLOGNE_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"


def build(output_path=None):
    hashes_before = raw_file_hashes()
    output_path = output_path or (INTERIM / "form_state_v1_deployment_post_cologne.json")

    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    included_ids = set(manifest.loc[manifest["history_status"] == "included", "match_id"].astype(int))
    official_cologne_ids = set(pd.read_parquet(CANONICAL_COLOGNE_PATH, engine="fastparquet")
                                ["source_match_id"].astype(int))
    deployment_cutoff = manifest.loc[manifest["history_status"] == "included", "datetime"].max()

    stream, info = load_series_form_stream(evaluation_groups=("development", "cologne_2026", "post_cologne"))
    before_n = len(stream)
    stream = stream[stream["match_id"].isin(included_ids)].copy()
    print(f"deployment form stream: {before_n} rows -> {len(stream)} after manifest anti-join "
          f"({before_n - len(stream)} dropped, expected 1 = the showmatch)")
    if before_n - len(stream) != 1:
        raise ValueError(f"STOP: expected exactly 1 row dropped (the showmatch), got {before_n - len(stream)}")

    # emit_features=False (matches Mode B): process_form_stream only populates
    # processed_rows/excluded_rows when emit_features=True (confirmed by direct testing -
    # both are empty with emit_features=False, even though Phase B state-writing still runs
    # unconditionally). So the consumption audit is derived directly from the stream's own
    # team1_eligible/team2_eligible columns, exactly mirroring the engine's own gate, rather
    # than relying on excluded_rows/reason output that emit_features=False never populates.
    store = TeamFormStateStore()
    process_form_stream(store, stream, emit_features=False)

    uid_to_match_id = dict(zip(stream["canonical_match_uid"], stream["match_id"].astype(int)))
    processed_match_ids = {uid_to_match_id[u] for u in store.processed_match_uids if u in uid_to_match_id}
    unreachable_cologne = official_cologne_ids - processed_match_ids
    if unreachable_cologne:
        raise ValueError(f"STOP: {len(unreachable_cologne)} official Cologne match_ids never reached the "
                          f"form engine: {sorted(unreachable_cologne)[:10]}")

    audit_rows = []
    for _, r in stream.iterrows():
        mid = int(r["match_id"])
        t1e, t2e = bool(r["team1_eligible"]), bool(r["team2_eligible"])
        eligible = t1e or t2e
        consumed = mid in processed_match_ids and eligible
        if t1e and t2e:
            reason = "both sides identity-eligible: both histories updated"
        elif eligible:
            missing = "team1" if not t1e else "team2"
            reason = f"{missing}_not_identity_eligible - only the eligible side's own history updated"
        else:
            reason = "team1_not_identity_eligible;team2_not_identity_eligible - no state mutation"
        audit_rows.append({"state_type": "form", "match_id": mid, "deployment_history_status": "included",
                            "eligible_for_state": eligible, "consumed_by_state": consumed,
                            "eligibility_reason": reason})
        if eligible and not consumed:
            raise ValueError(f"STOP: match_id {mid} is eligible for the form state but was not consumed.")

    n_trusted = sum(1 for st in store.teams.values() for h in st.history if h.opponent_identity_trusted)
    n_untrusted = sum(1 for st in store.teams.values() for h in st.history if not h.opponent_identity_trusted)
    max_history_dt = max((h.dt for st in store.teams.values() for h in st.history), default=None)
    if max_history_dt is not None and str(max_history_dt) > str(deployment_cutoff):
        raise ValueError(f"STOP: a form history entry ({max_history_dt}) is later than the deployment cutoff")

    summary_df = store.snapshot_summary_df().sort_values("canonical_team_name").reset_index(drop=True)
    parquet_path = output_path.with_suffix(".parquet")
    summary_df.to_parquet(parquet_path, engine="fastparquet", index=False)

    meta = {
        "historical_replay_state": "pre_cologne", "deployment_state": "deployment_post_cologne",
        "form_engine_version": FORM_ENGINE_VERSION, "snapshot_id": "form_state_v1_deployment_post_cologne",
        "state_source": "deployment_history_manifest_v1.parquet + canonical series stream (fresh rebuild), "
                         "NOT series_features_v3_form.parquet",
        "deployment_history_cutoff": str(deployment_cutoff),
        "max_source_series_datetime": str(stream["datetime"].max()),
        "team_states": len(store.teams), "matches_processed": len(store.processed_match_uids),
        "official_cologne_match_ids_included": len(official_cologne_ids),
        "history_entries_trusted_opponent": n_trusted, "history_entries_untrusted_opponent": n_untrusted,
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(output_path, meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the deployment build"
    print(f"Wrote {output_path} ({len(store.teams)} teams, {len(store.processed_match_uids)} matches)")
    print(f"Wrote {parquet_path} ({len(summary_df)} teams)")
    return store, pd.DataFrame(audit_rows), meta


if __name__ == "__main__":
    build()
