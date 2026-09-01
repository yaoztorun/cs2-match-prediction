"""
Phase 9A deployment series state: `deployment_post_cologne`, built as a
FRESH rebuild (never resuming/patching `pre_cologne_team_state_v1_full.json`
or the Mode-A full-development state) from the deployment-history manifest,
through the unmodified `feature_engine.StateStore` /
`process_chronological_stream` engine. Mirrors
`feature_engineering/state/build_pre_cologne_snapshot.py` exactly, just fed the wider
manifest-gated pool instead of a strict pre-Cologne cutoff.
"""

import pandas as pd

from _common import INTERIM, ROOT, raw_file_hashes
from feature_engineering.series.feature_engine import StateStore, process_chronological_stream, ENGINE_VERSION
from feature_engineering.series.build_series_features_v1 import build_stream_rows
import feature_engineering.state.phase9a_common as p9a

DEPLOY = ROOT / "data" / "deployment"
MANIFEST_PATH = DEPLOY / "deployment_history_manifest_v1.parquet"
CANONICAL_COLOGNE_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"


def build(output_path=None):
    hashes_before = raw_file_hashes()
    output_path = output_path or (ROOT / "data" / "features" / "series_team_state_v1_deployment_post_cologne.json")

    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    included_ids = set(manifest.loc[manifest["history_status"] == "included", "match_id"].astype(int))
    official_cologne_ids = set(pd.read_parquet(CANONICAL_COLOGNE_PATH, engine="fastparquet")
                                ["source_match_id"].astype(int))
    deployment_cutoff = manifest.loc[manifest["history_status"] == "included", "datetime"].max()

    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet")
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    merged = sb.merge(em, on="match_id", how="inner")
    pool = merged[merged["evaluation_group"].isin(("development", "cologne_2026", "post_cologne"))].copy()
    rows = build_stream_rows(pool)

    # defensive anti-join against the manifest's authoritative included set (drops the
    # showmatch even though the evaluation_group filter above would otherwise admit it)
    before_n = len(rows)
    rows = rows[rows["match_id"].isin(included_ids)].copy()
    print(f"deployment input pool: {before_n} rows -> {len(rows)} after manifest anti-join "
          f"({before_n - len(rows)} dropped, expected 1 = the showmatch)")
    if before_n - len(rows) != 1:
        raise ValueError(f"STOP: expected exactly 1 row dropped by the manifest anti-join (the showmatch), "
                          f"got {before_n - len(rows)}")

    # FRESH store - never resumes Mode A's development store or patches pre_cologne_team_state_v1_full.json
    store = StateStore()
    processed_rows, excluded_rows = process_chronological_stream(store, rows)
    assert len(processed_rows) + len(excluded_rows) == len(rows)

    # ---- amendment #2 style contamination/positivity checks ----
    processed_ids = {int(r["match_id"]) for r in processed_rows} | {int(r["match_id"]) for r in excluded_rows}
    reachable_cologne = official_cologne_ids & processed_ids
    unreachable_cologne = official_cologne_ids - processed_ids
    if unreachable_cologne:
        raise ValueError(f"STOP: {len(unreachable_cologne)} official Cologne match_ids never reached the "
                          f"series engine at all: {sorted(unreachable_cologne)[:10]}")

    # ---- consumption audit (amendment #3/#4): eligible = >=1 side identity-eligible ----
    excluded_by_id = {int(r["match_id"]): r["reason"] for r in excluded_rows}
    processed_id_set = {int(r["match_id"]) for r in processed_rows}
    audit_rows = []
    for _, r in rows.iterrows():
        mid = int(r["match_id"])
        t1e, t2e = bool(r["team1_eligible"]), bool(r["team2_eligible"])
        eligible = t1e or t2e
        if mid in processed_id_set:
            consumed, reason = True, "both sides identity-eligible: full feature row + both histories + ELO updated"
        elif mid in excluded_by_id:
            reason = excluded_by_id[mid]
            consumed = t1e or t2e  # exactly one side eligible -> that side's own history still updated
        else:
            consumed, reason = False, "unexpected: row never reached the engine"
        deployment_status = "included"
        audit_rows.append({"state_type": "series", "match_id": mid,
                            "deployment_history_status": deployment_status,
                            "eligible_for_state": eligible, "consumed_by_state": consumed,
                            "eligibility_reason": reason})
        if eligible and not consumed:
            raise ValueError(f"STOP: match_id {mid} is eligible for the series state but was not consumed - "
                              f"unexplained gap.")

    n_trusted = sum(1 for ts in store.teams.values() for h in ts.history if h.opponent_identity_trusted)
    n_untrusted = sum(1 for ts in store.teams.values() for h in ts.history if not h.opponent_identity_trusted)
    max_history_dt = max((h.dt for ts in store.teams.values() for h in ts.history), default=None)
    if max_history_dt is not None and str(max_history_dt) > str(deployment_cutoff):
        raise ValueError(f"STOP: a history entry ({max_history_dt}) is later than the deployment cutoff "
                          f"({deployment_cutoff})")

    meta = {
        "historical_replay_state": "pre_cologne", "deployment_state": "deployment_post_cologne",
        "engine_version": ENGINE_VERSION, "snapshot_id": "series_team_state_v1_deployment_post_cologne",
        "state_source": "deployment_history_manifest_v1.parquet + canonical series stream (fresh rebuild, "
                         "never resumes Mode A's development store or patches the pre-Cologne snapshot)",
        "deployment_history_cutoff": str(deployment_cutoff),
        "n_input_rows": int(len(rows)), "n_feature_rows": len(processed_rows), "n_excluded_rows": len(excluded_rows),
        "official_cologne_match_ids_included": len(reachable_cologne),
        "history_entries_trusted_opponent": n_trusted, "history_entries_untrusted_opponent": n_untrusted,
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(output_path, meta=meta)

    summary = store.snapshot_summary_df().sort_values("canonical_team_name").reset_index(drop=True)
    parquet_path = output_path.with_suffix(".parquet")
    summary.to_parquet(parquet_path, engine="fastparquet", index=False)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the deployment build"
    print(f"Wrote {output_path} ({len(store.teams)} teams, {len(store.processed_match_uids)} matches, "
          f"{len(reachable_cologne)}/106 official Cologne matches included)")
    print(f"Wrote {parquet_path} ({len(summary)} teams)")
    return store, pd.DataFrame(audit_rows), meta


if __name__ == "__main__":
    build()
