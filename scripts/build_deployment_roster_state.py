"""
Phase 9A deployment player/roster state: `deployment_post_cologne`, a FRESH
rebuild through the unmodified `player_roster_feature_engine.
PlayerRosterStateStore` / `process_player_roster_stream` engine, gated by
the deployment-history manifest. Mirrors
`scripts/build_pre_cologne_player_roster_state_v1.py`.

`process_player_roster_stream` has NO exclusion-tracking mechanism at all
(confirmed by direct engine inspection: an ineligible series request is
silently `continue`d with zero trace) - so, exactly as for the map engine,
the consumption audit here is derived directly from the stream's own
`team_eligible`/`has_usable_stats` columns, never from guessed-at internal
logic. A legitimate series may have zero surviving player-observation rows
at all (roster data is joined from `map_base.parquet`, so the same 7
official Cologne matches with zero map rows structurally cannot contribute
here either) - this is expected and recorded with its own reason, never
forced.
"""

import pandas as pd

from _common import INTERIM, ROOT, raw_file_hashes
from player_roster_feature_engine import (
    PLAYER_ROSTER_ENGINE_VERSION, ROSTER_LOOKBACK_DAYS, PLAYER_FORM_HALF_LIFE_DAYS,
    PlayerRosterStateStore, process_player_roster_stream,
)
from player_roster_stream_common import load_player_roster_stream

DEPLOY = ROOT / "data" / "deployment"
MANIFEST_PATH = DEPLOY / "deployment_history_manifest_v1.parquet"
CANONICAL_COLOGNE_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"


def build(output_path=None):
    hashes_before = raw_file_hashes()
    output_path = output_path or (INTERIM / "player_roster_state_v1_deployment_post_cologne.json")

    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    included_ids = set(manifest.loc[manifest["history_status"] == "included", "match_id"].astype(int))
    official_cologne_ids = set(pd.read_parquet(CANONICAL_COLOGNE_PATH, engine="fastparquet")
                                ["source_match_id"].astype(int))
    deployment_cutoff = manifest.loc[manifest["history_status"] == "included", "datetime"].max()

    stream, info = load_player_roster_stream(evaluation_groups=("development", "cologne_2026", "post_cologne"))
    before_n = len(stream)
    stream = stream[stream["match_id"].isin(included_ids)].copy()
    print(f"deployment player-observation stream: {before_n} rows -> {len(stream)} after manifest anti-join")

    matches_with_player_rows = set(stream["match_id"].astype(int))
    cologne_without_player_rows = official_cologne_ids - matches_with_player_rows

    store = PlayerRosterStateStore()
    process_player_roster_stream(store, stream, series_requests=None, emit_features=False)

    match_id_by_game_id = dict(zip(stream["game_id"].astype(int), stream["match_id"].astype(int)))
    consumed_match_ids = {match_id_by_game_id[gid] for gid in
                           {int(x) for x in store.processed_map_uids} if gid in match_id_by_game_id}

    # ---- consumption audit (amendment #3/#4) at match_id granularity ----
    elig_by_match = stream.groupby("match_id")["team_eligible"].any().to_dict()
    usable_by_match = stream.groupby("match_id")["has_usable_stats"].any().to_dict()
    audit_rows = []
    for mid in included_ids:
        mid = int(mid)
        if mid not in matches_with_player_rows:
            eligible, consumed = False, False
            reason = "no surviving player-observation rows for this match_id (roster data is joined from " \
                     "map_base.parquet, which has zero surviving rows for this match - Phase 2 structural, " \
                     "independent of team identity)"
        else:
            eligible = bool(elig_by_match.get(mid, False)) or bool(usable_by_match.get(mid, False))
            consumed = mid in consumed_match_ids
            reason_parts = []
            if elig_by_match.get(mid, False):
                reason_parts.append(">=1 team-identity-eligible appearance recorded")
            if usable_by_match.get(mid, False):
                reason_parts.append(">=1 player observation with usable stats recorded (team-independent)")
            reason = "; ".join(reason_parts) if reason_parts else \
                "no team-eligible appearance and no usable-stats player observation for this match"
        audit_rows.append({"state_type": "roster", "match_id": mid, "deployment_history_status": "included",
                            "eligible_for_state": eligible, "consumed_by_state": consumed,
                            "eligibility_reason": reason})
        if eligible and not consumed:
            raise ValueError(f"STOP: match_id {mid} is eligible for the roster state but was not consumed.")

    n_official_with_rows = len(official_cologne_ids) - len(cologne_without_player_rows)
    print(f"official Cologne matches with surviving player-observation rows: {n_official_with_rows}/106 "
          f"({len(cologne_without_player_rows)} legitimately excluded: no rows survived Phase 2)")

    max_history_dt = max((h.series_dt for st in store.players.values() for h in st.history), default=None)
    if max_history_dt is not None and str(max_history_dt) > str(deployment_cutoff):
        raise ValueError(f"STOP: a player history entry ({max_history_dt}) is later than the deployment cutoff")

    n_transfers = sum(1 for st in store.players.values() if len({h.team_canonical for h in st.history}) > 1)

    summary_df = store.snapshot_summary_df().sort_values("player_id").reset_index(drop=True)
    parquet_path = output_path.with_suffix(".parquet")
    summary_df.to_parquet(parquet_path, engine="fastparquet", index=False)

    meta = {
        "historical_replay_state": "pre_cologne", "deployment_state": "deployment_post_cologne",
        "player_roster_engine_version": PLAYER_ROSTER_ENGINE_VERSION,
        "snapshot_id": "player_roster_state_v1_deployment_post_cologne",
        "state_source": "deployment_history_manifest_v1.parquet + canonical player observation stream "
                         "(fresh rebuild), NOT series_features_v4_roster.parquet",
        "deployment_history_cutoff": str(deployment_cutoff),
        "roster_lookback_days": ROSTER_LOOKBACK_DAYS, "player_form_half_life_days": PLAYER_FORM_HALF_LIFE_DAYS,
        "player_states": len(store.players), "team_states": len(store.teams),
        "official_cologne_matches_with_player_rows": n_official_with_rows,
        "official_cologne_matches_without_player_rows": sorted(int(x) for x in cologne_without_player_rows),
        "players_with_multiple_teams_observed": n_transfers,
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(output_path, meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the deployment build"
    print(f"Wrote {output_path} ({len(store.players)} players, {len(store.teams)} teams)")
    print(f"Wrote {parquet_path} ({len(summary_df)} players)")
    return store, pd.DataFrame(audit_rows), meta


if __name__ == "__main__":
    build()
