"""
Phase 9A deployment modern-map state: `deployment_post_cologne`, a FRESH
rebuild through the unmodified `modern_map_feature_engine.ModernMapStateStore`
/ `apply_selected_map_team_result` / `apply_selected_map_player_observation`
engine, gated by the deployment-history manifest. Mirrors
`feature_engineering/state/build_pre_cologne_modern_map_state_v1.py`.

The engine has no map-name allowlist of any kind (confirmed by direct
inspection - any map_name string is accepted generically), so eligibility
here is governed purely by team-identity eligibility and by whether the
underlying map/player rows exist at all (same 7-match map_base gap as the
map/roster engines, since both of this engine's input streams are built on
top of map_base.parquet via modern_map_stream_common.load_modern_map_streams).
"""

import pandas as pd

from _common import INTERIM, ROOT, raw_file_hashes
from feature_engineering.maps.modern_map_feature_engine import (
    MODERN_MAP_ENGINE_VERSION, MAP_FORM_HALF_LIFE_DAYS,
    ModernMapStateStore, apply_selected_map_team_result, apply_selected_map_player_observation,
)
from feature_engineering.maps.modern_map_stream_common import load_modern_map_streams

DEPLOY = ROOT / "data" / "deployment"
MANIFEST_PATH = DEPLOY / "deployment_history_manifest_v1.parquet"
CANONICAL_COLOGNE_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"


def build(output_path=None):
    hashes_before = raw_file_hashes()
    output_path = output_path or (INTERIM / "modern_map_state_v1_deployment_post_cologne.json")

    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    included_ids = set(manifest.loc[manifest["history_status"] == "included", "match_id"].astype(int))
    official_cologne_ids = set(pd.read_parquet(CANONICAL_COLOGNE_PATH, engine="fastparquet")
                                ["source_match_id"].astype(int))
    deployment_cutoff = manifest.loc[manifest["history_status"] == "included", "datetime"].max()

    map_rows, player_rows, info = load_modern_map_streams(
        evaluation_groups=("development", "cologne_2026", "post_cologne"))
    before_map_n, before_player_n = len(map_rows), len(player_rows)
    map_rows = map_rows[map_rows["match_id"].isin(included_ids)].copy()
    player_rows = player_rows[player_rows["match_id"].isin(included_ids)].copy()
    print(f"deployment modern-map stream: map {before_map_n} -> {len(map_rows)}, "
          f"player {before_player_n} -> {len(player_rows)} after manifest anti-join")

    matches_with_map_rows = set(map_rows["match_id"].astype(int))
    matches_with_player_rows = set(player_rows["match_id"].astype(int))
    cologne_without_any_rows = official_cologne_ids - (matches_with_map_rows | matches_with_player_rows)

    store = ModernMapStateStore()
    for _, r in map_rows.iterrows():
        apply_selected_map_team_result(store, r)
    for _, r in player_rows.iterrows():
        apply_selected_map_player_observation(store, r)

    consumed_team_map_ids = {int(h.match_id) for hist in store.team_map.values() for h in hist}
    game_id_to_match_id = dict(zip(player_rows["game_id"].astype(int), player_rows["match_id"].astype(int)))
    consumed_player_map_ids = {game_id_to_match_id[int(h.game_id)] for hist in store.player_map.values()
                                for h in hist if int(h.game_id) in game_id_to_match_id}

    # ---- consumption audit (amendment #3/#4) at match_id granularity ----
    team_elig_by_match = map_rows.groupby("match_id").apply(
        lambda g: bool((g["team1_eligible"] & g["team2_eligible"]).any())).to_dict() if len(map_rows) else {}
    any_team_elig_by_match = map_rows.groupby("match_id").apply(
        lambda g: bool((g["team1_eligible"] | g["team2_eligible"]).any())).to_dict() if len(map_rows) else {}
    player_usable_by_match = player_rows.groupby("match_id")["has_usable_stats"].any().to_dict() \
        if len(player_rows) else {}

    audit_rows = []
    for mid in included_ids:
        mid = int(mid)
        has_map, has_player = mid in matches_with_map_rows, mid in matches_with_player_rows
        if not has_map and not has_player:
            eligible, consumed = False, False
            reason = "no surviving map_base-derived rows (map or player) for this match_id (Phase 2 " \
                     "structural, independent of team identity)"
        else:
            team_eligible_any = any_team_elig_by_match.get(mid, False)
            player_usable = player_usable_by_match.get(mid, False)
            eligible = bool(team_eligible_any or player_usable)
            consumed = (mid in consumed_team_map_ids) or (mid in consumed_player_map_ids)
            parts = []
            if team_eligible_any:
                parts.append("team_map ledger: >=1 identity-eligible side")
            if player_usable:
                parts.append("player_map ledger: >=1 observation with usable stats (team-independent)")
            reason = "; ".join(parts) if parts else "map/player rows exist but neither ledger's admission " \
                                                      "criterion was met"
        audit_rows.append({"state_type": "modern_map", "match_id": mid, "deployment_history_status": "included",
                            "eligible_for_state": eligible, "consumed_by_state": consumed,
                            "eligibility_reason": reason})
        if eligible and not consumed:
            raise ValueError(f"STOP: match_id {mid} is eligible for the modern-map state but was not consumed.")

    n_official_with_rows = len(official_cologne_ids) - len(cologne_without_any_rows)
    print(f"official Cologne matches with surviving modern-map rows: {n_official_with_rows}/106 "
          f"({len(cologne_without_any_rows)} legitimately excluded)")

    all_series_dt = [h.series_dt for hist in store.team_map.values() for h in hist] + \
        [h.series_dt for hist in store.player_map.values() for h in hist]
    max_history_dt = max(all_series_dt, default=None)
    if max_history_dt is not None and str(max_history_dt) > str(deployment_cutoff):
        raise ValueError(f"STOP: a modern-map history entry ({max_history_dt}) is later than the deployment cutoff")

    meta = {
        "historical_replay_state": "pre_cologne", "deployment_state": "deployment_post_cologne",
        "modern_map_engine_version": MODERN_MAP_ENGINE_VERSION,
        "snapshot_id": "modern_map_state_v1_deployment_post_cologne",
        "state_source": "deployment_history_manifest_v1.parquet + canonical modern-map streams (fresh rebuild)",
        "companion_deployment_snapshots": [
            "data/interim/map_state_v1_deployment_post_cologne.json",
            "data/interim/player_roster_state_v1_deployment_post_cologne.json",
        ],
        "deployment_history_cutoff": str(deployment_cutoff),
        "map_form_half_life_days": MAP_FORM_HALF_LIFE_DAYS,
        "team_map_ledgers": len(store.team_map), "player_map_ledgers": len(store.player_map),
        "team_map_roster_ledgers": len(store.team_map_roster),
        "official_cologne_matches_with_rows": n_official_with_rows,
        "official_cologne_matches_without_rows": sorted(int(x) for x in cologne_without_any_rows),
        "no_map_name_allowlist_note": "This engine accepts any map_name string generically - it does not "
                                       "filter to an Active-Duty pool. The deployment cutoff (2026-06-28) "
                                       "predates any later Active-Duty change, so no legitimate post-change "
                                       "map experience (e.g. Cache) exists in this snapshot regardless.",
        "generated_at": str(pd.Timestamp.now()),
    }
    store.to_json(output_path, meta=meta)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the deployment build"
    print(f"Wrote {output_path} ({len(store.team_map)} team-map, {len(store.player_map)} player-map, "
          f"{len(store.team_map_roster)} team-map-roster ledgers)")
    return store, pd.DataFrame(audit_rows), meta


if __name__ == "__main__":
    build()
