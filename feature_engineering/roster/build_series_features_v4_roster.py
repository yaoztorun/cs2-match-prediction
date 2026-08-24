"""
Phase 5C - build data/features/series_features_v4_roster.parquet.

Extends V3 (series_features_v3_form.parquet) with leakage-safe player-form and
roster-stability features produced by feature_engineering/roster/player_roster_feature_engine.py.

Prediction task: PRE-VETO SERIES. The target series' actual lineup and box
score are FORBIDDEN as inputs; each team's expected lineup is inferred from
that team's strictly-prior appearances only.

Contract with V3 (enforced here and re-checked in validate_phase5c.py):
  * identical match_id universe (9,456), identical targets, identical ordering;
  * every V3 feature column preserved value-for-value;
  * exactly 21 player/roster columns appended.

Every one of the 9,456 series gets roster features - including the roughly
half that have no map/player rows of their own, because roster features come
from the two teams' PRIOR history, not from the target match. That is why the
build drives the engine with an explicit series-request frame.

Read-only against data/raw/ and data/interim/.
"""

import json

import numpy as np
import pandas as pd
import yaml

from _common import INTERIM, ROOT, raw_file_hashes
from feature_engineering.maps.map_stream_common import canonical_eligibility_map
from feature_engineering.roster.player_roster_feature_engine import (
    PLAYER_ROSTER_ENGINE_VERSION, ROSTER_LOOKBACK_DAYS, PLAYER_FORM_HALF_LIFE_DAYS, ROSTER_SIZE,
    PlayerRosterStateStore, process_player_roster_stream,
    ROSTER_DIRECTIONAL_FEATURES, ROSTER_SYMMETRIC_FEATURES, ROSTER_PERFORMANCE_DIFFS,
)
from feature_engineering.roster.player_roster_stream_common import load_player_roster_stream, MIN_ROUNDS_FOR_VALID_MAP

FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "features" / "series_features_v4_roster.yaml"
V3_PATH = FEATURES_DIR / "series_features_v3_form.parquet"

EXPECTED_ROWS = 9456


def main():
    hashes_before = raw_file_hashes()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    v3 = pd.read_parquet(V3_PATH, engine="fastparquet")
    print(f"series_features_v3_form: {len(v3)} rows x {len(v3.columns)} cols")
    assert len(v3) == EXPECTED_ROWS

    # ---- 1. the canonical player observation stream (authoritative series datetime) ----
    stream, info = load_player_roster_stream(evaluation_groups=("development",))
    print(f"player observations: {len(stream)} "
          f"({info['distinct_players']} players, {info['distinct_maps']} maps, "
          f"{info['distinct_matches']} matches)")
    print(f"  malformed handling: {info['maps_excluded_player_on_both_sides']} maps excluded "
          f"(player on both sides), {info['duplicate_player_slots_collapsed']} duplicate slot groups "
          f"collapsed, {info['duplicate_player_slots_conflicting_excluded']} conflicting excluded")

    # ---- 2. one PRE-VETO request per V3 series, whether or not it has player rows ----
    canon_elig = canonical_eligibility_map()
    reqs = pd.DataFrame({
        "match_id": v3["match_id"].astype(str),
        "series_datetime": v3["datetime"],
        "team1_canonical": v3["team1_canonical"],
        "team2_canonical": v3["team2_canonical"],
        "team1_eligible": v3["team1_canonical"].map(canon_elig),
        "team2_eligible": v3["team2_canonical"].map(canon_elig),
    })
    assert reqs["team1_eligible"].all() and reqs["team2_eligible"].all(), \
        "series_features_v3 contains an identity-ineligible pair - V3/V4 coverage would diverge"

    matches_with_own_players = int(reqs["match_id"].isin(set(stream["match_id"].astype(str))).sum())
    print(f"series requests: {len(reqs)} ({matches_with_own_players} have own player rows, "
          f"{len(reqs) - matches_with_own_players} do not)")

    # ---- 3. run the engine ----
    store = PlayerRosterStateStore()
    emitted = process_player_roster_stream(store, stream, reqs, emit_features=True)
    roster = pd.DataFrame(emitted)
    print(f"roster feature rows emitted: {len(roster)}")
    assert len(roster) == len(v3), f"emitted {len(roster)} rows for {len(v3)} series"

    new_cols = ROSTER_DIRECTIONAL_FEATURES + ROSTER_SYMMETRIC_FEATURES
    assert len(new_cols) == 21, f"expected 21 new features, got {len(new_cols)}"
    declared_new = set(cfg["directional_features"] + cfg["symmetric_features"]) - set(v3.columns)
    assert set(new_cols) == declared_new, \
        f"engine columns and the config whitelist disagree: {set(new_cols) ^ declared_new}"

    # ---- 4. structural invariants of the two state types ----
    for pid, st in store.players.items():
        keys = [(h.game_id, pid) for h in st.history]
        assert len(keys) == len(set(keys)), f"player {pid} has duplicate (game_id, player_id) entries"
    for team, st in store.teams.items():
        keys = [(a.game_id, a.player_id) for a in st.appearances]
        assert len(keys) == len(set(keys)), f"team {team} has duplicate (game_id, player_id) appearances"
    print(f"state: {len(store.players)} players, {len(store.teams)} teams, "
          f"{len(store.player_map_keys)} player-map entries, {len(store.appearance_keys)} appearances")

    # ---- 5. append to V3, preserving V3 exactly ----
    roster["match_id"] = roster["match_id"].astype(str)
    v3_key = v3.copy()
    v3_key["_mid"] = v3_key["match_id"].astype(str)
    merged = v3_key.merge(roster[["match_id"] + new_cols].rename(columns={"match_id": "_mid"}),
                           on="_mid", how="left", validate="one_to_one").drop(columns=["_mid"])
    assert len(merged) == len(v3)
    assert merged["match_id"].tolist() == v3["match_id"].tolist(), "row ordering diverged from V3"
    for c in v3.columns:
        a, b = v3[c], merged[c]
        if pd.api.types.is_numeric_dtype(a):
            assert np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), c
        else:
            assert a.equals(b), c

    final = merged[list(v3.columns) + new_cols]

    # ---- 6. guard rails ----
    forbidden = {"map_name", "map_id", "game_id", "score1_game", "score2_game", "team1_map_win",
                 "score1", "score2", "score1_match", "score2_match", "team1_win", "player_id"}
    leaked = forbidden & set(final.columns)
    assert not leaked, f"target-series-leaking column reached the table: {leaked}"
    player_tokens = ("player1", "player2", "player3", "player4", "player5",
                     "kills", "deaths", "kddiff", "kast_raw", "adr_raw")
    bad = [c for c in new_cols if any(t in c.lower() for t in player_tokens)]
    assert not bad, f"raw player identity/box-score column reached the table: {bad}"

    # NaN is permitted ONLY on the ten performance diffs, and ONLY where the
    # confidence feature says there is no evidence on at least one side.
    non_perf = [c for c in new_cols if c not in ROSTER_PERFORMANCE_DIFFS]
    assert np.isfinite(final[non_perf].to_numpy(dtype=float)).all(), \
        "a non-performance roster feature is non-finite"
    no_evidence = final["roster_form_players_min"] == 0
    for c in ROSTER_PERFORMANCE_DIFFS:
        isna = final[c].isna()
        assert isna.equals(no_evidence), \
            (f"{c}: NaN pattern does not match roster_form_players_min == 0 "
             f"({int(isna.sum())} NaN vs {int(no_evidence.sum())} no-evidence rows)")
        assert np.isfinite(final.loc[~isna, c].to_numpy(dtype=float)).all(), f"{c} has a non-finite value"
    print(f"cold-start rows (no usable player history on >=1 side): {int(no_evidence.sum())} "
          f"({100 * no_evidence.mean():.2f}%)")

    final.to_parquet(FEATURES_DIR / "series_features_v4_roster.parquet",
                     engine="fastparquet", index=False)

    # ---- 7. audit-only per-side raw roster values ----
    audit_cols = ["match_id", "series_datetime"] + \
        [c for c in roster.columns if c.startswith("team1_") or c.startswith("team2_")]
    audit = roster[audit_cols].sort_values(["series_datetime", "match_id"]).reset_index(drop=True)
    audit.to_parquet(FEATURES_DIR / "series_roster_states_v1.parquet", engine="fastparquet", index=False)

    summary = {
        "player_roster_engine_version": PLAYER_ROSTER_ENGINE_VERSION,
        "task_id": "series_features_v4_roster",
        "prediction_task": "pre_veto_series",
        "roster_lookback_days": ROSTER_LOOKBACK_DAYS,
        "player_form_half_life_days": PLAYER_FORM_HALF_LIFE_DAYS,
        "roster_size": ROSTER_SIZE,
        "min_rounds_for_valid_map": MIN_ROUNDS_FOR_VALID_MAP,
        "rows": int(len(final)),
        "v3_rows": int(len(v3)),
        "n_new_directional": len(ROSTER_DIRECTIONAL_FEATURES),
        "n_new_symmetric": len(ROSTER_SYMMETRIC_FEATURES),
        "n_new_total": len(new_cols),
        "n_total_features": len(cfg["directional_features"]) + len(cfg["symmetric_features"])
                             + len(cfg["categorical_context"]),
        "matches_with_own_player_rows": matches_with_own_players,
        "matches_without_own_player_rows": int(len(reqs) - matches_with_own_players),
        "rows_with_no_usable_player_history": int(no_evidence.sum()),
        "state_players": len(store.players),
        "state_teams": len(store.teams),
        "state_player_map_entries": len(store.player_map_keys),
        "state_appearances": len(store.appearance_keys),
        "stream_info": info,
    }
    with open(INTERIM / "series_features_v4_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"\nWrote {FEATURES_DIR / 'series_features_v4_roster.parquet'} "
          f"({len(final)} rows x {len(final.columns)} cols)")
    print(f"Wrote {FEATURES_DIR / 'series_roster_states_v1.parquet'} (AUDIT ONLY, {len(audit)} rows)")
    print(f"Wrote {INTERIM / 'series_features_v4_build_summary.json'}")


if __name__ == "__main__":
    main()
