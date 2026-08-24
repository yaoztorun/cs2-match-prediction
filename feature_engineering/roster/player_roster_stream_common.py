"""
Dataset-specific glue between the Kaggle export and the Phase 5C
player/roster engine (feature_engineering/roster/player_roster_feature_engine.py). The engine
itself knows nothing about "Kaggle", "development" or "Cologne" - every such
concept lives here, mirroring feature_engineering/maps/map_stream_common.py's role for the
Phase 5A map engine.

AUTHORITATIVE SERIES DATETIME (do not reintroduce a map-derived cutoff)
-----------------------------------------------------------------------
The prediction cutoff for a series is the AUTHORITATIVE series datetime taken
from the canonical series source (data/interim/series_base.parquet, which is
also where series_features_v1/v2/v3's `datetime` column comes from), joined
onto every map/player observation of that match_id. It is deliberately NOT
`groupby(match_id).datetime.min()` over map rows: map timestamps are
PROVENANCE ONLY and must never decide what a series may see. Every player
observation of a series therefore carries one identical `series_datetime`,
and a map whose own timestamp differs from it still cannot leak into its own
series' pre-series features.

MALFORMED SOURCE ROWS (quantified in the Phase 5C schema audit)
---------------------------------------------------------------
The raw export contains two structural defects that would silently corrupt a
per-player streaming state. Both are handled here, at load time, and counted
in the returned info dict:

  1. The SAME player_id appearing on BOTH sides of one map (5 maps). That
     player would be credited with a win and a loss for one map, and both
     teams would record them as a roster member. -> the ENTIRE map is
     excluded from player-performance AND roster-appearance updates.
  2. The same player_id occupying MORE THAN ONE slot on one side (14 maps),
     which also means the real 5th player is missing. -> collapsed to a
     single observation when the duplicated slots carry consistent stats;
     if the duplicated slots DISAGREE, that player's entry is excluded from
     that map (the other players on the map are unaffected).

Consequently a completeness test must use `nunique` over player ids, never
`notna().sum()`.

Never read here: `kddiff` (100% collinear with kills-deaths) and the player
NAME columns (1,316 slots carry an id with a blank name, so ids are the
strictly broader and only reliable identity signal).
"""

import numpy as np
import pandas as pd

from _common import INTERIM
from feature_engineering.maps.map_stream_common import canonical_eligibility_map

SOURCE = "kaggle_ektarr"

# Minimum rounds for a map's player box score to be treated as usable. The
# two sub-5-round rows in the export are forfeits, whose per-round rates
# would be meaningless. Fixed engineering constant, not tuned.
MIN_ROUNDS_FOR_VALID_MAP = 5

STAT_SUFFIXES = ["kills", "deaths", "assists", "adr", "kast"]   # kddiff deliberately excluded
_STAT_TUPLE_COLS = ["kills", "deaths", "assists", "adr", "kast"]


def _map_base_columns():
    base = ["match_id", "game_id", "map_name", "datetime", "tier", "bestOf",
            "team1_canonical", "team2_canonical", "score1_game", "score2_game"]
    for t in (1, 2):
        for i in range(1, 6):
            base.append(f"team{t}_player{i}_id")
            for s in STAT_SUFFIXES:
                base.append(f"team{t}_player{i}_{s}")
    return base


def _melt_players(mb):
    """Wide -> long. One row per (game_id, side, slot) with a non-null player
    id. The side is read straight from the column name, which is why the
    player -> team assignment needs no inference."""
    frames = []
    for side in (1, 2):
        for slot in range(1, 6):
            frames.append(pd.DataFrame({
                "match_id": mb["match_id"].to_numpy(),
                "game_id": mb["game_id"].to_numpy(),
                "series_datetime": mb["series_datetime"].to_numpy(),
                "map_datetime": mb["datetime"].to_numpy(),
                "side": side,
                "slot": slot,
                "team_canonical": mb[f"team{side}_canonical"].to_numpy(),
                "team_eligible": mb[f"team{side}_eligible"].to_numpy(),
                "rounds": mb["rounds"].to_numpy(),
                "player_id": mb[f"team{side}_player{slot}_id"].to_numpy(),
                **{s: mb[f"team{side}_player{slot}_{s}"].to_numpy() for s in STAT_SUFFIXES},
            }))
    long = pd.concat(frames, ignore_index=True)
    long = long[long["player_id"].notna()].reset_index(drop=True)
    long["player_id"] = long["player_id"].astype("int64")
    return long


def _handle_malformed(long):
    """Applies the two documented malformed-row rules. Returns
    (clean_long, info)."""
    info = {}

    # ---- rule 1: same player on BOTH sides of one map -> drop the whole map ----
    sides_per = long.groupby(["game_id", "player_id"])["side"].nunique()
    both_sides = sides_per[sides_per > 1]
    bad_games = set(long.loc[
        long.set_index(["game_id", "player_id"]).index.isin(both_sides.index), "game_id"])
    info["maps_excluded_player_on_both_sides"] = int(len(bad_games))
    info["excluded_both_sides_game_ids"] = sorted(int(g) for g in bad_games)
    long = long[~long["game_id"].isin(bad_games)].reset_index(drop=True)

    # ---- rule 2: same player in >1 slot on one side ----
    grp = long.groupby(["game_id", "side", "player_id"])
    sizes = grp.size()
    dup_keys = sizes[sizes > 1].index
    info["duplicate_player_slot_groups"] = int(len(dup_keys))
    if len(dup_keys):
        dup_mask = long.set_index(["game_id", "side", "player_id"]).index.isin(dup_keys)
        dup_rows = long[dup_mask]
        # consistent == every duplicated slot carries an identical stat tuple
        # (NaN == NaN counts as consistent, hence fillna to a sentinel first).
        consistency = dup_rows.assign(
            **{c: dup_rows[c].fillna(-999999.0) for c in _STAT_TUPLE_COLS}
        ).groupby(["game_id", "side", "player_id"])[_STAT_TUPLE_COLS].nunique().max(axis=1)
        conflicting = set(consistency[consistency > 1].index)
        collapsible = set(consistency[consistency <= 1].index)
        info["duplicate_player_slots_collapsed"] = int(len(collapsible))
        info["duplicate_player_slots_conflicting_excluded"] = int(len(conflicting))

        idx = long.set_index(["game_id", "side", "player_id"]).index
        if conflicting:
            long = long[~idx.isin(conflicting)].reset_index(drop=True)
        # collapse: keep the first slot only, per (game_id, side, player_id)
        long = long.sort_values(["game_id", "side", "slot"]).drop_duplicates(
            subset=["game_id", "side", "player_id"], keep="first").reset_index(drop=True)
    else:
        info["duplicate_player_slots_collapsed"] = 0
        info["duplicate_player_slots_conflicting_excluded"] = 0

    return long, info


def load_player_roster_stream(evaluation_groups=("development",), max_exclusive_datetime=None):
    """Returns (long_player_observations, info). One row per
    (game_id, side, player_id), carrying the AUTHORITATIVE `series_datetime`.

    `max_exclusive_datetime`, if given, keeps only observations with
    `series_datetime < cutoff` (used by the pre-Cologne snapshot builder)."""
    mb = pd.read_parquet(INTERIM / "map_base.parquet", engine="fastparquet",
                         columns=_map_base_columns())
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet",
                         columns=["match_id", "datetime"])
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")

    info = {"map_base_rows": int(len(mb))}

    # ---- AUTHORITATIVE series datetime: from the canonical SERIES source ----
    authoritative = sb.rename(columns={"datetime": "series_datetime"})
    before = len(mb)
    mb = mb.merge(authoritative, on="match_id", how="inner", validate="many_to_one")
    info["map_rows_without_canonical_series_row"] = int(before - len(mb))
    assert mb["series_datetime"].notna().all(), "authoritative series datetime missing after join"
    info["map_rows_whose_map_datetime_differs_from_series_datetime"] = int(
        (pd.to_datetime(mb["datetime"]) != pd.to_datetime(mb["series_datetime"])).sum())

    mb = mb.merge(em, on="match_id", how="inner")
    mb = mb[mb["evaluation_group"].isin(evaluation_groups)].reset_index(drop=True)
    info["rows_selected_by_evaluation_group"] = int(len(mb))

    if max_exclusive_datetime is not None:
        cutoff = pd.Timestamp(max_exclusive_datetime)
        n_before = len(mb)
        mb = mb[mb["series_datetime"] < cutoff].reset_index(drop=True)
        info["rows_removed_by_datetime_cutoff"] = int(n_before - len(mb))
    else:
        info["rows_removed_by_datetime_cutoff"] = 0

    canon_elig = canonical_eligibility_map()
    for side in (1, 2):
        elig = mb[f"team{side}_canonical"].map(canon_elig)
        if elig.isna().any():
            missing = set(mb.loc[elig.isna(), f"team{side}_canonical"])
            raise ValueError(f"canonical names missing from team_identity_policy.csv: {missing}")
        mb[f"team{side}_eligible"] = elig.astype(bool)

    mb["rounds"] = mb["score1_game"].astype("int64") + mb["score2_game"].astype("int64")

    long = _melt_players(mb)
    info["player_slot_observations_before_cleaning"] = int(len(long))

    long, malformed_info = _handle_malformed(long)
    info.update(malformed_info)
    info["player_slot_observations_after_cleaning"] = int(len(long))

    # ---- derived per-observation statistics ----
    kills = long["kills"].to_numpy(dtype=float)
    deaths = long["deaths"].to_numpy(dtype=float)
    assists = long["assists"].to_numpy(dtype=float)
    rounds = long["rounds"].to_numpy(dtype=float)

    denom = np.maximum(kills + deaths, 1.0)
    long["kd_balance"] = (kills - deaths) / denom           # bounded [-1, 1], cannot explode
    long["assists_per_round"] = assists / np.maximum(rounds, 1.0)
    # adr and kast are ALREADY round-normalized rates in the source - never re-divide.

    stats_present = long[_STAT_TUPLE_COLS].notna().all(axis=1)
    rounds_ok = long["rounds"] >= MIN_ROUNDS_FOR_VALID_MAP
    long["has_usable_stats"] = (stats_present & rounds_ok).to_numpy()

    info["observations_with_usable_stats"] = int(long["has_usable_stats"].sum())
    info["observations_without_usable_stats"] = int((~long["has_usable_stats"]).sum())
    info["distinct_players"] = int(long["player_id"].nunique())
    info["distinct_maps"] = int(long["game_id"].nunique())
    info["distinct_matches"] = int(long["match_id"].nunique())
    info["min_series_datetime"] = str(long["series_datetime"].min()) if len(long) else None
    info["max_series_datetime"] = str(long["series_datetime"].max()) if len(long) else None

    long = long.sort_values(["series_datetime", "match_id", "game_id", "side", "slot"]).reset_index(drop=True)
    return long, info
