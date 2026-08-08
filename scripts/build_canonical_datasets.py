"""
Build canonical series-level and map-level datasets (Phase 2, items 2-4).

Reads the raw games CSVs and data/interim/team_aliases.csv (must already
exist - run scripts/team_identity_analysis.py first), and writes:

    data/interim/series_base.parquet
    data/interim/map_base.parquet
    data/interim/rejected_series_rows.csv
    data/interim/rejected_map_rows.csv
    data/interim/evaluation_manifest.csv

No feature engineering, no ELO, no rolling stats, no modelling. Read-only
against data/raw/ (verified via a before/after content-hash check).
"""

import numpy as np
import pandas as pd

from _common import (
    load_games_tiered, raw_file_hashes, to_num,
    PLAYER_ID_COLS, PLAYER_NAME_COLS, PLAYER_STAT_COLS, INTERIM,
)

COLOGNE_2026 = "IEM Cologne Major 2026"


def normalize_bestof(series):
    """Parse '1'/'1.0'/'3'/'3.0'/'5'/'5.0'/'' -> nullable Int64 in {1,3,5} or <NA>."""
    num = to_num(series)
    valid = num.isin([1.0, 3.0, 5.0])
    out = pd.array([int(v) if ok else pd.NA for v, ok in zip(num, valid)], dtype="Int64")
    return out


def join_canonical(df, aliases, col):
    m = aliases.set_index("original_team_name")["canonical_team_name"]
    return df[col].map(m).fillna(df[col])


def build_series(df, aliases, hashes_before):
    tt = df[df["is_total"] == "True"].copy()
    tt["score1_match_num"] = to_num(tt["score1_match"])
    tt["score2_match_num"] = to_num(tt["score2_match"])
    tt["bestOf_norm"] = normalize_bestof(tt["bestOf"])

    reasons = pd.Series([""] * len(tt), index=tt.index)

    missing_score = tt["score1_match_num"].isna() | tt["score2_match_num"].isna()
    tie = (~missing_score) & (tt["score1_match_num"] == tt["score2_match_num"])
    missing_bestof = tt["bestOf_norm"].isna()

    for mask, name in [(missing_score, "missing_score"), (tie, "tie"), (missing_bestof, "missing_bestOf")]:
        reasons = reasons.mask(mask & (reasons != ""), reasons + ";" + name)
        reasons = reasons.mask(mask & (reasons == ""), name)

    tt["reject_reason"] = reasons
    rejected = tt[tt["reject_reason"] != ""].copy()
    retained = tt[tt["reject_reason"] == ""].copy()

    retained["team1_series_win"] = (retained["score1_match_num"] > retained["score2_match_num"]).astype("Int64")

    retained["team1_canonical"] = join_canonical(retained, aliases, "team1")
    retained["team2_canonical"] = join_canonical(retained, aliases, "team2")
    retained["datetime_parsed"] = pd.to_datetime(retained["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    series_base = pd.DataFrame({
        "match_id": to_num(retained["match_id"]).astype("Int64"),
        "tournament": retained["tournament"],
        "tier": retained["tier"],
        "team1": retained["team1"],
        "team1_canonical": retained["team1_canonical"],
        "team1_id": to_num(retained["team1_id"]).astype("Int64"),
        "team2": retained["team2"],
        "team2_canonical": retained["team2_canonical"],
        "team2_id": to_num(retained["team2_id"]).astype("Int64"),
        "bestOf": retained["bestOf_norm"],
        "datetime": retained["datetime_parsed"],
        "score1_match": retained["score1_match_num"].astype("Int64"),
        "score2_match": retained["score2_match_num"].astype("Int64"),
        "team1_series_win": retained["team1_series_win"],
    }).reset_index(drop=True)

    assert set(series_base.columns).isdisjoint({"team1_win"})
    assert series_base["match_id"].is_unique, "series_base must have one row per match_id"
    assert series_base["team1_series_win"].isin([0, 1]).all()
    assert series_base["bestOf"].dropna().isin([1, 3, 5]).all()

    return series_base, rejected, tt


def build_map(df, aliases):
    tf = df[df["is_total"] == "False"].copy()
    tf["score1_game_num"] = to_num(tf["score1_game"])
    tf["score2_game_num"] = to_num(tf["score2_game"])
    tf["bestOf_norm"] = normalize_bestof(tf["bestOf"])  # kept, NEVER a rejection criterion

    map_id_blank = (tf["map_id"] == "") | (tf["map_name"] == "")
    score_blank = tf["score1_game_num"].isna() | tf["score2_game_num"].isna()

    blank_map_row = score_blank & map_id_blank
    missing_map_name = (~score_blank) & map_id_blank
    tie_or_other = (~map_id_blank) & (~score_blank) & (tf["score1_game_num"] == tf["score2_game_num"])
    other_invalid = (~map_id_blank) & score_blank  # map identity known but score missing - still not usable

    reasons = pd.Series([""] * len(tf), index=tf.index)
    for mask, name in [
        (blank_map_row, "blank_map_row"),
        (missing_map_name, "missing_map_name"),
        (tie_or_other, "tie"),
        (other_invalid, "other"),
    ]:
        reasons = reasons.mask(mask & (reasons != ""), reasons + ";" + name)
        reasons = reasons.mask(mask & (reasons == ""), name)

    tf["reject_reason"] = reasons
    rejected = tf[tf["reject_reason"] != ""].copy()
    retained = tf[tf["reject_reason"] == ""].copy()

    retained["team1_map_win"] = (retained["score1_game_num"] > retained["score2_game_num"]).astype("Int64")
    retained["team1_canonical"] = join_canonical(retained, aliases, "team1")
    retained["team2_canonical"] = join_canonical(retained, aliases, "team2")
    retained["datetime_parsed"] = pd.to_datetime(retained["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    data = {
        "match_id": to_num(retained["match_id"]).astype("Int64"),
        "game_id": to_num(retained["game_id"]).astype("Int64"),
        "map_id": to_num(retained["map_id"]).astype("Int64"),
        "map_name": retained["map_name"],
        "tier": retained["tier"],
        "tournament": retained["tournament"],
        "datetime": retained["datetime_parsed"],
        "bestOf": retained["bestOf_norm"],
        "team1": retained["team1"],
        "team1_canonical": retained["team1_canonical"],
        "team1_id": to_num(retained["team1_id"]).astype("Int64"),
        "team2": retained["team2"],
        "team2_canonical": retained["team2_canonical"],
        "team2_id": to_num(retained["team2_id"]).astype("Int64"),
    }
    for c in PLAYER_ID_COLS:
        data[c] = to_num(retained[c]).astype("Int64")
    for c in PLAYER_NAME_COLS:
        data[c] = retained[c]
    for c in PLAYER_STAT_COLS:
        data[c] = to_num(retained[c])
    data["score1_game"] = retained["score1_game_num"].astype("Int64")
    data["score2_game"] = retained["score2_game_num"].astype("Int64")
    data["team1_map_win"] = retained["team1_map_win"]

    map_base = pd.DataFrame(data).reset_index(drop=True)

    assert set(map_base.columns).isdisjoint({"team1_win", "score1_match", "score2_match", "games_played"})
    assert map_base["game_id"].is_unique, "map_base must have one row per game_id"
    assert map_base["team1_map_win"].isin([0, 1]).all()

    return map_base, rejected, tf


def build_evaluation_manifest(all_series_rows):
    tt = all_series_rows.copy()
    tt["dt"] = pd.to_datetime(tt["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    cologne_mask = tt["tournament"] == COLOGNE_2026
    cologne_max_dt = tt.loc[cologne_mask, "dt"].max()

    group = pd.Series("development", index=tt.index)
    group = group.mask(cologne_mask, "cologne_2026")
    group = group.mask((~cologne_mask) & (tt["dt"] > cologne_max_dt), "post_cologne")

    manifest = pd.DataFrame({
        "match_id": to_num(tt["match_id"]).astype("Int64"),
        "evaluation_group": group,
    }).drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    assert manifest["match_id"].is_unique
    assert set(manifest["evaluation_group"].unique()) <= {"development", "cologne_2026", "post_cologne"}
    return manifest, cologne_max_dt


def main():
    hashes_before = raw_file_hashes()

    df = load_games_tiered()
    aliases = pd.read_csv(INTERIM / "team_aliases.csv", dtype=str, keep_default_na=False)

    series_base, rejected_series, all_series = build_series(df, aliases, hashes_before)
    map_base, rejected_map, all_map = build_map(df, aliases)
    manifest, cologne_max_dt = build_evaluation_manifest(all_series)

    # every map's match_id must exist somewhere in the raw match_id universe (all series rows),
    # even if that series row was itself rejected
    known_match_ids = set(to_num(all_series["match_id"]).astype("Int64"))
    assert set(map_base["match_id"]) <= known_match_ids, "map_base has match_ids outside the raw match_id universe"

    # pyarrow's native DLL is blocked by this machine's Application Control policy;
    # fastparquet is used as the parquet engine instead (verified round-trip of
    # nullable Int64 dtypes before relying on it here).
    series_base.to_parquet(INTERIM / "series_base.parquet", engine="fastparquet", index=False)
    map_base.to_parquet(INTERIM / "map_base.parquet", engine="fastparquet", index=False)
    rejected_series.to_csv(INTERIM / "rejected_series_rows.csv", index=False, encoding="utf-8")
    rejected_map.to_csv(INTERIM / "rejected_map_rows.csv", index=False, encoding="utf-8")
    manifest.to_csv(INTERIM / "evaluation_manifest.csv", index=False, encoding="utf-8")

    hashes_after = raw_file_hashes()
    assert hashes_before == hashes_after, "data/raw/ was modified during Phase 2 build - this must never happen"

    print(f"series_base: {len(series_base)} retained / {len(rejected_series)} rejected "
          f"(of {len(all_series)} total series rows)")
    print(f"  rejected reasons: {rejected_series['reject_reason'].value_counts().to_dict()}")
    print(f"map_base: {len(map_base)} retained / {len(rejected_map)} rejected "
          f"(of {len(all_map)} total map rows)")
    print(f"  rejected reasons: {rejected_map['reject_reason'].value_counts().to_dict()}")
    print(f"evaluation_manifest: {len(manifest)} matches, groups={manifest['evaluation_group'].value_counts().to_dict()}")
    print(f"  cologne_2026 max datetime: {cologne_max_dt}")
    print("raw/ files unchanged: OK")


if __name__ == "__main__":
    main()
