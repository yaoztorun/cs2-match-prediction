"""
Dataset-specific glue between the Kaggle export and the Phase 5A map engine.

The engine (map_feature_engine.py) knows nothing about "Kaggle", "development"
or "Cologne" - every such concept lives here, so the same engine can later be
pointed at a different provider. Shared by build_map_features_v1.py,
build_series_features_v2_map_pool.py and build_pre_cologne_map_state_v1.py so
that all three see byte-identical inputs.
"""

import pandas as pd

from _common import INTERIM

SOURCE = "kaggle_ektarr"

MAP_BASE_COLUMNS = [
    "match_id", "game_id", "map_id", "map_name", "tier", "tournament", "datetime",
    "bestOf", "team1", "team1_canonical", "team2", "team2_canonical",
    "score1_game", "score2_game", "team1_map_win",
]


def canonical_eligibility_map():
    """Identical derivation to Phase 3 (build_series_features_v1.py:38) - a
    canonical name is eligible only if EVERY raw alias mapping to it is."""
    policy = pd.read_csv(INTERIM / "team_identity_policy.csv")
    return policy.groupby("canonical_team_name")["identity_feature_eligible"].min().astype(bool).to_dict()


def assert_map_target_integrity(df):
    """Re-assert the target from the raw scores rather than trusting the stored
    flag. Phase 2 already cleaned this; the check is cheap and the cost of a
    silent target error is total."""
    problems = []
    if df["score1_game"].isna().any() or df["score2_game"].isna().any():
        problems.append(f"null map scores: {int(df[['score1_game', 'score2_game']].isna().any(axis=1).sum())} rows")
    ties = int((df["score1_game"] == df["score2_game"]).sum())
    if ties:
        problems.append(f"tied maps (no defined winner): {ties} rows")
    expected = (df["score1_game"] > df["score2_game"]).astype(int)
    mismatch = int((expected != df["team1_map_win"].astype(int)).sum())
    if mismatch:
        problems.append(f"team1_map_win disagrees with score comparison: {mismatch} rows")
    if problems:
        raise ValueError("map target integrity failed -> " + "; ".join(problems))
    return {"rows_checked": int(len(df)), "ties": 0, "null_scores": 0, "target_mismatches": 0}


def load_map_stream(evaluation_groups=("development",), max_exclusive_datetime=None):
    """Build the canonical map event stream.

    SERIES_DATETIME vs MAP_DATETIME
    -------------------------------
    `series_datetime` (the prediction cutoff) is derived as the MINIMUM raw
    datetime across a match's map rows, so it is well defined even if a future
    provider timestamps each map separately. `map_datetime` keeps the row's own
    raw timestamp for provenance. In the current export the two coincide for
    every row - that is a PROPERTY OF THIS EXPORT, not something the pipeline
    relies on, and `map_datetime_differs_from_series` records where it fails.

    Returns (stream_df, info_dict).
    """
    mb = pd.read_parquet(INTERIM / "map_base.parquet", engine="fastparquet",
                         columns=MAP_BASE_COLUMNS)
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    merged = mb.merge(em, on="match_id", how="left")

    n_all = len(merged)
    unlabeled = int(merged["evaluation_group"].isna().sum())
    sel = merged[merged["evaluation_group"].isin(list(evaluation_groups))].copy()

    integrity = assert_map_target_integrity(sel)

    sel["series_datetime"] = sel.groupby("match_id")["datetime"].transform("min")
    sel["map_datetime"] = sel["datetime"]
    n_differing = int((sel["map_datetime"] != sel["series_datetime"]).sum())

    if max_exclusive_datetime is not None:
        cutoff = pd.Timestamp(max_exclusive_datetime)
        before = len(sel)
        sel = sel[sel["series_datetime"] < cutoff].copy()
        cut_removed = before - len(sel)
    else:
        cut_removed = 0

    canon_elig = canonical_eligibility_map()
    t1 = sel["team1_canonical"].map(canon_elig)
    t2 = sel["team2_canonical"].map(canon_elig)
    if t1.isna().any() or t2.isna().any():
        missing = set(sel.loc[t1.isna(), "team1_canonical"]) | set(sel.loc[t2.isna(), "team2_canonical"])
        raise ValueError(f"canonical names missing from team_identity_policy.csv: {sorted(missing)}")
    sel["team1_eligible"] = t1.astype(bool)
    sel["team2_eligible"] = t2.astype(bool)

    sel["source"] = SOURCE
    sel["canonical_map_uid"] = SOURCE + ":" + sel["match_id"].astype(str) + ":" + sel["game_id"].astype(str)

    sel = sel.sort_values(["series_datetime", "match_id", "game_id"]).reset_index(drop=True)

    info = {
        "map_base_rows": n_all,
        "rows_without_manifest_label": unlabeled,
        "evaluation_groups": list(evaluation_groups),
        "rows_selected": int(len(sel)),
        "rows_removed_by_datetime_cutoff": int(cut_removed),
        "distinct_matches": int(sel["match_id"].nunique()),
        "map_datetime_differs_from_series": n_differing,
        "both_eligible_rows": int((sel["team1_eligible"] & sel["team2_eligible"]).sum()),
        "rows_touching_ineligible_identity": int((~(sel["team1_eligible"] & sel["team2_eligible"])).sum()),
        "min_series_datetime": str(sel["series_datetime"].min()) if len(sel) else None,
        "max_series_datetime": str(sel["series_datetime"].max()) if len(sel) else None,
        "target_integrity": integrity,
    }
    return sel, info


def cologne_cutoff():
    """Cologne boundary derived PRIMARILY from evaluation_manifest.csv (the
    Phase 2 grouping), never from tournament-name matching. Returns
    (first_cologne_datetime, set_of_cologne_match_ids)."""
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"] == "cologne_2026", "match_id"])
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet",
                         columns=["match_id", "datetime", "tournament"])
    col = sb[sb["match_id"].isin(cologne_ids)]
    if col.empty:
        raise ValueError("no Cologne matches found in series_base for the manifest's cologne_2026 ids")
    return pd.Timestamp(col["datetime"].min()), cologne_ids
