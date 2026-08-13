"""
Phase 6A - build data/modeling/map_split_v1.csv.

Map-level modeling split for the known-map task. CRITICAL: maps from the
SAME match_id must never appear across different partitions - the map split
is entirely DERIVED from the existing series split
(data/modeling/series_split_v1.csv), never independently assigned. If a
series is TRAIN, every one of its maps is TRAIN; if VALIDATION, every map is
VALIDATION; if TEST, every map is TEST. Cologne match_ids never appear in
series_split_v1.csv (development-only universe) and therefore never appear
here either.

Read-only against data/raw/ and data/interim/.
"""

import pandas as pd

from _common import ROOT, INTERIM, raw_file_hashes

FEATURES_DIR = ROOT / "data" / "features"
MODELING_DIR = ROOT / "data" / "modeling"
SPLIT_PATH = MODELING_DIR / "series_split_v1.csv"


def main():
    hashes_before = raw_file_hashes()

    mv1 = pd.read_parquet(FEATURES_DIR / "map_features_v1.parquet", engine="fastparquet",
                          columns=["match_id", "game_id", "series_datetime"])
    split = pd.read_csv(SPLIT_PATH)

    merged = mv1.merge(split[["match_id", "split"]], on="match_id", how="left", validate="many_to_one")
    assert len(merged) == len(mv1), "join changed row count"
    assert merged["split"].notna().all(), \
        f"{int(merged['split'].isna().sum())} map rows have no series split assignment"

    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    cologne_ids = set(em.loc[em["evaluation_group"].isin(["cologne_2026", "post_cologne"]), "match_id"])
    assert not (set(merged["match_id"]) & cologne_ids), "a Cologne/post-Cologne match_id reached the map split"

    out = merged.rename(columns={"series_datetime": "datetime"})[["match_id", "game_id", "datetime", "split"]]
    out = out.sort_values(["datetime", "match_id", "game_id"]).reset_index(drop=True)

    # CRITICAL invariant: zero match_id crosses partitions
    crossing = out.groupby("match_id")["split"].nunique()
    bad = crossing[crossing > 1]
    assert len(bad) == 0, f"{len(bad)} match_id(s) span more than one partition: {bad.index.tolist()[:10]}"

    MODELING_DIR.mkdir(exist_ok=True, parents=True)
    out.to_csv(MODELING_DIR / "map_split_v1.csv", index=False, encoding="utf-8")

    counts = out["split"].value_counts()
    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"Wrote {MODELING_DIR / 'map_split_v1.csv'} ({len(out)} map rows)")
    print(f"  train={counts.get('train', 0)} validation={counts.get('validation', 0)} test={counts.get('test', 0)}")
    print(f"  zero match_id crosses a partition (verified over {out['match_id'].nunique()} distinct matches)")


if __name__ == "__main__":
    main()
