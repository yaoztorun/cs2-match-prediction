"""
Phase 6A - build data/modeling/map_cv_folds_v1.csv.

TRAIN-only chronological expanding-window CV folds for the future known-map
model. Derived entirely from the EXISTING series-level fold manifest
(data/modeling/random_forest_cv_folds_v2.csv, reused byte-identically, never
regenerated) so maps from the same series stay atomic and exact-timestamp
groups stay together automatically - every map of a series inherits that
series' own (fold, role) assignment(s). Series with no map rows of their own
simply contribute nothing (inner join). Map targets are never read anywhere
in this script - folds are purely a join on (match_id, fold, role).

Read-only against data/raw/ and data/interim/.
"""

import pandas as pd

from _common import ROOT, raw_file_hashes

FEATURES_DIR = ROOT / "data" / "features"
MODELING_DIR = ROOT / "data" / "modeling"
CV_FOLDS_PATH = MODELING_DIR / "random_forest_cv_folds_v2.csv"


def main():
    hashes_before = raw_file_hashes()

    mv1 = pd.read_parquet(FEATURES_DIR / "map_features_v1.parquet", engine="fastparquet",
                          columns=["match_id", "game_id", "series_datetime"])
    series_cv = pd.read_csv(CV_FOLDS_PATH, parse_dates=["datetime"])

    out = mv1.merge(series_cv[["match_id", "fold", "role"]], on="match_id", how="inner")
    out = out.rename(columns={"series_datetime": "datetime"})[["match_id", "game_id", "datetime", "fold", "role"]]
    out = out.sort_values(["fold", "role", "datetime", "match_id", "game_id"]).reset_index(drop=True)

    # ---- guarantees, re-derived rather than assumed ----
    n_folds = int(series_cv["fold"].max())
    for fold in range(1, n_folds + 1):
        train_dt = out.loc[(out.fold == fold) & (out.role == "train"), "datetime"]
        val_dt = out.loc[(out.fold == fold) & (out.role == "validation"), "datetime"]
        if len(train_dt) and len(val_dt):
            assert train_dt.max() < val_dt.min(), f"fold {fold}: map-level chronology violated"

    matches_covered = set(out["match_id"])
    matches_missing_maps = set(series_cv["match_id"]) - matches_covered
    assert matches_covered <= set(series_cv["match_id"]), "a map's match_id is outside the series CV manifest"

    MODELING_DIR.mkdir(exist_ok=True, parents=True)
    out.to_csv(MODELING_DIR / "map_cv_folds_v1.csv", index=False, encoding="utf-8")

    assert hashes_before == raw_file_hashes(), "data/raw/ was modified during the build"
    print(f"Wrote {MODELING_DIR / 'map_cv_folds_v1.csv'} ({len(out)} map-fold rows, "
          f"{out['match_id'].nunique()} distinct matches, {n_folds} folds)")
    print(f"  series in the CV manifest with no map rows of their own (contribute nothing): "
          f"{len(matches_missing_maps)}")
    for fold in range(1, n_folds + 1):
        sub = out[out.fold == fold]
        tr = sub[sub.role == "train"]
        va = sub[sub.role == "validation"]
        print(f"  fold {fold}: train rows={len(tr)} [{tr['datetime'].min()} .. {tr['datetime'].max()}], "
              f"validation rows={len(va)} [{va['datetime'].min() if len(va) else 'n/a'} .. "
              f"{va['datetime'].max() if len(va) else 'n/a'}]")


if __name__ == "__main__":
    main()
