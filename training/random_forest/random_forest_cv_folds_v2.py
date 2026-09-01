"""
Phase 4B.1: expanding-window chronological CV folds for Random Forest V2
hyperparameter selection. Built ONLY from the 6,619-match TRAIN partition of
data/modeling/series_split_v1.csv - the 1,419-match main VALIDATION
partition and the 1,418-match TEST partition are never read here.

The 6,619 training matches are split (by exact `datetime` group - never
split a group) into 5 contiguous chronological blocks of near-equal size,
giving 4 expanding-window folds:

    fold 1: train=block1            val=block2
    fold 2: train=block1+2          val=block3
    fold 3: train=block1+2+3        val=block4
    fold 4: train=block1+2+3+4      val=block5

Writes:
    data/modeling/random_forest_cv_folds_v2.csv  (match_id, datetime, fold, role)
    reports/random_forest_cv_folds_v2.md
"""

import pandas as pd

from _common import ROOT, REPORTS
from feature_engineering.series.build_series_split_v1 import pick_boundary_index

FEATURES_PATH = ROOT / "data" / "features" / "series_features_v1.parquet"
SPLIT_PATH = ROOT / "data" / "modeling" / "series_split_v1.csv"
MODELING_DIR = ROOT / "data" / "modeling"

N_BLOCKS = 5
N_FOLDS = 4


def assign_blocks(train_df, n_blocks=N_BLOCKS):
    """Pure function: given a dataframe with `datetime` (+ any other columns),
    returns a copy with a new `block` column (1..n_blocks), splitting the
    cumulative row count into n_blocks near-equal, chronologically contiguous
    groups without ever splitting a `datetime` group across a block boundary."""
    train = train_df.sort_values(["datetime"]).reset_index(drop=True)
    total = len(train)

    group_sizes = train.groupby("datetime").size().sort_index()
    datetimes = group_sizes.index.tolist()
    counts = group_sizes.values.tolist()
    cum_counts = []
    running = 0
    for c in counts:
        running += c
        cum_counts.append(running)

    boundaries = []
    for k in range(1, n_blocks):
        target = k * total / n_blocks
        boundaries.append(pick_boundary_index(cum_counts, target))
    boundaries = sorted(set(boundaries))
    assert len(boundaries) == n_blocks - 1, f"expected {n_blocks - 1} distinct block boundaries, got {boundaries}"

    block_of_datetime = {}
    block_start = 0
    for block_idx in range(n_blocks):
        end = boundaries[block_idx] if block_idx < len(boundaries) else len(datetimes) - 1
        for i in range(block_start, end + 1):
            block_of_datetime[datetimes[i]] = block_idx + 1  # 1-indexed blocks
        block_start = end + 1

    train = train.copy()
    train["block"] = train["datetime"].map(block_of_datetime)
    assert train["block"].isna().sum() == 0
    assert set(train["block"].unique()) == set(range(1, n_blocks + 1))
    return train


def build_cv_folds(train_with_block, n_folds=N_FOLDS):
    """Pure function: given a dataframe with `match_id`, `datetime`, `block`,
    returns the long-format CV assignment dataframe (match_id, datetime,
    fold, role) for expanding-window folds 1..n_folds, where fold f trains
    on blocks 1..f and validates on block f+1. Asserts fold-train ends
    strictly before fold-validation begins, for every fold."""
    fold_rows = []
    for fold in range(1, n_folds + 1):
        train_blocks = list(range(1, fold + 1))
        val_block = fold + 1
        fold_train = train_with_block[train_with_block["block"].isin(train_blocks)]
        fold_val = train_with_block[train_with_block["block"] == val_block]

        assert fold_train["datetime"].max() < fold_val["datetime"].min(), \
            f"fold {fold}: train/validation chronology violated"

        for _, r in fold_train.iterrows():
            fold_rows.append({"match_id": r["match_id"], "datetime": r["datetime"], "fold": fold, "role": "train"})
        for _, r in fold_val.iterrows():
            fold_rows.append({"match_id": r["match_id"], "datetime": r["datetime"], "fold": fold, "role": "validation"})

    return pd.DataFrame(fold_rows).sort_values(["fold", "role", "datetime", "match_id"]).reset_index(drop=True)


def main():
    features = pd.read_parquet(FEATURES_PATH, engine="fastparquet")
    split = pd.read_csv(SPLIT_PATH)
    df = features.merge(split[["match_id", "split"]], on="match_id", how="inner")
    train = df[df["split"] == "train"].copy()
    train = train.sort_values(["datetime", "match_id"]).reset_index(drop=True)
    total = len(train)
    assert total == 6619, f"expected 6619 train matches, got {total}"

    train = assign_blocks(train, N_BLOCKS)
    block_stats = train.groupby("block").agg(n=("match_id", "size"), first_dt=("datetime", "min"),
                                              last_dt=("datetime", "max"))

    cv_df = build_cv_folds(train, N_FOLDS)

    # ---------------- integrity checks ----------------
    # no datetime group crosses a block boundary (and therefore a fold boundary,
    # since folds are built from whole blocks)
    per_dt_block = train.groupby("datetime")["block"].nunique()
    assert (per_dt_block == 1).all(), "a datetime group was split across blocks"

    # every match_id used in the CV file must be a TRAIN match_id (never validation/test)
    train_ids = set(train["match_id"])
    assert set(cv_df["match_id"]) <= train_ids, "CV fold file references a non-TRAIN match_id"

    MODELING_DIR.mkdir(exist_ok=True, parents=True)
    cv_df.to_csv(MODELING_DIR / "random_forest_cv_folds_v2.csv", index=False, encoding="utf-8")

    # ---------------- report ----------------
    md = []
    md.append("# Random Forest V2 - Expanding-Window CV Folds\n")
    md.append("Generated by `training/random_forest/random_forest_cv_folds_v2.py`, built **only** from the 6,619-match TRAIN "
              "partition of `data/modeling/series_split_v1.csv` - the 1,419-match main VALIDATION partition and "
              "the 1,418-match TEST partition are never read by this script.\n")

    md.append("## Chronological blocks\n")
    md.append("| block | n | first datetime | last datetime |")
    md.append("|---|---|---|---|")
    for b, row in block_stats.iterrows():
        md.append(f"| {b} | {int(row['n'])} | {row['first_dt']} | {row['last_dt']} |")
    md.append("")

    md.append("## Expanding-window folds\n")
    md.append("| fold | fold-train blocks | fold-train n | fold-train last dt | fold-val block | fold-val n | fold-val first dt |")
    md.append("|---|---|---|---|---|---|---|")
    for fold in range(1, N_FOLDS + 1):
        train_blocks = list(range(1, fold + 1))
        val_block = fold + 1
        ft = cv_df[(cv_df["fold"] == fold) & (cv_df["role"] == "train")]
        fv = cv_df[(cv_df["fold"] == fold) & (cv_df["role"] == "validation")]
        md.append(f"| {fold} | {train_blocks} | {len(ft)} | {ft['datetime'].max()} | {val_block} | {len(fv)} | "
                  f"{fv['datetime'].min()} |")
    md.append("\nVerified for every fold: fold-train's maximum `datetime` is strictly less than fold-validation's "
              "minimum `datetime` (no leakage), and no `datetime` group is split across a block/fold boundary.\n")

    (REPORTS / "phases" / "random_forest_cv_folds_v2.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote data/modeling/random_forest_cv_folds_v2.csv ({len(cv_df)} rows)")
    print(f"Wrote reports/random_forest_cv_folds_v2.md")
    for fold in range(1, N_FOLDS + 1):
        ft = cv_df[(cv_df["fold"] == fold) & (cv_df["role"] == "train")]
        fv = cv_df[(cv_df["fold"] == fold) & (cv_df["role"] == "validation")]
        print(f"fold {fold}: train={len(ft)} val={len(fv)}")


if __name__ == "__main__":
    main()
