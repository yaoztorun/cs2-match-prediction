"""
Dataset-specific glue between the Kaggle export and the Phase 5B.2 team-form
engine (feature_engineering/form/team_form_engine.py). The engine itself knows nothing about
"Kaggle", "development" or "Cologne" - every such concept lives here,
mirroring feature_engineering/maps/map_stream_common.py's role for the Phase 5A map engine.

Reuses (imports, never modifies) feature_engineering/series/build_series_features_v1.py's own
`build_stream_rows()` (which itself calls `canonical_eligibility_map()`) so
the series-level stream fed to this phase's engine is byte-identical to the
one Phase 3's own engine run consumed - same eligibility tagging, same
score1/score2/team1_win columns, same REQUIRED_STREAM_COLUMNS shape.
"""

import pandas as pd

from _common import INTERIM
from feature_engineering.series.build_series_features_v1 import build_stream_rows


def load_series_form_stream(evaluation_groups=("development",), max_exclusive_datetime=None):
    """Returns (stream_df, info_dict). `stream_df` satisfies
    feature_engine.REQUIRED_STREAM_COLUMNS exactly (the same shape
    build_series_features_v1.py itself builds and feeds to Phase 3's engine).
    `max_exclusive_datetime`, if given, keeps only rows with
    `datetime < cutoff` (used by the pre-Cologne snapshot builder)."""
    sb = pd.read_parquet(INTERIM / "series_base.parquet", engine="fastparquet")
    em = pd.read_csv(INTERIM / "evaluation_manifest.csv")
    merged = sb.merge(em, on="match_id", how="inner")
    dev = merged[merged["evaluation_group"].isin(evaluation_groups)].copy()

    rows = build_stream_rows(dev)

    rows_removed_by_cutoff = 0
    if max_exclusive_datetime is not None:
        cutoff = pd.Timestamp(max_exclusive_datetime)
        before = len(rows)
        rows = rows[rows["datetime"] < cutoff].copy()
        rows_removed_by_cutoff = before - len(rows)

    info = {
        "evaluation_groups": list(evaluation_groups),
        "development_pool_rows": int(len(dev)),
        "rows_removed_by_datetime_cutoff": int(rows_removed_by_cutoff),
        "rows_selected": int(len(rows)),
        "min_datetime": str(rows["datetime"].min()) if len(rows) else None,
        "max_datetime": str(rows["datetime"].max()) if len(rows) else None,
    }
    return rows, info
