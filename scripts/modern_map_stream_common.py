"""
Dataset-specific glue between the Kaggle export and the Phase 6C modern-map
engine (scripts/modern_map_feature_engine.py). Mirrors map_stream_common.py's
role for Phase 5A. Adds NO new cleaning logic of its own - only joins two
STATIC, already-frozen columns onto the two existing stream loaders' output.

WHY team1_pre_series_elo / team2_pre_series_elo COME FROM A FROZEN AUDIT FILE
-------------------------------------------------------------------------
data/features/series_team_form_states_v1.parquet is Phase 5B.2's own
audit-only artifact: one row per (development) series, carrying
team1_elo_before/team2_elo_before EXACTLY as of that series' start - the same
ELO trajectory Phase 3's own state produces (exhaustively parity-checked when
Phase 5B.2 was built). Every map of a series shares one series_datetime, so
this one static per-match_id value is correct for every map of that series -
no new ELO replay is needed, and scripts/build_map_features_v3_modern_map.py
independently re-verifies this parity (exhaustively, all rows THAT ENTER
map_features_v2_rich.parquet) against that file's own elo_diff before
building anything.

SOME map_stream_common ROWS HAVE NO series_team_form_states_v1 MATCH
----------------------------------------------------------------------
69 (of 5,021) map_stream_common match_ids have no row in
series_team_form_states_v1.parquet (13 are entirely absent from
series_base.parquet; the other 56 were excluded from Phase 3's own
series-level stream by validity checks map_stream_common's map-level
admission doesn't apply). Empirically confirmed: NONE of the 30 both-eligible
map rows among them ever reach map_features_v2_rich.parquet (Phase 6A's own
join already excludes every one) - they are pure historical noise this
module must not let corrupt anything. `team1/2_pre_series_elo` is filled with
`ELO_INITIAL` for these rows purely so the arithmetic in
modern_map_feature_engine.apply_selected_map_team_result does not crash; the
companion boolean column `pre_series_elo_known=False` is ALSO emitted, and
that engine ANDs it into `opponent_identity_trusted` so the fabricated value
can never enter a trusted-population aggregate (map_state's own ELO-driven
replay is completely unaffected - `apply_map_result` never reads these
columns at all).
"""

import pandas as pd

from _common import INTERIM, ROOT
from feature_engine import ELO_INITIAL
import map_stream_common
import player_roster_stream_common

FEATURES_DIR = ROOT / "data" / "features"


def load_modern_map_streams(evaluation_groups=("development",), max_exclusive_datetime=None):
    """Returns (map_rows, player_rows, info).

    map_rows: map_stream_common.load_map_stream's own output, left-joined
    with team1_elo_before/team2_elo_before (renamed team1/2_pre_series_elo)
    from series_team_form_states_v1.parquet on match_id, plus a
    `pre_series_elo_known` boolean (see module docstring for the small
    fraction of rows where this join legitimately misses).

    player_rows: player_roster_stream_common.load_player_roster_stream's own
    output, left-joined with map_name from map_base.parquet on
    (match_id, game_id). The join must be total.
    """
    map_rows, map_info = map_stream_common.load_map_stream(evaluation_groups, max_exclusive_datetime)

    elo_before = pd.read_parquet(
        FEATURES_DIR / "series_team_form_states_v1.parquet", engine="fastparquet",
        columns=["match_id", "team1_elo_before", "team2_elo_before"],
    ).rename(columns={"team1_elo_before": "team1_pre_series_elo", "team2_elo_before": "team2_pre_series_elo"})

    n_before = len(map_rows)
    map_rows = map_rows.merge(elo_before, on="match_id", how="left", validate="many_to_one")
    n_missing_elo = int(map_rows["team1_pre_series_elo"].isna().sum())
    map_rows["pre_series_elo_known"] = map_rows["team1_pre_series_elo"].notna()
    map_rows["team1_pre_series_elo"] = map_rows["team1_pre_series_elo"].fillna(ELO_INITIAL)
    map_rows["team2_pre_series_elo"] = map_rows["team2_pre_series_elo"].fillna(ELO_INITIAL)
    assert len(map_rows) == n_before, "map row count changed during the pre-series-ELO join"

    player_rows, player_info = player_roster_stream_common.load_player_roster_stream(
        evaluation_groups, max_exclusive_datetime)

    map_names = pd.read_parquet(INTERIM / "map_base.parquet", engine="fastparquet",
                                 columns=["match_id", "game_id", "map_name"])
    n_before_p = len(player_rows)
    player_rows = player_rows.merge(map_names, on=["match_id", "game_id"], how="left", validate="many_to_one")
    n_missing_map = int(player_rows["map_name"].isna().sum())
    if n_missing_map:
        raise ValueError(f"{n_missing_map} player rows have no map_base map_name match")
    assert len(player_rows) == n_before_p, "player row count changed during the map_name join"

    info = {
        "map_rows": int(len(map_rows)), "player_rows": int(len(player_rows)),
        "map_stream_info": map_info, "player_stream_info": player_info,
        "map_rows_missing_pre_series_elo": n_missing_elo, "map_name_join_total": True,
    }
    return map_rows, player_rows, info
