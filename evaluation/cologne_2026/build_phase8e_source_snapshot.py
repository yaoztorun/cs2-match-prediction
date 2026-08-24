"""
Builds the per-match provenance snapshot (amendment #1): one record per
official match with enough fields to independently reconstruct the
canonical actual-results table, plus a pointer to which external source
page corroborated that match's stage/round placement (see
data/tournaments/iem_cologne_major_2026_actual_results_sources.json for the
page-level source records themselves).
"""

import json

import pandas as pd

from _common import ROOT

EVAL = ROOT / "data" / "evaluation"
OUT_PATH = ROOT / "data" / "tournaments" / "iem_cologne_major_2026_actual_result_source_snapshot_v1.json"

STAGE_SOURCE_URL = {
    "stage_1": "https://liquipedia.net/counterstrike/index.php?title=Intel_Extreme_Masters/2026/Cologne/Stage_1&action=raw",
    "stage_2": "https://liquipedia.net/counterstrike/index.php?title=Intel_Extreme_Masters/2026/Cologne/Stage_2&action=raw",
    "stage_3": "https://liquipedia.net/counterstrike/index.php?title=Intel_Extreme_Masters/2026/Cologne/Stage_3&action=raw",
    "playoffs": "https://liquipedia.net/counterstrike/index.php?title=Intel_Extreme_Masters/2026/Cologne/Playoffs&action=raw",
}


def build():
    canonical = pd.read_parquet(EVAL / "cologne_2026_actual_series_results_v1.parquet", engine="fastparquet")
    matches = []
    for row in canonical.itertuples(index=False):
        matches.append({
            "source_match_id": int(row.source_match_id),
            "stage": row.stage, "round_number": int(row.round_number), "record_group": row.record_group,
            "team_1": row.team_1, "team_2": row.team_2, "best_of": int(row.best_of),
            "score_team_1": int(row.score_team_1), "score_team_2": int(row.score_team_2),
            "winner": row.winner, "datetime_source": row.datetime_source,
            "primary_score_source": "dataset(kaggle_ektarr) series_base.parquet score1_match/score2_match",
            "structural_corroboration_source": STAGE_SOURCE_URL[row.stage],
        })
    snapshot = {
        "event_id": "iem_cologne_major_2026", "phase": "phase8e_actual_result_provenance",
        "n_matches": len(matches),
        "note": "Provenance snapshot, not model input. Score/winner authority is always the dataset's own "
                "score1_match/score2_match fields (see reconciliation_policy.winner_derivation in the "
                "Phase 8E protocol); structural_corroboration_source is the independently-fetched raw "
                "wikitext page whose per-round pairings matched this match's stage/round placement.",
        "matches": matches,
    }
    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(matches)} matches)")
    return snapshot


if __name__ == "__main__":
    build()
