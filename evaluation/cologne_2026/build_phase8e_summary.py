"""
Writes the Phase 8E presentation-ready outputs that other artifacts depend
on: the 32-team milestone comparison CSV and the compact simulation-vs-
reality summary JSON. No hardcoded metric values - everything is pulled
from phase8e_metrics.compute_all_metrics().
"""

import json

from _common import ROOT
import evaluation.cologne_2026.phase8e_metrics as pm

EVAL = ROOT / "data" / "evaluation"
MILESTONE_CSV = EVAL / "cologne_2026_actual_milestone_comparison_v1.csv"
SUMMARY_JSON = EVAL / "cologne_2026_simulation_vs_reality_summary_v1.json"


def build():
    m = pm.compute_all_metrics()

    m["milestone_df"].to_csv(MILESTONE_CSV, index=False)
    print(f"wrote {MILESTONE_CSV}")

    mm = m["match_metrics"]
    champ = m["champion_analysis"]
    topk = m["top_k_set_comparisons"]
    fav = m["favorite_path_vs_reality"]

    summary = {
        "event_id": "iem_cologne_major_2026",
        "actual_champion": champ["actual_champion"],
        "actual_champion_pre_event_probability": champ["championship_probability"],
        "actual_champion_pre_event_rank": champ["championship_rank"],
        "predicted_favorite": fav["structural_set_comparisons"]["champion"]["predicted"][0],
        "predicted_favorite_championship_probability": None,  # filled below
        "favorite_won_tournament": fav["structural_set_comparisons"]["champion"]["overlap_count"] == 1,
        "actual_playoff_teams": m["playoff_teams"],
        "predicted_top8_playoff_teams": topk["playoffs_top8"]["predicted"],
        "playoff_overlap_count": topk["playoffs_top8"]["overlap_count"],
        "playoff_jaccard": topk["playoffs_top8"]["jaccard"],
        "actual_semifinalists": m["semifinalists"],
        "predicted_top4_semifinalists": topk["semifinal_top4"]["predicted"],
        "semifinal_overlap_count": topk["semifinal_top4"]["overlap_count"],
        "actual_finalists": m["finalists"],
        "predicted_top2_finalists": topk["final_top2"]["predicted"],
        "final_overlap_count": topk["final_top2"]["overlap_count"],
        "match_n": mm["overall"]["n"],
        "match_accuracy": mm["overall"]["accuracy"],
        "match_auc": mm["overall"]["roc_auc"],
        "match_log_loss": mm["overall"]["log_loss"],
        "match_brier": mm["overall"]["brier"],
        "baseline_log_loss": mm["baseline_p0_5"]["log_loss"],
        "baseline_brier": mm["baseline_p0_5"]["brier"],
        "favorite_path_champion": fav["structural_set_comparisons"]["champion"]["predicted"][0],
        "favorite_path_champion_correct": fav["structural_set_comparisons"]["champion"]["overlap_count"] == 1,
        "showmatch_excluded": True,
        "original_cologne_tagged_rows": 107,
        "official_major_rows": 106,
    }

    import pandas as pd
    team_probs = pd.read_csv(EVAL / "cologne_2026_pre_event_team_probabilities_v1.csv")
    fav_row = team_probs[(team_probs.canonical_model_name == summary["predicted_favorite"])
                          & (team_probs.metric == "win_tournament")]
    summary["predicted_favorite_championship_probability"] = float(fav_row["probability"].iloc[0]) if len(fav_row) else None

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {SUMMARY_JSON}")
    return summary, m


if __name__ == "__main__":
    build()
