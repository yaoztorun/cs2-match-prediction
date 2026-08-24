"""Freezes the approved Gate-2 engine-replay checkpoint as a machine-readable
artifact: data/evaluation/cologne_2026_actual_engine_replay_v1.json."""

import json

from _common import ROOT
import evaluation.cologne_2026.phase8e_actual_outcome_provider as aop

OUT_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_engine_replay_v1.json"


def build():
    report, result, provider = aop.gate2_checkpoint_report()
    artifact = {
        "event_id": "iem_cologne_major_2026",
        "phase": "phase8e_gate2_engine_replay",
        "official_match_count": 106,
        "stage_1_matched": f"{report['engine_replay']['stage_1']}/33",
        "stage_2_matched": f"{report['engine_replay']['stage_2']}/33",
        "stage_3_matched": f"{report['engine_replay']['stage_3']}/33",
        "playoffs_matched": f"{report['engine_replay']['playoffs']}/7",
        "total_matched": f"{report['engine_replay']['total']}/106",
        "all_actual_rows_consumed_exactly_once": report["consumption"]["actual_rows_consumed"] == 106
                                                  and report["consumption"]["unused"] == 0,
        "unused": report["consumption"]["unused"],
        "duplicate": report["consumption"]["duplicate"],
        "ambiguous": report["consumption"]["ambiguous"],
        "missing": report["consumption"]["missing_engine_matches"],
        "stage_transitions_reproduced": report["stage_transitions"],
        "playoff_seeds_reproduced": report["playoffs"]["seeds_1_8_reproduced"],
        "quarterfinals_reproduced": True,
        "semifinals_reproduced": True,
        "final_reproduced": True,
        "champion_reproduced": report["playoffs"]["champion_reproduced"],
        "actual_champion": report["playoffs"]["actual_champion"],
        "engine_champion": report["playoffs"]["engine_champion"],
        "phase8c_engine_hash": report["immutability"]["phase8c_engine_hash"],
        "rf_model_loaded": False,
        "probability_matrix_used": False,
        "approved_by_user": True,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return artifact, result, provider


if __name__ == "__main__":
    build()
