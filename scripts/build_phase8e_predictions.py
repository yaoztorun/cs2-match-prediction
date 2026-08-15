"""
Phase 8E actual-match predictions - PURE frozen-matrix lookup.

Deliberately does NOT import joblib, the RF V2 model, feature_engine, RF
preprocessing, or pre-Cologne state (amendment #7/#18) - statically checked
by tests/test_phase8e_cologne_simulation_vs_reality.py via an AST import
guard. Only the validated Gate-2 actual replay trace and the frozen Phase 8D
2,976-row probability matrix are used. No RF re-inference of any kind.
"""

import pandas as pd

from _common import ROOT
import phase8e_actual_outcome_provider as aop

MATRIX_PATH = ROOT / "data" / "evaluation" / "cologne_2026_pre_event_matchup_probabilities_v1.parquet"
OUT_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_match_predictions_v1.parquet"


def build_matrix_lookup(df):
    """Identical shape to cologne_pre_event_simulation.build_matrix_lookup:
    {(team_a, team_b, best_of): probability_team_a}. Reimplemented locally
    (not imported) so this module has zero dependency on any simulation
    script that might import the model elsewhere in its own module body."""
    return {(r.team_a, r.team_b, r.best_of): r.probability_team_a for r in df.itertuples(index=False)}


def build_predictions():
    result, provider = aop.replay_actual_tournament()
    matrix_df = pd.read_parquet(MATRIX_PATH, engine="fastparquet")
    lookup = build_matrix_lookup(matrix_df)

    rows = []
    for entry in result.full_trace():
        m = entry.match
        key = (m.team_a, m.team_b, m.best_of)
        if key not in lookup:
            raise ValueError(f"GATE STOP: exact matrix lookup failed for {key} (match_id={m.match_id}); "
                              "no orientation reversal / approximate matching / symmetrization allowed.")
        p_a = lookup[key]
        winner = entry.resolution.winner
        actual_team_a_win = int(winner == m.team_a)
        predicted_winner_0_5 = m.team_a if p_a >= 0.5 else m.team_b
        rows.append({
            "actual_match_id": entry.resolution.provider_metadata["source_match_id"],
            "engine_match_id": m.match_id,
            "stage": m.stage, "round": m.round_number, "record_group": m.record_group,
            "team_a": m.team_a, "team_b": m.team_b, "seed_a": m.seed_a, "seed_b": m.seed_b,
            "best_of": m.best_of,
            "probability_team_a": p_a, "probability_team_b": 1.0 - p_a,
            "predicted_winner_0_5": predicted_winner_0_5,
            "actual_winner": winner,
            "prediction_correct": int(predicted_winner_0_5 == winner),
            "actual_team_a_win": actual_team_a_win,
        })

    df = pd.DataFrame(rows)
    if len(df) != 106:
        raise ValueError(f"expected 106 actual match predictions, got {len(df)}")
    if df["engine_match_id"].duplicated().any():
        raise ValueError("duplicate engine_match_id in predictions table")
    n_exact_lookups = len(df)  # every row above only exists because its lookup succeeded
    print(f"exact matrix lookups: {n_exact_lookups}/106")

    df.to_parquet(OUT_PATH, engine="fastparquet", index=False)
    print(f"wrote {OUT_PATH}")
    return df


if __name__ == "__main__":
    build_predictions()
