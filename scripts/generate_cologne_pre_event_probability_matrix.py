"""
Phase 8D probability matrix: every ordered pair of the 32 real Cologne
teams x {Bo1, Bo3, Bo5}, scored ONCE via the frozen RF V2 adapter, before
any Monte Carlo simulation touches them. 32 x 31 x 3 = 2,976 rows.

Validation never clips, smooths, or otherwise alters a legitimate RF
probability (amendment #3) - it only asserts finiteness and the
[0,1]/complementarity contract, and separately COUNTS any exact-endpoint
values rather than treating them as errors.
"""

import numpy as np
import pandas as pd

import phase8d_common as pc
import pre_veto_series_predictor as pvp

BEST_OF_VALUES = (1, 3, 5)
EXPECTED_ROW_COUNT = 32 * 31 * 3


def build_probability_matrix(context, teams=None, prediction_datetime=None):
    teams = teams or pc.load_cologne_teams()
    if prediction_datetime is None:
        from build_phase8d_protocol import PREDICTION_CUTOFF
        prediction_datetime = PREDICTION_CUTOFF

    rows = []
    for team_a in teams:
        for team_b in teams:
            if team_a["canonical_model_name"] == team_b["canonical_model_name"]:
                continue
            for best_of in BEST_OF_VALUES:
                pred = pvp.predict_series_unknown_maps(
                    context, team_a["canonical_model_name"], team_b["canonical_model_name"],
                    best_of, prediction_datetime)
                rows.append({
                    "team_a": pred["team_a"], "team_b": pred["team_b"],
                    "team_a_display": team_a["display_name"], "team_b_display": team_b["display_name"],
                    "best_of": pred["best_of"], "prediction_datetime": pred["prediction_datetime"],
                    "tier": pred["tier"], "probability_team_a": pred["probability_team_a"],
                    "probability_team_b": pred["probability_team_b"], "model_id": pred["model_id"],
                })
    return pd.DataFrame(rows)


def validate_probability_matrix(df):
    """Raises on any real defect. Never mutates df. Returns a report dict
    including exact-endpoint counts (0 and 1 are legitimate RF outputs, not
    errors - amendment #3)."""
    report = {}
    report["row_count"] = len(df)
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(f"expected {EXPECTED_ROW_COUNT} rows, got {len(df)}")

    key = df[["team_a", "team_b", "best_of"]]
    report["unique_keys"] = int(len(key.drop_duplicates()))
    if report["unique_keys"] != EXPECTED_ROW_COUNT:
        raise ValueError(f"(team_a,team_b,best_of) is not a unique key: {report['unique_keys']} unique of "
                          f"{EXPECTED_ROW_COUNT}")

    p_a = df["probability_team_a"].to_numpy(dtype=float)
    p_b = df["probability_team_b"].to_numpy(dtype=float)
    if not np.isfinite(p_a).all():
        raise ValueError("non-finite probability_team_a present")
    if not ((p_a >= 0.0) & (p_a <= 1.0)).all():
        raise ValueError("probability_team_a outside [0,1]")
    report["exact_zero_count"] = int((p_a == 0.0).sum())
    report["exact_one_count"] = int((p_a == 1.0).sum())

    complement_diff = np.abs(p_b - (1.0 - p_a))
    report["max_complement_violation"] = float(complement_diff.max())
    if not (complement_diff < 1e-9).all():
        raise ValueError(f"probability_team_b != 1 - probability_team_a for some rows "
                          f"(max diff {report['max_complement_violation']})")

    forbidden_cols = {"winner", "result", "actual_matchup", "champion", "score"}
    present_forbidden = forbidden_cols & set(df.columns)
    if present_forbidden:
        raise ValueError(f"forbidden result-shaped column(s) present: {present_forbidden}")

    report["all_checks_pass"] = True
    return report


def compute_orientation_diagnostic(df):
    """Diagnostic only (amendment #10 of Phase 8D brief / brief section 10) -
    never used to alter, average, or correct the matrix."""
    indexed = df.set_index(["team_a", "team_b", "best_of"])["probability_team_a"]
    errors = []
    seen = set()
    for (team_a, team_b, best_of), p_ab in indexed.items():
        pair_key = (frozenset((team_a, team_b)), best_of)
        if pair_key in seen:
            continue
        seen.add(pair_key)
        p_ba = indexed.get((team_b, team_a, best_of))
        if p_ba is None:
            continue
        errors.append(abs(p_ab - (1.0 - p_ba)))
    errors = np.array(errors, dtype=float)
    return {
        "n_unordered_pairs_by_format": int(len(errors)),
        "mean": float(errors.mean()), "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)), "max": float(errors.max()),
    }
