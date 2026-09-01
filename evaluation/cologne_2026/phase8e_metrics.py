"""
Phase 8E metrics: match-level external performance, tournament-milestone
comparison, stage/BO breakdowns, Swiss-record and playoff-seed probability
lookups, top-K set comparisons, and the favorite-wins-path structural
comparison. Every number here is computed from already-frozen local
artifacts (Gate-2 replay + Phase 8D aggregates) - nothing here calls RF V2
or touches feature_engine/state.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                              log_loss, precision_score, recall_score, roc_auc_score)

from _common import ROOT
import evaluation.cologne_2026.phase8e_actual_outcome_provider as aop
import tournament.engine.tournament_engine as te

EVAL = ROOT / "data" / "evaluation"
PRED_PATH = EVAL / "cologne_2026_actual_match_predictions_v1.parquet"
TEAM_PROB_PATH = EVAL / "cologne_2026_pre_event_team_probabilities_v1.csv"
SWISS_RECORD_PATH = EVAL / "cologne_2026_pre_event_swiss_record_distributions_v1.csv"
PLAYOFF_SEED_PATH = EVAL / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv"
FAVORITE_PATH_PATH = EVAL / "cologne_2026_pre_event_favorite_path_v1.json"
MATRIX_PATH = EVAL / "cologne_2026_pre_event_matchup_probabilities_v1.parquet"
TOURNAMENT_YAML = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"

DEV_VALIDATION = {"log_loss": 0.6514, "brier": 0.2298, "roc_auc": 0.6566, "accuracy": 0.6068}


# ---------------------------------------------------------------------------
# 1. Binary match-level metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"n": 0, "note": "empty group"}
    pred = (p >= 0.5).astype(int)
    out = {
        "n": n,
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(set(y_true.tolist())) < 2:
        out["roc_auc"] = None
        out["roc_auc_note"] = "N/A - only one class present in this group"
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
    if n <= 1:
        out["warning"] = "INSUFFICIENT SAMPLE FOR GENERAL INFERENCE"
    return out


def compute_match_metrics(pred_df):
    y_true, p = pred_df["actual_team_a_win"].values, pred_df["probability_team_a"].values
    overall = binary_metrics(y_true, p)
    baseline = binary_metrics(y_true, np.full(len(y_true), 0.5))
    baseline["roc_auc"] = 0.5
    baseline.pop("roc_auc_note", None)
    baseline["accuracy_note"] = ("with p>=0.5 -> team_a for every match, this 'accuracy' is simply the "
                                  "observed team_a-win prevalence, not a meaningful tuned classifier.")
    by_stage = {stage: binary_metrics(g["actual_team_a_win"].values, g["probability_team_a"].values)
                for stage, g in pred_df.groupby("stage")}
    by_bo = {f"BO{int(bo)}": binary_metrics(g["actual_team_a_win"].values, g["probability_team_a"].values)
             for bo, g in pred_df.groupby("best_of")}
    dev_diff = {k: {"cologne": overall[k], "dev_validation": DEV_VALIDATION[k],
                     "external_event_metric_difference": overall[k] - DEV_VALIDATION[k]}
                for k in ("log_loss", "brier", "roc_auc", "accuracy") if overall.get(k) is not None}
    return {"overall": overall, "baseline_p0_5": baseline, "by_stage": by_stage, "by_best_of": by_bo,
            "vs_development_validation": dev_diff,
            "language_note": "Differences above are reported as 'external-event metric differences' only - "
                              "not improvement/degradation claims and not new model-selection evidence."}


# ---------------------------------------------------------------------------
# 2. Actual-winner probability distribution + realized-path log score
# ---------------------------------------------------------------------------

def actual_winner_probability_analysis(pred_df):
    df = pred_df.copy()
    df["p_actual_winner"] = np.where(df["actual_team_a_win"] == 1, df["probability_team_a"], df["probability_team_b"])
    stats = {
        "mean": float(df["p_actual_winner"].mean()), "median": float(df["p_actual_winner"].median()),
        "q1": float(df["p_actual_winner"].quantile(0.25)), "q3": float(df["p_actual_winner"].quantile(0.75)),
        "min": float(df["p_actual_winner"].min()), "max": float(df["p_actual_winner"].max()),
    }
    correct, incorrect = df[df.prediction_correct == 1], df[df.prediction_correct == 0]

    def row_summary(row):
        return {"actual_match_id": int(row["actual_match_id"]), "stage": row["stage"], "round": int(row["round"]),
                "team_a": row["team_a"], "team_b": row["team_b"], "actual_winner": row["actual_winner"],
                "p_actual_winner": float(row["p_actual_winner"]), "best_of": int(row["best_of"])}

    highest_conf_correct = row_summary(correct.loc[correct["p_actual_winner"].idxmax()]) if len(correct) else None
    highest_conf_incorrect = row_summary(incorrect.loc[incorrect["p_actual_winner"].idxmin()]) if len(incorrect) else None
    biggest_upset = row_summary(df.loc[df["p_actual_winner"].idxmin()])
    return {"distribution": stats, "highest_confidence_correct_prediction": highest_conf_correct,
            "highest_confidence_incorrect_prediction": highest_conf_incorrect, "biggest_model_upset": biggest_upset,
            "note": "biggest_model_upset is defined as the actual winner with the lowest frozen "
                    "p_actual_winner; it will typically coincide with the highest-confidence-incorrect "
                    "row since both select the minimum p_actual_winner."}


def realized_path_log_score(pred_df):
    y_true, p_a = pred_df["actual_team_a_win"].values, pred_df["probability_team_a"].values
    p_actual = np.where(y_true == 1, p_a, 1.0 - p_a)
    log_p = np.log(p_actual)
    total = float(np.sum(log_p))
    mean_neg = float(-np.mean(log_p))
    ll = float(log_loss(y_true, p_a, labels=[0, 1]))
    if abs(mean_neg - ll) > 1e-9:
        raise ValueError(f"realized-path mean negative log prob {mean_neg} != match log loss {ll}")
    return {
        "sum_log_probability_actual_winners": total, "mean_negative_log_probability": mean_neg,
        "equals_match_log_loss": True, "match_log_loss_cross_check": ll,
        "naming": "conditional realized-path log probability / realized tournament path log score - "
                  "conditional on the deterministic pairing rules induced by preceding outcomes, NOT an "
                  "unconditional probability of the entire Major.",
        "raw_path_probability_note": "exp(sum_log_probability) is astronomically small and is not "
                                      "foregrounded; the sum stays in log-space.",
    }


# ---------------------------------------------------------------------------
# 3. Champion analysis
# ---------------------------------------------------------------------------

def champion_analysis(actual_champion, team_probs):
    champ = team_probs[team_probs.metric == "win_tournament"].copy()
    champ = champ.sort_values(["probability", "canonical_model_name"], ascending=[False, True]).reset_index(drop=True)
    champ["rank"] = champ.index + 1
    row = champ[champ.canonical_model_name == actual_champion].iloc[0]
    rank = int(row["rank"])
    cum_above = float(champ.iloc[: rank - 1]["probability"].sum())
    p_champ = float(row["probability"])

    def metric_p(metric):
        r = team_probs[(team_probs.canonical_model_name == actual_champion) & (team_probs.metric == metric)]
        return float(r["probability"].iloc[0]) if len(r) else None

    return {
        "actual_champion": actual_champion, "championship_probability": p_champ, "championship_rank": rank,
        "cumulative_championship_probability_of_teams_ranked_above": cum_above,
        "top1": rank <= 1, "top3": rank <= 3, "top5": rank <= 5, "top8": rank <= 8,
        "reach_playoffs_probability": metric_p("reach_playoffs"),
        "reach_semifinal_probability": metric_p("reach_semifinal"),
        "reach_final_probability": metric_p("reach_final"),
        "multiclass_champion_log_score": -float(np.log(p_champ)) if p_champ > 0 else float("inf"),
        "note": "The 32-team champion distribution is a single multiclass outcome, not 32 independent "
                "binary observations.",
    }


# ---------------------------------------------------------------------------
# 4. Milestone comparison table + probability quality
# ---------------------------------------------------------------------------

def milestone_labels_from_replay(result):
    qf = [e for e in result.playoffs.trace if e.match.round_number == 1]
    sf = [e for e in result.playoffs.trace if e.match.round_number == 2]
    fn = [e for e in result.playoffs.trace if e.match.round_number == 3]
    playoff_teams = {e.match.team_a for e in qf} | {e.match.team_b for e in qf}
    semifinalists = {e.match.team_a for e in sf} | {e.match.team_b for e in sf}
    finalists = {e.match.team_a for e in fn} | {e.match.team_b for e in fn}
    return playoff_teams, semifinalists, finalists, result.champion


def build_milestone_comparison(team_probs, roster, playoff_teams, semifinalists, finalists, champion):
    def get_p(team, metric):
        r = team_probs[(team_probs.canonical_model_name == team) & (team_probs.metric == metric)]
        return float(r["probability"].iloc[0]) if len(r) else None

    rows = []
    for team in sorted(roster):
        rows.append({
            "team": team,
            "actual_reached_playoffs": team in playoff_teams, "predicted_p_playoffs": get_p(team, "reach_playoffs"),
            "actual_reached_semifinal": team in semifinalists, "predicted_p_semifinal": get_p(team, "reach_semifinal"),
            "actual_reached_final": team in finalists, "predicted_p_final": get_p(team, "reach_final"),
            "actual_champion": team == champion, "predicted_p_champion": get_p(team, "win_tournament"),
        })
    return pd.DataFrame(rows)


def milestone_probability_quality(milestone_df):
    out = {}
    for label, actual_col, pred_col in [("reach_playoffs", "actual_reached_playoffs", "predicted_p_playoffs"),
                                         ("reach_semifinal", "actual_reached_semifinal", "predicted_p_semifinal"),
                                         ("reach_final", "actual_reached_final", "predicted_p_final")]:
        y = milestone_df[actual_col].astype(int).values
        p = milestone_df[pred_col].values
        out[label] = {"n": int(len(y)), "brier": float(brier_score_loss(y, p)),
                      "log_loss": float(log_loss(y, p, labels=[0, 1]))}
    return out


# ---------------------------------------------------------------------------
# 5. Top-K set comparisons (frozen tie-break: probability -> pre_event_seed -> name)
# ---------------------------------------------------------------------------

def top_k_by_metric(team_probs, roster, metric, k):
    rows = team_probs[team_probs.metric == metric].copy()
    rows["pre_event_seed"] = rows["canonical_model_name"].map(lambda t: roster[t]["pre_event_seed"])
    rows = rows.sort_values(by=["probability", "pre_event_seed", "canonical_model_name"],
                             ascending=[False, True, True])
    return list(rows["canonical_model_name"].iloc[:k])


def set_comparison(predicted, actual):
    predicted, actual = set(predicted), set(actual)
    overlap = predicted & actual
    union = predicted | actual
    return {
        "overlap_count": len(overlap),
        "precision": (len(overlap) / len(predicted)) if predicted else None,
        "recall": (len(overlap) / len(actual)) if actual else None,
        "jaccard": (len(overlap) / len(union)) if union else None,
        "predicted": sorted(predicted), "actual": sorted(actual),
    }


def top_k_set_comparisons(team_probs, roster, playoff_teams, semifinalists, finalists, champion):
    return {
        "playoffs_top8": set_comparison(top_k_by_metric(team_probs, roster, "reach_playoffs", 8), playoff_teams),
        "semifinal_top4": set_comparison(top_k_by_metric(team_probs, roster, "reach_semifinal", 4), semifinalists),
        "final_top2": set_comparison(top_k_by_metric(team_probs, roster, "reach_final", 2), finalists),
        "champion_top1": set_comparison(top_k_by_metric(team_probs, roster, "win_tournament", 1), {champion}),
    }


# ---------------------------------------------------------------------------
# 6. Stage advancement evaluation
# ---------------------------------------------------------------------------

def stage_advancement_evaluation(result, team_probs, roster):
    out = {}
    for stage_key, stage_result in [("stage_1", result.stage1), ("stage_2", result.stage2),
                                     ("stage_3", result.stage3)]:
        metric = f"advance_from_{stage_key}"
        advanced_ids = {s.team_id for s in stage_result.advancers}
        participant_ids = {s.team_id for s in stage_result.entrants}
        y, p, teams = [], [], []
        for tid in sorted(participant_ids):
            r = team_probs[(team_probs.canonical_model_name == tid) & (team_probs.metric == metric)]
            if not len(r) or pd.isna(r["probability"].iloc[0]):
                continue  # team never participates in this stage across the 50,000 sims - no conditional estimate exists
            y.append(int(tid in advanced_ids))
            p.append(float(r["probability"].iloc[0]))
            teams.append(tid)
        y_arr, p_arr = np.array(y), np.array(p)
        metrics = {"n": len(y), "brier": float(brier_score_loss(y_arr, p_arr)) if len(y) else None,
                   "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])) if len(y) else None}
        top8_predicted = top_k_by_metric(team_probs[team_probs.canonical_model_name.isin(participant_ids)],
                                          roster, metric, 8)
        out[stage_key] = {"metrics": metrics, "n_participants_scored": len(y),
                           "top8_vs_actual": set_comparison(top8_predicted, advanced_ids)}
    return out


# ---------------------------------------------------------------------------
# 7. Swiss terminal record evaluation
# ---------------------------------------------------------------------------

def swiss_terminal_record_evaluation(result, swiss_records):
    rows = []
    for stage_key, stage_result in [("stage_1", result.stage1), ("stage_2", result.stage2),
                                     ("stage_3", result.stage3)]:
        for s in stage_result.final_order:
            actual_record = f"{s.wins}-{s.losses}"
            team_rows = swiss_records[(swiss_records.canonical_model_name == s.team_id)
                                       & (swiss_records.stage == stage_key)]
            realized = team_rows[team_rows.record == actual_record]
            realized_p = float(realized["conditional_on_participating_probability"].iloc[0]) if len(realized) else None
            if len(team_rows):
                modal_row = team_rows.loc[team_rows["conditional_on_participating_probability"].idxmax()]
                modal_record = modal_row["record"]
            else:
                modal_record = None
            rows.append({"team": s.team_id, "stage": stage_key, "actual_terminal_record": actual_record,
                         "realized_record_probability": realized_p, "most_probable_terminal_record": modal_record,
                         "modal_matched_reality": modal_record == actual_record})
    df = pd.DataFrame(rows)
    valid = df["realized_record_probability"].dropna()
    summary = {"mean": float(valid.mean()), "median": float(valid.median()),
               "min": float(valid.min()), "max": float(valid.max()), "n": int(len(valid))}
    return df, summary


# ---------------------------------------------------------------------------
# 8. Playoff seed evaluation
# ---------------------------------------------------------------------------

def playoff_seed_evaluation(result, playoff_seed_dist, team_probs):
    rows = []
    for i, s in enumerate(result.stage3.advancers, start=1):
        team_rows = playoff_seed_dist[playoff_seed_dist.canonical_model_name == s.team_id]
        realized = team_rows[team_rows.playoff_seed == i]
        realized_p = float(realized["conditional_on_reaching_playoffs_probability"].iloc[0]) if len(realized) else None
        modal_row = team_rows.loc[team_rows["conditional_on_reaching_playoffs_probability"].idxmax()] if len(team_rows) else None
        p_playoffs_row = team_probs[(team_probs.canonical_model_name == s.team_id) & (team_probs.metric == "reach_playoffs")]
        rows.append({
            "team": s.team_id, "actual_seed": i, "probability_of_actual_seed": realized_p,
            "most_probable_seed_given_playoffs": int(modal_row["playoff_seed"]) if modal_row is not None else None,
            "modal_seed_matched_actual": (int(modal_row["playoff_seed"]) == i) if modal_row is not None else None,
            "reach_playoffs_probability": float(p_playoffs_row["probability"].iloc[0]) if len(p_playoffs_row) else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Favorite-wins path vs reality
# ---------------------------------------------------------------------------

def replay_favorite_path():
    import tournament.simulation.cologne_pre_event_simulation as sim
    import tournament.cologne_2026.phase8d_common as p8d
    matrix_df = pd.read_parquet(MATRIX_PATH, engine="fastparquet")
    lookup = sim.build_matrix_lookup(matrix_df)
    rules = te.load_frozen_rules()
    stage1, stage2_direct, stage3_direct = p8d.build_cologne_entrants()
    provider = sim.FavoriteWinsProvider(lookup)
    return te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)


def favorite_path_vs_reality(actual_result, favorite_result, favorite_path_json):
    if favorite_result.champion != favorite_path_json["champion"]:
        raise ValueError(f"favorite-path replay mismatch: got champion {favorite_result.champion}, "
                          f"frozen favorite_path.json says {favorite_path_json['champion']}")

    def sets_for(result):
        pt, sf, fi, ch = milestone_labels_from_replay(result)
        return {"stage_1_advancers": {s.team_id for s in result.stage1.advancers},
                "stage_2_advancers": {s.team_id for s in result.stage2.advancers},
                "stage_3_advancers_playoff_teams": pt, "semifinalists": sf, "finalists": fi, "champion": {ch}}

    actual_sets, favorite_sets = sets_for(actual_result), sets_for(favorite_result)
    structural = {k: set_comparison(favorite_sets[k], actual_sets[k]) for k in actual_sets}

    def match_index(result):
        idx = {}
        for e in result.full_trace():
            m = e.match
            key = (m.stage, m.round_number, frozenset({m.team_a, m.team_b}))
            idx[key] = {"team_a": m.team_a, "team_b": m.team_b, "best_of": m.best_of,
                        "predicted_winner": e.resolution.winner, "probability_team_a": e.resolution.probability_team_a}
        return idx

    actual_idx, favorite_idx = match_index(actual_result), match_index(favorite_result)
    shared_keys = set(actual_idx) & set(favorite_idx)
    matched_predictions = []
    for key in shared_keys:
        a, f = actual_idx[key], favorite_idx[key]
        actual_match = None
        for e in actual_result.full_trace():
            m = e.match
            if (m.stage, m.round_number, frozenset({m.team_a, m.team_b})) == key:
                actual_match = e
                break
        matched_predictions.append({
            "stage": key[0], "round": key[1], "team_a": a["team_a"], "team_b": a["team_b"],
            "favorite_path_predicted_winner": f["predicted_winner"], "actual_winner": actual_match.resolution.winner,
            "correct": f["predicted_winner"] == actual_match.resolution.winner,
        })
    n_correct = sum(1 for r in matched_predictions if r["correct"])
    return {
        "structural_set_comparisons": structural,
        "matched_identical_matchups": {
            "n_shared_matchups_between_favorite_path_and_reality": len(shared_keys),
            "n_correct_among_shared": n_correct,
            "accuracy_among_shared_matchups_only": (n_correct / len(shared_keys)) if shared_keys else None,
            "detail": matched_predictions,
        },
        "note": "ACTUAL-MATCH MODEL ACCURACY (see match-level metrics) is kept strictly separate from "
                "DETERMINISTIC FAVORITE-PATH SIMILARITY here. Individual predictions are compared only "
                "where the exact same (stage, round, unordered team pair) occurs in both paths - "
                "divergent downstream favorite-path matchups that never happened in reality are never "
                "penalized as wrong predictions.",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_roster():
    import yaml
    cfg = yaml.safe_load(TOURNAMENT_YAML.read_bytes())
    p = cfg["participants"]
    roster = {}
    for group in ("stage_1_entrants", "stage_2_direct_entrants", "stage_3_direct_entrants"):
        for t in p[group]:
            roster[t["canonical_model_name"]] = {"pre_event_seed": t["pre_event_seed"]}
    return roster


def compute_all_metrics():
    pred_df = pd.read_parquet(PRED_PATH, engine="fastparquet")
    team_probs = pd.read_csv(TEAM_PROB_PATH)
    swiss_records = pd.read_csv(SWISS_RECORD_PATH)
    playoff_seed_dist = pd.read_csv(PLAYOFF_SEED_PATH)
    favorite_path_json = json.loads(FAVORITE_PATH_PATH.read_text(encoding="utf-8"))
    roster = load_roster()

    result, _provider = aop.replay_actual_tournament()
    playoff_teams, semifinalists, finalists, champion = milestone_labels_from_replay(result)

    match_metrics = compute_match_metrics(pred_df)
    winner_prob_analysis = actual_winner_probability_analysis(pred_df)
    path_log_score = realized_path_log_score(pred_df)
    champ_analysis = champion_analysis(champion, team_probs)
    milestone_df = build_milestone_comparison(team_probs, roster, playoff_teams, semifinalists, finalists, champion)
    milestone_quality = milestone_probability_quality(milestone_df)
    top_k = top_k_set_comparisons(team_probs, roster, playoff_teams, semifinalists, finalists, champion)
    advancement = stage_advancement_evaluation(result, team_probs, roster)
    swiss_record_df, swiss_record_summary = swiss_terminal_record_evaluation(result, swiss_records)
    playoff_seed_df = playoff_seed_evaluation(result, playoff_seed_dist, team_probs)
    favorite_result = replay_favorite_path()
    favorite_comparison = favorite_path_vs_reality(result, favorite_result, favorite_path_json)

    return {
        "match_metrics": match_metrics,
        "winner_probability_analysis": winner_prob_analysis,
        "realized_path_log_score": path_log_score,
        "champion_analysis": champ_analysis,
        "milestone_probability_quality": milestone_quality,
        "top_k_set_comparisons": top_k,
        "stage_advancement_evaluation": advancement,
        "swiss_terminal_record_summary": swiss_record_summary,
        "favorite_path_vs_reality": favorite_comparison,
        "milestone_df": milestone_df, "swiss_record_df": swiss_record_df, "playoff_seed_df": playoff_seed_df,
        "playoff_teams": sorted(playoff_teams), "semifinalists": sorted(semifinalists),
        "finalists": sorted(finalists), "champion": champion,
        "pred_df": pred_df,
    }


if __name__ == "__main__":
    m = compute_all_metrics()
    printable = {k: v for k, v in m.items() if not k.endswith("_df") and k != "pred_df"}
    print(json.dumps(printable, indent=2, default=str))
