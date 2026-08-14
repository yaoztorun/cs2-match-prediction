"""
Phase 8D Monte Carlo driver. Consumes ONLY the frozen, already-validated
probability matrix - never calls RF V2 again. Runs 50,000 complete Majors
over the REAL 32 Cologne teams via tournament_engine.run_major_tournament
(unmodified from Phase 8C), aggregating team/stage/record/playoff/matchup
statistics in memory. Full traces are kept only for a small fixed set of
simulation indices (reproducibility/debug samples); everything else is
integer counts and probabilities derived from those counts.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

import tournament_engine as te

SWISS_STAGES = ("stage_1", "stage_2", "stage_3")


class CachedProbabilityOutcomeProvider(te.OutcomeProvider):
    """Looks up (team_a, team_b, best_of) in the frozen matrix and samples a
    winner from a per-simulation RNG. Lives outside tournament_engine.py -
    that module is not modified for Phase 8D."""

    def __init__(self, matrix_lookup, rng):
        self.matrix_lookup = matrix_lookup
        self.rng = rng

    def resolve_match(self, match: te.MatchSpec) -> te.MatchResolution:
        key = (match.team_a, match.team_b, match.best_of)
        if key not in self.matrix_lookup:
            raise ValueError(f"{match.match_id}: no cached probability for {key}")
        p_a = self.matrix_lookup[key]
        u = self.rng.random()
        winner = match.team_a if u < p_a else match.team_b
        return te.MatchResolution(winner=winner, probability_team_a=p_a,
                                   provider_metadata={"model_id": "series_random_forest_v2",
                                                       "prediction_mode": "pre_veto"})


class FavoriteWinsProvider(te.OutcomeProvider):
    """Zero-randomness deterministic favorite-wins path (amendment #10 tie
    policy): if P(team_a) >= 0.5, team_a wins; else team_b wins. Exact
    p=0.5 resolves to team_a."""

    def __init__(self, matrix_lookup):
        self.matrix_lookup = matrix_lookup

    def resolve_match(self, match: te.MatchSpec) -> te.MatchResolution:
        key = (match.team_a, match.team_b, match.best_of)
        p_a = self.matrix_lookup[key]
        winner = match.team_a if p_a >= 0.5 else match.team_b
        return te.MatchResolution(winner=winner, probability_team_a=p_a,
                                   provider_metadata={"model_id": "series_random_forest_v2",
                                                       "prediction_mode": "pre_veto"})


def build_matrix_lookup(df):
    return {(row.team_a, row.team_b, row.best_of): float(row.probability_team_a)
            for row in df.itertuples(index=False)}


def simulation_rng(base_seed, simulation_index):
    seed_seq = np.random.SeedSequence([base_seed, simulation_index])
    return np.random.default_rng(seed_seq), seed_seq


def run_one_simulation(simulation_index, base_seed, entrants, rules, matrix_lookup):
    rng, _ = simulation_rng(base_seed, simulation_index)
    provider = CachedProbabilityOutcomeProvider(matrix_lookup, rng)
    stage1, stage2_direct, stage3_direct = entrants
    return te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class Aggregator:
    all_team_ids: list
    team: dict = None
    matchup: Counter = None
    matchup_wins: dict = None
    rematch_by_stage_round: dict = None
    rematch_pool_events_by_stage: Counter = None
    rematch_max_by_stage: Counter = None
    champion_counts: Counter = None
    n_simulations: int = 0

    def __post_init__(self):
        self.team = {tid: {
            "participate_stage_1": 0, "participate_stage_2": 0, "participate_stage_3": 0,
            "reach_playoffs": 0, "reach_semifinal": 0, "reach_final": 0, "win_tournament": 0,
            "stage_1_participations": 0, "stage_1_advances": 0,
            "stage_2_participations": 0, "stage_2_advances": 0,
            "stage_3_participations": 0, "stage_3_advances": 0,
            "swiss_records": {s: Counter() for s in SWISS_STAGES},
            "playoff_seed_counts": Counter(),
        } for tid in self.all_team_ids}
        self.matchup = Counter()
        self.matchup_wins = defaultdict(lambda: Counter())
        self.rematch_by_stage_round = defaultdict(lambda: {"total_matches": 0, "zero_rematch_matches": 0,
                                                             "rematch_matches": 0})
        self.rematch_pool_events_by_stage = Counter()
        self.rematch_max_by_stage = Counter()
        self.champion_counts = Counter()

    def absorb(self, result: te.TournamentResult):
        self.n_simulations += 1

        for stage_result in (result.stage1, result.stage2, result.stage3):
            stage = stage_result.stage
            participant_ids = {e.team_id for e in stage_result.entrants}
            advancer_ids = {t.team_id for t in stage_result.advancers}
            for tid in participant_ids:
                t = self.team[tid]
                t[f"participate_{stage}"] += 1
                t[f"{stage}_participations"] += 1
                if tid in advancer_ids:
                    t[f"{stage}_advances"] += 1
                final_state = next(s for s in stage_result.final_order if s.team_id == tid)
                record_key = f"{final_state.wins}-{final_state.losses}"
                t["swiss_records"][stage][record_key] += 1

            self.rematch_pool_events_by_stage[stage] += stage_result.rematch_fallback_events
            self.rematch_max_by_stage[stage] = max(self.rematch_max_by_stage[stage],
                                                     stage_result.max_rematches_in_a_pairing)

            for entry in stage_result.trace:
                m = entry.match
                key = (stage, m.round_number)
                bucket = self.rematch_by_stage_round[key]
                bucket["total_matches"] += 1
                if m.rematch_count == 0:
                    bucket["zero_rematch_matches"] += 1
                else:
                    bucket["rematch_matches"] += 1
                mkey = (m.stage, m.round_number, m.best_of, m.team_a, m.team_b)
                self.matchup[mkey] += 1
                self.matchup_wins[mkey][entry.resolution.winner] += 1

        for i, tid in enumerate([e.team_id for e in result.stage3.advancers], start=1):
            t = self.team[tid]
            t["reach_playoffs"] += 1
            t["playoff_seed_counts"][i] += 1

        for entry in result.playoffs.trace:
            m = entry.match
            mkey = (m.stage, m.round_number, m.best_of, m.team_a, m.team_b)
            self.matchup[mkey] += 1
            self.matchup_wins[mkey][entry.resolution.winner] += 1
            if m.round_number == 2:   # semifinal: both participants reached the semifinal
                for tid in (m.team_a, m.team_b):
                    self.team[tid]["reach_semifinal"] += 1
            if m.round_number == 3:   # final: both participants reached the final
                for tid in (m.team_a, m.team_b):
                    self.team[tid]["reach_final"] += 1

        self.champion_counts[result.champion] += 1
        self.team[result.champion]["win_tournament"] += 1


def run_monte_carlo(n_simulations, base_seed, entrants, rules, matrix_lookup, all_team_ids,
                     sample_indices=()):
    agg = Aggregator(all_team_ids=all_team_ids)
    sample_traces = []
    sample_set = set(sample_indices)
    for i in range(n_simulations):
        result = run_one_simulation(i, base_seed, entrants, rules, matrix_lookup)
        agg.absorb(result)
        if i in sample_set:
            _, seed_seq = simulation_rng(base_seed, i)
            sample_traces.append({
                "simulation_index": i,
                "seed_sequence_entropy": list(seed_seq.entropy),
                "seed_sequence_generated_state": [int(x) for x in seed_seq.generate_state(4)],
                "champion": result.champion,
                "canonical_trace_hash": te.trace_hash(result.to_dict()),
                "trace": result.to_dict(),
            })
    return agg, sample_traces


# ---------------------------------------------------------------------------
# Monte Carlo statistics
# ---------------------------------------------------------------------------

def mc_stats(numerator_count, denominator_count):
    """amendment #4: correct conditional denominator, null when denom==0."""
    if denominator_count == 0:
        return {"numerator_count": 0, "denominator_count": 0, "probability": None,
                "mc_standard_error": None, "mc_ci95_low": None, "mc_ci95_high": None}
    p = numerator_count / denominator_count
    se = (p * (1.0 - p) / denominator_count) ** 0.5
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return {"numerator_count": numerator_count, "denominator_count": denominator_count,
            "probability": p, "mc_standard_error": se, "mc_ci95_low": lo, "mc_ci95_high": hi}


# ---------------------------------------------------------------------------
# Output tables (amendment #5: counts are the primary source of truth -
# every probability here is literally numerator_count/denominator_count,
# independently recomputable and re-checked by the validator)
# ---------------------------------------------------------------------------

UNCONDITIONAL_METRICS = ["participate_stage_1", "participate_stage_2", "participate_stage_3",
                          "reach_playoffs", "reach_semifinal", "reach_final", "win_tournament"]
CONDITIONAL_ADVANCE_METRICS = [("advance_from_stage_1", "stage_1_advances", "stage_1_participations"),
                                ("advance_from_stage_2", "stage_2_advances", "stage_2_participations"),
                                ("advance_from_stage_3", "stage_3_advances", "stage_3_participations")]


def build_team_probabilities_df(agg, teams):
    n = agg.n_simulations
    rows = []
    team_meta = {t["canonical_model_name"]: t for t in teams}
    for tid, t in agg.team.items():
        meta = team_meta[tid]
        for metric in UNCONDITIONAL_METRICS:
            stats = mc_stats(t[metric], n)
            rows.append({"canonical_model_name": tid, "display_name": meta["display_name"],
                         "starting_stage": meta["starting_stage"], "metric": metric,
                         "denominator_type": "unconditional_n_simulations", **stats})
        for metric, num_key, den_key in CONDITIONAL_ADVANCE_METRICS:
            stats = mc_stats(t[num_key], t[den_key])
            rows.append({"canonical_model_name": tid, "display_name": meta["display_name"],
                         "starting_stage": meta["starting_stage"], "metric": metric,
                         "denominator_type": "conditional_on_stage_participation", **stats})
    return pd.DataFrame(rows)


def build_swiss_record_distributions_df(agg, teams):
    n = agg.n_simulations
    rows = []
    team_meta = {t["canonical_model_name"]: t for t in teams}
    all_records = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    for tid, t in agg.team.items():
        meta = team_meta[tid]
        for stage in SWISS_STAGES:
            participations = t[f"{stage}_participations"]
            if participations == 0:
                continue
            counts = t["swiss_records"][stage]
            for record in all_records:
                count = counts.get(record, 0)
                unconditional = mc_stats(count, n)
                conditional = mc_stats(count, participations)
                rows.append({
                    "canonical_model_name": tid, "display_name": meta["display_name"], "stage": stage,
                    "record": record, "count": count,
                    "unconditional_probability": unconditional["probability"],
                    "unconditional_mc_standard_error": unconditional["mc_standard_error"],
                    "unconditional_mc_ci95_low": unconditional["mc_ci95_low"],
                    "unconditional_mc_ci95_high": unconditional["mc_ci95_high"],
                    "conditional_on_participating_probability": conditional["probability"],
                    "conditional_on_participating_denominator": conditional["denominator_count"],
                    "conditional_mc_standard_error": conditional["mc_standard_error"],
                    "conditional_mc_ci95_low": conditional["mc_ci95_low"],
                    "conditional_mc_ci95_high": conditional["mc_ci95_high"],
                })
    return pd.DataFrame(rows)


def build_playoff_seed_distributions_df(agg, teams):
    n = agg.n_simulations
    rows = []
    team_meta = {t["canonical_model_name"]: t for t in teams}
    for tid, t in agg.team.items():
        reach_playoffs_count = t["reach_playoffs"]
        if reach_playoffs_count == 0:
            continue
        meta = team_meta[tid]
        for seed in range(1, 9):
            count = t["playoff_seed_counts"].get(seed, 0)
            unconditional = mc_stats(count, n)
            conditional = mc_stats(count, reach_playoffs_count)
            rows.append({
                "canonical_model_name": tid, "display_name": meta["display_name"], "playoff_seed": seed,
                "count": count,
                "unconditional_probability": unconditional["probability"],
                "unconditional_mc_standard_error": unconditional["mc_standard_error"],
                "conditional_on_reaching_playoffs_probability": conditional["probability"],
                "conditional_on_reaching_playoffs_denominator": conditional["denominator_count"],
                "conditional_mc_standard_error": conditional["mc_standard_error"],
                "conditional_mc_ci95_low": conditional["mc_ci95_low"],
                "conditional_mc_ci95_high": conditional["mc_ci95_high"],
            })
    return pd.DataFrame(rows)


def build_matchup_frequencies_df(agg, matrix_lookup, n_simulations):
    rows = []
    for mkey, count in agg.matchup.items():
        stage, round_number, best_of, team_a, team_b = mkey
        wins = agg.matchup_wins[mkey]
        rows.append({
            "stage": stage, "round": round_number, "best_of": best_of,
            "team_a": team_a, "team_b": team_b,
            "times_matchup_occurred": count,
            "fraction_of_simulations": count / n_simulations,
            "model_probability_team_a": matrix_lookup.get((team_a, team_b, best_of)),
            "team_a_win_count": wins.get(team_a, 0),
            "team_b_win_count": wins.get(team_b, 0),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["stage", "round", "team_a", "team_b", "best_of"]).reset_index(drop=True)


def build_rematch_fallback_report(agg):
    """amendment #6: report only - never alters Phase 8C fallback behavior."""
    total_matches = sum(b["total_matches"] for b in agg.rematch_by_stage_round.values())
    zero_rematch = sum(b["zero_rematch_matches"] for b in agg.rematch_by_stage_round.values())
    rematch_matches = sum(b["rematch_matches"] for b in agg.rematch_by_stage_round.values())
    by_stage = {}
    for stage in SWISS_STAGES:
        stage_buckets = {k: v for k, v in agg.rematch_by_stage_round.items() if k[0] == stage}
        by_stage[stage] = {
            "total_matches": sum(b["total_matches"] for b in stage_buckets.values()),
            "zero_rematch_matches": sum(b["zero_rematch_matches"] for b in stage_buckets.values()),
            "rematch_matches": sum(b["rematch_matches"] for b in stage_buckets.values()),
            "pool_level_fallback_events": agg.rematch_pool_events_by_stage[stage],
            "max_rematches_in_any_one_pairing_event": agg.rematch_max_by_stage[stage],
        }
    by_stage_round = {f"{stage}_round_{round_number}": bucket
                       for (stage, round_number), bucket in sorted(agg.rematch_by_stage_round.items())}
    return {
        "total_swiss_pairing_events": total_matches,
        "zero_rematch_pairing_events": zero_rematch,
        "rematch_events": rematch_matches,
        "total_unavoidable_rematches": rematch_matches,
        "pool_level_fallback_events_total": sum(agg.rematch_pool_events_by_stage.values()),
        "max_rematches_in_any_one_pairing_event": max(agg.rematch_max_by_stage.values(), default=0),
        "breakdown_by_stage": by_stage,
        "breakdown_by_stage_round": by_stage_round,
    }


def build_favorite_path_trace(entrants, rules, matrix_lookup):
    stage1, stage2_direct, stage3_direct = entrants
    provider = FavoriteWinsProvider(matrix_lookup)
    result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
    matches = []
    for entry in result.full_trace():
        m = entry.match
        matches.append({
            "match_id": m.match_id, "stage": m.stage, "round": m.round_number,
            "record_group": m.record_group, "team_a": m.team_a, "team_b": m.team_b,
            "best_of": m.best_of, "probability_team_a": entry.resolution.probability_team_a,
            "predicted_winner": entry.resolution.winner,
        })
    return {"path_name": "deterministic favorite-wins path",
            "tie_policy": "if P(team_a) >= 0.5: team_a wins; else team_b wins",
            "champion": result.champion, "matches": matches,
            "canonical_trace_hash": te.trace_hash(result.to_dict())}
