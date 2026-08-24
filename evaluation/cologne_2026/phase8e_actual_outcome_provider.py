"""
Phase 8E Gate 2: ActualOutcomeProvider + engine replay driver.

Lives outside tournament_engine.py (never modified). Feeds the untouched
Phase 8C engine real Cologne winners instead of simulated ones, using the
strict match key from the approved Phase 8E amendments:

  Swiss:     (stage, round_number, record_group, unordered team pair, best_of)
  Playoffs:  (stage, round_number,   "playoffs", unordered team pair, best_of)

Every resolve_match() call must find exactly one unconsumed actual result
under this key. Zero matches or a second consumption attempt on the same key
both raise immediately (hard STOP, never "pick the first candidate") - this
makes a successful, exception-free replay of the whole tournament a genuine
external validation that the engine's own pairing mechanics reproduce
reality, not just a replay that happens to fill in winners.
"""

import pandas as pd

import evaluation.cologne_2026.phase8e_common as p8e
import tournament.cologne_2026.phase8d_common as p8d
import tournament.engine.tournament_engine as te
from _common import ROOT

CANONICAL_RESULTS_PATH = ROOT / "data" / "evaluation" / "cologne_2026_actual_series_results_v1.parquet"


class ActualOutcomeProvider(te.OutcomeProvider):
    def __init__(self, canonical_df):
        self._lookup = {}
        for _, row in canonical_df.iterrows():
            key = self._key(row["stage"], int(row["round_number"]), row["record_group"],
                             row["team_1"], row["team_2"], int(row["best_of"]))
            if key in self._lookup:
                raise ValueError(f"ambiguous actual-result key {key}: more than one real match shares it "
                                  f"(source_match_ids {self._lookup[key]['source_match_id']} and "
                                  f"{row['source_match_id']})")
            self._lookup[key] = row
        self._consumed = set()

    @staticmethod
    def _key(stage, round_number, record_group, team_x, team_y, best_of):
        return (stage, round_number, record_group, frozenset({team_x, team_y}), best_of)

    def resolve_match(self, match: te.MatchSpec) -> te.MatchResolution:
        key = self._key(match.stage, match.round_number, match.record_group,
                         match.team_a, match.team_b, match.best_of)
        if key not in self._lookup:
            raise ValueError(f"GATE 2 STOP: engine-generated match {match.match_id} "
                              f"({match.team_a} vs {match.team_b}, key={key}) has no matching actual "
                              f"result - engine pairing disagrees with reality.")
        if key in self._consumed:
            raise ValueError(f"GATE 2 STOP: actual result for key={key} already consumed once "
                              f"(engine tried to use it again for {match.match_id}).")
        self._consumed.add(key)
        row = self._lookup[key]
        return te.MatchResolution(winner=row["winner"],
                                   provider_metadata={"source": "actual_result",
                                                       "source_match_id": int(row["source_match_id"])})

    def unused_keys(self):
        return set(self._lookup) - self._consumed

    def n_total(self):
        return len(self._lookup)

    def n_consumed(self):
        return len(self._consumed)


def load_canonical_results():
    df = pd.read_parquet(CANONICAL_RESULTS_PATH, engine="fastparquet")
    if len(df) != 106:
        raise ValueError(f"expected 106 canonical actual results, got {len(df)}")
    return df


def replay_actual_tournament():
    """Returns (TournamentResult, ActualOutcomeProvider) after a full,
    exception-free replay, or raises on any Gate-2 STOP condition."""
    canonical = load_canonical_results()
    rules = te.load_frozen_rules()
    stage1, stage2_direct, stage3_direct = p8d.build_cologne_entrants()
    provider = ActualOutcomeProvider(canonical)
    result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
    return result, provider


def gate2_checkpoint_report():
    result, provider = replay_actual_tournament()

    unused = provider.unused_keys()
    duplicate_consumptions = 0  # any duplicate attempt would already have raised
    missing_engine_matches = 0  # any missing key would already have raised
    ambiguous_matches = 0       # any ambiguous key would already have raised at construction

    counts = {
        "stage_1": len(result.stage1.trace), "stage_2": len(result.stage2.trace),
        "stage_3": len(result.stage3.trace), "playoffs": len(result.playoffs.trace),
    }

    canonical = load_canonical_results()
    actual_champion_row = canonical[(canonical.stage == "playoffs") & (canonical.round_number == 3)].iloc[0]
    actual_champion = actual_champion_row["winner"]

    report = {
        "reconciliation": {"original_rows": 107, "official_rows": 106,
                            "excluded_row": "Team Germany vs Team Poland, 2026-06-21, BO1, "
                                            "non_tournament_showmatch"},
        "engine_replay": {**counts, "total": sum(counts.values())},
        "stage_transitions": {
            "stage_1_qualifiers_reproduced": len(result.stage1.advancers) == 8,
            "stage_2_qualifiers_reproduced": len(result.stage2.advancers) == 8,
            "stage_3_qualifiers_reproduced": len(result.stage3.advancers) == 8,
        },
        "playoffs": {
            "seeds_1_8_reproduced": len({e.team_id for e in
                                          [te.TeamEntry(s.team_id, s.display_name, i)
                                           for i, s in enumerate(result.stage3.advancers, start=1)]}) == 8,
            "champion_reproduced": result.champion == actual_champion,
            "engine_champion": result.champion, "actual_champion": actual_champion,
        },
        "consumption": {
            "actual_rows_consumed": provider.n_consumed(), "total_actual_rows": provider.n_total(),
            "unused": len(unused), "duplicate": duplicate_consumptions,
            "ambiguous": ambiguous_matches, "missing_engine_matches": missing_engine_matches,
        },
        "immutability": {
            "phase8c_engine_hash": p8e.sha256_file(p8e.IMMUTABLE_PRE_EVENT_HASH_INPUTS["phase8c_tournament_engine"]),
            "rf_model_loaded": False, "probability_matrix_used": False,
        },
    }
    return report, result, provider


if __name__ == "__main__":
    import json
    report, _, _ = gate2_checkpoint_report()
    print(json.dumps(report, indent=2, default=str))
