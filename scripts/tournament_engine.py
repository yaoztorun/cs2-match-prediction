"""
Pure Major-tournament mechanics engine (Phase 8C).

Implements the Swiss-stage and playoff-bracket mechanics frozen in
config/tournaments/iem_cologne_major_2026_pre_event.yaml (Phase 8B) as
deterministic, outcome-provider-driven simulation. This module knows
NOTHING about machine learning, feature engineering, or Cologne's real
teams: it never imports joblib/xgboost/sklearn, never reads data/raw or
data/interim, and never reads the frozen YAML's `participants` section
(only `swiss_rules` / `match_formats` / `playoff_bracket` / `metadata` are
read, via load_frozen_rules()). Match winners are supplied entirely through
an injected OutcomeProvider - the engine only ever asks "who won this
match?" and never infers or predicts an answer itself.

DESIGN
------
Mirrors the rest of the repository's "engine" convention (team_form_engine.py,
map_feature_engine.py): plain dataclasses hold state, module-level pure
functions operate on that state. `MatchSpec` is a FROZEN dataclass - once a
round's pairings are generated from the round-start state (round-start
atomicity, see run_swiss_round), nothing about that pairing (including its
recorded pre-match record/Difficulty-Score snapshot) is ever mutated again,
even though the live team state keeps changing in later rounds. This is a
deliberate, load-bearing invariant (see difficulty_a_before/difficulty_b_before)
and is directly enforced by dataclass immutability rather than by convention
alone.

REMATCH FALLBACK (documented convention for the Valve rulebook's
under-specified "if possible" clause)
------------------------------------------------------------------
The frozen rule says teams should avoid a same-stage rematch "if possible."
This engine treats that as an explicit minimize-then-fallback contract, not
a hard constraint that can crash an otherwise-valid simulated tournament:
  1. Search for a pairing arrangement with zero rematches.
  2. If one exists, it is used - no rematch is ever permitted when a
     rematch-free arrangement exists.
  3. Only if EVERY possible arrangement contains at least one rematch
     (proven by exhaustive search, not assumed) does the engine fall back
     to the arrangement with the fewest rematches, chosen deterministically
     (see constrained_pool_pairing / priority_table_pairing), and records
     `rematch_fallback_used` / `rematch_count` on every affected match so
     the fallback is always visible in the trace, never silent.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from _common import ROOT

FROZEN_YAML_PATH = ROOT / "config" / "tournaments" / "iem_cologne_major_2026_pre_event.yaml"
FROZEN_YAML_SHA256 = "e481ca4dc3ab5bdf63636ad53eeeba8d3677305b643ecb98b7391e6419383ba3"

STAGE_1, STAGE_2, STAGE_3, PLAYOFFS = "stage_1", "stage_2", "stage_3", "playoffs"
SWISS_STAGES = (STAGE_1, STAGE_2, STAGE_3)

# Round-number -> expected {(wins, losses): pool_size} shape of the ACTIVE
# teams at the moment that round's pairings are generated. This is an
# assertion, never a dispatcher: which pairing ALGORITHM runs is selected
# by round_number alone (see run_swiss_round). A stage that produces a
# different shape than this is a bug, not a variant to silently handle.
EXPECTED_GROUPS_BY_ROUND = {
    1: {(0, 0): 16},
    2: {(1, 0): 8, (0, 1): 8},
    3: {(2, 0): 4, (1, 1): 8, (0, 2): 4},
    4: {(2, 1): 6, (1, 2): 6},
    5: {(2, 2): 6},
}
SWISS_ROUNDS = (1, 2, 3, 4, 5)
EXPECTED_STAGE_MATCH_COUNT = 8 + 8 + 8 + 6 + 3  # 33


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class TeamEntry:
    """A team entering a stage, seeded but with no record yet."""
    team_id: str
    display_name: str
    initial_stage_seed: int

    def to_dict(self):
        return {"team_id": self.team_id, "display_name": self.display_name,
                "initial_stage_seed": self.initial_stage_seed}


@dataclass
class TeamStageState:
    """Mutable in-stage state for one team. Reset fresh at every stage
    transition - no wins/losses/opponents/status carries over."""
    team_id: str
    display_name: str
    initial_stage_seed: int
    current_seed: int
    wins: int = 0
    losses: int = 0
    opponents: list = field(default_factory=list)   # list[team_id], stage-local only
    status: str = "active"                            # active | advanced | eliminated

    def to_dict(self):
        return {"team_id": self.team_id, "display_name": self.display_name,
                "initial_stage_seed": self.initial_stage_seed, "current_seed": self.current_seed,
                "wins": self.wins, "losses": self.losses, "opponents": list(self.opponents),
                "status": self.status}


@dataclass(frozen=True)
class MatchSpec:
    """A frozen matchup, generated once from round-start state and never
    mutated afterward. team_a is always the better (lower-numbered) current
    seed at pairing time, team_b the worse - an ENGINE REPRESENTATION
    convention only (never CT/T, never map-veto Team A/B)."""
    match_id: str
    stage: str
    round_number: int
    record_group: str
    team_a: str
    team_b: str
    seed_a: int
    seed_b: int
    best_of: int
    team_a_record_before: Optional[tuple] = None
    team_b_record_before: Optional[tuple] = None
    difficulty_a_before: Optional[int] = None
    difficulty_b_before: Optional[int] = None
    priority_pattern_used: Optional[int] = None
    rematch_fallback_used: bool = False
    rematch_count: int = 0
    source_a: Optional[str] = None
    source_b: Optional[str] = None
    bracket_side: Optional[str] = None

    def to_dict(self):
        return {
            "match_id": self.match_id, "stage": self.stage, "round_number": self.round_number,
            "record_group": self.record_group, "team_a": self.team_a, "team_b": self.team_b,
            "seed_a": self.seed_a, "seed_b": self.seed_b, "best_of": self.best_of,
            "team_a_record_before": list(self.team_a_record_before) if self.team_a_record_before else None,
            "team_b_record_before": list(self.team_b_record_before) if self.team_b_record_before else None,
            "difficulty_a_before": self.difficulty_a_before, "difficulty_b_before": self.difficulty_b_before,
            "priority_pattern_used": self.priority_pattern_used,
            "rematch_fallback_used": self.rematch_fallback_used, "rematch_count": self.rematch_count,
            "source_a": self.source_a, "source_b": self.source_b, "bracket_side": self.bracket_side,
        }


@dataclass(frozen=True)
class MatchResolution:
    """winner/loser are team_ids. probability_team_a/provider_metadata are
    opaque - the engine carries them through the trace but never interprets
    them (reserved for Phase 8D)."""
    winner: str
    loser: Optional[str] = None
    probability_team_a: Optional[float] = None
    provider_metadata: Optional[dict] = None

    def to_dict(self):
        return {"winner": self.winner, "loser": self.loser,
                "probability_team_a": self.probability_team_a,
                "provider_metadata": self.provider_metadata}


@dataclass
class MatchTraceEntry:
    match: MatchSpec
    resolution: MatchResolution
    team_a_record_after: tuple
    team_b_record_after: tuple

    def to_dict(self):
        d = self.match.to_dict()
        d["winner"] = self.resolution.winner
        d["loser"] = self.resolution.loser
        d["provider_metadata"] = self.resolution.provider_metadata
        d["probability_team_a"] = self.resolution.probability_team_a
        d["team_a_record_after"] = list(self.team_a_record_after) if self.team_a_record_after is not None else None
        d["team_b_record_after"] = list(self.team_b_record_after) if self.team_b_record_after is not None else None
        return d


@dataclass
class SwissStageResult:
    stage: str
    entrants: list                  # list[TeamEntry]
    final_order: list               # list[TeamStageState], all 16, best first
    advancers: list                 # list[TeamStageState], the 8 in final order
    eliminated: list                # list[TeamStageState], the 8 in final order
    trace: list                     # list[MatchTraceEntry]
    rematch_fallback_events: int = 0
    max_rematches_in_a_pairing: int = 0

    def to_dict(self):
        return {
            "stage": self.stage,
            "entrants": [e.to_dict() for e in self.entrants],
            "final_order": [t.to_dict() for t in self.final_order],
            "advancers": [t.to_dict() for t in self.advancers],
            "eliminated": [t.to_dict() for t in self.eliminated],
            "trace": [m.to_dict() for m in self.trace],
            "rematch_fallback_events": self.rematch_fallback_events,
            "max_rematches_in_a_pairing": self.max_rematches_in_a_pairing,
        }


@dataclass
class PlayoffResult:
    trace: list                     # list[MatchTraceEntry], 7 total
    champion: str
    runner_up: str

    def to_dict(self):
        return {"trace": [m.to_dict() for m in self.trace],
                "champion": self.champion, "runner_up": self.runner_up}


@dataclass
class TournamentResult:
    stage1: SwissStageResult
    stage2: SwissStageResult
    stage3: SwissStageResult
    playoffs: PlayoffResult
    champion: str

    def to_dict(self):
        return {"stage1": self.stage1.to_dict(), "stage2": self.stage2.to_dict(),
                "stage3": self.stage3.to_dict(), "playoffs": self.playoffs.to_dict(),
                "champion": self.champion}

    def full_trace(self):
        return (self.stage1.trace + self.stage2.trace + self.stage3.trace + self.playoffs.trace)


@dataclass
class TournamentRules:
    """Extracted ONCE from the frozen YAML's swiss_rules/match_formats/
    playoff_bracket/metadata keys. NEVER reads `participants` - that key is
    intentionally never touched anywhere in this module."""
    advancement_wins: int
    elimination_losses: int
    teams_per_stage: int
    round4_plus_priority_table: list
    stage1_default_bo: int
    stage2_default_bo: int
    stage3_all_bo3: bool
    playoff_qf_bo: int
    playoff_sf_bo: int
    playoff_final_bo: int
    bracket_a: list        # [(1,8), (4,5)]
    bracket_b: list        # [(2,7), (3,6)]
    source_yaml_sha256: str


def _parse_bo(label):
    return int(str(label).lower().removeprefix("bo"))


def _parse_bracket(pairs):
    return [tuple(int(x) for x in p.lower().split("v")) for p in pairs]


def load_frozen_rules(path=FROZEN_YAML_PATH, expected_sha256=FROZEN_YAML_SHA256):
    """Reads ONLY swiss_rules / match_formats / playoff_bracket / metadata
    from the frozen tournament YAML. Raises ValueError if the file's hash
    has drifted from the frozen constant. This function - and this module -
    never reads the YAML's `participants` key."""
    path = Path(path)
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_sha256:
        raise ValueError(f"frozen tournament YAML hash mismatch: expected {expected_sha256}, got {actual_hash}")
    cfg = yaml.safe_load(raw)
    sw = cfg["swiss_rules"]
    mf = cfg["match_formats"]
    pb = cfg["playoff_bracket"]
    meta = cfg["metadata"]
    return TournamentRules(
        advancement_wins=sw["advancement_wins"],
        elimination_losses=sw["elimination_losses"],
        teams_per_stage=meta["teams_per_swiss_stage"],
        round4_plus_priority_table=sw["round_4_plus_priority_table"],
        stage1_default_bo=_parse_bo(mf["stage_1"]["default"]),
        stage2_default_bo=_parse_bo(mf["stage_2"]["default"]),
        stage3_all_bo3=(mf["stage_3"]["default"] == "bo3"),
        playoff_qf_bo=mf["playoffs"]["quarterfinal"]["best_of"],
        playoff_sf_bo=mf["playoffs"]["semifinal"]["best_of"],
        playoff_final_bo=mf["playoffs"]["grand_final"]["best_of"],
        bracket_a=_parse_bracket(pb["bracket_a"]),
        bracket_b=_parse_bracket(pb["bracket_b"]),
        source_yaml_sha256=actual_hash,
    )


# ---------------------------------------------------------------------------
# Outcome-provider interface (synthetic/test providers only - no ML here)
# ---------------------------------------------------------------------------

class OutcomeProvider:
    def resolve_match(self, match: MatchSpec) -> MatchResolution:
        raise NotImplementedError


def _finalize_resolution(match: MatchSpec, resolution: MatchResolution) -> MatchResolution:
    """Validates winner in {team_a, team_b} and derives/confirms loser.
    Every provider's output passes through here before the engine uses it,
    so providers themselves never need to duplicate this logic."""
    if resolution.winner not in (match.team_a, match.team_b):
        raise ValueError(f"{match.match_id}: winner {resolution.winner!r} is not one of the two paired teams "
                          f"({match.team_a}, {match.team_b})")
    derived_loser = match.team_b if resolution.winner == match.team_a else match.team_a
    if resolution.loser is not None and resolution.loser != derived_loser:
        raise ValueError(f"{match.match_id}: resolution.loser={resolution.loser!r} contradicts "
                          f"winner={resolution.winner!r} (expected loser {derived_loser!r})")
    if resolution.loser is None:
        resolution = replace(resolution, loser=derived_loser)
    return resolution


class ScriptedOutcomeProvider(OutcomeProvider):
    """script: match_id -> winning team_id."""
    def __init__(self, script: dict):
        self.script = dict(script)

    def resolve_match(self, match: MatchSpec) -> MatchResolution:
        if match.match_id not in self.script:
            raise ValueError(f"no scripted outcome for match_id={match.match_id!r}")
        return MatchResolution(winner=self.script[match.match_id])


class HigherSeedWinsProvider(OutcomeProvider):
    """Fully deterministic: the better (lower-numbered) current seed always
    wins. team_a is always the better seed by engine-orientation convention,
    but this is checked explicitly rather than assumed."""
    def resolve_match(self, match: MatchSpec) -> MatchResolution:
        if match.seed_a == match.seed_b:
            raise ValueError(f"{match.match_id}: seed_a == seed_b == {match.seed_a}, cannot break tie")
        winner = match.team_a if match.seed_a < match.seed_b else match.team_b
        return MatchResolution(winner=winner)


class SeededRandomOutcomeProvider(OutcomeProvider):
    """Owns a single np.random.default_rng(seed), advanced once per
    resolve_match call. Reproducible given a fixed seed AND the engine's
    fixed round-start-atomic call order. Stress-testing only - never a
    stand-in for a real prediction."""
    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def resolve_match(self, match: MatchSpec) -> MatchResolution:
        winner = match.team_a if self._rng.random() < 0.5 else match.team_b
        return MatchResolution(winner=winner)


# ---------------------------------------------------------------------------
# Seeding / Difficulty Score
# ---------------------------------------------------------------------------

def difficulty_score(team_id, states_by_id):
    """Dynamic Buchholz: sum over every opponent already faced of
    (opponent.current_wins - opponent.current_losses), using each
    opponent's CURRENT record. Never cached - always recomputed from
    states_by_id at the moment of the call."""
    team = states_by_id[team_id]
    return sum(states_by_id[opp].wins - states_by_id[opp].losses for opp in team.opponents)


def compute_seed_order(states):
    """Canonical ordering: (record, Difficulty Score, initial seed). Sorts
    ALL passed teams (active and terminal alike) and assigns
    current_seed = rank (1 = best). The one seeding implementation used for
    both mid-stage pairing (callers filter to active teams first) and final
    stage order (section 17) - same key, same tie-break chain."""
    states_by_id = {s.team_id: s for s in states}

    def key(s):
        return (-s.wins, s.losses, -difficulty_score(s.team_id, states_by_id), s.initial_stage_seed)

    ordered = sorted(states, key=key)
    for rank, s in enumerate(ordered, start=1):
        s.current_seed = rank
    return ordered


def group_by_record(states):
    groups = {}
    for s in states:
        groups.setdefault((s.wins, s.losses), []).append(s)
    return groups


def _assert_group_shape(groups, expected, round_number):
    actual = {k: len(v) for k, v in groups.items()}
    if actual != expected:
        raise ValueError(f"round {round_number}: unexpected active record-group shape. "
                          f"expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
# Pairing algorithms
# ---------------------------------------------------------------------------

def _is_rematch(a_id, b_id, history):
    return b_id in history.get(a_id, set())


def constrained_pool_pairing(pool, history):
    """Round 2/3: exhaustive backtracking over a same-record pool.
    Minimizes total rematches first (a zero-rematch arrangement always beats
    any arrangement with a rematch); among arrangements achieving the
    minimum, prefers - at every step - the highest remaining seed paired
    against the worst available (lowest-seed) opponent, moving toward
    better opponents only when required. Pool sizes here are always <= 8
    (an EXPECTED_GROUPS_BY_ROUND-guaranteed property of rounds 2/3), so an
    O((n-1)!!) exhaustive search (<=105 matchings for n=8) is trivial.

    Returns (pairs, rematch_fallback_used, total_rematch_count)."""
    ordered_ids = [s.team_id for s in sorted(pool, key=lambda s: s.current_seed)]

    def best(remaining):
        if not remaining:
            return (0, [])
        highest = remaining[0]
        rest = remaining[1:]
        candidates = list(reversed(rest))     # worst (lowest) seed first, moving toward better seeds
        best_total, best_pairs = None, None
        for cand in candidates:
            this_rematch = 1 if _is_rematch(highest, cand, history) else 0
            sub_remaining = [t for t in rest if t != cand]
            sub_total, sub_pairs = best(sub_remaining)
            total = this_rematch + sub_total
            if best_total is None or total < best_total:
                best_total, best_pairs = total, [(highest, cand)] + sub_pairs
        return best_total, best_pairs

    total_rematches, pairs = best(ordered_ids)
    return pairs, total_rematches > 0, total_rematches


def priority_table_pairing(pool, history, priority_table):
    """Round 4/5: the frozen 15-row priority table applied to a six-team
    pool, ranked 1-6 by current_seed (pool-relative rank, matching the
    frozen YAML's own note that the table's pair labels are pod-relative).
    First pass: the earliest row with zero rematches wins outright. If none
    of the 15 rows achieves zero, the earliest row achieving the true
    minimum rematch count across all 15 is used instead (documented
    fallback, never a silent invention).

    Returns (pairs, priority_row_used, rematch_fallback_used, rematch_count)."""
    if len(pool) != 6:
        raise ValueError(f"priority_table_pairing requires exactly 6 teams, got {len(pool)}")
    ordered = sorted(pool, key=lambda s: s.current_seed)
    rank_to_id = {i + 1: s.team_id for i, s in enumerate(ordered)}

    best_row, best_pairs, best_count = None, None, None
    for row in priority_table:
        parsed = [tuple(int(x) for x in p.lower().split("v")) for p in row["pairs"]]
        team_pairs = [(rank_to_id[a], rank_to_id[b]) for a, b in parsed]
        count = sum(1 for a, b in team_pairs if _is_rematch(a, b, history))
        if best_count is None or count < best_count:
            best_row, best_pairs, best_count = row["priority"], team_pairs, count
        if best_count == 0:
            break
    if best_pairs is None:
        raise ValueError(f"no priority-table row could be evaluated for pool {[s.team_id for s in pool]}")
    return best_pairs, best_row, best_count > 0, best_count


def round1_pairing_ids(entrants):
    """1v9, 2v10, ..., 8v16 on initial_stage_seed."""
    ordered = sorted(entrants, key=lambda e: e.initial_stage_seed)
    seeds = [e.initial_stage_seed for e in ordered]
    if seeds != list(range(1, len(entrants) + 1)):
        raise ValueError(f"round1_pairing_ids requires contiguous seeds 1..N, got {seeds}")
    half = len(entrants) // 2
    return [(ordered[i].team_id, ordered[i + half].team_id) for i in range(half)]


# ---------------------------------------------------------------------------
# Format selection
# ---------------------------------------------------------------------------

def select_format(stage, wins, losses, rules: TournamentRules):
    """Derived from the shared pre-match record, never from round number.
    Caller (see _build_swiss_matches) asserts both paired teams share this
    record BEFORE calling this function."""
    if stage == STAGE_3:
        return 3 if rules.stage3_all_bo3 else rules.stage2_default_bo
    would_advance = (wins + 1) == rules.advancement_wins
    would_eliminate = (losses + 1) == rules.elimination_losses
    default_bo = rules.stage1_default_bo if stage == STAGE_1 else rules.stage2_default_bo
    return 3 if (would_advance or would_eliminate) else default_bo


# ---------------------------------------------------------------------------
# Match construction / result application
# ---------------------------------------------------------------------------

def make_swiss_match_id(stage, round_number, index):
    return f"{stage}_r{round_number}_m{index:02d}"


def make_playoff_match_id(round_label, index):
    return f"playoff_{round_label}_{index:02d}"


def _build_swiss_matches(pair_records, states_by_id, stage, round_number, rules):
    """pair_records: list of dicts with keys id_a, id_b, rematch_fallback_used,
    priority_pattern_used. Orients team_a=better seed, asserts the
    shared-record invariant (amendment #4) BEFORE deriving the format, and
    snapshots the pre-round Difficulty Score for both teams."""
    matches = []
    for idx, rec in enumerate(pair_records, start=1):
        a, b = states_by_id[rec["id_a"]], states_by_id[rec["id_b"]]
        if a.current_seed > b.current_seed:
            a, b = b, a
        if (a.wins, a.losses) != (b.wins, b.losses):
            raise ValueError(f"{stage} round {round_number}: paired teams do not share a record - "
                              f"{a.team_id}={a.wins}-{a.losses} vs {b.team_id}={b.wins}-{b.losses}")
        best_of = select_format(stage, a.wins, a.losses, rules)
        matches.append(MatchSpec(
            match_id=make_swiss_match_id(stage, round_number, idx), stage=stage, round_number=round_number,
            record_group=f"{a.wins}-{a.losses}", team_a=a.team_id, team_b=b.team_id,
            seed_a=a.current_seed, seed_b=b.current_seed, best_of=best_of,
            team_a_record_before=(a.wins, a.losses), team_b_record_before=(b.wins, b.losses),
            difficulty_a_before=difficulty_score(a.team_id, states_by_id),
            difficulty_b_before=difficulty_score(b.team_id, states_by_id),
            priority_pattern_used=rec.get("priority_pattern_used"),
            rematch_fallback_used=rec.get("rematch_fallback_used", False),
            rematch_count=rec.get("rematch_count", 0),
        ))
    return matches


def apply_result(states_by_id, match: MatchSpec, resolution: MatchResolution, rules: TournamentRules):
    winner, loser = states_by_id[resolution.winner], states_by_id[resolution.loser]
    for team in (winner, loser):
        if team.status != "active":
            raise ValueError(f"{match.match_id}: team {team.team_id} is already terminal ({team.status})")
    winner.wins += 1
    loser.losses += 1
    winner.opponents.append(loser.team_id)
    loser.opponents.append(winner.team_id)
    if winner.wins > rules.advancement_wins or loser.losses > rules.elimination_losses:
        raise ValueError(f"{match.match_id}: record exceeded terminal thresholds after applying result")
    winner.status = "advanced" if winner.wins == rules.advancement_wins else "active"
    loser.status = "eliminated" if loser.losses == rules.elimination_losses else "active"


def build_trace_entry(match, resolution, a_after, b_after):
    return MatchTraceEntry(match=match, resolution=resolution,
                            team_a_record_after=a_after, team_b_record_after=b_after)


# ---------------------------------------------------------------------------
# Round / stage orchestration
# ---------------------------------------------------------------------------

def run_swiss_round(states_by_id, history, stage, round_number, rules, outcome_provider):
    """Round-start atomicity: (1) recompute seed order + Difficulty Score
    from the complete post-previous-round state, (2) group active teams by
    record and assert the expected shape for this round_number, (3) select
    the pairing ALGORITHM by round_number alone (never by pool size -
    pool size is only the assertion in step 2), (4) freeze all matchups,
    (5) only then resolve outcomes and apply results."""
    compute_seed_order(list(states_by_id.values()))
    active = [s for s in states_by_id.values() if s.status == "active"]
    groups = group_by_record(active)
    _assert_group_shape(groups, EXPECTED_GROUPS_BY_ROUND[round_number], round_number)

    pool_order = sorted(groups.keys(), key=lambda k: (-k[0], k[1]))
    pair_records = []
    pool_fallback_events, pool_max_rematches = 0, 0

    if round_number == 1:
        entrants = [TeamEntry(s.team_id, s.display_name, s.initial_stage_seed) for s in active]
        for id_a, id_b in round1_pairing_ids(entrants):
            pair_records.append({"id_a": id_a, "id_b": id_b, "rematch_fallback_used": False,
                                  "rematch_count": 0, "priority_pattern_used": None})
    elif round_number in (2, 3):
        for key in pool_order:
            pairs, fallback_used, total_count = constrained_pool_pairing(groups[key], history)
            if fallback_used:
                pool_fallback_events += 1
                pool_max_rematches = max(pool_max_rematches, total_count)
            for id_a, id_b in pairs:
                pair_records.append({"id_a": id_a, "id_b": id_b, "rematch_fallback_used": fallback_used,
                                      "rematch_count": 1 if _is_rematch(id_a, id_b, history) else 0,
                                      "priority_pattern_used": None})
    elif round_number in (4, 5):
        for key in pool_order:
            pairs, row_used, fallback_used, total_count = priority_table_pairing(
                groups[key], history, rules.round4_plus_priority_table)
            if fallback_used:
                pool_fallback_events += 1
                pool_max_rematches = max(pool_max_rematches, total_count)
            for id_a, id_b in pairs:
                pair_records.append({"id_a": id_a, "id_b": id_b, "rematch_fallback_used": fallback_used,
                                      "rematch_count": 1 if _is_rematch(id_a, id_b, history) else 0,
                                      "priority_pattern_used": row_used})
    else:
        raise ValueError(f"unexpected swiss round_number={round_number}")

    matches = _build_swiss_matches(pair_records, states_by_id, stage, round_number, rules)

    wins_before_round = sum(s.wins for s in states_by_id.values())
    losses_before_round = sum(s.losses for s in states_by_id.values())
    trace_entries = []
    seen_teams = set()
    for match in matches:
        if match.team_a in seen_teams or match.team_b in seen_teams:
            raise ValueError(f"{match.match_id}: a team appears in more than one match this round")
        seen_teams.add(match.team_a)
        seen_teams.add(match.team_b)
        resolution = _finalize_resolution(match, outcome_provider.resolve_match(match))
        apply_result(states_by_id, match, resolution, rules)
        history.setdefault(match.team_a, set()).add(match.team_b)
        history.setdefault(match.team_b, set()).add(match.team_a)
        a_after = states_by_id[match.team_a]
        b_after = states_by_id[match.team_b]
        trace_entries.append(build_trace_entry(match, resolution, (a_after.wins, a_after.losses),
                                                (b_after.wins, b_after.losses)))

    wins_after_round = sum(s.wins for s in states_by_id.values())
    losses_after_round = sum(s.losses for s in states_by_id.values())
    validate_round_invariants(matches, trace_entries, active, states_by_id,
                               wins_before_round, losses_before_round, wins_after_round, losses_after_round)
    return trace_entries, pool_fallback_events, pool_max_rematches


def run_swiss_stage(entrants, stage, rules, outcome_provider):
    if len(entrants) != rules.teams_per_stage:
        raise ValueError(f"{stage}: expected {rules.teams_per_stage} entrants, got {len(entrants)}")
    states_by_id = {e.team_id: TeamStageState(e.team_id, e.display_name, e.initial_stage_seed,
                                               current_seed=e.initial_stage_seed) for e in entrants}
    history = {tid: set() for tid in states_by_id}

    trace = []
    total_fallback_events, max_rematches = 0, 0
    for round_number in SWISS_ROUNDS:
        entries, fb_events, fb_max = run_swiss_round(states_by_id, history, stage, round_number, rules, outcome_provider)
        trace.extend(entries)
        total_fallback_events += fb_events
        max_rematches = max(max_rematches, fb_max)

    final_order, advancers, eliminated = finalize_stage(states_by_id)
    validate_stage_completion(entrants, final_order, advancers, eliminated, trace)
    return SwissStageResult(stage=stage, entrants=entrants, final_order=final_order,
                             advancers=advancers, eliminated=eliminated, trace=trace,
                             rematch_fallback_events=total_fallback_events,
                             max_rematches_in_a_pairing=max_rematches)


def finalize_stage(states_by_id):
    """Explicit final-order computation (amendment #6): record -> Difficulty
    Score -> initial seed over ALL 16 teams, then split by terminal status.
    Advancers/eliminated are read off THIS order - never off the order in
    which a team happened to reach its terminal record."""
    final_order = compute_seed_order(list(states_by_id.values()))
    advancers = [s for s in final_order if s.status == "advanced"]
    eliminated = [s for s in final_order if s.status == "eliminated"]
    if len(advancers) != 8 or len(eliminated) != 8:
        raise ValueError(f"stage completion invariant violated: {len(advancers)} advanced, "
                          f"{len(eliminated)} eliminated (expected 8/8)")
    return final_order, advancers, eliminated


def next_stage_entrants(direct_entrants, previous_result: SwissStageResult):
    """Direct entrants keep seeds 1-8 as given; the 8 advancers (already in
    final stage order) become seeds 9-16 in that order."""
    seeds = sorted(e.initial_stage_seed for e in direct_entrants)
    if seeds != list(range(1, 9)):
        raise ValueError(f"direct entrants must carry seeds 1..8, got {seeds}")
    advancer_entrants = [TeamEntry(s.team_id, s.display_name, 8 + i)
                          for i, s in enumerate(previous_result.advancers, start=1)]
    return list(direct_entrants) + advancer_entrants


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------

def validate_round_invariants(matches, trace_entries, active_before, states_by_id,
                               wins_before_round=None, losses_before_round=None,
                               wins_after_round=None, losses_after_round=None):
    active_ids_before = {s.team_id for s in active_before}
    paired_ids = set()
    for m in matches:
        if m.team_a == m.team_b:
            raise ValueError(f"{m.match_id}: a team cannot play itself")
        if m.team_a in paired_ids or m.team_b in paired_ids:
            raise ValueError(f"{m.match_id}: a team appears in more than one match this round")
        paired_ids.add(m.team_a)
        paired_ids.add(m.team_b)
        if m.team_a not in active_ids_before or m.team_b not in active_ids_before:
            raise ValueError(f"{m.match_id}: a paired team was not active before this round")
        if m.team_a_record_before != m.team_b_record_before:
            raise ValueError(f"{m.match_id}: paired teams did not share a pre-match record")
    if paired_ids != active_ids_before:
        raise ValueError("every active team must appear exactly once in the round's pairings")
    if wins_before_round is not None:
        wins_delta = wins_after_round - wins_before_round
        losses_delta = losses_after_round - losses_before_round
        if wins_delta != losses_delta or wins_delta != len(matches):
            raise ValueError(f"total wins delta ({wins_delta}) must equal total losses delta ({losses_delta}) "
                              f"must equal number of completed matches ({len(matches)})")
    for e in trace_entries:
        team = states_by_id[e.resolution.winner]
        if team.status not in ("active", "advanced"):
            raise ValueError(f"winner {team.team_id} has an unexpected post-match status {team.status}")


def validate_stage_completion(entrants, final_order, advancers, eliminated, trace):
    entrant_ids = {e.team_id for e in entrants}
    if len(entrant_ids) != 16:
        raise ValueError(f"expected 16 unique entrants, got {len(entrant_ids)}")
    if len(advancers) != 8:
        raise ValueError(f"expected 8 advancers, got {len(advancers)}")
    if len(eliminated) != 8:
        raise ValueError(f"expected 8 eliminated, got {len(eliminated)}")
    if len(trace) != EXPECTED_STAGE_MATCH_COUNT:
        raise ValueError(f"expected {EXPECTED_STAGE_MATCH_COUNT} matches for the stage, got {len(trace)}")
    match_counts = {}
    for e in trace:
        match_counts[e.match.team_a] = match_counts.get(e.match.team_a, 0) + 1
        match_counts[e.match.team_b] = match_counts.get(e.match.team_b, 0) + 1
    for tid, n in match_counts.items():
        if not (3 <= n <= 5):
            raise ValueError(f"team {tid} played {n} matches, expected between 3 and 5")
    final_ids = {t.team_id for t in final_order}
    if final_ids != entrant_ids:
        raise ValueError("final stage order does not contain exactly the 16 entrants")


# ---------------------------------------------------------------------------
# Playoffs
# ---------------------------------------------------------------------------

def _seed_lookup(entrants):
    return {e.initial_stage_seed: e for e in entrants}


def run_playoffs(seed_1_to_8, rules: TournamentRules, outcome_provider):
    seeds = sorted(e.initial_stage_seed for e in seed_1_to_8)
    if seeds != list(range(1, 9)):
        raise ValueError(f"playoffs require exactly seeds 1..8, got {seeds}")
    by_seed = _seed_lookup(seed_1_to_8)

    qf_pairs = rules.bracket_a + rules.bracket_b   # [(1,8),(4,5),(2,7),(3,6)]
    trace = []
    qf_winner_by_index = {}
    for i, (sa, sb) in enumerate(qf_pairs, start=1):
        a, b = by_seed[sa], by_seed[sb]
        if sa > sb:
            a, b = b, a
        match = MatchSpec(match_id=make_playoff_match_id("qf", i), stage=PLAYOFFS, round_number=1,
                           record_group="playoffs", team_a=a.team_id, team_b=b.team_id,
                           seed_a=a.initial_stage_seed, seed_b=b.initial_stage_seed, best_of=rules.playoff_qf_bo,
                           source_a=f"seed_{a.initial_stage_seed}", source_b=f"seed_{b.initial_stage_seed}",
                           bracket_side="A" if i <= 2 else "B")
        resolution = _finalize_resolution(match, outcome_provider.resolve_match(match))
        trace.append(build_trace_entry(match, resolution, None, None))
        qf_winner_by_index[i] = (match.match_id, resolution.winner)

    def sf_match(qf_x_idx, qf_y_idx, sf_index, bracket_side):
        (id_x, team_x), (id_y, team_y) = qf_winner_by_index[qf_x_idx], qf_winner_by_index[qf_y_idx]
        seed_x, seed_y = by_seed_of(team_x, seed_1_to_8), by_seed_of(team_y, seed_1_to_8)
        pairs = sorted([(id_x, team_x, seed_x), (id_y, team_y, seed_y)], key=lambda t: t[2])
        (source_a, team_a, seed_a), (source_b, team_b, seed_b) = pairs
        return MatchSpec(match_id=make_playoff_match_id("sf", sf_index), stage=PLAYOFFS, round_number=2,
                          record_group="playoffs", team_a=team_a, team_b=team_b, seed_a=seed_a, seed_b=seed_b,
                          best_of=rules.playoff_sf_bo, source_a=f"winner_{source_a}", source_b=f"winner_{source_b}",
                          bracket_side=bracket_side)

    sf1 = sf_match(1, 2, 1, "A")
    r1 = _finalize_resolution(sf1, outcome_provider.resolve_match(sf1))
    trace.append(build_trace_entry(sf1, r1, None, None))
    sf2 = sf_match(3, 4, 2, "B")
    r2 = _finalize_resolution(sf2, outcome_provider.resolve_match(sf2))
    trace.append(build_trace_entry(sf2, r2, None, None))

    (id_x, team_x), (id_y, team_y) = (sf1.match_id, r1.winner), (sf2.match_id, r2.winner)
    seed_x, seed_y = by_seed_of(team_x, seed_1_to_8), by_seed_of(team_y, seed_1_to_8)
    pairs = sorted([(id_x, team_x, seed_x), (id_y, team_y, seed_y)], key=lambda t: t[2])
    (source_a, team_a, seed_a), (source_b, team_b, seed_b) = pairs
    final = MatchSpec(match_id=make_playoff_match_id("final", 1), stage=PLAYOFFS, round_number=3,
                       record_group="playoffs", team_a=team_a, team_b=team_b, seed_a=seed_a, seed_b=seed_b,
                       best_of=rules.playoff_final_bo, source_a=f"winner_{source_a}", source_b=f"winner_{source_b}",
                       bracket_side=None)
    rf = _finalize_resolution(final, outcome_provider.resolve_match(final))
    trace.append(build_trace_entry(final, rf, None, None))

    if len(trace) != 7:
        raise ValueError(f"expected 7 playoff matches (4 QF + 2 SF + 1 Final), got {len(trace)}")
    return PlayoffResult(trace=trace, champion=rf.winner, runner_up=rf.loser)


def by_seed_of(team_id, entrants):
    for e in entrants:
        if e.team_id == team_id:
            return e.initial_stage_seed
    raise ValueError(f"team_id {team_id!r} not found among playoff entrants")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_major_tournament(stage1_entrants, stage2_direct_entrants, stage3_direct_entrants, rules, outcome_provider):
    stage1_result = run_swiss_stage(stage1_entrants, STAGE_1, rules, outcome_provider)
    stage2_entrants = next_stage_entrants(stage2_direct_entrants, stage1_result)
    stage2_result = run_swiss_stage(stage2_entrants, STAGE_2, rules, outcome_provider)
    stage3_entrants = next_stage_entrants(stage3_direct_entrants, stage2_result)
    stage3_result = run_swiss_stage(stage3_entrants, STAGE_3, rules, outcome_provider)

    playoff_seeds = [TeamEntry(s.team_id, s.display_name, i)
                     for i, s in enumerate(stage3_result.advancers, start=1)]
    playoff_result = run_playoffs(playoff_seeds, rules, outcome_provider)

    return TournamentResult(stage1=stage1_result, stage2=stage2_result, stage3=stage3_result,
                             playoffs=playoff_result, champion=playoff_result.champion)


# ---------------------------------------------------------------------------
# Canonical serialization / trace hashing
# ---------------------------------------------------------------------------

def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def trace_hash(result_dict):
    return hashlib.sha256(canonical_json_bytes(result_dict)).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic fixture (test/demo/stress use only - never real Cologne teams)
# ---------------------------------------------------------------------------

def build_synthetic_fixture():
    """32 wholly synthetic teams mirroring the frozen 16/8/8 entry
    structure, with no relation to any real Cologne participant."""
    stage1 = [TeamEntry(f"S1_{i:02d}", f"S1_{i:02d}", i) for i in range(1, 17)]
    stage2_direct = [TeamEntry(f"S2_{i:02d}", f"S2_{i:02d}", i) for i in range(1, 9)]
    stage3_direct = [TeamEntry(f"S3_{i:02d}", f"S3_{i:02d}", i) for i in range(1, 9)]
    return stage1, stage2_direct, stage3_direct


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class StressTestSummary:
    n_runs: int
    total_stages: int
    total_matches: int
    total_no_rematch_pairings: int
    total_rematch_fallback_events: int
    max_rematches_in_any_fallback_pairing: int
    invariant_failures: int
    max_rounds_per_stage: int
    failures: list = field(default_factory=list)

    def to_dict(self):
        return {"n_runs": self.n_runs, "total_stages": self.total_stages, "total_matches": self.total_matches,
                "total_no_rematch_pairings": self.total_no_rematch_pairings,
                "total_rematch_fallback_events": self.total_rematch_fallback_events,
                "max_rematches_in_any_fallback_pairing": self.max_rematches_in_any_fallback_pairing,
                "invariant_failures": self.invariant_failures, "max_rounds_per_stage": self.max_rounds_per_stage,
                "failures": self.failures}


def run_stress_test(n_runs=500, base_seed=0, rules=None):
    """Runs n_runs full synthetic Majors through SeededRandomOutcomeProvider
    (a distinct seed per run), asserting all engine invariants on every run
    (invariants are already enforced inside run_swiss_stage/run_playoffs -
    any violation raises, which this function catches and records rather
    than letting a single bad run abort the whole batch)."""
    rules = rules or load_frozen_rules()
    stage1, stage2_direct, stage3_direct = build_synthetic_fixture()

    total_matches, total_no_rematch, total_fallback_events, max_rematches = 0, 0, 0, 0
    invariant_failures, failures = 0, []
    for run_idx in range(n_runs):
        provider = SeededRandomOutcomeProvider(seed=base_seed + run_idx)
        try:
            result = run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
        except ValueError as e:
            invariant_failures += 1
            failures.append({"run_idx": run_idx, "error": str(e)})
            continue
        for stage_result in (result.stage1, result.stage2, result.stage3):
            total_fallback_events += stage_result.rematch_fallback_events
            max_rematches = max(max_rematches, stage_result.max_rematches_in_a_pairing)
            for entry in stage_result.trace:
                total_matches += 1
                if entry.match.rematch_count == 0:
                    total_no_rematch += 1
        total_matches += len(result.playoffs.trace)

    return StressTestSummary(
        n_runs=n_runs, total_stages=n_runs * 3, total_matches=total_matches,
        total_no_rematch_pairings=total_no_rematch, total_rematch_fallback_events=total_fallback_events,
        max_rematches_in_any_fallback_pairing=max_rematches, invariant_failures=invariant_failures,
        max_rounds_per_stage=5, failures=failures,
    )
