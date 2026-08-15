"""
Phase 9E Major Simulation Application Service. Sits ABOVE the frozen Phase
8C tournament engine (tournament_engine.py, never modified here) and wraps
three more already-frozen subsystems: Phase 8C (engine mechanics), Phase 8D
(historical Cologne probability matrix + 50,000-simulation Monte Carlo), and
Phase 8E (simulation-vs-reality evaluation). Never touches XGB V3 - every
tournament prediction, historical or interactive, is RF V2 pre-veto only
(`application_inference.predict_series_unknown_maps`).

TWO PERMANENT MODES, never conflated:
  - historical: `get_historical_cologne_pre_event`/`get_historical_cologne_
    results` are pure FILE-BACKED VIEWS over already-frozen Phase 8D/8E
    artifacts - no RF call, no matrix build, no engine run, no Monte Carlo,
    for a normal GET. Historical participant/team identifiers use the
    frozen Phase 8B YAML's own `canonical_model_name` values AS RECORDED
    (via phase8d_common.build_cologne_entrants) - never Phase 9B's
    deployment identity policy (`ai.resolve_team`). The two identity
    contracts are deliberately kept separate; drift in
    team_identity_policy.csv must never change whether a frozen historical
    endpoint can be served.
  - interactive: `validate_tournament_participants`/`build_tournament_
    probability_matrix`/`predict_tournament_path`/`simulate_tournament` use
    Phase 9B's `deployment_post_cologne_v1` identity resolution
    (`ai.resolve_team`) and build a FRESH 2,976-row RF matrix on demand
    (never at API startup, never persisted to research artifacts).

MATRIX IDENTITY != TOURNAMENT SCENARIO IDENTITY: the probability matrix only
depends on the 32-team SET (plus context/tier/prediction_datetime/RF
pipeline version) - never on seed/stage assignment, since it always
contains every ordered pair x BO1/3/5. Seed/stage assignment (and manual
overrides) determine the deterministic path / Monte Carlo RESULT, which is
therefore never cached by matrix key alone (amendment #2) - Phase 9E V1
keeps simulation results fully stateless/recomputed (amendment #21).
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from _common import ROOT
import application_inference as ai
import phase8d_common
import tournament_engine as te

CONFIG = ROOT / "config"
DATA = ROOT / "data"
EVAL = DATA / "evaluation"
SCRIPTS = ROOT / "scripts"

RULESETS_REGISTRY_PATH = CONFIG / "application_tournament_rulesets_v1.yaml"
DEFAULT_RULESET_ID = "iem_cologne_major_2026_format_v1"

VALID_STAGES = (te.STAGE_1, te.STAGE_2, te.STAGE_3)
PLAYOFF_ROUND_LABELS = {1: "quarterfinal", 2: "semifinal", 3: "grand_final"}

MAX_SIMULATION_COUNT = 50_000                 # amendment #10: bounded integer 1 <= n <= 50,000
PROCESS_POOL_MIN_SIMULATIONS = 2_000          # amendment #9: frozen after benchmarking (see report section P)
PROCESS_POOL_MAX_WORKERS = min(4, os.cpu_count() or 1)
MAX_MATRIX_CACHE_ENTRIES = 8                  # amendment #7: bounded, deterministic LRU eviction
MAX_MANUAL_OVERRIDES = te.EXPECTED_STAGE_MATCH_COUNT * 3 + 7  # 106: max matches in one complete path

_err = ai._err


# ---------------------------------------------------------------------------
# Ruleset registry (amendment #15: application/version METADATA only - the
# frozen te.load_frozen_rules() remains the sole executable rules authority)
# ---------------------------------------------------------------------------

def _load_ruleset_registry():
    return yaml.safe_load(RULESETS_REGISTRY_PATH.read_text(encoding="utf-8"))


def list_tournament_rulesets():
    reg = _load_ruleset_registry()
    return [{"ruleset_id": rid, **entry} for rid, entry in reg["rulesets"].items()]


def get_tournament_ruleset(ruleset_id):
    reg = _load_ruleset_registry()
    if ruleset_id not in reg["rulesets"]:
        _err("unknown_ruleset", f"unknown ruleset_id {ruleset_id!r} - must be one of "
             f"{sorted(reg['rulesets'])}", requested_ruleset_id=ruleset_id, available=sorted(reg["rulesets"]))
    return {"ruleset_id": ruleset_id, **reg["rulesets"][ruleset_id]}


def verify_ruleset_matches_engine_rules(ruleset_id=DEFAULT_RULESET_ID):
    """Amendment #15: the registry's descriptive facts must agree with what
    the frozen engine actually executes - never a second rules engine."""
    entry = get_tournament_ruleset(ruleset_id)
    rules = te.load_frozen_rules()
    checks = {
        "stage1_teams": rules.teams_per_stage == entry["stage1_teams"],
        "advancement_wins": rules.advancement_wins == entry["advancement_wins"],
        "elimination_losses": rules.elimination_losses == entry["elimination_losses"],
        "stage1_default_best_of": rules.stage1_default_bo == entry["stage1_default_best_of"],
        "stage2_default_best_of": rules.stage2_default_bo == entry["stage2_default_best_of"],
        "stage3_all_best_of_3": rules.stage3_all_bo3 == entry["stage3_all_best_of_3"],
        "playoff_quarterfinal_best_of": rules.playoff_qf_bo == entry["playoff_quarterfinal_best_of"],
        "playoff_semifinal_best_of": rules.playoff_sf_bo == entry["playoff_semifinal_best_of"],
        "playoff_grand_final_best_of": rules.playoff_final_bo == entry["playoff_grand_final_best_of"],
        "source_yaml_sha256": rules.source_yaml_sha256 == entry["source_yaml_sha256"],
        "tournament_engine_py_sha256": (hashlib.sha256((SCRIPTS / "tournament_engine.py").read_bytes()).hexdigest()
                                         == entry["tournament_engine_py_sha256"]),
    }
    return all(checks.values()), checks


# ---------------------------------------------------------------------------
# Interactive participant validation (Phase 9B deployment identity policy -
# amendment #10/#17: never fuzzy, never forced onto historical replay)
# ---------------------------------------------------------------------------

_STAGE_SIZES = {"stage1": 16, "stage2_direct": 8, "stage3_direct": 8}
_STAGE_TO_ENGINE_LABEL = {"stage1": te.STAGE_1, "stage2_direct": te.STAGE_2, "stage3_direct": te.STAGE_3}


def validate_tournament_participants(ruleset_id, participants, context_id):
    """participants: {"stage1": [{"team":.., "seed":..}, x16], "stage2_direct": [x8],
    "stage3_direct": [x8]}. Returns canonical structure + tournament_engine.TeamEntry
    lists ready for run_major_tournament."""
    get_tournament_ruleset(ruleset_id)  # 404 unknown_ruleset if not registered
    ctx = ai.get_context(context_id)    # 404 unknown_context via ai's own error path

    for key, expected_n in _STAGE_SIZES.items():
        group = participants.get(key)
        if group is None or len(group) != expected_n:
            _err("invalid_participant_count", f"{key!r} requires exactly {expected_n} entries, got "
                 f"{0 if group is None else len(group)}", stage=key, expected=expected_n,
                 got=None if group is None else len(group))

    canonical_by_stage = {}
    all_canonical = []
    for key in _STAGE_SIZES:
        seeds_seen = set()
        entries = []
        for row in participants[key]:
            team, seed = row["team"], row["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                _err("invalid_seed", f"{key}: seed must be an int, got {seed!r}", stage=key, team=team, seed=seed)
            if seed in seeds_seen:
                _err("invalid_seed", f"{key}: duplicate seed {seed}", stage=key, seed=seed)
            seeds_seen.add(seed)
            canonical = ai.resolve_team(team, ctx.identity_policy)  # unknown_team/ambiguous_team as appropriate
            entries.append({"team": team, "canonical_name": canonical, "seed": seed})
        expected_seeds = set(range(1, _STAGE_SIZES[key] + 1))
        if seeds_seen != expected_seeds:
            _err("missing_seed" if seeds_seen < expected_seeds else "invalid_seed",
                 f"{key}: seeds must be exactly {sorted(expected_seeds)}, got {sorted(seeds_seen)}",
                 stage=key, expected=sorted(expected_seeds), got=sorted(seeds_seen))
        entries.sort(key=lambda e: e["seed"])
        canonical_by_stage[key] = entries
        all_canonical.extend(e["canonical_name"] for e in entries)

    if len(set(all_canonical)) != len(all_canonical):
        seen, dupes = set(), set()
        for name in all_canonical:
            (dupes if name in seen else seen).add(name)
        _err("duplicate_team", f"the same canonical team appears more than once across stages: {sorted(dupes)}",
             duplicates=sorted(dupes))

    entrants = {}
    for key in _STAGE_SIZES:
        entrants[key] = [te.TeamEntry(team_id=e["canonical_name"], display_name=e["team"], initial_stage_seed=e["seed"])
                          for e in canonical_by_stage[key]]

    history_meta = {name: ai._team_history_metadata(ctx.rf_context.store, name) for name in all_canonical}
    return {
        "canonical_by_stage": canonical_by_stage,
        "entrants": (entrants["stage1"], entrants["stage2_direct"], entrants["stage3_direct"]),
        "all_canonical_teams": all_canonical,
        "history": history_meta,
    }


# ---------------------------------------------------------------------------
# Probability matrix construction + thread-safe bounded LRU cache
# (amendment #1: lock -> check -> unlock -> build -> lock -> re-check ->
# insert/evict -> unlock, with per-key single-flight so two concurrent
# first-requests for the SAME key don't both pay for 2,976 RF calls)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CachedMatrix:
    lookup: MappingProxyType             # {(team_a, team_b, best_of): p_a}, read-only
    matrix_hash: str
    context_id: str
    tier: str
    prediction_datetime: Optional[str]
    team_count: int
    built_at: str


_MATRIX_CACHE: "OrderedDict[tuple, _CachedMatrix]" = OrderedDict()
_MATRIX_CACHE_LOCK = threading.RLock()
_MATRIX_BUILD_LOCKS: dict = {}


def _canonical_matrix_content_hash(rows):
    """Hashes ONLY semantically relevant prediction content (team_a, team_b,
    best_of, probability_team_a/b, context/tier metadata) - never pandas row
    index/display ordering. Deterministic float repr via Python's own
    repr(float) (round-trippable, locale-independent)."""
    canon = sorted(
        {"team_a": r["team_a"], "team_b": r["team_b"], "best_of": r["best_of"],
         "probability_team_a": repr(r["probability_team_a"]), "probability_team_b": repr(r["probability_team_b"]),
         "model_id": r["model_id"], "context_id": r["context_id"], "tier": r["tier"],
         "prediction_datetime": r["prediction_datetime"]}.items()
        for r in rows
    )
    payload = [dict(row) for row in canon]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _matrix_cache_key(context_id, canonical_teams, tier, prediction_datetime):
    """Amendment #2: keyed on the 32-team SET (never seed/stage assignment -
    the matrix contains every ordered pair regardless of bracket seeding),
    context, tier, prediction_datetime, and the RF pipeline's own hash
    fingerprint (so a future RF re-freeze automatically invalidates stale
    cache entries)."""
    rf_fingerprint = tuple(sorted(ai._registry()["rf_unknown_map_pipeline"].items()))
    return (context_id, tier, prediction_datetime, frozenset(canonical_teams), rf_fingerprint)


def _build_matrix_uncached(context_id, canonical_teams, tier, prediction_datetime):
    rows = []
    for team_a in canonical_teams:
        for team_b in canonical_teams:
            if team_a == team_b:
                continue
            for best_of in (1, 3, 5):
                r = ai.predict_series_unknown_maps(context_id, team_a, team_b, best_of,
                                                    prediction_datetime=prediction_datetime, tier=tier)
                rows.append({"team_a": team_a, "team_b": team_b, "best_of": best_of,
                             "probability_team_a": r["probability_team_a"],
                             "probability_team_b": r["probability_team_b"],
                             "model_id": r["model_id"], "context_id": context_id, "tier": r["tier"],
                             "prediction_datetime": r["prediction_datetime"]})

    n = len(canonical_teams)
    expected_rows = n * (n - 1) * 3
    if len(rows) != expected_rows:
        _err("probability_matrix_incomplete", f"expected {expected_rows} rows for {n} teams, built {len(rows)}",
             expected=expected_rows, got=len(rows))
    keys_seen = set()
    lookup = {}
    for row in rows:
        key = (row["team_a"], row["team_b"], row["best_of"])
        if key in keys_seen:
            _err("probability_matrix_incomplete", f"duplicate matrix key {key}", key=list(key))
        keys_seen.add(key)
        p_a = row["probability_team_a"]
        if not isinstance(p_a, float) or not (0.0 <= p_a <= 1.0) or not np.isfinite(p_a):
            _err("probability_matrix_incomplete", f"non-finite or out-of-range probability for {key}: {p_a}", key=list(key))
        if abs(row["probability_team_b"] - (1.0 - p_a)) > 0.0:
            _err("probability_matrix_incomplete", f"complement violated for {key}", key=list(key))
        lookup[key] = p_a
    for a in canonical_teams:
        for b in canonical_teams:
            if a == b:
                continue
            for bo in (1, 3, 5):
                if (a, b, bo) not in lookup:
                    _err("probability_matrix_incomplete", f"missing matrix entry for ({a}, {b}, {bo})")

    matrix_hash = _canonical_matrix_content_hash(rows)
    effective_datetime = rows[0]["prediction_datetime"] if rows else prediction_datetime
    return _CachedMatrix(lookup=MappingProxyType(lookup), matrix_hash=matrix_hash, context_id=context_id, tier=tier,
                          prediction_datetime=effective_datetime, team_count=n,
                          built_at=pd.Timestamp.now("UTC").isoformat())


def build_tournament_probability_matrix(context_id, canonical_teams, tier, prediction_datetime):
    key = _matrix_cache_key(context_id, canonical_teams, tier, prediction_datetime)

    with _MATRIX_CACHE_LOCK:
        cached = _MATRIX_CACHE.get(key)
        if cached is not None:
            _MATRIX_CACHE.move_to_end(key)
            return cached
        build_lock = _MATRIX_BUILD_LOCKS.setdefault(key, threading.Lock())

    with build_lock:
        with _MATRIX_CACHE_LOCK:
            cached = _MATRIX_CACHE.get(key)
            if cached is not None:
                _MATRIX_CACHE.move_to_end(key)
                return cached

        matrix = _build_matrix_uncached(context_id, canonical_teams, tier, prediction_datetime)  # NOT holding the lock

        with _MATRIX_CACHE_LOCK:
            _MATRIX_CACHE[key] = matrix
            _MATRIX_CACHE.move_to_end(key)
            while len(_MATRIX_CACHE) > MAX_MATRIX_CACHE_ENTRIES:
                evicted_key, _ = _MATRIX_CACHE.popitem(last=False)
                _MATRIX_BUILD_LOCKS.pop(evicted_key, None)
        return matrix


def _matrix_cache_size():
    with _MATRIX_CACHE_LOCK:
        return len(_MATRIX_CACHE)


# ---------------------------------------------------------------------------
# Manual override validation (amendment #4/#5/#12/#24)
# ---------------------------------------------------------------------------

def _match_identity(match: te.MatchSpec):
    if match.stage == te.PLAYOFFS:
        return ("playoffs", PLAYOFF_ROUND_LABELS[match.round_number], None, frozenset({match.team_a, match.team_b}))
    return (match.stage, str(match.round_number), match.record_group, frozenset({match.team_a, match.team_b}))


def _override_identity(ov, identity_policy):
    """Resolves + validates one raw override dict. Raises invalid_override /
    unknown_team / override_team_mismatch. Does NOT check duplicates across
    the set - that is done separately over the whole validated list."""
    stage = ov.get("stage")
    team_1, team_2, winner = ov.get("team_1"), ov.get("team_2"), ov.get("winner")
    if not (team_1 and team_2 and winner):
        _err("invalid_override", "team_1, team_2, and winner are all required", override=ov)
    canon_1 = ai.resolve_team(team_1, identity_policy)
    canon_2 = ai.resolve_team(team_2, identity_policy)
    canon_winner = ai.resolve_team(winner, identity_policy)
    if canon_1 == canon_2:
        _err("invalid_override", "team_1 and team_2 must be different teams", override=ov)
    if canon_winner not in (canon_1, canon_2):
        _err("override_team_mismatch", f"winner {winner!r} is not one of the declared pair ({team_1}, {team_2})",
             override=ov)

    if stage in VALID_STAGES:
        round_number, record_group = ov.get("round_number"), ov.get("record_group")
        if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number not in (1, 2, 3, 4, 5):
            _err("invalid_override", f"{stage}: round_number must be an int 1..5, got {round_number!r}", override=ov)
        if not record_group or not isinstance(record_group, str):
            _err("invalid_override", f"{stage}: record_group is required (e.g. '1-0')", override=ov)
        identity = (stage, str(round_number), record_group, frozenset({canon_1, canon_2}))
    elif stage == "playoffs":
        playoff_round = ov.get("playoff_round")
        if playoff_round not in ("quarterfinal", "semifinal", "grand_final"):
            _err("invalid_override", f"playoffs: playoff_round must be one of quarterfinal/semifinal/grand_final, "
                 f"got {playoff_round!r}", override=ov)
        identity = ("playoffs", playoff_round, None, frozenset({canon_1, canon_2}))
    else:
        _err("invalid_override", f"stage must be one of {VALID_STAGES + ('playoffs',)}, got {stage!r}", override=ov)

    return {"identity": identity, "winner": canon_winner, "declared_best_of": ov.get("best_of"), "raw": ov}


def validate_manual_overrides(overrides, identity_policy):
    """Amendment #5: duplicate/contradictory overrides are rejected up
    front, never resolved by dictionary overwrite order. Amendment #24:
    bounded list length."""
    if len(overrides) > MAX_MANUAL_OVERRIDES:
        _err("invalid_override", f"at most {MAX_MANUAL_OVERRIDES} manual overrides are accepted, got "
             f"{len(overrides)}", max_allowed=MAX_MANUAL_OVERRIDES, got=len(overrides))
    validated = [_override_identity(ov, identity_policy) for ov in overrides]
    by_identity = {}
    for v in validated:
        ident = v["identity"]
        if ident in by_identity:
            prior = by_identity[ident]
            if prior["winner"] == v["winner"]:
                _err("duplicate_override", "the same matchup and winner was overridden more than once",
                     override=v["raw"])
            _err("contradictory_override", "the same matchup was overridden with two different winners",
                 override_a=prior["raw"], override_b=v["raw"])
        by_identity[ident] = v
    return validated


# ---------------------------------------------------------------------------
# Outcome providers (RF V2 pre-veto ONLY - static guard: this module must
# never import predict_map/predict_series_known_maps or XGB)
# ---------------------------------------------------------------------------

class _OverrideAwareFavoriteProvider(te.OutcomeProvider):
    """Deterministic favorite-wins path (p_a >= 0.5 -> team_a), with manual
    overrides taking precedence when the exact semantic matchup occurs.
    Deliberately `>=`, distinct from Phase 9B's display-only `favored_team=
    None` tie convention (see module docstring / report section F).

    provider_metadata is kept BYTE-IDENTICAL to Phase 8D's own
    FavoriteWinsProvider ({"model_id": ..., "prediction_mode": "pre_veto"} -
    no extra keys) so that a zero-override run's canonical_trace_hash can be
    compared directly against the frozen Phase 8D favorite-path hash
    (the hard parity gate). `selection_source` (model vs user) is derived
    separately at response-projection time from the override_index, never
    folded into the engine-level trace content."""

    def __init__(self, matrix_lookup, override_index, usage):
        self.matrix_lookup = matrix_lookup
        self.override_index = override_index
        self.usage = usage

    def resolve_match(self, match: te.MatchSpec) -> te.MatchResolution:
        key = (match.team_a, match.team_b, match.best_of)
        p_a = self.matrix_lookup[key]
        identity = _match_identity(match)
        if identity in self.override_index:
            self.usage[identity]["reached"] += 1
            self.usage[identity]["applied"] += 1
            winner = self.override_index[identity]
        else:
            winner = match.team_a if p_a >= 0.5 else match.team_b
        return te.MatchResolution(winner=winner, probability_team_a=p_a,
                                   provider_metadata={"model_id": "series_random_forest_v2",
                                                       "prediction_mode": "pre_veto"})


class _MonteCarloOverrideAwareProvider(te.OutcomeProvider):
    """Bernoulli sampling from the frozen matrix (rng.random() < p_a), with
    manual overrides forced when the exact semantic matchup occurs in this
    simulation. Never `p > 0.5 -> winner` - that would collapse Monte Carlo
    into repeated deterministic brackets."""

    def __init__(self, matrix_lookup, rng, override_index, usage):
        self.matrix_lookup = matrix_lookup
        self.rng = rng
        self.override_index = override_index
        self.usage = usage

    def resolve_match(self, match: te.MatchSpec) -> te.MatchResolution:
        key = (match.team_a, match.team_b, match.best_of)
        p_a = self.matrix_lookup[key]
        identity = _match_identity(match)
        if identity in self.override_index:
            self.usage[identity]["reached"] += 1
            self.usage[identity]["applied"] += 1
            winner = self.override_index[identity]
        else:
            winner = match.team_a if self.rng.random() < p_a else match.team_b
        return te.MatchResolution(winner=winner, probability_team_a=p_a,
                                   provider_metadata={"model_id": "series_random_forest_v2", "prediction_mode": "pre_veto"})


def _fresh_usage(validated_overrides):
    return {v["identity"]: {"reached": 0, "applied": 0} for v in validated_overrides}


def _override_index(validated_overrides):
    return {v["identity"]: v["winner"] for v in validated_overrides}


def _project_match(entry: "te.MatchTraceEntry", override_index=None):
    """override_index (identity -> winner) is consulted directly, rather
    than stashing selection_source inside provider_metadata, so the engine
    trace content stays byte-identical to Phase 8D's own zero-override
    provider (required for the historical favorite-path hash parity gate)."""
    m, r = entry.match, entry.resolution
    identity = _match_identity(m)
    selection_source = "user" if (override_index and identity in override_index) else "model"
    return {"match_id": m.match_id, "stage": m.stage, "round_number": m.round_number,
            "record_group": m.record_group, "team_a": m.team_a, "team_b": m.team_b, "best_of": m.best_of,
            "probability_team_a": r.probability_team_a, "winner": r.winner, "loser": r.loser,
            "selection_source": selection_source}


def _override_usage_report(validated_overrides, usage, total_simulations=1):
    per_override = []
    for v in validated_overrides:
        u = usage[v["identity"]]
        reached, applied = u["reached"], u["applied"]
        per_override.append({
            "stage": v["raw"].get("stage"), "team_1": v["raw"].get("team_1"), "team_2": v["raw"].get("team_2"),
            "winner": v["raw"].get("winner"), "simulations_matchup_reached": reached,
            "simulations_override_applied": applied, "simulations_not_reached": total_simulations - reached,
            "application_rate": applied / total_simulations if total_simulations else None,
            "conditional_application_rate": (applied / reached) if reached else None,
        })
    return {
        "overrides_supplied": len(validated_overrides),
        "overrides_used": sum(1 for p in per_override if p["simulations_override_applied"] > 0),
        "overrides_not_reached": sum(1 for p in per_override if p["simulations_matchup_reached"] == 0),
        "invalid_overrides": [],
        "per_override": per_override,
    }


def _run_deterministic_path(matrix_lookup, rules, entrants, validated_overrides):
    stage1, stage2_direct, stage3_direct = entrants
    override_index, usage = _override_index(validated_overrides), _fresh_usage(validated_overrides)
    provider = _OverrideAwareFavoriteProvider(matrix_lookup, override_index, usage)
    result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
    stages = {}
    for label, stage_result in (("stage_1", result.stage1), ("stage_2", result.stage2), ("stage_3", result.stage3)):
        stages[label] = {"stage": label, "matches": [_project_match(e, override_index) for e in stage_result.trace]}
    playoffs = {"matches": [_project_match(e, override_index) for e in result.playoffs.trace],
                "champion": result.playoffs.champion, "runner_up": result.playoffs.runner_up}
    return {
        "stage_1": stages["stage_1"], "stage_2": stages["stage_2"], "stage_3": stages["stage_3"],
        "playoffs": playoffs, "champion": result.champion,
        "canonical_trace_hash": te.trace_hash(result.to_dict()),
        "override_usage": _override_usage_report(validated_overrides, usage, total_simulations=1),
    }


def predict_tournament_path(ruleset_id, context_id, tier, prediction_datetime, participants, manual_overrides=None):
    manual_overrides = manual_overrides or []
    ctx = ai.get_context(context_id)
    canon = validate_tournament_participants(ruleset_id, participants, context_id)
    tier_value, _ = ai.validate_tier(tier)
    validated_overrides = validate_manual_overrides(manual_overrides, ctx.identity_policy)

    matrix = build_tournament_probability_matrix(context_id, canon["all_canonical_teams"], tier_value,
                                                  prediction_datetime)
    rules = te.load_frozen_rules()
    path = _run_deterministic_path(matrix.lookup, rules, canon["entrants"], validated_overrides)

    return {
        "ruleset_id": ruleset_id, "context_id": context_id, "tier": tier_value,
        "prediction_datetime": matrix.prediction_datetime,
        "canonical_participants": canon["canonical_by_stage"], "probability_matrix_hash": matrix.matrix_hash,
        **path,
    }


# ---------------------------------------------------------------------------
# Monte Carlo (amendment #6/#7/#8/#9/#20)
# ---------------------------------------------------------------------------

_PROCESS_POOL = None
_PROCESS_POOL_LOCK = threading.Lock()


def _get_process_pool():
    global _PROCESS_POOL
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is None:
            _PROCESS_POOL = ProcessPoolExecutor(max_workers=PROCESS_POOL_MAX_WORKERS)
        return _PROCESS_POOL


def shutdown_process_pool():
    """Owned by the FastAPI lifespan shutdown phase (amendment #6) - never
    leaves orphan worker processes after application shutdown/tests."""
    global _PROCESS_POOL
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is not None:
            _PROCESS_POOL.shutdown(wait=True, cancel_futures=False)
            _PROCESS_POOL = None


_SWISS_STAGE_LABELS = (te.STAGE_1, te.STAGE_2, te.STAGE_3)
_ALL_RECORDS = ("3-0", "3-1", "3-2", "2-3", "1-3", "0-3")


def _empty_team_bucket():
    return {
        "participate_stage_1": 0, "participate_stage_2": 0, "participate_stage_3": 0,
        "reach_playoffs": 0, "reach_semifinal": 0, "reach_final": 0, "win_tournament": 0,
        "stage_1_participations": 0, "stage_1_advances": 0, "stage_2_participations": 0, "stage_2_advances": 0,
        "stage_3_participations": 0, "stage_3_advances": 0,
        "swiss_records": {s: {r: 0 for r in _ALL_RECORDS} for s in _SWISS_STAGE_LABELS},
        "playoff_seed_counts": {str(i): 0 for i in range(1, 9)},
    }


def _run_monte_carlo_batch(payload):
    """Module-level, picklable, PURE-DATA worker (amendment #7): consumes
    only plain participant/rules/matrix/override/seed data - never imports
    or calls application_inference, never loads a model/state artifact.
    te.load_frozen_rules() is the only "loader" used here, and it reads
    ONLY the frozen tournament YAML's structure keys (hash-gated), never
    `participants`, never ML/state files - see tournament_engine.py's own
    module docstring/AST-checked import guard (Phase 8C)."""
    entrants_raw, matrix_lookup, override_payload, base_seed, start_index, count, all_team_ids = (
        payload["entrants"], payload["matrix_lookup"], payload["overrides"], payload["base_seed"],
        payload["start_index"], payload["count"], payload["all_team_ids"])

    stage1 = [te.TeamEntry(**e) for e in entrants_raw["stage1"]]
    stage2_direct = [te.TeamEntry(**e) for e in entrants_raw["stage2_direct"]]
    stage3_direct = [te.TeamEntry(**e) for e in entrants_raw["stage3_direct"]]
    rules = te.load_frozen_rules()

    override_index = {ov["identity"]: ov["winner"] for ov in override_payload}
    usage = {ov["identity"]: {"reached": 0, "applied": 0} for ov in override_payload}

    team = {tid: _empty_team_bucket() for tid in all_team_ids}
    champion_counts = {}
    n_done = 0

    for sim_index in range(start_index, start_index + count):
        seed_seq = np.random.SeedSequence([base_seed, sim_index])
        rng = np.random.default_rng(seed_seq)
        provider = _MonteCarloOverrideAwareProvider(matrix_lookup, rng, override_index, usage)
        result = te.run_major_tournament(stage1, stage2_direct, stage3_direct, rules, provider)
        n_done += 1

        for stage_result in (result.stage1, result.stage2, result.stage3):
            stage = stage_result.stage
            advancer_ids = {t.team_id for t in stage_result.advancers}
            for e in stage_result.entrants:
                tid = e.team_id
                t = team[tid]
                t[f"participate_{stage}"] += 1
                t[f"{stage}_participations"] += 1
                if tid in advancer_ids:
                    t[f"{stage}_advances"] += 1
                final_state = next(s for s in stage_result.final_order if s.team_id == tid)
                record_key = f"{final_state.wins}-{final_state.losses}"
                t["swiss_records"][stage][record_key] += 1

        for i, e in enumerate(result.stage3.advancers, start=1):
            t = team[e.team_id]
            t["reach_playoffs"] += 1
            t["playoff_seed_counts"][str(i)] += 1

        for entry in result.playoffs.trace:
            m = entry.match
            if m.round_number == 2:
                for tid in (m.team_a, m.team_b):
                    team[tid]["reach_semifinal"] += 1
            if m.round_number == 3:
                for tid in (m.team_a, m.team_b):
                    team[tid]["reach_final"] += 1

        champion_counts[result.champion] = champion_counts.get(result.champion, 0) + 1
        team[result.champion]["win_tournament"] += 1

    return {"n_simulations": n_done, "champion_counts": champion_counts, "team": team, "override_usage": usage}


def _merge_int_dict(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _merge_partial_aggregates(partials):
    n_simulations = sum(p["n_simulations"] for p in partials)
    champion_counts = {}
    for p in partials:
        champion_counts = _merge_int_dict(champion_counts, p["champion_counts"])
    all_team_ids = partials[0]["team"].keys()
    team = {}
    for tid in all_team_ids:
        merged = _empty_team_bucket()
        for p in partials:
            src = p["team"][tid]
            for k in ("participate_stage_1", "participate_stage_2", "participate_stage_3", "reach_playoffs",
                      "reach_semifinal", "reach_final", "win_tournament", "stage_1_participations",
                      "stage_1_advances", "stage_2_participations", "stage_2_advances", "stage_3_participations",
                      "stage_3_advances"):
                merged[k] += src[k]
            for stage in _SWISS_STAGE_LABELS:
                for rec in _ALL_RECORDS:
                    merged["swiss_records"][stage][rec] += src["swiss_records"][stage][rec]
            for seed in map(str, range(1, 9)):
                merged["playoff_seed_counts"][seed] += src["playoff_seed_counts"][seed]
        team[tid] = merged

    override_usage = {}
    for p in partials:
        for identity, u in p["override_usage"].items():
            bucket = override_usage.setdefault(identity, {"reached": 0, "applied": 0})
            bucket["reached"] += u["reached"]
            bucket["applied"] += u["applied"]

    return {"n_simulations": n_simulations, "champion_counts": champion_counts, "team": team,
            "override_usage": override_usage}


def _mc_stat(numerator, denominator):
    if denominator == 0:
        return {"numerator_count": 0, "denominator_count": 0, "probability": None, "mc_standard_error": None}
    p = numerator / denominator
    se = (p * (1.0 - p) / denominator) ** 0.5
    return {"numerator_count": numerator, "denominator_count": denominator, "probability": p, "mc_standard_error": se}


def _verify_aggregate_conservation(agg, all_canonical_teams):
    """Amendment #19: corruption detector for parallel aggregate merging."""
    n = agg["n_simulations"]
    checks = {
        "champion_sum_equals_n": sum(agg["champion_counts"].values()) == n,
        "playoff_sum_equals_8n": sum(t["reach_playoffs"] for t in agg["team"].values()) == 8 * n,
        "semifinal_sum_equals_4n": sum(t["reach_semifinal"] for t in agg["team"].values()) == 4 * n,
        "final_sum_equals_2n": sum(t["reach_final"] for t in agg["team"].values()) == 2 * n,
    }
    for stage in _SWISS_STAGE_LABELS:
        advancing = sum(t[f"{stage}_advances"] for t in agg["team"].values())
        participating = sum(t[f"{stage}_participations"] for t in agg["team"].values())
        record_total = sum(sum(t["swiss_records"][stage].values()) for t in agg["team"].values())
        checks[f"{stage}_advancing_equals_8n"] = advancing == 8 * n
        checks[f"{stage}_eliminated_equals_8n"] = (participating - advancing) == 8 * n
        checks[f"{stage}_terminal_record_total_equals_16n"] = record_total == 16 * n
    if not all(checks.values()):
        _err("missing_state_support", f"Monte Carlo aggregate conservation check failed: {checks}", checks=checks)
    return checks


def simulate_tournament(ruleset_id, context_id, tier, prediction_datetime, participants, simulation_count,
                         seed=42, manual_overrides=None):
    manual_overrides = manual_overrides or []
    if not isinstance(simulation_count, int) or isinstance(simulation_count, bool) or not (1 <= simulation_count <= MAX_SIMULATION_COUNT):
        _err("invalid_simulation_count", f"simulation_count must be an int in [1, {MAX_SIMULATION_COUNT}], got "
             f"{simulation_count!r}", requested=simulation_count, max_allowed=MAX_SIMULATION_COUNT)
    if not isinstance(seed, int) or isinstance(seed, bool):
        _err("invalid_simulation_count", f"seed must be an int, got {seed!r}", requested_seed=seed)

    ctx = ai.get_context(context_id)
    canon = validate_tournament_participants(ruleset_id, participants, context_id)
    tier_value, _ = ai.validate_tier(tier)
    validated_overrides = validate_manual_overrides(manual_overrides, ctx.identity_policy)

    matrix = build_tournament_probability_matrix(context_id, canon["all_canonical_teams"], tier_value,
                                                  prediction_datetime)
    entrants_raw = {
        "stage1": [e.to_dict() for e in canon["entrants"][0]],
        "stage2_direct": [e.to_dict() for e in canon["entrants"][1]],
        "stage3_direct": [e.to_dict() for e in canon["entrants"][2]],
    }
    override_payload = [{"identity": v["identity"], "winner": v["winner"]} for v in validated_overrides]

    n_chunks = 1 if simulation_count < PROCESS_POOL_MIN_SIMULATIONS else PROCESS_POOL_MAX_WORKERS
    bounds = np.linspace(0, simulation_count, n_chunks + 1, dtype=int)
    chunks = [(int(bounds[i]), int(bounds[i + 1] - bounds[i])) for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    start = time.perf_counter()
    matrix_lookup_plain = dict(matrix.lookup)
    payloads = [{"entrants": entrants_raw, "matrix_lookup": matrix_lookup_plain, "overrides": override_payload,
                 "base_seed": seed, "start_index": s, "count": c, "all_team_ids": canon["all_canonical_teams"]}
                for s, c in chunks]

    if n_chunks == 1:
        execution_mode = "synchronous"
        partials = [_run_monte_carlo_batch(payloads[0])]
    else:
        execution_mode = "process_pool"
        pool = _get_process_pool()
        futures = [pool.submit(_run_monte_carlo_batch, p) for p in payloads]
        try:
            partials = [f.result() for f in futures]
        except Exception:
            for f in futures:
                f.cancel()
            raise  # amendment #20: all-or-nothing, no partial aggregate is ever returned or cached

    elapsed_s = time.perf_counter() - start
    agg = _merge_partial_aggregates(partials)
    _verify_aggregate_conservation(agg, canon["all_canonical_teams"])

    n = agg["n_simulations"]
    champion_ranking = sorted(
        ({"team": tid, **_mc_stat(count, n)} for tid, count in agg["champion_counts"].items()),
        key=lambda r: (-(r["probability"] or 0), r["team"]))

    teams_out = []
    for tid, t in agg["team"].items():
        row = {"team": tid, "participate_stage_1": _mc_stat(t["participate_stage_1"], n),
               "participate_stage_2": _mc_stat(t["participate_stage_2"], n),
               "participate_stage_3": _mc_stat(t["participate_stage_3"], n),
               "advance_from_stage_1": _mc_stat(t["stage_1_advances"], t["stage_1_participations"]),
               "advance_from_stage_2": _mc_stat(t["stage_2_advances"], t["stage_2_participations"]),
               "advance_from_stage_3": _mc_stat(t["stage_3_advances"], t["stage_3_participations"]),
               "reach_playoffs": _mc_stat(t["reach_playoffs"], n),
               "reach_semifinal": _mc_stat(t["reach_semifinal"], n),
               "reach_final": _mc_stat(t["reach_final"], n),
               "win_tournament": _mc_stat(t["win_tournament"], n),
               "swiss_record_distribution": {
                   stage: {rec: _mc_stat(t["swiss_records"][stage][rec], t[f"{stage}_participations"])
                           for rec in _ALL_RECORDS} for stage in _SWISS_STAGE_LABELS},
               "playoff_seed_distribution": {seed_str: _mc_stat(t["playoff_seed_counts"][seed_str], t["reach_playoffs"])
                                              for seed_str in map(str, range(1, 9))}}
        teams_out.append(row)
    teams_out.sort(key=lambda r: r["team"])

    override_report = _override_usage_report(validated_overrides, agg["override_usage"], total_simulations=n)

    return {
        "ruleset_id": ruleset_id, "context_id": context_id, "tier": tier_value,
        "prediction_datetime": matrix.prediction_datetime,
        "canonical_participants": canon["canonical_by_stage"], "probability_matrix_hash": matrix.matrix_hash,
        "monte_carlo": {"simulation_count": simulation_count, "seed": seed, "execution_mode": execution_mode,
                         "n_chunks": len(chunks), "elapsed_seconds": elapsed_s,
                         "simulation_conditioned_on_manual_overrides": len(validated_overrides) > 0},
        "champion_ranking": champion_ranking, "teams": teams_out, "override_usage": override_report,
    }


# ---------------------------------------------------------------------------
# Historical Cologne - pure file-backed views (amendment #13/#14/#16/#17)
# ---------------------------------------------------------------------------

_HISTORICAL_CACHE_LOCK = threading.Lock()
_HISTORICAL_CACHE = {}

_PHASE8D_RECEIPT = EVAL / "cologne_2026_pre_event_simulation_receipt_v1.json"
_PHASE8D_MATRIX = EVAL / "cologne_2026_pre_event_matchup_probabilities_v1.parquet"
_PHASE8D_SUMMARY = EVAL / "cologne_2026_pre_event_simulation_summary_v1.json"
_PHASE8D_TEAM_PROBS = EVAL / "cologne_2026_pre_event_team_probabilities_v1.csv"
_PHASE8D_SWISS_DIST = EVAL / "cologne_2026_pre_event_swiss_record_distributions_v1.csv"
_PHASE8D_PLAYOFF_DIST = EVAL / "cologne_2026_pre_event_playoff_seed_distributions_v1.csv"
_PHASE8D_FAVORITE_PATH = EVAL / "cologne_2026_pre_event_favorite_path_v1.json"
_PHASE8D_SAMPLE_TRACES = EVAL / "cologne_2026_pre_event_sample_traces_v1.json"

_PHASE8E_RECEIPT = EVAL / "cologne_2026_simulation_vs_reality_receipt_v1.json"
_PHASE8E_SUMMARY = EVAL / "cologne_2026_simulation_vs_reality_summary_v1.json"
_PHASE8E_METRICS_DETAIL = EVAL / "cologne_2026_phase8e_metrics_detail_v1.json"
_PHASE8E_RECONCILIATION = EVAL / "cologne_2026_result_reconciliation_v1.csv"


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _df_records_json_safe(df):
    """pandas round-trips an empty CSV cell (Phase 8D writes None for a
    zero-denominator mc_stat) back as NaN, which is not JSON-compliant -
    normalize to None without altering any other value."""
    return df.astype(object).where(pd.notnull(df), None).to_dict("records")


def verify_historical_cologne_contract():
    """Amendment #13: verifies the frozen receipts' referenced artifact
    hashes TRANSITIVELY for every artifact actually served - never just the
    receipt JSON itself. Cheap (file hashing only) - no RF, no matrix
    build, no engine run, no Monte Carlo."""
    detail = {}
    ok = True
    try:
        d_receipt = json.loads(_PHASE8D_RECEIPT.read_text(encoding="utf-8"))
        d_hashes = d_receipt["hashes"]
        d_checks = {
            "matrix": _sha256_file(_PHASE8D_MATRIX) == d_hashes["cologne_2026_pre_event_matchup_probabilities_v1.parquet"],
            "summary": _sha256_file(_PHASE8D_SUMMARY) == d_hashes["cologne_2026_pre_event_simulation_summary_v1.json"],
            "team_probabilities": _sha256_file(_PHASE8D_TEAM_PROBS) == d_hashes["cologne_2026_pre_event_team_probabilities_v1.csv"],
            "swiss_distributions": _sha256_file(_PHASE8D_SWISS_DIST) == d_hashes["cologne_2026_pre_event_swiss_record_distributions_v1.csv"],
            "playoff_distributions": _sha256_file(_PHASE8D_PLAYOFF_DIST) == d_hashes["cologne_2026_pre_event_playoff_seed_distributions_v1.csv"],
            "favorite_path": _sha256_file(_PHASE8D_FAVORITE_PATH) == d_hashes["cologne_2026_pre_event_favorite_path_v1.json"],
            "sample_traces": _sha256_file(_PHASE8D_SAMPLE_TRACES) == d_hashes["cologne_2026_pre_event_sample_traces_v1.json"],
            "tournament_yaml": _sha256_file(CONFIG / "tournaments" / "iem_cologne_major_2026_pre_event.yaml") == d_hashes["tournament_yaml"],
            "tournament_engine": _sha256_file(SCRIPTS / "tournament_engine.py") == d_hashes["tournament_engine"],
            "receipt_committed": d_receipt.get("created_before_results_opened") is True,
        }
        detail["phase8d"] = d_checks
        ok = ok and all(d_checks.values())

        e_receipt = json.loads(_PHASE8E_RECEIPT.read_text(encoding="utf-8"))
        e_hashes = e_receipt["hashes"]["evaluation_artifacts"]
        e_checks = {
            "summary": _sha256_file(_PHASE8E_SUMMARY) == e_hashes["simulation_vs_reality_summary"],
            "metrics_detail": _sha256_file(_PHASE8E_METRICS_DETAIL) == e_hashes["metrics_detail"],
            "reconciliation": _sha256_file(_PHASE8E_RECONCILIATION) == e_hashes["reconciliation_table"],
            "receipt_committed": e_receipt.get("committed") is True,
        }
        detail["phase8e"] = e_checks
        ok = ok and all(e_checks.values())
    except Exception as e:  # noqa: BLE001
        ok = False
        detail["exception"] = f"{type(e).__name__}: {e}"
    return ok, detail


def get_historical_cologne_pre_event():
    with _HISTORICAL_CACHE_LOCK:
        cached = _HISTORICAL_CACHE.get("pre_event")
        if cached is not None:
            return cached

    ok, detail = verify_historical_cologne_contract()
    if not ok:
        _err("missing_state_support", "historical Cologne contract verification failed", detail=detail)

    teams = phase8d_common.load_cologne_teams()
    summary = json.loads(_PHASE8D_SUMMARY.read_text(encoding="utf-8"))
    favorite_path = json.loads(_PHASE8D_FAVORITE_PATH.read_text(encoding="utf-8"))
    team_probs = pd.read_csv(_PHASE8D_TEAM_PROBS)
    swiss_dist = pd.read_csv(_PHASE8D_SWISS_DIST)
    playoff_dist = pd.read_csv(_PHASE8D_PLAYOFF_DIST)
    receipt = json.loads(_PHASE8D_RECEIPT.read_text(encoding="utf-8"))

    n = summary["n_simulations"]
    champion_counts = summary["champion_counts"]
    favorite = max(champion_counts, key=champion_counts.get)
    championship_probabilities = sorted(
        [{"team": t, "count": c, "probability": c / n} for t, c in champion_counts.items()],
        key=lambda r: (-r["probability"], r["team"]))

    matrix_file_hash = _sha256_file(_PHASE8D_MATRIX)
    matrix_df = pd.read_parquet(_PHASE8D_MATRIX, engine="fastparquet")
    content_rows = [{"team_a": r.team_a, "team_b": r.team_b, "best_of": int(r.best_of),
                      "probability_team_a": float(r.probability_team_a), "probability_team_b": float(r.probability_team_b),
                      "model_id": r.model_id, "context_id": "historical_cologne_pre_event", "tier": r.tier,
                      "prediction_datetime": r.prediction_datetime}
                     for r in matrix_df.itertuples(index=False)]
    application_matrix_content_hash = _canonical_matrix_content_hash(content_rows)

    result = {
        "event_id": "iem_cologne_major_2026", "ruleset_id": DEFAULT_RULESET_ID,
        "prediction_cutoff": receipt["prediction_cutoff"], "tier": receipt["tier"],
        "n_simulations": n, "base_seed": receipt["base_seed"],
        "participants": teams,
        "favorite": favorite, "favorite_championship_probability": champion_counts[favorite] / n,
        "championship_probabilities": championship_probabilities,
        "stage_advancement_probabilities": _df_records_json_safe(
            team_probs[team_probs["metric"].str.startswith("advance_from_")]),
        "playoff_qualification_probabilities": _df_records_json_safe(team_probs[team_probs["metric"] == "reach_playoffs"]),
        "playoff_seed_distributions": _df_records_json_safe(playoff_dist),
        "swiss_record_distributions": _df_records_json_safe(swiss_dist),
        "favorite_wins_path": favorite_path,
        "artifact_file_sha256": matrix_file_hash,
        "application_matrix_content_hash": application_matrix_content_hash,
        "matrix_hash_semantics_note": (
            "artifact_file_sha256 is the frozen Phase 8D receipt's own recorded file-level SHA-256 "
            "of cologne_2026_pre_event_matchup_probabilities_v1.parquet (the authoritative frozen "
            "value). application_matrix_content_hash is a SEPARATELY DERIVED canonical-content hash "
            "(same hashing function Phase 9E uses for freshly-built interactive matrices) computed "
            "over that same frozen data at serve time - useful for like-for-like comparison, but it "
            "is not itself a value Phase 8D ever recorded."
        ),
        "historical": True, "immutable": True,
    }
    with _HISTORICAL_CACHE_LOCK:
        _HISTORICAL_CACHE["pre_event"] = result
    return result


def get_historical_cologne_results():
    with _HISTORICAL_CACHE_LOCK:
        cached = _HISTORICAL_CACHE.get("results")
        if cached is not None:
            return cached

    ok, detail = verify_historical_cologne_contract()
    if not ok:
        _err("missing_state_support", "historical Cologne contract verification failed", detail=detail)

    summary = json.loads(_PHASE8E_SUMMARY.read_text(encoding="utf-8"))
    reconciliation = pd.read_csv(_PHASE8E_RECONCILIATION)
    excluded = reconciliation[~reconciliation["included_in_official_event"]]
    excluded_rows = excluded[["team1", "team2", "datetime", "reconciliation_status"]].to_dict("records")

    result = {
        "event_id": "iem_cologne_major_2026", "historical": True, "immutable": True,
        **summary,
        "original_cologne_tagged_rows": int(len(reconciliation)),
        "official_major_matches": int(reconciliation["included_in_official_event"].sum()),
        "excluded_non_tournament_rows": int((~reconciliation["included_in_official_event"]).sum()),
        "excluded_rows_detail": excluded_rows,
    }
    with _HISTORICAL_CACHE_LOCK:
        _HISTORICAL_CACHE["results"] = result
    return result
