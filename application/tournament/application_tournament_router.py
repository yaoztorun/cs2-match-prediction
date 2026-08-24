"""
Phase 9E HTTP transport for the Major tournament service. Thin - all
business logic lives in application_tournament_service.py. Reuses Phase 9D's
request-ID middleware, structured error envelope, and readiness pattern
(application_api.py) rather than duplicating any of it.
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

import application.tournament.application_tournament_service as ats

router = APIRouter(prefix="/api/v1/major", tags=["major"])

DEFAULT_RULESET_ID = ats.DEFAULT_RULESET_ID


# ---------------------------------------------------------------------------
# Typed request/response contracts (amendment #17): the new /major/path and
# /major/simulate contracts are NOT untyped Dict[str, Any] blobs. Deep
# historical artifact payloads (already-versioned Phase 8D/8E schemas) stay
# as controlled dict passthrough - re-declaring them field-by-field would be
# pure duplication risk for no benefit.
# ---------------------------------------------------------------------------

class TournamentParticipant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team: str = Field(min_length=1)
    seed: StrictInt


class TournamentParticipants(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage1: List[TournamentParticipant] = Field(min_length=16, max_length=16)
    stage2_direct: List[TournamentParticipant] = Field(min_length=8, max_length=8)
    stage3_direct: List[TournamentParticipant] = Field(min_length=8, max_length=8)


class ManualOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["stage_1", "stage_2", "stage_3", "playoffs"]
    round_number: Optional[StrictInt] = Field(
        default=None, description="Required for stage_1/2/3 (1..5). Must be null for playoffs.")
    record_group: Optional[str] = Field(
        default=None, description="Required for stage_1/2/3 (e.g. '1-0'), part of the Swiss semantic "
                                   "matchup identity so overrides survive rebracketing. Must be null for playoffs.")
    playoff_round: Optional[Literal["quarterfinal", "semifinal", "grand_final"]] = Field(
        default=None, description="Required for playoffs. Must be null for stage_1/2/3.")
    team_1: str = Field(min_length=1)
    team_2: str = Field(min_length=1)
    winner: str = Field(min_length=1, description="Must be exactly team_1 or team_2.")
    best_of: Optional[StrictInt] = Field(
        default=None, description="Optional invariant check only - not part of matchup identity matching.")


class MajorPathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ruleset_id: str = Field(default=DEFAULT_RULESET_ID, min_length=1)
    context_id: str = Field(default="deployment_post_cologne_v1", min_length=1)
    tier: Optional[Literal["tier1", "tier2", "tier3"]] = None
    prediction_datetime: Optional[str] = None
    participants: TournamentParticipants
    manual_overrides: List[ManualOverride] = Field(default_factory=list)


class MajorSimulateRequest(MajorPathRequest):
    simulation_count: StrictInt = Field(description=f"1 <= simulation_count <= {ats.MAX_SIMULATION_COUNT}.")
    seed: StrictInt = 42


class MatchOutcome(BaseModel):
    match_id: str
    stage: str
    round_number: int
    record_group: Optional[str]
    team_a: str
    team_b: str
    best_of: int
    probability_team_a: Optional[float]
    winner: str
    loser: str
    selection_source: Literal["model", "user"]


class StageMatches(BaseModel):
    stage: str
    matches: List[MatchOutcome]


class PlayoffMatches(BaseModel):
    matches: List[MatchOutcome]
    champion: str
    runner_up: str


class OverrideDiagnostic(BaseModel):
    stage: Optional[str]
    team_1: Optional[str]
    team_2: Optional[str]
    winner: Optional[str]
    simulations_matchup_reached: int
    simulations_override_applied: int
    simulations_not_reached: int
    application_rate: Optional[float]
    conditional_application_rate: Optional[float]


class OverrideUsageReport(BaseModel):
    overrides_supplied: int
    overrides_used: int
    overrides_not_reached: int
    invalid_overrides: List[Dict[str, Any]] = Field(default_factory=list)
    per_override: List[OverrideDiagnostic] = Field(default_factory=list)


class TournamentResponseMetadata(BaseModel):
    ruleset_id: str
    context_id: str
    prediction_contract: str = "phase9b"
    probability_matrix_hash: str
    state_data_through: str
    state_is_live: bool = False
    historical: bool = False
    immutable: bool = False


class TournamentPathResult(BaseModel):
    ruleset_id: str
    context_id: str
    tier: str
    prediction_datetime: str
    canonical_participants: Dict[str, Any]
    probability_matrix_hash: str
    stage_1: StageMatches
    stage_2: StageMatches
    stage_3: StageMatches
    playoffs: PlayoffMatches
    champion: str
    canonical_trace_hash: str
    override_usage: OverrideUsageReport


class TournamentPathResponse(BaseModel):
    request_id: str
    api_version: str
    result: TournamentPathResult
    metadata: TournamentResponseMetadata


class ProbabilityStat(BaseModel):
    numerator_count: int
    denominator_count: int
    probability: Optional[float]
    mc_standard_error: Optional[float]


class ChampionshipProbability(ProbabilityStat):
    team: str


class TeamAggregate(BaseModel):
    team: str
    participate_stage_1: ProbabilityStat
    participate_stage_2: ProbabilityStat
    participate_stage_3: ProbabilityStat
    advance_from_stage_1: ProbabilityStat
    advance_from_stage_2: ProbabilityStat
    advance_from_stage_3: ProbabilityStat
    reach_playoffs: ProbabilityStat
    reach_semifinal: ProbabilityStat
    reach_final: ProbabilityStat
    win_tournament: ProbabilityStat
    swiss_record_distribution: Dict[str, Dict[str, ProbabilityStat]]
    playoff_seed_distribution: Dict[str, ProbabilityStat]


class MonteCarloMetadata(BaseModel):
    simulation_count: int
    seed: int
    execution_mode: Literal["synchronous", "process_pool"]
    n_chunks: int
    elapsed_seconds: float
    simulation_conditioned_on_manual_overrides: bool


class TournamentSimulationResult(BaseModel):
    ruleset_id: str
    context_id: str
    tier: str
    prediction_datetime: str
    canonical_participants: Dict[str, Any]
    probability_matrix_hash: str
    monte_carlo: MonteCarloMetadata
    champion_ranking: List[ChampionshipProbability]
    teams: List[TeamAggregate]
    override_usage: OverrideUsageReport


class TournamentSimulationResponse(BaseModel):
    request_id: str
    api_version: str
    result: TournamentSimulationResult
    metadata: TournamentResponseMetadata


# ---------------------------------------------------------------------------
# Readiness helper (reuses application_api's shared _STARTUP_STATE contract)
# ---------------------------------------------------------------------------

def _require_subsystem_ready(subsystem):
    import application.api.application_api as api
    if not api._STARTUP_STATE["subsystems"].get(subsystem, False):
        raise api.ServiceUnavailableError({"subsystem": subsystem,
                                            "detail": api._STARTUP_STATE["subsystem_detail"].get(subsystem)})


def _api_version():
    import application.api.application_api as api
    return api.API_VERSION


def _to_champion_ranking(rows):
    return [{"team": r["team"], "numerator_count": r["numerator_count"], "denominator_count": r["denominator_count"],
             "probability": r["probability"], "mc_standard_error": r["mc_standard_error"]} for r in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/rulesets", summary="List registered tournament rulesets (application/version metadata)")
def get_rulesets(request: Request):
    _require_subsystem_ready("tournament_engine_ready")
    return {"request_id": request.state.request_id, "rulesets": ats.list_tournament_rulesets()}


@router.get(
    "/historical/cologne-2026",
    summary="Frozen Phase 8D pre-event Cologne simulation - file-backed view, never re-simulated",
)
def get_historical_pre_event(request: Request):
    _require_subsystem_ready("historical_cologne_ready")
    result = ats.get_historical_cologne_pre_event()
    return {"request_id": request.state.request_id, "api_version": _api_version(), "result": result}


@router.get(
    "/historical/cologne-2026/results",
    summary="Frozen Phase 8E simulation-vs-reality evaluation - file-backed view, never recomputed",
)
def get_historical_results(request: Request):
    _require_subsystem_ready("historical_cologne_ready")
    result = ats.get_historical_cologne_results()
    return {"request_id": request.state.request_id, "api_version": _api_version(), "result": result}


def _participants_to_plain(participants: TournamentParticipants):
    return {
        "stage1": [{"team": p.team, "seed": p.seed} for p in participants.stage1],
        "stage2_direct": [{"team": p.team, "seed": p.seed} for p in participants.stage2_direct],
        "stage3_direct": [{"team": p.team, "seed": p.seed} for p in participants.stage3_direct],
    }


def _overrides_to_plain(overrides: List[ManualOverride]):
    return [{"stage": o.stage, "round_number": o.round_number, "record_group": o.record_group,
              "playoff_round": o.playoff_round, "team_1": o.team_1, "team_2": o.team_2, "winner": o.winner,
              "best_of": o.best_of} for o in overrides]


@router.post(
    "/path",
    response_model=TournamentPathResponse,
    summary="Deterministic model-favorite tournament path with optional manual (Pick'Em) overrides",
)
def post_path(payload: MajorPathRequest, request: Request):
    _require_subsystem_ready("prediction_ready")
    _require_subsystem_ready("tournament_engine_ready")
    request.state.context_id = payload.context_id
    request.state.mode = "major_path"

    result = ats.predict_tournament_path(
        payload.ruleset_id, payload.context_id, payload.tier, payload.prediction_datetime,
        _participants_to_plain(payload.participants), manual_overrides=_overrides_to_plain(payload.manual_overrides))

    metadata = {"ruleset_id": payload.ruleset_id, "context_id": payload.context_id,
                "probability_matrix_hash": result["probability_matrix_hash"],
                "state_data_through": result["prediction_datetime"], "state_is_live": False,
                "historical": False, "immutable": False}
    return {"request_id": request.state.request_id, "api_version": _api_version(), "result": result,
            "metadata": metadata}


@router.post(
    "/simulate",
    response_model=TournamentSimulationResponse,
    summary="Monte Carlo Major simulation - Bernoulli sampling from the frozen RF probability matrix",
)
def post_simulate(payload: MajorSimulateRequest, request: Request):
    _require_subsystem_ready("prediction_ready")
    _require_subsystem_ready("tournament_engine_ready")
    request.state.context_id = payload.context_id
    request.state.mode = "major_simulate"

    result = ats.simulate_tournament(
        payload.ruleset_id, payload.context_id, payload.tier, payload.prediction_datetime,
        _participants_to_plain(payload.participants), payload.simulation_count, seed=payload.seed,
        manual_overrides=_overrides_to_plain(payload.manual_overrides))
    result = dict(result, champion_ranking=_to_champion_ranking(result["champion_ranking"]))

    metadata = {"ruleset_id": payload.ruleset_id, "context_id": payload.context_id,
                "probability_matrix_hash": result["probability_matrix_hash"],
                "state_data_through": result["prediction_datetime"], "state_is_live": False,
                "historical": False, "immutable": False}
    return {"request_id": request.state.request_id, "api_version": _api_version(), "result": result,
            "metadata": metadata}
