/**
 * Typed contracts for the Phase 9D prediction API (`/api/v1`).
 *
 * These types are hand-written but VERIFIED against real captured fixtures in
 * `web/test/fixtures/` (Phase 10A amendment #21) — no field here was invented;
 * every one exists in a deterministic response captured from the running
 * FastAPI service. The backend contract is frozen (Phase 9D receipt); the
 * frontend adapts to it, never the reverse.
 */

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

export type BestOf = 1 | 3 | 5;
export type PredictionMode = "pre_veto" | "known_maps";
export type FactorDirection = "team_a" | "team_b" | "neutral";

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    detail: Record<string, unknown>;
  };
  request_id: string;
}

export interface ResponseMetadata {
  context_id: string;
  state_data_through: string;
  state_is_live: boolean;
  prediction_contract: string;
  explanation_version: string;
}

// ---------------------------------------------------------------------------
// Discovery endpoints
// ---------------------------------------------------------------------------

export interface MetaResponse {
  request_id: string;
  api_version: string;
  prediction_contract: string;
  explanation_version: string;
  default_context_id: string;
  available_context_ids: string[];
  model_ids: { series_unknown_map: string; known_map: string };
  deployment_state_data_through: string;
  state_is_live: boolean;
  explanation_causal: boolean;
}

export interface HealthReadyResponse {
  status: string;
  default_context_id: string;
  checked_at: string;
  subsystems?: Record<string, boolean>;
}

export interface TeamInfo {
  canonical_name: string;
  identity_eligible: boolean;
  history_available: boolean;
  history_match_count: number;
  cold_start: boolean;
}

export interface TeamsResponse {
  request_id: string;
  context_id: string;
  count: number;
  teams: TeamInfo[];
}

export interface MapInfo {
  map_name: string;
  model_supported: boolean;
  historical_context_available: boolean;
  deployment_context_observed: boolean;
  cologne_2026_competitive_pool: boolean;
  competitive_pool_status: string;
}

export interface MapsResponse {
  request_id: string;
  context_id: string;
  maps: MapInfo[];
}

// ---------------------------------------------------------------------------
// Prediction requests
// ---------------------------------------------------------------------------

export interface PredictSeriesRequest {
  context_id: string;
  mode: PredictionMode;
  team_a: string;
  team_b: string;
  best_of: BestOf;
  tier?: "tier1" | "tier2" | "tier3";
  ordered_maps?: string[];
  include_explanation?: boolean;
  explanation_detail?: "summary" | "full";
}

// ---------------------------------------------------------------------------
// Prediction responses — pre-veto (RF V2)
// ---------------------------------------------------------------------------

export interface TeamHistoryMeta {
  identity_known: boolean;
  available: boolean;
  cold_start: boolean;
  matches: number;
}

export interface DataFreshness {
  state_data_through: string;
  state_is_live: boolean;
  requested_prediction_datetime: string;
  staleness_days: number;
  snapshot_id: string;
  mode?: string; // "hypothetical_future_from_stale_snapshot" when stale
  warning?: string;
}

export interface PreVetoPrediction {
  prediction_mode: "pre_veto";
  team_a: string;
  team_b: string;
  best_of: number;
  tier: string;
  tier_source: string;
  probability_team_a: number;
  probability_team_b: number;
  favored_team: string | null;
  prediction_is_tied: boolean;
  model_id: string;
  context_id: string;
  state_cutoff: string;
  prediction_datetime: string;
  data_freshness: DataFreshness;
  team_a_history: TeamHistoryMeta;
  team_b_history: TeamHistoryMeta;
}

// ---------------------------------------------------------------------------
// Explanations (summary detail level — the product path)
// ---------------------------------------------------------------------------

export interface GroupedFactor {
  factor_group: string;
  direction: FactorDirection;
  signed_contribution: number;
  absolute_importance: number;
  rank: number;
  attribution_output_space: "probability" | "log_odds";
}

export interface StateSupport {
  overall_state_available: boolean;
  map_state_available: boolean;
  form_state_available: boolean;
  player_roster_state_available: boolean;
  modern_map_state_available: boolean;
  player_map_state_available: boolean;
  fallbacks_used: string[];
}

export interface InputProvenance {
  team_a_cold_start: boolean;
  team_b_cold_start: boolean;
  notes: string[];
}

export interface SummaryExplanation {
  explanation_type: string;
  causal: boolean;
  attribution_method: string;
  attribution_output_space: "probability" | "log_odds";
  base_value: number;
  grouped_factors: GroupedFactor[];
  team_a_factors: GroupedFactor[];
  team_b_factors: GroupedFactor[];
  neutral_factors: GroupedFactor[];
  top_positive_factors: GroupedFactor[];
  top_negative_factors: GroupedFactor[];
  human_readable_summary: string[];
  reconstruction_check: Record<string, unknown>;
  detail_level: string;
  input_provenance?: InputProvenance; // RF pre-veto only
  state_support?: StateSupport; // XGB known-map only
}

// ---------------------------------------------------------------------------
// Prediction responses — known maps (XGB V3 + series DP)
// ---------------------------------------------------------------------------

export interface OrderedMapProbability {
  map_number: number;
  map_name: string;
  probability_team_a: number;
  probability_team_b: number;
}

export interface KnownMapsPrediction {
  prediction_mode: "known_maps";
  team_a: string;
  team_b: string;
  best_of: number;
  ordered_maps: OrderedMapProbability[];
  series_probability_team_a: number;
  series_probability_team_b: number;
  favored_team: string | null;
  prediction_is_tied: boolean;
  model_id: string;
  composition_method: string;
  context_id: string;
  state_cutoff: string;
}

export interface MapLevelExplanation {
  map_number: number;
  map_name: string;
  probability_team_a: number;
  probability_team_b: number;
  explanation: SummaryExplanation;
}

export interface SeriesCompositionEntry {
  map_number: number;
  map_name: string;
  probability_team_a: number;
  probability_map_is_reached: number;
  series_composition_leverage: number;
}

export interface KnownMapsExplanation {
  map_level_explanations: MapLevelExplanation[];
  series_composition: SeriesCompositionEntry[];
  note: string;
}

// ---------------------------------------------------------------------------
// Envelopes
// ---------------------------------------------------------------------------

export interface PreVetoEnvelope {
  request_id: string;
  api_version: string;
  prediction: PreVetoPrediction;
  explanation: SummaryExplanation | null;
  metadata: ResponseMetadata;
}

export interface KnownMapsEnvelope {
  request_id: string;
  api_version: string;
  prediction: KnownMapsPrediction;
  explanation: KnownMapsExplanation | null;
  metadata: ResponseMetadata;
}

export type PredictSeriesEnvelope = PreVetoEnvelope | KnownMapsEnvelope;

export function isKnownMapsEnvelope(
  e: PredictSeriesEnvelope,
): e is KnownMapsEnvelope {
  return e.prediction.prediction_mode === "known_maps";
}
