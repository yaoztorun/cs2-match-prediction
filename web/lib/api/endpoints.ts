/**
 * Typed endpoint functions for the Phase 9D prediction API. The frontend
 * never sets `prediction_datetime` (the backend's frozen snapshot cutoff
 * governs freshness) and defaults to summary explanations — the product
 * contract per Phase 10A spec §18/§30/§32.
 */
import { DEFAULT_CONTEXT_ID } from "@/lib/app-config";
import { apiGet, apiPost, type RequestOptions } from "./client";
import type {
  BestOf,
  HealthReadyResponse,
  MapsResponse,
  MetaResponse,
  PredictSeriesEnvelope,
  PredictionMode,
  TeamsResponse,
} from "./types";

export function getMeta(options?: RequestOptions): Promise<MetaResponse> {
  return apiGet<MetaResponse>("/api/v1/meta", undefined, options);
}

export function getHealthReady(
  options?: RequestOptions,
): Promise<HealthReadyResponse> {
  return apiGet<HealthReadyResponse>("/api/v1/health/ready", undefined, options);
}

export function getContexts(options?: RequestOptions): Promise<unknown> {
  return apiGet("/api/v1/contexts", undefined, options);
}

export function searchTeams(
  q: string,
  limit = 20,
  contextId = DEFAULT_CONTEXT_ID,
  options?: RequestOptions,
): Promise<TeamsResponse> {
  return apiGet<TeamsResponse>(
    "/api/v1/teams",
    { context_id: contextId, q, limit },
    options,
  );
}

export function getMaps(
  contextId = DEFAULT_CONTEXT_ID,
  options?: RequestOptions,
): Promise<MapsResponse> {
  return apiGet<MapsResponse>("/api/v1/maps", { context_id: contextId }, options);
}

export interface PredictSeriesInput {
  teamA: string;
  teamB: string;
  bestOf: BestOf;
  mode: PredictionMode;
  orderedMaps?: string[];
  explanationDetail?: "summary" | "full";
}

export function predictSeries(
  input: PredictSeriesInput,
  options?: RequestOptions,
): Promise<PredictSeriesEnvelope> {
  return apiPost<PredictSeriesEnvelope>(
    "/api/v1/predict/series",
    {
      context_id: DEFAULT_CONTEXT_ID,
      mode: input.mode,
      team_a: input.teamA,
      team_b: input.teamB,
      best_of: input.bestOf,
      ...(input.mode === "known_maps"
        ? { ordered_maps: input.orderedMaps }
        : {}),
      include_explanation: true,
      explanation_detail: input.explanationDetail ?? "summary",
    },
    options,
  );
}

export function predictMap(
  teamA: string,
  teamB: string,
  mapName: string,
  bestOf: BestOf,
  options?: RequestOptions,
): Promise<unknown> {
  return apiPost(
    "/api/v1/predict/map",
    {
      context_id: DEFAULT_CONTEXT_ID,
      team_a: teamA,
      team_b: teamB,
      map_name: mapName,
      best_of: bestOf,
      include_explanation: true,
      explanation_detail: "summary",
    },
    options,
  );
}
