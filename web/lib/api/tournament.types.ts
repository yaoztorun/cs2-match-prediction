/**
 * Phase 9E Major tournament API type definitions — PREPARED FOR PHASE 10B.
 * No Major UI exists in Phase 10A; these types document the frozen backend
 * contract (`/api/v1/major/*`) so the next frontend phase can build against
 * them without re-deriving shapes.
 */
import type { BestOf } from "./types";

export interface TournamentParticipantInput {
  team: string;
  seed: number;
}

export interface TournamentParticipantsInput {
  stage1: TournamentParticipantInput[]; // exactly 16, seeds 1-16
  stage2_direct: TournamentParticipantInput[]; // exactly 8, seeds 1-8
  stage3_direct: TournamentParticipantInput[]; // exactly 8, seeds 1-8
}

export interface ManualOverrideInput {
  stage: "stage_1" | "stage_2" | "stage_3" | "playoffs";
  round_number?: number; // stages: 1..5
  record_group?: string; // stages: e.g. "1-0"
  playoff_round?: "quarterfinal" | "semifinal" | "grand_final";
  team_1: string;
  team_2: string;
  winner: string;
  best_of?: BestOf;
}

export interface MajorPathRequest {
  ruleset_id?: string;
  context_id?: string;
  tier?: "tier1" | "tier2" | "tier3";
  participants: TournamentParticipantsInput;
  manual_overrides?: ManualOverrideInput[];
}

export interface MajorSimulateRequest extends MajorPathRequest {
  simulation_count: number; // 1..50000
  seed?: number; // default 42
}

export interface ProbabilityStat {
  numerator_count: number;
  denominator_count: number;
  probability: number | null;
  mc_standard_error: number | null;
}
