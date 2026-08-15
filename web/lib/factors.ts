/**
 * Factor-group display vocabulary + display-copy construction.
 *
 * Amendment #6: frontend display copy is built from STRUCTURED fields
 * (factor_group, direction, canonical team names) — the backend's
 * deterministic `human_readable_summary` prose is NEVER string-mutated. The
 * verbatim backend summary remains available unchanged in the details
 * drawer. Only factor groups actually returned by the API are ever rendered.
 */
import type { FactorDirection, GroupedFactor } from "@/lib/api/types";

/** Friendly labels for the factor_group slugs the Phase 9C contract emits.
 * Unknown slugs fall back to a de-slugged form — never dropped. */
const FACTOR_LABELS: Record<string, string> = {
  overall_strength: "Overall strength",
  recent_performance: "Recent performance",
  recent_form: "Recent form",
  historical_experience: "Historical experience",
  event_context: "Series format & tier",
  opponent_strength: "Opponent strength",
  map_pool: "Map-pool depth",
  selected_map_strength: "Selected-map strength",
  map_experience: "Map experience",
  player_strength: "Player performance",
  roster_stability: "Roster stability",
  roster_map_familiarity: "Roster map familiarity",
};

export function factorLabel(factorGroup: string): string {
  return (
    FACTOR_LABELS[factorGroup] ??
    factorGroup.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase())
  );
}

/** Display copy built purely from structured fields, e.g.
 * "Overall strength favors Team Vitality". */
export function factorSentence(
  factor: GroupedFactor,
  teamAName: string,
  teamBName: string,
): string {
  const label = factorLabel(factor.factor_group);
  if (factor.direction === "neutral") {
    return `${label}: negligible net effect`;
  }
  const team = factor.direction === "team_a" ? teamAName : teamBName;
  return `${label} favors ${team}`;
}

export function teamForDirection(
  direction: FactorDirection,
  teamAName: string,
  teamBName: string,
): string | null {
  if (direction === "team_a") return teamAName;
  if (direction === "team_b") return teamBName;
  return null;
}

/**
 * Relative magnitude within ONE explanation only (amendment #7): normalized
 * to the max absolute grouped contribution of this explanation. Never
 * comparable across models/matches/maps or attribution spaces; never shown
 * with a "%" label.
 */
export function relativeMagnitude(
  factor: GroupedFactor,
  allFactors: GroupedFactor[],
): number {
  const maxAbs = Math.max(...allFactors.map((f) => f.absolute_importance), 0);
  if (maxAbs <= 0) return 0;
  return factor.absolute_importance / maxAbs;
}

/** Friendly phrases for the Phase 9B state_support fallback flags. Data
 * quality copy, kept visually separate from model factors (spec §29). */
const FALLBACK_PHRASES: Record<string, string> = {
  both_teams_have_history: "overall match history",
  both_teams_have_map_history: "map-specific history",
  both_teams_have_5_adjusted_matches: "recent-form history",
  both_teams_have_5_inferred_players: "roster data",
  both_teams_have_map_pool_history: "map-pool history",
  both_teams_have_recent_selected_map_history:
    "recent history on the selected map",
};

export function fallbackPhrase(key: string): string {
  return FALLBACK_PHRASES[key] ?? key.replace(/_/g, " ");
}

export function describeFallbacks(fallbacksUsed: string[]): string | null {
  if (fallbacksUsed.length === 0) return null;
  const phrases = fallbacksUsed.map(fallbackPhrase);
  const list =
    phrases.length === 1
      ? phrases[0]
      : `${phrases.slice(0, -1).join(", ")} and ${phrases[phrases.length - 1]}`;
  return `Limited ${list} for this matchup — the model used neutral defaults where data was missing.`;
}

export function attributionMethodLabel(method: string): string {
  if (method === "saabas_path_decomposition") {
    return "Saabas-style tree path decomposition";
  }
  if (method === "xgboost_native_treeshap") {
    return "TreeSHAP (native XGBoost)";
  }
  return method;
}
