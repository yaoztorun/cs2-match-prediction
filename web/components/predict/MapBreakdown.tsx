"use client";

/**
 * Known-maps breakdown. Preserves the Phase 9C separation strictly
 * (amendments #8/#9):
 *   MAP level    — each map's own TreeSHAP factor explanation (expandable,
 *                  clearly labeled per map; never merged into one series
 *                  factor ranking).
 *   SERIES level — composition mechanics only (chance the map is played +
 *                  series leverage), rendered as metadata rows visually
 *                  distinct from factor cards.
 */
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type {
  KnownMapsExplanation,
  KnownMapsPrediction,
} from "@/lib/api/types";
import { formatPercent } from "@/lib/format";
import { FactorCards } from "./FactorCards";
import { ProbabilityBar } from "./ProbabilityBar";

export function MapBreakdown({
  prediction,
  explanation,
}: {
  prediction: KnownMapsPrediction;
  explanation: KnownMapsExplanation | null;
}) {
  const [expandedMap, setExpandedMap] = useState<number | null>(null);
  const composition = explanation?.series_composition ?? null;
  const mapExplanations = explanation?.map_level_explanations ?? null;

  return (
    <section aria-label="Map breakdown">
      <h3 className="t-heading text-ink">Map-by-map</h3>
      <p className="t-meta mt-1">
        Later maps are only played if the series is still alive — the chance
        each map is played is shown alongside its probability.
      </p>
      <ul className="mt-3 space-y-2.5">
        {prediction.ordered_maps.map((map) => {
          const comp = composition?.find(
            (c) => c.map_number === map.map_number,
          );
          const mapExp = mapExplanations?.find(
            (m) => m.map_number === map.map_number,
          );
          const expanded = expandedMap === map.map_number;
          return (
            <li key={map.map_number} className="panel px-4 py-3.5">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <div className="flex items-baseline gap-2.5">
                  <span className="t-label">Map {map.map_number}</span>
                  <span className="text-sm font-semibold text-ink">
                    {map.map_name}
                  </span>
                </div>
                <span className="t-meta">
                  Favored:{" "}
                  <span className="text-ink-2">
                    {map.probability_team_a === 0.5
                      ? "even"
                      : map.probability_team_a > 0.5
                        ? prediction.team_a
                        : prediction.team_b}
                  </span>
                </span>
              </div>
              <div className="mt-2.5">
                <ProbabilityBar
                  probabilityTeamA={map.probability_team_a}
                  teamAName={prediction.team_a}
                  teamBName={prediction.team_b}
                  size="sm"
                />
              </div>
              {/* Series mechanics — NOT model factors (amendment #9) */}
              {comp && (
                <div className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1 border-t border-line pt-2.5">
                  <span className="t-meta">
                    Chance map is played:{" "}
                    <span className="t-stat text-ink-2">
                      {formatPercent(comp.probability_map_is_reached)}
                    </span>
                  </span>
                  <span
                    className="t-meta"
                    title="How strongly this map slot can influence the final series probability."
                  >
                    Series leverage:{" "}
                    <span className="t-stat text-ink-2">
                      {comp.series_composition_leverage.toFixed(2)}
                    </span>
                  </span>
                </div>
              )}
              {mapExp && (
                <div className="mt-2.5">
                  <button
                    type="button"
                    aria-expanded={expanded}
                    onClick={() =>
                      setExpandedMap(expanded ? null : map.map_number)
                    }
                    className="t-meta flex items-center gap-1 rounded-md py-1 text-ink-2 hover:text-ink"
                  >
                    <ChevronDown
                      size={14}
                      aria-hidden
                      className={`transition-transform ${expanded ? "rotate-180" : ""}`}
                    />
                    {expanded ? "Hide" : "Show"} factors for {map.map_name}
                  </button>
                  {expanded && (
                    <div className="mt-2">
                      <FactorCards
                        explanation={mapExp.explanation}
                        teamAName={prediction.team_a}
                        teamBName={prediction.team_b}
                        heading={`Why the model leans this way on ${map.map_name}`}
                      />
                    </div>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
