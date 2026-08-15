"use client";

/**
 * Result-first presentation: the hero (team names + large probabilities +
 * one split bar), factor explanation, known-maps breakdown, data-quality
 * notices, freshness metadata, and the methodology drawer.
 */
import type { PredictSeriesEnvelope } from "@/lib/api/types";
import { isKnownMapsEnvelope } from "@/lib/api/types";
import { formatPercent, formatDataThrough } from "@/lib/format";
import { FactorCards } from "./FactorCards";
import { MapBreakdown } from "./MapBreakdown";
import { PredictionDetails } from "./PredictionDetails";
import { ProbabilityBar } from "./ProbabilityBar";
import { SupportNotice } from "./SupportNotice";

export function PredictionResult({
  envelope,
}: {
  envelope: PredictSeriesEnvelope;
}) {
  const knownMaps = isKnownMapsEnvelope(envelope);
  const p = envelope.prediction;
  const probA = knownMaps
    ? envelope.prediction.series_probability_team_a
    : envelope.prediction.probability_team_a;
  const tied = p.prediction_is_tied;
  const staleFreshness = !knownMaps
    ? envelope.prediction.data_freshness
    : null;

  return (
    <div className="space-y-6">
      {/* Hero */}
      <section aria-label="Prediction result" className="panel px-5 py-6">
        <div className="t-meta">
          {knownMaps ? "Known-map series prediction" : "Pre-veto prediction"}
          {" · BO"}
          {p.best_of}
        </div>
        <div className="mt-3 flex items-end justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="t-heading truncate text-team-a">{p.team_a}</div>
            <div className="t-stat mt-1 text-4xl font-semibold text-ink sm:text-5xl">
              {formatPercent(probA)}
            </div>
          </div>
          <div className="t-label pb-3 text-ink-3">vs</div>
          <div className="min-w-0 flex-1 text-right">
            <div className="t-heading truncate text-team-b">{p.team_b}</div>
            <div className="t-stat mt-1 text-4xl font-semibold text-ink sm:text-5xl">
              {formatPercent(1 - probA)}
            </div>
          </div>
        </div>
        <div className="mt-4">
          <ProbabilityBar
            probabilityTeamA={probA}
            teamAName={p.team_a}
            teamBName={p.team_b}
            showLabels={false}
          />
        </div>
        <div className="mt-3 text-sm text-ink-2">
          {tied ? (
            <span className="font-medium text-ink">Even matchup</span>
          ) : (
            <>
              Favored:{" "}
              <span className="font-semibold text-ink">{p.favored_team}</span>
            </>
          )}
        </div>
        {staleFreshness?.mode === "hypothetical_future_from_stale_snapshot" &&
          staleFreshness.warning && (
            <p className="t-meta mt-3 rounded-md border border-warn/30 bg-warn/5 px-3 py-2 text-warn">
              {staleFreshness.warning}
            </p>
          )}
      </section>

      <SupportNotice envelope={envelope} />

      {knownMaps ? (
        <>
          <MapBreakdown
            prediction={envelope.prediction}
            explanation={envelope.explanation}
          />
          <p className="t-meta">
            The series probability above is composed from the per-map
            probabilities; per-map factors are listed under each map and are
            never merged into a single series factor ranking.
          </p>
        </>
      ) : (
        envelope.explanation && (
          <FactorCards
            explanation={envelope.explanation}
            teamAName={p.team_a}
            teamBName={p.team_b}
          />
        )
      )}

      <footer className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-t border-line pt-4">
        <span className="t-meta">
          Data through{" "}
          <span className="t-stat text-ink-2">
            {formatDataThrough(envelope.metadata.state_data_through)}
          </span>
        </span>
        <PredictionDetails envelope={envelope} />
      </footer>
    </div>
  );
}
