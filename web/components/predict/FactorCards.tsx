"use client";

/**
 * "Why the model leans this way" — grouped model factors, split by the team
 * they support. Renders ONLY groups actually returned by the API. Display
 * copy is built from structured fields (factor_group + direction + canonical
 * names, amendment #6); the verbatim backend summary lives in the details
 * drawer, untouched. The magnitude bar is relative WITHIN this one
 * explanation only and never carries a % label (amendment #7).
 */
import { Info } from "lucide-react";
import type { GroupedFactor, SummaryExplanation } from "@/lib/api/types";
import { factorLabel, factorSentence, relativeMagnitude } from "@/lib/factors";

function FactorCard({
  factor,
  all,
  teamAName,
  teamBName,
}: {
  factor: GroupedFactor;
  all: GroupedFactor[];
  teamAName: string;
  teamBName: string;
}) {
  const magnitude = relativeMagnitude(factor, all);
  const barColor =
    factor.direction === "team_a"
      ? "bg-team-a/70"
      : factor.direction === "team_b"
        ? "bg-team-b/70"
        : "bg-neutral-mid/50";
  return (
    <li className="panel-2 px-3.5 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-semibold text-ink">
          {factorLabel(factor.factor_group)}
        </span>
      </div>
      <p className="t-meta mt-0.5">
        {factorSentence(factor, teamAName, teamBName)}
      </p>
      <div
        className="mt-2 h-1 w-full overflow-hidden rounded-full bg-canvas"
        title="Relative model contribution (within this prediction only)"
      >
        <div
          className={`h-full rounded-full ${barColor}`}
          style={{ width: `${Math.round(magnitude * 100)}%` }}
        />
      </div>
    </li>
  );
}

export function FactorCards({
  explanation,
  teamAName,
  teamBName,
  heading = "Why the model leans this way",
}: {
  explanation: SummaryExplanation;
  teamAName: string;
  teamBName: string;
  heading?: string;
}) {
  const all = explanation.grouped_factors;
  const teamA = explanation.team_a_factors;
  const teamB = explanation.team_b_factors;

  return (
    <section aria-label={heading}>
      <h3 className="t-heading text-ink">{heading}</h3>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <div>
          <div className="t-label text-team-a">Supporting {teamAName}</div>
          {teamA.length === 0 ? (
            <p className="t-meta mt-2">No net factors on this side.</p>
          ) : (
            <ul className="mt-2 space-y-2">
              {teamA.map((f) => (
                <FactorCard
                  key={f.factor_group}
                  factor={f}
                  all={all}
                  teamAName={teamAName}
                  teamBName={teamBName}
                />
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="t-label text-team-b">Supporting {teamBName}</div>
          {teamB.length === 0 ? (
            <p className="t-meta mt-2">No net factors on this side.</p>
          ) : (
            <ul className="mt-2 space-y-2">
              {teamB.map((f) => (
                <FactorCard
                  key={f.factor_group}
                  factor={f}
                  all={all}
                  teamAName={teamAName}
                  teamBName={teamBName}
                />
              ))}
            </ul>
          )}
        </div>
      </div>
      <p className="t-meta mt-3 flex items-start gap-1.5">
        <Info size={13} aria-hidden className="mt-0.5 shrink-0" />
        Factors describe what influenced the model prediction; they do not
        prove causal reasons for a real match outcome.
      </p>
    </section>
  );
}
