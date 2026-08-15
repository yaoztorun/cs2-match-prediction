"use client";

/**
 * The single probability visualization: a horizontal split bar. Neutral
 * around 50/50 (a midpoint tick marks even), percentage labels at both
 * ends, one entry transition that CSS disables entirely under
 * prefers-reduced-motion (see .prob-fill in globals.css — amendment #29).
 */
import { useEffect, useState } from "react";
import { formatPercent } from "@/lib/format";

export function ProbabilityBar({
  probabilityTeamA,
  teamAName,
  teamBName,
  size = "md",
  showLabels = true,
}: {
  probabilityTeamA: number;
  teamAName: string;
  teamBName: string;
  size?: "sm" | "md";
  showLabels?: boolean;
}) {
  // Start at 50/50 and transition once to the real value on mount.
  const [animatedIn, setAnimatedIn] = useState(false);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnimatedIn(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const pctA = probabilityTeamA * 100;
  const widthA = animatedIn ? pctA : 50;
  const height = size === "md" ? "h-3.5" : "h-2";

  return (
    <div>
      {showLabels && (
        <div className="mb-1 flex items-baseline justify-between">
          <span className="t-stat text-sm text-team-a">
            {formatPercent(probabilityTeamA)}
          </span>
          <span className="t-stat text-sm text-team-b">
            {formatPercent(1 - probabilityTeamA)}
          </span>
        </div>
      )}
      <div
        role="img"
        aria-label={`${teamAName} ${formatPercent(probabilityTeamA)} — ${teamBName} ${formatPercent(1 - probabilityTeamA)}`}
        className={`relative ${height} w-full overflow-hidden rounded-full bg-team-b-deep`}
      >
        <div
          data-testid="prob-fill-a"
          className="prob-fill absolute inset-y-0 left-0 rounded-l-full bg-team-a/80"
          style={{ width: `${widthA}%` }}
        />
        <div
          aria-hidden
          className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-neutral-mid/50"
        />
      </div>
    </div>
  );
}
