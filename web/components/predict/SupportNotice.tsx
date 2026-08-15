"use client";

/**
 * DATA QUALITY notices — cold starts and state-support fallbacks, kept
 * visually separate from model factors (spec §29). A missing-data fallback
 * is never framed as a team weakness.
 */
import { Database } from "lucide-react";
import type { PredictSeriesEnvelope } from "@/lib/api/types";
import { isKnownMapsEnvelope } from "@/lib/api/types";
import { describeFallbacks } from "@/lib/factors";

export function SupportNotice({
  envelope,
}: {
  envelope: PredictSeriesEnvelope;
}) {
  const notices: string[] = [];

  if (isKnownMapsEnvelope(envelope)) {
    const p = envelope.prediction;
    const fallbackKeys = new Set<string>();
    for (const mle of envelope.explanation?.map_level_explanations ?? []) {
      for (const key of mle.explanation.state_support?.fallbacks_used ?? []) {
        fallbackKeys.add(key);
      }
    }
    const described = describeFallbacks([...fallbackKeys]);
    if (described) notices.push(described);
    void p;
  } else {
    const { team_a_history, team_b_history, team_a, team_b } =
      envelope.prediction;
    if (team_a_history.cold_start) {
      notices.push(
        `${team_a} has no match history in this data snapshot — the model used neutral default inputs for it.`,
      );
    }
    if (team_b_history.cold_start) {
      notices.push(
        `${team_b} has no match history in this data snapshot — the model used neutral default inputs for it.`,
      );
    }
  }

  if (notices.length === 0) return null;

  return (
    <aside
      aria-label="Data coverage"
      className="rounded-[10px] border border-warn/30 bg-warn/5 px-3.5 py-3"
    >
      <div className="flex items-center gap-1.5">
        <Database size={13} aria-hidden className="text-warn" />
        <span className="t-label text-warn">Data coverage</span>
      </div>
      <ul className="mt-1.5 space-y-1">
        {notices.map((n, i) => (
          <li key={i} className="t-body text-ink-2">
            {n}
          </li>
        ))}
      </ul>
    </aside>
  );
}
