"use client";

/**
 * Collapsible methodology drawer. Model family names and attribution
 * methods live here rather than in the primary UX (spec §22/§31). RF is
 * labeled Saabas-style tree path decomposition — never SHAP. The verbatim
 * backend explanation summaries are preserved unchanged here (amendment #6).
 */
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { PredictSeriesEnvelope } from "@/lib/api/types";
import { isKnownMapsEnvelope } from "@/lib/api/types";
import { attributionMethodLabel } from "@/lib/factors";
import { formatDataThrough } from "@/lib/format";

export function PredictionDetails({
  envelope,
}: {
  envelope: PredictSeriesEnvelope;
}) {
  const [open, setOpen] = useState(false);
  const knownMaps = isKnownMapsEnvelope(envelope);

  const rows: [string, string][] = [];
  rows.push([
    "Prediction mode",
    knownMaps ? "Known-map series prediction" : "Pre-veto prediction",
  ]);
  rows.push(["Best of", String(envelope.prediction.best_of)]);
  rows.push(["Context", envelope.metadata.context_id]);
  rows.push(["Model", envelope.prediction.model_id]);
  if (knownMaps) {
    rows.push(["Series composition", envelope.prediction.composition_method]);
    const method =
      envelope.explanation?.map_level_explanations[0]?.explanation
        .attribution_method;
    if (method) rows.push(["Explanation method", attributionMethodLabel(method)]);
  } else if (envelope.explanation) {
    rows.push([
      "Explanation method",
      attributionMethodLabel(envelope.explanation.attribution_method),
    ]);
  }
  rows.push([
    "Data through",
    formatDataThrough(envelope.metadata.state_data_through),
  ]);
  rows.push([
    "Snapshot timestamp",
    `${envelope.metadata.state_data_through} (source timezone unspecified)`,
  ]);
  rows.push(["Causal claims", "None — model feature attribution only"]);

  const verbatimSummaries = knownMaps
    ? envelope.explanation?.map_level_explanations.flatMap((m) =>
        m.explanation.human_readable_summary.map(
          (s) => `[${m.map_name}] ${s}`,
        ),
      )
    : envelope.explanation?.human_readable_summary;

  return (
    <section aria-label="Prediction details">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="t-meta flex items-center gap-1 rounded-md py-1 text-ink-2 hover:text-ink"
      >
        <ChevronDown
          size={14}
          aria-hidden
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        />
        Prediction details
      </button>
      {open && (
        <div className="panel-2 mt-2 px-4 py-3.5">
          <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {rows.map(([term, value]) => (
              <div key={term} className="flex flex-col">
                <dt className="t-label">{term}</dt>
                <dd className="t-body mt-0.5 text-ink-2">{value}</dd>
              </div>
            ))}
          </dl>
          {verbatimSummaries && verbatimSummaries.length > 0 && (
            <div className="mt-4 border-t border-line pt-3">
              <div className="t-label">Model summary (verbatim)</div>
              <ul className="mt-1.5 space-y-1">
                {verbatimSummaries.map((s, i) => (
                  <li key={i} className="t-meta">
                    {s}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
