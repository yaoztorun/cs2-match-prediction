"use client";

import type { PredictionMode } from "@/lib/api/types";

const MODES: {
  value: PredictionMode;
  title: string;
  description: string;
}[] = [
  {
    value: "pre_veto",
    title: "Maps unknown",
    description: "Before veto / map selection",
  },
  {
    value: "known_maps",
    title: "Maps known",
    description: "Use the exact ordered maps",
  },
];

/** Two clearly separated prediction modes. Model names (RF V2 / XGB V3)
 * intentionally stay out of the primary labels — they live in the details
 * drawer (spec §16/§22). */
export function MapModeSelector({
  value,
  onChange,
}: {
  value: PredictionMode;
  onChange: (mode: PredictionMode) => void;
}) {
  return (
    <div>
      <div className="t-label" id="mapmode-label">
        Map mode
      </div>
      <div
        role="radiogroup"
        aria-labelledby="mapmode-label"
        className="mt-1.5 grid grid-cols-2 gap-2"
      >
        {MODES.map((mode) => {
          const active = value === mode.value;
          return (
            <button
              key={mode.value}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(mode.value)}
              className={`rounded-[10px] border px-3.5 py-2.5 text-left transition-colors ${
                active
                  ? "border-accent/50 bg-accent-deep/60"
                  : "border-line bg-panel-2 hover:border-line-strong"
              }`}
            >
              <div className={`text-sm font-semibold ${active ? "text-ink" : "text-ink-2"}`}>
                {mode.title}
              </div>
              <div className="t-meta mt-0.5">{mode.description}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
