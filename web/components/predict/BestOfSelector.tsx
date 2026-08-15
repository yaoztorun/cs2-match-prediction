"use client";

import type { BestOf } from "@/lib/api/types";

const OPTIONS: BestOf[] = [1, 3, 5];

/** Compact segmented BO1/BO3/BO5 control — typed 1|3|5 to the API. */
export function BestOfSelector({
  value,
  onChange,
}: {
  value: BestOf;
  onChange: (bo: BestOf) => void;
}) {
  return (
    <div>
      <div className="t-label" id="bestof-label">
        Best of
      </div>
      <div
        role="radiogroup"
        aria-labelledby="bestof-label"
        className="mt-1.5 inline-flex rounded-[10px] border border-line bg-panel-2 p-0.5"
      >
        {OPTIONS.map((bo) => (
          <button
            key={bo}
            type="button"
            role="radio"
            aria-checked={value === bo}
            onClick={() => onChange(bo)}
            className={`t-stat rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
              value === bo
                ? "bg-accent-deep text-ink"
                : "text-ink-2 hover:text-ink"
            }`}
          >
            BO{bo}
          </button>
        ))}
      </div>
    </div>
  );
}
