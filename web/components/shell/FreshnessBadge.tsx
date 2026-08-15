"use client";

import { useMeta } from "@/lib/hooks/useMeta";
import { formatDataThrough } from "@/lib/format";

/**
 * Subtle "Data through 28 Jun 2026" chip. The date derives exclusively from
 * the API (/meta or an explicit override string from a prediction response's
 * metadata) — never hardcoded (amendment #19). Renders nothing until the
 * source is known; the shell never blocks on it.
 */
export function FreshnessBadge({
  dataThrough,
  className = "",
}: {
  /** Optional explicit timestamp (e.g. prediction metadata.state_data_through). */
  dataThrough?: string;
  className?: string;
}) {
  const { meta } = useMeta();
  const source = dataThrough ?? meta?.deployment_state_data_through;
  if (!source) return null;
  return (
    <span className={`t-meta whitespace-nowrap ${className}`}>
      Data through <span className="t-stat text-ink-2">{formatDataThrough(source)}</span>
    </span>
  );
}
