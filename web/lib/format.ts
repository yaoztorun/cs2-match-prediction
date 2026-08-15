/**
 * Display-level formatting only — the frontend never recomputes model
 * quantities (Phase 10A spec §39); it formats numbers/strings the API
 * already produced.
 */

/** 0.5523 -> "55%" (default) or "55.2%" with decimals=1 */
export function formatPercent(p: number, decimals = 0): string {
  return `${(p * 100).toFixed(decimals)}%`;
}

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
] as const;

/**
 * Formats the backend's snapshot timestamp string ("2026-06-28T20:00:00" or
 * "2026-06-28 20:00:00") as "28 Jun 2026" by reading the DATE COMPONENTS
 * DIRECTLY FROM THE STRING — never via `new Date(...)`/browser timezone
 * conversion, which could shift the calendar date (amendment #20: the
 * backend timestamp's timezone semantics are intentionally unspecified).
 */
export function formatDataThrough(timestamp: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(timestamp);
  if (!m) return timestamp;
  const [, year, month, day] = m;
  const monthName = MONTHS[Number(month) - 1] ?? month;
  return `${Number(day)} ${monthName} ${year}`;
}
