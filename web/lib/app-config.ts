/**
 * Centralized product identity + app-level constants (Phase 10A amendment #3).
 * "CS2 Match Intel" is a WORKING name, not established branding — renaming the
 * product means editing this file only. No permanent visual logo exists yet;
 * PWA icons are explicitly placeholder typographic tiles derived from
 * APP_INITIALS below.
 */
export const APP_NAME = "CS2 Match Intel";
export const APP_SHORT_NAME = "Match Intel";
export const APP_INITIALS = "MI";
export const APP_DESCRIPTION =
  "Model-grounded CS2 match probabilities: pre-veto series prediction, known-map analysis, and Major simulation.";

/** Theme colors surfaced to the PWA manifest / browser chrome. Keep in sync
 * with the --canvas / --accent tokens in app/globals.css. */
export const THEME_COLOR = "#0b0d12";
export const BACKGROUND_COLOR = "#0b0d12";

/**
 * Backend origin (Phase 10A spec §9). Two supported modes:
 *  - unset/empty (default): same-origin `/api/*` calls, proxied to the
 *    FastAPI backend by the rewrite in next.config.ts (dev target
 *    http://localhost:8000 via API_PROXY_TARGET).
 *  - set (e.g. "http://localhost:8000"): direct cross-origin calls; the
 *    backend's CORS allowlist must then include the frontend origin.
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export const DEFAULT_CONTEXT_ID = "deployment_post_cologne_v1";
