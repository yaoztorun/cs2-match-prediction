/**
 * Single fetch wrapper for the Phase 9D API. All backend traffic flows
 * through here — components never call fetch("http://localhost:8000/...")
 * ad hoc. Parses the backend's structured `{error:{code,message,detail},
 * request_id}` envelope into a typed ApiError; raw error JSON never reaches
 * the UI layer.
 */
import { API_BASE_URL } from "@/lib/app-config";
import type { ApiErrorBody } from "./types";

export class ApiError extends Error {
  readonly code: string;
  readonly requestId: string | null;
  readonly status: number;
  readonly detail: Record<string, unknown>;

  constructor(
    status: number,
    code: string,
    message: string,
    requestId: string | null = null,
    detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.detail = detail;
  }
}

/** Thrown when the backend cannot be reached at all (network layer). */
export class ApiUnreachableError extends Error {
  constructor(cause?: unknown) {
    super("The prediction service could not be reached.");
    this.name = "ApiUnreachableError";
    this.cause = cause;
  }
}

export interface RequestOptions {
  signal?: AbortSignal;
}

async function parseErrorBody(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as ApiErrorBody;
    if (body?.error?.code) {
      return new ApiError(
        res.status,
        body.error.code,
        body.error.message,
        body.request_id ?? null,
        body.error.detail ?? {},
      );
    }
  } catch {
    // non-JSON error body — fall through to generic
  }
  return new ApiError(res.status, "http_error", `Request failed (${res.status})`);
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number | undefined>,
  options?: RequestOptions,
): Promise<T> {
  // Empty base URL = same-origin relative request (proxied by next.config.ts).
  let url = `${API_BASE_URL}${path}`;
  if (params) {
    const search = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) search.set(k, String(v));
    }
    const qs = search.toString();
    if (qs) url += `?${qs}`;
  }
  let res: Response;
  try {
    res = await fetch(url, { signal: options?.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiUnreachableError(e);
  }
  if (!res.ok) throw await parseErrorBody(res);
  return (await res.json()) as T;
}

export async function apiPost<T>(
  path: string,
  body: unknown,
  options?: RequestOptions,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options?.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiUnreachableError(e);
  }
  if (!res.ok) throw await parseErrorBody(res);
  return (await res.json()) as T;
}
