import { vi } from "vitest";

export interface FetchRoute {
  /** Substring matched against the request URL. */
  match: string;
  /** Response body (object → JSON). */
  body: unknown;
  status?: number;
  /** Optional artificial delay in ms (for stale-response tests). */
  delayMs?: number;
}

/** Installs a fetch stub that routes by URL substring. Later routes win so
 * tests can layer specific matches over general ones. */
export function mockFetchRoutes(routes: FetchRoute[]): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      const route = [...routes].reverse().find((r) => url.includes(r.match));
      if (!route) {
        throw new TypeError(`mockFetch: no route for ${url}`);
      }
      if (route.delayMs) {
        await new Promise((resolve) => setTimeout(resolve, route.delayMs));
      }
      if (init?.signal?.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }
      return new Response(JSON.stringify(route.body), {
        status: route.status ?? 200,
        headers: { "Content-Type": "application/json" },
      });
    },
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
