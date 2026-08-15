"use client";

/**
 * Single source of truth for the data-through date (amendment #19). Every
 * "Data through …" string in the UI derives from GET /api/v1/meta (or a
 * prediction response's own metadata) — the date is never hardcoded in
 * components, so a future state release updates the UI without code changes.
 * One module-level in-flight promise means the whole app performs at most
 * one /meta request per page load.
 */
import { useEffect, useState } from "react";
import { getMeta } from "@/lib/api/endpoints";
import type { MetaResponse } from "@/lib/api/types";

let metaPromise: Promise<MetaResponse> | null = null;
let cachedMeta: MetaResponse | null = null;

export function fetchMetaOnce(): Promise<MetaResponse> {
  if (!metaPromise) {
    metaPromise = getMeta().then((m) => {
      cachedMeta = m;
      return m;
    });
    metaPromise.catch(() => {
      metaPromise = null; // allow retry after failure
    });
  }
  return metaPromise;
}

/** Test-only reset. */
export function __resetMetaCache(): void {
  metaPromise = null;
  cachedMeta = null;
}

export function useMeta(): {
  meta: MetaResponse | null;
  error: boolean;
} {
  const [meta, setMeta] = useState<MetaResponse | null>(cachedMeta);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchMetaOnce()
      .then((m) => {
        if (!cancelled) setMeta(m);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { meta, error };
}
