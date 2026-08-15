"use client";

/**
 * Backend readiness probe. Determines API-dependent capabilities only — the
 * static app shell, home content, and navigation render regardless of
 * backend availability (amendment #18). Only API-dependent actions gate on
 * this.
 */
import { useCallback, useEffect, useState } from "react";
import { getHealthReady } from "@/lib/api/endpoints";

export type HealthState = "checking" | "ready" | "unavailable";

export function useHealth(): { health: HealthState; retry: () => void } {
  const [health, setHealth] = useState<HealthState>("checking");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getHealthReady()
      .then((r) => {
        if (!cancelled) setHealth(r.status === "ready" ? "ready" : "unavailable");
      })
      .catch(() => {
        if (!cancelled) setHealth("unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setHealth("checking");
    setAttempt((a) => a + 1);
  }, []);
  return { health, retry };
}
