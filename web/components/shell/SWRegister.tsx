"use client";

import { useEffect } from "react";

/** Registers the service worker in production builds only — SW caching
 * during `next dev` would mask live changes. */
export function SWRegister() {
  useEffect(() => {
    if (
      process.env.NODE_ENV === "production" &&
      typeof navigator !== "undefined" &&
      "serviceWorker" in navigator
    ) {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        // Installability is progressive enhancement — never surface a failure.
      });
    }
  }, []);
  return null;
}
