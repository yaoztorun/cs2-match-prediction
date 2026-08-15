import type { Metadata } from "next";
import { WifiOff } from "lucide-react";
import { APP_NAME } from "@/lib/app-config";

export const metadata: Metadata = { title: "Offline" };

/** Minimal offline fallback served by the service worker when navigation
 * fails without connectivity. Predictions genuinely require the backend —
 * nothing here pretends otherwise. */
export default function OfflinePage() {
  return (
    <div className="mx-auto max-w-md pt-16 text-center">
      <WifiOff size={28} aria-hidden className="mx-auto text-ink-3" />
      <h1 className="t-heading mt-4 text-ink">You&apos;re offline</h1>
      <p className="t-body mt-2 text-ink-2">
        {APP_NAME} needs a connection to the prediction service to compute
        probabilities. Reconnect and try again.
      </p>
    </div>
  );
}
