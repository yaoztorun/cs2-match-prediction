import Link from "next/link";
import { ArrowRight, Crosshair, Trophy } from "lucide-react";
import { FreshnessBadge } from "@/components/shell/FreshnessBadge";

const CAPABILITIES = [
  {
    title: "Pre-veto series probability",
    body: "Series win probability before any map is picked, from the frozen series model.",
  },
  {
    title: "Known-map analysis",
    body: "Per-map probabilities for an exact ordered map slate, composed into a series probability.",
  },
  {
    title: "Model-grounded factors",
    body: "Every prediction explains which factor groups pushed it — derived from the model itself.",
  },
  {
    title: "Major simulation",
    body: "Full Swiss + playoff Monte Carlo over a frozen probability table.",
  },
] as const;

export default function HomePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-10 pt-6 sm:pt-12">
      <section>
        <h1 className="t-display text-ink">
          Probabilities for CS2 matches,
          <br className="hidden sm:block" /> grounded in a frozen model.
        </h1>
        <p className="t-body mt-3 max-w-xl text-ink-2">
          Series and map win probabilities with model-grounded explanations —
          no hand-tuned power rankings, no vibes.
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-2" aria-label="Actions">
        <Link
          href="/predict"
          className="panel group flex items-center justify-between gap-3 px-5 py-5 transition-colors hover:border-line-strong"
        >
          <span className="flex items-center gap-3">
            <Crosshair size={19} aria-hidden className="text-team-a" />
            <span>
              <span className="block text-sm font-semibold text-ink">
                Predict a match
              </span>
              <span className="t-meta">Any two teams, BO1 to BO5</span>
            </span>
          </span>
          <ArrowRight
            size={16}
            aria-hidden
            className="text-ink-3 transition-transform group-hover:translate-x-0.5"
          />
        </Link>
        <Link
          href="/major"
          className="panel group flex items-center justify-between gap-3 px-5 py-5 transition-colors hover:border-line-strong"
        >
          <span className="flex items-center gap-3">
            <Trophy size={19} aria-hidden className="text-team-b" />
            <span>
              <span className="block text-sm font-semibold text-ink">
                Simulate a Major
              </span>
              <span className="t-meta">Arriving in the next release</span>
            </span>
          </span>
          <ArrowRight
            size={16}
            aria-hidden
            className="text-ink-3 transition-transform group-hover:translate-x-0.5"
          />
        </Link>
      </section>

      <section aria-label="What this provides">
        <ul className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
          {CAPABILITIES.map((c) => (
            <li key={c.title}>
              <div className="text-sm font-semibold text-ink">{c.title}</div>
              <p className="t-meta mt-0.5">{c.body}</p>
            </li>
          ))}
        </ul>
      </section>

      <footer className="border-t border-line pt-4">
        <FreshnessBadge />
        <p className="t-meta mt-1">
          Predictions come from a frozen model snapshot — they describe model
          expectations, not guaranteed outcomes.
        </p>
      </footer>
    </div>
  );
}
