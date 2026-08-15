import Link from "next/link";
import { ArrowLeft } from "lucide-react";

/**
 * Shared shell for the intentionally-unimplemented Major routes
 * (Phase 10A spec §48/amendment #30): a clear "next release" statement,
 * no fake brackets, no fake percentages, no inert controls.
 */
export function ComingNextPhase({
  title,
  description,
  upcoming,
}: {
  title: string;
  description: string;
  upcoming: string[];
}) {
  return (
    <div className="mx-auto max-w-xl pt-10 sm:pt-16">
      <div className="panel px-6 py-8">
        <div className="t-label text-accent">Next release</div>
        <h1 className="t-display mt-2 text-ink">{title}</h1>
        <p className="t-body mt-3 text-ink-2">{description}</p>
        <ul className="mt-5 space-y-2">
          {upcoming.map((item) => (
            <li key={item} className="t-body flex gap-2 text-ink-2">
              <span aria-hidden className="text-ink-3">
                —
              </span>
              {item}
            </li>
          ))}
        </ul>
        <Link
          href="/predict"
          className="mt-7 inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel-2 px-4 py-2 text-sm font-medium text-ink hover:border-line-strong"
        >
          <ArrowLeft size={14} aria-hidden />
          Predict a match instead
        </Link>
      </div>
    </div>
  );
}
