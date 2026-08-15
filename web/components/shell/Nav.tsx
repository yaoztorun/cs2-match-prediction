"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Crosshair, Trophy } from "lucide-react";
import { APP_NAME } from "@/lib/app-config";
import { FreshnessBadge } from "./FreshnessBadge";

const DESTINATIONS = [
  { href: "/predict", label: "Predict", icon: Crosshair },
  { href: "/major", label: "Major", icon: Trophy },
] as const;

function isActive(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

/** Desktop: compact top bar. Mobile: fixed bottom navigation with
 * safe-area padding (amendment #28). */
export function Nav() {
  const pathname = usePathname();

  return (
    <>
      {/* Top bar (all widths; nav links hidden on mobile) */}
      <header className="sticky top-0 z-40 border-b border-line bg-panel-glass backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-6 px-4">
          <Link
            href="/"
            className="t-heading tracking-tight text-ink hover:text-accent"
          >
            {APP_NAME}
          </Link>
          <nav aria-label="Primary" className="hidden gap-1 md:flex">
            {DESTINATIONS.map(({ href, label, icon: Icon }) => {
              const active = isActive(pathname, href);
              return (
                <Link
                  key={href}
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    active
                      ? "bg-panel-2 text-ink"
                      : "text-ink-2 hover:bg-panel-2 hover:text-ink"
                  }`}
                >
                  <Icon size={15} aria-hidden />
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto hidden md:block">
            <FreshnessBadge />
          </div>
        </div>
      </header>

      {/* Mobile bottom navigation */}
      <nav
        aria-label="Primary"
        className="bottom-nav-safe fixed inset-x-0 bottom-0 z-40 border-t border-line bg-panel-glass backdrop-blur-md md:hidden"
      >
        <div className="flex h-16 items-stretch justify-around">
          {DESTINATIONS.map(({ href, label, icon: Icon }) => {
            const active = isActive(pathname, href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`flex flex-1 flex-col items-center justify-center gap-1 text-xs font-medium ${
                  active ? "text-accent" : "text-ink-2"
                }`}
              >
                <Icon size={19} aria-hidden />
                {label}
              </Link>
            );
          })}
        </div>
      </nav>
    </>
  );
}
