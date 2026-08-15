"use client";

/**
 * Searchable team selector — full ARIA combobox pattern (amendment #27):
 * combobox input with aria-expanded/aria-controls/aria-activedescendant,
 * listbox options, Arrow/Enter/Escape keyboard behavior.
 *
 * Server search only (GET /api/v1/teams): 250 ms debounce, minimum query
 * length 1, and BOTH AbortController cancellation and a monotonic
 * latest-query guard so a slow older response can never overwrite results
 * from a newer query (amendment #10). No client-side fuzzy matching; no
 * preloading of the full team list.
 */
import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { Search, X } from "lucide-react";
import { searchTeams } from "@/lib/api/endpoints";
import type { TeamInfo } from "@/lib/api/types";

const DEBOUNCE_MS = 250;
const MIN_QUERY_LENGTH = 1;

export interface TeamSelectorProps {
  label: string;
  side: "a" | "b";
  selected: TeamInfo | null;
  onSelect: (team: TeamInfo | null) => void;
  /** Canonical name selected on the other side — not selectable here. */
  excludeName?: string;
}

export function TeamSelector({
  label,
  side,
  selected,
  onSelect,
  excludeName,
}: TeamSelectorProps) {
  const id = useId();
  const listboxId = `${id}-listbox`;
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TeamInfo[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [searching, setSearching] = useState(false);
  const [searchFailed, setSearchFailed] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const requestSeq = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const runSearch = useCallback((q: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const seq = ++requestSeq.current;
    setSearching(true);
    setSearchFailed(false);
    searchTeams(q, 20, undefined, { signal: controller.signal })
      .then((res) => {
        if (seq !== requestSeq.current) return; // stale response — discard
        setResults(res.teams);
        setOpen(true);
        setActiveIndex(res.teams.length > 0 ? 0 : -1);
        setSearching(false);
      })
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        if (seq !== requestSeq.current) return;
        setSearching(false);
        setSearchFailed(true);
        setResults([]);
        setOpen(true);
      });
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (query.trim().length < MIN_QUERY_LENGTH) return;
    debounceRef.current = setTimeout(() => runSearch(query.trim()), DEBOUNCE_MS);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, runSearch]);

  function onQueryChange(value: string) {
    setQuery(value);
    if (value.trim().length < MIN_QUERY_LENGTH) {
      setResults([]);
      setOpen(false);
      setActiveIndex(-1);
    }
  }

  // Close on outside click
  useEffect(() => {
    function onPointerDown(e: PointerEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, []);

  function choose(team: TeamInfo) {
    if (team.canonical_name === excludeName) return;
    onSelect(team);
    setQuery("");
    setResults([]);
    setOpen(false);
    setActiveIndex(-1);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || results.length === 0) {
      if (e.key === "Escape") setOpen(false);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (i + 1) % results.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (i - 1 + results.length) % results.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < results.length) {
        choose(results[activeIndex]);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
    }
  }

  const sideColor = side === "a" ? "text-team-a" : "text-team-b";
  const sideBorder = side === "a" ? "border-team-a/40" : "border-team-b/40";

  if (selected) {
    return (
      <div
        className={`panel-2 flex items-center justify-between gap-3 border-l-2 ${sideBorder} px-3.5 py-3`}
      >
        <div className="min-w-0">
          <div className={`t-label ${sideColor}`}>{label}</div>
          <div className="t-heading mt-0.5 truncate text-ink">
            {selected.canonical_name}
          </div>
          <div className="t-meta mt-0.5">
            {selected.cold_start
              ? "No match history in this snapshot"
              : `${selected.history_match_count} matches in snapshot`}
          </div>
        </div>
        <button
          type="button"
          onClick={() => onSelect(null)}
          aria-label={`Clear ${label}`}
          className="shrink-0 rounded-md p-1.5 text-ink-3 hover:bg-panel hover:text-ink"
        >
          <X size={16} aria-hidden />
        </button>
      </div>
    );
  }

  return (
    <div ref={rootRef} className="relative">
      <label htmlFor={`${id}-input`} className={`t-label ${sideColor}`}>
        {label}
      </label>
      <div className="relative mt-1.5">
        <Search
          size={15}
          aria-hidden
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-3"
        />
        <input
          id={`${id}-input`}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls={listboxId}
          aria-activedescendant={
            activeIndex >= 0 ? `${id}-opt-${activeIndex}` : undefined
          }
          aria-autocomplete="list"
          autoComplete="off"
          spellCheck={false}
          placeholder="Search teams…"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={onKeyDown}
          className="w-full rounded-[10px] border border-line bg-panel-2 py-2.5 pl-9 pr-3 text-sm text-ink placeholder:text-ink-3"
        />
      </div>
      {open && (
        <ul
          id={listboxId}
          role="listbox"
          aria-label={`${label} suggestions`}
          className="panel absolute z-30 mt-1.5 max-h-72 w-full overflow-auto py-1 shadow-xl shadow-black/40"
        >
          {searching && (
            <li className="t-meta px-3.5 py-2" role="presentation">
              Searching…
            </li>
          )}
          {!searching && searchFailed && (
            <li className="t-meta px-3.5 py-2 text-danger" role="presentation">
              Team search unavailable — check the prediction service.
            </li>
          )}
          {!searching && !searchFailed && results.length === 0 && (
            <li className="t-meta px-3.5 py-2" role="presentation">
              No teams match this search.
            </li>
          )}
          {!searching &&
            results.map((team, i) => {
              const excluded = team.canonical_name === excludeName;
              return (
                <li
                  key={team.canonical_name}
                  id={`${id}-opt-${i}`}
                  role="option"
                  aria-selected={i === activeIndex}
                  aria-disabled={excluded || undefined}
                  onPointerDown={(e) => {
                    e.preventDefault();
                    if (!excluded) choose(team);
                  }}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={`flex cursor-pointer items-baseline justify-between gap-3 px-3.5 py-2 text-sm ${
                    excluded
                      ? "cursor-not-allowed text-ink-3"
                      : i === activeIndex
                        ? "bg-panel-2 text-ink"
                        : "text-ink-2"
                  }`}
                >
                  <span className="truncate">{team.canonical_name}</span>
                  <span className="t-meta shrink-0">
                    {excluded
                      ? "already selected"
                      : team.cold_start
                        ? "no history"
                        : `${team.history_match_count} matches`}
                  </span>
                </li>
              );
            })}
        </ul>
      )}
    </div>
  );
}
