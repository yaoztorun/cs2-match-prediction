"use client";

/**
 * /predict orchestration.
 *
 * Result integrity (amendment #12): every response is stored with the
 * canonical `predictionRequestSignature` of the inputs that produced it;
 * the result renders ONLY while the current input signature matches. Any
 * prediction-defining change collapses the result to an explicit
 * "inputs changed" notice until Predict is run again.
 *
 * URL state (amendments #13/#14/#15): inputs sync to ?a=&b=&bo=&mode=&maps=
 * via shallow history.replaceState (no navigation, no implicit inference).
 * Auto-run happens at most once per mount, only when the URL already
 * contains a complete valid request. Canonical team names are used verbatim
 * (standard URL encoding, no invented slugs) and known-maps order is
 * preserved exactly.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import { ApiError, ApiUnreachableError } from "@/lib/api/client";
import { predictSeries, searchTeams } from "@/lib/api/endpoints";
import type {
  BestOf,
  PredictSeriesEnvelope,
  PredictionMode,
  TeamInfo,
} from "@/lib/api/types";
import { useHealth } from "@/lib/hooks/useHealth";
import { BestOfSelector } from "./BestOfSelector";
import { MapModeSelector } from "./MapModeSelector";
import { OrderedMapSelector } from "./OrderedMapSelector";
import { PredictionResult } from "./PredictionResult";
import { TeamSelector } from "./TeamSelector";

const ERROR_COPY: Record<string, string> = {
  unknown_team: "That team isn't available in this model snapshot.",
  ambiguous_team:
    "That team name is ambiguous in this model snapshot — pick it from the suggestions.",
  unsupported_map: "This map isn't supported by the frozen map model.",
  service_unavailable: "Prediction service is temporarily unavailable.",
  invalid_map_count: "Pick the exact number of maps for this series format.",
  duplicate_map: "Each map can only be picked once.",
};

function errorMessage(e: unknown): string {
  if (e instanceof ApiError) {
    return ERROR_COPY[e.code] ?? "The prediction couldn't be completed — adjust the inputs and try again.";
  }
  if (e instanceof ApiUnreachableError) {
    return "The prediction service could not be reached.";
  }
  return "Something went wrong while predicting this matchup.";
}

interface Inputs {
  teamA: TeamInfo | null;
  teamB: TeamInfo | null;
  bestOf: BestOf;
  mode: PredictionMode;
  maps: string[];
}

function inputSignature(i: Inputs): string {
  return JSON.stringify({
    a: i.teamA?.canonical_name ?? null,
    b: i.teamB?.canonical_name ?? null,
    bo: i.bestOf,
    mode: i.mode,
    maps: i.mode === "known_maps" ? i.maps.slice(0, i.bestOf) : null,
  });
}

function inputsComplete(i: Inputs): boolean {
  if (!i.teamA || !i.teamB) return false;
  if (i.teamA.canonical_name === i.teamB.canonical_name) return false;
  if (i.mode === "known_maps") {
    const maps = i.maps.slice(0, i.bestOf);
    if (maps.length !== i.bestOf || maps.some((m) => !m)) return false;
    if (new Set(maps).size !== maps.length) return false;
  }
  return true;
}

function parseBestOf(raw: string | null): BestOf | null {
  if (raw === "1") return 1;
  if (raw === "3") return 3;
  if (raw === "5") return 5;
  return null;
}

export function PredictClient() {
  const { health, retry } = useHealth();

  const [teamA, setTeamA] = useState<TeamInfo | null>(null);
  const [teamB, setTeamB] = useState<TeamInfo | null>(null);
  const [bestOf, setBestOf] = useState<BestOf>(3);
  const [mode, setMode] = useState<PredictionMode>("pre_veto");
  const [maps, setMaps] = useState<string[]>([]);

  const [result, setResult] = useState<{
    signature: string;
    envelope: PredictSeriesEnvelope;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [urlRestoreNotice, setUrlRestoreNotice] = useState<string | null>(null);

  const restoredRef = useRef(false);
  const autoRanRef = useRef(false);

  const inputs: Inputs = useMemo(
    () => ({ teamA, teamB, bestOf, mode, maps }),
    [teamA, teamB, bestOf, mode, maps],
  );
  const signature = inputSignature(inputs);
  const complete = inputsComplete(inputs);
  const resultIsCurrent = result !== null && result.signature === signature;
  const resultIsStale = result !== null && !resultIsCurrent;

  const runPredict = useCallback(
    async (i: Inputs) => {
      if (!inputsComplete(i) || !i.teamA || !i.teamB) return;
      const sig = inputSignature(i);
      setLoading(true);
      setError(null);
      try {
        const envelope = await predictSeries({
          teamA: i.teamA.canonical_name,
          teamB: i.teamB.canonical_name,
          bestOf: i.bestOf,
          mode: i.mode,
          orderedMaps:
            i.mode === "known_maps" ? i.maps.slice(0, i.bestOf) : undefined,
        });
        setResult({ signature: sig, envelope });
      } catch (e) {
        setError(errorMessage(e));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // --- URL restore (once per mount) + optional single auto-run -------------
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    const params = new URLSearchParams(window.location.search);
    const a = params.get("a");
    const b = params.get("b");
    const bo = parseBestOf(params.get("bo"));
    const urlMode = params.get("mode");
    const urlMaps = params.get("maps")?.split(",").filter(Boolean) ?? [];

    const nextBo = bo ?? 3;
    const nextMode: PredictionMode =
      urlMode === "known_maps" ? "known_maps" : "pre_veto";

    (async () => {
      setBestOf(nextBo);
      setMode(nextMode);
      if (nextMode === "known_maps") setMaps(urlMaps);
      if (!a && !b) return;
      // Resolve URL team names through the API contract — exact canonical
      // match only, never a silent substitution (amendment #14).
      async function resolveExact(name: string): Promise<TeamInfo | null> {
        try {
          const res = await searchTeams(name, 50);
          return res.teams.find((t) => t.canonical_name === name) ?? null;
        } catch {
          return null;
        }
      }
      const [resolvedA, resolvedB] = await Promise.all([
        a ? resolveExact(a) : Promise.resolve(null),
        b ? resolveExact(b) : Promise.resolve(null),
      ]);
      const missing: string[] = [];
      if (a && !resolvedA) missing.push(a);
      if (b && !resolvedB) missing.push(b);
      if (missing.length > 0) {
        setUrlRestoreNotice(
          `${missing.join(" and ")} ${missing.length === 1 ? "isn't" : "aren't"} available in this model snapshot.`,
        );
      }
      if (resolvedA) setTeamA(resolvedA);
      if (resolvedB) setTeamB(resolvedB);

      const restored: Inputs = {
        teamA: resolvedA,
        teamB: resolvedB,
        bestOf: nextBo,
        mode: nextMode,
        maps: urlMaps,
      };
      if (!autoRanRef.current && inputsComplete(restored)) {
        autoRanRef.current = true;
        void runPredict(restored);
      }
    })();
  }, [runPredict]);

  // --- URL sync (shallow, never triggers prediction — amendment #13) -------
  useEffect(() => {
    if (!restoredRef.current) return;
    const params = new URLSearchParams();
    if (teamA) params.set("a", teamA.canonical_name);
    if (teamB) params.set("b", teamB.canonical_name);
    params.set("bo", String(bestOf));
    params.set("mode", mode);
    if (mode === "known_maps") {
      const activeMaps = maps.slice(0, bestOf).filter(Boolean);
      if (activeMaps.length > 0) params.set("maps", activeMaps.join(","));
    }
    const qs = params.toString();
    window.history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
  }, [teamA, teamB, bestOf, mode, maps]);

  function changeBestOf(bo: BestOf) {
    setBestOf(bo);
    setMaps((m) => m.slice(0, bo));
  }

  const devMode = process.env.NODE_ENV === "development";

  return (
    <div className="space-y-6">
      {health === "unavailable" && (
        <div
          role="alert"
          className="panel flex flex-wrap items-center justify-between gap-3 border-danger/30 px-4 py-3.5"
        >
          <div>
            <div className="text-sm font-semibold text-ink">
              Prediction service unavailable
            </div>
            <p className="t-meta mt-0.5">
              The model backend isn&apos;t reachable right now. Predictions
              need a live connection to it.
            </p>
          </div>
          <button
            type="button"
            onClick={retry}
            className="flex items-center gap-1.5 rounded-lg border border-line bg-panel-2 px-3.5 py-2 text-sm font-medium text-ink hover:border-line-strong"
          >
            <RefreshCw size={14} aria-hidden />
            Retry
          </button>
        </div>
      )}

      {/* Input card */}
      <section aria-label="Matchup" className="panel px-5 py-5">
        <div className="grid items-start gap-3 md:grid-cols-[1fr_auto_1fr]">
          <TeamSelector
            label="Team A"
            side="a"
            selected={teamA}
            onSelect={setTeamA}
            excludeName={teamB?.canonical_name}
          />
          <div className="t-label hidden self-center px-1 pt-5 text-ink-3 md:block">
            vs
          </div>
          <TeamSelector
            label="Team B"
            side="b"
            selected={teamB}
            onSelect={setTeamB}
            excludeName={teamA?.canonical_name}
          />
        </div>
        {urlRestoreNotice && (
          <p className="t-meta mt-3 text-warn" role="status">
            {urlRestoreNotice}
          </p>
        )}
        <div className="mt-5 flex flex-wrap items-start gap-x-8 gap-y-4">
          <BestOfSelector value={bestOf} onChange={changeBestOf} />
          <div className="min-w-56 flex-1">
            <MapModeSelector value={mode} onChange={setMode} />
          </div>
        </div>
        {mode === "known_maps" && (
          <div className="mt-4">
            <OrderedMapSelector
              bestOf={bestOf}
              value={maps}
              onChange={setMaps}
            />
          </div>
        )}
        <div className="mt-5 flex items-center gap-4">
          <button
            type="button"
            disabled={!complete || loading || health === "unavailable"}
            onClick={() => void runPredict(inputs)}
            className="rounded-[10px] bg-accent px-6 py-2.5 text-sm font-semibold text-canvas transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Predict match
          </button>
          {resultIsStale && !loading && (
            <span className="t-meta text-warn" role="status">
              Inputs changed — run Predict to update the result.
            </span>
          )}
        </div>
      </section>

      {/* Result region — fixed minimum height to avoid layout jumps */}
      <div aria-live="polite" className="min-h-40">
        {loading && (
          <div className="panel flex min-h-40 items-center justify-center px-5 py-6">
            <span className="t-body text-ink-2">Analyzing matchup…</span>
          </div>
        )}
        {!loading && error && (
          <div role="alert" className="panel border-danger/30 px-5 py-5">
            <div className="text-sm font-semibold text-ink">
              Prediction failed
            </div>
            <p className="t-body mt-1 text-ink-2">{error}</p>
          </div>
        )}
        {!loading && !error && resultIsCurrent && result && (
          <PredictionResult envelope={result.envelope} />
        )}
      </div>

      {devMode && result && (
        <button
          type="button"
          className="t-meta rounded-md border border-line px-2.5 py-1.5 text-ink-3 hover:text-ink"
          onClick={async () => {
            if (!teamA || !teamB) return;
            const full = await predictSeries({
              teamA: teamA.canonical_name,
              teamB: teamB.canonical_name,
              bestOf,
              mode,
              orderedMaps:
                mode === "known_maps" ? maps.slice(0, bestOf) : undefined,
              explanationDetail: "full",
            });
            // Developer diagnostics only — never rendered in the product UI.
            console.log("[dev] full explanation payload", full);
          }}
        >
          Log full explanation to console (dev only)
        </button>
      )}
    </div>
  );
}
