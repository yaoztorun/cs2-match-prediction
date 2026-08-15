"use client";

/**
 * Ordered map slots for known-maps mode: exactly 1/3/5 numbered slots per
 * BO. Options come from GET /api/v1/maps filtered by the STRUCTURED
 * `model_supported` field (amendment #16 — the nine names are never
 * hardcoded; Cache is simply never offered because the API never returns it
 * as supported). Duplicates are disabled. Labeled "Supported maps", never
 * "map pool" / "Active Duty" (amendment #17).
 */
import { useEffect, useState } from "react";
import { getMaps } from "@/lib/api/endpoints";
import type { BestOf, MapInfo } from "@/lib/api/types";

let mapsPromise: Promise<MapInfo[]> | null = null;

function fetchSupportedMapsOnce(): Promise<MapInfo[]> {
  if (!mapsPromise) {
    mapsPromise = getMaps().then((res) =>
      res.maps.filter((m) => m.model_supported === true),
    );
    mapsPromise.catch(() => {
      mapsPromise = null;
    });
  }
  return mapsPromise;
}

/** Test-only reset. */
export function __resetMapsCache(): void {
  mapsPromise = null;
}

export function OrderedMapSelector({
  bestOf,
  value,
  onChange,
}: {
  bestOf: BestOf;
  /** value[i] is the map for slot i+1; "" = unselected. Length === bestOf. */
  value: string[];
  onChange: (maps: string[]) => void;
}) {
  const [maps, setMaps] = useState<MapInfo[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchSupportedMapsOnce()
      .then((m) => {
        if (!cancelled) setMaps(m);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const slots = Array.from({ length: bestOf }, (_, i) => value[i] ?? "");
  const chosen = new Set(slots.filter(Boolean));

  function setSlot(index: number, mapName: string) {
    const next = [...slots];
    next[index] = mapName;
    onChange(next);
  }

  return (
    <div>
      <div className="t-label">Ordered maps · supported maps only</div>
      {failed && (
        <p className="t-meta mt-1.5 text-danger">
          Supported maps could not be loaded — check the prediction service.
        </p>
      )}
      <div className="mt-1.5 grid gap-2 sm:grid-cols-3 md:grid-cols-5">
        {slots.map((slotValue, i) => (
          <div key={i}>
            <label htmlFor={`map-slot-${i}`} className="t-meta">
              Map {i + 1}
            </label>
            <select
              id={`map-slot-${i}`}
              value={slotValue}
              onChange={(e) => setSlot(i, e.target.value)}
              disabled={maps === null}
              className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2.5 py-2 text-sm text-ink disabled:opacity-50"
            >
              <option value="">Select…</option>
              {(maps ?? []).map((m) => {
                const takenElsewhere =
                  chosen.has(m.map_name) && slotValue !== m.map_name;
                return (
                  <option
                    key={m.map_name}
                    value={m.map_name}
                    disabled={takenElsewhere}
                  >
                    {m.map_name}
                    {takenElsewhere ? " (already picked)" : ""}
                  </option>
                );
              })}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
}
