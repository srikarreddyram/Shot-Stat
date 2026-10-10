import { useEffect, useState } from "react";

// Shot-map data for the Stat Engine profiles (src/inference/shot_maps.py): a
// smooth shot-frequency density on a 1-ft grid (baseline row first) and a
// per-zone table carrying each zone's adjusted FG% vs the league.
// Drawn by components/stat-engine/ShotMap.tsx.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type ShotMapZone = {
  zone: string;
  fga: number;
  fgm: number;
  fg_pct: number | null;
  league_fg_pct: number | null;
  vs_league: number | null;
  /** vs_league after shrinkage toward the league — what the map colours. */
  vs_league_adjusted: number | null;
  /** 0-1: how much of the colour to show, by how many shots are behind it. */
  support: number;
  share: number;
};

export type ShotMapData = {
  season: string;
  /** Every season the shots come from: one, or the three ending in `season`. */
  seasons: string[];
  season_type: "regular" | "playoffs";
  grid: { x0: number; y0: number; cell: number; nx: number; ny: number };
  attempts: number;
  makes: number;
  drawn: number;
  frequency: (number | null)[];
  zones: ShotMapZone[];
};

export type ShotMetric = "frequency" | "vs_league";

export type ShotMapState = { data: ShotMapData | null; missing: boolean; error: string | null };

export function useShotMap(path: string | null): ShotMapState {
  const [state, setState] = useState<ShotMapState>({ data: null, missing: false, error: null });
  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    setState({ data: null, missing: false, error: null });
    fetch(`${API_BASE}${path}`)
      .then((r) => {
        if (r.status === 404) return null;
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: ShotMapData | null) => {
        if (!cancelled) setState({ data: d, missing: d == null, error: null });
      })
      .catch((e) => {
        if (!cancelled)
          setState({
            data: null,
            missing: false,
            error: e instanceof Error ? e.message : "Could not load",
          });
      });
    return () => {
      cancelled = true;
    };
  }, [path]);
  return state;
}
