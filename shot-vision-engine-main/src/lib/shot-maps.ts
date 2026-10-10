import { useEffect, useState } from "react";

// Shot-map data for the Stat Engine profiles (src/inference/shot_maps.py): a
// smooth shot-frequency density on a 1-ft grid (baseline row first) and a
// per-zone table carrying each zone's adjusted FG% vs the league.
// Drawn by components/stat-engine/ShotMap.tsx.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type ShotGrid = { x0: number; y0: number; cell: number; nx: number; ny: number };

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
  grid: ShotGrid;
  attempts: number;
  makes: number;
  fg_pct: number | null;
  /** The league's FG% on these shots' exact zone mix. */
  expected_fg_pct: number | null;
  vs_expected: number | null;
  drawn: number;
  frequency: (number | null)[];
  zones: ShotMapZone[];
};

/** One map of a group: a subset of the player's shots (or his assignments'). */
export type SubsetMap = ShotMapData & {
  key: string;
  label: string;
  note: string;
  /** This subset's share of all the shots in the group. */
  share: number;
};

export type ShotMapGroup = {
  season: string;
  seasons: string[];
  season_type: "regular" | "playoffs";
  group: "shot_types" | "clutch" | "defense";
  maps: SubsetMap[];
};

export type ShotChangeZone = {
  zone: string;
  fga: number;
  previous_fga: number;
  share: number;
  previous_share: number;
  share_change: number;
  fg_pct: number | null;
  previous_fg_pct: number | null;
};

export type ShotChange = {
  season: string;
  previous_season: string;
  grid: ShotGrid;
  attempts: number;
  previous_attempts: number;
  /** This season's share of shots per square foot minus last season's. */
  change: (number | null)[];
  /** 0-1 per square foot: how far the change is beyond sampling noise
   *  (from 2 standard errors to full at 3.5) — the change map's opacity. */
  confidence: (number | null)[];
  /** Colour scale for `change`: the larger season's peak density. */
  scale: number;
  zones: ShotChangeZone[];
};

export type ShotMetric = "frequency" | "vs_league";

export type FetchState<T> = { data: T | null; missing: boolean; error: string | null };

/** GET an engine path; a 404 is "nothing to show" (`missing`), not an error. */
export function useShotMap<T = ShotMapData>(path: string | null): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({ data: null, missing: false, error: null });
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
      .then((d: T | null) => {
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
