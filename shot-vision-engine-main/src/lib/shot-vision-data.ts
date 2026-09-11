export interface Player {
  id: string;
  name: string;
  pos: string;
  // null = not on record. Never fabricated with a league-average or
  // height-derived guess — this project treats physical measurements as
  // exact data or nothing, same policy as the backend's ingestion pipeline.
  heightIn: number | null;
  weightLbs: number | null;
  wingspanIn: number | null;
  // null when the backend has no rating for this player — too few shots, or
  // too few defended attempts to publish a defensive number. The UI shows a
  // dash rather than substituting a plausible-looking default.
  offRtg: number | null;
  defRtgOvr: number | null;
  ratingComponents?: {
    scoring: number | null;
    playmaking: number | null;
    volume: number | null;
    creation: number | null;
    defence: number | null;
  };
  fg: number;
  tp: number;
  rim: number;
  defRtg: number;
  contest: string | number;
  ast?: number;
  tov?: number;
  ftPct?: number;
  rimPct?: number;
  midPct?: number;
  headshotUrl?: string;
  // "measured" = real recorded stats. "prior" = this player has no NBA shot
  // history yet (e.g. hasn't played a game this season) and the numbers
  // shown are a labeled position-based projection, not measured data.
  statsSource?: "measured" | "prior";
  resolvedSeason?: string | null;
}

// PLAYERS mock removed - now fetched from API

export interface Zone {
  key: string;
  label: string;
  shortLabel: string;
  baseEP: number;
  shotValue: number;
  color: string;
}

export const ZONES: Zone[] = [
  { key: "restrictedArea", label: "Restricted Area", shortLabel: "RA", baseEP: 1.36, shotValue: 2, color: "#16A34A" },
  { key: "paintNonRA", label: "In The Paint (Non-RA)", shortLabel: "PAINT", baseEP: 1.04, shotValue: 2, color: "#65A830" },
  { key: "leftCorner3", label: "Left Corner 3", shortLabel: "LC3", baseEP: 1.23, shotValue: 3, color: "#C9A84C" },
  { key: "rightCorner3", label: "Right Corner 3", shortLabel: "RC3", baseEP: 1.14, shotValue: 3, color: "#C9A84C" },
  { key: "aboveBreak3", label: "Above the Break 3", shortLabel: "AB3", baseEP: 1.08, shotValue: 3, color: "#B8860B" },
  { key: "midRange", label: "Mid-Range", shortLabel: "MID", baseEP: 0.88, shotValue: 2, color: "#DC2626" },
];

export interface EngineInput {
  attacker: Player;
  primaryDef: Player;
  secondaryDef: Player | null;
  doubleTeam: boolean;
  quarter: 1 | 2 | 3 | 4;
  minutes: number;
  seconds: number;
  scoreDiff: number;
  home: boolean;
}

export interface ZoneResult {
  zone: Zone;
  ep: number;
  makeProb: number;
  /**
   * Historical attempts backing this estimate, from the API's
   * `attempts_behind`. Previously this read `attempts`/`actual_fg_pct`, which
   * the backend has never sent — so the supporting caption fell through to a
   * placeholder on every card, in every session.
   */
  attemptsBehind?: number;
  /**
   * Distance (feet) of the specific location this recommendation refers to
   * — the zone's best-scoring spot (api.py ZoneSummary.shot_distance). A
   * zone spans a range of distances, so this is one shot's distance, not a
   * zone average.
   */
  shotDistance?: number;
}

// ── Types mirroring the FastAPI response shapes (src/inference/api.py) ──────

export interface HeatmapPoint {
  zone: string;
  loc_x: number;
  loc_y: number;
  /** Coarse: only ever "2PT Field Goal" / "3PT Field Goal". */
  shot_type: string;
  shot_distance: number;
  make_probability: number;
  expected_points: number;
  shot_quality_score: number;
  difficulty_score: number;
  shot_volume?: number;
  confidence?: number;

  // The backend sends these too. They were undeclared, so the UI could not
  // read them — which left the shot map able to show how good a spot is but
  // not what shot it actually is. Optional so an older backend still works.

  /** The shot the model picks here: "stepback", "cutting", "driving", … */
  best_mechanic?: string;
  /** Model confidence in that mechanic, 0-1. */
  best_mechanic_prob?: number;
  /** Expected-points interval, for showing the estimate honestly. */
  ep_low?: number;
  ep_high?: number;
  /** How reachable this shot is for this player in this matchup, 0-1. */
  attainability?: number;
  /** Expected points here relative to the player's own average. */
  ep_vs_own_average?: number;
  /** Historical attempts the estimate is drawn from. */
  attempts_behind?: number;
  score?: number;
  player_name?: string;
  model?: string;
}

/** One trait behind an attainability estimate, from GET /explain/attainability. */
export interface AttainabilityFactor {
  feature: string;
  label: string;
  value: number | null;
  display_value: string;
  /** Approximate league percentile for this player's value, 0-100. */
  percentile: number | null;
  /** Exact TreeSHAP contribution, in attainability share. */
  impact: number;
  direction: "raises" | "lowers";
  detail: string;
}

/**
 * Who generates this player's shots from a spot. Attainability is a frequency
 * and cannot separate a look he manufactures at will from one that only exists
 * when a teammate finds him; this does.
 */
export interface AttainabilityCreation {
  /** Shrunk share of his makes here that were unassisted, 0-1. */
  self_created_share: number | null;
  /** The league's rate for the same zone, for comparison. */
  league_self_created_share: number | null;
  makes: number;
  self_makes: number;
  /** Reader-facing sentence, or null when the sample is too thin to say. */
  note: string | null;
}

export interface AttainabilityExplanation {
  player_id: string;
  player_name?: string;
  season: string;
  zone: string;
  attainability: number;
  /** What a league-average player would get in this zone. */
  baseline: number;
  /** This player's deviation from that baseline. */
  player_effect: number;
  league_zone_share?: number | null;
  position?: string | null;
  factors: AttainabilityFactor[];
  summary: string;
  creation?: AttainabilityCreation;
}

export interface BackendZoneSummary {
  zone: string;
  best_make_prob: number;
  avg_make_prob: number;
  best_ep: number;
  avg_ep: number;
  shot_type: string;
  point_count?: number;
  /**
   * Historical attempts backing the estimate (api.py ZoneSummary). The
   * `attempts` / `actual_fg_pct` fields declared here previously were never
   * emitted by the backend, so any UI reading them silently got `undefined`.
   */
  attempts_behind?: number;
  best_quality?: number;
  shot_distance?: number;
  best_mechanic?: string;
  best_mechanic_prob?: number;
}

export interface ZoneVolume {
  attempts: number;
  makes: number;
  fg_pct: number;
}

export interface HeatmapResponse {
  player_id: string;
  player_name: string;
  season: string;
  heatmap: HeatmapPoint[];
  zone_summary: BackendZoneSummary[];
  shot_volume: Record<string, ZoneVolume>;
  recommendations: HeatmapPoint[];
  grid_size: number;
  // "prior" means one side has no real NBA data yet (e.g. this season hasn't
  // started) and the numbers are a labeled position-based projection.
  attacker_stats_source: "measured" | "prior";
  attacker_resolved_season: string | null;
  defender_stats_source: "measured" | "prior" | null;
  defender_resolved_season: string | null;
}

export interface MatchupPlayerRef {
  player_id: string;
  name: string;
  position: string;
  headshot_url: string;
  stats_source: "measured" | "prior";
  resolved_season: string | null;
}

export interface MatchupPhysicalStat {
  attacker: number | null;
  defender: number | null;
  diff: number | null;
}

export interface MatchupExploitZone {
  zone: string;
  attacker_fg_pct: number | null;
  defender_fg_pct_allowed: number | null;
  matchup_advantage: number | null;
  exploit: boolean;
}

export interface MatchupResponse {
  attacker: MatchupPlayerRef;
  defender: MatchupPlayerRef;
  season: string;
  physical_comparison: {
    height: MatchupPhysicalStat;
    weight: MatchupPhysicalStat;
    wingspan: MatchupPhysicalStat;
  };
  size_mismatch: boolean;
  defender_quality: { fg_pct_allowed: number | null; plus_minus: number | null };
  exploit_zones: MatchupExploitZone[];
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  features: number;
  latest_season: string | null;
}

// runEngine mock removed - now using API

export function heightToFeet(inches: number | null): string {
  if (inches == null) return "—";
  const ft = Math.floor(inches / 12);
  const inc = inches % 12;
  return `${ft}'${inc}"`;
}