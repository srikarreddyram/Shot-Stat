// Pure helpers shared by the Stat Engine page and its comparison view.
// Kept free of React so the ranking/percentile logic — the part that is
// actually easy to get wrong — can be reasoned about and tested on its own.
//
// Everything here operates on the league table the API already sent
// (/stats/players), so a rank or percentile costs no round trip. Column
// labels, formats and grouping come from the server; this file only ever
// interprets values, never names them.

export type Column = {
  key: string;
  label: string;
  fmt: string;
  group?: string;
  group_label?: string;
  derived?: boolean;
};

export type Row = Record<string, string | number | null>;

export type Stat = { key: string; label: string; fmt: string; value: string | number | null };
export type Section = { group: string; label: string; stats: Stat[] };

export type PlayerProfile = {
  player_id: string;
  name: string;
  team_id: string | null;
  season: string;
  sections: Section[];
};

export type RosterEntry = {
  player_id: string;
  name: string;
  position: string | null;
  off_rating: number | null;
  def_rating_ours: number | null;
};

export type TeamProfile = {
  team_id: string;
  team_name: string | null;
  team_abbrev: string | null;
  def_rating: number | null;
  season: string;
  roster: RosterEntry[];
};

// ── Formatting ──────────────────────────────────────────────────────────────
// `fmt` is the backend's vocabulary (stat_engine.STAT_GROUPS): "pct" is a
// 0-1 proportion, "in" is inches, "count" a whole-number tally, "text" a
// string, anything else a plain number.

// rating_source carries the engine's own identifiers ("measured",
// "2k_fallback") — the same strings EnginePage keys its 2K badge off, so
// they stay as they are on the wire and get a reader-facing name here.
const VALUE_LABELS: Record<string, string> = {
  measured: "Measured from play-by-play",
  "2k_fallback": "NBA 2K (no measured rating)",
};

export function formatStat(value: string | number | null | undefined, fmt: string): string {
  if (value == null || value === "") return "—";
  if (fmt === "text") return VALUE_LABELS[String(value)] ?? String(value);
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (fmt === "pct") return `${(n * 100).toFixed(1)}%`;
  // Round to whole inches BEFORE splitting, or 83.75" renders as 6'12".
  if (fmt === "in") {
    const inches = Math.round(n);
    return `${Math.floor(inches / 12)}'${inches % 12}"`;
  }
  if (fmt === "count") return n.toFixed(0);
  return Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(2);
}

export function numericValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// ── Direction ───────────────────────────────────────────────────────────────
// Stats where a LOWER number is the better performance, so a percentile or
// rank has to be inverted or it will paint a lockdown defender as terrible.
//
// Note what is NOT here: off_rating and def_rating_ours are our own
// percentile BANDS (player_ratings.to_rating_band), where 99 is the best
// defender in the league, not the worst — inverting those would rank the
// league upside down. Same for every roster_avg_ of them. Only genuinely
// "fewer is better" raw measurements belong in this set.
export const LOWER_IS_BETTER = new Set([
  "def_fg_pct_allowed", "roster_avg_def_fg_pct_allowed",
  "def_plus_minus",
  "def_rating", // team points allowed per 100 possessions
  "tov",
]);

// Stats with no meaningful better/worse direction at all. Being tall is not
// an achievement and neither is taking most of your shots off the catch —
// ranking these implies a verdict the number does not support.
export const DIRECTIONLESS = new Set([
  "height", "weight", "wingspan",
  "catch_shoot_share", "pullup_share", "open_share", "tight_share",
  "roster_size",
]);

export function isRankable(key: string, fmt: string): boolean {
  return fmt !== "text" && !DIRECTIONLESS.has(key);
}

// ── League context ──────────────────────────────────────────────────────────
// Ascending list of the values that actually exist for a column. Missing
// values are excluded rather than treated as zero: "we never measured this"
// and "this player does none of it" are different claims, and conflating
// them ranks every un-tracked player last.
export function numericColumn(rows: Row[], key: string): number[] {
  const values: number[] = [];
  for (const r of rows) {
    const v = numericValue(r[key]);
    if (v != null) values.push(v);
  }
  return values.sort((a, b) => a - b);
}

/** Fraction of the league this value beats, 0-1, already direction-corrected. */
export function percentileOf(value: number, sorted: number[], key: string): number | null {
  if (sorted.length < 5) return null;
  let below = 0;
  for (const v of sorted) {
    if (v < value) below++;
    else break;
  }
  const raw = below / sorted.length;
  return LOWER_IS_BETTER.has(key) ? 1 - raw : raw;
}

/** 1-based league rank, where 1 is best. Ties share the better rank. */
export function rankOf(
  value: number,
  sorted: number[],
  key: string
): { rank: number; outOf: number } | null {
  if (sorted.length === 0) return null;
  const lower = LOWER_IS_BETTER.has(key);
  let better = 0;
  if (lower) {
    for (const v of sorted) {
      if (v < value) better++;
      else break;
    }
  } else {
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (sorted[i] > value) better++;
      else break;
    }
  }
  return { rank: better + 1, outOf: sorted.length };
}

/** Ascending-sorted values for every column named, computed once. */
export function buildDistributions(rows: Row[], keys: string[]): Map<string, number[]> {
  const map = new Map<string, number[]>();
  for (const key of keys) {
    if (!map.has(key)) map.set(key, numericColumn(rows, key));
  }
  return map;
}

// ── Search ──────────────────────────────────────────────────────────────────
// Names in the database carry real diacritics and people type them without.
// NFD splits a letter from its accent and the combining-mark range is then
// strippable, so a plain-ASCII query matches an accented name.
//
// The pattern is built from character codes rather than written as a regex
// literal for U+0300-U+036F because pasting the literal escape sequence into
// a source file has, in this codebase, been silently rewritten by editing
// tooling into something that no longer matched. This form is inert text
// until it runs. EnginePage.tsx carries its own copy on purpose — that file
// is the core recommendation flow and is deliberately not coupled to this
// one.
const DIACRITIC_MARKS = new RegExp(
  String.fromCharCode(0x5b, 0x5c, 0x75, 0x30, 0x33, 0x30, 0x30, 0x2d, 0x5c, 0x75, 0x30, 0x33, 0x36, 0x66, 0x5d),
  "g"
);

export function foldAccents(s: string): string {
  return s.normalize("NFD").replace(DIACRITIC_MARKS, "");
}

export function matchesQuery(haystack: string, foldedQuery: string): boolean {
  return foldAccents(haystack.toLowerCase()).includes(foldedQuery);
}

// ── Sorting ─────────────────────────────────────────────────────────────────
// Missing values always sink to the bottom regardless of direction. A null is
// "we don't have this", not a very small number, so it should never win a
// "worst in the league" sort either.
export function sortRows(rows: Row[], key: string, dir: "asc" | "desc"): Row[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = a[key], bv = b[key];
    const aMissing = av == null || av === "";
    const bMissing = bv == null || bv === "";
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * sign;
    return String(av).localeCompare(String(bv)) * sign;
  });
}

// ── Radar axes ──────────────────────────────────────────────────────────────
// Each axis is the unweighted MEAN OF PERCENTILE RANKS of its member stats —
// not a rating, and deliberately not a weighted formula. This project has
// already been bitten once by a frontend "rating" assembled from invented
// weights and clamps (see the note at the top of src/lib/api.ts), where the
// number looked authoritative and meant nothing. An unweighted mean of
// percentiles over a short, named list of measured stats is a summary the
// reader can check, and every component stat is shown in full underneath.
export type RadarAxis = { key: string; label: string; stats: string[] };

export const RADAR_AXES: RadarAxis[] = [
  { key: "shooting", label: "Shooting", stats: ["season_fg_pct", "career_3p_pct", "ft_pct"] },
  { key: "rim", label: "Rim pressure", stats: ["rim_pressure", "drives_per_min", "zone_fga_rim"] },
  { key: "creation", label: "Creation", stats: ["self_creation_index", "avg_drib_per_touch"] },
  { key: "playmaking", label: "Playmaking", stats: ["playmaking_gravity", "ast"] },
  { key: "activity", label: "Defensive activity", stats: ["stl_per_min", "blk_per_min", "deflections_per_min"] },
  { key: "impact", label: "Defensive impact", stats: ["def_fg_pct_allowed", "def_plus_minus"] },
];

/**
 * A player's 0-1 score on one axis, or null when we measured none of its
 * component stats for them. A partial axis uses the components we do have
 * rather than dropping the player — but "none at all" stays null instead of
 * becoming a confident zero.
 */
export function axisScore(
  row: Row,
  axis: RadarAxis,
  distributions: Map<string, number[]>
): number | null {
  const parts: number[] = [];
  for (const key of axis.stats) {
    const value = numericValue(row[key]);
    if (value == null) continue;
    const pct = percentileOf(value, distributions.get(key) ?? [], key);
    if (pct != null) parts.push(pct);
  }
  if (parts.length === 0) return null;
  return parts.reduce((a, b) => a + b, 0) / parts.length;
}

/** Every stat key any radar axis depends on — for pre-building distributions. */
export const RADAR_STAT_KEYS = Array.from(new Set(RADAR_AXES.flatMap((a) => a.stats)));

// A fixed per-slot palette for comparison. Distinct hues rather than a
// lightness ramp: the players being compared are categories, not magnitudes.
export const COMPARE_COLORS = ["#C9A84C", "#4C9ED9", "#5CB85C"];
