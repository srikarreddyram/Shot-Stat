// Pure helpers shared by the Stat Engine page and its comparison view.
// Kept free of React so the ranking/percentile logic — the part that is
// actually easy to get wrong — can be reasoned about and tested on its own.
//
// Everything here operates on the league table the API already sent
// (/stats/players), so a rank or percentile costs no round trip. Column
// labels, formats and grouping come from the server; this file only ever
// interprets values, never names them.

/** Which side of the ball a group belongs on. "info" (bio, ratings) sits in
 *  a persistent header rather than inside either tab — sent by the API
 *  (stat_engine.GROUP_SIDE / career_stats.GROUP_SIDE) so the split is
 *  defined once, not guessed independently by every view that needs it. */
export type Side = "offense" | "defense" | "info";

export type Column = {
  key: string;
  label: string;
  fmt: string;
  group?: string;
  group_label?: string;
  side?: Side;
  derived?: boolean;
  /** Column counting the attempts behind a rate, and the minimum needed to be
   *  ranked. Sent by the API (stat_engine.QUALIFIERS) so the threshold lives
   *  in one place rather than being guessed per view. */
  qualify_key?: string;
  qualify_min?: number;
};

export type Row = Record<string, string | number | null>;

export type Stat = { key: string; label: string; fmt: string; value: string | number | null };
export type Section = { group: string; label: string; side: Side; stats: Stat[] };

export type PlayerProfile = {
  player_id: string;
  name: string;
  team_id: string | null;
  season: string;
  sections: Section[];
};

/** One point in a career trend line: what the stat WAS in that season, not a
 *  running average up to that point. */
export type SeasonPoint = { season: string; value: number };

export type CareerStat = {
  key: string;
  label: string;
  group: string;
  fmt: string;
  kind: string;
  current: number | null;
  previous: number | null;
  career_avg: number | null;
  career_total: number | null;
  first_season: string | null;
  last_season: string | null;
  seasons: number;
  series: SeasonPoint[];
};

export type CareerSection = { group: string; label: string; side: Side; stats: CareerStat[] };

export type CareerPanelData = {
  player_id: string;
  season: string;
  previous_season: string;
  first_season: string | null;
  last_season: string | null;
  seasons_covered: number;
  sections: CareerSection[];
};

export function sideOf<T extends { side?: Side }>(entries: T[], side: Side): T[] {
  return entries.filter((e) => e.side === side);
}

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
  // Career totals run to five figures ("15614 shots defended"), which is
  // unreadable without separators; small counts are unaffected.
  if (fmt === "count") return Math.round(n).toLocaleString("en-US");
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
const DEF_SLUGS = ["rim", "paint", "two_pt", "mid_long", "perimeter"];

export const LOWER_IS_BETTER = new Set([
  "def_fg_pct_allowed", "roster_avg_def_fg_pct_allowed",
  "def_plus_minus",
  "def_rating", // team points allowed per 100 possessions
  "tov",
  // Every per-category defensive rate: allowing a LOWER percentage, and
  // holding shooters further below league normal, is better defence.
  ...DEF_SLUGS.map((s) => `def_${s}_fg_pct`),
  ...DEF_SLUGS.map((s) => `def_${s}_pm`),
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

/**
 * Ascending-sorted values per column, computed once.
 *
 * When a column carries a qualifier, only rows meeting it enter the
 * distribution. A rate built on two attempts is not a worse or better
 * measurement than one built on six hundred — it is not a measurement of the
 * same thing at all, and letting it into the distribution puts whoever
 * defended one shot at #1.
 */
export function buildDistributions(
  rows: Row[],
  keys: string[],
  columns: Column[] = []
): Map<string, number[]> {
  const byKey = new Map(columns.map((c) => [c.key, c]));
  const map = new Map<string, number[]>();
  for (const key of keys) {
    if (map.has(key)) continue;
    const column = byKey.get(key);
    const eligible = qualifies(column) ? rows.filter((r) => meetsQualifier(r, column!)) : rows;
    map.set(key, numericColumn(eligible, key));
  }
  return map;
}

function qualifies(column: Column | undefined): boolean {
  return !!(column?.qualify_key && column.qualify_min != null);
}

/** Does this row have enough volume behind the stat to be ranked on it? */
export function meetsQualifier(row: Row, column: Column): boolean {
  if (!qualifies(column)) return true;
  const volume = numericValue(row[column.qualify_key!]);
  return volume != null && volume >= column.qualify_min!;
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
export type RadarAxis = { key: string; label: string; blurb: string; stats: string[] };

// Axis names must say what they measure, not a word that sounds right.
// "Creation" was read — correctly, by anyone who follows basketball — as
// "creating shots, including for team-mates", and it then looked absurd that
// the league's best passer scored 39 on it. The underlying number was fine:
// self_creation_index is pull-up share, self-created dribble share, dribbles
// per touch and drives per minute (src/features/creation.py's _COMPOSITES),
// with no passing term at all. Creating FOR others is the playmaking axis,
// where that player is 99th percentile. So the axis is now named for what it
// actually is, and every axis carries a blurb naming its inputs.
export const RADAR_AXES: RadarAxis[] = [
  // Zone percentages, not blended FG%. A blended rate confounds shooting
  // SKILL with shot DIET: a player whose diet is pull-up threes converts a
  // lower share of his attempts than one who lives at the rim, however good
  // he is. Comparing like-for-like within each zone removes that. FT% is the
  // one shot with no defender and no diet effect, so it stays as a clean
  // stroke reading.
  { key: "shooting", label: "Shooting", blurb: "Accuracy WITHIN each zone — rim, mid-range, above-the-break three, free throws. Compared like-for-like, so a hard shot diet is not punished",
    stats: ["zone_fg_pct_rim", "zone_fg_pct_midrange", "zone_fg_pct_above_break3", "ft_pct"] },
  // drives_per_min is deliberately absent: it is already 45% of
  // rim_pressure, and listing it again here counted it roughly twice.
  { key: "rim", label: "Rim pressure", blurb: "Getting to the basket and finishing there — drives, drive efficiency, fouls drawn, rim volume",
    stats: ["rim_pressure", "zone_fga_rim"] },
  // Likewise avg_drib_per_touch is 20% of self_creation_index. The composite
  // already blends pull-up share, self-created dribble share, dribbles per
  // touch and drives; it does not need one of its own ingredients re-added
  // as half the axis.
  { key: "creation", label: "Self-creation", blurb: "Generating your OWN shot off the dribble — pull-ups, self-created dribbles, drives. No passing in this one",
    stats: ["self_creation_index"] },
  { key: "playmaking", label: "Playmaking", blurb: "Creating shots for team-mates — assists, potential assists, points created",
    stats: ["playmaking_gravity", "ast"] },
  { key: "activity", label: "Defensive activity", blurb: "Blocks, steals and deflections per minute",
    stats: ["stl_per_min", "blk_per_min", "deflections_per_min"] },
  { key: "impact", label: "Defensive impact", blurb: "FG% you allow, and how far below league normal you hold the shooters you guard",
    stats: ["def_fg_pct_allowed", "def_plus_minus"] },
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
  distributions: Map<string, number[]>,
  columns: Column[] = []
): number | null {
  const byKey = new Map(columns.map((c) => [c.key, c]));
  const parts: number[] = [];
  for (const key of axis.stats) {
    const value = numericValue(row[key]);
    if (value == null) continue;
    // A component built on four attempts is not evidence about this player's
    // shooting; it is skipped rather than allowed to swing the axis.
    const column = byKey.get(key);
    if (column && !meetsQualifier(row, column)) continue;
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

// ── Archetypes ────────────────────────────────────────────────────────────
// "Urban NBA" vocabulary (3&D Wing, Pick-and-Roll Hub, Rim Protector,
// Chucker, ...) computed server-side from measured stats — see
// src/inference/archetypes.py's module docstring for the full reasoning.
// Every label ships with the trait percentiles behind it, so a client can
// always answer "why" rather than presenting the name as a bare verdict.

export type ArchetypeCatalogueEntry = { key: string; label: string; category: string; blurb: string };

export type SecondaryArchetype = { key: string; label: string; category: string; blurb: string; score: number };

export type PlayerArchetype = {
  player_id: string;
  archetype_key: string | null;
  archetype_label: string | null;
  archetype_score: number | null;
  archetype_secondary: SecondaryArchetype[];
};

export type ArchetypesResponse = {
  season: string | null;
  catalogue: ArchetypeCatalogueEntry[];
  players: PlayerArchetype[];
};

export type TraitScore = { key: string; label: string; percentile: number | null; stats: string[] };

export type ArchetypeDetail = {
  player_id: string;
  primary: { key: string; label: string; category: string | null; blurb: string | null; score: number } | null;
  secondary: SecondaryArchetype[];
  qualified: { key: string; label: string; category: string; blurb: string; score: number }[];
  traits: TraitScore[];
  eligible: boolean;
};
