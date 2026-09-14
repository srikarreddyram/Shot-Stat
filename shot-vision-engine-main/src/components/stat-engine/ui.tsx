import { ReactNode } from "react";
import { formatStat, SeasonPoint } from "../../lib/stat-engine";

// Shared visual language for the Stat Engine. The tokens here are taken from
// EnginePage.tsx rather than invented, so /stats reads as the same product as
// the engine itself — the page previously drifted onto its own near-miss
// palette (#0B0B0F vs #0a0a0f, #15151D vs #0e0e16), which is the kind of gap
// that looks like a rendering bug rather than a design choice.

// The accent is a CSS custom property, not a constant, so opening a team or
// player profile can re-theme the whole page to that team's colours without
// threading an accent prop through every component. accentVars() below
// produces the values; DEFAULT_ACCENT is SHOT VISION's own gold.
export const DEFAULT_ACCENT = "#C9A84C";

export const C = {
  bg: "#0a0a0f",
  surface: "#0e0e16",
  raised: "#16161f",
  hover: "var(--se-hover)",
  edge: "var(--se-edge)",
  edgeStrong: "var(--se-edge-strong)",
  rule: "rgba(255,255,255,0.04)",
  text: "#F0F0F0",
  dim: "rgba(240,240,240,0.62)",
  faint: "rgba(240,240,240,0.38)",
  muted: "#64748b",
  /** The active accent. Follows the team in a profile; gold elsewhere. */
  gold: "var(--se-accent)",
  red: "#DC2626",
};

/** Inline style block establishing the accent for a subtree. */
export function accentVars(accent: string): React.CSSProperties {
  return {
    // Eight-digit hex: the last byte is alpha, so one base colour yields the
    // whole set of tints without a colour library.
    "--se-accent": accent,
    "--se-edge": `${accent}1A`,
    "--se-edge-strong": `${accent}3D`,
    "--se-hover": `${accent}14`,
    "--se-on-accent": readableOn(accent),
  } as React.CSSProperties;
}

/** Black or white, whichever is legible on `hex`. Several teams' colours are
 *  dark enough (Boston green, Dallas blue) that fixed black text on an accent
 *  fill would be unreadable. */
export function readableOn(hex: string): string {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  // Relative luminance, per WCAG.
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return L > 0.45 ? "#000" : "#fff";
}

export const F = {
  display: "'Bebas Neue', sans-serif",
  mono: "'JetBrains Mono', monospace",
  body: "'Inter', sans-serif",
};

// Numerals must be tabular or a column of figures will not line up, which is
// what makes a dense stat table readable at a glance rather than a wall.
export const NUM: React.CSSProperties = {
  fontFamily: F.mono,
  fontVariantNumeric: "tabular-nums",
};

// ── Quality tiers ───────────────────────────────────────────────────────────
// A five-step ramp over league percentile. This is the single thing that makes
// a dense table scannable: the reader sees colour before they read a number.
// Gold sits in the middle as "average" on purpose — it is the brand accent, so
// an average player reads as neutral rather than as a judgement.
const TIERS = [
  { min: 0.9, color: "#16A34A", label: "Elite" },
  { min: 0.7, color: "#84CC16", label: "Strong" },
  { min: 0.4, color: "#C9A84C", label: "Average" },
  { min: 0.2, color: "#EA580C", label: "Below average" },
  { min: 0, color: "#DC2626", label: "Poor" },
];

export function tier(pct: number | null | undefined) {
  if (pct == null) return null;
  return TIERS.find((t) => pct >= t.min) ?? TIERS[TIERS.length - 1];
}

export function tierColor(pct: number | null | undefined): string {
  return tier(pct)?.color ?? C.muted;
}

// ── Primitives ──────────────────────────────────────────────────────────────

export function Card({ children, accent, style }: {
  children: ReactNode;
  accent?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.edge}`,
      borderLeft: accent ? `3px solid ${accent}` : `1px solid ${C.edge}`,
      borderRadius: 6,
      ...style,
    }}>
      {children}
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{
      fontFamily: F.mono, fontSize: 10, color: C.gold,
      letterSpacing: "0.4em", textTransform: "uppercase",
    }}>
      {children}
    </div>
  );
}

/** Segmented control. Active segment switches to the display face, matching
 *  the engine's mode buttons, so the selected state reads at a glance. */
export function Segmented({ options, value, onChange }: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "inline-flex", background: C.surface, border: `1px solid ${C.edge}`, borderRadius: 4, padding: 3, gap: 3 }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            aria-pressed={active}
            style={{
              background: active ? C.gold : "transparent",
              color: active ? "var(--se-on-accent, #000)" : C.dim,
              border: "none", borderRadius: 2,
              padding: active ? "7px 18px" : "7px 18px",
              cursor: "pointer",
              fontFamily: active ? F.display : F.mono,
              fontSize: active ? 15 : 10,
              letterSpacing: active ? "0.06em" : "0.22em",
              lineHeight: active ? "16px" : "16px",
              transition: "all 200ms",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Smaller pill row, for the within-view column-group chooser. */
export function Chips({ options, value, onChange }: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            aria-pressed={active}
            style={{
              background: active ? C.hover : "transparent",
              color: active ? C.gold : C.muted,
              border: `1px solid ${active ? C.edgeStrong : C.edge}`,
              borderRadius: 3, padding: "6px 13px", cursor: "pointer",
              fontFamily: F.mono, fontSize: 9.5, letterSpacing: "0.2em",
              textTransform: "uppercase", transition: "all 200ms",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Percentile bar. Width is the percentile; colour is its tier. */
export function Bar({ pct, color, height = 3 }: { pct: number; color?: string; height?: number }) {
  return (
    <div style={{ flex: 1, height, background: "rgba(255,255,255,0.05)", borderRadius: height / 2, overflow: "hidden" }}>
      <div style={{
        width: `${Math.max(2, pct * 100)}%`, height: "100%",
        background: color ?? tierColor(pct), borderRadius: height / 2,
        transition: "width 300ms",
      }} />
    </div>
  );
}

/** League rank, as a tier-tinted pill: "#74 / 582". */
export function RankPill({ rank, outOf, pct }: { rank: number; outOf: number; pct: number | null }) {
  const color = tierColor(pct);
  return (
    <span style={{
      ...NUM, fontSize: 9.5, color,
      background: `${color}1A`, border: `1px solid ${color}33`,
      borderRadius: 2, padding: "1px 5px", whiteSpace: "nowrap",
    }}>
      #{rank}
      <span style={{ color: C.muted, marginLeft: 3 }}>/{outOf}</span>
    </span>
  );
}

export function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      style={{
        color: C.muted, fontFamily: F.mono, fontSize: 11,
        letterSpacing: "0.3em", textDecoration: "none", transition: "color 200ms",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.color = C.gold)}
      onMouseLeave={(e) => (e.currentTarget.style.color = C.muted)}
    >
      {children}
    </a>
  );
}

export function GhostButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "transparent", border: `1px solid ${C.edge}`, borderRadius: 3,
        padding: "7px 16px", color: C.muted, cursor: "pointer",
        fontFamily: F.mono, fontSize: 10, letterSpacing: "0.25em", transition: "all 200ms",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.color = C.gold; e.currentTarget.style.borderColor = C.edgeStrong; }}
      onMouseLeave={(e) => { e.currentTarget.style.color = C.muted; e.currentTarget.style.borderColor = C.edge; }}
    >
      {children}
    </button>
  );
}

export function SearchInput({ value, onChange, placeholder, width = 320 }: {
  value: string; onChange: (v: string) => void; placeholder: string; width?: number | string;
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      style={{
        flex: `0 1 ${typeof width === "number" ? `${width}px` : width}`,
        background: C.surface, border: `1px solid ${C.edge}`, borderRadius: 3,
        padding: "9px 13px", color: C.text, fontFamily: F.body, fontSize: 13,
        outline: "none", transition: "border-color 200ms",
      }}
      onFocus={(e) => (e.currentTarget.style.borderColor = C.edgeStrong)}
      onBlur={(e) => (e.currentTarget.style.borderColor = C.edge)}
    />
  );
}

/** Shaped placeholder rows, so arriving content does not shift the layout the
 *  way a single "Loading…" line does. */
export function TableSkeleton({ rows = 10 }: { rows?: number }) {
  return (
    <Card style={{ overflow: "hidden" }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          style={{
            display: "flex", gap: 16, padding: "11px 14px",
            borderBottom: `1px solid ${C.rule}`, alignItems: "center",
            opacity: 1 - i * (0.6 / rows),
          }}
        >
          <div style={{ width: 22, height: 9, background: "rgba(255,255,255,0.05)", borderRadius: 2 }} />
          <div style={{ width: 150, height: 9, background: "rgba(255,255,255,0.07)", borderRadius: 2 }} />
          <div style={{ flex: 1 }} />
          {[54, 54, 54, 54].map((w, j) => (
            <div key={j} style={{ width: w, height: 9, background: "rgba(255,255,255,0.05)", borderRadius: 2 }} />
          ))}
        </div>
      ))}
    </Card>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div style={{ padding: "44px 24px", textAlign: "center", color: C.muted, fontFamily: F.mono, fontSize: 11, letterSpacing: "0.18em" }}>
      {children}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <Card accent={C.red} style={{ padding: "16px 20px", marginBottom: 16 }}>
      <div style={{ fontFamily: F.body, fontWeight: 600, fontSize: 14, color: C.text }}>
        The engine didn't answer
      </div>
      <div style={{ fontFamily: F.mono, fontSize: 11.5, color: C.muted, marginTop: 5, lineHeight: 1.6 }}>
        {message} — check that ./scripts/dev.sh is still running.
      </div>
    </Card>
  );
}

// ── NBA team identity ───────────────────────────────────────────────────────
// Official team colours, keyed by the abbreviation the API already returns.
//
// Several teams' true primary is near-black or a very dark navy (Brooklyn,
// Denver, Milwaukee, Utah, Minnesota) and would vanish against this app's
// background, so for those the entry is the team's official SECONDARY —
// Denver's gold, Milwaukee's cream, Utah's yellow. The rule is legibility on
// a dark ground, not slavish primary-first, because a colour nobody can see
// identifies nothing.
export const TEAM_COLORS: Record<string, string> = {
  ATL: "#E03A3E", BOS: "#007A33", BKN: "#C4CED4", CHA: "#00788C",
  CHI: "#CE1141", CLE: "#C8365A", DAL: "#0076CE", DEN: "#FEC524",
  DET: "#C8102E", GSW: "#FFC72C", HOU: "#CE1141", IND: "#FDBB30",
  LAC: "#C8102E", LAL: "#FDB927", MEM: "#7B9BD1", MIA: "#D4245C",
  MIL: "#EEE1C6", MIN: "#78BE20", NOP: "#B4975A", NYK: "#F58426",
  OKC: "#007AC1", ORL: "#0077C0", PHI: "#3B84D8", PHX: "#E56020",
  POR: "#E03A3E", SAC: "#8A63B8", SAS: "#C4CED4", TOR: "#CE1141",
  UTA: "#F9A01B", WAS: "#E31837",
};

export function teamColor(abbrev: string | null | undefined): string {
  if (!abbrev) return C.muted;
  return TEAM_COLORS[abbrev.toUpperCase()] ?? C.muted;
}

/** A team's abbreviation in its own colours — the cheapest way to make a
 *  league-wide table read as basketball rather than as a spreadsheet. */
export function TeamChip({ abbrev, size = "sm" }: { abbrev: string | null | undefined; size?: "sm" | "lg" }) {
  if (!abbrev) return <span style={{ color: C.muted, fontFamily: F.mono, fontSize: 10.5 }}>—</span>;
  const color = teamColor(abbrev);
  const lg = size === "lg";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: lg ? 7 : 5,
      fontFamily: F.mono, fontSize: lg ? 11 : 10.5,
      letterSpacing: "0.18em", color,
    }}>
      <span style={{
        width: 3, height: lg ? 13 : 11, borderRadius: 1,
        background: color, flexShrink: 0,
      }} />
      {abbrev.toUpperCase()}
    </span>
  );
}

// A half-court arc, echoing the peach hardwood of the engine's shot map, sat
// behind the page header at very low opacity. Enough to read as basketball;
// not enough to compete with the data.
export function CourtWatermark() {
  return (
    <>
      {/* A wash of the shot map's hardwood, full width. */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          background: "linear-gradient(180deg, rgba(220,180,137,0.045) 0%, rgba(220,180,137,0) 100%)",
        }}
      />
      {/* The arcs are drawn at a FIXED size and centred rather than stretched
          across the header. Scaling them to the full width blew the radii up
          until the lane lines read as two hard vertical strokes through the
          middle of the chrome rather than as a court. */}
      <svg
        viewBox="0 0 460 76"
        width="460"
        height="76"
        aria-hidden="true"
        style={{
          position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)",
          pointerEvents: "none",
        }}
      >
        {/* Radii are sized so BOTH arcs close inside the header's height.
            An arc wider than the box gets clipped flat by overflow:hidden and
            stops reading as a circle. And no baseline rule here — a line only
            as wide as this SVG ends abruptly mid-header; the header's own
            border-bottom already carries the full width. */}
        <g fill="none" stroke="#DCB489" strokeOpacity="0.14" strokeWidth="1">
          <circle cx="230" cy="76" r="30" />
          <circle cx="230" cy="76" r="62" />
        </g>
      </svg>
    </>
  );
}

/** Team logo, served by our own API, which fetches it from NBA's CDN once and
 *  caches it (see /team/{id}/logo in src/inference/api.py).
 *
 *  Pointing an <img> straight at cdn.nba.com does NOT work reliably: the CDN
 *  serves a plain HTTP client fine but rejects browser requests with
 *  ERR_HTTP2_PROTOCOL_ERROR. Proxying also keeps NBA's marks out of the repo
 *  and makes the logos work with no network once cached. Every use must still
 *  tolerate the image not arriving. */
const LOGO_API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export function teamLogoUrl(teamId: string | null | undefined): string | null {
  if (!teamId) return null;
  return `${LOGO_API_BASE}/team/${encodeURIComponent(String(teamId))}/logo`;
}

/** The league's own wordmark — same caching proxy as team logos (see
 *  /league/logo in src/inference/api.py), used to brand app chrome rather
 *  than any single team. */
export function leagueLogoUrl(): string {
  return `${LOGO_API_BASE}/league/logo`;
}

// A red/white/blue palette echoing the league's own wordmark colours,
// layered in as a SECOND accent alongside this app's established gold —
// gold stays the primary "brand" colour (buttons, ratings, highlights);
// these show up as structural/ambient touches (dividers, glows, home/away)
// so the UI reads as "NBA" rather than as a single unbranded gold app.
export const NBA_RED = "#C8102E";
export const NBA_BLUE = "#1D428A";

/** A persistent red/blue frame down both edges of the viewport — the one
 *  piece of NBA branding every page in the app shares, team-themed pages
 *  included, so the app reads as "one NBA product" rather than a themed
 *  page here and an unbranded one there. Fixed position, so it stays put
 *  through scrolling; pointer-events none, so it never steals a click. */
export function NBAEdgeBezel() {
  return (
    <>
      <div aria-hidden="true" style={{ position: "fixed", left: 0, top: 0, bottom: 0, width: 4, background: NBA_RED, opacity: 0.55, zIndex: 40, pointerEvents: "none" }} />
      <div aria-hidden="true" style={{ position: "fixed", right: 0, top: 0, bottom: 0, width: 4, background: NBA_BLUE, opacity: 0.55, zIndex: 40, pointerEvents: "none" }} />
    </>
  );
}

/** The default page wash for anywhere with no specific team to theme
 *  against — red top-left, blue top-right. Any page with a real team in
 *  play (a team profile, a player's team) should use that team's own
 *  TeamPageTint instead; this is what fills the gap everywhere else so the
 *  app is never left plain gold-on-black. */
export function NBADefaultWash() {
  return (
    <div aria-hidden="true" style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
      <div style={{ position: "absolute", inset: 0, background: `radial-gradient(65% 55% at 8% 0%, ${NBA_RED}2E 0%, transparent 62%)` }} />
      <div style={{ position: "absolute", inset: 0, background: `radial-gradient(65% 55% at 92% 0%, ${NBA_BLUE}38 0%, transparent 62%)` }} />
    </div>
  );
}

/** The league wordmark plus a full-strength red/white/blue stripe, sized to
 *  sit next to a page's own title in its header. One consistent brand unit
 *  reused by every page's header rather than each re-implementing it. */
export function LeagueBrandMark({ height = 30 }: { height?: number }) {
  return (
    <img
      src={leagueLogoUrl()}
      alt=""
      aria-hidden="true"
      onError={(e) => { e.currentTarget.style.display = "none"; }}
      style={{ height, width: "auto" }}
    />
  );
}

/** `bottom` is a raw CSS offset (default flush with the header's own bottom
 *  edge); pass a negative value to hang the stripe just below a header that
 *  already has its own bottom border, so the two don't overlap. */
export function NBAHeaderStripe({ bottom = 0 }: { bottom?: number | string }) {
  return (
    <div aria-hidden="true" style={{ position: "absolute", left: 0, right: 0, bottom, height: 5, display: "flex" }}>
      <div style={{ flex: 1, background: NBA_RED }} />
      <div style={{ flex: 1, background: "#F0F0F0" }} />
      <div style={{ flex: 1, background: NBA_BLUE }} />
    </div>
  );
}

/** A team's colour wash plus its logo, sat behind a profile hero.
 *  Decorative only: aria-hidden, non-interactive, and it renders correctly
 *  with no logo at all if the CDN does not answer. */
export function TeamBackdrop({ teamId, abbrev, height = 320 }: {
  teamId: string | null | undefined;
  abbrev: string | null | undefined;
  height?: number;
}) {
  const color = teamColor(abbrev);
  const src = teamLogoUrl(teamId);
  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute", inset: 0, overflow: "hidden",
        pointerEvents: "none", borderRadius: 6,
      }}
    >
      <div style={{
        position: "absolute", inset: 0,
        background: `radial-gradient(130% 150% at 88% 0%, ${color}45 0%, ${color}16 42%, transparent 72%)`,
      }} />
      {src && (
        <img
          src={src}
          alt=""
          loading="lazy"
          // A missing logo must leave the wash looking deliberate rather than
          // leaving a broken-image glyph over the hero.
          onError={(e) => { e.currentTarget.style.display = "none"; }}
          // Cropped at the edge and held faint on purpose: the hero's small
          // mono labels sit over this, and a busy mark (Denver's ring, Boston's
          // leprechaun) renders 9px text unreadable long before it stops being
          // recognisable as a logo.
          style={{
            position: "absolute", right: -height * 0.22, top: "50%",
            transform: "translateY(-50%)", height, width: "auto",
            maxWidth: "none", maxHeight: "none",
            opacity: 0.16, filter: "saturate(1.4)",
          }}
        />
      )}
    </div>
  );
}

/** Full-page tint for a themed profile — a faint column of team colour down
 *  the page behind the cards, so the theming is not confined to the hero. */
export function TeamPageTint({ abbrev }: { abbrev: string | null | undefined }) {
  const color = teamColor(abbrev);
  return (
    <div
      aria-hidden="true"
      style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
        background: `radial-gradient(80% 55% at 50% 0%, ${color}38 0%, transparent 68%)`,
      }}
    />
  );
}

/** Player headshot, served by our own API (see /player/{id}/headshot), which
 *  fetches from NBA's CDN once and caches it — same reasoning as the team
 *  logo above.
 *
 *  "small" is a ~15 KB 260x190; "large" is a ~200 KB 1040x760. A table of 582
 *  rows must use "small", or the page pulls well over a hundred megabytes. */
export function playerHeadshotUrl(playerId: string | null | undefined, size: "small" | "large" = "small"): string | null {
  if (!playerId) return null;
  return `${LOGO_API_BASE}/player/${encodeURIComponent(String(playerId))}/headshot?size=${size}`;
}

export function PlayerAvatar({ playerId, teamAbbrev, size = 28, variant = "small" }: {
  playerId: string | null | undefined;
  teamAbbrev?: string | null;
  size?: number;
  variant?: "small" | "large";
}) {
  const src = playerHeadshotUrl(playerId, variant);
  const ring = teamAbbrev ? teamColor(teamAbbrev) : C.edgeStrong;
  return (
    <span
      style={{
        display: "inline-block", width: size, height: size, borderRadius: "50%",
        // The headshots are cut-outs on a transparent ground, so they need a
        // filled circle behind them or they read as floating heads.
        background: `linear-gradient(160deg, ${ring}33, ${C.raised})`,
        border: `1px solid ${ring}55`,
        overflow: "hidden", flexShrink: 0, position: "relative",
      }}
    >
      {src && (
        <img
          src={src}
          alt=""
          loading="lazy"
          decoding="async"
          // A missing photo must leave the tinted circle, not a broken glyph.
          onError={(e) => { e.currentTarget.style.display = "none"; }}
          style={{
            // The source is a 4:3 crop with the head high in the frame;
            // scaling up and nudging down centres the face in a circle.
            //
            // maxWidth/maxHeight: "none" is load-bearing, not decoration. A
            // global reset (Tailwind preflight's img{max-width:100%}) caps
            // this element's WIDTH at the parent's 100% while leaving HEIGHT
            // free to reach the full 132% — an asymmetric clamp that turns a
            // square 132%x132% crop box into a narrow, too-tall one. The
            // marginLeft below was computed assuming a symmetric box, so
            // every headshot rendered visibly shifted (too much of the right
            // side of the face showing, background visible on the left).
            // Confirmed live: computed width capped at 84px against an
            // uncapped height of 110.875px in an 86px box, both meant to be
            // 113.5px. Overriding the cap here restores the symmetric box
            // the centering math actually assumes.
            maxWidth: "none", maxHeight: "none",
            width: "132%", height: "132%", objectFit: "cover",
            objectPosition: "50% 12%", marginLeft: "-16%", marginTop: "-6%",
            display: "block",
          }}
        />
      )}
    </span>
  );
}

// ── Career trend chart ───────────────────────────────────────────────────────
// A season-by-season line, not a sparkline: labelled axes, a dot per real
// season (never interpolated across a gap — an SVG polyline drawn straight
// through a missing season would imply a value that was never measured), and
// the actual season labels along the bottom so "when" is never ambiguous.


const TREND_W = 460;
const TREND_H = 150;
const TREND_PAD = { top: 14, right: 14, bottom: 22, left: 8 };

export function TrendChart({ points, fmt, color = C.gold, label }: {
  points: SeasonPoint[];
  fmt: string;
  color?: string;
  label?: string;
}) {
  if (points.length === 0) {
    return (
      <div style={{ height: TREND_H, display: "flex", alignItems: "center", justifyContent: "center", color: C.muted, fontSize: 11 }}>
        No seasons measured
      </div>
    );
  }

  const values = points.map((p) => p.value);
  const lo = Math.min(...values), hi = Math.max(...values);
  // A dead-flat series (every season identical) needs an artificial span or
  // every point lands on the same horizontal line at 50% — still correct,
  // just given a hair of headroom so the line doesn't hug the chart edges.
  const span = hi - lo || Math.abs(hi || 1) * 0.1 || 1;
  const padSpan = span * 0.18;
  const yMin = lo - padSpan, yMax = hi + padSpan;

  const innerW = TREND_W - TREND_PAD.left - TREND_PAD.right;
  const innerH = TREND_H - TREND_PAD.top - TREND_PAD.bottom;
  const x = (i: number) => TREND_PAD.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
  const y = (v: number) => TREND_PAD.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(p.value).toFixed(1)}`).join(" ");
  const area = `${path} L ${x(points.length - 1).toFixed(1)} ${(TREND_PAD.top + innerH).toFixed(1)} `
    + `L ${x(0).toFixed(1)} ${(TREND_PAD.top + innerH).toFixed(1)} Z`;

  // Thinning season labels when there are many, so they never overlap —
  // every point still gets a dot, just not every point gets a printed label.
  const labelEvery = Math.max(1, Math.ceil(points.length / 6));

  return (
    <div>
      {label && <div style={{ fontSize: 11.5, color: C.dim, marginBottom: 6 }}>{label}</div>}
      <svg viewBox={`0 0 ${TREND_W} ${TREND_H}`} style={{ width: "100%", height: "auto", display: "block" }}>
        <defs>
          <linearGradient id={`trend-fill-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* Zero-ish baseline only when zero actually sits inside the visible
            range — drawing it outside that range would just be a stray line
            at the chart's edge. */}
        {yMin < 0 && yMax > 0 && (
          <line x1={TREND_PAD.left} x2={TREND_W - TREND_PAD.right} y1={y(0)} y2={y(0)}
                stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />
        )}
        <path d={area} fill={`url(#trend-fill-${color.replace("#", "")})`} />
        <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => (
          <circle key={p.season} cx={x(i)} cy={y(p.value)} r={i === points.length - 1 ? 3.5 : 2.5}
                  fill={i === points.length - 1 ? color : C.bg} stroke={color} strokeWidth={1.5} />
        ))}
        {points.map((p, i) => (
          (i % labelEvery === 0 || i === points.length - 1) && (
            <text key={p.season} x={x(i)} y={TREND_H - 6} textAnchor="middle"
                  fill={C.muted} style={{ fontFamily: F.mono, fontSize: 8.5 }}>
              {p.season.slice(2, 5)}
            </text>
          )
        ))}
        {/* The current (rightmost) value, printed at its own point. */}
        <text x={x(points.length - 1)} y={y(points[points.length - 1].value) - 9} textAnchor="middle"
              fill={color} style={{ fontFamily: F.mono, fontSize: 10.5, fontWeight: 700 }}>
          {formatStat(points[points.length - 1].value, fmt)}
        </text>
      </svg>
    </div>
  );
}
