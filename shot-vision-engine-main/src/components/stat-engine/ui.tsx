import { ReactNode } from "react";

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
        background: `radial-gradient(130% 150% at 88% 0%, ${color}2E 0%, ${color}0A 42%, transparent 72%)`,
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
            opacity: 0.075, filter: "saturate(1.4)",
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
        background: `radial-gradient(80% 55% at 50% 0%, ${color}1C 0%, transparent 68%)`,
      }}
    />
  );
}
