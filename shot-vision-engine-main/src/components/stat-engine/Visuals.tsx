import { useEffect, useState } from "react";
import { ArchetypeDetail, PlayerProfile, formatStat, numericValue } from "../../lib/stat-engine";
import { Bar, C, F, NUM, Card, SectionLabel, tierColor } from "./ui";

// Three season-view visuals, replacing what used to be three more rows in a
// stat table: a shot chart by zone, a Synergy-style play-type bar chart, and
// a defensive FG%-allowed-by-category chart. Each reads straight off the
// PlayerProfile the page already fetched — no extra request.

function findStat(profile: PlayerProfile, key: string) {
  for (const section of profile.sections) {
    const hit = section.stats.find((s) => s.key === key);
    if (hit) return hit;
  }
  return null;
}

// ── Shot chart by zone ───────────────────────────────────────────────────────
// Court geometry duplicated from EnginePage.tsx's CourtCanvas / archetypes.tsx
// on purpose (see those files' own notes) — a small amount of repetition here
// is the deliberate trade for zero risk to the core recommendation flow this
// geometry also drives. Peach hardwood, heat as a tint rather than a cover,
// matching the shot map's established look.
const FT = {
  courtWidth: 50, hoopFromBaseline: 5.25, rimRadius: 0.75, restrictedRadius: 4,
  laneWidth: 16, laneDepth: 19, ftCircleRadius: 6, cornerThreeX: 22,
  arcRadius: 23.75, visibleDepth: 32,
};
const CORNER_JOIN_Y = Math.sqrt(FT.arcRadius ** 2 - FT.cornerThreeX ** 2);
const COURT_CW = 420;
const COURT_SCALE = COURT_CW / FT.courtWidth;
const COURT_CH = Math.round(FT.visibleDepth * COURT_SCALE);
const HOOP_X = COURT_CW / 2;
const HOOP_Y = COURT_CH - FT.hoopFromBaseline * COURT_SCALE;
const ft = (feet: number) => feet * COURT_SCALE;
const pt = (x: number, y: number) => `${(HOOP_X + ft(x)).toFixed(1)},${(HOOP_Y - ft(y)).toFixed(1)}`;

const COURT_SURFACE = "#DCB489";
const COURT_LINE = "rgba(92, 58, 30, 0.5)";
const COURT_LINE_STRONG = "rgba(58, 35, 16, 0.85)";
const COURT_RIM = "#B4491C";

// Where each zone's badge sits on the court, in feet from the hoop.
const ZONE_BADGE_POS: Record<string, [number, number]> = {
  rim: [0, 3.2],
  paint: [0, 11],
  midrange: [-13, 17],
  left_corner3: [-21, 3],
  right_corner3: [21, 3],
  above_break3: [0, 25.5],
};
const ZONE_BADGE_LABEL: Record<string, string> = {
  rim: "RIM", paint: "PAINT", midrange: "MID", left_corner3: "L CORNER",
  right_corner3: "R CORNER", above_break3: "3PT",
};

export function ShotZoneCourt({ profile }: { profile: PlayerProfile }) {
  const zones = Object.keys(ZONE_BADGE_POS).map((slug) => {
    const pctStat = findStat(profile, `zone_fg_pct_${slug}`);
    const fgaStat = findStat(profile, `zone_fga_${slug}`);
    const pct = numericValue(pctStat?.value);
    const fga = numericValue(fgaStat?.value);
    return { slug, pct, fga, pos: ZONE_BADGE_POS[slug] };
  });

  const hasAny = zones.some((z) => z.pct != null);
  if (!hasAny) return null;

  // League-relative colour needs the league distribution, which this
  // component doesn't have — so each badge is coloured against a fixed,
  // honest 30-70% FG% scale instead of a percentile. That is a real,
  // reader-checkable claim ("above 55% at the rim is good") rather than an
  // invented comparison this component would have no data to back up.
  const fixedTier = (pct: number | null) => {
    if (pct == null) return C.muted;
    if (pct >= 0.60) return "#16A34A";
    if (pct >= 0.48) return "#84CC16";
    if (pct >= 0.38) return "#C9A84C";
    if (pct >= 0.28) return "#EA580C";
    return "#DC2626";
  };

  return (
    <Card style={{ padding: "16px 20px 18px", marginBottom: 14 }}>
      <div style={{ marginBottom: 4 }}><SectionLabel>Shot chart by zone</SectionLabel></div>
      <div style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5 }}>
        FG% within each zone this season. Coloured against a fixed scale (60%+ elite, under 28% poor)
        rather than league percentile — the same bar for every position, since a rim FG% and a
        three-point FG% are not the same kind of shot.
      </div>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        <svg viewBox={`0 0 ${COURT_CW} ${COURT_CH}`} style={{ width: "100%", maxWidth: 420, height: "auto", borderRadius: 4, overflow: "hidden" }}>
          <rect x={0} y={0} width={COURT_CW} height={COURT_CH} fill={COURT_SURFACE} />
          {/* Court lines: lane, ft circle top, restricted area, three-point line */}
          <g fill="none" stroke={COURT_LINE} strokeWidth={1.5}>
            <rect x={HOOP_X - ft(FT.laneWidth / 2)} y={HOOP_Y - ft(FT.laneDepth)} width={ft(FT.laneWidth)} height={ft(FT.laneDepth)} />
            <path d={`M ${pt(-FT.ftCircleRadius, FT.laneDepth)} A ${ft(FT.ftCircleRadius)} ${ft(FT.ftCircleRadius)} 0 0 1 ${pt(FT.ftCircleRadius, FT.laneDepth)}`} />
            <path d={`M ${pt(-FT.restrictedRadius, 0)} A ${ft(FT.restrictedRadius)} ${ft(FT.restrictedRadius)} 0 0 1 ${pt(FT.restrictedRadius, 0)}`} />
            <line x1={HOOP_X - ft(FT.cornerThreeX)} y1={HOOP_Y} x2={HOOP_X - ft(FT.cornerThreeX)} y2={HOOP_Y - ft(CORNER_JOIN_Y)} />
            <line x1={HOOP_X + ft(FT.cornerThreeX)} y1={HOOP_Y} x2={HOOP_X + ft(FT.cornerThreeX)} y2={HOOP_Y - ft(CORNER_JOIN_Y)} />
            <path d={`M ${pt(-FT.cornerThreeX, CORNER_JOIN_Y)} A ${ft(FT.arcRadius)} ${ft(FT.arcRadius)} 0 0 1 ${pt(FT.cornerThreeX, CORNER_JOIN_Y)}`} />
          </g>
          <circle cx={HOOP_X} cy={HOOP_Y} r={ft(FT.rimRadius)} fill="none" stroke={COURT_RIM} strokeWidth={2} />
          <line x1={HOOP_X - ft(3)} y1={HOOP_Y + ft(0.6)} x2={HOOP_X + ft(3)} y2={HOOP_Y + ft(0.6)} stroke={COURT_LINE_STRONG} strokeWidth={2} />

          {zones.map((z) => {
            const [x, y] = z.pos;
            const cx = HOOP_X + ft(x), cy = HOOP_Y - ft(y);
            const color = fixedTier(z.pct);
            return (
              <g key={z.slug}>
                <circle cx={cx} cy={cy} r={26} fill={color} fillOpacity={z.pct == null ? 0.08 : 0.22} stroke={color} strokeWidth={1.5} />
                <text x={cx} y={cy - 3} textAnchor="middle" fill={z.pct == null ? C.muted : "#fff"} style={{ fontFamily: F.display, fontSize: 15 }}>
                  {z.pct == null ? "—" : `${(z.pct * 100).toFixed(0)}%`}
                </text>
                <text x={cx} y={cy + 11} textAnchor="middle" fill={z.pct == null ? C.muted : "rgba(255,255,255,0.85)"} style={{ fontFamily: F.mono, fontSize: 7.5, letterSpacing: "0.08em" }}>
                  {ZONE_BADGE_LABEL[z.slug]}
                </text>
              </g>
            );
          })}
        </svg>
        <div style={{ flex: "1 1 160px", minWidth: 150 }}>
          {zones.map((z) => (
            <div key={z.slug} style={{ display: "flex", justifyContent: "space-between", gap: 10, padding: "5px 0", borderBottom: `1px solid ${C.rule}` }}>
              <span style={{ fontSize: 11.5, color: C.dim }}>{ZONE_BADGE_LABEL[z.slug]}</span>
              <span style={{ ...NUM, fontSize: 11, color: C.muted }}>
                {z.fga == null ? "—" : `${z.fga.toFixed(0)} att`}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

// ── Play-type bar chart ──────────────────────────────────────────────────────
// A Synergy card's own layout: bar length is how often a player runs each
// action, colour is how efficient they are at it. Sorted by frequency so the
// player's actual offensive identity reads top-to-bottom at a glance.
const PLAY_TYPE_SLUGS = [
  "isolation", "transition", "pnr_ball_handler", "pnr_roll_man", "postup",
  "spotup", "handoff", "cut", "off_screen", "putback",
];
const PLAY_TYPE_LABEL: Record<string, string> = {
  isolation: "Isolation", transition: "Transition",
  pnr_ball_handler: "PnR (ball-handler)", pnr_roll_man: "PnR (roll man)",
  postup: "Post-up", spotup: "Spot-up", handoff: "Hand-off", cut: "Cut",
  off_screen: "Off-screen", putback: "Putback",
};

// PPP tiers: 1.0 PPP is roughly league-average offensive efficiency across
// most play types, so the scale is centred there rather than needing the
// league distribution this component doesn't have.
function pppColor(ppp: number | null): string {
  if (ppp == null) return C.muted;
  if (ppp >= 1.15) return "#16A34A";
  if (ppp >= 1.0) return "#84CC16";
  if (ppp >= 0.85) return "#C9A84C";
  if (ppp >= 0.7) return "#EA580C";
  return "#DC2626";
}

export function PlayTypeBars({ profile }: { profile: PlayerProfile }) {
  const rows = PLAY_TYPE_SLUGS.map((slug) => {
    const possPct = numericValue(findStat(profile, `playtype_${slug}_poss_pct`)?.value);
    const poss = numericValue(findStat(profile, `playtype_${slug}_poss`)?.value);
    const ppp = numericValue(findStat(profile, `playtype_${slug}_ppp`)?.value);
    return { slug, possPct, poss, ppp };
  })
    .filter((r) => r.possPct != null && r.possPct > 0)
    .sort((a, b) => (b.possPct ?? 0) - (a.possPct ?? 0));

  if (rows.length === 0) return null;
  const maxShare = Math.max(...rows.map((r) => r.possPct ?? 0));

  return (
    <Card style={{ padding: "16px 20px 14px", marginBottom: 14 }}>
      <div style={{ marginBottom: 4 }}><SectionLabel>Play type (Synergy)</SectionLabel></div>
      <div style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5 }}>
        Bar length is share of offensive possessions run as this play type; colour is points per
        possession — 1.00 PPP is roughly league-average offense. Needs at least 25 possessions to
        be shown at all.
      </div>
      {rows.map((r) => {
        const color = pppColor(r.ppp);
        return (
          <div key={r.slug} style={{ marginBottom: 9 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 3 }}>
              <span style={{ color: C.dim }}>{PLAY_TYPE_LABEL[r.slug]}</span>
              <span style={{ ...NUM, color, fontWeight: 600 }}>
                {formatStat(r.ppp, "num")} PPP
                <span style={{ color: C.muted, fontWeight: 400, marginLeft: 8 }}>{formatStat(r.possPct, "pct")}</span>
              </span>
            </div>
            <div style={{ height: 8, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                width: `${Math.max(2, ((r.possPct ?? 0) / maxShare) * 100)}%`, height: "100%",
                background: color, opacity: 0.85, borderRadius: 3, transition: "width 300ms",
              }} />
            </div>
          </div>
        );
      })}
    </Card>
  );
}

// ── Defense-by-category chart ────────────────────────────────────────────────
// FG% allowed at each shot type this player defends, inside-out. Colour is
// vs-league-normal for that same category — the fair comparison, since raw
// FG% allowed at the rim and on threes are not naturally on the same scale.
const DEF_CATEGORIES = [
  { slug: "rim", label: "Rim (< 6 ft)" },
  { slug: "paint", label: "Paint (< 10 ft)" },
  { slug: "two_pt", label: "Two-point" },
  { slug: "mid_long", label: "Long two (15+ ft)" },
  { slug: "perimeter", label: "Perimeter (3PT)" },
];

function pmColor(pm: number | null): string {
  if (pm == null) return C.muted;
  // Lower (more negative) is better defence — holding shooters below normal.
  if (pm <= -0.06) return "#16A34A";
  if (pm <= -0.02) return "#84CC16";
  if (pm <= 0.02) return "#C9A84C";
  if (pm <= 0.06) return "#EA580C";
  return "#DC2626";
}

export function DefenseCategoryChart({ profile }: { profile: PlayerProfile }) {
  const rows = DEF_CATEGORIES.map((c) => {
    const fgPct = numericValue(findStat(profile, `def_${c.slug}_fg_pct`)?.value);
    const pm = numericValue(findStat(profile, `def_${c.slug}_pm`)?.value);
    const fga = numericValue(findStat(profile, `def_${c.slug}_fga`)?.value);
    return { ...c, fgPct, pm, fga };
  }).filter((r) => r.fgPct != null);

  if (rows.length === 0) return null;
  const maxPct = Math.max(...rows.map((r) => r.fgPct ?? 0), 0.5);

  return (
    <Card style={{ padding: "16px 20px 14px", marginBottom: 14 }}>
      <div style={{ marginBottom: 4 }}><SectionLabel>FG% allowed by category</SectionLabel></div>
      <div style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5 }}>
        Bar length is FG% allowed; colour is how that compares to league normal for the same shot
        type — green means holding shooters well below what they'd normally make there.
      </div>
      {rows.map((r) => {
        const color = pmColor(r.pm);
        return (
          <div key={r.slug} style={{ marginBottom: 9 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 3 }}>
              <span style={{ color: C.dim }}>{r.label}</span>
              <span style={{ ...NUM, color, fontWeight: 600 }}>
                {formatStat(r.fgPct, "pct")}
                <span style={{ color: C.muted, fontWeight: 400, marginLeft: 8 }}>
                  {r.pm != null ? `${r.pm > 0 ? "+" : ""}${formatStat(r.pm, "pct")} vs normal` : ""}
                </span>
              </span>
            </div>
            <div style={{ height: 8, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                width: `${Math.max(2, ((r.fgPct ?? 0) / maxPct) * 100)}%`, height: "100%",
                background: color, opacity: 0.85, borderRadius: 3, transition: "width 300ms",
              }} />
            </div>
          </div>
        );
      })}
    </Card>
  );
}

// ── Archetype card ────────────────────────────────────────────────────────
// The "Urban NBA" label(s), computed server-side from measured stats — see
// src/inference/archetypes.py's module docstring. Fetches its own data
// (small, independent of the season/career/offense-defense toggles this
// page already has) and shows the FULL trait breakdown alongside the label,
// never the label alone — an archetype is a summary of the numbers below
// it, not a verdict a reader has to take on faith.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export function ArchetypeCard({ playerId }: { playerId: string }) {
  const [detail, setDetail] = useState<ArchetypeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  // Which archetype's definition is currently expanded, by key — null means
  // none. A single slot rather than a Set: opening a second definition
  // closes whichever was open, since two blurbs open at once just reads as
  // clutter on a card this narrow.
  const [openBlurb, setOpenBlurb] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    fetch(`${API_BASE}/stats/player/${encodeURIComponent(playerId)}/archetype`)
      .then((r) => {
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: ArchetypeDetail) => { if (!cancelled) setDetail(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [playerId]);

  if (error || !detail) return null; // a missing archetype is not worth an error banner on the profile

  if (!detail.eligible) {
    return (
      <Card style={{ padding: "15px 19px", marginBottom: 16, breakInside: "avoid" }}>
        <div style={{ marginBottom: 4 }}><SectionLabel>Archetype</SectionLabel></div>
        <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.5 }}>
          Not enough games played this season to assign an archetype yet.
        </div>
      </Card>
    );
  }

  const measuredTraits = detail.traits.filter((t) => t.percentile != null);

  return (
    <Card style={{ padding: "15px 19px 14px", marginBottom: 16, breakInside: "avoid" }}>
      <div style={{ marginBottom: 10 }}><SectionLabel>Archetype</SectionLabel></div>

      {detail.primary ? (
        <button
          onClick={() => setOpenBlurb((k) => (k === detail.primary!.key ? null : detail.primary!.key))}
          title="Click for what this archetype means"
          style={{
            display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 8,
            background: "transparent", border: "none", padding: 0, cursor: "pointer", textAlign: "left",
          }}
        >
          <span style={{
            fontFamily: F.display, fontSize: 22, color: tierColor(detail.primary.score / 100),
            letterSpacing: "0.02em", borderBottom: `1px dotted ${C.muted}`,
          }}>
            {detail.primary.label}
          </span>
          <span style={{ ...NUM, fontSize: 10, color: C.muted }}>{detail.primary.score}</span>
        </button>
      ) : (
        <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 8 }}>
          Doesn't cleanly fit any defined archetype this season.
        </div>
      )}

      {detail.primary && openBlurb === detail.primary.key && detail.primary.blurb && (
        <div style={{
          fontSize: 11, color: C.dim, lineHeight: 1.5, marginBottom: 10,
          padding: "8px 10px", background: "rgba(255,255,255,0.04)", borderRadius: 4,
          borderLeft: `2px solid ${tierColor(detail.primary.score / 100)}`,
        }}>
          {detail.primary.blurb}
        </div>
      )}

      {detail.secondary.length > 0 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
          {detail.secondary.map((s) => (
            <div key={s.key}>
              <button
                onClick={() => setOpenBlurb((k) => (k === s.key ? null : s.key))}
                title="Click for what this archetype means"
                style={{
                  fontFamily: F.mono, fontSize: 10, color: C.dim, cursor: "pointer",
                  background: openBlurb === s.key ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.05)",
                  border: "none", borderRadius: 3, padding: "3px 8px",
                }}
              >
                {s.label} · {s.score}
              </button>
              {openBlurb === s.key && s.blurb && (
                <div style={{
                  fontSize: 11, color: C.dim, lineHeight: 1.5, marginTop: 6,
                  padding: "8px 10px", background: "rgba(255,255,255,0.04)", borderRadius: 4,
                  borderLeft: `2px solid ${C.muted}`, width: "100%",
                }}>
                  {s.blurb}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <button
        onClick={() => setExpanded((v) => !v)}
        style={{
          background: "transparent", border: "none", color: C.gold, cursor: "pointer",
          fontFamily: F.mono, fontSize: 10, letterSpacing: "0.15em", padding: 0, marginBottom: expanded ? 10 : 0,
        }}
      >
        {expanded ? "▾ HIDE TRAIT BREAKDOWN" : "▸ WHY? SHOW TRAIT BREAKDOWN"}
      </button>

      {expanded && (
        <div>
          <div style={{ fontSize: 10.5, color: C.faint, marginBottom: 8, lineHeight: 1.5 }}>
            Every archetype is a combination of these percentiles — nothing here is hand-assigned.
            A dash means none of that trait's underlying stats are measured for this player.
          </div>
          {measuredTraits.map((t) => (
            <div key={t.key} style={{ padding: "5px 0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 3 }}>
                <span style={{ color: C.dim }}>{t.label}</span>
                <span style={{ ...NUM, color: tierColor((t.percentile ?? 0) / 100) }}>{t.percentile}</span>
              </div>
              <Bar pct={(t.percentile ?? 0) / 100} />
            </div>
          ))}
          {detail.qualified.length > 1 && (
            <div style={{ fontSize: 10.5, color: C.faint, marginTop: 10, lineHeight: 1.5 }}>
              Also qualified for: {detail.qualified.filter((q) => q.key !== detail.primary?.key)
                .map((q) => `${q.label} (${q.score})`).join(", ")}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
