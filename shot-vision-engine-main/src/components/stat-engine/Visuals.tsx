import { useEffect, useState } from "react";
import {
  ArchetypeDetail, PlayerProfile, formatStat, isRankable, numericValue, percentileOf, rankOf,
} from "../../lib/stat-engine";
import {
  Bar, C, ChartHeader, F, MiniTable, NUM, Card, RankPill, SectionLabel, ViewMode, tierColor, useViewMode,
} from "./ui";

// Season-view visuals, replacing what used to be more rows in a stat table:
// a Synergy-style play-type bar chart and a defensive FG%-allowed-by-category
// chart. Each reads straight off the PlayerProfile the page already fetched —
// no extra request. (The shot chart lives in ShotMap.tsx, drawn from real
// shot locations on the engine's court.)

function findStat(profile: PlayerProfile, key: string) {
  for (const section of profile.sections) {
    const hit = section.stats.find((s) => s.key === key);
    if (hit) return hit;
  }
  return null;
}

/** League rank pill for one value, or nothing when it cannot be ranked. */
function leagueRank(value: number | null, key: string, distributions?: Map<string, number[]>) {
  if (value == null || !distributions) return null;
  const sorted = distributions.get(key) ?? [];
  const pct = percentileOf(value, sorted, key);
  const rank = rankOf(value, sorted, key);
  return rank && pct != null ? <RankPill rank={rank.rank} outOf={rank.outOf} pct={pct} /> : null;
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

export function PlayTypeBars({ profile, distributions, defaultMode = "chart" }: {
  profile: PlayerProfile; distributions?: Map<string, number[]>; defaultMode?: ViewMode;
}) {
  const [mode, setMode] = useViewMode(defaultMode);
  const rows = PLAY_TYPE_SLUGS.map((slug) => {
    const possPct = numericValue(findStat(profile, `playtype_${slug}_poss_pct`)?.value);
    const poss = numericValue(findStat(profile, `playtype_${slug}_poss`)?.value);
    const ppp = numericValue(findStat(profile, `playtype_${slug}_ppp`)?.value);
    const fgPct = numericValue(findStat(profile, `playtype_${slug}_fg_pct`)?.value);
    return { slug, possPct, poss, ppp, fgPct };
  })
    .filter((r) => r.possPct != null && r.possPct > 0)
    .sort((a, b) => (b.possPct ?? 0) - (a.possPct ?? 0));

  if (rows.length === 0) return null;
  const maxShare = Math.max(...rows.map((r) => r.possPct ?? 0));

  return (
    <Card style={{ padding: "16px 20px 14px", marginBottom: 14 }}>
      <ChartHeader title="Play type (Synergy)" mode={mode} onChange={setMode} />
      <div style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5 }}>
        Bar length is share of offensive possessions run as this play type; colour is points per
        possession — 1.00 PPP is roughly league-average offense. Needs at least 25 possessions to
        be shown at all.
      </div>
      {mode === "table" ? (
        <MiniTable
          headers={["Play type", "Share", "Poss", "PPP", "PPP rank", "FG%"]}
          rows={rows.map((r) => [
            PLAY_TYPE_LABEL[r.slug],
            formatStat(r.possPct, "pct"),
            r.poss == null ? "—" : r.poss.toFixed(0),
            <span style={{ color: pppColor(r.ppp) }}>{formatStat(r.ppp, "num")}</span>,
            leagueRank(r.ppp, `playtype_${r.slug}_ppp`, distributions) ?? "—",
            formatStat(r.fgPct, "pct"),
          ])}
        />
      ) : rows.map((r) => {
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
              <div className="se-bar-fill" style={{
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

export function DefenseCategoryChart({ profile, distributions, defaultMode = "chart" }: {
  profile: PlayerProfile; distributions?: Map<string, number[]>; defaultMode?: ViewMode;
}) {
  const [mode, setMode] = useViewMode(defaultMode);
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
      <ChartHeader title="FG% allowed by category" mode={mode} onChange={setMode} />
      <div style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5 }}>
        Bar length is FG% allowed; colour is how that compares to league normal for the same shot
        type — green means holding shooters well below what they'd normally make there.
      </div>
      {mode === "table" ? (
        <MiniTable
          headers={["Shot type", "FG% allowed", "vs normal", "League rank", "FGA faced"]}
          rows={rows.map((r) => [
            r.label,
            formatStat(r.fgPct, "pct"),
            <span style={{ color: pmColor(r.pm) }}>
              {r.pm != null ? `${r.pm > 0 ? "+" : ""}${formatStat(r.pm, "pct")}` : "—"}
            </span>,
            leagueRank(r.pm, `def_${r.slug}_pm`, distributions) ?? "—",
            r.fga == null ? "—" : r.fga.toFixed(0),
          ])}
        />
      ) : rows.map((r) => {
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
              <div className="se-bar-fill" style={{
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

// ── Strengths & weaknesses ───────────────────────────────────────────────────
// The whole profile in one glance: of every ranked stat this season, the
// handful where this player sits furthest ABOVE the league median and the
// handful furthest BELOW it. Bars diverge from the 50th percentile, so length
// is literally "how far from average" and side is "which way" — direction is
// carried by position and the printed percentile, not by colour alone.
// Info sections (overview, composite ratings) are left out: they summarise
// the stats below rather than being stats in their own right.
const EXTREMES_PER_SIDE = 5;

export function StrengthsChart({ profile, distributions, defaultMode = "chart" }: {
  profile: PlayerProfile;
  distributions: Map<string, number[]>;
  defaultMode?: ViewMode;
}) {
  const [mode, setMode] = useViewMode(defaultMode);
  const ranked = profile.sections
    .filter((s) => s.side !== "info")
    .flatMap((s) => s.stats.map((stat) => ({ stat, section: s.label })))
    .map(({ stat, section }) => {
      const v = numericValue(stat.value);
      if (v == null || !isRankable(stat.key, stat.fmt)) return null;
      const sorted = distributions.get(stat.key) ?? [];
      const pct = percentileOf(v, sorted, stat.key);
      if (pct == null) return null;
      return { stat, section, pct, rank: rankOf(v, sorted, stat.key) };
    })
    .filter((x): x is NonNullable<typeof x> => x != null);

  if (ranked.length < EXTREMES_PER_SIDE * 2) return null;

  const byPct = [...ranked].sort((a, b) => b.pct - a.pct);
  const strengths = byPct.slice(0, EXTREMES_PER_SIDE).filter((r) => r.pct > 0.5);
  const weaknesses = byPct.slice(-EXTREMES_PER_SIDE).filter((r) => r.pct < 0.5).reverse();
  if (strengths.length === 0 && weaknesses.length === 0) return null;

  const ordinal = (p: number) => {
    const n = Math.round(p * 100);
    const suffix = n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th";
    return `${n}${suffix}`;
  };

  const row = (r: (typeof ranked)[number], i: number) => {
    const above = r.pct >= 0.5;
    const reach = Math.abs(r.pct - 0.5) * 2; // 0 at the median, 1 at the extreme
    const color = tierColor(r.pct);
    return (
      <div
        key={r.stat.key}
        title={r.rank ? `${r.stat.label}: ${formatStat(r.stat.value, r.stat.fmt)} — #${r.rank.rank} of ${r.rank.outOf}` : undefined}
        style={{ display: "grid", gridTemplateColumns: "minmax(120px, 1.1fr) 2fr 44px", alignItems: "center", gap: 12, padding: "5px 0" }}
      >
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 11.5, color: C.dim, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {r.stat.label}
          </div>
          <div style={{ fontFamily: F.mono, fontSize: 8.5, color: C.muted, letterSpacing: "0.1em", textTransform: "uppercase" }}>
            {r.section} · {formatStat(r.stat.value, r.stat.fmt)}
          </div>
        </div>
        <div style={{ position: "relative", height: 10 }}>
          <div style={{ position: "absolute", left: "50%", top: -3, bottom: -3, width: 1, background: "rgba(255,255,255,0.18)" }} />
          <div
            className="se-bar-fill"
            style={{
              position: "absolute", top: 1, height: 8, borderRadius: 3, background: color, opacity: 0.9,
              width: `${Math.max(1.5, reach * 50)}%`,
              ...(above ? { left: "50%" } : { right: "50%" }),
              transformOrigin: above ? "left center" : "right center",
              animationDelay: `${120 + i * 60}ms`,
            }}
          />
        </div>
        <div style={{ ...NUM, fontSize: 11, color: C.text, textAlign: "right", fontWeight: 600 }}>
          {ordinal(r.pct)}
        </div>
      </div>
    );
  };

  return (
    <Card style={{ padding: "16px 20px 14px", marginBottom: 16 }}>
      <ChartHeader title={<>Strengths &amp; weaknesses</>} mode={mode} onChange={setMode} />
      <div style={{ fontSize: 11, color: C.faint, margin: "6px 0 12px", lineHeight: 1.5, maxWidth: 720 }}>
        Of the {ranked.length} stats this player is ranked in, the ones furthest above and below
        the league median. Bars grow out from the centre line (50th percentile); the number is his
        league percentile. Hover a row for the raw value and exact rank.
      </div>
      {mode === "table" ? (
        <MiniTable
          headers={["Stat", "Value", "Percentile", "Rank"]}
          rows={[...strengths, ...weaknesses].map((r) => [
            r.stat.label,
            formatStat(r.stat.value, r.stat.fmt),
            <span style={{ color: tierColor(r.pct) }}>{ordinal(r.pct)}</span>,
            r.rank ? `#${r.rank.rank} / ${r.rank.outOf}` : "—",
          ])}
        />
      ) : <>
      {strengths.length > 0 && (
        <>
          <div style={{ fontFamily: F.mono, fontSize: 8.5, color: C.muted, letterSpacing: "0.2em", marginBottom: 2 }}>
            ▲ ABOVE MEDIAN
          </div>
          {strengths.map((r, i) => row(r, i))}
        </>
      )}
      {weaknesses.length > 0 && (
        <>
          <div style={{ fontFamily: F.mono, fontSize: 8.5, color: C.muted, letterSpacing: "0.2em", margin: "10px 0 2px" }}>
            ▼ BELOW MEDIAN
          </div>
          {weaknesses.map((r, i) => row(r, i + strengths.length))}
        </>
      )}
      </>}
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

// Label chips by kind: a role (identity) gets the accent, a skill stays
// neutral, a negative pattern is tinted red — the kind reads before the word.
const TIER_CHIP: Record<string, React.CSSProperties> = {
  identity: { color: C.text, background: "var(--se-hover)", border: "1px solid var(--se-edge-strong)" },
  skill: { color: C.dim, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" },
  pattern: { color: "#F2A3A3", background: "rgba(220,38,38,0.08)", border: "1px solid rgba(220,38,38,0.3)" },
};
const TIER_TITLE: Record<string, string> = {
  identity: "Role on the floor",
  skill: "Something he is elite at",
  pattern: "A pattern worth flagging",
};

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

  if (!detail.primary && !detail.eligible) {
    return (
      <Card style={{ padding: "15px 19px", marginBottom: 16, breakInside: "avoid" }}>
        <div style={{ marginBottom: 4 }}><SectionLabel>Archetype</SectionLabel></div>
        <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.5 }}>
          Not enough games or minutes this season (or last) to assign an archetype yet.
        </div>
      </Card>
    );
  }

  const measuredTraits = detail.traits.filter((t) => t.percentile != null);
  // Labels from his last full season, when injury kept him under the bar.
  const fromEarlier = detail.based_on_season && detail.season && detail.based_on_season !== detail.season;

  return (
    <Card style={{ padding: "15px 19px 14px", marginBottom: 16, breakInside: "avoid" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
        <SectionLabel>{detail.secondary.length > 0 ? "Archetypes" : "Archetype"}</SectionLabel>
        {fromEarlier && (
          <span style={{ ...NUM, fontSize: 9.5, color: C.muted }} title="Too few games this season to judge on its own">
            from {detail.based_on_season} · {detail.games_this_season} games this season
          </span>
        )}
      </div>

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
                title={`${TIER_TITLE[s.tier ?? "identity"]} — click for what it means`}
                style={{
                  fontFamily: F.mono, fontSize: 10, cursor: "pointer", borderRadius: 3, padding: "3px 8px",
                  ...TIER_CHIP[s.tier ?? "identity"],
                  ...(openBlurb === s.key ? { background: "rgba(255,255,255,0.12)" } : null),
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
          <div style={{ fontSize: 10.5, color: C.faint, marginTop: 10, lineHeight: 1.5 }}>
            A player carries every archetype he qualifies for. The headline is who he is on the floor —
            the most complete description that fits — then what he's elite at, then any pattern worth
            flagging. The number beside each is how strongly he fits it.
          </div>
        </div>
      )}
    </Card>
  );
}
