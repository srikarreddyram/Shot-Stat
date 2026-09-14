import { useEffect, useState } from "react";
import {
  CareerPanelData, CareerStat, Side, formatStat, LOWER_IS_BETTER,
} from "../../lib/stat-engine";
import { C, F, NUM, Card, SectionLabel, TrendChart } from "./ui";

// One player's stats over time: this season, last season as a reference
// point, a career average, a career total, and — the point of this file —
// the full season-by-season line, since a career is a shape over time, not
// four numbers.
//
// Split by OFFENSE / DEFENSE (see `side` on each section, from the API's
// career_stats.GROUP_SIDE) for the same reason the season view is: one flat
// list of ~140 rows is a spreadsheet, not a stats page.
//
// A curated few headline stats per side get an actual trend LINE at the top
// — every stat has a full `series`, but drawing a chart for all ~140 of them
// would just be clutter with extra steps. The rest stay in the table below,
// still fully available, one click away rather than the first thing shown.
//
// "Career" is career within this project's data. The span column is not
// decoration — the window genuinely differs per stat, because the NBA started
// publishing tracking data partway through the period covered here, so a
// career deflections figure rests on fewer seasons than a career FG%-allowed.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const GOOD = "#16A34A";
const BAD = "#DC2626";

// (stat key, chart colour) per side. Picked from stats every player who has
// tracking data at all will have SOME history for, so the headline row is
// rarely empty. Anything not listed here still appears in the table below.
const HEADLINE_TRENDS: Record<Side, { key: string; color: string }[]> = {
  offense: [
    { key: "zone_fg_pct_rim", color: "#16A34A" },
    { key: "zone_fg_pct_above_break3", color: "#4C9ED9" },
    { key: "potential_ast", color: "#C9A84C" },
    { key: "playtype_isolation_ppp", color: "#9B7ED9" },
  ],
  defense: [
    { key: "def_overall_pm", color: "#16A34A" },
    { key: "def_rim_pm", color: "#4C9ED9" },
    { key: "blocks", color: "#C9A84C" },
    { key: "deflections", color: "#E0793C" },
  ],
  info: [],
};

export function CareerPanel({ playerId, side }: { playerId: string; side: Side }) {
  const [data, setData] = useState<CareerPanelData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    fetch(`${API_BASE}/stats/player/${encodeURIComponent(playerId)}/career`)
      .then((r) => {
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: CareerPanelData) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [playerId]);

  if (error) return <div style={{ color: BAD, fontSize: 13 }}>{error}</div>;
  if (!data) return <div style={{ color: C.muted, fontSize: 13 }}>Loading career history…</div>;

  const sections = data.sections.filter((s) => s.side === side);

  if (sections.length === 0) {
    return (
      <Card style={{ padding: "44px 24px", textAlign: "center" }}>
        <div style={{ fontFamily: F.mono, fontSize: 11, color: C.muted, letterSpacing: "0.2em" }}>
          NO {side.toUpperCase()} HISTORY FOR THIS PLAYER
        </div>
      </Card>
    );
  }

  const allStats = sections.flatMap((s) => s.stats);
  const trendStats = HEADLINE_TRENDS[side]
    .map((t) => ({ ...t, stat: allStats.find((s) => s.key === t.key) }))
    .filter((t): t is { key: string; color: string; stat: CareerStat } => !!t.stat && t.stat.series.length >= 2);

  return (
    <>
      <div style={{ fontSize: 11.5, color: C.faint, marginBottom: 14, lineHeight: 1.6, maxWidth: 820 }}>
        Career figures are computed from summed makes and attempts, and from total games — not by
        averaging each season's number, which would weight a ten-game season the same as a full one.
        Rates have no meaningful total, so their volume is listed as its own row.
        {data.first_season ? ` This player's data runs ${data.first_season} to ${data.last_season}.` : ""}
      </div>

      {trendStats.length > 0 && (
        <Card style={{ padding: "16px 20px 8px", marginBottom: 14 }}>
          <div style={{ marginBottom: 12 }}><SectionLabel>Career trend</SectionLabel></div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 20 }}>
            {trendStats.map((t) => (
              <TrendChart key={t.key} points={t.stat.series} fmt={t.stat.fmt} color={t.color} label={t.stat.label} />
            ))}
          </div>
        </Card>
      )}

      {sections.map((section) => (
        <Card key={section.group} style={{ padding: "15px 18px 6px", marginBottom: 12, overflow: "hidden" }}>
          <div style={{ marginBottom: 10 }}><SectionLabel>{section.label}</SectionLabel></div>
          <div style={{ overflowX: "auto" }}>
            <table className="se-table" style={{ minWidth: 640 }}>
              <thead>
                <tr>
                  <th className="se-left" style={{ cursor: "default", position: "static" }}>Stat</th>
                  <th style={{ cursor: "default", position: "static" }}>{data.season}</th>
                  <th style={{ cursor: "default", position: "static" }}>{data.previous_season}</th>
                  <th style={{ cursor: "default", position: "static" }}>Career avg</th>
                  <th style={{ cursor: "default", position: "static" }}>Career total</th>
                  <th style={{ cursor: "default", position: "static" }}>Span</th>
                </tr>
              </thead>
              <tbody>
                {section.stats.map((stat) => (
                  <CareerRow key={stat.key} stat={stat} />
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ))}
    </>
  );
}

function CareerRow({ stat }: { stat: CareerStat }) {
  const { current, career_avg } = stat;
  const lowerIsBetter = LOWER_IS_BETTER.has(stat.key) || stat.kind === "diff" || stat.key.endsWith("_fg_pct") && stat.key.startsWith("def_");

  let delta: { text: string; color: string } | null = null;
  if (current != null && career_avg != null && career_avg !== 0) {
    const diff = current - career_avg;
    // A hair either side of the career average is noise, not a trend.
    if (Math.abs(diff) > Math.abs(career_avg) * 0.02) {
      const better = lowerIsBetter ? diff < 0 : diff > 0;
      delta = {
        text: `${diff > 0 ? "▲" : "▼"} ${formatStat(Math.abs(diff), stat.fmt)}`,
        color: better ? GOOD : BAD,
      };
    }
  }

  return (
    <tr style={{ cursor: "default" }}>
      <td className="se-left" style={{ fontFamily: F.body, fontSize: 12.5, color: C.dim }}>
        {stat.label}
      </td>
      <td style={{ color: C.text, fontWeight: 600 }}>
        {formatStat(current, stat.fmt)}
        {delta && (
          <span style={{ color: delta.color, fontSize: 9.5, marginLeft: 6, fontWeight: 400 }}>
            {delta.text}
          </span>
        )}
      </td>
      <td style={{ color: "rgba(240,240,240,0.55)" }}>{formatStat(stat.previous, stat.fmt)}</td>
      <td style={{ color: C.gold }}>{formatStat(career_avg, stat.fmt)}</td>
      <td style={{ color: stat.career_total == null ? C.muted : C.text }}>
        {stat.career_total == null ? "—" : formatStat(stat.career_total, "count")}
      </td>
      <td style={{ ...NUM, color: C.muted, fontSize: 10 }}>
        {stat.seasons}{stat.seasons === 1 ? " szn" : " szns"}
      </td>
    </tr>
  );
}
