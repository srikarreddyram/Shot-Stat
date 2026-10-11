import { useEffect, useMemo, useState } from "react";
import { C, Card, ChartHeader, Chips, F, NUM, Segmented, TrendChart, tierColor, useViewMode } from "./ui";
import { formatStat } from "../../lib/stat-engine";

// Season-by-season averages of the major box-score stats, for one player or
// one team — src/inference/season_averages.py. TABLE is the default view
// (one row per season, newest first, with the career line under it); CHART
// draws any one stat across seasons. Each column's best season is picked out
// in the accent colour; team cells also carry the league rank that season.
// set="advanced" shows the NBA's own advanced stats the same way (no career
// line: those rates cannot be combined across seasons from averages).

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type Col = { key: string; label: string; fmt: string };
type Row = {
  season: string; team?: string; seasons?: number; ranks?: Record<string, number>; of?: number;
  [key: string]: string | number | null | undefined | Record<string, number>;
};
type Block = { seasons: Row[]; career: Row | null };
type Response = {
  columns: Col[]; regular: Block; playoffs: Block;
  /** Columns where fewer is better, and columns with no better or worse at
   *  all (pace, games). Sent by the advanced endpoint; the box-score one
   *  falls back to the defaults below. */
  lower_is_better?: string[]; no_best?: string[];
};

// A short season (a 9-game rookie stint, a first-round sweep) shouldn't be
// able to claim "best season" in a column.
const MIN_GAMES_FOR_BEST = { regular: 20, playoffs: 4 };
const PLAYER_LOWER_IS_BETTER = new Set(["tov", "pf"]);
const NO_BEST = new Set(["gp", "w", "l"]);

const ordinal = (n: number) => {
  const suffix = n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th";
  return `${n}${suffix}`;
};

export function SeasonAverages({ kind, id, set = "box" }: { kind: "player" | "team"; id: string; set?: "box" | "advanced" }) {
  const [data, setData] = useState<Response | null>(null);
  const [missing, setMissing] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [type, setType] = useState<"regular" | "playoffs">("regular");
  const [mode, setMode] = useViewMode("table");
  const [stat, setStat] = useState("pts");

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setMissing(false);
    setErr(null);
    fetch(`${API_BASE}/stats/${kind}/${encodeURIComponent(id)}/seasons${set === "advanced" ? "?set=advanced" : ""}`)
      .then((r) => {
        if (r.status === 404) { if (!cancelled) setMissing(true); return null; }
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: Response | null) => { if (!cancelled && d) setData(d); })
      .catch((e) => { if (!cancelled) setErr(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [kind, id, set]);

  const block = data?.[type];
  const lowerBetter = useMemo(
    () => (data?.lower_is_better ? new Set(data.lower_is_better) : kind === "team" ? new Set<string>() : PLAYER_LOWER_IS_BETTER),
    [kind, data],
  );
  const noBest = useMemo(() => (data?.no_best ? new Set(data.no_best) : NO_BEST), [data]);

  // The season each column peaks in, among seasons with enough games.
  const best = useMemo(() => {
    const out: Record<string, string> = {};
    if (!data || !block) return out;
    const eligible = block.seasons.filter((r) => Number(r.gp ?? 0) >= MIN_GAMES_FOR_BEST[type]);
    if (eligible.length < 2) return out;
    for (const c of data.columns) {
      if (noBest.has(c.key)) continue;
      let pick: Row | null = null;
      for (const r of eligible) {
        const v = r[c.key];
        if (typeof v !== "number") continue;
        const pv = pick?.[c.key] as number | undefined;
        if (pick == null || (lowerBetter.has(c.key) ? v < (pv as number) : v > (pv as number))) pick = r;
      }
      if (pick) out[c.key] = pick.season;
    }
    return out;
  }, [data, block, type, lowerBetter, noBest]);

  if (missing) return null;
  if (err) return <Card style={{ padding: 16, marginBottom: 16, color: C.red, fontSize: 12.5 }}>{err}</Card>;
  if (!data || !block) {
    return <Card style={{ padding: "18px 20px", marginBottom: 16, color: C.muted, fontSize: 12 }}>Loading {set === "advanced" ? "advanced stats" : "season averages"}…</Card>;
  }

  const cols = data.columns;
  const chartCol =
    cols.find((c) => c.key === stat) ?? cols.find((c) => c.key === "pts") ?? cols.find((c) => !noBest.has(c.key)) ?? cols[0];
  const points = block.seasons
    .filter((r) => typeof r[chartCol.key] === "number")
    .map((r) => ({ season: r.season, value: r[chartCol.key] as number }))
    .reverse();

  const cell = (r: Row, c: Col, isCareer: boolean) => {
    const v = r[c.key] as number | null | undefined;
    const isBest = !isCareer && best[c.key] === r.season;
    const rank = r.ranks?.[c.key];
    return (
      <td key={c.key} style={{ color: isBest ? C.gold : isCareer ? C.text : undefined, fontWeight: isBest || isCareer ? 700 : 400 }}>
        {formatStat(v ?? null, c.fmt)}
        {rank != null && r.of != null && (
          <div style={{ fontSize: 8.5, fontWeight: 400, color: tierColor(1 - (rank - 1) / Math.max(r.of - 1, 1)) }}>
            {ordinal(rank)}
          </div>
        )}
      </td>
    );
  };

  return (
    <Card style={{ padding: "16px 20px 14px", marginBottom: 16 }}>
      <ChartHeader title={set === "advanced" ? "Advanced stats by season (official NBA)" : "Season averages"} mode={mode} onChange={setMode} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", margin: "10px 0 12px" }}>
        <div style={{ fontSize: 11, color: C.faint, lineHeight: 1.5, maxWidth: 640 }}>
          {set === "advanced" ? (
            <>
              The NBA's own advanced numbers, every season in the data: usage, true shooting, assist,
              rebound and turnover rates, and {kind === "team" ? "points scored and allowed" : "the team's points scored and allowed while he plays"},{" "}
              per 100 possessions. No career line — these rates can't be combined across seasons from averages.{" "}
            </>
          ) : (
            <>
              Per game, every season in the data. Shooting percentages are total makes over total
              attempts; eFG% counts a three as 1.5 makes and TS% also credits free throws.{" "}
            </>
          )}
          <span style={{ color: C.gold }}>Gold</span> marks the best season in each column
          {kind === "team" ? "; the small number under each value is the league rank that season." : "."}
        </div>
        <Segmented value={type} onChange={(v) => setType(v as "regular" | "playoffs")}
                   options={[{ value: "regular", label: "REGULAR SEASON" }, { value: "playoffs", label: "PLAYOFFS" }]} />
      </div>

      {block.seasons.length === 0 ? (
        <div style={{ fontFamily: F.mono, fontSize: 10.5, color: C.muted, letterSpacing: "0.2em", padding: "22px 0", textAlign: "center" }}>
          NO {type === "playoffs" ? "PLAYOFF" : "REGULAR-SEASON"} GAMES IN THE DATA
        </div>
      ) : mode === "chart" ? (
        <div className="se-rise">
          <div style={{ marginBottom: 12 }}>
            <Chips value={chartCol.key} onChange={setStat}
                   options={cols.filter((c) => !noBest.has(c.key)).map((c) => ({ value: c.key, label: c.label }))} />
          </div>
          {points.length >= 2 ? (
            <div style={{ maxWidth: 760 }}>
              <TrendChart points={points} fmt={chartCol.fmt} label={`${chartCol.label}, season by season`} />
            </div>
          ) : (
            <div style={{ fontSize: 12, color: C.muted, padding: "12px 0" }}>Only one season — nothing to trend yet.</div>
          )}
        </div>
      ) : (
        <div className="se-rise" style={{ overflowX: "auto" }}>
          <table className="se-table" style={{ minWidth: 0 }}>
            <thead>
              <tr>
                <th className="se-left" style={{ cursor: "default", position: "static" }}>Season</th>
                {kind === "player" && <th className="se-left" style={{ cursor: "default", position: "static" }}>Team</th>}
                {cols.map((c) => (
                  <th key={c.key} style={{ cursor: "default", position: "static" }}>{c.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.seasons.map((r) => (
                <tr key={r.season} style={{ cursor: "default" }}>
                  <td className="se-left" style={{ fontFamily: F.mono, color: C.text }}>{r.season}</td>
                  {kind === "player" && <td className="se-left" style={{ ...NUM, fontSize: 11, color: C.muted }}>{r.team}</td>}
                  {cols.map((c) => cell(r, c, false))}
                </tr>
              ))}
              {block.career && (
                <tr style={{ cursor: "default" }}>
                  <td className="se-left" style={{ fontFamily: F.mono, color: C.gold, borderTop: `1px solid ${C.edgeStrong}` }}>CAREER</td>
                  {kind === "player" && (
                    <td className="se-left" style={{ ...NUM, fontSize: 11, color: C.muted, borderTop: `1px solid ${C.edgeStrong}` }}>
                      {block.career.seasons} SZN
                    </td>
                  )}
                  {cols.map((c) => cell(block.career!, c, true))}
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
