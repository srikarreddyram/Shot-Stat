import { ReactNode, useEffect, useRef, useState } from "react";
import { ShotMapData, ShotMapZone, ShotMetric, useShotMap } from "../../lib/shot-maps";
import { drawCourtLines, drawHardwood, fillZone } from "../shot-vision/engine/court/courtDrawing";
import {
  COURT_AREA_H,
  COURT_CW,
  DIVERGING_RAMP,
  HEAT_RAMP,
  LEGEND_BG,
  LEGEND_H,
  divergingColor,
  heatColor,
  heatSwatch,
} from "../shot-vision/engine/court/geometry";
import { formatStat, percentileOf, rankOf } from "../../lib/stat-engine";
import {
  C,
  Card,
  ChartHeader,
  Chips,
  F,
  MiniTable,
  NUM,
  RankPill,
  Segmented,
  ViewMode,
  useViewMode,
} from "./ui";

// Shot maps on the Stat Engine profiles: where a player shoots from, and how
// well in each zone compared with the league. The statistics (smoothing, the
// shrinkage toward the league, and why efficiency is shown by zone rather
// than spot by spot) all live server-side in src/inference/shot_maps.py;
// this file only paints the fields onto the same hardwood court the engine
// uses (courtDrawing.ts), with the same multiply tint so the wood shows
// through the heat.

/** ±6 percentage points spans the whole diverging ramp: the true spread
 *  between players in any zone is only 3-5.5 points (shot_maps.py), so a
 *  wider scale would leave every map pale. Fixed rather than fitted per
 *  player, so the same colour means the same gap on every map. */
const VS_LEAGUE_SCALE = 0.06;

const ZONE_SHORT: Record<string, string> = {
  "Restricted Area": "Rim",
  "In The Paint (Non-RA)": "Paint",
  "Mid-Range": "Mid-range",
  "Left Corner 3": "Left corner 3",
  "Right Corner 3": "Right corner 3",
  "Above the Break 3": "Above the break 3",
};

// The Stat Engine's own zone FG% columns, for the league rank.
const ZONE_SLUG: Record<string, string> = {
  "Restricted Area": "rim",
  "In The Paint (Non-RA)": "paint",
  "Mid-Range": "midrange",
  "Left Corner 3": "left_corner3",
  "Right Corner 3": "right_corner3",
  "Above the Break 3": "above_break3",
};

// ── Painting ─────────────────────────────────────────────────────────────────
/** Lay an offscreen layer onto the court area as a multiply tint, so the
 *  wood's grain and seams show through at every intensity. */
function multiplyOnto(ctx: CanvasRenderingContext2D, layer: CanvasImageSource, blur: string) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, COURT_CW, COURT_AREA_H);
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.filter = blur;
  ctx.globalCompositeOperation = "multiply";
  ctx.drawImage(layer, 0, 0, COURT_CW, COURT_AREA_H);
  ctx.restore();
}

/** Where he shoots: the density field, one value per square foot. */
function paintFrequency(ctx: CanvasRenderingContext2D, data: ShotMapData) {
  const { nx, ny } = data.grid;
  const field = ctx.createImageData(nx, ny);
  let fmax = 1e-9;
  for (const v of data.frequency) if (v != null && v > fmax) fmax = v;
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      // Baseline row first in the data; canvas rows run top-down.
      const px = ((ny - 1 - iy) * nx + ix) * 4;
      // A power below 1 lifts the arc out from under the rim, which
      // otherwise takes most of the range for any player who drives.
      const t = Math.pow((data.frequency[iy * nx + ix] ?? 0) / fmax, 0.6);
      const { r, g, b } = heatColor(t);
      field.data[px] = r;
      field.data[px + 1] = g;
      field.data[px + 2] = b;
      field.data[px + 3] = Math.round(246 * Math.min(1, t / 0.12));
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = nx;
  tmp.height = ny;
  tmp.getContext("2d")!.putImageData(field, 0, 0);
  multiplyOnto(ctx, tmp, "blur(3px)");
}

/** How well: each zone filled in its own shape, by its adjusted gap to the
 *  league, faded where few of his shots are behind it. */
function paintZones(ctx: CanvasRenderingContext2D, zones: ShotMapZone[]) {
  const layer = document.createElement("canvas");
  layer.width = COURT_CW;
  layer.height = COURT_AREA_H;
  const lctx = layer.getContext("2d")!;
  for (const z of zones) {
    if (z.vs_league_adjusted == null || z.support <= 0) continue;
    const { r, g, b } = divergingColor(z.vs_league_adjusted / VS_LEAGUE_SCALE);
    fillZone(lctx, z.zone, `rgba(${r}, ${g}, ${b}, ${(0.96 * z.support).toFixed(3)})`);
  }
  multiplyOnto(ctx, layer, "blur(1.5px)");
}

// ── The court ────────────────────────────────────────────────────────────────
export function ShotHeatmap({
  data,
  metric,
  label,
}: {
  data: ShotMapData;
  metric: ShotMetric;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const ch = COURT_AREA_H + LEGEND_H;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, COURT_CW, ch);
    drawHardwood(ctx, ch);

    if (metric === "frequency") paintFrequency(ctx, data);
    else paintZones(ctx, data.zones);

    drawCourtLines(ctx);

    // Legend strip, on the same dark band the engine uses; the ramp is
    // composited against the floor so it shows the colours on the court.
    ctx.fillStyle = LEGEND_BG;
    ctx.fillRect(0, COURT_AREA_H, COURT_CW, LEGEND_H);
    const barX = 96;
    const barW = COURT_CW - barX - 64;
    const barY = COURT_AREA_H + 12;
    const ramp = metric === "frequency" ? HEAT_RAMP : DIVERGING_RAMP;
    const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
    ramp.forEach(([r, g, b], i) => grad.addColorStop(i / (ramp.length - 1), heatSwatch(r, g, b)));
    ctx.fillStyle = grad;
    ctx.fillRect(barX, barY, barW, 6);

    ctx.font = "9px JetBrains Mono, monospace";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#9BA8A0";
    const ticks =
      metric === "frequency"
        ? [
            ["RARELY", "left"],
            ["", "center"],
            ["OFTEN", "right"],
          ]
        : [
            ["−6 PTS", "left"],
            ["LEAGUE AVG", "center"],
            ["+6 PTS", "right"],
          ];
    ticks.forEach(([text, align], i) => {
      ctx.textAlign = align as CanvasTextAlign;
      ctx.fillText(text, barX + (barW * i) / 2, barY + 11);
    });
    ctx.textAlign = "right";
    ctx.fillStyle = "#71807A";
    ctx.fillText(metric === "frequency" ? "WHERE" : "FG% VS LG", barX - 10, barY + 11);
  }, [data, metric, ch]);

  return (
    <canvas
      ref={ref}
      width={COURT_CW}
      height={ch}
      role="img"
      aria-label={label}
      style={{ width: "100%", height: "auto", display: "block", borderRadius: 4 }}
    />
  );
}

// ── Zone table ───────────────────────────────────────────────────────────────
function signed(v: number | null) {
  if (v == null) return "—";
  const pts = v * 100;
  return `${pts > 0 ? "+" : pts < 0 ? "−" : ""}${Math.abs(pts).toFixed(1)}`;
}

function gapColor(v: number | null) {
  if (v == null || Math.abs(v) < 0.01) return C.muted;
  return v > 0 ? "#16A34A" : "#DC2626";
}

export function ZoneTable({
  data,
  distributions,
}: {
  data: ShotMapData;
  distributions?: Map<string, number[]>;
}) {
  // League ranks exist for the regular season only (the leaderboard's zone
  // columns), and only for players who meet that zone's attempt minimum.
  // League ranks are the leaderboard's: one regular season, qualified
  // players only (the page empties the distribution when he isn't).
  const ranked =
    data.season_type === "regular" && data.seasons.length === 1 && distributions != null;
  const rank = (z: ShotMapZone): ReactNode => {
    if (!ranked || z.fg_pct == null) return "—";
    const key = `zone_fg_pct_${ZONE_SLUG[z.zone]}`;
    const sorted = distributions!.get(key) ?? [];
    const pct = percentileOf(z.fg_pct, sorted, key);
    const r = rankOf(z.fg_pct, sorted, key);
    return r && pct != null ? <RankPill rank={r.rank} outOf={r.outOf} pct={pct} /> : "—";
  };
  return (
    <MiniTable
      headers={[
        "Zone",
        "FGA",
        "Of his shots",
        "FG%",
        "League",
        "+/− pts",
        "Adjusted",
        ...(ranked ? ["Rank"] : []),
      ]}
      rows={data.zones.map((z) => [
        ZONE_SHORT[z.zone] ?? z.zone,
        z.fga,
        formatStat(z.share, "pct"),
        formatStat(z.fg_pct, "pct"),
        formatStat(z.league_fg_pct, "pct"),
        <span style={{ color: gapColor(z.vs_league) }}>{signed(z.vs_league)}</span>,
        <span style={{ color: gapColor(z.vs_league_adjusted) }}>
          {signed(z.vs_league_adjusted)}
        </span>,
        ...(ranked ? [rank(z)] : []),
      ])}
    />
  );
}

// ── Basic-view card ──────────────────────────────────────────────────────────
const CAPTION: Record<ShotMetric, string> = {
  frequency:
    "Where his shots came from — darker means more often. Heaves from beyond 33 ft are counted but not drawn.",
  vs_league:
    "Each zone's FG% against the league's in that zone, adjusted for sample size: a zone is pulled toward league average in proportion to how few shots back it, because one season rarely separates a hot zone from a lucky one. Raw and adjusted gaps are in the table; three seasons gives the steadier picture.",
};

export function ShotMapCard({
  playerId,
  season,
  distributions,
  defaultMode = "chart",
}: {
  playerId: string;
  season: string;
  distributions?: Map<string, number[]>;
  defaultMode?: ViewMode;
}) {
  const [mode, setMode] = useViewMode(defaultMode);
  const [metric, setMetric] = useState<ShotMetric>("frequency");
  const [type, setType] = useState<"regular" | "playoffs">("regular");
  const [span, setSpan] = useState<"1" | "3">("1");
  const { data, missing, error } = useShotMap(
    `/stats/player/${encodeURIComponent(playerId)}/shot-map?season=${season}&season_type=${type}&span=${span}`,
  );

  // No regular-season shots this season at all: nothing to map, so no card.
  if (missing && type === "regular" && span === "1") return null;

  return (
    <Card style={{ padding: "16px 20px 18px", marginBottom: 16 }}>
      <ChartHeader title="Shot map" mode={mode} onChange={setMode} />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          margin: "10px 0 10px",
        }}
      >
        <Segmented
          value={metric}
          onChange={(v) => setMetric(v as ShotMetric)}
          options={[
            { value: "frequency", label: "WHERE HE SHOOTS" },
            { value: "vs_league", label: "FG% VS LEAGUE" },
          ]}
        />
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <Chips
            value={span}
            onChange={(v) => setSpan(v as "1" | "3")}
            options={[
              { value: "1", label: "This season" },
              { value: "3", label: "Last 3 seasons" },
            ]}
          />
          <Chips
            value={type}
            onChange={(v) => setType(v as "regular" | "playoffs")}
            options={[
              { value: "regular", label: "Regular season" },
              { value: "playoffs", label: "Playoffs" },
            ]}
          />
        </div>
      </div>
      <div
        style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5, maxWidth: 760 }}
      >
        {CAPTION[metric]}
        {data && (
          <span style={{ ...NUM, color: C.muted }}>
            {"  "}
            {data.attempts} shots ·{" "}
            {formatStat(data.attempts ? data.makes / data.attempts : null, "pct")} FG ·{" "}
            {data.seasons.length > 1 ? `${data.seasons[0]} to ${data.season}` : data.season}{" "}
            {type === "playoffs" ? "playoffs" : "regular season"}
          </span>
        )}
      </div>

      {error ? (
        <div style={{ color: C.red, fontSize: 12 }}>{error}</div>
      ) : missing ? (
        <div
          style={{
            fontFamily: F.mono,
            fontSize: 10.5,
            color: C.muted,
            letterSpacing: "0.2em",
            padding: "22px 0",
            textAlign: "center",
          }}
        >
          NO {type === "playoffs" ? "PLAYOFF" : "REGULAR-SEASON"} SHOTS
          {span === "3" ? " IN THE LAST 3 SEASONS" : ` IN ${season}`}
        </div>
      ) : !data ? (
        <div style={{ fontSize: 12, color: C.muted, padding: "18px 0" }}>Loading shot map…</div>
      ) : mode === "table" ? (
        <ZoneTable data={data} distributions={distributions} />
      ) : (
        <div
          className="se-rise"
          style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}
        >
          <div style={{ flex: "1 1 340px", maxWidth: 460 }}>
            <ShotHeatmap
              data={data}
              metric={metric}
              label={`${metric === "frequency" ? "Where he shoots" : "FG% compared with the league"}, ${data.season}`}
            />
          </div>
          <div style={{ flex: "1 1 300px", minWidth: 0 }}>
            <ZoneTable data={data} distributions={distributions} />
          </div>
        </div>
      )}
    </Card>
  );
}
