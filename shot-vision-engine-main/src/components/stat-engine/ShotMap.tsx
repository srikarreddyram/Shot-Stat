import { ReactNode, useEffect, useRef, useState } from "react";
import {
  ShotChange,
  ShotChangeZone,
  ShotGrid,
  ShotMapData,
  ShotMapGroup,
  ShotMapZone,
  ShotList,
  ShotMetric,
  SubsetMap,
  ZONE_SHORT,
  useShotMap,
} from "../../lib/shot-maps";
import { ShotFilter, ShotReplay } from "./ShotReplay";
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

// Shot maps on the Stat Engine profiles: where a player (or a team, or its
// opponents) shoots from, and how well in each zone compared with the
// league. The statistics (smoothing, the shrinkage toward the league, why
// efficiency is shown by zone rather than spot by spot, and the like-for-like
// league baselines) all live server-side in src/inference/shot_maps.py; this
// file only paints the results onto the same hardwood court the engine uses
// (courtDrawing.ts), with the same multiply tint so the wood shows through.

/** ±6 percentage points spans the whole diverging ramp: the true spread
 *  between players in any zone is only 3-5.5 points (shot_maps.py), so a
 *  wider scale would leave every map pale. Fixed rather than fitted per
 *  player, so the same colour means the same gap on every map. */
const VS_LEAGUE_SCALE = 0.06;

/** Below this many shots a subset's headline gap is shown but not coloured:
 *  two dunks are not a dunking record. */
const SMALL_SAMPLE = 25;

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
type Paint =
  | { kind: "frequency"; grid: ShotGrid; values: (number | null)[] }
  | { kind: "vs_league"; zones: ShotMapZone[] }
  | {
      kind: "change";
      grid: ShotGrid;
      values: (number | null)[];
      confidence: (number | null)[];
      scale: number;
    };

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

/** A per-square-foot field (baseline row first) → a small image, coloured
 *  and faded by `shade`, then blurred up onto the court. */
function paintField(
  ctx: CanvasRenderingContext2D,
  grid: ShotGrid,
  shade: (i: number) => { r: number; g: number; b: number; a: number } | null,
) {
  const { nx, ny } = grid;
  const field = ctx.createImageData(nx, ny);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const c = shade(iy * nx + ix);
      if (!c) continue;
      const px = ((ny - 1 - iy) * nx + ix) * 4; // canvas rows run top-down
      field.data[px] = c.r;
      field.data[px + 1] = c.g;
      field.data[px + 2] = c.b;
      field.data[px + 3] = Math.round(246 * c.a);
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = nx;
  tmp.height = ny;
  tmp.getContext("2d")!.putImageData(field, 0, 0);
  multiplyOnto(ctx, tmp, "blur(3px)");
}

function paint(ctx: CanvasRenderingContext2D, p: Paint) {
  if (p.kind === "frequency") {
    let max = 1e-9;
    for (const v of p.values) if (v != null && v > max) max = v;
    paintField(ctx, p.grid, (i) => {
      // A power below 1 lifts the arc out from under the rim, which
      // otherwise takes most of the range for any player who drives.
      const t = Math.pow((p.values[i] ?? 0) / max, 0.6);
      return { ...heatColor(t), a: Math.min(1, t / 0.12) };
    });
  } else if (p.kind === "change") {
    paintField(ctx, p.grid, (i) => {
      const t = (p.values[i] ?? 0) / Math.max(p.scale, 1e-9);
      // Only changes well beyond sampling noise show (shot_maps.py).
      return {
        ...divergingColor(Math.sign(t) * Math.pow(Math.min(1, Math.abs(t)), 0.6)),
        a: p.confidence[i] ?? 0,
      };
    });
  } else {
    // Each zone filled in its own shape, by its adjusted gap to the league,
    // faded where few shots are behind it.
    const layer = document.createElement("canvas");
    layer.width = COURT_CW;
    layer.height = COURT_AREA_H;
    const lctx = layer.getContext("2d")!;
    for (const z of p.zones) {
      if (z.vs_league_adjusted == null || z.support <= 0) continue;
      const { r, g, b } = divergingColor(z.vs_league_adjusted / VS_LEAGUE_SCALE);
      fillZone(lctx, z.zone, `rgba(${r}, ${g}, ${b}, ${(0.96 * z.support).toFixed(3)})`);
    }
    multiplyOnto(ctx, layer, "blur(1.5px)");
  }
}

const LEGEND: Record<
  Paint["kind"],
  { label: string; ticks: [string, string, string]; ramp: typeof HEAT_RAMP }
> = {
  frequency: { label: "WHERE", ticks: ["RARELY", "", "OFTEN"], ramp: HEAT_RAMP },
  vs_league: {
    label: "FG% VS LG",
    ticks: ["−6 PTS", "LEAGUE AVG", "+6 PTS"],
    ramp: DIVERGING_RAMP,
  },
  change: { label: "VS LAST YR", ticks: ["FEWER", "SAME", "MORE"], ramp: DIVERGING_RAMP },
};

function drawLegend(ctx: CanvasRenderingContext2D, kind: Paint["kind"]) {
  const { label, ticks, ramp } = LEGEND[kind];
  ctx.fillStyle = LEGEND_BG;
  ctx.fillRect(0, COURT_AREA_H, COURT_CW, LEGEND_H);
  const barX = 96;
  const barW = COURT_CW - barX - 64;
  const barY = COURT_AREA_H + 12;
  const grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
  ramp.forEach(([r, g, b], i) => grad.addColorStop(i / (ramp.length - 1), heatSwatch(r, g, b)));
  ctx.fillStyle = grad;
  ctx.fillRect(barX, barY, barW, 6);
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.textBaseline = "top";
  ctx.fillStyle = "#9BA8A0";
  (["left", "center", "right"] as const).forEach((align, i) => {
    ctx.textAlign = align;
    ctx.fillText(ticks[i], barX + (barW * i) / 2, barY + 11);
  });
  ctx.textAlign = "right";
  ctx.fillStyle = "#71807A";
  ctx.fillText(label, barX - 10, barY + 11);
}

/** The court with one map on it. `legend` adds the scale strip under the
 *  court; a gallery of small maps shares one legend instead. */
function CourtMap({
  paint: p,
  label,
  legend = true,
}: {
  paint: Paint;
  label: string;
  legend?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const ch = COURT_AREA_H + (legend ? LEGEND_H : 0);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, COURT_CW, ch);
    drawHardwood(ctx, ch);
    paint(ctx, p);
    drawCourtLines(ctx);
    if (legend) drawLegend(ctx, p.kind);
  }, [p, ch, legend]);
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

function paintFor(data: ShotMapData, metric: ShotMetric): Paint {
  return metric !== "vs_league"
    ? { kind: "frequency", grid: data.grid, values: data.frequency }
    : { kind: "vs_league", zones: data.zones };
}

/** The legend strip as page elements, for a gallery's shared scale. */
function LegendBar({ kind }: { kind: Paint["kind"] }) {
  const { label, ticks, ramp } = LEGEND[kind];
  const stops = ramp.map(
    ([r, g, b], i) => `${heatSwatch(r, g, b)} ${(i / (ramp.length - 1)) * 100}%`,
  );
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        maxWidth: 420,
        margin: "4px 0 14px",
      }}
    >
      <span
        style={{
          ...NUM,
          fontSize: 9,
          color: C.muted,
          letterSpacing: "0.12em",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
      <div style={{ flex: 1 }}>
        <div
          style={{
            height: 6,
            borderRadius: 1,
            background: `linear-gradient(to right, ${stops.join(", ")})`,
          }}
        />
        <div
          style={{
            ...NUM,
            display: "flex",
            justifyContent: "space-between",
            fontSize: 8.5,
            color: C.muted,
            marginTop: 4,
          }}
        >
          {ticks.map((t, i) => (
            <span key={i}>{t}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Numbers ──────────────────────────────────────────────────────────────────
function signed(v: number | null) {
  if (v == null) return "—";
  const pts = v * 100;
  return `${pts > 0 ? "+" : pts < 0 ? "−" : ""}${Math.abs(pts).toFixed(1)}`;
}

function gapColor(v: number | null, attempts = Infinity) {
  if (v == null || Math.abs(v) < 0.01 || attempts < SMALL_SAMPLE) return C.muted;
  return v > 0 ? "#16A34A" : "#DC2626";
}

export function ZoneTable({
  data,
  distributions,
}: {
  data: ShotMapData;
  distributions?: Map<string, number[]>;
}) {
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
        "Of shots",
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

const muted: React.CSSProperties = {
  fontFamily: F.mono,
  fontSize: 10.5,
  color: C.muted,
  letterSpacing: "0.2em",
  padding: "22px 0",
  textAlign: "center",
};

// ── One shot map, with its controls (basic view) ────────────────────────────
type Range = { type: "regular" | "playoffs"; span: "1" | "3" };

function seasonsLabel(data: { season: string; seasons: string[] }, type: Range["type"]) {
  const range = data.seasons.length > 1 ? `${data.seasons[0]} to ${data.season}` : data.season;
  return `${range} ${type === "playoffs" ? "playoffs" : "regular season"}`;
}

function RangeChips({ range, onChange }: { range: Range; onChange: (r: Range) => void }) {
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
      <Chips
        value={range.span}
        onChange={(v) => onChange({ ...range, span: v as Range["span"] })}
        options={[
          { value: "1", label: "This season" },
          { value: "3", label: "Last 3 seasons" },
        ]}
      />
      <Chips
        value={range.type}
        onChange={(v) => onChange({ ...range, type: v as Range["type"] })}
        options={[
          { value: "regular", label: "Regular season" },
          { value: "playoffs", label: "Playoffs" },
        ]}
      />
    </div>
  );
}

function ShotMapPanel({
  title,
  pathFor,
  shotsPathFor,
  captions,
  noun,
  distributions,
  defaultMode = "chart",
  extra,
}: {
  title: string;
  pathFor: (range: Range) => string;
  /** Where the individual shots come from, for the SHOTS replay; without it
   *  the replay is not offered. */
  shotsPathFor?: (range: Range) => string;
  captions: Partial<Record<ShotMetric, string>>;
  /** "WHERE HE SHOOTS" / "WHERE THEY SHOOT" */
  noun: string;
  distributions?: Map<string, number[]>;
  defaultMode?: ViewMode;
  extra?: ReactNode;
}) {
  const [mode, setMode] = useViewMode(defaultMode);
  const [metric, setMetric] = useState<ShotMetric>(shotsPathFor ? "shots" : "frequency");
  const [range, setRange] = useState<Range>({ type: "regular", span: "1" });
  const [filter, setFilter] = useState<ShotFilter>("all");
  const { data, missing, error } = useShotMap(pathFor(range));
  const shots = useShotMap<ShotList>(
    metric === "shots" && shotsPathFor ? shotsPathFor(range) : null,
  );

  // No regular-season shots this season at all: nothing to map, so no card.
  if (missing && range.type === "regular" && range.span === "1" && !extra) return null;

  return (
    <Card style={{ padding: "16px 20px 18px", marginBottom: 16 }}>
      <ChartHeader title={title} mode={mode} onChange={setMode} />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          margin: "10px 0",
        }}
      >
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          {extra}
          <Segmented
            value={metric}
            onChange={(v) => setMetric(v as ShotMetric)}
            options={[
              ...(shotsPathFor ? [{ value: "shots", label: "SHOTS" }] : []),
              { value: "frequency", label: noun },
              { value: "vs_league", label: "FG% VS LEAGUE" },
            ]}
          />
        </div>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          {metric === "shots" && (
            <Chips
              value={filter}
              onChange={(v) => setFilter(v as ShotFilter)}
              options={[
                { value: "all", label: "All" },
                { value: "made", label: "Makes" },
                { value: "missed", label: "Misses" },
              ]}
            />
          )}
          <RangeChips range={range} onChange={setRange} />
        </div>
      </div>
      <div
        style={{ fontSize: 11, color: C.faint, marginBottom: 12, lineHeight: 1.5, maxWidth: 780 }}
      >
        {captions[metric]}
        {data && (
          <span style={{ ...NUM, color: C.muted }}>
            {"  "}
            {data.attempts.toLocaleString()} shots · {formatStat(data.fg_pct, "pct")} FG ·{" "}
            {seasonsLabel(data, range.type)}
          </span>
        )}
      </div>

      {error ? (
        <div style={{ color: C.red, fontSize: 12 }}>{error}</div>
      ) : missing ? (
        <div style={muted}>NO SHOTS FOR THIS SELECTION</div>
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
            {metric === "shots" ? (
              shots.data ? (
                <ShotReplay
                  data={shots.data}
                  filter={filter}
                  label={`${title}: every shot, replayed in order`}
                />
              ) : (
                <div style={{ ...muted, padding: "120px 0" }}>
                  {shots.error ?? "LOADING SHOTS…"}
                </div>
              )
            ) : (
              <CourtMap
                paint={paintFor(data, metric)}
                label={`${title}: ${metric === "frequency" ? "shot locations" : "FG% vs league by zone"}`}
              />
            )}
          </div>
          <div style={{ flex: "1 1 300px", minWidth: 0 }}>
            <ZoneTable data={data} distributions={distributions} />
          </div>
        </div>
      )}
    </Card>
  );
}

const VS_LEAGUE_CAPTION = (who: string) =>
  `Each zone's FG% against the league's in that zone, adjusted for sample size: a zone is pulled toward league average in proportion to how few shots back it, because one season rarely separates a hot zone from a lucky one. Raw and adjusted gaps are in the table; ${who}`;

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
  return (
    <ShotMapPanel
      title="Shot map"
      noun="WHERE HE SHOOTS"
      distributions={distributions}
      defaultMode={defaultMode}
      pathFor={(r) =>
        `/stats/player/${encodeURIComponent(playerId)}/shot-map?season=${season}&season_type=${r.type}&span=${r.span}`
      }
      shotsPathFor={(r) =>
        `/stats/player/${encodeURIComponent(playerId)}/shots?season=${season}&season_type=${r.type}&span=${r.span}`
      }
      captions={{
        shots:
          "Every shot, replayed in the order he took it: each ball flies from its spot to the rim and leaves a green dot if it went in, a red cross if it didn't.",
        frequency:
          "Where his shots came from — darker means more often. Heaves from beyond 33 ft are counted but not drawn.",
        vs_league: VS_LEAGUE_CAPTION("three seasons gives the steadier picture."),
      }}
    />
  );
}

export function TeamShotMapCard({
  teamId,
  season,
  defaultMode = "chart",
}: {
  teamId: string;
  season: string;
  defaultMode?: ViewMode;
}) {
  const [side, setSide] = useState<"offense" | "defense">("offense");
  return (
    <ShotMapPanel
      title={side === "offense" ? "Shot map — their shots" : "Shot map — opponents' shots"}
      noun={side === "offense" ? "WHERE THEY SHOOT" : "WHERE OPPONENTS SHOOT"}
      defaultMode={defaultMode}
      extra={
        <Segmented
          value={side}
          onChange={(v) => setSide(v as "offense" | "defense")}
          options={[
            { value: "offense", label: "OFFENSE" },
            { value: "defense", label: "DEFENSE" },
          ]}
        />
      }
      pathFor={(r) =>
        `/stats/team/${encodeURIComponent(teamId)}/shot-map?season=${season}&side=${side}&season_type=${r.type}&span=${r.span}`
      }
      shotsPathFor={(r) =>
        `/stats/team/${encodeURIComponent(teamId)}/shots?season=${season}&side=${side}&season_type=${r.type}&span=${r.span}`
      }
      captions={
        side === "offense"
          ? {
              shots:
                "Their season, shot by shot: each ball flies from its spot to the rim and leaves a green dot if it went in, a red cross if it didn't (a full season is thinned evenly to 2,500 shots).",
              frequency: "Where the team's shots came from — darker means more often.",
              vs_league: VS_LEAGUE_CAPTION(
                "a team's thousands of shots mean the two barely differ.",
              ),
            }
          : {
              shots:
                "Every shot opponents took against them, replayed in order — green went in, red didn't (thinned evenly to 2,500 shots).",
              frequency: "Where opponents shot from against them — darker means more often.",
              vs_league: VS_LEAGUE_CAPTION(
                "for a defense, blue (opponents below league average) is good.",
              ),
            }
      }
    />
  );
}

// ── More shot maps (advanced view) ──────────────────────────────────────────
type GalleryTab = "shot_types" | "clutch" | "defense" | "change";

const GALLERY_CAPTION: Record<GalleryTab, string> = {
  shot_types:
    "His shots split by the league's own play-by-play shot type. Each kind is compared with the league on the same kind — pull-ups against pull-ups — and 'expected' is what an average player makes on his exact mix of those shots and zones.",
  clutch:
    "Clutch time is the last five minutes of the fourth quarter or overtime with the score within five. Both maps are compared with the league in the same situation. Clutch samples are small; three seasons helps.",
  defense:
    "Shots by the player he guarded most in each game, compared with the league from the same zones — blue means his assignments shot worse than usual. The NBA assigns the matchup per game, not per shot, so this shows where his assignments shot, not who contested each one.",
  change:
    "Where his shots moved since last season: red where a bigger share of his shots now comes from, blue where a smaller one. A spot is only coloured when the change is well beyond what two samples of the same shooter would show by chance (2+ standard errors).",
};

function MiniMap({ map, metric }: { map: SubsetMap; metric: ShotMetric }) {
  const thin = map.attempts < SMALL_SAMPLE;
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 8,
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 12.5, color: C.text, fontWeight: 600 }}>{map.label}</span>
        <span style={{ ...NUM, fontSize: 10, color: C.muted }}>
          {formatStat(map.share, "pct")} of shots
        </span>
      </div>
      {map.attempts === 0 ? (
        <div
          style={{ ...muted, border: `1px dashed ${C.edge}`, borderRadius: 4, padding: "60px 0" }}
        >
          NONE
        </div>
      ) : (
        <CourtMap
          paint={paintFor(map, metric)}
          label={`${map.label}: ${map.note}`}
          legend={false}
        />
      )}
      <div style={{ ...NUM, fontSize: 10.5, color: C.dim, marginTop: 6, lineHeight: 1.5 }}>
        {map.attempts} shots · {formatStat(map.fg_pct, "pct")}
        {map.expected_fg_pct != null && (
          <span style={{ color: C.muted }}> (exp. {formatStat(map.expected_fg_pct, "pct")})</span>
        )}{" "}
        <span style={{ color: gapColor(map.vs_expected, map.attempts) }}>
          {signed(map.vs_expected)}
        </span>
        {thin && map.attempts > 0 && <span style={{ color: C.muted }}> · small sample</span>}
      </div>
      <div style={{ fontSize: 10.5, color: C.faint, marginTop: 2, lineHeight: 1.45 }}>
        {map.note}
      </div>
    </div>
  );
}

function ChangeView({ data, mode }: { data: ShotChange; mode: ViewMode }) {
  const table = (
    <MiniTable
      headers={[
        "Zone",
        `${data.previous_season} share`,
        `${data.season} share`,
        "Change",
        "FG% then",
        "FG% now",
      ]}
      rows={data.zones.map((z: ShotChangeZone) => [
        ZONE_SHORT[z.zone] ?? z.zone,
        formatStat(z.previous_share, "pct"),
        formatStat(z.share, "pct"),
        <span
          style={{
            color:
              Math.abs(z.share_change) < 0.01
                ? C.muted
                : z.share_change > 0
                  ? "#B4491C"
                  : "#5E86C4",
          }}
        >
          {signed(z.share_change)}
        </span>,
        formatStat(z.previous_fg_pct, "pct"),
        formatStat(z.fg_pct, "pct"),
      ])}
    />
  );
  if (mode === "table") return table;
  return (
    <div
      className="se-rise"
      style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}
    >
      <div style={{ flex: "1 1 340px", maxWidth: 460 }}>
        <CourtMap
          paint={{
            kind: "change",
            grid: data.grid,
            values: data.change,
            confidence: data.confidence,
            scale: data.scale,
          }}
          label={`Change in shot locations from ${data.previous_season} to ${data.season}`}
        />
      </div>
      <div style={{ flex: "1 1 300px", minWidth: 0 }}>
        <div style={{ ...NUM, fontSize: 10.5, color: C.muted, marginBottom: 8 }}>
          {data.previous_attempts} shots in {data.previous_season} → {data.attempts} in{" "}
          {data.season}
        </div>
        {table}
      </div>
    </div>
  );
}

export function ShotMapGallery({
  playerId,
  season,
  defaultMode = "chart",
}: {
  playerId: string;
  season: string;
  defaultMode?: ViewMode;
}) {
  const [mode, setMode] = useViewMode(defaultMode);
  const [tab, setTab] = useState<GalleryTab>("shot_types");
  const [metric, setMetric] = useState<Exclude<ShotMetric, "shots">>("vs_league");
  const [range, setRange] = useState<Range>({ type: "regular", span: "1" });
  const id = encodeURIComponent(playerId);
  const group = useShotMap<ShotMapGroup>(
    tab === "change"
      ? null
      : `/stats/player/${id}/shot-maps?group=${tab}&season=${season}&season_type=${range.type}&span=${range.span}`,
  );
  const change = useShotMap<ShotChange>(
    tab === "change"
      ? `/stats/player/${id}/shot-map/change?season=${season}&season_type=${range.type}`
      : null,
  );
  const state = tab === "change" ? change : group;

  return (
    <Card style={{ padding: "16px 20px 18px", marginBottom: 16 }}>
      <ChartHeader title="More shot maps" mode={mode} onChange={setMode} />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          margin: "10px 0",
        }}
      >
        <Segmented
          value={tab}
          onChange={(v) => setTab(v as GalleryTab)}
          options={[
            { value: "shot_types", label: "SHOT TYPES" },
            { value: "clutch", label: "CLUTCH" },
            { value: "defense", label: "DEFENSE" },
            { value: "change", label: "VS LAST SEASON" },
          ]}
        />
        {tab !== "change" && (
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            <Chips
              value={metric}
              onChange={(v) => setMetric(v as Exclude<ShotMetric, "shots">)}
              options={[
                { value: "frequency", label: "Where" },
                { value: "vs_league", label: "FG% vs league" },
              ]}
            />
            <RangeChips range={range} onChange={setRange} />
          </div>
        )}
      </div>
      <div
        style={{ fontSize: 11, color: C.faint, marginBottom: 8, lineHeight: 1.5, maxWidth: 780 }}
      >
        {GALLERY_CAPTION[tab]}
        {group.data && tab !== "change" && (
          <span style={{ ...NUM, color: C.muted }}>
            {"  "}
            {seasonsLabel(group.data, range.type)}
          </span>
        )}
      </div>

      {state.error ? (
        <div style={{ color: C.red, fontSize: 12 }}>{state.error}</div>
      ) : state.missing ? (
        <div style={muted}>
          {tab === "change" ? "NEEDS SHOTS IN BOTH SEASONS" : "NO SHOTS FOR THIS SELECTION"}
        </div>
      ) : tab === "change" ? (
        change.data ? (
          <ChangeView data={change.data} mode={mode} />
        ) : (
          <div style={{ fontSize: 12, color: C.muted, padding: "18px 0" }}>Loading…</div>
        )
      ) : !group.data ? (
        <div style={{ fontSize: 12, color: C.muted, padding: "18px 0" }}>Loading shot maps…</div>
      ) : mode === "table" ? (
        <MiniTable
          headers={["Map", "Shots", "Of shots", "FG%", "Expected", "+/− pts"]}
          rows={group.data.maps.map((m) => [
            m.label,
            m.attempts,
            formatStat(m.share, "pct"),
            formatStat(m.fg_pct, "pct"),
            formatStat(m.expected_fg_pct, "pct"),
            <span style={{ color: gapColor(m.vs_expected, m.attempts) }}>
              {signed(m.vs_expected)}
            </span>,
          ])}
        />
      ) : (
        <>
          <LegendBar kind={metric} />
          <div
            className="se-rise"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))",
              gap: 18,
            }}
          >
            {group.data.maps.map((m) => (
              <MiniMap key={m.key} map={m} metric={metric} />
            ))}
          </div>
        </>
      )}
    </Card>
  );
}

// ── Compare: shot maps side by side ─────────────────────────────────────────
function ComparedMap({
  playerId,
  name,
  season,
  color,
  metric,
}: {
  playerId: string;
  name: string;
  season: string;
  color: string;
  metric: Exclude<ShotMetric, "shots">;
}) {
  const { data, missing, error } = useShotMap(
    `/stats/player/${encodeURIComponent(playerId)}/shot-map?season=${season}`,
  );
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ width: 8, height: 8, borderRadius: 2, background: color, flexShrink: 0 }} />
        <span style={{ fontSize: 12.5, color: C.text, fontWeight: 600 }}>{name}</span>
      </div>
      {error ? (
        <div style={{ color: C.red, fontSize: 12 }}>{error}</div>
      ) : missing ? (
        <div
          style={{ ...muted, border: `1px dashed ${C.edge}`, borderRadius: 4, padding: "60px 0" }}
        >
          NO SHOTS
        </div>
      ) : !data ? (
        <div style={{ ...muted, padding: "80px 0" }}>LOADING…</div>
      ) : (
        <>
          <div style={{ borderRadius: 4, boxShadow: `0 0 0 1px ${color}55` }}>
            <CourtMap
              paint={paintFor(data, metric)}
              label={`${name}: ${metric === "frequency" ? "shot locations" : "FG% vs league by zone"}`}
              legend={false}
            />
          </div>
          <div style={{ ...NUM, fontSize: 10.5, color: C.dim, marginTop: 6, lineHeight: 1.5 }}>
            {data.attempts} shots · {formatStat(data.fg_pct, "pct")}
            {data.expected_fg_pct != null && (
              <span style={{ color: C.muted }}>
                {" "}
                (exp. {formatStat(data.expected_fg_pct, "pct")})
              </span>
            )}{" "}
            <span style={{ color: gapColor(data.vs_expected, data.attempts) }}>
              {signed(data.vs_expected)}
            </span>
          </div>
          <div style={{ ...NUM, fontSize: 9.5, color: C.muted, marginTop: 2 }}>
            {data.zones
              .filter((z) => z.share >= 0.15)
              .sort((a, b) => b.share - a.share)
              .map((z) => `${ZONE_SHORT[z.zone] ?? z.zone} ${formatStat(z.share, "pct")}`)
              .join(" · ")}
          </div>
        </>
      )}
    </div>
  );
}

/** Two or three players' shot maps on one shared scale — where each shoots,
 *  or how well by zone. */
export function CompareShotMaps({
  players,
  season,
}: {
  players: { id: string; name: string; color: string }[];
  season: string;
}) {
  const [metric, setMetric] = useState<Exclude<ShotMetric, "shots">>("frequency");
  return (
    <Card style={{ padding: "16px 20px 18px", marginBottom: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <SectionLabelLike>Shot maps</SectionLabelLike>
        <Chips
          value={metric}
          onChange={(v) => setMetric(v as Exclude<ShotMetric, "shots">)}
          options={[
            { value: "frequency", label: "Where" },
            { value: "vs_league", label: "FG% vs league" },
          ]}
        />
      </div>
      <div
        style={{
          fontSize: 11,
          color: C.faint,
          margin: "8px 0 4px",
          lineHeight: 1.5,
          maxWidth: 760,
        }}
      >
        {metric === "frequency"
          ? `Where each player's ${season} shots came from — darker means more often. The line under each map is his share of shots from his main zones.`
          : "Each zone's FG% against the league's there, adjusted for sample size (see a player's own shot map for the raw numbers)."}
      </div>
      <LegendBar kind={metric} />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 18,
        }}
      >
        {players.map((p) => (
          <ComparedMap
            key={p.id}
            playerId={p.id}
            name={p.name}
            season={season}
            color={p.color}
            metric={metric}
          />
        ))}
      </div>
    </Card>
  );
}

function SectionLabelLike({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontFamily: F.mono,
        fontSize: 10,
        color: C.gold,
        letterSpacing: "0.4em",
        textTransform: "uppercase",
      }}
    >
      {children}
    </div>
  );
}
