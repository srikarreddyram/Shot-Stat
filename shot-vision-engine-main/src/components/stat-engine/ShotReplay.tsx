import { useEffect, useRef, useState } from "react";
import { ShotList } from "../../lib/shot-maps";
import { drawCourtLines, drawHardwood } from "../shot-vision/engine/court/courtDrawing";
import {
  COURT_AREA_H,
  COURT_CW,
  HOOP_X,
  HOOP_Y,
  LEGEND_BG,
  LEGEND_H,
  courtToCanvas,
} from "../shot-vision/engine/court/geometry";
import { F } from "./ui";

// The shot replay: every shot of a player's season played back in the order
// he took it. Each ball leaves its spot, arcs to the rim (a shadow on the
// floor and a ball that grows with height give the arc in a top-down view),
// and leaves a mark where it was taken — green for a make, a red cross for a
// miss — until the floor holds the whole season. Under prefers-reduced-motion
// the finished chart is drawn straight away.

export type ShotFilter = "all" | "made" | "missed";

const MADE = "#1F8A5B";
const MISSED = "#C0392B";
const BALL = "#E07B39";
const BALL_EDGE = "#7A3A12";
const FLIGHT_MS = 540;

function drawMark(ctx: CanvasRenderingContext2D, x: number, y: number, made: boolean) {
  if (made) {
    ctx.beginPath();
    ctx.arc(x, y, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = MADE;
    ctx.fill();
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.stroke();
  } else {
    const r = 2.3;
    ctx.beginPath();
    ctx.moveTo(x - r, y - r);
    ctx.lineTo(x + r, y + r);
    ctx.moveTo(x + r, y - r);
    ctx.lineTo(x - r, y + r);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = MISSED;
    ctx.stroke();
  }
}

function drawBall(ctx: CanvasRenderingContext2D, ox: number, oy: number, p: number) {
  // Along the floor from the spot to the rim, with height as an arc.
  const gx = ox + (HOOP_X - ox) * p;
  const gy = oy + (HOOP_Y - oy) * p;
  const dist = Math.hypot(HOOP_X - ox, HOOP_Y - oy);
  const h = (18 + dist * 0.22) * 4 * p * (1 - p);
  ctx.beginPath();
  ctx.ellipse(gx, gy, 3.2, 1.6, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(40, 22, 8, 0.22)";
  ctx.fill();
  const r = 3.2 + h * 0.035;
  ctx.beginPath();
  ctx.arc(gx, gy - h, r, 0, Math.PI * 2);
  ctx.fillStyle = BALL;
  ctx.fill();
  ctx.lineWidth = 0.8;
  ctx.strokeStyle = BALL_EDGE;
  ctx.stroke();
}

function drawStrip(
  ctx: CanvasRenderingContext2D,
  made: number,
  taken: number,
  game: number,
  games: number,
  thinned: boolean,
  filter: ShotFilter,
) {
  const y = COURT_AREA_H;
  ctx.fillStyle = LEGEND_BG;
  ctx.fillRect(0, y, COURT_CW, LEGEND_H);
  ctx.font = "9px JetBrains Mono, monospace";
  ctx.textBaseline = "middle";
  const mid = y + LEGEND_H / 2;
  drawMark(ctx, 16, mid, true);
  ctx.fillStyle = "#9BA8A0";
  ctx.textAlign = "left";
  ctx.fillText("MADE", 24, mid);
  drawMark(ctx, 64, mid, false);
  ctx.fillStyle = "#9BA8A0";
  ctx.fillText("MISSED", 72, mid);
  ctx.textAlign = "center";
  ctx.fillStyle = "#71807A";
  ctx.fillText(games > 0 ? `GAME ${game} / ${games}` : "", COURT_CW / 2 + 12, mid);
  ctx.textAlign = "right";
  ctx.fillStyle = "#E9E4D8";
  // With one outcome filtered out, a make rate is meaningless (0/424):
  // count what is shown instead.
  const pct = taken ? ` · ${((made / taken) * 100).toFixed(1)}%` : "";
  const tally =
    filter === "all"
      ? `${made}/${taken}${pct}`
      : `${taken} ${filter === "made" ? "makes" : "misses"}`;
  ctx.fillText(`${tally}${thinned ? "*" : ""}`, COURT_CW - 12, mid);
}

export function ShotReplay({
  data,
  filter,
  label,
}: {
  data: ShotList;
  filter: ShotFilter;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [run, setRun] = useState(0);
  const ch = COURT_AREA_H + LEGEND_H;

  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;

    // The floor never changes during a replay: draw it once.
    const court = document.createElement("canvas");
    court.width = COURT_CW;
    court.height = COURT_AREA_H;
    const courtCtx = court.getContext("2d")!;
    drawHardwood(courtCtx, COURT_AREA_H);
    drawCourtLines(courtCtx);
    // Landed shots accumulate on their own layer, so a frame costs two
    // image copies plus the balls in the air, not a redraw of every mark.
    const marks = document.createElement("canvas");
    marks.width = COURT_CW;
    marks.height = COURT_AREA_H;
    const marksCtx = marks.getContext("2d")!;

    const shots = data.shots
      .filter(([, , made]) => filter === "all" || (filter === "made" ? made === 1 : made === 0))
      .map(([x, y, made, game]) => ({ ...courtToCanvas(x, y), made: made === 1, game }))
      .filter((s) => s.y >= 0 && s.y <= COURT_AREA_H);
    const n = shots.length;
    const thinned = data.shots.length < data.attempts;

    const paint = (landed: number, made: number, game: number) => {
      ctx.clearRect(0, 0, COURT_CW, ch);
      ctx.drawImage(court, 0, 0);
      ctx.drawImage(marks, 0, 0);
      drawStrip(ctx, made, landed, game, data.games, thinned, filter);
    };

    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || n === 0) {
      let made = 0;
      for (const s of shots) {
        drawMark(marksCtx, s.x, s.y, s.made);
        made += s.made ? 1 : 0;
      }
      paint(n, made, n ? shots[n - 1].game : 0);
      return;
    }

    // Launches spread over a few seconds whatever the season's size.
    const span = Math.min(7000, Math.max(2600, n * 7));
    const step = n > 1 ? span / (n - 1) : 0;
    const pulses: { t: number; made: boolean }[] = [];
    let landed = 0;
    let made = 0;
    let game = 0;
    let raf = 0;
    const start = performance.now();

    const frame = (now: number) => {
      const t = now - start;
      while (landed < n && t >= landed * step + FLIGHT_MS) {
        const s = shots[landed];
        drawMark(marksCtx, s.x, s.y, s.made);
        pulses.push({ t: landed * step + FLIGHT_MS, made: s.made });
        made += s.made ? 1 : 0;
        game = s.game;
        landed += 1;
      }
      paint(landed, made, game);
      for (let i = landed; i < n && i * step <= t; i++) {
        drawBall(ctx, shots[i].x, shots[i].y, Math.min(1, (t - i * step) / FLIGHT_MS));
      }
      // A ring at the rim as each ball arrives: green through, red off.
      while (pulses.length && t - pulses[0].t > 320) pulses.shift();
      for (const pulse of pulses.slice(-12)) {
        const k = (t - pulse.t) / 320;
        ctx.beginPath();
        ctx.arc(HOOP_X, HOOP_Y, 5 + k * 12, 0, Math.PI * 2);
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = pulse.made
          ? `rgba(31,138,91,${0.8 * (1 - k)})`
          : `rgba(192,57,43,${0.6 * (1 - k)})`;
        ctx.stroke();
      }
      if (landed < n || pulses.length) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [data, filter, run, ch]);

  return (
    <div style={{ position: "relative" }}>
      <canvas
        ref={ref}
        width={COURT_CW}
        height={ch}
        role="img"
        aria-label={label}
        style={{ width: "100%", height: "auto", display: "block", borderRadius: 4 }}
      />
      <button
        onClick={() => setRun((r) => r + 1)}
        aria-label="Replay the shots"
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          background: "rgba(10,10,15,0.72)",
          color: "#E9E4D8",
          border: "1px solid rgba(255,255,255,0.18)",
          borderRadius: 3,
          padding: "4px 9px",
          cursor: "pointer",
          fontFamily: F.mono,
          fontSize: 9.5,
          letterSpacing: "0.18em",
        }}
      >
        ↻ REPLAY
      </button>
    </div>
  );
}
