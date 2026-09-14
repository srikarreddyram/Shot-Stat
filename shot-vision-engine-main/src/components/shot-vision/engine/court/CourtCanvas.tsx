import { useEffect, useMemo, useRef, useState } from "react";
import { ZONES, type HeatmapPoint, type Player, type ZoneResult } from "@/lib/shot-vision-data";
import {
  COURT_AREA_H, COURT_CH, COURT_CW, COURT_LINE, COURT_LINE_STRONG, COURT_PLANK, COURT_GRAIN,
  COURT_RIM, COURT_SURFACE, CORNER_JOIN_Y, FIELD_REACH, FT, HEAT_RAMP, HOOP_X, HOOP_Y, LEGEND_BG,
  LEGEND_H, courtToCanvas, ft, heatColor, heatSwatch,
} from "./geometry";
import { drawMatchupBadge, useFontsReady, useHeadshot } from "./badgeDrawing";
import { ShotDetail } from "./ShotDetail";

export function CourtCanvas({ bestKey, results, heatmapPoints, showProb, attacker, defender, season, selected, setSelected }: { bestKey: string; results: ZoneResult[]; heatmapPoints: HeatmapPoint[]; showProb: boolean; attacker: Player; defender: Player; season: string; selected: number | null; setSelected: (i: number | null) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Click-to-inspect. Every grid point already carries its shot type,
  // distance and model outputs; before this they were only ever aggregated
  // into the heat field, so a specific shot could be seen but not read.
  // Lifted up to ResultsOutput (see there) so the ranked zone cards can
  // open the same panel a direct court click does.

  // Both players, on the floor. Without them the top of the court is a large
  // empty stretch of boards, and the chart carries no reminder of whose
  // matchup it is once you have scrolled the inputs off screen.
  const attackerImg = useHeadshot(attacker.headshotUrl);
  const defenderImg = useHeadshot(defender.headshotUrl);
  const fontsReady = useFontsReady();
  const [nearCursor, setNearCursor] = useState(false);

  // Canvas-space position of every point, computed once per result set and
  // reused for hit-testing so a click never re-runs the field interpolation.
  const plotted = useMemo(
    () => heatmapPoints.map((pt) => ({ pt, ...courtToCanvas(pt.loc_x, pt.loc_y) })),
    [heatmapPoints]
  );

  // A new run replaces the grid, so any previous pick is meaningless.
  useEffect(() => { setSelected(null); }, [heatmapPoints]);

  useEffect(() => {
    if (selected === null) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSelected(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

  // The canvas is 400px internally but rendered at width:100%, so pointer
  // coordinates have to be mapped through the displayed size, not the buffer.
  const toCanvasSpace = (e: { clientX: number; clientY: number }) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: ((e.clientX - rect.left) / rect.width) * COURT_CW,
      y: ((e.clientY - rect.top) / rect.height) * COURT_CH,
    };
  };

  // Matching the field's reach means click coverage equals painted coverage:
  // if a spot shows heat it is selectable, and bare floor still clears.
  // Mid-range samples sit up to 24px apart, so a tighter radius left dead
  // patches in the middle of bright blobs.
  const HIT_RADIUS = FIELD_REACH;
  const nearestTo = (x: number, y: number) => {
    let bestIdx = -1;
    let bestD2 = HIT_RADIUS * HIT_RADIUS;
    for (let i = 0; i < plotted.length; i++) {
      const dx = plotted[i].x - x;
      const dy = plotted[i].y - y;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bestD2) { bestD2 = d2; bestIdx = i; }
    }
    return bestIdx;
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = toCanvasSpace(e);
    if (!c) return;
    const i = nearestTo(c.x, c.y);
    // Clicking bare floor clears, so the card is never stuck open.
    setSelected(i >= 0 ? i : null);
  };

  const handleMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = toCanvasSpace(e);
    setNearCursor(!!c && nearestTo(c.x, c.y) >= 0);
  };

  // Centroid (in canvas px) of each zone's real grid points — used both to
  // place the aggregate EP/probability label and to anchor the best-zone
  // pulse ring, so both track the actual shot locations instead of
  // hand-picked coordinates.
  const centroids = useMemo(() => {
    const sums: Record<string, { sx: number; sy: number; n: number }> = {};
    for (const p of heatmapPoints) {
      const c = courtToCanvas(p.loc_x, p.loc_y);
      const s = sums[p.zone] || (sums[p.zone] = { sx: 0, sy: 0, n: 0 });
      s.sx += c.x;
      s.sy += c.y;
      s.n += 1;
    }
    const out: Record<string, { x: number; y: number }> = {};
    for (const [zone, s] of Object.entries(sums)) {
      out[zone] = { x: s.sx / s.n, y: s.sy / s.n };
    }
    return out;
  }, [heatmapPoints]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Canvas size
    const cw = COURT_CW;
    const ch = COURT_CH;

    // Filled in by the field pass, read by the legend so the scale ticks
    // show the values actually on screen rather than invented endpoints.
    let heatScale: { lo: number; hi: number } | null = null;

    // ── STEP 1: PLAYING SURFACE ───────────────────────────────────────────
    // Peach hardwood. An earlier version of this had a vertical gradient,
    // grain lines every 12px and plank seams at 25/50/75% — enough contrast
    // that the eye had to discard it before it could read the data. The wood
    // here is deliberately quiet: seams at a realistic plank width and a grain
    // whisper, both far below the weakest heat step, so the boards read as
    // texture and never as signal.
    ctx.fillStyle = COURT_SURFACE;
    ctx.fillRect(0, 0, cw, ch);

    // Planks run baseline to baseline, i.e. vertically in this view.
    const PLANK_W = ft(2.5);
    ctx.lineWidth = 1;
    ctx.strokeStyle = COURT_PLANK;
    for (let x = PLANK_W / 2; x < cw; x += PLANK_W) {
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, 0);
      ctx.lineTo(Math.round(x) + 0.5, COURT_AREA_H);
      ctx.stroke();
    }

    // Grain: short dashes along the planks, offset per plank so the boards
    // do not line up into visible rows.
    ctx.strokeStyle = COURT_GRAIN;
    ctx.setLineDash([9, 15]);
    let plankIndex = 0;
    for (let x = PLANK_W / 2; x < cw; x += PLANK_W) {
      ctx.lineDashOffset = (plankIndex % 4) * 7;
      ctx.beginPath();
      ctx.moveTo(Math.round(x - PLANK_W / 3) + 0.5, 0);
      ctx.lineTo(Math.round(x - PLANK_W / 3) + 0.5, COURT_AREA_H);
      ctx.stroke();
      plankIndex += 1;
    }
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;

    // ── STEP 2: SHOT-QUALITY FIELD ────────────────────────────────────────
    //
    // An earlier implementation splatted one additive radial blob per grid
    // point with `globalCompositeOperation = "screen"`. Overlapping blobs SUM,
    // so brightness tracked how densely the backend happened to sample a
    // region, not how good the shot was — mid-range (12 angles x 4 radii) blazed
    // while the corners (4 x 5, and far apart) stayed dim regardless of their
    // actual value. Per-zone blob radii were then hand-tuned to paper over the
    // gaps, which is a fix for the symptom.
    //
    // This computes a genuine scalar field instead: at each cell, a
    // distance-WEIGHTED AVERAGE (Shepard / inverse-distance interpolation) of
    // the nearby grid points. An average is invariant to how many points sit
    // nearby, so density drops out and brightness means value. No per-zone
    // tuning constants remain.
    const values = heatmapPoints.map((p) => (showProb ? p.make_probability : p.expected_points));

    if (values.length > 0) {
      const pts = heatmapPoints.map((p, i) => {
        const c = courtToCanvas(p.loc_x, p.loc_y);
        return { x: c.x, y: c.y, v: values[i] };
      });

      // Computed on a coarse lattice and upscaled with the browser's bilinear
      // smoothing — visually identical to per-pixel work at a fraction of the
      // cost, and the field is smooth by construction anyway.
      const CELL = 4;
      const gw = Math.ceil(cw / CELL);
      const gh = Math.ceil(ch / CELL);
      const field = ctx.createImageData(gw, gh);

      // Interpolation falloff. Also the honesty control: past this radius from
      // any real sample the field fades out rather than inventing values for
      // parts of the floor nobody shoots from.
      const REACH = FIELD_REACH;
      // Total kernel weight at which the field is fully opaque. Below it the
      // field fades out, so thinly-sampled edges of the court recede instead
      // of asserting a value.
      const WEIGHT_FULL = 1.6;

      // PASS 1 — interpolate. The ramp is scaled to the values that actually
      // end up ON the lattice, not to the raw point values: averaging pulls
      // peaks in, so scaling to the point maximum meant the top of the ramp
      // was never reached and the legend promised a red that never appeared.
      const cellValue = new Float32Array(gw * gh).fill(NaN);
      const cellAlpha = new Float32Array(gw * gh);
      let lo = Infinity;
      let hi = -Infinity;

      for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
          const px = gx * CELL + CELL / 2;
          const py = gy * CELL + CELL / 2;

          let wsum = 0;
          let vsum = 0;

          for (let i = 0; i < pts.length; i++) {
            const dx = px - pts[i].x;
            const dy = py - pts[i].y;
            const d2 = dx * dx + dy * dy;
            if (d2 > REACH * REACH) continue;
            // Gaussian weight — smoother than 1/d^2, and it cannot blow up
            // when a cell lands exactly on a sample.
            const w = Math.exp(-(d2) / (2 * (REACH / 2.2) ** 2));
            wsum += w;
            vsum += w * pts[i].v;
          }

          const cell = gy * gw + gx;
          if (wsum <= 0) continue;

          const value = vsum / wsum;
          cellValue[cell] = value;

          // Fade on TOTAL weight rather than distance to the nearest sample.
          // Nearest-distance gives every point its own circular cutoff, so the
          // boundary of the field is a union of circles and renders as a
          // scalloped, bubbly edge. Total weight decays smoothly and its
          // contours follow the shape of the sampled region, which is both
          // better looking and a more honest depiction of where the estimate
          // is actually supported.
          const edge = Math.max(0, Math.min(1, wsum / WEIGHT_FULL));
          // Cubed rather than squared: the field should die off quickly past
          // the last real sample. With a gentler falloff it flooded the whole
          // upper half of the court with colour in places nobody shoots from,
          // which reads as data and is not.
          cellAlpha[cell] = edge * edge * edge;

          // Only cells that are actually drawn get to set the scale — a cell
          // that fades to nothing should not stretch the legend.
          if (cellAlpha[cell] > 0.02) {
            if (value < lo) lo = value;
            if (value > hi) hi = value;
          }
        }
      }

      if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
      if (hi - lo < 1e-6) { lo -= 0.05; hi += 0.05; }

      // PASS 2 — colourise against the range the field actually spans.
      for (let cell = 0; cell < cellValue.length; cell++) {
        const idx = cell * 4;
        const value = cellValue[cell];
        if (Number.isNaN(value)) { field.data[idx + 3] = 0; continue; }
        const { r, g, b } = heatColor((value - lo) / (hi - lo));
        field.data[idx] = r;
        field.data[idx + 1] = g;
        field.data[idx + 2] = b;
        // Multiply preserves the grain at any alpha, so the field no longer has to
        // be held back to keep the boards visible.
        field.data[idx + 3] = Math.round(246 * cellAlpha[cell]);
      }

      // Blit the lattice, then let drawImage upscale it smoothly.
      const tmp = document.createElement("canvas");
      tmp.width = gw;
      tmp.height = gh;
      tmp.getContext("2d")!.putImageData(field, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      // The extra blur is what makes the blobs read as organic rather than as
      // an upscaled lattice. Applied to the field only, before the lines are
      // drawn, so the court markings stay crisp on top.
      ctx.filter = "blur(3.5px)";
      ctx.globalCompositeOperation = "multiply";
      ctx.drawImage(tmp, 0, 0, cw, ch);
      ctx.globalCompositeOperation = "source-over";
      ctx.filter = "none";

      heatScale = { lo, hi };
    }

    // ── STEP 3: COURT LINES ───────────────────────────────────────────────
    // Every line below is placed from the real dimensions in FT, through the
    // same transform the data uses, so the arc and the above-the-break cluster
    // cannot drift apart the way they did before.
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = COURT_LINE;
    ctx.lineWidth = 1.2;

    const baselineY = HOOP_Y + ft(FT.hoopFromBaseline);

    // THREE-POINT LINE: two straight corner segments joined by the arc.
    ctx.beginPath();
    ctx.moveTo(HOOP_X - ft(FT.cornerThreeX), baselineY);
    ctx.lineTo(HOOP_X - ft(FT.cornerThreeX), HOOP_Y - ft(CORNER_JOIN_Y));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(HOOP_X + ft(FT.cornerThreeX), baselineY);
    ctx.lineTo(HOOP_X + ft(FT.cornerThreeX), HOOP_Y - ft(CORNER_JOIN_Y));
    ctx.stroke();

    // The arc spans exactly the angles where it is outside the corner lines.
    const arcSweep = Math.acos(FT.cornerThreeX / FT.arcRadius);
    ctx.beginPath();
    ctx.arc(HOOP_X, HOOP_Y, ft(FT.arcRadius), Math.PI + arcSweep, -arcSweep, false);
    ctx.stroke();

    // PAINT
    ctx.strokeRect(
      HOOP_X - ft(FT.laneWidth / 2),
      baselineY - ft(FT.laneDepth),
      ft(FT.laneWidth),
      ft(FT.laneDepth),
    );

    // FREE-THROW CIRCLE — solid away from the basket, dashed toward it.
    const ftLineY = baselineY - ft(FT.laneDepth);
    ctx.beginPath();
    ctx.arc(HOOP_X, ftLineY, ft(FT.ftCircleRadius), Math.PI, 0, false);
    ctx.stroke();
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.arc(HOOP_X, ftLineY, ft(FT.ftCircleRadius), 0, Math.PI, false);
    ctx.stroke();
    ctx.setLineDash([]);

    // RESTRICTED AREA
    ctx.beginPath();
    ctx.arc(HOOP_X, HOOP_Y, ft(FT.restrictedRadius), Math.PI, 0, false);
    ctx.stroke();

    // BACKBOARD
    ctx.strokeStyle = COURT_LINE_STRONG;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(HOOP_X - ft(FT.backboardWidth / 2), HOOP_Y + ft(FT.backboardFromHoop));
    ctx.lineTo(HOOP_X + ft(FT.backboardWidth / 2), HOOP_Y + ft(FT.backboardFromHoop));
    ctx.stroke();

    // RIM
    ctx.strokeStyle = COURT_RIM;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(HOOP_X, HOOP_Y, ft(FT.rimRadius), 0, Math.PI * 2);
    ctx.stroke();

    // BASELINE + SIDELINES
    ctx.strokeStyle = COURT_LINE;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(1, baselineY);
    ctx.lineTo(cw - 1, baselineY);
    ctx.moveTo(1, 0); ctx.lineTo(1, baselineY);
    ctx.moveTo(cw - 1, 0); ctx.lineTo(cw - 1, baselineY);
    ctx.stroke();

    // The floor varnish and the per-zone EP numbers that used to be painted
    // here are both gone. The varnish was a wood effect with nothing left to
    // sit on, and stamping values onto the surface is the least SofaScore
    // thing on the chart — clicking a spot now gives the full readout, which
    // is strictly more information than five numbers baked into the image.

    // ── STEP 5b: MATCHUP BADGES ───────────────────────────────────────────
    // Backcourt corners: the two areas of floor the heat field never reaches,
    // since nobody shoots from there. Gold left / red right is the same
    // attacker/defender convention the rest of the page uses.
    const BADGE_R = 21;
    const BADGE_Y = 36;
    drawMatchupBadge(ctx, { cx: 14 + BADGE_R, cy: BADGE_Y, r: BADGE_R, side: "left", accent: "#C9A84C", ink: "#6B4E0A", role: "ATTACKER", name: attacker.name, img: attackerImg });
    drawMatchupBadge(ctx, { cx: cw - 14 - BADGE_R, cy: BADGE_Y, r: BADGE_R, side: "right", accent: "#DC2626", ink: "#8A1512", role: "DEFENDER", name: defender.name, img: defenderImg });

    // ── STEP 6: SCALE LEGEND ──────────────────────────────────────────────
    // An earlier legend was a bare gradient labelled "LOW EP" / "HIGH EP",
    // which tells a reader the direction but not the magnitude — two players'
    // charts looked identical whether they differed by 0.02 expected points
    // or 0.6. Heat is a multi-hue ramp, and the house rule is that it always
    // ships with a scale, so this one carries the real endpoints and midpoint.
    ctx.fillStyle = LEGEND_BG;
    ctx.fillRect(0, COURT_AREA_H, cw, LEGEND_H);

    const barX = 80;
    const barW = cw - barX * 2;
    const barY = COURT_AREA_H + 12;

    const legGrad = ctx.createLinearGradient(barX, barY, barX + barW, barY);
    for (let i = 0; i < HEAT_RAMP.length; i++) {
      const [r, g, b] = HEAT_RAMP[i];
      legGrad.addColorStop(i / (HEAT_RAMP.length - 1), heatSwatch(r, g, b));
    }
    ctx.fillStyle = legGrad;
    ctx.fillRect(barX, barY, barW, 6);

    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillStyle = "#9BA8A0";
    ctx.textBaseline = "top";

    const fmt = (v: number) => (showProb ? `${Math.round(v * 100)}%` : v.toFixed(2));
    if (heatScale) {
      ctx.textAlign = "left";
      ctx.fillText(fmt(heatScale.lo), barX, barY + 11);
      ctx.textAlign = "center";
      ctx.fillText(fmt((heatScale.lo + heatScale.hi) / 2), barX + barW / 2, barY + 11);
      ctx.textAlign = "right";
      ctx.fillText(fmt(heatScale.hi), barX + barW, barY + 11);
    }

    // One label, right-aligned into the gutter left of the bar. An earlier
    // version also drew "COLD" at x=8, which overlapped this text.
    ctx.textAlign = "right";
    ctx.fillStyle = "#71807A";
    ctx.fillText(showProb ? "MAKE %" : "EXP. PTS", barX - 10, barY + 11);

  }, [results, heatmapPoints, centroids, showProb, attacker, defender, attackerImg, defenderImg, fontsReady]);

  // STEP 7 - BEST ZONE PULSE RING, anchored to the real centroid of that
  // zone's grid points (falls back to a rough manual position if missing).
  const ringFallback: Record<string, { x: number; y: number }> = {
    restrictedArea: { x: 200, y: 318 },
    paintNonRA: { x: 200, y: 230 },
    leftCorner3: { x: 32, y: 310 },
    rightCorner3: { x: 368, y: 310 },
    aboveBreak3: { x: 200, y: 100 },
    midRange: { x: 200, y: 250 },
  };
  const bestZone = ZONES.find((z) => z.key === bestKey);
  const ring = (bestZone && centroids[bestZone.label]) ?? ringFallback[bestKey];
  const sel = selected !== null ? plotted[selected] : null;

  // The ranked zone cards below the court can also set `selected`, but this
  // panel renders right after the canvas — scrolled well above where a
  // click on the #2 or #3 card happened. Without this, picking one of
  // those looked like nothing happened.
  const detailRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selected !== null) detailRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected]);

  return (
    <>
    <div style={{ position: "relative", width: "100%", borderRadius: 3, overflow: "hidden", border: "1px solid rgba(255,255,255,0.05)" }}>
      <style>{`
        @keyframes pulseRing {
          0%, 100% { opacity: 1; transform: translate(-50%, -50%) scale(1); box-shadow: 0 0 8px #C9A84C; }
          50% { opacity: 0; transform: translate(-50%, -50%) scale(1.6); box-shadow: 0 0 24px #C9A84C; }
        }
      `}</style>
      <canvas
        ref={canvasRef}
        width={COURT_CW}
        height={COURT_CH}
        onClick={handleClick}
        onMouseMove={handleMove}
        onMouseLeave={() => setNearCursor(false)}
        role="img"
        aria-label={`Shot quality map for ${attacker.name} against ${defender.name}. Click a spot on the floor for that shot's numbers.`}
        style={{ width: "100%", height: "auto", display: "block", cursor: nearCursor ? "pointer" : "default" }}
      />
      {ring && (
        <div style={{
          position: "absolute",
          left: `${(ring.x / COURT_CW) * 100}%`,
          top: `${(ring.y / COURT_CH) * 100}%`,
          width: 44, height: 44,
          borderRadius: "50%",
          border: "2px solid #C9A84C",
          pointerEvents: "none",
          transformOrigin: "top left",
          animation: "pulseRing 2s ease-in-out infinite"
        }} />
      )}
      {sel && (
        <div style={{
          position: "absolute",
          left: `${(sel.x / COURT_CW) * 100}%`,
          top: `${(sel.y / COURT_CH) * 100}%`,
          width: 16, height: 16,
          marginLeft: -8, marginTop: -8,
          borderRadius: "50%",
          border: "2px solid #FFFFFF",
          boxShadow: "0 0 0 2px rgba(0,0,0,0.55), 0 0 12px rgba(255,255,255,0.5)",
          pointerEvents: "none",
        }} />
      )}
    </div>
    {sel && (
      <div ref={detailRef}>
      <ShotDetail
        point={sel.pt}
        attacker={attacker}
        defender={defender}
        showProb={showProb}
        season={season}
        onClose={() => setSelected(null)}
      />
      </div>
    )}
    </>
  );
}
