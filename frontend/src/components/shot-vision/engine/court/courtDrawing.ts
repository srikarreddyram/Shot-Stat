// The hardwood floor and the court markings, shared by every court in the
// app: the engine's shot-quality map (CourtCanvas) and the Stat Engine's
// shot maps. Moved here verbatim from CourtCanvas so both draw the identical
// court; geometry.ts holds the dimensions both are drawn from.
import {
  COURT_AREA_H,
  COURT_CW,
  COURT_GRAIN,
  COURT_LINE,
  COURT_LINE_STRONG,
  COURT_PLANK,
  COURT_RIM,
  COURT_SURFACE,
  CORNER_JOIN_Y,
  FT,
  HOOP_X,
  HOOP_Y,
  ft,
} from "./geometry";

/** Peach hardwood over the whole canvas (`ch` tall), planks over the court area. */
export function drawHardwood(ctx: CanvasRenderingContext2D, ch: number) {
  const cw = COURT_CW;
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
}

/** Three-point line, paint, free-throw circle, restricted area, backboard,
 *  rim, baseline and sidelines — drawn on top of whatever field is below. */
export function drawCourtLines(ctx: CanvasRenderingContext2D) {
  const cw = COURT_CW;
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
  ctx.moveTo(1, 0);
  ctx.lineTo(1, baselineY);
  ctx.moveTo(cw - 1, 0);
  ctx.lineTo(cw - 1, baselineY);
  ctx.stroke();
}

/**
 * Fill one of the league's six shot zones (the shots table's `zone` names),
 * traced from the same dimensions drawCourtLines uses, so a filled zone
 * meets its lines exactly: restricted area within 4 ft of the hoop, the
 * paint the 16-ft lane up to the free-throw line, the corners below the
 * point where the corner line meets the arc, and the arc itself.
 */
export function fillZone(ctx: CanvasRenderingContext2D, zone: string, fill: string) {
  const baselineY = HOOP_Y + ft(FT.hoopFromBaseline);
  const cornerTopY = HOOP_Y - ft(CORNER_JOIN_Y);
  const ftLineY = baselineY - ft(FT.laneDepth);
  const laneLeft = HOOP_X - ft(FT.laneWidth / 2);
  const arcSweep = Math.acos(FT.cornerThreeX / FT.arcRadius);
  const p = new Path2D();
  const restricted = () => {
    p.moveTo(HOOP_X + ft(FT.restrictedRadius), HOOP_Y);
    p.arc(HOOP_X, HOOP_Y, ft(FT.restrictedRadius), 0, Math.PI * 2);
  };
  const lane = () => p.rect(laneLeft, ftLineY, ft(FT.laneWidth), baselineY - ftLineY);

  ctx.save();
  ctx.fillStyle = fill;
  switch (zone) {
    case "Restricted Area":
      restricted();
      ctx.fill(p);
      break;
    case "In The Paint (Non-RA)":
      lane();
      restricted();
      ctx.fill(p, "evenodd");
      break;
    case "Left Corner 3":
      p.rect(0, cornerTopY, HOOP_X - ft(FT.cornerThreeX), baselineY - cornerTopY);
      ctx.fill(p);
      break;
    case "Right Corner 3":
      p.rect(
        HOOP_X + ft(FT.cornerThreeX),
        cornerTopY,
        COURT_CW - HOOP_X - ft(FT.cornerThreeX),
        baselineY - cornerTopY,
      );
      ctx.fill(p);
      break;
    case "Above the Break 3": {
      // Everything above the corners that is outside the arc.
      const above = new Path2D();
      above.rect(0, 0, COURT_CW, cornerTopY);
      ctx.clip(above);
      p.rect(0, 0, COURT_CW, cornerTopY);
      p.moveTo(HOOP_X + ft(FT.arcRadius), HOOP_Y);
      p.arc(HOOP_X, HOOP_Y, ft(FT.arcRadius), 0, Math.PI * 2);
      ctx.fill(p, "evenodd");
      break;
    }
    case "Mid-Range":
      // Inside the three-point line, outside the lane.
      p.moveTo(HOOP_X - ft(FT.cornerThreeX), baselineY);
      p.lineTo(HOOP_X - ft(FT.cornerThreeX), cornerTopY);
      p.arc(HOOP_X, HOOP_Y, ft(FT.arcRadius), Math.PI + arcSweep, -arcSweep, false);
      p.lineTo(HOOP_X + ft(FT.cornerThreeX), baselineY);
      p.closePath();
      lane();
      ctx.fill(p, "evenodd");
      break;
  }
  ctx.restore();
}
