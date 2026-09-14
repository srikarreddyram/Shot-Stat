// ─────────────────────────────────────────────────────────────────────────────
// Court geometry — derived from real NBA dimensions, in FEET from the hoop.
//
// An earlier version drew the court from invented proportions (`cw * 0.42`,
// `ch * 0.35`, …) while plotting the heatmap from the backend's real
// coordinates. The two could never line up, and they didn't: the three-point
// arc was drawn at 21.0 ft against a real 23.75 ft, so every above-the-break
// blob rendered 22-48px OUTSIDE the arc; the free-throw line sat 11 ft too far
// out and the paint was 8 ft too deep.
//
// Everything below now comes from one scale and one origin, so the lines and
// the data are drawn in the same space by construction.
// ─────────────────────────────────────────────────────────────────────────────
export const FT = {
  courtWidth: 50,        // sideline to sideline
  hoopFromBaseline: 5.25,
  rimRadius: 0.75,
  backboardFromHoop: 1.25,   // behind the hoop centre
  backboardWidth: 6,
  restrictedRadius: 4,
  laneWidth: 16,
  laneDepth: 19,             // baseline to the free-throw line
  ftCircleRadius: 6,
  cornerThreeX: 22,          // corner-3 line, from the centre of the hoop
  arcRadius: 23.75,          // above-the-break three
  visibleDepth: 38,          // how far up-court to render
};

// The corner-3 line runs from the baseline until it meets the arc.
export const CORNER_JOIN_Y = Math.sqrt(FT.arcRadius ** 2 - FT.cornerThreeX ** 2); // ≈ 8.95 ft

export const COURT_CW = 400;
export const COURT_SCALE = COURT_CW / FT.courtWidth;   // 8 px per foot
export const COURT_AREA_H = Math.round(FT.visibleDepth * COURT_SCALE);

// A dedicated strip below the baseline for the scale legend. The legend used
// to be drawn at y=345 on a 375px canvas with the court occupying all of it,
// so it sat on top of the floor near the baseline; giving it its own band keeps
// it off the data and lets it carry real tick values.
export const LEGEND_H = 38;
export const COURT_CH = COURT_AREA_H + LEGEND_H;

export const HOOP_X = COURT_CW / 2;
export const HOOP_Y = COURT_AREA_H - FT.hoopFromBaseline * COURT_SCALE;

export const ft = (feet: number) => feet * COURT_SCALE;

// How far the heat field reaches from a real sample point. This is the single
// source of truth for it: the renderer fades the field out past this radius,
// and click hit-testing uses the same number so that anywhere the map shows
// colour is somewhere you can click. They were separate values briefly, and
// the result was large visibly-hot regions that silently ignored clicks.
export const FIELD_REACH = ft(7.0);

// Peach hardwood. Markings are dark on a light floor, the way a real court
// is painted — light lines would disappear into the boards.
export const COURT_SURFACE = "#DCB489";
export const COURT_PLANK = "rgba(140, 95, 52, 0.22)";
export const COURT_GRAIN = "rgba(112, 74, 40, 0.075)";
export const COURT_LINE = "rgba(92, 58, 30, 0.5)";
export const COURT_LINE_STRONG = "rgba(58, 35, 16, 0.85)";
export const COURT_RIM = "#B4491C";
export const LEGEND_BG = "#15110D";

// Backend grid coordinates are tenths of a foot, basket-centred, y toward
// half-court. This is the ONLY place that conversion happens.
export function courtToCanvas(locX: number, locY: number) {
  return {
    x: HOOP_X + (locX / 10) * COURT_SCALE,
    y: HOOP_Y - (locY / 10) * COURT_SCALE,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential ramp — semantic heat, strictly increasing in lightness.
//
// An earlier ramp ran blue → green → red, which is a rainbow: its OKLab
// lightness went 0.66 → 0.87 → 0.65, i.e. up and back down. The worst shot on
// the floor and the best shot on the floor rendered at nearly the same
// lightness, so in greyscale, in print, or to a red-green colourblind viewer
// they were indistinguishable — the encoding carried no information at all.
//
// These steps rise monotonically (0.27 → 0.95, minimum step 0.09), so magnitude
// survives loss of hue. Heat is the one multi-hue sequential exception the
// house style allows, and it is the domain's own vocabulary for this chart.
// ─────────────────────────────────────────────────────────────────────────────
// Heat ramp for a light hardwood floor — MULTIPLY tints, not opaque colours.
//
// Painting the field over the boards with source-over hid the wood: at any
// alpha strong enough to read as heat, the plank seams and grain underneath
// were gone. These entries are multiplicative tints instead, composited with
// globalCompositeOperation = "multiply", so every pixel of heat is the wood's
// own value scaled down. A seam that is 12% darker than the surface stays 12%
// darker inside the hottest blob on the floor — the texture is preserved by
// construction, at every intensity, rather than by choosing a timid alpha.
//
// The cold end is near white, which is the identity for multiply and so leaves
// the boards untouched. From there the tint deepens and saturates; over
// #DCB489 the ramp resolves to roughly:
//
//   d5af86 -> d69f60 -> d6893c -> d16b28 -> c04a20 -> a0301d -> 74151a
//
// Lightness falls monotonically across that range, so magnitude still survives
// greyscale and red-green colour blindness.
export const HEAT_RAMP: Array<[number, number, number]> = [
  [0xf7, 0xf4, 0xef],
  [0xf9, 0xe2, 0xb2],
  [0xf9, 0xc4, 0x72],
  [0xf2, 0x9a, 0x4c],
  [0xdd, 0x6b, 0x3c],
  [0xb9, 0x45, 0x36],
  [0x86, 0x1e, 0x30],
];

// The legend sits on a dark panel, where a multiply tint would be meaningless.
// Compositing it against the floor first means the legend shows the colours
// that actually appear on the court.
export function heatSwatch(r: number, g: number, b: number) {
  const surface = [0xdc, 0xb4, 0x89];
  return `rgb(${Math.round((r * surface[0]) / 255)}, ${Math.round((g * surface[1]) / 255)}, ${Math.round((b * surface[2]) / 255)})`;
}

export function heatColor(t: number) {
  const clamped = Math.max(0, Math.min(1, t));
  const pos = clamped * (HEAT_RAMP.length - 1);
  const i = Math.min(HEAT_RAMP.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = HEAT_RAMP[i];
  const b = HEAT_RAMP[i + 1];
  return {
    r: Math.round(a[0] + (b[0] - a[0]) * f),
    g: Math.round(a[1] + (b[1] - a[1]) * f),
    b: Math.round(a[2] + (b[2] - a[2]) * f),
  };
}
