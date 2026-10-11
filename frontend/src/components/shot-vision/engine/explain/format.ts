// Attainability spans two orders of magnitude across the floor — a corner
// three can sit near 1% while the rim is over 30% — so a single fixed
// precision either rounds the small end into noise or clutters the large end.
export function formatAttainability(v: number) {
  return `${(v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0)}%`;
}

export function ordinal(p: number) {
  const n = Math.round(p);
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
