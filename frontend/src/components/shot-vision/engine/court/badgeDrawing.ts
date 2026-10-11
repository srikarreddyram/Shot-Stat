import { useEffect, useState } from "react";
import { COURT_LINE_STRONG } from "./geometry";

// Headshot for the on-floor matchup badges, as a decoded <img> the canvas can
// draw. Deliberately NOT crossOrigin: the NBA CDN is not guaranteed to send
// CORS headers, and a rejected CORS load is a permanently blank badge, whereas
// tainting this canvas costs nothing — the field is composed in a separate
// offscreen canvas and no code ever reads pixels back off the visible one.
export function useHeadshot(url?: string) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    setImg(null);
    if (!url) return;
    let alive = true;
    const im = new Image();
    im.onload = () => { if (alive) setImg(im); };
    im.onerror = () => { if (alive) setImg(null); };
    im.src = url;
    return () => { alive = false; };
  }, [url]);
  return img;
}

// Canvas text silently falls back to a system font if the webfont has not
// finished loading when the effect runs, which is a coin flip on first paint.
// Flipping this state after document.fonts settles gives the draw a second
// pass with the real faces.
export function useFontsReady() {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let alive = true;
    document.fonts?.ready.then(() => { if (alive) setReady(true); });
    return () => { alive = false; };
  }, []);
  return ready;
}

// "Victor Wembanyama" -> "V. WEMBANYAMA". Full names do not fit beside a
// 42px badge on a 400px floor, and the surname is the identifying half.
export function badgeName(name: string) {
  const parts = name.trim().split(/\s+/);
  if (parts.length < 2) return name.toUpperCase();
  return `${parts[0][0]}. ${parts.slice(1).join(" ")}`.toUpperCase();
}

export function initialsOf(name: string) {
  return name.split(" ").filter(Boolean).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

// One corner decal: circular headshot, accent ring, role label and name
// painted onto the boards. `side` decides which way the text runs so both
// badges hug their own corner.
export function drawMatchupBadge(
  ctx: CanvasRenderingContext2D,
  opts: { cx: number; cy: number; r: number; side: "left" | "right"; accent: string; ink: string; role: string; name: string; img: HTMLImageElement | null },
) {
  const { cx, cy, r, side, accent, ink, role, name, img } = opts;

  ctx.save();

  // Seat the disc on the floor with a soft shadow, so it reads as an object
  // on the boards rather than a hole cut through them.
  ctx.shadowColor = "rgba(40, 20, 6, 0.45)";
  ctx.shadowBlur = 7;
  ctx.shadowOffsetY = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = "#12121A";
  ctx.fill();
  ctx.restore();

  if (img) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.clip();
    // NBA headshots are 1040x760 with the player centred and shoulders-up.
    // Crop a square around the head rather than squashing the full frame
    // into the circle.
    const s = Math.min(img.naturalWidth, img.naturalHeight) * 0.72;
    const sx = (img.naturalWidth - s) / 2;
    const sy = Math.max(0, img.naturalHeight * 0.06);
    ctx.drawImage(img, sx, sy, s, s, cx - r, cy - r, r * 2, r * 2);
    ctx.restore();
  } else {
    ctx.fillStyle = accent;
    ctx.font = "16px 'Bebas Neue', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(initialsOf(name) || "?", cx, cy + 1);
  }

  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.stroke();

  const tx = side === "left" ? cx + r + 9 : cx - r - 9;
  ctx.textAlign = side === "left" ? "left" : "right";
  ctx.textBaseline = "alphabetic";

  // The role label is a darkened version of the accent, not the accent itself:
  // #C9A84C gold on #DCB489 peach is barely a colour difference, and the ring
  // already carries the attacker/defender coding at full strength. Darkening
  // it keeps the association and makes it readable as pigment on wood.
  ctx.font = "8px 'JetBrains Mono', monospace";
  ctx.fillStyle = ink;
  ctx.fillText(role, tx, cy - 7);

  // The name is painted in the same dark pigment as the court markings, so it
  // belongs to the floor. It stays legible over the heat because the ramp is
  // monotonic in lightness and never gets light enough to swallow it.
  ctx.font = "18px 'Bebas Neue', sans-serif";
  ctx.fillStyle = COURT_LINE_STRONG;
  ctx.fillText(badgeName(name), tx, cy + 13);

  ctx.restore();
}
