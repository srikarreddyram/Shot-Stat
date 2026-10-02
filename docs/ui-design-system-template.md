# UI Design System — reusable template

Extracted from Shot Vision's frontend (`shot-vision-engine-main/`, React 19 +
TanStack Start + Vite, inline-style components, no CSS framework beyond
Tailwind utilities for a handful of pages). Everything below is written to be
copy-portable into a different project: the specific hex values and copy are
this app's choices, but the *patterns* — how theming, states, layout rhythm,
and animation are structured — are the reusable part.

Nothing here needs Tailwind, Radix, or any particular framework. All of it is
plain inline `style` objects and CSS keyframes in a `<style>` tag, which is
exactly why it drops into a new project cleanly: there's no build-step
dependency to bring with it.

---

## 1. Design tokens

One object per concern, imported everywhere instead of re-typed. This is the
single highest-leverage thing to copy: a page that reaches for `C.bg` and
`F.mono` instead of `"#0a0a0f"` and `"monospace"` cannot drift from the system
by typo.

```ts
// tokens.ts
export const C = {
  bg: "#0a0a0f",           // page background
  surface: "#0e0e16",      // card background, one step up from bg
  raised: "#16161f",       // a card floating above another card
  rule: "rgba(255,255,255,0.04)",  // hairline dividers — barely-there on purpose
  text: "#F0F0F0",         // primary text (never pure #fff — too harsh on this bg)
  dim: "rgba(240,240,240,0.62)",   // secondary text
  faint: "rgba(240,240,240,0.38)", // tertiary / placeholder text
  muted: "#64748b",        // captions, disabled states
  gold: "var(--se-accent)",        // the ACTIVE accent — see §2, it's dynamic
  red: "#DC2626",
};

export const F = {
  display: "'Bebas Neue', sans-serif",   // headlines, big numbers — condensed, loud
  mono: "'JetBrains Mono', monospace",   // ALL data, labels, captions, eyebrows
  body: "'Inter', sans-serif",           // paragraphs, longer descriptive text
};

// Numerals in a mono face still aren't guaranteed to line up in columns
// unless you ask for it explicitly.
export const NUM: React.CSSProperties = {
  fontFamily: F.mono,
  fontVariantNumeric: "tabular-nums",
};
```

**Three-typeface rule.** One display face (character, used sparingly, for
things that should feel like a scoreboard/hero moment), one body face (for
anything a user actually reads at length), one mono face (for *every* number,
label, eyebrow, and caption — a mono face used this consistently is what makes
a data-dense page look designed rather than defaulted). Don't reach for a
fourth.

**Why `gold` is a CSS variable, not a hex constant** — see §2. Any token that
should be able to re-theme per page (a brand color, a per-entity accent) needs
to be a `var()`, not a literal, or re-theming means threading a prop through
every component that uses it.

---

## 2. Dynamic theming via CSS custom properties

The whole app can re-skin to a specific color (a sports team, a client brand,
a status) by setting four CSS variables on a wrapping element — no prop
drilling, no context provider, no re-render of the subtree's own styles.

```ts
export const DEFAULT_ACCENT = "#C9A84C"; // the app's own baseline brand color

export function accentVars(accent: string): React.CSSProperties {
  return {
    // 8-digit hex: the last byte IS the alpha channel, so one base color
    // yields a whole tint set with no color-math library.
    "--se-accent": accent,
    "--se-edge": `${accent}1A`,        // ~10% — hairline borders
    "--se-edge-strong": `${accent}3D`, // ~24% — emphasized borders
    "--se-hover": `${accent}14`,       // ~8%  — hover fills
    "--se-on-accent": readableOn(accent), // legible text COLOR on top of the accent
  } as React.CSSProperties;
}

// Every component that wants "the current accent" reads `var(--se-accent)`
// (or C.gold, which IS that variable) rather than a passed-in prop. Wrap any
// subtree in `<div style={accentVars(brandColor)}>` and every Card, Segmented
// control, and Bar inside it re-themes for free.
export function readableOn(hex: string): string {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return L > 0.45 ? "#000" : "#fff"; // WCAG relative luminance — never hardcode black-on-accent
}
```

**Real usage in this app**: a player or team profile page wraps its content in
`accentVars(teamColor)` so every badge, border, and progress bar on that page
picks up the team's real color, while a generic leaderboard page wraps in
`accentVars(DEFAULT_ACCENT)` and looks like the "home" palette. Same
components, zero conditional styling inside them.

**The `readableOn` function is not optional** if you ever theme against a
palette you don't control (team colors, user-picked brand colors, status
colors) — several real entries in this app's own color map are dark enough
(a deep green, a navy) that fixed white-on-accent text would fail contrast,
and fixed black-on-accent would fail on the light entries. Compute it.

---

## 3. Core primitives

Small, composable, and every one of them reads its colors from the token
object above — never a literal.

### Card
```tsx
function Card({ children, accent, style }: { children: ReactNode; accent?: string; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${C.edge}`,
      borderLeft: accent ? `3px solid ${accent}` : `1px solid ${C.edge}`,
      borderRadius: 6,
      ...style,
    }}>
      {children}
    </div>
  );
}
```
An accent is a left-border stripe, not a full-card fill — reserves saturated
color for something that actually needs to draw the eye (an error card, a
"this is the featured one" card), keeps the base case quiet.

### SectionLabel (the eyebrow)
```tsx
function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontFamily: F.mono, fontSize: 10, color: C.gold, letterSpacing: "0.4em", textTransform: "uppercase" }}>
      {children}
    </div>
  );
}
```
Wide letter-spacing + mono + small size + accent color is the one recurring
"this labels a block below it" signature used everywhere: section headers,
stat-block eyebrows, hero page kickers. Reuse this exact recipe rather than
inventing a new label style per screen.

### Segmented control (the two/three-way toggle)
```tsx
function Segmented({ options, value, onChange }: {
  options: { value: string; label: string }[]; value: string; onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "inline-flex", background: C.surface, border: `1px solid ${C.edge}`, borderRadius: 4, padding: 3, gap: 3 }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button key={o.value} onClick={() => onChange(o.value)} aria-pressed={active} style={{
            background: active ? C.gold : "transparent",
            color: active ? "var(--se-on-accent, #000)" : C.dim,
            border: "none", borderRadius: 2, padding: "7px 18px", cursor: "pointer",
            fontFamily: active ? F.display : F.mono,
            fontSize: active ? 15 : 10,
            letterSpacing: active ? "0.06em" : "0.22em",
            transition: "all 200ms",
          }}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
```
Notice the active option switches typeface (mono → display) as well as color
— the whole point is that "which one is active" should be readable from
20 feet away, not just from a slightly-different gray. **This is the
"alternate between two views" pattern** — used in this app for EP-vs-Make%,
season-vs-career, visual-vs-list density, and (see §9) best-vs-worst
rankings. Any time a feature says "let the user flip between two structurally
identical views," reach for this component before inventing a new toggle.

### Percentile bar + tier color
```tsx
const TIERS = [
  { min: 0.9, color: "#16A34A", label: "Elite" },
  { min: 0.7, color: "#84CC16", label: "Strong" },
  { min: 0.4, color: "#C9A84C", label: "Average" }, // brand accent sits at "average" ON PURPOSE
  { min: 0.2, color: "#EA580C", label: "Below average" },
  { min: 0,   color: "#DC2626", label: "Poor" },
];
function tierColor(pct: number | null | undefined): string {
  if (pct == null) return C.muted;
  return (TIERS.find((t) => pct >= t.min) ?? TIERS[TIERS.length - 1]).color;
}
function Bar({ pct, color, height = 3 }: { pct: number; color?: string; height?: number }) {
  return (
    <div style={{ flex: 1, height, background: "rgba(255,255,255,0.05)", borderRadius: height / 2, overflow: "hidden" }}>
      <div style={{ width: `${Math.max(2, pct * 100)}%`, height: "100%", background: color ?? tierColor(pct), borderRadius: height / 2, transition: "width 300ms" }} />
    </div>
  );
}
```
Putting the brand accent color at the *middle* of the quality ramp (not at
the top) is a deliberate choice: it means "this is merely average" reads as
neutral information, not as the app's own celebratory color being spent on a
mediocre result. Pick where your own accent sits in a quality ramp
on purpose.

### Loading skeleton (shape-matched, not a generic shimmer bar)
```tsx
function TableSkeleton({ rows = 10 }: { rows?: number }) {
  return (
    <Card style={{ overflow: "hidden" }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{ display: "flex", gap: 16, padding: "11px 14px", borderBottom: `1px solid ${C.rule}`, alignItems: "center", opacity: 1 - i * (0.6 / rows) }}>
          <div style={{ width: 22, height: 9, background: "rgba(255,255,255,0.05)", borderRadius: 2 }} />
          <div style={{ width: 150, height: 9, background: "rgba(255,255,255,0.07)", borderRadius: 2 }} />
          <div style={{ flex: 1 }} />
          {[54, 54, 54, 54].map((w, j) => <div key={j} style={{ width: w, height: 9, background: "rgba(255,255,255,0.05)", borderRadius: 2 }} />)}
        </div>
      ))}
    </Card>
  );
}
```
The skeleton's row shape mirrors the REAL row shape it's standing in for
(rank column, name column, four stat columns) — and rows fade toward the
bottom (`1 - i * (0.6/rows)`), which reads as "more content below" rather
than a hard cutoff. Build a skeleton per real layout, not one generic
shimmer block reused everywhere.

### Empty / error states
```tsx
function EmptyState({ children }: { children: ReactNode }) {
  return <div style={{ padding: "44px 24px", textAlign: "center", color: C.muted, fontFamily: F.mono, fontSize: 11, letterSpacing: "0.18em" }}>{children}</div>;
}
function ErrorState({ message }: { message: string }) {
  return (
    <Card accent={C.red} style={{ padding: "16px 20px", marginBottom: 16 }}>
      <div style={{ fontFamily: F.body, fontWeight: 600, fontSize: 14, color: C.text }}>The engine didn't answer</div>
      <div style={{ fontFamily: F.mono, fontSize: 11.5, color: C.muted, marginTop: 5, lineHeight: 1.6 }}>{message} — check that the backend is still running.</div>
    </Card>
  );
}
```
An error state names what a person can actually go check, not just "an error
occurred." Adapt the second line's suggestion to whatever your own likely
failure mode is (dev server down, auth expired, network offline).

---

## 4. Brand chrome (make every page feel like the same product)

A small set of "always-on" decorative elements, added once, that make every
page in an app read as one coherent product instead of a pile of
independently-styled screens.

```tsx
const BRAND_PRIMARY = "#C8102E";
const BRAND_SECONDARY = "#1D428A";

// A persistent colored frame down both edges of the viewport — fixed
// position (survives scrolling), pointer-events none (never steals a click).
// The one piece of brand identity every page shares, themed pages included.
function EdgeBezel() {
  return (
    <>
      <div aria-hidden style={{ position: "fixed", left: 0, top: 0, bottom: 0, width: 4, background: BRAND_PRIMARY, opacity: 0.55, zIndex: 40, pointerEvents: "none" }} />
      <div aria-hidden style={{ position: "fixed", right: 0, top: 0, bottom: 0, width: 4, background: BRAND_SECONDARY, opacity: 0.55, zIndex: 40, pointerEvents: "none" }} />
    </>
  );
}

// Default ambient page wash for anywhere with no specific entity to theme
// against — corner-anchored radial glows, never a full-bleed gradient (which
// would fight the content for attention).
function DefaultWash() {
  return (
    <div aria-hidden style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
      <div style={{ position: "absolute", inset: 0, background: `radial-gradient(65% 55% at 8% 0%, ${BRAND_PRIMARY}2E 0%, transparent 62%)` }} />
      <div style={{ position: "absolute", inset: 0, background: `radial-gradient(65% 55% at 92% 0%, ${BRAND_SECONDARY}38 0%, transparent 62%)` }} />
    </div>
  );
}

// A three-band brand stripe under a page header — cheap, instantly
// recognizable, reused rather than reinvented per page.
function HeaderStripe({ bottom = 0 }: { bottom?: number | string }) {
  return (
    <div aria-hidden style={{ position: "absolute", left: 0, right: 0, bottom, height: 5, display: "flex" }}>
      <div style={{ flex: 1, background: BRAND_PRIMARY }} />
      <div style={{ flex: 1, background: "#F0F0F0" }} />
      <div style={{ flex: 1, background: BRAND_SECONDARY }} />
    </div>
  );
}
```

**The pattern worth copying**: pick ONE default wash for generic pages, and
let any page with a specific entity in play (a team, a user, a project)
override it with that entity's own color via §2's `accentVars` — so the app
is never "plain" on a generic page, and never fighting a competing color
scheme on an entity-specific one.

---

## 5. Layout rhythm

- **CSS columns for uneven card heights**, not CSS grid: `columnWidth: 350,
  columnGap: 16` on a wrapper, `breakInside: "avoid"` on each card. A grid
  leaves a tall gap under a short card next to a tall one; columns pack them
  Pinterest-style with zero JS measurement.
- **Segmented controls travel in pairs**, right-aligned against a page's
  content width, never centered — `justifyContent: "space-between"` with a
  page-context label on the left and the toggle(s) on the right.
- **A hero card, then content** — nearly every detail page (a player, a
  team, a matchup result) opens with one wide `Card` containing an avatar/
  identity block on the left and 2-4 headline numbers on the right, THEN the
  detail sections begin below it. Establish identity once, at the top, never
  re-litigate it in every section beneath.
- **Side gutter discipline**: outer page padding is set once (16-20px) on a
  top-level wrapper; nothing inside ever adds its own left/right margin to
  fight it.

---

## 6. Animation catalog

Every keyframe this app defines, what it's for, and when to reach for it.
None of these need a library — they're all plain CSS `@keyframes` dropped in
a `<style>` tag scoped to the component that uses them (so an animation
declared inside a component that isn't mounted can't silently exist nowhere,
a real bug this app hit once).

```css
/* Word-by-word drop-in for a hero headline — staggered per word via
   animation-delay, not a single fade on the whole line. Reads as the
   headline being TYPESET in front of you rather than just appearing. */
@keyframes wordDrop {
  from { opacity: 0; transform: translateY(-36px); filter: blur(5px); }
  to   { opacity: 1; transform: translateY(0);      filter: blur(0);   }
}

/* The generic "this block just entered" animation — content cards, result
   panels, anything that appears after a load or a state change. */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0);     }
}

/* A stat/number "landing" with a slight overshoot — pairs with a
   cubic-bezier(0.34,1.56,0.64,1) easing (a spring-like bounce) for a number
   or badge that should feel like it POPPED into place, not just faded. */
@keyframes statPop {
  from { opacity: 0; transform: scale(0.85) translateY(12px); }
  to   { opacity: 1; transform: scale(1)    translateY(0);     }
}

/* Ambient pulsing glow — a loading state's "still working" heartbeat, or a
   call-to-action button that should breathe without being obnoxious. */
@keyframes goldPulse {
  0%, 100% { box-shadow: 0 0 20px rgba(201,168,76,0.2); }
  50%      { box-shadow: 0 0 48px rgba(201,168,76,0.5); }
}

/* A tiny vertical nudge + opacity pulse for a "scroll for more" affordance. */
@keyframes scrollNudge {
  0%, 100% { opacity: 0.3; transform: scaleY(0.8); }
  50%      { opacity: 1;   transform: scaleY(1.1); }
}

/* A progress/percentile bar growing in from zero width on first render. */
@keyframes barFill { from { width: 0; } }

/* A slow ambient opacity breathe for a background decoration (a watermark,
   a soft glow) — subtle enough to sit behind content without competing. */
@keyframes ambientPulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.9; } }

/* A location marker "radar ping" — expands and fades out, repeats. Good for
   "here's the thing I want your eye on" on a map, chart, or canvas overlay. */
@keyframes pulseRing {
  0%, 100%  { opacity: 1; transform: translate(-50%,-50%) scale(1);   box-shadow: 0 0 8px currentColor; }
  50%       { opacity: 0; transform: translate(-50%,-50%) scale(1.6); box-shadow: 0 0 24px currentColor; }
}
```

**Rules of thumb this app follows:**
- Loading dots / pulses use `ease` timing, never `linear` — a linear pulse
  reads as mechanical rather than alive.
- A "pop into place" animation (stat reveal, badge appearing) uses an
  overshoot easing (`cubic-bezier(0.34,1.56,0.64,1)`) so it feels like it has
  weight; a plain fade-in never does.
- Hover states are handled with `onMouseEnter`/`onMouseLeave` setting
  `opacity` directly (`0.8` on hover) rather than a CSS `:hover` pseudo-class
  — because these components are built as inline-style objects with no
  separate stylesheet to attach a pseudo-class to. If your project DOES have
  a stylesheet, prefer real `:hover` — this is a workaround, not an
  aspiration.
- Color and background transitions default to `200-300ms`; layout-affecting
  transitions (width, height) get `700ms` — slow enough to see the change
  happen, not so slow it feels laggy.

---

## 7. The scrollytelling landing-page pattern

The splash/marketing page in this app is the single most distinctive piece
of motion design in the codebase, and it's a fully generic pattern worth
lifting wholesale into a different project's landing page.

**The mechanism**: a very tall scroll container (e.g. `height: 600vh`) wraps
a `position: sticky; top: 0; height: 100vh` inner viewport. As the user
scrolls through the tall container, the STICKY viewport stays pinned to the
screen while the scroll position is read and mapped to a "section index"
(0 through N-1). Each section's content is absolutely positioned over the
same sticky viewport and cross-fades in/out based on whether it's the active
section — so the page LOOKS like distinct full-screen slides, but is
actually one continuously-scrollable element with no scroll-jacking and no
JS-driven smooth-scroll interception (real browser scroll the whole way).

```tsx
const SECTIONS = 6;

function ScrollyPage() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [section, setSection] = useState(0);
  const sectionRef = useRef(0);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onScroll = () => {
      const rect = el.getBoundingClientRect();
      const scrolled = Math.min(1, Math.max(0, -rect.top / (rect.height - window.innerHeight)));
      const idx = Math.min(SECTIONS - 1, Math.floor(scrolled * SECTIONS));
      if (idx !== sectionRef.current) { sectionRef.current = idx; setSection(idx); }
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div ref={wrapRef} style={{ height: `${SECTIONS * 100}vh`, position: "relative" }}>
      <div style={{ position: "sticky", top: 0, height: "100vh", overflow: "hidden" }}>
        {/* background media / gradient that can also key off `section` */}
        {Array.from({ length: SECTIONS }).map((_, i) => (
          <SectionOverlay key={i} active={section === i}>
            {/* this section's headline/content */}
          </SectionOverlay>
        ))}
        {/* optional: nav dots, see below */}
      </div>
    </div>
  );
}

function SectionOverlay({ children, active }: { children: React.ReactNode; active: boolean }) {
  return (
    <div style={{
      position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
      opacity: active ? 1 : 0,
      pointerEvents: active ? "auto" : "none",
      transition: "opacity 0.55s ease",
    }}>
      <div style={{ width: "100%" }}>{children}</div>
    </div>
  );
}
```

**Supporting details that make it feel finished, not just functional:**

- **Per-section legibility scrim** — if there's a photo/video background
  behind every section, don't rely on a single global darkening overlay;
  give each section its own gradient (biased toward whichever side the copy
  sits on) and cross-fade the scrim's `background` on a `700ms` transition
  when the section changes.
- **Word-by-word headline reveal** — split a headline into words, animate
  each with `wordDrop` (see §6) at a staggered `animation-delay` (`i * 0.09s`
  per word). A whole-line fade never reads as intentional the way a
  staggered word cascade does.
- **Click-to-jump nav dots**, synced bidirectionally with scroll position:
  ```tsx
  {Array.from({ length: SECTIONS }).map((_, i) => (
    <div
      key={i}
      onClick={() => {
        const el = wrapRef.current!;
        const total = el.offsetHeight - window.innerHeight;
        window.scrollTo({ top: el.offsetTop + (total * i) / SECTIONS + 10, behavior: "smooth" });
      }}
      style={{
        width: section === i ? 3 : 2,
        height: section === i ? 28 : 8,     // the active dot GROWS into a dash
        background: section === i ? BRAND_ACCENT : `${BRAND_ACCENT}33`,
        borderRadius: 2, cursor: "pointer",
        transition: "all 400ms cubic-bezier(0.34,1.56,0.64,1)",
      }}
    />
  ))}
  ```
  The active dot stretching into a short dash (rather than just changing
  color) is what makes the nav readable as a progress indicator, not just a
  row of bullets.
- **Static chrome stays static.** The brand mark and edge bezel (§4) are
  rendered ONCE, outside the per-section animation logic, explicitly NOT
  wired to `section` or any keyframe — a persistent frame around a moving
  scene, not part of the scene.

---

## 8. The "alternate view" toggle pattern (beyond a simple two-way switch)

Worth calling out as its own pattern because it recurs constantly: **any
time a feature computes a ranked or scored list, its inverse (the WORST N,
not just the BEST N) is nearly free to add** if the full list is already
being computed and sorted — and presenting it as a toggle that reuses the
exact same card/hero markup, just re-colored and re-sliced, is far better
than a whole separate view.

```tsx
const PALETTES = {
  best:  { colors: ["#16A34A", "#C9A84C", "#B8860B"], heroValue: "#16A34A", label: "TOP RESULT" },
  worst: { colors: ["#DC2626", "#B91C1C", "#7F1D1D"], heroValue: "#DC2626", label: "WEAKEST RESULT" },
} as const;

function RankedPanel({ allResultsSortedBestFirst }: { allResultsSortedBestFirst: Result[] }) {
  const [view, setView] = useState<"best" | "worst">("best");
  const palette = PALETTES[view];
  const ranked = view === "best"
    ? allResultsSortedBestFirst.slice(0, 3)
    : allResultsSortedBestFirst.slice(-3).reverse(); // worst-first when reversed
  const hero = ranked[0];

  return (
    <>
      <Segmented value={view} onChange={(v) => setView(v as "best" | "worst")}
        options={[{ value: "best", label: "BEST" }, { value: "worst", label: "WORST" }]} />
      {/* hero + ranked cards below read `palette` and `ranked`/`hero` —
          IDENTICAL markup to the single-view version, just parameterized */}
    </>
  );
}
```

The key discipline: **build one render path parameterized by a small
palette/slice object, not two copy-pasted render paths** — otherwise a future
change to the "best" card's markup silently doesn't apply to "worst," which
is exactly the kind of drift a toggle like this should never allow.

---

## 9. Data visualization conventions

- **A canvas-drawn heatmap, not a chart library**, for anything spatial (a
  court, a map, a floor plan): draw a radial gradient per data point directly
  to a `<canvas>`, and overlay a plain absolutely-positioned `<div>` (with the
  `pulseRing` keyframe from §6) for the single highlighted point — mixing a
  canvas heat layer with a DOM overlay for the ONE interactive marker is far
  simpler than making the whole thing DOM-based or the whole thing
  canvas-interactive.
- **A radar/spider chart is hand-rolled SVG**, not a charting library, when
  the shape is simple (a handful of axes, one or two series): compute each
  axis's angle as `-Math.PI/2 + i * 2*Math.PI/n` (start at 12 o'clock, go
  clockwise), draw concentric `<polygon>` gridlines at fixed radii (25/50/
  75/100%), and plot each series as one filled, semi-transparent `<polygon>`
  plus a `<circle>` per vertex. Under ~8 axes and ~3 series, hand-rolled SVG
  is less code and more controllable than pulling in a charting dependency.
- **A progress/percentile bar's color IS the data** (tier-colored per §3),
  not a fixed brand color — the reader should see "how good" before they
  read the number next to it.
- **Every numeric column is tabular-nums** (`NUM` token, §1) — non-negotiable
  for anything presented as a list of comparable numbers.

---

## 10. Adapting this template to a new project

1. Pick your three typefaces (display / body / mono) and your `DEFAULT_ACCENT`.
   Run the accent through `readableOn()` before you trust any "accent
   background + fixed text color" combination.
2. Copy §1's token objects verbatim, replacing values only.
3. Copy §3's primitives verbatim — they have zero app-specific logic.
4. Decide your brand chrome (§4): a two-color edge bezel is a cheap, highly
   recognizable identity marker if your brand has two colors; skip it
   entirely for a single-color brand and just use the `DefaultWash`.
5. If you have a marketing/landing page, seriously consider the scrollytelling
   pattern in §7 over a conventional stack-of-sections page — it costs one
   scroll-listener and pays for itself in how distinctive it feels.
6. Any feature that ranks or scores things: ask whether the inverse view
   (§8) is worth exposing before you ship only the "top N."
7. Keep the animation catalog (§6) as a single shared `<style>` block (or a
   CSS module) rather than letting individual components redeclare
   `@keyframes` under slightly different names — this app's own `goldPulse`
   exists in two places today for exactly that avoidable reason.
