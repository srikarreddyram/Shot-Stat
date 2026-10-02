import { useEffect, useRef, useState } from "react";
import VideoBackdrop from "./video-backdrop";
import {
  BrandLockup, LeagueBrandMark, NBAEdgeBezel, ShotVisionMark, SiteNav, teamLogoUrl,
} from "@/components/stat-engine/ui";

interface Props {
  onLaunch: () => void;
}

const SECTIONS = 6;

const KEYFRAMES = `
@keyframes wordDrop { from { opacity:0; transform:translateY(-36px); filter:blur(5px);} to { opacity:1; transform:translateY(0); filter:blur(0);} }
@keyframes fadeUp { from { opacity:0; transform:translateY(14px);} to { opacity:1; transform:translateY(0);} }
@keyframes statPop { from { opacity:0; transform:scale(0.85) translateY(12px);} to { opacity:1; transform:scale(1) translateY(0);} }
@keyframes goldPulse { 0%,100% { box-shadow: 0 0 20px rgba(201,168,76,0.2);} 50% { box-shadow: 0 0 48px rgba(201,168,76,0.5);} }
@keyframes scrollNudge { 0%,100% { opacity:0.3; transform:scaleY(0.8);} 50% { opacity:1; transform:scaleY(1.1);} }
@keyframes barFill { from { width:0;} }
@keyframes ambientPulse { 0%,100% { opacity:0.4;} 50% { opacity:0.9;} }
`;

/** Per-section darkening, biased toward whichever edge the copy sits on. */
const SCRIMS = [
  "linear-gradient(to bottom, rgba(10,10,15,0.72) 0%, rgba(10,10,15,0.42) 55%, rgba(10,10,15,0.78) 100%)",
  "linear-gradient(100deg, rgba(10,10,15,0.94) 0%, rgba(10,10,15,0.78) 42%, rgba(10,10,15,0.34) 100%)",
  "linear-gradient(260deg, rgba(10,10,15,0.94) 0%, rgba(10,10,15,0.78) 42%, rgba(10,10,15,0.34) 100%)",
  "linear-gradient(100deg, rgba(10,10,15,0.95) 0%, rgba(10,10,15,0.8) 45%, rgba(10,10,15,0.36) 100%)",
  "linear-gradient(260deg, rgba(10,10,15,0.94) 0%, rgba(10,10,15,0.78) 42%, rgba(10,10,15,0.34) 100%)",
  "linear-gradient(to bottom, rgba(10,10,15,0.62) 0%, rgba(10,10,15,0.86) 35%, rgba(10,10,15,0.86) 80%, rgba(10,10,15,0.66) 100%)",
];
/**
 * Luka Dončić's actual 2025-26 zone splits from player_zone_stats — the same
 * table the engine reads. Expected points is fg_pct x shot value, so the
 * spread between his best and worst zone makes the section's argument with a
 * real player instead of a hypothetical one.
 *
 * Names and public performance statistics are factual sports data, which is
 * what this product analyses; no footage or likeness is used.
 */
const LUKA_SEASON = "2025-26";
const LUKA_ZONES = [
  { zone: "RESTRICTED AREA", fgm: 133, fga: 163, value: 2 },
  { zone: "ABOVE THE BREAK 3", fgm: 246, fga: 649, value: 3 },
  { zone: "IN THE PAINT", fgm: 202, fga: 365, value: 2 },
  { zone: "MID-RANGE", fgm: 104, fga: 235, value: 2 },
  { zone: "LEFT CORNER 3", fgm: 4, fga: 30, value: 3 },
].map((z) => ({ ...z, ep: (z.fgm / z.fga) * z.value }));

/**
 * Stephen Curry's 2025-26 profile as the Stat Engine shows it: headline
 * ratings with league rank, and three of the stats he ranks first in.
 * Straight off /stats (player_stat_table), not rounded up for effect.
 */
const CURRY = {
  season: "2025-26",
  ratings: [
    { label: "OFFENSE", value: 99, rank: "#1 / 542", color: "#16A34A" },
    { label: "DEFENSE", value: 66, rank: "#292 / 541", color: "#C9A84C" },
    { label: "NBA 2K", value: 93, rank: "#9 / 481", color: "#16A34A" },
  ],
  tops: [
    { stat: "THREES MADE / GAME", value: "4.40", rank: "#1 OF 582" },
    { stat: "THREES ATTEMPTED / GAME", value: "11.3", rank: "#1 OF 582" },
    { stat: "SECONDARY ASSISTS / GAME", value: "1.30", rank: "#1 OF 582" },
  ],
} as const;

/**
 * The seven shot archetypes found by PCA + K-Means over 2,234,150 shots
 * (models/shot_archetypes_metadata.json, pca_kmeans_v1): share of all shots
 * and FG% within each. The same clusters the /archetypes page draws.
 */
const ARCHETYPES = {
  shots: 2234150,
  rows: [
    { label: "Open above-the-break spot-up three", share: 0.354, fg: 0.362 },
    { label: "Contested driving layup at the rim", share: 0.264, fg: 0.558 },
    { label: "Open above-the-break pull-up three", share: 0.121, fg: 0.408 },
    { label: "Contested short driving floater", share: 0.083, fg: 0.452 },
    { label: "Contested short post-up jumper", share: 0.073, fg: 0.447 },
    { label: "Contested alley-oop dunk", share: 0.062, fg: 0.873 },
    { label: "Open above-the-break step-back three", share: 0.044, fg: 0.394 },
  ],
} as const;

const PRODUCTS = [
  { key: "engine", name: "SHOT ENGINE", href: "/#engine", blurb: "Where to shoot against any defender — make %, expected points and why." },
  { key: "stats", name: "STAT ENGINE", href: "/stats", blurb: "Every stat for every player and team, ranked, compared and charted." },
  { key: "archetypes", name: "ARCHETYPES", href: "/archetypes", blurb: "The kinds of shot the league takes, found by clustering 2M+ shots." },
] as const;

const LUKA_BEST = LUKA_ZONES[0];
const LUKA_FGA = LUKA_ZONES.reduce((n, z) => n + z.fga, 0);

function WordLine({
  text,
  active,
  delay = 0,
  size = 110,
}: {
  text: string;
  active: boolean;
  delay?: number;
  size?: number;
}) {
  const words = text.split(" ");
  return (
    <div
      style={{
        fontFamily: "'Bebas Neue', sans-serif",
        fontSize: size,
        color: "#F0F0F0",
        lineHeight: 0.9,
        letterSpacing: "0.01em",
        display: "flex",
        flexWrap: "wrap",
        gap: "0.28em",
      }}
    >
      {words.map((w, i) => (
        <span
          key={`${w}-${i}`}
          style={{
            display: "inline-block",
            opacity: 0,
            animation: active
              ? `wordDrop 0.7s cubic-bezier(0.2,0.7,0.2,1) ${delay + i * 0.09}s forwards`
              : undefined,
          }}
        >
          {w}
        </span>
      ))}
    </div>
  );
}

export default function SplashPage({ onLaunch }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [section, setSection] = useState(0);
  const sectionRef = useRef(0);

  // Scroll -> section
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onScroll = () => {
      const rect = el.getBoundingClientRect();
      const scrolled = Math.min(1, Math.max(0, -rect.top / (rect.height - window.innerHeight)));
      const idx = Math.min(SECTIONS - 1, Math.floor(scrolled * SECTIONS));
      if (idx !== sectionRef.current) {
        sectionRef.current = idx;
        setSection(idx);
      }
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <>
      <div ref={wrapRef} style={{ height: "600vh", background: "#0a0a0f", position: "relative" }}>
        <style>{KEYFRAMES}</style>
        <div
          style={{
            position: "sticky",
            top: 0,
            height: "100vh",
            overflow: "hidden",
            background: "#0a0a0f",
          }}
        >
          <VideoBackdrop section={section} />
          {/* Legibility scrim. The court runs full-bleed behind every section,
            so the copy needs a gradient darkening the side it sits on rather
            than the old trick of fading the whole canvas out after section 2. */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 5,
              pointerEvents: "none",
              background: SCRIMS[section] ?? SCRIMS[0],
              transition: "background 700ms ease",
            }}
          />
          {/* Vignette — keeps the fogged floor edge from reading as a hard line. */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 6,
              pointerEvents: "none",
              background:
                "radial-gradient(ellipse at 50% 48%, rgba(10,10,15,0) 52%, rgba(10,10,15,0.55) 100%)",
            }}
          />

          {/* Brand + navigation chrome — static, not part of the scroll
              sequence. The same lockup and nav every other page carries, so
              the front page reads as the door to the whole app rather than
              to the shot engine alone. */}
          <div style={{ position: "absolute", left: 32, top: 28, zIndex: 20, display: "flex", alignItems: "center", gap: 14 }}>
            <LeagueBrandMark height={32} />
            <div style={{ width: 1, height: 28, background: "rgba(255,255,255,0.14)" }} />
            <BrandLockup size={34} sub="NBA INTELLIGENCE" />
          </div>
          <div style={{ position: "absolute", right: 32, top: 36, zIndex: 20 }}>
            <SiteNav onEngine={onLaunch} />
          </div>
          {/* Same red/blue edge bezel every other page in the app uses (see
              NBAEdgeBezel in stat-engine/ui.tsx) — fixed positioning reaches
              past this sticky container to the real viewport edge, so it
              still lines up here. Purely an added frame; touches nothing
              about VideoBackdrop, the scrims, or any keyframe above. */}
          <NBAEdgeBezel />

          {/* Nav dots */}
          <div
            style={{
              position: "absolute",
              right: 32,
              top: "50%",
              transform: "translateY(-50%)",
              zIndex: 20,
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            {Array.from({ length: SECTIONS }).map((_, i) => (
              <div
                key={i}
                onClick={() => {
                  const el = wrapRef.current;
                  if (!el) return;
                  const total = el.offsetHeight - window.innerHeight;
                  window.scrollTo({
                    top: el.offsetTop + (total * i) / SECTIONS + 10,
                    behavior: "smooth",
                  });
                }}
                style={{
                  width: section === i ? 3 : 2,
                  height: section === i ? 28 : 8,
                  background: section === i ? "#C9A84C" : "rgba(201,168,76,0.2)",
                  borderRadius: 2,
                  cursor: "pointer",
                  transition: "all 400ms cubic-bezier(0.34,1.56,0.64,1)",
                }}
              />
            ))}
          </div>

          {/* SECTION 0 — what SHOT VISION is now: three products, one engine */}
          <SectionOverlay active={section === 0} align="center">
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 200, color: "rgba(201,168,76,0.018)", letterSpacing: "0.02em" }}>
                VISION
              </div>
            </div>
            <div style={{ position: "relative", maxWidth: 820, margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
              <div style={{ display: "flex", justifyContent: "center", opacity: 0,
                            animation: section === 0 ? "statPop 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.05s forwards" : undefined }}>
                <ShotVisionMark size={84} />
              </div>
              <div style={{ marginTop: 22 }}><Tag>SHOT VISION · NBA INTELLIGENCE</Tag></div>
              <div style={{ marginTop: 22, display: "flex", flexDirection: "column", alignItems: "center" }}>
                <WordLine text="Read the game" active={section === 0} />
                <WordLine text="before it's played." active={section === 0} delay={0.18} />
              </div>
              <p style={{ marginTop: 28, opacity: 0, animation: section === 0 ? "fadeUp 0.7s ease 0.9s forwards" : undefined,
                          fontFamily: "'Inter', sans-serif", fontSize: 17, color: "#9aa7bd", maxWidth: 560,
                          marginLeft: "auto", marginRight: "auto", lineHeight: 1.55 }}>
                Where to shoot against any defender, every stat for every player, and the shapes of the
                shots the whole league takes — one engine, built on ten seasons of real games.
              </p>
              <div style={{ marginTop: 26, display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap", opacity: 0,
                            animation: section === 0 ? "fadeUp 0.7s ease 1.15s forwards" : undefined }}>
                {PRODUCTS.map((p) => (
                  <span key={p.key} style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, letterSpacing: "0.26em",
                                             color: "#C9A84C", border: "1px solid rgba(201,168,76,0.3)", borderRadius: 3,
                                             padding: "6px 12px", background: "rgba(201,168,76,0.06)" }}>
                    {p.name}
                  </span>
                ))}
              </div>
            </div>
            <div style={{ position: "absolute", bottom: 40, left: "50%", transform: "translateX(-50%)", textAlign: "center" }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#C9A84C", letterSpacing: "0.4em" }}>SCROLL</div>
              <div style={{ width: 1, height: 40, background: "#C9A84C", margin: "10px auto 0",
                            animation: "scrollNudge 1.6s ease infinite", transformOrigin: "top" }} />
            </div>
          </SectionOverlay>

          {/* SECTION 1 — SHOT ENGINE */}
          <SectionOverlay active={section === 1} align="left">
            <div style={{ paddingLeft: "8vw", maxWidth: 580 }}>
              <Tag>01 — SHOT ENGINE</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="Not all shots" active={section === 1} size={88} />
                <WordLine text="are equal." active={section === 1} delay={0.18} size={88} />
              </div>
              <p style={{ marginTop: 18, opacity: 0, animation: section === 1 ? "fadeUp 0.7s ease 0.8s forwards" : undefined,
                          fontFamily: "'Inter', sans-serif", fontSize: 15.5, color: "#9aa7bd", maxWidth: 440, lineHeight: 1.55 }}>
                Pick a shooter and the defender on him. The engine scores every spot on the floor — make
                probability, expected points, and why.
              </p>
              <div style={{ marginTop: 18, opacity: 0, animation: section === 1 ? "fadeUp 0.7s ease 0.9s forwards" : undefined,
                            display: "flex", alignItems: "center", gap: 8, fontFamily: "'JetBrains Mono', monospace",
                            fontSize: 11, color: "#C9A84C", letterSpacing: "0.22em" }}>
                <img src={teamLogoUrl("1610612747")!} alt="" aria-hidden="true" loading="lazy"
                     onError={(e) => { e.currentTarget.style.display = "none"; }}
                     style={{ height: 18, width: 18, objectFit: "contain", flexShrink: 0 }} />
                LUKA DONČIĆ · {LUKA_SEASON} · {LUKA_FGA.toLocaleString()} FGA
              </div>
              <div style={{ marginTop: 16, opacity: 0, animation: section === 1 ? "fadeUp 0.7s ease 1s forwards" : undefined }}>
                {LUKA_ZONES.map((z, i) => (
                  <div key={z.zone} style={{ marginTop: i === 0 ? 0 : 10 }}>
                    <ComparisonRow
                      label={z.zone}
                      // Scaled against his best zone, so the drop-off reads at a glance.
                      pct={(z.ep / LUKA_BEST.ep) * 100}
                      color={z.ep >= 1.2 ? "#C9A84C" : z.ep >= 0.85 ? "#B06A2C" : "#DC2626"}
                      epColor={z.ep >= 1.2 ? "#16A34A" : undefined}
                      ep={`${z.ep.toFixed(2)} EP`}
                      sub={`${z.fgm}/${z.fga} · ${((z.fgm / z.fga) * 100).toFixed(1)}%`}
                      active={section === 1}
                      delay={1.05 + i * 0.12}
                    />
                  </div>
                ))}
              </div>
              <SectionCta active={section === 1} delay={1.8} onClick={onLaunch}>OPEN THE SHOT ENGINE →</SectionCta>
            </div>
          </SectionOverlay>

          {/* SECTION 2 — STAT ENGINE */}
          <SectionOverlay active={section === 2} align="right">
            <div style={{ paddingRight: "8vw", maxWidth: 580, marginLeft: "auto", textAlign: "right" }}>
              <Tag>02 — STAT ENGINE</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="Every stat." active={section === 2} size={88} />
                <WordLine text="Every player." active={section === 2} delay={0.18} size={88} />
              </div>
              <p style={{ marginTop: 18, marginLeft: "auto", opacity: 0, animation: section === 2 ? "fadeUp 0.7s ease 0.8s forwards" : undefined,
                          fontFamily: "'Inter', sans-serif", fontSize: 15.5, color: "#9aa7bd", maxWidth: 440, lineHeight: 1.55 }}>
                Box score to tracking data, ranked against the whole league, with a career trend behind every
                number and a side-by-side compare for any three players.
              </p>
              <div style={{ marginTop: 20, display: "flex", gap: 30, justifyContent: "flex-end", opacity: 0,
                            animation: section === 2 ? "fadeUp 0.7s ease 0.95s forwards" : undefined }}>
                {CURRY.ratings.map((r) => (
                  <div key={r.label} style={{ textAlign: "right" }}>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#6b7a92", letterSpacing: "0.25em" }}>{r.label}</div>
                    <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 52, lineHeight: 1, color: r.color }}>{r.value}</div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9.5, color: "#6b7a92" }}>{r.rank}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 18, opacity: 0, animation: section === 2 ? "fadeUp 0.7s ease 1.1s forwards" : undefined }}>
                <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 8, fontFamily: "'JetBrains Mono', monospace",
                              fontSize: 11, color: "#C9A84C", letterSpacing: "0.22em", marginBottom: 10 }}>
                  STEPHEN CURRY · {CURRY.season} · LEAGUE RANK
                  <img src={teamLogoUrl("1610612744")!} alt="" aria-hidden="true" loading="lazy"
                       onError={(e) => { e.currentTarget.style.display = "none"; }}
                       style={{ height: 18, width: 18, objectFit: "contain", flexShrink: 0 }} />
                </div>
                {CURRY.tops.map((t, i) => (
                  <div key={t.stat} style={{ marginTop: i === 0 ? 0 : 10 }}>
                    <ComparisonRow label={t.stat} pct={100} color="#16A34A" ep={t.value} sub={t.rank}
                                   active={section === 2} delay={1.2 + i * 0.12} />
                  </div>
                ))}
              </div>
              <SectionCta active={section === 2} delay={1.8} href="/stats" align="right">BROWSE THE STAT ENGINE →</SectionCta>
            </div>
          </SectionOverlay>

          {/* SECTION 3 — ARCHETYPES */}
          <SectionOverlay active={section === 3} align="left">
            <div style={{ paddingLeft: "8vw", maxWidth: 700, paddingTop: 40 }}>
              <Tag>03 — SHOT ARCHETYPES</Tag>
              <div style={{ marginTop: 16 }}>
                <WordLine text="The league takes" active={section === 3} size={74} />
                <WordLine text="seven shots." active={section === 3} delay={0.18} size={74} />
              </div>
              <p style={{ marginTop: 14, opacity: 0, animation: section === 3 ? "fadeUp 0.7s ease 0.8s forwards" : undefined,
                          fontFamily: "'Inter', sans-serif", fontSize: 15, color: "#9aa7bd", maxWidth: 520, lineHeight: 1.5 }}>
                {ARCHETYPES.shots.toLocaleString()} real shots, clustered by where, how and how contested they were.
                Bar length is how often each kind is taken; the number is how often it goes in.
              </p>
              <div style={{ marginTop: 16, maxWidth: 520 }}>
                {ARCHETYPES.rows.map((a, i) => (
                  <div key={a.label} style={{ marginTop: i === 0 ? 0 : 6 }}>
                    <ComparisonRow
                      label={a.label.toUpperCase()}
                      // Scaled to the most common archetype, so frequency reads at a glance.
                      pct={(a.share / ARCHETYPES.rows[0].share) * 100}
                      color={a.fg >= 0.5 ? "#16A34A" : a.fg >= 0.4 ? "#C9A84C" : "#B06A2C"}
                      ep={`${Math.round(a.fg * 100)}% FG`}
                      sub={`${(a.share * 100).toFixed(1)}% of all shots`}
                      active={section === 3}
                      delay={1.0 + i * 0.1}
                    />
                  </div>
                ))}
              </div>
              <SectionCta active={section === 3} delay={1.9} href="/archetypes" marginTop={18}>EXPLORE THE ARCHETYPES →</SectionCta>
            </div>
          </SectionOverlay>

          {/* SECTION 4 — the data, and how honest the models are about it */}
          <SectionOverlay active={section === 4} align="right">
            <div style={{ paddingRight: "8vw", maxWidth: 900, marginLeft: "auto", textAlign: "right" }}>
              <Tag>04 — UNDER THE HOOD</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="2.2 million shots." active={section === 4} size={88} />
                <WordLine text="Zero peeking." active={section === 4} delay={0.2} size={88} />
              </div>
              <p style={{ marginTop: 18, marginLeft: "auto", opacity: 0, animation: section === 4 ? "fadeUp 0.7s ease 0.9s forwards" : undefined,
                          fontFamily: "'Inter', sans-serif", fontSize: 15.5, color: "#9aa7bd", maxWidth: 480, lineHeight: 1.55 }}>
                Every feature is point-in-time: the model only ever sees games played before the shot it's scoring.
                Then it's tested on a season it never trained on — and the results are published, good and bad.
              </p>
              <div style={{ marginTop: 34, display: "flex", gap: 40, flexWrap: "wrap", justifyContent: "flex-end" }}>
                <StatCallout value="10" label="SEASONS" active={section === 4} delay={1.1} />
                <StatCallout value="2.2M" label="SHOTS MODELLED" active={section === 4} delay={1.25} />
                <StatCallout value="273K" label="PLAYER BOX SCORES" active={section === 4} delay={1.4} />
                <StatCallout value="582" label="PLAYERS RANKED" active={section === 4} delay={1.55} />
              </div>
              <div style={{ marginTop: 26, opacity: 0, animation: section === 4 ? "fadeUp 0.6s ease 1.8s forwards" : undefined,
                            fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#6b7a92", letterSpacing: "0.22em" }}>
                XGBOOST · TREESHAP · EMPIRICAL-BAYES SHRINKAGE · CALIBRATION-CHECKED
              </div>
            </div>
          </SectionOverlay>

          {/* SECTION 5 — CTA: pick where to start */}
          <SectionOverlay active={section === 5} align="center">
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
              <div style={{ width: 640, height: 640, borderRadius: "50%",
                            background: "radial-gradient(circle, rgba(201,168,76,0.08) 0%, rgba(201,168,76,0) 70%)",
                            animation: "ambientPulse 5s ease infinite" }} />
            </div>
            <div style={{ position: "relative", textAlign: "center", padding: "0 24px", maxWidth: 1080, margin: "0 auto" }}>
              <Tag>SHOT VISION</Tag>
              <div style={{ marginTop: 22, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                <WordLine text="Know before" active={section === 5} size={120} />
                <WordLine text="tip-off." active={section === 5} delay={0.22} size={120} />
              </div>
              <div style={{ marginTop: 40, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: 12 }}>
                {PRODUCTS.map((p, i) => (
                  <ProductTile key={p.key} product={p} active={section === 5} delay={1 + i * 0.12}
                               onClick={p.key === "engine" ? onLaunch : undefined} />
                ))}
              </div>
            </div>
          </SectionOverlay>
        </div>
      </div>
    </>
  );
}

function SectionOverlay({
  children,
  active,
  align,
}: {
  children: React.ReactNode;
  active: boolean;
  align: "left" | "right" | "center";
}) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 10,
        display: "flex",
        alignItems: "center",
        justifyContent: align === "left" ? "flex-start" : align === "right" ? "flex-end" : "center",
        opacity: active ? 1 : 0,
        pointerEvents: active ? "auto" : "none",
        transition: "opacity 0.55s ease",
      }}
    >
      <div style={{ width: "100%" }}>{children}</div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
        color: "#C9A84C",
        letterSpacing: "0.5em",
      }}
    >
      {children}
    </div>
  );
}

function ComparisonRow({
  label,
  pct,
  color,
  epColor,
  ep,
  sub,
  active,
  delay,
}: {
  label: string;
  pct: number;
  color: string;
  epColor?: string;
  ep: string;
  /** Makes-over-attempts, so the expected-points figure is auditable. */
  sub?: string;
  active: boolean;
  delay: number;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          marginBottom: 6,
          gap: 16,
        }}
      >
        <div style={{ textAlign: "left" }}>
          <div
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              color,
              letterSpacing: "0.2em",
            }}
          >
            {label}
          </div>
          {sub ? (
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                color: "#9aa7bd",
                letterSpacing: "0.12em",
                marginTop: 3,
              }}
            >
              {sub}
            </div>
          ) : null}
        </div>
        <div
          style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 32,
            color: epColor || color,
            lineHeight: 1,
          }}
        >
          {ep}
        </div>
      </div>
      <div style={{ height: 6, background: "#1a1a2a", borderRadius: 2, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: active ? `${pct}%` : 0,
            background: color,
            transition: `width 700ms ease ${delay}s`,
          }}
        />
      </div>
    </div>
  );
}

function SectionCta({ children, active, delay, onClick, href, align = "left", marginTop = 26 }: {
  children: React.ReactNode; active: boolean; delay: number; onClick?: () => void; href?: string;
  align?: "left" | "right"; marginTop?: number;
}) {
  const style: React.CSSProperties = {
    display: "inline-block", marginTop, opacity: 0,
    animation: active ? `fadeUp 0.6s ease ${delay}s forwards` : undefined,
    fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.26em", color: "#C9A84C",
    background: "rgba(201,168,76,0.08)", border: "1px solid rgba(201,168,76,0.4)", borderRadius: 3,
    padding: "10px 16px", cursor: "pointer", textDecoration: "none", transition: "background 200ms, color 200ms",
  };
  const hover = (on: boolean) => (e: React.MouseEvent<HTMLElement>) => {
    e.currentTarget.style.background = on ? "#C9A84C" : "rgba(201,168,76,0.08)";
    e.currentTarget.style.color = on ? "#000" : "#C9A84C";
  };
  return (
    <div style={{ textAlign: align }}>
      {href ? (
        <a href={href} style={style} onMouseEnter={hover(true)} onMouseLeave={hover(false)}>{children}</a>
      ) : (
        <button onClick={onClick} style={style} onMouseEnter={hover(true)} onMouseLeave={hover(false)}>{children}</button>
      )}
    </div>
  );
}

function ProductTile({ product, active, delay, onClick }: {
  product: (typeof PRODUCTS)[number]; active: boolean; delay: number; onClick?: () => void;
}) {
  const primary = product.key === "engine";
  const body = (
    <>
      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 26, letterSpacing: "0.05em", lineHeight: 1,
                    color: primary ? "#000" : "#F0F0F0" }}>
        {product.name} →
      </div>
      <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 12.5, lineHeight: 1.45, marginTop: 8,
                    color: primary ? "rgba(0,0,0,0.72)" : "#9aa7bd" }}>
        {product.blurb}
      </div>
    </>
  );
  const style: React.CSSProperties = {
    display: "block", textAlign: "left", cursor: "pointer", textDecoration: "none", borderRadius: 6,
    padding: "18px 18px 20px", opacity: 0, width: "100%",
    animation: active ? `statPop 0.55s cubic-bezier(0.34,1.56,0.64,1) ${delay}s forwards` : undefined,
    background: primary ? "#C9A84C" : "rgba(14,14,22,0.82)",
    border: `1px solid ${primary ? "#C9A84C" : "rgba(201,168,76,0.25)"}`,
    boxShadow: primary ? "0 0 40px rgba(201,168,76,0.25)" : "none",
    transition: "transform 200ms ease, border-color 200ms ease, box-shadow 200ms ease",
  };
  const hover = (on: boolean) => (e: React.MouseEvent<HTMLElement>) => {
    e.currentTarget.style.transform = on ? "translateY(-3px)" : "none";
    if (!primary) e.currentTarget.style.borderColor = on ? "rgba(201,168,76,0.7)" : "rgba(201,168,76,0.25)";
  };
  return onClick ? (
    <button onClick={onClick} style={style} onMouseEnter={hover(true)} onMouseLeave={hover(false)}>{body}</button>
  ) : (
    <a href={product.href} style={style} onMouseEnter={hover(true)} onMouseLeave={hover(false)}>{body}</a>
  );
}

function StatCallout({
  value,
  label,
  active,
  delay,
}: {
  value: string;
  label: string;
  active: boolean;
  delay: number;
}) {
  return (
    <div
      style={{
        opacity: 0,
        animation: active
          ? `statPop 0.6s cubic-bezier(0.34,1.56,0.64,1) ${delay}s forwards`
          : undefined,
      }}
    >
      <div
        style={{
          fontFamily: "'Bebas Neue', sans-serif",
          fontSize: 80,
          color: "#C9A84C",
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#9aa7bd",
          letterSpacing: "0.3em",
          marginTop: 4,
        }}
      >
        {label}
      </div>
    </div>
  );
}
