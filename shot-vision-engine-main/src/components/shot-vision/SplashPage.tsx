import { useEffect, useRef, useState } from "react";
import VideoBackdrop from "./video-backdrop";
import { LeagueBrandMark, NBAEdgeBezel, teamColor, teamLogoUrl } from "@/components/stat-engine/ui";

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
 * Best and worst scoring zone for four current stars, from the same
 * player_zone_stats table, 2025-26, minimum 15 attempts in a zone. These are
 * public performance statistics — the thing the engine exists to analyse.
 *
 * Tatum's sample is far smaller than the others'; `fga` is shown per player so
 * that is visible rather than buried.
 */
// team/teamId are each player's real current franchise — abbreviation feeds
// teamColor() for the row accent, teamId feeds the same /team/{id}/logo
// proxy the rest of the app uses, both confirmed against the live /teams
// endpoint rather than typed from memory.
const PLAYER_SPREAD = [
  {
    name: "LUKA DONČIĆ",
    team: "LAL",
    teamId: "1610612747",
    fga: 1457,
    best: ["RESTRICTED AREA", 1.63],
    worst: ["LEFT CORNER 3", 0.4],
  },
  {
    name: "VICTOR WEMBANYAMA",
    team: "SAS",
    teamId: "1610612759",
    fga: 1080,
    best: ["RESTRICTED AREA", 1.5],
    worst: ["MID-RANGE", 0.82],
  },
  {
    name: "CADE CUNNINGHAM",
    team: "DET",
    teamId: "1610612765",
    fga: 1189,
    best: ["RESTRICTED AREA", 1.26],
    worst: ["IN THE PAINT", 0.84],
  },
  {
    name: "JAYSON TATUM",
    team: "BOS",
    teamId: "1610612738",
    fga: 287,
    best: ["RESTRICTED AREA", 1.23],
    worst: ["MID-RANGE", 0.7],
  },
] as const;

const LUKA_BEST = LUKA_ZONES[0];
const LUKA_WORST = LUKA_ZONES[LUKA_ZONES.length - 1];
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

          {/* League wordmark — a static, persistent brand mark like the nav
              dots beside it. No animation, no motion: this is a chrome
              addition, not part of the hero's scroll-triggered sequence, and
              intentionally does not touch VideoBackdrop or any keyframe
              above. Silently disappears if the logo proxy can't be reached. */}
          <div style={{ position: "absolute", left: 32, top: 32, zIndex: 20 }}>
            <LeagueBrandMark height={34} />
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

          {/* SECTION 0 */}
          <SectionOverlay active={section === 0} align="center">
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                pointerEvents: "none",
              }}
            >
              <div
                style={{
                  fontFamily: "'Bebas Neue', sans-serif",
                  fontSize: 200,
                  color: "rgba(201,168,76,0.018)",
                  letterSpacing: "0.02em",
                }}
              >
                VISION
              </div>
            </div>
            <div
              style={{
                position: "relative",
                maxWidth: 700,
                margin: "0 auto",
                padding: "0 24px",
                textAlign: "center",
              }}
            >
              <Tag>SHOT VISION</Tag>
              <div style={{ marginTop: 24 }}>
                <WordLine text="Every bucket" active={section === 0} />
                <WordLine text="starts here." active={section === 0} delay={0.18} />
              </div>
              <p
                style={{
                  marginTop: 32,
                  opacity: 0,
                  animation: section === 0 ? "fadeUp 0.7s ease 0.9s forwards" : undefined,
                  fontFamily: "'Inter', sans-serif",
                  fontWeight: 400,
                  fontSize: 17,
                  color: "#9aa7bd",
                  maxWidth: 460,
                  marginLeft: "auto",
                  marginRight: "auto",
                  lineHeight: 1.55,
                }}
              >
                2 seconds. One defender. One decision. SHOT VISION already knows the answer.
              </p>
            </div>
            <div
              style={{
                position: "absolute",
                bottom: 40,
                left: "50%",
                transform: "translateX(-50%)",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 9,
                  color: "#C9A84C",
                  letterSpacing: "0.4em",
                }}
              >
                SCROLL
              </div>
              <div
                style={{
                  width: 1,
                  height: 40,
                  background: "#C9A84C",
                  margin: "10px auto 0",
                  animation: "scrollNudge 1.6s ease infinite",
                  transformOrigin: "top",
                }}
              />
            </div>
          </SectionOverlay>

          {/* SECTION 1 */}
          <SectionOverlay active={section === 1} align="left">
            <div style={{ paddingLeft: "8vw", maxWidth: 620 }}>
              <Tag>— THE MOMENT</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="The closeout" active={section === 1} size={90} />
                <WordLine text="is coming." active={section === 1} delay={0.2} size={90} />
              </div>
              <p
                style={{
                  marginTop: 28,
                  opacity: 0,
                  animation: section === 1 ? "fadeUp 0.7s ease 0.9s forwards" : undefined,
                  fontFamily: "'Inter', sans-serif",
                  fontSize: 16,
                  color: "#9aa7bd",
                  maxWidth: 400,
                  lineHeight: 1.55,
                }}
              >
                A 6'9" wing at full sprint. 80 inches of wingspan. You have 0.4 seconds. SHOT VISION
                already has the answer.
              </p>
              <div
                style={{
                  marginTop: 40,
                  opacity: 0,
                  animation:
                    section === 1
                      ? "statPop 0.6s cubic-bezier(0.34,1.56,0.64,1) 1.3s forwards"
                      : undefined,
                }}
              >
                <div
                  style={{
                    fontFamily: "'Bebas Neue', sans-serif",
                    fontSize: 72,
                    color: "#C9A84C",
                    lineHeight: 1,
                  }}
                >
                  0.4
                </div>
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 10,
                    color: "#9aa7bd",
                    letterSpacing: "0.4em",
                    marginTop: 6,
                  }}
                >
                  SECONDS TO DECIDE
                </div>
              </div>
            </div>
          </SectionOverlay>

          {/* SECTION 2 */}
          <SectionOverlay active={section === 2} align="right">
            <div
              style={{ paddingRight: "8vw", maxWidth: 560, marginLeft: "auto", textAlign: "right" }}
            >
              <Tag>— SHOT QUALITY</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="Not all shots" active={section === 2} size={90} />
                <WordLine text="are equal." active={section === 2} delay={0.18} size={90} />
              </div>
              <div
                style={{
                  marginTop: 14,
                  opacity: 0,
                  animation: section === 2 ? "fadeUp 0.7s ease 0.8s forwards" : undefined,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  gap: 8,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  color: "#C9A84C",
                  letterSpacing: "0.22em",
                }}
              >
                LUKA DONČIĆ · {LUKA_SEASON} · {LUKA_FGA.toLocaleString()} FGA
                {/* Same real crest proxy as the rest of the app — this is the
                    Lakers row, so it should look like one. */}
                <img
                  src={teamLogoUrl("1610612747")!}
                  alt=""
                  aria-hidden="true"
                  loading="lazy"
                  onError={(e) => { e.currentTarget.style.display = "none"; }}
                  style={{ height: 18, width: 18, objectFit: "contain", flexShrink: 0 }}
                />
              </div>
              <div
                style={{
                  marginTop: 22,
                  opacity: 0,
                  animation: section === 2 ? "fadeUp 0.7s ease 0.9s forwards" : undefined,
                }}
              >
                {LUKA_ZONES.map((z, i) => (
                  <div key={z.zone} style={{ marginTop: i === 0 ? 0 : 12 }}>
                    <ComparisonRow
                      label={z.zone}
                      // Bars are scaled against his best zone, so the drop-off
                      // reads at a glance rather than needing the numbers.
                      pct={(z.ep / LUKA_BEST.ep) * 100}
                      color={z.ep >= 1.2 ? "#C9A84C" : z.ep >= 0.85 ? "#B06A2C" : "#DC2626"}
                      epColor={z.ep >= 1.2 ? "#16A34A" : undefined}
                      ep={`${z.ep.toFixed(2)} EP`}
                      sub={`${z.fgm}/${z.fga} · ${((z.fgm / z.fga) * 100).toFixed(1)}%`}
                      active={section === 2}
                      delay={1.0 + i * 0.12}
                    />
                  </div>
                ))}
                <div
                  style={{
                    marginTop: 20,
                    opacity: 0,
                    animation: section === 2 ? "fadeUp 0.6s ease 2s forwards" : undefined,
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 12,
                    color: "#C9A84C",
                  }}
                >
                  {(LUKA_BEST.ep - LUKA_WORST.ep).toFixed(2)} EP between his best zone and his worst
                </div>
              </div>
            </div>
          </SectionOverlay>

          {/* SECTION 3 */}
          <SectionOverlay active={section === 3} align="left">
            <div style={{ paddingLeft: "8vw", maxWidth: 900 }}>
              <Tag>— THE DATASET</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="2.1 million shots." active={section === 3} size={90} />
                <WordLine text="10 seasons." active={section === 3} delay={0.2} size={90} />
              </div>
              <div style={{ marginTop: 44, display: "flex", gap: 64, flexWrap: "wrap" }}>
                <StatCallout
                  value="10"
                  label="SEASONS ANALYZED"
                  active={section === 3}
                  delay={1.0}
                />
                <StatCallout
                  value="2.1M"
                  label="SHOTS PROCESSED"
                  active={section === 3}
                  delay={1.2}
                />
                <StatCallout value="<200" label="MS INFERENCE" active={section === 3} delay={1.4} />
              </div>
            </div>
          </SectionOverlay>

          {/* SECTION 4 */}
          <SectionOverlay active={section === 4} align="right">
            <div
              style={{ paddingRight: "8vw", maxWidth: 620, marginLeft: "auto", textAlign: "right" }}
            >
              <Tag>— THE INTELLIGENCE</Tag>
              <div style={{ marginTop: 20 }}>
                <WordLine text="XGBoost." active={section === 4} size={90} />
                <WordLine text="Trained on" active={section === 4} delay={0.15} size={90} />
                <WordLine text="mismatches." active={section === 4} delay={0.3} size={90} />
              </div>
              <p
                style={{
                  marginTop: 32,
                  marginLeft: "auto",
                  opacity: 0,
                  animation: section === 4 ? "fadeUp 0.7s ease 1.1s forwards" : undefined,
                  fontFamily: "'Inter', sans-serif",
                  fontSize: 16,
                  color: "#9aa7bd",
                  maxWidth: 440,
                  lineHeight: 1.55,
                }}
              >
                Height differential. Wingspan. Contest rate. Zone tendencies. Game state. All of it
                in under 200ms.
              </p>
              <div
                style={{
                  marginTop: 30,
                  marginLeft: "auto",
                  maxWidth: 470,
                  opacity: 0,
                  animation: section === 4 ? "fadeUp 0.7s ease 1.35s forwards" : undefined,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 9,
                    color: "#6b7a92",
                    letterSpacing: "0.24em",
                    paddingBottom: 8,
                    borderBottom: "1px solid rgba(201,168,76,0.22)",
                  }}
                >
                  <span>PLAYER · {LUKA_SEASON}</span>
                  <span>BEST / WORST ZONE · EP</span>
                </div>
                {PLAYER_SPREAD.map((pl, i) => (
                  <div
                    key={pl.name}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 14,
                      padding: "9px 0",
                      borderBottom:
                        i === PLAYER_SPREAD.length - 1
                          ? "none"
                          : "1px solid rgba(255,255,255,0.06)",
                      textAlign: "left",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
                      {/* Real team crest, same proxy every other team logo in
                          the app uses — the one piece of this row that
                          wasn't here before, so these four real players read
                          as "on a real team" rather than as names in a list. */}
                      <img
                        src={teamLogoUrl(pl.teamId)!}
                        alt=""
                        aria-hidden="true"
                        loading="lazy"
                        onError={(e) => { e.currentTarget.style.display = "none"; }}
                        style={{ height: 26, width: 26, objectFit: "contain", flexShrink: 0 }}
                      />
                      <div style={{ minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                          <div
                            style={{
                              fontFamily: "'Bebas Neue', sans-serif",
                              fontSize: 19,
                              color: "#F0F0F0",
                              letterSpacing: "0.03em",
                              lineHeight: 1.1,
                            }}
                          >
                            {pl.name}
                          </div>
                          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: teamColor(pl.team), letterSpacing: "0.1em" }}>
                            {pl.team}
                          </div>
                        </div>
                        <div
                          style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontSize: 9,
                            color: "#6b7a92",
                            letterSpacing: "0.14em",
                          }}
                        >
                          {pl.fga.toLocaleString()} FGA
                        </div>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 18, flexShrink: 0 }}>
                      <ZoneChip zone={pl.best[0]} ep={pl.best[1]} tone="#C9A84C" />
                      <ZoneChip zone={pl.worst[0]} ep={pl.worst[1]} tone="#DC2626" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </SectionOverlay>

          {/* SECTION 5 - CTA */}
          <SectionOverlay active={section === 5} align="center">
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                pointerEvents: "none",
              }}
            >
              <div
                style={{
                  width: 600,
                  height: 600,
                  borderRadius: "50%",
                  background:
                    "radial-gradient(circle, rgba(201,168,76,0.08) 0%, rgba(201,168,76,0) 70%)",
                  animation: "ambientPulse 5s ease infinite",
                }}
              />
            </div>
            <div
              style={{
                position: "relative",
                textAlign: "center",
                padding: "0 24px",
                maxWidth: 1000,
                margin: "0 auto",
              }}
            >
              <Tag>SHOT VISION — NBA SHOT QUALITY ENGINE</Tag>
              <div
                style={{
                  marginTop: 28,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                <WordLine text="Know before" active={section === 5} size={130} />
                <WordLine text="you shoot." active={section === 5} delay={0.22} size={130} />
              </div>
              <div
                style={{
                  width: 48,
                  height: 1,
                  background: "#C9A84C",
                  margin: "36px auto 0",
                  opacity: 0,
                  animation: section === 5 ? "fadeUp 0.6s ease 1s forwards" : undefined,
                }}
              />
              <p
                style={{
                  marginTop: 20,
                  opacity: 0,
                  animation: section === 5 ? "fadeUp 0.7s ease 1.1s forwards" : undefined,
                  fontFamily: "'Inter', sans-serif",
                  fontSize: 17,
                  color: "#9aa7bd",
                }}
              >
                Input the matchup. Run the engine.
              </p>
              <button
                onClick={onLaunch}
                style={{
                  marginTop: 40,
                  opacity: 0,
                  animation:
                    section === 5
                      ? "statPop 0.6s cubic-bezier(0.34,1.56,0.64,1) 1.4s forwards"
                      : undefined,
                  background: "#C9A84C",
                  color: "#000",
                  border: "none",
                  fontFamily: "'Bebas Neue', sans-serif",
                  fontSize: 22,
                  letterSpacing: "0.05em",
                  padding: "18px 52px",
                  borderRadius: 3,
                  boxShadow: "0 0 40px rgba(201,168,76,0.25)",
                  cursor: "pointer",
                  transition: "all 300ms ease",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = "0 0 64px rgba(201,168,76,0.45)";
                  e.currentTarget.style.transform = "scale(1.03)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = "0 0 40px rgba(201,168,76,0.25)";
                  e.currentTarget.style.transform = "scale(1)";
                }}
              >
                RUN SHOT VISION →
              </button>
              <div
                style={{
                  marginTop: 40,
                  opacity: 0,
                  animation: section === 5 ? "fadeUp 0.6s ease 1.7s forwards" : undefined,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 9,
                  color: "#5a5a70",
                  letterSpacing: "0.3em",
                }}
              >
                SHOT VISION · XGBOOST · 12 SEASONS · &lt;200MS INFERENCE
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

function ZoneChip({ zone, ep, tone }: { zone: string; ep: number; tone: string }) {
  return (
    <div style={{ textAlign: "right", minWidth: 92 }}>
      <div
        style={{
          fontFamily: "'Bebas Neue', sans-serif",
          fontSize: 24,
          color: tone,
          lineHeight: 1,
        }}
      >
        {ep.toFixed(2)}
      </div>
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 8,
          color: "#6b7a92",
          letterSpacing: "0.12em",
          marginTop: 3,
        }}
      >
        {zone}
      </div>
    </div>
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
