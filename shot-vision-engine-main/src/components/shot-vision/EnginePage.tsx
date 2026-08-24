import { useState, useEffect, useRef, useMemo } from "react";
import { ZONES, heightToFeet, type Player, type ZoneResult, type HeatmapPoint, type MatchupResponse, type HealthStatus } from "@/lib/shot-vision-data";
import { searchPlayersAPI, getRecommendationHeatmapAPI, getMatchupAPI, checkHealthAPI } from "@/lib/api";

interface Props { onBack: () => void; }

const HEALTH_POLL_MS = 30000;

const LOADING_LINES = [
  "Loading player profiles...",
  "Computing zone tendencies...",
  "Running XGBoost inference...",
  "Ranking by expected points...",
];

export default function EnginePage({ onBack }: Props) {
  const [attacker, setAttacker] = useState<Player | null>(null);
  const [primaryDef, setPrimaryDef] = useState<Player | null>(null);
  const [secondaryDef, setSecondaryDef] = useState<Player | null>(null);
  const [doubleTeam, setDoubleTeam] = useState(false);
  const [quarter, setQuarter] = useState<1 | 2 | 3 | 4>(4);
  const [minutes, setMinutes] = useState(2);
  const [seconds, setSeconds] = useState(14);
  const [scoreDiff, setScoreDiff] = useState(0);
  const [home, setHome] = useState(true);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ZoneResult[] | null>(null);
  const [heatmapPoints, setHeatmapPoints] = useState<HeatmapPoint[]>([]);
  // Which side(s) of the last run had no real NBA stats (position-projected).
  const [projected, setProjected] = useState<{ attacker: boolean; defender: boolean }>({ attacker: false, defender: false });
  const [loadingIdx, setLoadingIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  // The most recent season with ingested data, fetched from the backend rather
  // than hardcoded — a hardcoded season string goes stale every year and hides
  // that season's rookies/trades from search and recommendations.
  const [season, setSeason] = useState<string | null>(null);

  const canRun = attacker && primaryDef && !!season && (!doubleTeam || secondaryDef);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      checkHealthAPI()
        .then((res) => {
          if (cancelled) return;
          setHealth(res.model_loaded ? "online" : "offline");
          if (res.latest_season) setSeason(res.latest_season);
        })
        .catch(() => { if (!cancelled) setHealth("offline"); });
    };
    poll();
    const iv = setInterval(poll, HEALTH_POLL_MS);
    return () => { cancelled = true; clearInterval(iv); };
  }, []);

  const run = () => {
    if (!canRun || !attacker || !primaryDef || !season) return;
    setLoading(true);
    setResults(null);
    setError(null);
    setLoadingIdx(0);
    const iv = setInterval(() => setLoadingIdx((i) => (i + 1) % LOADING_LINES.length), 320);

    getRecommendationHeatmapAPI(attacker.id, primaryDef.id, season, quarter, minutes * 60 + seconds, scoreDiff, home ? 1 : 0, doubleTeam ? secondaryDef?.id : null)
      .then((res) => {
        clearInterval(iv);
        const backendZones = res.zone_summary || [];
        const mappedResults: ZoneResult[] = ZONES.map(z => {
          const bZone = backendZones.find((bz) => bz.zone === z.label);
          return {
            zone: z,
            ep: bZone ? bZone.avg_ep : z.baseEP,
            makeProb: bZone ? bZone.avg_make_prob : 0.35,
            attempts: bZone?.attempts,
            actualFgPct: bZone?.actual_fg_pct,
          };
        }).sort((a, b) => b.ep - a.ep);
        setResults(mappedResults);
        setHeatmapPoints(res.heatmap || []);
        setProjected({
          attacker: res.attacker_stats_source === "prior",
          defender: res.defender_stats_source === "prior",
        });
        setLoading(false);
      })
      .catch((err) => {
        clearInterval(iv);
        console.error(err);
        setError(err instanceof Error ? err.message : "Something went wrong reaching the shot engine.");
        setLoading(false);
      });
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f", color: "#F0F0F0", fontFamily: "'Inter', sans-serif" }}>
      <header style={{ position: "sticky", top: 0, zIndex: 50, background: "rgba(10,10,15,0.92)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", borderBottom: "1px solid rgba(201,168,76,0.1)", padding: "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#F0F0F0", letterSpacing: "0.05em" }}>SHOT VISION</div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#C9A84C", letterSpacing: "0.4em", marginTop: 2 }}>SHOT QUALITY ENGINE</div>
        </div>
        <HealthIndicator status={health} />
        <button onClick={onBack} style={{ background: "transparent", border: "none", color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", cursor: "pointer", transition: "color 200ms" }} onMouseEnter={(e) => (e.currentTarget.style.color = "#C9A84C")} onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}>
          ← BACK
        </button>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "52% 48%", gap: 32, padding: "32px", maxWidth: 1440, margin: "0 auto", alignItems: "start" }}>
        {/* LEFT COL */}
        <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
          <InputSection label="ATTACKER">
            <PlayerSelect selected={attacker} onSelect={setAttacker} season={season} accent="#C9A84C" statLabels={["FG%", "3P%", "RIM%"]} statValues={(p) => [p.fg, p.tp, p.rim]} ratingValue={(p) => p.offRtg} excludeIds={[primaryDef?.id, secondaryDef?.id]} />
          </InputSection>

          <InputSection label="DEFENSE TYPE">
            <div style={{ display: "flex", gap: 8 }}>
              <Pill active={!doubleTeam} onClick={() => { setDoubleTeam(false); setSecondaryDef(null); }}>SINGLE COVERAGE</Pill>
              <Pill active={doubleTeam} onClick={() => setDoubleTeam(true)}>DOUBLE TEAM</Pill>
            </div>
          </InputSection>

          <InputSection label="PRIMARY DEFENDER">
            <PlayerSelect selected={primaryDef} onSelect={setPrimaryDef} season={season} accent="#DC2626" statLabels={["OPP FG%", "DFG DIFF%", "WINGSPAN"]} statValues={(p) => [`${p.defRtg}%`, p.contest, p.wingspanIn != null ? `${p.wingspanIn}"` : "—"]} ratingValue={(p) => p.defRtgOvr} excludeIds={[attacker?.id, secondaryDef?.id]} />
          </InputSection>

          <div style={{ maxHeight: doubleTeam ? 400 : 0, opacity: doubleTeam ? 1 : 0, overflow: doubleTeam ? "visible" : "hidden", transition: "all 400ms ease" }}>
            <InputSection label="SECONDARY DEFENDER">
              <PlayerSelect selected={secondaryDef} onSelect={setSecondaryDef} season={season} accent="rgba(220,38,38,0.5)" statLabels={["OPP FG%", "DFG DIFF%", "WINGSPAN"]} statValues={(p) => [`${p.defRtg}%`, p.contest, p.wingspanIn != null ? `${p.wingspanIn}"` : "—"]} ratingValue={(p) => p.defRtgOvr} excludeIds={[attacker?.id, primaryDef?.id]} />
            </InputSection>
          </div>

          <InputSection label="GAME STATE">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div>
                <MiniLabel>QUARTER</MiniLabel>
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                  {[1, 2, 3, 4].map((q) => (
                    <button key={q} onClick={() => setQuarter(q as 1 | 2 | 3 | 4)} style={{ flex: 1, padding: "10px 0", border: quarter === q ? "none" : "1px solid #2a2a3a", background: quarter === q ? "#C9A84C" : "#111118", color: quarter === q ? "#000" : "#F0F0F0", fontFamily: quarter === q ? "'Bebas Neue', sans-serif" : "'JetBrains Mono', monospace", fontSize: quarter === q ? 16 : 12, borderRadius: 3, cursor: "pointer", transition: "all 200ms" }}>
                      Q{q}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <MiniLabel>TIME REMAINING</MiniLabel>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8 }}>
                  <input type="number" min={0} max={12} value={minutes} onChange={(e) => setMinutes(Math.max(0, Math.min(12, +e.target.value || 0)))} style={inputStyle(68)} />
                  <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 24, color: "#C9A84C" }}>:</span>
                  <input type="number" min={0} max={59} value={seconds.toString().padStart(2, "0")} onChange={(e) => setSeconds(Math.max(0, Math.min(59, +e.target.value || 0)))} style={inputStyle(68)} />
                </div>
              </div>
              <div style={{ gridColumn: "1 / -1" }}>
                <MiniLabel>SCORE DIFFERENTIAL</MiniLabel>
                <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 8 }}>
                  <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 52, lineHeight: 1, minWidth: 90, color: scoreDiff > 0 ? "#16A34A" : scoreDiff < 0 ? "#DC2626" : "#64748b" }}>
                    {scoreDiff > 0 ? "+" : ""}{scoreDiff}
                  </div>
                  <div style={{ flex: 1 }}>
                    <input type="range" min={-30} max={30} value={scoreDiff} onChange={(e) => setScoreDiff(+e.target.value)} style={{ width: "100%", accentColor: scoreDiff > 0 ? "#16A34A" : scoreDiff < 0 ? "#DC2626" : "#C9A84C" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.2em" }}>
                      <span>YOUR TEAM</span>
                      <span>OPPONENT</span>
                    </div>
                  </div>
                </div>
              </div>
              <div style={{ gridColumn: "1 / -1" }}>
                <MiniLabel>HOME / AWAY</MiniLabel>
                <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                  <Pill active={home} onClick={() => setHome(true)}>HOME</Pill>
                  <Pill active={!home} onClick={() => setHome(false)}>AWAY</Pill>
                </div>
              </div>
            </div>
          </InputSection>

          <button
            disabled={!canRun || loading}
            onClick={run}
            style={{
              width: "100%",
              height: 58,
              marginTop: 8,
              border: "none",
              borderRadius: 3,
              cursor: canRun && !loading ? "pointer" : "not-allowed",
              background: canRun && !loading ? "#C9A84C" : "#1a1a2a",
              color: canRun && !loading ? "#000" : "#2a2a3a",
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: 22,
              letterSpacing: "0.05em",
              boxShadow: canRun && !loading ? "0 0 32px rgba(201,168,76,0.2)" : "none",
              transition: "all 250ms ease",
            }}
            onMouseEnter={(e) => { if (canRun && !loading) { e.currentTarget.style.transform = "scale(1.015)"; e.currentTarget.style.boxShadow = "0 0 56px rgba(201,168,76,0.4)"; } }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = "scale(1)"; if (canRun && !loading) e.currentTarget.style.boxShadow = "0 0 32px rgba(201,168,76,0.2)"; }}
          >
            RUN SHOT VISION →
          </button>
          {health === "offline" && (
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#DC2626", letterSpacing: "0.1em", textAlign: "center" }}>
              ⚠ ENGINE UNREACHABLE — requests will likely fail until the backend is back up
            </div>
          )}
        </div>

        {/* RIGHT COL */}
        <div style={{ position: "sticky", top: 88, background: "#0e0e16", border: "1px solid rgba(201,168,76,0.08)", borderRadius: 6, padding: 28, minHeight: 640 }}>
          {!loading && error && <ErrorOutput message={error} onRetry={run} />}
          {!loading && !error && !results && <DefaultOutput />}
          {loading && <LoadingOutput idx={loadingIdx} />}
          {!loading && !error && results && attacker && primaryDef && season && <ResultsOutput results={results} heatmapPoints={heatmapPoints} attacker={attacker} defender={primaryDef} secondaryDefender={doubleTeam ? secondaryDef : null} season={season} projected={projected} />}
        </div>
      </div>
      <div style={{ height: 40 }} />
    </div>
  );
}

const inputStyle = (width?: number): React.CSSProperties => ({
  background: "#111118",
  border: "1px solid rgba(255,255,255,0.07)",
  borderRadius: 3,
  padding: "12px 14px",
  color: "#F0F0F0",
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 18,
  outline: "none",
  width: width ?? "100%",
  textAlign: width ? "center" : "left",
});

function HealthIndicator({ status }: { status: "checking" | "online" | "offline" }) {
  const color = status === "online" ? "#16A34A" : status === "offline" ? "#DC2626" : "#64748b";
  const label = status === "online" ? "ENGINE ONLINE" : status === "offline" ? "ENGINE OFFLINE" : "CONNECTING…";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: status === "online" ? `0 0 6px ${color}` : "none", flexShrink: 0 }} />
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.25em" }}>{label}</span>
    </div>
  );
}

// Flags a player whose numbers come from src/training/position_priors.py's
// position-bucket fallback, not real recorded stats — shown whenever a
// player's/response's stats_source is "prior" (no NBA shot history yet,
// e.g. this year's draft class before their first real game).
function ProjectedBadge({ compact = false }: { compact?: boolean }) {
  return (
    <span
      title="No real NBA stats yet — this is a position-based projection, not measured data."
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: compact ? "1px 6px" : "2px 8px",
        borderRadius: 2,
        border: "1px solid rgba(201,168,76,0.35)",
        background: "rgba(201,168,76,0.08)",
        color: "#C9A84C",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: compact ? 8 : 9,
        letterSpacing: "0.15em",
        whiteSpace: "nowrap",
      }}
    >
      PROJECTED
    </span>
  );
}

function ErrorOutput({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 580, textAlign: "center", gap: 16, padding: "0 24px" }}>
      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 72, color: "#DC2626", lineHeight: 1 }}>!</div>
      <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 18, color: "#F0F0F0" }}>The engine couldn't complete that run</div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#64748b", letterSpacing: "0.03em", maxWidth: 380, lineHeight: 1.6 }}>{message}</div>
      <button
        onClick={onRetry}
        style={{ marginTop: 8, padding: "12px 28px", border: "1px solid rgba(220,38,38,0.4)", borderRadius: 3, background: "transparent", color: "#DC2626", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", cursor: "pointer", transition: "all 200ms" }}
        onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(220,38,38,0.1)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
      >
        RETRY
      </button>
    </div>
  );
}

function PlayerAvatar({ player, size = 40 }: { player: Player; size?: number }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => { setBroken(false); }, [player.id]);
  const initials = player.name
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const baseStyle: React.CSSProperties = {
    width: size,
    height: size,
    borderRadius: "50%",
    flexShrink: 0,
    background: "#1a1a2a",
    border: "1px solid rgba(201,168,76,0.2)",
  };

  if (!player.headshotUrl || broken) {
    return (
      <div style={{ ...baseStyle, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Bebas Neue', sans-serif", fontSize: Math.round(size * 0.4), color: "#C9A84C" }}>
        {initials || "?"}
      </div>
    );
  }

  return (
    <img
      src={player.headshotUrl}
      alt={player.name}
      onError={() => setBroken(true)}
      style={{ ...baseStyle, objectFit: "cover" }}
    />
  );
}

function MiniLabel({ children }: { children: React.ReactNode }) {
  return <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.3em" }}>{children}</div>;
}

function InputSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em", marginBottom: 12 }}>{label}</div>
      {children}
    </div>
  );
}

function Pill({ active, children, onClick, activeColor = "#C9A84C" }: { active: boolean; children: React.ReactNode; onClick: () => void; activeColor?: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 20px",
        borderRadius: 3,
        border: active ? "none" : "1px solid #2a2a3a",
        background: active ? activeColor : "transparent",
        color: active ? "#000" : "#64748b",
        fontFamily: active ? "'Bebas Neue', sans-serif" : "'JetBrains Mono', monospace",
        fontSize: active ? 15 : 11,
        letterSpacing: active ? "0.05em" : "0.25em",
        cursor: "pointer",
        transition: "all 200ms",
      }}
    >
      {children}
    </button>
  );
}

function PlayerSelect({ selected, onSelect, season, accent, statLabels, statValues, ratingValue, excludeIds }: {
  selected: Player | null;
  onSelect: (p: Player | null) => void;
  season: string | null;
  accent: string;
  statLabels: [string, string, string];
  statValues: (p: Player) => [number | string, number | string, number | string];
  ratingValue: (p: Player) => number | null;
  // Players already picked in another slot (e.g. the attacker, or the
  // primary defender when picking a secondary) — filtered out so a shot
  // can't be scored against a "matchup" that's really one player twice.
  excludeIds?: (string | undefined)[];
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [filtered, setFiltered] = useState<Player[]>([]);

  useEffect(() => {
    if (query.length < 2 || !season) {
      setFiltered([]);
      return;
    }
    const timer = setTimeout(() => {
      searchPlayersAPI(query, season).then(res => setFiltered(res)).catch(console.error);
    }, 250);
    return () => clearTimeout(timer);
  }, [query, season]);

  // Filtered at render time (not inside the fetch) so excluding a player
  // stays correct even if excludeIds changes without the user retyping —
  // e.g. picking the attacker after the defender dropdown already has
  // cached results.
  const visible = filtered.filter(p => !excludeIds?.includes(p.id)).slice(0, 8);

  if (selected) {
    const vals = statValues(selected);
    return (
      <div style={{ background: "#16161f", borderLeft: `3px solid ${accent}`, padding: "14px 18px", borderRadius: 3 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <PlayerAvatar player={selected} size={44} />
          <div style={{ background: accent, color: "#000", padding: "6px 10px", borderRadius: 2, fontFamily: "'Bebas Neue', sans-serif", fontSize: 20, letterSpacing: "0.02em", minWidth: 46, textAlign: "center" }}>
            {ratingValue(selected) ?? "—"}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#F0F0F0", letterSpacing: "0.02em", lineHeight: 1 }}>{selected.name}</div>
              {selected.statsSource === "prior" && <ProjectedBadge />}
            </div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b", marginTop: 4 }}>
              {selected.pos} · {heightToFeet(selected.heightIn)} · {selected.weightLbs != null ? `${selected.weightLbs}lb` : "—"}
            </div>
          </div>
          <button onClick={() => { onSelect(null); setQuery(""); }} style={{ background: "transparent", border: "none", color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, cursor: "pointer" }}>CHANGE</button>
        </div>
        <div style={{ height: 1, background: "rgba(201,168,76,0.15)", margin: "12px 0" }} />
        <div style={{ display: "flex", gap: 24 }}>
          {statLabels.map((lbl, i) => (
            <div key={lbl}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.15em" }}>{lbl}</div>
              <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 700, fontSize: 15, color: "#F0F0F0", marginTop: 2 }}>{vals[i]}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <input
        placeholder="Search player..."
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        style={{
          width: "100%",
          background: "#111118",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 3,
          padding: "14px 18px",
          color: "#F0F0F0",
          fontFamily: "'Inter', sans-serif",
          fontSize: 16,
          outline: "none",
        }}
        onFocusCapture={(e) => (e.currentTarget.style.borderColor = "rgba(201,168,76,0.4)")}
      />
      {open && visible.length > 0 && (
        <div style={{ position: "absolute", top: "100%", left: 0, right: 0, marginTop: 4, background: "#16161f", border: "1px solid rgba(201,168,76,0.15)", borderRadius: 3, maxHeight: 280, overflowY: "auto", zIndex: 20 }}>
          {visible.map((p) => (
            <div key={p.id} onMouseDown={() => { onSelect(p); setQuery(""); setOpen(false); }} style={{ padding: "10px 16px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.03)", transition: "background 150ms" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(201,168,76,0.06)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <PlayerAvatar player={p} size={28} />
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 14, color: "#F0F0F0" }}>{p.name}</div>
                    {p.statsSource === "prior" && <ProjectedBadge compact />}
                  </div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b" }}>{p.pos} · {heightToFeet(p.heightIn)}</div>
                </div>
              </div>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 18, color: "#C9A84C" }}>{ratingValue(p) ?? "—"}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DefaultOutput() {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 580, textAlign: "center" }}>
      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 120, color: "rgba(201,168,76,0.06)", lineHeight: 1 }}>?</div>
      <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 18, color: "#2a2a3a", marginTop: 8 }}>Configure the matchup</div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#1e1e2e", marginTop: 4, letterSpacing: "0.15em" }}>then run the engine.</div>
    </div>
  );
}

function LoadingOutput({ idx }: { idx: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 580, textAlign: "center", gap: 16 }}>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#C9A84C", letterSpacing: "0.4em" }}>ANALYZING MATCHUP</div>
      <div style={{ display: "flex", gap: 6 }}>
        {[0, 1, 2].map((i) => (
          <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: "#C9A84C", opacity: 0.3, animation: `goldPulse 1.2s ease ${i * 0.15}s infinite` }} />
        ))}
      </div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#334155", letterSpacing: "0.1em", marginTop: 24 }}>
        {LOADING_LINES[idx]}
      </div>
    </div>
  );
}

function ResultsOutput({ results, heatmapPoints, attacker, defender, secondaryDefender, season, projected }: { results: ZoneResult[]; heatmapPoints: HeatmapPoint[]; attacker: Player; defender: Player; secondaryDefender?: Player | null; season: string; projected: { attacker: boolean; defender: boolean } }) {
  const [showProb, setShowProb] = useState(false);
  const best = results[0];
  const worst = results[results.length - 1];

  const MetricToggle = () => (
    <div style={{ display: "flex", background: "#111118", borderRadius: 20, padding: 4, width: "fit-content", border: "1px solid rgba(255,255,255,0.05)" }}>
      <button
        onClick={() => setShowProb(false)}
        style={{
          background: !showProb ? "#C9A84C" : "transparent",
          color: !showProb ? "#000" : "#64748b",
          border: "none",
          padding: "4px 12px",
          borderRadius: 16,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          fontWeight: 700,
          cursor: "pointer",
          transition: "all 0.2s"
        }}
      >
        EP
      </button>
      <button
        onClick={() => setShowProb(true)}
        style={{
          background: showProb ? "#C9A84C" : "transparent",
          color: showProb ? "#000" : "#64748b",
          border: "none",
          padding: "4px 12px",
          borderRadius: 16,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          fontWeight: 700,
          cursor: "pointer",
          transition: "all 0.2s"
        }}
      >
        MAKE %
      </button>
    </div>
  );

  const projectedNames = [
    projected.attacker ? attacker.name : null,
    projected.defender ? defender.name : null,
  ].filter(Boolean);

  return (
    <div style={{ animation: "fadeUp 0.6s ease both", display: "flex", flexDirection: "column", gap: 20 }}>
      {projectedNames.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, background: "rgba(201,168,76,0.06)", border: "1px solid rgba(201,168,76,0.25)", borderRadius: 3, padding: "10px 14px" }}>
          <ProjectedBadge />
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#a8a29e", lineHeight: 1.5 }}>
            {projectedNames.join(" and ")} {projectedNames.length > 1 ? "have" : "has"} no real NBA shot data yet — these numbers use a position-based projection, not measured stats.
          </span>
        </div>
      )}
      {secondaryDefender && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, background: "rgba(220,38,38,0.06)", border: "1px solid rgba(220,38,38,0.25)", borderRadius: 3, padding: "10px 14px" }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#DC2626", letterSpacing: "0.15em", flexShrink: 0 }}>DOUBLE TEAM</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#a8a29e", lineHeight: 1.5 }}>
            Scored against {defender.name} + {secondaryDefender.name} help defense — the tougher of the two applies per zone (a heuristic, not a learned effect; no shot in this dataset actually records two defenders).
          </span>
        </div>
      )}
      <CourtCanvas bestKey={best.zone.key} results={results} heatmapPoints={heatmapPoints} showProb={showProb} />

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em" }}>OPTIMAL ZONE</div>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 38, color: "#F0F0F0", lineHeight: 1.05, marginTop: 4 }}>{best.zone.label}</div>
          </div>
          <MetricToggle />
        </div>
        
        <div style={{ display: "flex", alignItems: "baseline", gap: 20, marginTop: 8 }}>
          <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 64, color: "#16A34A", lineHeight: 1 }}>
            {showProb ? `${Math.round(best.makeProb * 100)}%` : best.ep.toFixed(2)}
          </div>
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.2em" }}>
              {showProb ? "EXPECTED PTS" : "MAKE PROB"}
            </div>
            <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 700, fontSize: 22, color: "#F0F0F0" }}>
              {showProb ? best.ep.toFixed(2) : `${Math.round(best.makeProb * 100)}%`}
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {results.slice(0, 3).map((r, i) => {
          const isTop = i === 0;
          const color = i === 0 ? "#16A34A" : i === 1 ? "#C9A84C" : "#B8860B";
          return (
            <div key={r.zone.key} style={{ background: "#111118", border: isTop ? "1px solid rgba(201,168,76,0.25)" : "1px solid rgba(255,255,255,0.05)", padding: "12px 16px", borderRadius: 3, boxShadow: isTop ? "0 0 32px rgba(201,168,76,0.08)" : "none" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                  <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 14, color }}>#{i + 1}</span>
                  <span style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 15, color: "#F0F0F0" }}>{r.zone.label}</span>
                </div>
                <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 32, color, lineHeight: 1 }}>
                  {showProb ? `${Math.round(r.makeProb * 100)}%` : r.ep.toFixed(2)}
                </div>
              </div>
              <div style={{ height: 3, background: "#1a1a2a", borderRadius: 2, marginTop: 8 }}>
                <div style={{ height: "100%", width: `${Math.min(100, showProb ? r.makeProb * 100 : (r.ep / 1.5) * 100)}%`, background: color, borderRadius: 2, transition: "width 700ms ease" }} />
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#334155", letterSpacing: "0.1em", marginTop: 6 }}>
                {r.attempts ? `${r.attempts} SEASON ATTEMPTS · ${r.actualFgPct?.toFixed(1)}% ACTUAL FG` : "NO SEASON VOLUME DATA"}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ background: "#0e0e16", borderLeft: "3px solid #DC2626", padding: "12px 16px" }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#DC2626", letterSpacing: "0.15em" }}>
          ⚠ AVOID · {worst.zone.label.toUpperCase()} · {showProb ? `${Math.round(worst.makeProb * 100)}% PROB` : `${worst.ep.toFixed(2)} EP`}
        </div>
      </div>

      <MatchupEdge attacker={attacker} defender={defender} season={season} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Court geometry — derived from real NBA dimensions, in FEET from the hoop.
//
// The previous version drew the court from invented proportions (`cw * 0.42`,
// `ch * 0.35`, …) while plotting the heatmap from the backend's real
// coordinates. The two could never line up, and they didn't: the three-point
// arc was drawn at 21.0 ft against a real 23.75 ft, so every above-the-break
// blob rendered 22-48px OUTSIDE the arc; the free-throw line sat 11 ft too far
// out and the paint was 8 ft too deep.
//
// Everything below now comes from one scale and one origin, so the lines and
// the data are drawn in the same space by construction.
// ─────────────────────────────────────────────────────────────────────────────
const FT = {
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
const CORNER_JOIN_Y = Math.sqrt(FT.arcRadius ** 2 - FT.cornerThreeX ** 2); // ≈ 8.95 ft

const COURT_CW = 400;
const COURT_SCALE = COURT_CW / FT.courtWidth;   // 8 px per foot
const COURT_AREA_H = Math.round(FT.visibleDepth * COURT_SCALE);

// A dedicated strip below the baseline for the scale legend. The legend used
// to be drawn at y=345 on a 375px canvas with the court occupying all of it,
// so it sat on top of the floor near the baseline; giving it its own band keeps
// it off the data and lets it carry real tick values.
const LEGEND_H = 38;
const COURT_CH = COURT_AREA_H + LEGEND_H;

const HOOP_X = COURT_CW / 2;
const HOOP_Y = COURT_AREA_H - FT.hoopFromBaseline * COURT_SCALE;

const ft = (feet: number) => feet * COURT_SCALE;

// Backend grid coordinates are tenths of a foot, basket-centred, y toward
// half-court. This is the ONLY place that conversion happens.
function courtToCanvas(locX: number, locY: number) {
  return {
    x: HOOP_X + (locX / 10) * COURT_SCALE,
    y: HOOP_Y - (locY / 10) * COURT_SCALE,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential ramp — semantic heat, strictly increasing in lightness.
//
// The old ramp ran blue → green → red, which is a rainbow: its OKLab lightness
// went 0.66 → 0.87 → 0.65, i.e. up and back down. The worst shot on the floor
// and the best shot on the floor rendered at nearly the same lightness, so in
// greyscale, in print, or to a red-green colourblind viewer they were
// indistinguishable — the encoding carried no information at all.
//
// These steps rise monotonically (0.27 → 0.95, minimum step 0.09), so magnitude
// survives loss of hue. Heat is the one multi-hue sequential exception the
// house style allows, and it is the domain's own vocabulary for this chart.
// ─────────────────────────────────────────────────────────────────────────────
const HEAT_RAMP: Array<[number, number, number]> = [
  [0x2b, 0x1a, 0x4d],
  [0x6a, 0x1f, 0x6e],
  [0xa7, 0x2c, 0x62],
  [0xd9, 0x4d, 0x3e],
  [0xf0, 0x87, 0x1f],
  [0xfb, 0xc9, 0x3d],
  [0xfd, 0xf0, 0xa8],
];

function heatColor(t: number) {
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

function CourtCanvas({ bestKey, results, heatmapPoints, showProb }: { bestKey: string; results: ZoneResult[]; heatmapPoints: HeatmapPoint[]; showProb: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

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

    // STEP 1 - HARDWOOD FLOOR BASE
    ctx.fillStyle = "#2C1A0A";
    ctx.fillRect(0, 0, cw, COURT_AREA_H);
    
    const baseGrad = ctx.createLinearGradient(0, 0, 0, ch);
    baseGrad.addColorStop(0.0, "#3D1F08");
    baseGrad.addColorStop(0.4, "#4A2810");
    baseGrad.addColorStop(0.7, "#3D1F08");
    baseGrad.addColorStop(1.0, "#2C1508");
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, cw, ch);
    
    for (let y = 0; y <= ch; y += 12) {
      ctx.strokeStyle = (y / 12) % 2 === 0 ? "rgba(180, 100, 30, 0.15)" : "rgba(120, 60, 15, 0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(cw, y);
      ctx.stroke();
    }
    
    ctx.strokeStyle = "rgba(100, 50, 10, 0.1)";
    ctx.lineWidth = 1;
    [cw * 0.25, cw * 0.5, cw * 0.75].forEach(x => {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, ch);
      ctx.stroke();
    });
    
    // ── STEP 2: SHOT-QUALITY FIELD ────────────────────────────────────────
    //
    // The previous implementation splatted one additive radial blob per grid
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
      // Scale the ramp to the data actually on screen rather than to hardcoded
      // limits, so a low-usage bench player's chart still uses the full ramp
      // instead of rendering as one flat colour.
      let lo = Math.min(...values);
      let hi = Math.max(...values);
      if (hi - lo < 1e-6) { lo -= 0.05; hi += 0.05; }

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
      const REACH = ft(7.0);
      // Total kernel weight at which the field is fully opaque. Below it the
      // field fades out, so thinly-sampled edges of the court recede instead
      // of asserting a value.
      const WEIGHT_FULL = 1.6;

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

          const idx = (gy * gw + gx) * 4;
          if (wsum <= 0) { field.data[idx + 3] = 0; continue; }

          const value = vsum / wsum;
          const { r, g, b } = heatColor((value - lo) / (hi - lo));

          // Fade on TOTAL weight rather than distance to the nearest sample.
          // Nearest-distance gives every point its own circular cutoff, so the
          // boundary of the field is a union of circles and renders as a
          // scalloped, bubbly edge. Total weight decays smoothly and its
          // contours follow the shape of the sampled region, which is both
          // better looking and a more honest depiction of where the estimate
          // is actually supported.
          const edge = Math.max(0, Math.min(1, wsum / WEIGHT_FULL));

          field.data[idx] = r;
          field.data[idx + 1] = g;
          field.data[idx + 2] = b;
          // Cubed rather than squared: the field should die off quickly past
          // the last real sample. With a gentler falloff it flooded the whole
          // upper half of the court with colour in places nobody shoots from,
          // which reads as data and is not.
          field.data[idx + 3] = Math.round(224 * edge * edge * edge);
        }
      }

      // Blit the lattice, then let drawImage upscale it smoothly.
      const tmp = document.createElement("canvas");
      tmp.width = gw;
      tmp.height = gh;
      tmp.getContext("2d")!.putImageData(field, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(tmp, 0, 0, cw, ch);

      heatScale = { lo, hi };
    }

    // ── STEP 3: COURT LINES ───────────────────────────────────────────────
    // Every line below is placed from the real dimensions in FT, through the
    // same transform the data uses, so the arc and the above-the-break cluster
    // cannot drift apart the way they did before.
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "rgba(255, 240, 200, 0.55)";
    ctx.lineWidth = 1.5;

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
    ctx.strokeStyle = "rgba(255,240,200,0.75)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(HOOP_X - ft(FT.backboardWidth / 2), HOOP_Y + ft(FT.backboardFromHoop));
    ctx.lineTo(HOOP_X + ft(FT.backboardWidth / 2), HOOP_Y + ft(FT.backboardFromHoop));
    ctx.stroke();

    // RIM
    ctx.strokeStyle = "#C9A84C";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(HOOP_X, HOOP_Y, ft(FT.rimRadius), 0, Math.PI * 2);
    ctx.stroke();

    // BASELINE + SIDELINES
    ctx.strokeStyle = "rgba(255,240,200,0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(1, baselineY);
    ctx.lineTo(cw - 1, baselineY);
    ctx.moveTo(1, 0); ctx.lineTo(1, baselineY);
    ctx.moveTo(cw - 1, 0); ctx.lineTo(cw - 1, baselineY);
    ctx.stroke();

    // STEP 4 - FLOOR VARNISH EFFECT
    const varnishGrad = ctx.createRadialGradient(cw * 0.5, COURT_AREA_H * 0.5, 0, cw * 0.5, COURT_AREA_H * 0.5, cw * 0.7);
    varnishGrad.addColorStop(0.0, "rgba(255,200,100,0.04)");
    varnishGrad.addColorStop(0.5, "rgba(255,150,50,0.02)");
    varnishGrad.addColorStop(1.0, "rgba(0,0,0,0)");
    ctx.fillStyle = varnishGrad;
    ctx.globalCompositeOperation = "overlay";
    ctx.fillRect(0, 0, cw, COURT_AREA_H);
    ctx.globalCompositeOperation = "source-over";
    
    // STEP 5 - EP LABELS
    ctx.font = "bold 12px JetBrains Mono";
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0,0,0,0.8)";
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 1;
    
    // Label each zone once at the centroid of its real grid points (falls
    // back to a rough manual position if a zone had no returned points).
    const fallbackPos: Record<string, { x: number; y: number }> = {
      restrictedArea: { x: 200, y: 318 },
      paintNonRA: { x: 200, y: 230 },
      leftCorner3: { x: 32, y: 310 },
      rightCorner3: { x: 368, y: 310 },
      aboveBreak3: { x: 200, y: 100 },
      midRange: { x: 200, y: 250 },
    };

    for (const z of ZONES) {
      const r = results.find((res) => res.zone.key === z.key);
      if (!r) continue;
      const val = showProb ? `${Math.round(r.makeProb * 100)}%` : r.ep.toFixed(2);
      const pos = centroids[z.label] ?? fallbackPos[z.key];
      if (!pos) continue;
      // Corner-3 centroids land ~12px from the sideline, so centred text ran
      // off the canvas and rendered as half a number.
      const lx = Math.min(cw - 22, Math.max(22, pos.x));
      const ly = Math.min(COURT_AREA_H - 14, Math.max(14, pos.y));
      ctx.fillText(val, lx, ly);
    }

    ctx.shadowColor = "transparent";
    
    // ── STEP 6: SCALE LEGEND ──────────────────────────────────────────────
    // The old legend was a bare gradient labelled "LOW EP" / "HIGH EP", which
    // tells a reader the direction but not the magnitude — two players' charts
    // looked identical whether they differed by 0.02 expected points or 0.6.
    // Heat is a multi-hue ramp, and the house rule is that it always ships with
    // a scale, so this one carries the real endpoints and midpoint.
    ctx.fillStyle = "#12100E";
    ctx.fillRect(0, COURT_AREA_H, cw, LEGEND_H);

    const barX = 80;
    const barW = cw - barX * 2;
    const barY = COURT_AREA_H + 12;

    const legGrad = ctx.createLinearGradient(barX, barY, barX + barW, barY);
    for (let i = 0; i < HEAT_RAMP.length; i++) {
      const [r, g, b] = HEAT_RAMP[i];
      legGrad.addColorStop(i / (HEAT_RAMP.length - 1), `rgb(${r}, ${g}, ${b})`);
    }
    ctx.fillStyle = legGrad;
    ctx.fillRect(barX, barY, barW, 6);

    ctx.font = "9px JetBrains Mono, monospace";
    ctx.fillStyle = "#8A8578";
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

    // One label, right-aligned into the gutter left of the bar. The previous
    // version also drew "COLD" at x=8, which overlapped this text.
    ctx.textAlign = "right";
    ctx.fillStyle = "#6B6759";
    ctx.fillText(showProb ? "MAKE %" : "EXP. PTS", barX - 10, barY + 11);

  }, [results, heatmapPoints, centroids, showProb]);

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

  return (
    <div style={{ position: "relative", width: "100%", borderRadius: 3, overflow: "hidden", border: "1px solid rgba(255,255,255,0.05)" }}>
      <style>{`
        @keyframes pulseRing {
          0%, 100% { opacity: 1; transform: translate(-50%, -50%) scale(1); box-shadow: 0 0 8px #C9A84C; }
          50% { opacity: 0; transform: translate(-50%, -50%) scale(1.6); box-shadow: 0 0 24px #C9A84C; }
        }
      `}</style>
      <canvas ref={canvasRef} width={COURT_CW} height={COURT_CH} style={{ width: "100%", height: "auto", display: "block" }} />
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
    </div>
  );
}

function MatchupEdge({ attacker, defender, season }: { attacker: Player; defender: Player; season: string }) {
  const [data, setData] = useState<MatchupResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    // Use the same season the run() call scores against so this stays
    // consistent with the zone results shown above it.
    getMatchupAPI(attacker.id, defender.id, season)
      .then((res) => { if (!cancelled) setData(res); })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load matchup data."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [attacker.id, defender.id, season]);

  const physicalRows = data
    ? (["height", "wingspan", "weight"] as const).map((key) => {
        const stat = data.physical_comparison[key];
        const fmt = (n: number | null) => {
          if (n == null) return "—";
          if (key === "height") return heightToFeet(n);
          if (key === "wingspan") return `${n}"`;
          return `${n}lb`;
        };
        // A delta gets its own formatter — feet'inches notation (e.g. 0'8")
        // reads as an absolute height for a small gap, not a difference.
        // Plain inches/lb makes an 8-inch or 5-pound gap unambiguous.
        const fmtDelta = (n: number) => (key === "weight" ? `${n}lb` : `${n}"`);
        const a = stat.attacker ?? 0;
        const d = stat.defender ?? 0;
        const diff = stat.diff;
        const advLabel = diff == null ? "N/A" : diff > 0 ? `+${fmtDelta(Math.abs(diff))} ATTACKER` : diff < 0 ? `${fmtDelta(Math.abs(diff))} DEFENDER` : "EVEN";
        // Each stat gets its OWN bar scale — height/wingspan (~70-96) and
        // weight (~180-260) live on completely different ranges, so a
        // shared max would make height/wingspan bars look artificially
        // tiny next to weight's.
        const rowMax = Math.max(1, a, d);
        return { key, label: key.toUpperCase(), a, d, rowMax, fmtA: fmt(stat.attacker), fmtD: fmt(stat.defender), advLabel, diff };
      })
    : [];

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em" }}>MATCHUP EDGE</div>
        {data?.size_mismatch && (
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#DC2626", letterSpacing: "0.15em" }}>⚠ SIZE MISMATCH</span>
        )}
        {(data?.attacker.stats_source === "prior" || data?.defender.stats_source === "prior") && <ProjectedBadge compact />}
      </div>

      {loading && (
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b" }}>Loading matchup data…</div>
      )}

      {!loading && error && (
        <div style={{ background: "#0e0e16", borderLeft: "3px solid #DC2626", padding: "10px 14px", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#DC2626" }}>
          {error}
        </div>
      )}

      {!loading && !error && data && (
        <>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {physicalRows.map((r) => (
              <div key={r.key}>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.2em" }}>
                  <span>{r.label}</span>
                  <span style={{ color: r.diff && r.diff > 0 ? "#C9A84C" : r.diff && r.diff < 0 ? "#DC2626" : "#64748b" }}>{r.advLabel}</span>
                </div>
                <div style={{ display: "flex", gap: 6, marginTop: 6, alignItems: "center" }}>
                  <div style={{ flex: r.a / r.rowMax, height: 6, background: "#C9A84C", borderRadius: 2 }} />
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", minWidth: 44, textAlign: "center" }}>{r.fmtA}</span>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#DC2626", minWidth: 44, textAlign: "center" }}>{r.fmtD}</span>
                  <div style={{ flex: r.d / r.rowMax, height: 6, background: "#DC2626", borderRadius: 2 }} />
                </div>
              </div>
            ))}
          </div>

          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.15em", marginTop: 16 }}>
            DEFENDER OVERALL FG% ALLOWED:{" "}
            <span style={{ color: "#F0F0F0" }}>
              {data.defender_quality.fg_pct_allowed != null ? `${(data.defender_quality.fg_pct_allowed * 100).toFixed(1)}%` : "N/A"}
            </span>
            {data.defender_quality.plus_minus != null && (
              <span style={{ marginLeft: 10, color: data.defender_quality.plus_minus < 0 ? "#16A34A" : "#DC2626" }}>
                {data.defender_quality.plus_minus > 0 ? "+" : ""}
                {(data.defender_quality.plus_minus * 100).toFixed(1)}% VS LEAGUE
              </span>
            )}
          </div>

          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.3em", margin: "16px 0 8px" }}>EXPLOIT ZONES</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {data.exploit_zones.map((z) => (
              <div
                key={z.zone}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "9px 12px",
                  background: "#111118",
                  borderRadius: 3,
                  border: z.exploit ? "1px solid rgba(22,163,74,0.3)" : "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <span style={{ fontFamily: "'Inter', sans-serif", fontSize: 13, color: "#F0F0F0" }}>{z.zone}</span>
                <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b" }}>
                    {z.attacker_fg_pct != null ? `${(z.attacker_fg_pct * 100).toFixed(0)}%` : "—"} vs {z.defender_fg_pct_allowed != null ? `${(z.defender_fg_pct_allowed * 100).toFixed(0)}%` : "—"}
                  </span>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 700, color: z.exploit ? "#16A34A" : "#64748b", minWidth: 56, textAlign: "right" }}>
                    {z.matchup_advantage != null ? `${z.matchup_advantage > 0 ? "+" : ""}${(z.matchup_advantage * 100).toFixed(1)}%` : "N/A"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}