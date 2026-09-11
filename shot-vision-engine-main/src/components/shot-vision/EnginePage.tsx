import { useState, useEffect, useRef, useMemo } from "react";
import { ZONES, heightToFeet, type Player, type ZoneResult, type HeatmapPoint, type MatchupResponse, type HealthStatus, type AttainabilityExplanation } from "@/lib/shot-vision-data";
import { searchPlayersAPI, getRecommendationHeatmapAPI, getMatchupAPI, checkHealthAPI, getAttainabilityExplanationAPI } from "@/lib/api";

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
            attemptsBehind: bZone?.attempts_behind,
            shotDistance: bZone?.shot_distance,
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
        {/* Status and nav sit together on the right. Under space-between the
            status pill landed dead-centre with nothing on its axis, reading as
            a stray element rather than as chrome. */}
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <HealthIndicator status={health} />
          <button onClick={onBack} style={{ background: "transparent", border: "none", color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", cursor: "pointer", transition: "color 200ms" }} onMouseEnter={(e) => (e.currentTarget.style.color = "#C9A84C")} onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}>
            ← BACK
          </button>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "52% 48%", gap: 32, padding: "32px", maxWidth: 1440, margin: "0 auto", alignItems: "start" }}>
        {/* LEFT COL
            Deliberately NOT position:sticky. Pinning it fills the empty space
            beside the taller results column, but sticky opens a stacking
            context around the player-search dropdowns nested in here and the
            defender dropdown stops receiving clicks — a broken control is a
            worse trade than uneven whitespace. */}
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
      <CourtCanvas bestKey={best.zone.key} results={results} heatmapPoints={heatmapPoints} showProb={showProb} attacker={attacker} defender={defender} season={season} />

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em" }}>OPTIMAL ZONE</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 38, color: "#F0F0F0", lineHeight: 1.05, marginTop: 4 }}>{best.zone.label}</div>
              {best.shotDistance != null ? (
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: "#64748b" }}>
                  {best.shotDistance.toFixed(1)} FT
                </div>
              ) : null}
            </div>
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
                  {r.shotDistance != null ? (
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b" }}>
                      {r.shotDistance.toFixed(1)} FT
                    </span>
                  ) : null}
                </div>
                <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 32, color, lineHeight: 1 }}>
                  {showProb ? `${Math.round(r.makeProb * 100)}%` : r.ep.toFixed(2)}
                </div>
              </div>
              <div style={{ height: 3, background: "#1a1a2a", borderRadius: 2, marginTop: 8 }}>
                <div style={{ height: "100%", width: `${Math.min(100, showProb ? r.makeProb * 100 : (r.ep / 1.5) * 100)}%`, background: color, borderRadius: 2, transition: "width 700ms ease" }} />
              </div>
              {/* Evidence behind the estimate. Rendered only when there is
                  some — an empty caption row is quieter than a placeholder
                  telling the reader what the app does not know. */}
              {r.attemptsBehind && r.attemptsBehind > 0 ? (
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em", marginTop: 6 }}>
                  {Math.round(r.attemptsBehind).toLocaleString()} SIMILAR SHOTS BEHIND THIS
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <div style={{ background: "#0e0e16", borderLeft: "3px solid #DC2626", padding: "12px 16px" }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#DC2626", letterSpacing: "0.15em" }}>
          ⚠ AVOID · {worst.zone.label.toUpperCase()}{worst.shotDistance != null ? ` · ${worst.shotDistance.toFixed(1)} FT` : ""} · {showProb ? `${Math.round(worst.makeProb * 100)}% PROB` : `${worst.ep.toFixed(2)} EP`}
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

// How far the heat field reaches from a real sample point. This is the single
// source of truth for it: the renderer fades the field out past this radius,
// and click hit-testing uses the same number so that anywhere the map shows
// colour is somewhere you can click. They were separate values briefly, and
// the result was large visibly-hot regions that silently ignored clicks.
const FIELD_REACH = ft(7.0);

// Peach hardwood. Markings are dark on a light floor, the way a real court
// is painted — light lines would disappear into the boards.
const COURT_SURFACE = "#DCB489";
const COURT_PLANK = "rgba(140, 95, 52, 0.22)";
const COURT_GRAIN = "rgba(112, 74, 40, 0.075)";
const COURT_LINE = "rgba(92, 58, 30, 0.5)";
const COURT_LINE_STRONG = "rgba(58, 35, 16, 0.85)";
const COURT_RIM = "#B4491C";
const LEGEND_BG = "#15110D";

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
const HEAT_RAMP: Array<[number, number, number]> = [
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
function heatSwatch(r: number, g: number, b: number) {
  const surface = [0xdc, 0xb4, 0x89];
  return `rgb(${Math.round((r * surface[0]) / 255)}, ${Math.round((g * surface[1]) / 255)}, ${Math.round((b * surface[2]) / 255)})`;
}

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

// Headshot for the on-floor matchup badges, as a decoded <img> the canvas can
// draw. Deliberately NOT crossOrigin: the NBA CDN is not guaranteed to send
// CORS headers, and a rejected CORS load is a permanently blank badge, whereas
// tainting this canvas costs nothing — the field is composed in a separate
// offscreen canvas and no code ever reads pixels back off the visible one.
function useHeadshot(url?: string) {
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
function useFontsReady() {
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
function badgeName(name: string) {
  const parts = name.trim().split(/\s+/);
  if (parts.length < 2) return name.toUpperCase();
  return `${parts[0][0]}. ${parts.slice(1).join(" ")}`.toUpperCase();
}

function initialsOf(name: string) {
  return name.split(" ").filter(Boolean).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

// One corner decal: circular headshot, accent ring, role label and name
// painted onto the boards. `side` decides which way the text runs so both
// badges hug their own corner.
function drawMatchupBadge(
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

function CourtCanvas({ bestKey, results, heatmapPoints, showProb, attacker, defender, season }: { bestKey: string; results: ZoneResult[]; heatmapPoints: HeatmapPoint[]; showProb: boolean; attacker: Player; defender: Player; season: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Click-to-inspect. Every grid point already carries its shot type,
  // distance and model outputs; before this they were only ever aggregated
  // into the heat field, so a specific shot could be seen but not read.
  const [selected, setSelected] = useState<number | null>(null);

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
    // Peach hardwood. The earlier version of this had a vertical gradient,
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
    // The old legend was a bare gradient labelled "LOW EP" / "HIGH EP", which
    // tells a reader the direction but not the magnitude — two players' charts
    // looked identical whether they differed by 0.02 expected points or 0.6.
    // Heat is a multi-hue ramp, and the house rule is that it always ships with
    // a scale, so this one carries the real endpoints and midpoint.
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

    // One label, right-aligned into the gutter left of the bar. The previous
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
      <ShotDetail
        point={sel.pt}
        attacker={attacker}
        defender={defender}
        showProb={showProb}
        season={season}
        onClose={() => setSelected(null)}
      />
    )}
    </>
  );
}

// One grid point, read out in full. The heat field can only show magnitude;
// this is where the shot the model is actually scoring becomes legible.
function ShotDetail({ point, attacker, defender, showProb, season, onClose }: { point: HeatmapPoint; attacker: Player; defender: Player; showProb: boolean; season: string; onClose: () => void }) {
  // `shot_type` is only ever "2PT/3PT Field Goal". `best_mechanic` is the
  // actual shot — stepback, cutting, driving — so it leads.
  const mechanic = point.best_mechanic ? point.best_mechanic.toUpperCase() : null;
  const isThree = /3PT/.test(point.shot_type);

  // The attainability breakdown is fetched only when opened. Six zone
  // explanations per heatmap request would be computed for every shot a user
  // never asks about.
  const [whyOpen, setWhyOpen] = useState(false);
  const [why, setWhy] = useState<AttainabilityExplanation | null>(null);
  const [whyError, setWhyError] = useState<string | null>(null);

  // A new shot means the open explanation describes the wrong zone.
  useEffect(() => {
    setWhyOpen(false);
    setWhy(null);
    setWhyError(null);
  }, [point.zone, point.loc_x, point.loc_y, attacker.id, season]);

  useEffect(() => {
    if (!whyOpen || why || whyError) return;
    let cancelled = false;
    getAttainabilityExplanationAPI(attacker.id, point.zone, season, point.loc_x, point.loc_y)
      .then((res) => { if (!cancelled) setWhy(res); })
      .catch((e) => { if (!cancelled) setWhyError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [whyOpen, why, whyError, attacker.id, point.zone, point.loc_x, point.loc_y, season]);

  const band =
    point.ep_low != null && point.ep_high != null
      ? `${point.ep_low.toFixed(2)}–${point.ep_high.toFixed(2)}`
      : null;

  const rows: Array<{ label: string; value: string; accent?: string; note?: string; expandable?: boolean }> = [
    {
      label: "MAKE PROB",
      value: `${(point.make_probability * 100).toFixed(1)}%`,
      accent: showProb ? "#C9A84C" : undefined,
    },
    {
      label: "EXPECTED PTS",
      value: point.expected_points.toFixed(2),
      accent: showProb ? undefined : "#C9A84C",
      note: band ? `95% ${band}` : undefined,
    },
    { label: "SHOT QUALITY", value: point.shot_quality_score.toFixed(2) },
    { label: "DIFFICULTY", value: point.difficulty_score.toFixed(2) },
  ];

  if (point.ep_vs_own_average != null) {
    const v = point.ep_vs_own_average;
    rows.push({
      label: "VS HIS AVERAGE",
      value: `${v >= 0 ? "+" : ""}${v.toFixed(2)} EP`,
      accent: v >= 0 ? "#16A34A" : "#DC2626",
    });
  }
  if (point.attainability != null) {
    rows.push({
      label: "ATTAINABILITY",
      // Same precision the breakdown uses. Rounding a 1.6% to "2%" here while
      // the panel below reads "1.6%" makes one number look like two.
      value: formatAttainability(point.attainability),
      note: "tap for why",
      expandable: true,
    });
  }
  if (point.attempts_behind != null) {
    rows.push({
      label: "SAMPLE",
      value: point.attempts_behind.toLocaleString(),
      note: "similar shots",
    });
  }
  if (typeof point.shot_volume === "number") rows.push({ label: "ATTEMPTS HERE", value: String(point.shot_volume) });

  return (
    <div style={{
      marginTop: 10,
      background: "#111118",
      border: "1px solid rgba(201,168,76,0.28)",
      borderRadius: 3,
      padding: "12px 14px 14px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#C9A84C", letterSpacing: "0.3em" }}>
            SELECTED SHOT
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 3, flexWrap: "wrap" }}>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 28, color: "#F0F0F0", lineHeight: 1.05 }}>
              {mechanic ?? point.shot_type}
            </div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#9aa7bd", letterSpacing: "0.1em" }}>
              {point.shot_distance.toFixed(1)} FT · {isThree ? "3PT" : "2PT"} · {point.zone.toUpperCase()}
            </div>
          </div>
          <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 11, color: "#64748b", marginTop: 3 }}>
            {attacker.name} vs {defender.name}
            {point.best_mechanic_prob != null && mechanic
              ? ` · model picks ${mechanic.toLowerCase()} ${(point.best_mechanic_prob * 100).toFixed(0)}% of the time here`
              : ""}
          </div>
        </div>
        <button
          onClick={onClose}
          aria-label="Clear selected shot"
          style={{
            background: "transparent",
            border: "1px solid rgba(255,255,255,0.14)",
            color: "#9aa7bd",
            borderRadius: 3,
            fontSize: 11,
            lineHeight: 1,
            padding: "5px 8px",
            cursor: "pointer",
            flexShrink: 0,
          }}
        >
          CLEAR
        </button>
      </div>
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(104px, 1fr))",
        gap: "10px 14px",
        marginTop: 12,
        paddingTop: 12,
        borderTop: "1px solid rgba(255,255,255,0.07)",
      }}>
        {rows.map((r) => {
          const body = (
            <>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
                {r.label}
              </div>
              <div style={{
                fontFamily: "'Inter', sans-serif",
                fontWeight: 600,
                fontSize: 15,
                color: r.accent || "#F0F0F0",
                marginTop: 3,
              }}>
                {r.value}
                {r.expandable && (
                  <span style={{ fontSize: 9, color: "#C9A84C", marginLeft: 5 }}>
                    {whyOpen ? "▾" : "▸"}
                  </span>
                )}
              </div>
              {r.note && (
                <div style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: r.expandable ? 9 : 8,
                  letterSpacing: r.expandable ? "0.08em" : undefined,
                  color: r.expandable ? "#C9A84C" : "#4a5568",
                  marginTop: r.expandable ? 4 : 2,
                  // The expandable one reads as a control rather than a
                  // caption: a bordered chip at 8px in the same muted grey as
                  // every other label was the single least discoverable thing
                  // on the panel, and it is the entry point to the whole
                  // explanation feature.
                  ...(r.expandable ? {
                    display: "inline-block",
                    border: "1px solid rgba(201,168,76,0.45)",
                    borderRadius: 3,
                    padding: "2px 6px",
                    background: "rgba(201,168,76,0.08)",
                  } : {}),
                }}>
                  {r.note}
                </div>
              )}
            </>
          );

          if (!r.expandable) return <div key={r.label}>{body}</div>;
          return (
            <button
              key={r.label}
              onClick={() => setWhyOpen((v) => !v)}
              aria-expanded={whyOpen}
              aria-label={`${r.label} ${r.value}. Show why.`}
              style={{
                background: "transparent",
                border: "none",
                padding: 0,
                textAlign: "left",
                cursor: "pointer",
                font: "inherit",
                transition: "opacity 160ms ease",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.opacity = "0.75"; }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = "1"; }}
            >
              {body}
            </button>
          );
        })}
      </div>
      {whyOpen && (
        <AttainabilityWhy explanation={why} error={whyError} />
      )}
    </div>
  );
}

// The attainability breakdown, opened from the shot detail panel.
//
// Reports the split the model actually makes: what any player would get in
// this zone, then what this player's own game adds or gives up. Collapsing
// those into one number is what makes a bare "4%" unreadable — it hides
// whether the shot is rare for everybody or rare for him.
function AttainabilityWhy({ explanation, error }: { explanation: AttainabilityExplanation | null; error: string | null }) {
  const wrap: React.CSSProperties = {
    marginTop: 10,
    paddingTop: 10,
    borderTop: "1px solid rgba(201,168,76,0.2)",
  };

  if (error) {
    return (
      <div style={{ ...wrap, fontFamily: "'Inter', sans-serif", fontSize: 11, color: "#DC2626" }}>
        {error}
      </div>
    );
  }
  if (!explanation) {
    return (
      <div style={{ ...wrap, fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.18em" }}>
        LOADING…
      </div>
    );
  }

  const pct = formatAttainability;
  const delta = explanation.player_effect;

  return (
    <div style={wrap}>
      <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 12, color: "#CBD5E1", lineHeight: 1.5 }}>
        {explanation.summary}
      </div>

      <div style={{ display: "flex", gap: 18, marginTop: 10, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
            TYPICAL PLAYER HERE
          </div>
          <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 14, color: "#94A3B8", marginTop: 2 }}>
            {pct(explanation.baseline)}
          </div>
        </div>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
            HIS GAME
          </div>
          <div style={{
            fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 14,
            color: delta >= 0 ? "#16A34A" : "#DC2626", marginTop: 2,
          }}>
            {delta >= 0 ? "+" : ""}{pct(delta)}
          </div>
        </div>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
            RESULT
          </div>
          <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 14, color: "#C9A84C", marginTop: 2 }}>
            {pct(explanation.attainability)}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
        {explanation.factors.map((f) => {
          const lowers = f.direction === "lowers";
          // Bar width is relative to the largest factor shown, so the ranking
          // stays legible whether the spread is 6 points or half a point.
          const max = Math.max(...explanation.factors.map((x) => Math.abs(x.impact)), 1e-9);
          const width = `${Math.max(4, (Math.abs(f.impact) / max) * 100)}%`;
          // Enough precision to separate the factors actually being compared.
          // Fixed at one decimal, a spread like 0.14/0.11/0.09pp collapsed to
          // three identical "0.1pp" labels sitting beside three visibly
          // different bars, which reads as a rendering fault rather than as
          // the small-but-real differences it is.
          const decimals = max * 100 < 1 ? 2 : 1;
          return (
            <div key={f.feature}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 11, color: "#E2E8F0", minWidth: 0 }}>
                  {f.detail}
                  <span style={{ color: "#64748b" }}>
                    {" — "}{f.label.toLowerCase()} {f.display_value}
                    {f.percentile != null && `, ${ordinal(f.percentile)} pct`}
                  </span>
                </div>
                <div style={{
                  fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
                  color: lowers ? "#DC2626" : "#16A34A", whiteSpace: "nowrap",
                }}>
                  {lowers ? "−" : "+"}{(Math.abs(f.impact) * 100).toFixed(decimals)}pp
                </div>
              </div>
              <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, marginTop: 4 }}>
                <div style={{
                  width, height: "100%", borderRadius: 2,
                  background: lowers ? "#DC2626" : "#16A34A", opacity: 0.75,
                }} />
              </div>
            </div>
          );
        })}
      </div>

      {explanation.creation?.note && (
        <div style={{
          marginTop: 12,
          paddingTop: 10,
          borderTop: "1px solid rgba(255,255,255,0.07)",
        }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
            WHO CREATES IT
          </div>
          {/* Attainability says how OFTEN he shoots here. This says whether he
              can go get it or has to be found — the same number means very
              different things in those two cases. */}
          {explanation.creation.self_created_share != null
            && explanation.creation.league_self_created_share != null && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0 5px" }}>
              <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden", display: "flex" }}>
                <div style={{
                  width: `${explanation.creation.self_created_share * 100}%`,
                  background: "#C9A84C",
                }} />
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#94A3B8", whiteSpace: "nowrap" }}>
                {Math.round(explanation.creation.self_created_share * 100)}% SELF
                <span style={{ color: "#4a5568" }}>
                  {" / LG "}{Math.round(explanation.creation.league_self_created_share * 100)}%
                </span>
              </div>
            </div>
          )}
          <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 11, color: "#CBD5E1", lineHeight: 1.5 }}>
            {explanation.creation.note}
          </div>
        </div>
      )}

      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#4a5568", marginTop: 10, lineHeight: 1.5 }}>
        CONTRIBUTIONS ARE EXACT (TREESHAP) AND SUM TO THE ESTIMATE. PERCENTILES ARE LEAGUE-WIDE.
      </div>
    </div>
  );
}

// Attainability spans two orders of magnitude across the floor — a corner
// three can sit near 1% while the rim is over 30% — so a single fixed
// precision either rounds the small end into noise or clutters the large end.
function formatAttainability(v: number) {
  return `${(v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0)}%`;
}

function ordinal(p: number) {
  const n = Math.round(p);
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

/**
 * A tappable row that expands an explanation in place.
 *
 * The matchup panel is dense with paired numbers ("79% vs 47%", "8\" DEFENDER")
 * that are meaningless unless you already know which side is which and what
 * the pairing is measuring. Rather than shrink the data down or bolt on
 * tooltips that never appear on touch, each block can be opened for a
 * plain-English reading of the exact numbers on screen.
 */
function Disclosure({ label, children, tone = "#C9A84C", dense = false }: { label: React.ReactNode; children: React.ReactNode; tone?: string; dense?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          background: "transparent",
          border: "none",
          padding: dense ? 0 : "2px 0",
          cursor: "pointer",
          textAlign: "left",
          color: "inherit",
          font: "inherit",
        }}
      >
        {label}
        <span
          aria-hidden
          style={{
            marginLeft: "auto",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: tone,
            border: `1px solid ${tone}55`,
            borderRadius: 2,
            padding: "1px 5px",
            flexShrink: 0,
          }}
        >
          {open ? "HIDE" : "WHAT?"}
        </span>
      </button>
      {open && (
        <div
          style={{
            marginTop: 8,
            padding: "10px 12px",
            background: "#0d0d14",
            borderLeft: `2px solid ${tone}`,
            borderRadius: 2,
            fontFamily: "'Inter', sans-serif",
            fontSize: 12.5,
            lineHeight: 1.6,
            color: "#9aa7bd",
          }}
        >
          {children}
        </div>
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
      <div style={{ marginBottom: 12 }}>
        <Disclosure
          label={
            <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em" }}>MATCHUP EDGE</span>
              {data?.size_mismatch && (
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#DC2626", letterSpacing: "0.15em" }}>⚠ SIZE MISMATCH</span>
              )}
              {(data?.attacker.stats_source === "prior" || data?.defender.stats_source === "prior") && <ProjectedBadge compact />}
            </span>
          }
        >
          <p style={{ margin: 0 }}>
            Everything in this panel compares <strong style={{ color: "#C9A84C" }}>{attacker.name}</strong> (gold, left)
            against <strong style={{ color: "#DC2626" }}>{defender.name}</strong> (red, right).
          </p>
          <p style={{ margin: "8px 0 0" }}>
            <strong style={{ color: "#F0F0F0" }}>The bars</strong> are raw physicals. The label on the right names who
            holds the advantage and by how much — so <em>8&quot; DEFENDER</em> means the defender is eight inches taller.
          </p>
          <p style={{ margin: "8px 0 0" }}>
            <strong style={{ color: "#F0F0F0" }}>FG% allowed</strong> is how well opponents shoot with this defender on
            them, across the whole floor. <em>Lower is better for the defender</em>, so a negative &quot;vs league&quot;
            figure means he is tougher than the average defender, not worse.
          </p>
          <p style={{ margin: "8px 0 0" }}>
            <strong style={{ color: "#F0F0F0" }}>Exploit zones</strong> put the attacker&apos;s shooting in a zone next to
            what the defender allows there. A green row is where that gap is biggest — the mismatch to attack. Tap any
            row for its own numbers spelled out.
          </p>
          {data?.size_mismatch && (
            <p style={{ margin: "8px 0 0", color: "#DC2626" }}>
              <strong>Size mismatch</strong> flags a physical gap wide enough that the model expects it to change shot
              quality on its own.
            </p>
          )}
        </Disclosure>
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

          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.15em", marginTop: 16, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
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
            {data.exploit_zones.map((z) => {
              const pct = (v: number | null) => (v != null ? `${(v * 100).toFixed(0)}%` : "—");
              const adv = z.matchup_advantage;
              return (
                <div
                  key={z.zone}
                  style={{
                    padding: "9px 12px",
                    background: "#111118",
                    borderRadius: 3,
                    border: z.exploit ? "1px solid rgba(22,163,74,0.3)" : "1px solid rgba(255,255,255,0.05)",
                  }}
                >
                  <Disclosure
                    dense
                    tone={z.exploit ? "#16A34A" : "#64748b"}
                    label={
                      <span style={{ display: "flex", alignItems: "center", gap: 14, flex: 1, minWidth: 0 }}>
                        <span style={{ fontFamily: "'Inter', sans-serif", fontSize: 13, color: "#F0F0F0", flex: 1 }}>{z.zone}</span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b" }}>
                          {pct(z.attacker_fg_pct)} vs {pct(z.defender_fg_pct_allowed)}
                        </span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 700, color: z.exploit ? "#16A34A" : "#64748b", minWidth: 56, textAlign: "right" }}>
                          {adv != null ? `${adv > 0 ? "+" : ""}${(adv * 100).toFixed(1)}%` : "N/A"}
                        </span>
                      </span>
                    }
                  >
                    {z.attacker_fg_pct != null && z.defender_fg_pct_allowed != null && adv != null ? (
                      <>
                        <p style={{ margin: 0 }}>
                          <strong style={{ color: "#C9A84C" }}>{attacker.name}</strong> shoots{" "}
                          <strong style={{ color: "#F0F0F0" }}>{pct(z.attacker_fg_pct)}</strong> from the{" "}
                          {z.zone.toLowerCase()}.
                        </p>
                        <p style={{ margin: "6px 0 0" }}>
                          <strong style={{ color: "#DC2626" }}>{defender.name}</strong> lets opponents shoot{" "}
                          <strong style={{ color: "#F0F0F0" }}>{pct(z.defender_fg_pct_allowed)}</strong> there.
                        </p>
                        <p style={{ margin: "8px 0 0", color: adv > 0 ? "#16A34A" : "#DC2626" }}>
                          {adv > 0 ? (
                            <>
                              Gap of <strong>{(adv * 100).toFixed(1)} points</strong> in the attacker&apos;s favour
                              {z.exploit ? " — flagged as a zone worth attacking." : "."}
                            </>
                          ) : (
                            <>
                              Gap of <strong>{Math.abs(adv * 100).toFixed(1)} points</strong> against the attacker — the
                              defender is stronger here than {attacker.name.split(" ").slice(-1)[0]} is.
                            </>
                          )}
                        </p>
                        <p style={{ margin: "8px 0 0", fontSize: 11.5, color: "#64748b" }}>
                          Season averages, not a model output — it ignores game state and who else is on the floor. The
                          shot map above is the model&apos;s actual answer.
                        </p>
                      </>
                    ) : (
                      <p style={{ margin: 0 }}>
                        Not enough recorded shots in this zone for one or both players this season, so the comparison is
                        left blank rather than guessed at.
                      </p>
                    )}
                  </Disclosure>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}