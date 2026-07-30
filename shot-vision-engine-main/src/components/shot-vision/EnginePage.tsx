import { useState, useEffect, useRef } from "react";
import { ZONES, heightToFeet, type Player, type ZoneResult } from "@/lib/shot-vision-data";
import { searchPlayersAPI, getRecommendationHeatmapAPI } from "@/lib/api";

interface Props { onBack: () => void; }

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
  const [loadingIdx, setLoadingIdx] = useState(0);

  const canRun = attacker && primaryDef && (!doubleTeam || secondaryDef);

  const run = () => {
    if (!canRun || !attacker || !primaryDef) return;
    setLoading(true);
    setResults(null);
    setLoadingIdx(0);
    const iv = setInterval(() => setLoadingIdx((i) => (i + 1) % LOADING_LINES.length), 320);
    
    getRecommendationHeatmapAPI(attacker.id, primaryDef.id, quarter, minutes * 60 + seconds, scoreDiff, home ? 1 : 0)
      .then((res) => {
        clearInterval(iv);
        const backendZones = res.zone_summary || [];
        const mappedResults: ZoneResult[] = ZONES.map(z => {
          const bZone = backendZones.find((bz: any) => bz.zone === z.label);
          return {
            zone: z,
            ep: bZone ? bZone.avg_ep : z.baseEP,
            makeProb: bZone ? bZone.avg_make_prob : 0.35,
          };
        }).sort((a, b) => b.ep - a.ep);
        setResults(mappedResults);
        setLoading(false);
      })
      .catch((err) => {
        clearInterval(iv);
        console.error(err);
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
        <button onClick={onBack} style={{ background: "transparent", border: "none", color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", cursor: "pointer", transition: "color 200ms" }} onMouseEnter={(e) => (e.currentTarget.style.color = "#C9A84C")} onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}>
          ← BACK
        </button>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "52% 48%", gap: 32, padding: "32px", maxWidth: 1440, margin: "0 auto", alignItems: "start" }}>
        {/* LEFT COL */}
        <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
          <InputSection label="ATTACKER">
            <PlayerSelect selected={attacker} onSelect={setAttacker} accent="#C9A84C" statLabels={["FG%", "3P%", "RIM%"]} statValues={(p) => [p.fg, p.tp, p.rim]} ratingValue={(p) => p.offRtg} />
          </InputSection>

          <InputSection label="DEFENSE TYPE">
            <div style={{ display: "flex", gap: 8 }}>
              <Pill active={!doubleTeam} onClick={() => { setDoubleTeam(false); setSecondaryDef(null); }}>SINGLE COVERAGE</Pill>
              <Pill active={doubleTeam} onClick={() => setDoubleTeam(true)}>DOUBLE TEAM</Pill>
            </div>
          </InputSection>

          <InputSection label="PRIMARY DEFENDER">
            <PlayerSelect selected={primaryDef} onSelect={setPrimaryDef} accent="#DC2626" statLabels={["OPP FG%", "DFG DIFF%", "WINGSPAN"]} statValues={(p) => [`${p.defRtg}%`, p.contest, `${p.wingspanIn}"`]} ratingValue={(p) => p.defRtgOvr} />
          </InputSection>

          <div style={{ maxHeight: doubleTeam ? 400 : 0, opacity: doubleTeam ? 1 : 0, overflow: doubleTeam ? "visible" : "hidden", transition: "all 400ms ease" }}>
            <InputSection label="SECONDARY DEFENDER">
              <PlayerSelect selected={secondaryDef} onSelect={setSecondaryDef} accent="rgba(220,38,38,0.5)" statLabels={["OPP FG%", "DFG DIFF%", "WINGSPAN"]} statValues={(p) => [`${p.defRtg}%`, p.contest, `${p.wingspanIn}"`]} ratingValue={(p) => p.defRtgOvr} />
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
                  <Pill active={home} onClick={() => setHome(true)} activeColor="#16A34A">HOME</Pill>
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
        </div>

        {/* RIGHT COL */}
        <div style={{ position: "sticky", top: 88, background: "#0e0e16", border: "1px solid rgba(201,168,76,0.08)", borderRadius: 6, padding: 28, minHeight: 640 }}>
          {!loading && !results && <DefaultOutput />}
          {loading && <LoadingOutput idx={loadingIdx} />}
          {!loading && results && attacker && primaryDef && <ResultsOutput results={results} attacker={attacker} defender={primaryDef} />}
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

function PlayerSelect({ selected, onSelect, accent, statLabels, statValues, ratingValue }: {
  selected: Player | null;
  onSelect: (p: Player | null) => void;
  accent: string;
  statLabels: [string, string, string];
  statValues: (p: Player) => [number | string, number | string, number | string];
  ratingValue: (p: Player) => number;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [filtered, setFiltered] = useState<Player[]>([]);

  useEffect(() => {
    if (query.length < 2) {
      setFiltered([]);
      return;
    }
    const timer = setTimeout(() => {
      searchPlayersAPI(query).then(res => setFiltered(res.slice(0, 8))).catch(console.error);
    }, 250);
    return () => clearTimeout(timer);
  }, [query]);

  if (selected) {
    const vals = statValues(selected);
    return (
      <div style={{ background: "#16161f", borderLeft: `3px solid ${accent}`, padding: "14px 18px", borderRadius: 3 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ background: accent, color: "#000", padding: "6px 10px", borderRadius: 2, fontFamily: "'Bebas Neue', sans-serif", fontSize: 20, letterSpacing: "0.02em", minWidth: 46, textAlign: "center" }}>
            {ratingValue(selected)}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#F0F0F0", letterSpacing: "0.02em", lineHeight: 1 }}>{selected.name}</div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b", marginTop: 4 }}>
              {selected.pos} · {heightToFeet(selected.heightIn)} · {selected.weightLbs}lb
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
      {open && filtered.length > 0 && (
        <div style={{ position: "absolute", top: "100%", left: 0, right: 0, marginTop: 4, background: "#16161f", border: "1px solid rgba(201,168,76,0.15)", borderRadius: 3, maxHeight: 280, overflowY: "auto", zIndex: 20 }}>
          {filtered.map((p) => (
            <div key={p.id} onMouseDown={() => { onSelect(p); setQuery(""); setOpen(false); }} style={{ padding: "10px 16px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.03)", transition: "background 150ms" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(201,168,76,0.06)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div>
                <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 14, color: "#F0F0F0" }}>{p.name}</div>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b" }}>{p.pos} · {heightToFeet(p.heightIn)}</div>
              </div>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 18, color: "#C9A84C" }}>{ratingValue(p)}</div>
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

function ResultsOutput({ results, attacker, defender }: { results: ZoneResult[]; attacker: Player; defender: Player }) {
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

  return (
    <div style={{ animation: "fadeUp 0.6s ease both", display: "flex", flexDirection: "column", gap: 20 }}>
      <CourtCanvas bestKey={best.zone.key} results={results} showProb={showProb} />

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
            </div>
          );
        })}
      </div>

      <div style={{ background: "#0e0e16", borderLeft: "3px solid #DC2626", padding: "12px 16px" }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#DC2626", letterSpacing: "0.15em" }}>
          ⚠ AVOID · {worst.zone.label.toUpperCase()} · {showProb ? `${Math.round(worst.makeProb * 100)}% PROB` : `${worst.ep.toFixed(2)} EP`}
        </div>
      </div>

      <MatchupEdge attacker={attacker} defender={defender} />
    </div>
  );
}

function CourtCanvas({ bestKey, results, showProb }: { bestKey: string; results: ZoneResult[]; showProb: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Canvas size
    const cw = 400;
    const ch = 375;
    
    // STEP 1 - HARDWOOD FLOOR BASE
    ctx.fillStyle = "#2C1A0A";
    ctx.fillRect(0, 0, cw, ch);
    
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
    
    // STEP 2 - HEATMAP GRADIENT BLOBS
    ctx.globalCompositeOperation = "screen";
    
    const getZoneEP = (key: string) => {
      const z = results.find(r => r.zone.key === key);
      return z ? z.ep : 0;
    };

    const getBaseColor = (ep: number) => {
      // Map EP to a value between 0 and 1 (assume min EP ~0.6, max ~1.6)
      const t = Math.max(0, Math.min(1, (ep - 0.6) / 1.0));
      let r, g, b;
      if (t < 0.5) {
        // Blue (0,150,255) to Green (0,255,100)
        const pct = t * 2;
        r = 0;
        g = Math.round(150 + 105 * pct);
        b = Math.round(255 - 155 * pct);
      } else {
        // Green (0,255,100) to Red (255,50,50)
        const pct = (t - 0.5) * 2;
        r = Math.round(255 * pct);
        g = Math.round(255 - 205 * pct);
        b = Math.round(100 - 50 * pct);
      }
      return { r, g, b };
    };
    
    const drawBlob = (cx: number, cy: number, radius: number, ep: number) => {
      if (ep <= 0) return;
      const { r, g, b } = getBaseColor(ep);
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      
      const opacity = Math.max(0.3, Math.min(0.85, ep / 1.8));

      grad.addColorStop(0.0, `rgba(${r}, ${g}, ${b}, ${opacity})`);
      grad.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${opacity * 0.6})`);
      grad.addColorStop(1.0, `rgba(${r}, ${g}, ${b}, 0)`);

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    };

    // BLOB 1: Restricted Area
    drawBlob(200, 318, 55, getZoneEP("restrictedArea"));

    // BLOB 2: Paint Non-RA
    drawBlob(200, 240, 90, getZoneEP("paintNonRA"));

    // BLOB 3 & 4: Corners
    drawBlob(30, 320, 70, getZoneEP("leftCorner3"));
    drawBlob(370, 320, 70, getZoneEP("rightCorner3"));

    // BLOB 5, 6, 7: Above Break 3 (Left, Right, Top)
    const ab3EP = getZoneEP("aboveBreak3");
    drawBlob(80, 180, 85, ab3EP);
    drawBlob(320, 180, 85, ab3EP);
    drawBlob(200, 120, 80, ab3EP);

    // BLOB 8 & 9: Mid Range (Left, Right)
    const midEP = getZoneEP("midRange");
    drawBlob(100, 260, 70, midEP);
    drawBlob(300, 260, 70, midEP);
    
    // STEP 3 - COURT LINES
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "rgba(255, 240, 200, 0.7)";
    ctx.lineWidth = 2;
    
    // THREE POINT ARC
    ctx.beginPath();
    ctx.moveTo(20, ch);
    ctx.lineTo(20, ch * 0.52);
    ctx.arc(cw / 2, ch * 0.88, cw * 0.42, Math.PI, 0, false);
    ctx.lineTo(cw - 20, ch * 0.52);
    ctx.lineTo(cw - 20, ch);
    ctx.stroke();
    
    // PAINT RECTANGLE
    ctx.strokeRect(cw * 0.33, ch * 0.35, cw * 0.34, ch * 0.58);
    
    // FREE THROW CIRCLE top half (solid)
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(cw / 2, ch * 0.35, cw * 0.17, Math.PI, 0, false);
    ctx.stroke();
    
    // FREE THROW CIRCLE bottom half (dashed)
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.arc(cw / 2, ch * 0.35, cw * 0.17, 0, Math.PI, false);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // RESTRICTED AREA
    ctx.beginPath();
    ctx.arc(cw / 2, ch * 0.88, cw * 0.08, Math.PI, 0, false);
    ctx.stroke();
    
    // LANE LINES inside paint (subtle)
    ctx.strokeStyle = "rgba(255,240,200,0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cw * 0.38, ch * 0.35); ctx.lineTo(cw * 0.38, ch);
    ctx.moveTo(cw * 0.62, ch * 0.35); ctx.lineTo(cw * 0.62, ch);
    ctx.stroke();
    
    // BACKBOARD
    ctx.strokeStyle = "rgba(255,240,200,0.8)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cw * 0.42, ch * 0.91);
    ctx.lineTo(cw * 0.58, ch * 0.91);
    ctx.stroke();
    
    // HOOP
    ctx.strokeStyle = "#C9A84C";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(cw / 2, ch * 0.88, cw * 0.04, 0, Math.PI * 2);
    ctx.stroke();
    
    // CENTER COURT LOGO area
    ctx.strokeStyle = "rgba(255,240,200,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cw / 2, ch * 0.15, cw * 0.15, 0, Math.PI * 2);
    ctx.stroke();
    
    // OUTER BOUNDARY
    ctx.strokeStyle = "rgba(255,240,200,0.5)";
    ctx.lineWidth = 2;
    ctx.strokeRect(8, 8, cw - 16, ch - 16);
    
    // STEP 4 - FLOOR VARNISH EFFECT
    const varnishGrad = ctx.createRadialGradient(cw * 0.5, ch * 0.5, 0, cw * 0.5, ch * 0.5, cw * 0.7);
    varnishGrad.addColorStop(0.0, "rgba(255,200,100,0.04)");
    varnishGrad.addColorStop(0.5, "rgba(255,150,50,0.02)");
    varnishGrad.addColorStop(1.0, "rgba(0,0,0,0)");
    ctx.fillStyle = varnishGrad;
    ctx.globalCompositeOperation = "overlay";
    ctx.fillRect(0, 0, cw, ch);
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
    
    const getVal = (key: string) => {
      const z = results.find(r => r.zone.key === key);
      return z ? (showProb ? `${Math.round(z.makeProb * 100)}%` : z.ep.toFixed(2)) : "";
    };
    
    ctx.fillText(getVal("restrictedArea"), 200, 318);
    ctx.fillText(getVal("paintNonRA"), 200, 230);
    ctx.fillText(getVal("leftCorner3"), 32, 310);
    ctx.fillText(getVal("rightCorner3"), 368, 310);
    ctx.fillText(getVal("aboveBreak3"), 55, 175);
    ctx.fillText(getVal("aboveBreak3"), 345, 175);
    ctx.fillText(getVal("aboveBreak3"), 200, 85);
    ctx.fillText(getVal("midRange"), 90, 260);
    ctx.fillText(getVal("midRange"), 310, 260);
    
    ctx.shadowColor = "transparent";
    
    // STEP 6 - LEGEND
    const legGrad = ctx.createLinearGradient(100, 345, 300, 345);
    legGrad.addColorStop(0, "rgb(0, 150, 255)");   // Blue
    legGrad.addColorStop(0.5, "rgb(0, 255, 100)"); // Green
    legGrad.addColorStop(1, "rgb(255, 50, 50)");   // Red
    ctx.fillStyle = legGrad;
    ctx.fillRect(100, 345, 200, 6);
    
    ctx.font = "9px JetBrains Mono";
    ctx.fillStyle = "#64748b";
    ctx.fillText("LOW EP", 100, 362);
    ctx.fillText("HIGH EP", 300, 362);

  }, [results, showProb]);

  // STEP 7 - BEST ZONE PULSE RING
  const ringCoords: Record<string, {x: number, y: number}> = {
    restrictedArea: {x: 200, y: 318},
    paintNonRA: {x: 200, y: 230},
    leftCorner3: {x: 32, y: 310},
    rightCorner3: {x: 368, y: 310},
    aboveBreak3: {x: 200, y: 85}, 
    midRange: {x: 310, y: 260},
  };
  const ring = ringCoords[bestKey];

  return (
    <div style={{ position: "relative", width: "100%", borderRadius: 3, overflow: "hidden", border: "1px solid rgba(255,255,255,0.05)" }}>
      <style>{`
        @keyframes pulseRing {
          0%, 100% { opacity: 1; transform: translate(-50%, -50%) scale(1); box-shadow: 0 0 8px #C9A84C; }
          50% { opacity: 0; transform: translate(-50%, -50%) scale(1.6); box-shadow: 0 0 24px #C9A84C; }
        }
      `}</style>
      <canvas ref={canvasRef} width={400} height={375} style={{ width: "100%", height: "auto", display: "block" }} />
      {ring && (
        <div style={{
          position: "absolute",
          left: `${(ring.x / 400) * 100}%`,
          top: `${(ring.y / 375) * 100}%`,
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

function MatchupEdge({ attacker, defender }: { attacker: Player; defender: Player }) {
  const rows = [
    { label: "HEIGHT", a: attacker.heightIn, d: defender.heightIn, fmt: (n: number) => heightToFeet(n) },
    { label: "WINGSPAN", a: attacker.wingspanIn, d: defender.wingspanIn, fmt: (n: number) => `${n}"` },
    { label: "REACH", a: attacker.heightIn + attacker.wingspanIn / 2, d: defender.heightIn + defender.wingspanIn / 2, fmt: (n: number) => `${n.toFixed(1)}"` },
  ];
  const max = Math.max(...rows.flatMap((r) => [r.a, r.d]));
  return (
    <div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em", marginBottom: 12 }}>MATCHUP EDGE</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {rows.map((r) => {
          const adv = r.a - r.d;
          const advLabel = adv > 0 ? `+${r.fmt(adv)} ATTACKER` : adv < 0 ? `${r.fmt(Math.abs(adv))} DEFENDER` : "EVEN";
          return (
            <div key={r.label}>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.2em" }}>
                <span>{r.label}</span>
                <span style={{ color: adv > 0 ? "#C9A84C" : adv < 0 ? "#DC2626" : "#64748b" }}>{advLabel}</span>
              </div>
              <div style={{ display: "flex", gap: 6, marginTop: 6, alignItems: "center" }}>
                <div style={{ flex: r.a / max, height: 6, background: "#C9A84C", borderRadius: 2 }} />
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", minWidth: 44, textAlign: "center" }}>{r.fmt(r.a)}</span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#DC2626", minWidth: 44, textAlign: "center" }}>{r.fmt(r.d)}</span>
                <div style={{ flex: r.d / max, height: 6, background: "#DC2626", borderRadius: 2 }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}