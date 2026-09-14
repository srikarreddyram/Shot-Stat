import { useEffect, useState } from "react";
import type { HeatmapPoint, MatchupResponse, Player, ZoneResult } from "@/lib/shot-vision-data";
import { getMatchupAPI } from "@/lib/api";
import { CourtCanvas } from "./court/CourtCanvas";
import { MatchupEdge } from "./MatchupEdge";
import { ProjectedBadge } from "./badges";

const LOADING_LINES = [
  "Loading player profiles...",
  "Computing zone tendencies...",
  "Running XGBoost inference...",
  "Ranking by expected points...",
];

export function ErrorOutput({ message, onRetry }: { message: string; onRetry: () => void }) {
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

export function DefaultOutput() {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 580, textAlign: "center" }}>
      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 120, color: "rgba(201,168,76,0.06)", lineHeight: 1 }}>?</div>
      <div style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: 18, color: "#2a2a3a", marginTop: 8 }}>Configure the matchup</div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#1e1e2e", marginTop: 4, letterSpacing: "0.15em" }}>then run the engine.</div>
    </div>
  );
}

export function LoadingOutput({ idx }: { idx: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 580, textAlign: "center", gap: 16 }}>
      {/* Scoped locally rather than assumed global: this component and
          SplashPage.tsx are never mounted at the same time, so a keyframe
          declared only in SplashPage's own <style> block (as this one used
          to be) silently never existed here — the dots below animated
          nowhere for the entire lifetime of this page. */}
      <style>{`@keyframes goldPulse { 0%,100% { box-shadow: 0 0 20px rgba(201,168,76,0.2); } 50% { box-shadow: 0 0 48px rgba(201,168,76,0.5); } }`}</style>
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

export function ResultsOutput({ results, heatmapPoints, attacker, defender, secondaryDefender, season, projected }: { results: ZoneResult[]; heatmapPoints: HeatmapPoint[]; attacker: Player; defender: Player; secondaryDefender?: Player | null; season: string; projected: { attacker: boolean; defender: boolean } }) {
  const [showProb, setShowProb] = useState(false);
  const best = results[0];
  const worst = results[results.length - 1];

  // Lifted out of CourtCanvas so the ranked zone cards below can open the
  // exact same shot-detail panel a court click does — both are just two
  // different ways of picking an index into `heatmapPoints`.
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => { setSelected(null); }, [heatmapPoints]);

  // The single best-scoring point within a zone — the ranked card's "#1
  // Restricted Area 1.19" is a zone-level aggregate, not any one shot, so
  // opening it needs a specific representative location to show.
  const selectZone = (zoneLabel: string) => {
    let bestIdx = -1;
    let bestEp = -Infinity;
    heatmapPoints.forEach((pt, i) => {
      if (pt.zone === zoneLabel && pt.expected_points > bestEp) {
        bestEp = pt.expected_points;
        bestIdx = i;
      }
    });
    if (bestIdx >= 0) setSelected(bestIdx);
  };

  // Season-average attacker-vs-defender numbers per zone (exploit_zones) —
  // fetched once here and shared by both MatchupEdge below and the ranked
  // zone cards above it, which previously showed a bare EP number with
  // nothing about WHY that zone ranked where it did. This used to be
  // MatchupEdge's own private fetch; lifting it up avoids a second request
  // for numbers the ranked cards need too.
  const [matchupData, setMatchupData] = useState<MatchupResponse | null>(null);
  const [matchupLoading, setMatchupLoading] = useState(true);
  const [matchupError, setMatchupError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setMatchupLoading(true);
    setMatchupError(null);
    getMatchupAPI(attacker.id, defender.id, season)
      .then((res) => { if (!cancelled) setMatchupData(res); })
      .catch((err) => { if (!cancelled) setMatchupError(err instanceof Error ? err.message : "Failed to load matchup data."); })
      .finally(() => { if (!cancelled) setMatchupLoading(false); });
    return () => { cancelled = true; };
  }, [attacker.id, defender.id, season]);

  const exploitZoneFor = (zoneLabel: string) =>
    matchupData?.exploit_zones.find((z) => z.zone === zoneLabel) ?? null;

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
      <CourtCanvas bestKey={best.zone.key} results={results} heatmapPoints={heatmapPoints} showProb={showProb} attacker={attacker} defender={defender} season={season} selected={selected} setSelected={setSelected} />

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
            <div
              key={r.zone.key}
              role="button"
              tabIndex={0}
              onClick={() => selectZone(r.zone.label)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectZone(r.zone.label); } }}
              aria-label={`Inspect the best ${r.zone.label} shot`}
              style={{
                background: "#111118",
                border: isTop ? "1px solid rgba(201,168,76,0.25)" : "1px solid rgba(255,255,255,0.05)",
                padding: "12px 16px", borderRadius: 3,
                boxShadow: isTop ? "0 0 32px rgba(201,168,76,0.08)" : "none",
                cursor: "pointer", transition: "opacity 160ms ease",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.opacity = "0.8"; }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = "1"; }}
            >
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
              {/* The attacker-vs-defender season-average behind WHY this zone
                  ranked where it did — the same numbers MATCHUP EDGE's
                  exploit-zones panel shows, not a second, differently-built
                  comparison. A bare EP number said nothing about whether the
                  rank came from the attacker being great here or the
                  defender being exploitable here; this does. */}
              {(() => {
                const ez = exploitZoneFor(r.zone.label);
                if (!ez || ez.attacker_fg_pct == null || ez.defender_fg_pct_allowed == null) return null;
                const gap = ez.matchup_advantage ?? (ez.attacker_fg_pct - ez.defender_fg_pct_allowed);
                const favorsAttacker = gap >= 0;
                return (
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#94A3B8", marginTop: 6 }}>
                    {Math.round(ez.attacker_fg_pct * 100)}% vs {Math.round(ez.defender_fg_pct_allowed * 100)}% allowed
                    <span style={{ color: favorsAttacker ? "#16A34A" : "#DC2626", marginLeft: 6 }}>
                      {favorsAttacker ? "+" : ""}{(gap * 100).toFixed(1)}pp {favorsAttacker ? "attacker" : "defender"}
                    </span>
                  </div>
                );
              })()}
              {/* Evidence behind the estimate. Rendered only when there is
                  some — an empty caption row is quieter than a placeholder
                  telling the reader what the app does not know. */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 6 }}>
                {r.attemptsBehind && r.attemptsBehind > 0 ? (
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em" }}>
                    {Math.round(r.attemptsBehind).toLocaleString()} SIMILAR SHOTS BEHIND THIS
                  </div>
                ) : <span />}
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#C9A84C", letterSpacing: "0.1em" }}>
                  TAP TO INSPECT ▸
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ background: "#0e0e16", borderLeft: "3px solid #DC2626", padding: "12px 16px" }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#DC2626", letterSpacing: "0.15em" }}>
          ⚠ AVOID · {worst.zone.label.toUpperCase()}{worst.shotDistance != null ? ` · ${worst.shotDistance.toFixed(1)} FT` : ""} · {showProb ? `${Math.round(worst.makeProb * 100)}% PROB` : `${worst.ep.toFixed(2)} EP`}
        </div>
      </div>

      <MatchupEdge attacker={attacker} defender={defender} data={matchupData} loading={matchupLoading} error={matchupError} />
    </div>
  );
}
