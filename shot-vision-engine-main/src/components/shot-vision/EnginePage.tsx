import { useState, useEffect, useRef } from "react";
import { ZONES, type Player, type ZoneResult, type HeatmapPoint, type Team } from "@/lib/shot-vision-data";
import { getRecommendationHeatmapAPI, checkHealthAPI, getTeamsAPI, getTeamRosterAPI } from "@/lib/api";
// Reusing the Stat Engine's team-identity building blocks rather than
// re-deriving colours/logos here: one TEAM_COLORS map, one logo endpoint,
// used by both halves of the app.
import { teamColor, NBA_RED, NBA_BLUE, NBAEdgeBezel, NBADefaultWash, LeagueBrandMark, NBAHeaderStripe } from "@/components/stat-engine/ui";

// This file is the orchestrator only: state, effects, the run() call, and
// the page layout. Every visual piece below has its own file under
// ./engine/ — the court canvas, the two pickers, the results panel, the
// matchup/attainability "why" breakdowns — so a change to any one of them
// doesn't require reading the other ~2,500 lines this file used to be.
import { HealthIndicator } from "./engine/HealthIndicator";
import { InputSection, MiniLabel, Pill } from "./engine/formControls";
import { TeamSelect } from "./engine/TeamSelect";
import { PlayerSelect } from "./engine/PlayerSelect";
import { DefaultOutput, ErrorOutput, LoadingOutput, ResultsOutput } from "./engine/ResultsOutput";

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

  // Team-vs-team mode: pick a real offense and defense roster first, then
  // any attacker/defender pair from within them — "Luka vs Curry", then
  // just swap the defender to "Butler vs Luka" without re-searching the
  // league. SEARCH mode (the original flow) stays the default.
  const [matchupSource, setMatchupSource] = useState<"search" | "team">("search");
  const [allTeams, setAllTeams] = useState<Team[]>([]);
  const [offenseTeam, setOffenseTeam] = useState<Team | null>(null);
  const [defenseTeam, setDefenseTeam] = useState<Team | null>(null);
  const [offenseRoster, setOffenseRoster] = useState<Player[]>([]);
  const [defenseRoster, setDefenseRoster] = useState<Player[]>([]);

  useEffect(() => {
    getTeamsAPI().then(setAllTeams).catch(console.error);
  }, []);

  // A new team (or a season that just resolved) invalidates whichever
  // roster it feeds — the roster fetch below reloads it. Clearing the
  // already-picked player here (rather than letting a stale one linger)
  // means the picker never shows someone alongside the wrong team's roster.
  useEffect(() => {
    if (!offenseTeam || !season) { setOffenseRoster([]); return; }
    setAttacker(null);
    getTeamRosterAPI(offenseTeam.teamId, season).then(setOffenseRoster).catch(console.error);
  }, [offenseTeam, season]);

  useEffect(() => {
    if (!defenseTeam || !season) { setDefenseRoster([]); return; }
    setPrimaryDef(null);
    setSecondaryDef(null);
    getTeamRosterAPI(defenseTeam.teamId, season).then(setDefenseRoster).catch(console.error);
  }, [defenseTeam, season]);

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

  // Every run gets a generation number. There is no real HTTP cancellation
  // here (getRecommendationHeatmapAPI doesn't take an AbortSignal), so a
  // "killed" request still finishes on the wire — this just makes sure its
  // answer is thrown away instead of clobbering whatever the user has
  // changed the inputs to in the meantime. `killRun` is what the input
  // guard below calls to invalidate the in-flight run.
  const runIdRef = useRef(0);
  const loadingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const killRun = () => {
    if (loadingIntervalRef.current != null) clearInterval(loadingIntervalRef.current);
    runIdRef.current += 1;
    setLoading(false);
  };

  const run = () => {
    if (!canRun || !attacker || !primaryDef || !season) return;
    const myRunId = ++runIdRef.current;
    setLoading(true);
    setResults(null);
    setError(null);
    setLoadingIdx(0);
    loadingIntervalRef.current = setInterval(() => setLoadingIdx((i) => (i + 1) % LOADING_LINES.length), 320);

    getRecommendationHeatmapAPI(attacker.id, primaryDef.id, season, quarter, minutes * 60 + seconds, scoreDiff, home ? 1 : 0, doubleTeam ? secondaryDef?.id : null)
      .then((res) => {
        if (myRunId !== runIdRef.current) return; // superseded or killed — discard
        clearInterval(loadingIntervalRef.current!);
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
        if (myRunId !== runIdRef.current) return; // superseded or killed — discard
        clearInterval(loadingIntervalRef.current!);
        console.error(err);
        setError(err instanceof Error ? err.message : "Something went wrong reaching the shot engine.");
        setLoading(false);
      });
  };

  // Once a real team is picked in TEAM VS TEAM mode, the attacker/defender
  // pickers below borrow that team's own colour instead of the generic
  // gold/red — so picking the Lakers doesn't just fill a "gold slot", the
  // whole offense side of the form reads as Lakers. SEARCH mode has no team
  // to borrow a colour from, so it keeps the original scheme.
  const offenseAccent = matchupSource === "team" && offenseTeam ? teamColor(offenseTeam.abbreviation) : "#C9A84C";
  const defenseAccent = matchupSource === "team" && defenseTeam ? teamColor(defenseTeam.abbreviation) : "#DC2626";
  // The secondary defender keeps a visually distinct, faded version of the
  // same team colour rather than an identical block next to the primary
  // defender's — otherwise a double-team on one team reads as one repeated
  // swatch instead of two distinguishable slots.
  const defenseAccentSecondary = matchupSource === "team" && defenseTeam ? `${teamColor(defenseTeam.abbreviation)}80` : "rgba(220,38,38,0.5)";

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f", color: "#F0F0F0", fontFamily: "'Inter', sans-serif" }}>
      {/* Ambient page tint for TEAM VS TEAM mode: offense colour bleeding in
          from the top-left, defense from the top-right, so the theming isn't
          confined to the two picker boxes — the whole page reads as "this
          matchup" once both sides are picked. Fixed + z-index 0 so it sits
          behind every real control and never intercepts a click. Absent
          entirely in SEARCH mode or before a team is chosen — there's no
          real colour to wash the page in yet. */}
      {matchupSource === "team" && (offenseTeam || defenseTeam) ? (
        <div aria-hidden="true" style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
          {offenseTeam && (
            <div style={{ position: "absolute", inset: 0, background: `radial-gradient(60% 45% at 15% 0%, ${teamColor(offenseTeam.abbreviation)}38 0%, transparent 65%)` }} />
          )}
          {defenseTeam && (
            <div style={{ position: "absolute", inset: 0, background: `radial-gradient(60% 45% at 85% 0%, ${teamColor(defenseTeam.abbreviation)}38 0%, transparent 65%)` }} />
          )}
        </div>
      ) : (
        // No specific team picked (yet) to theme the page with — falls back
        // to the league's own red/blue (shared with every other page) rather
        // than leaving the page pure gold-on-black. Swapped out the instant
        // a real team is chosen above.
        <NBADefaultWash />
      )}
      {/* A persistent red/blue bezel down both edges of the viewport — the
          "broadcast frame" look of NBA League Pass / NBA TV chrome, visible
          no matter how far down the form you've scrolled, not just at the
          very top of the page. Shared with every other page in the app —
          see NBAEdgeBezel in stat-engine/ui.tsx. */}
      <NBAEdgeBezel />
      <header style={{ position: "sticky", top: 0, zIndex: 50, background: "rgba(10,10,15,0.92)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", borderBottom: "1px solid rgba(201,168,76,0.1)", padding: "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        {/* A full-strength red/white/blue stripe under the header — the one
            place on this page that borrows the league's wordmark colours
            directly rather than mixing them into a gradient, so the "NBA"
            identity reads at a glance without competing with the gold
            chrome above it. */}
        <NBAHeaderStripe bottom={-5} />
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <LeagueBrandMark height={40} />
          <div style={{ width: 1, height: 32, background: "rgba(255,255,255,0.12)" }} />
          <div>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#F0F0F0", letterSpacing: "0.05em" }}>SHOT VISION</div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#C9A84C", letterSpacing: "0.4em", marginTop: 2 }}>SHOT QUALITY ENGINE</div>
          </div>
        </div>
        {/* Status and nav sit together on the right. Under space-between the
            status pill landed dead-centre with nothing on its axis, reading as
            a stray element rather than as chrome. */}
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <HealthIndicator status={health} />
          {/* Plain <a>, not a router Link: /stats, /archetypes, and
              /model-features are separate top-level routes, and this
              component is rendered inside the "/" route's own splash/engine
              toggle rather than being a route itself. A full navigation is
              the honest thing here and costs nothing at this size. */}
          <a href="/stats" style={{ color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", textDecoration: "none", transition: "color 200ms" }} onMouseEnter={(e) => (e.currentTarget.style.color = "#C9A84C")} onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}>
            STAT ENGINE
          </a>
          <a href="/archetypes" style={{ color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", textDecoration: "none", transition: "color 200ms" }} onMouseEnter={(e) => (e.currentTarget.style.color = "#C9A84C")} onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}>
            ARCHETYPES
          </a>
          <a href="/model-features" style={{ color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", textDecoration: "none", transition: "color 200ms" }} onMouseEnter={(e) => (e.currentTarget.style.color = "#C9A84C")} onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}>
            MODEL FEATURES
          </a>
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
        {/* Capture-phase, not bubble: this needs to fire and kill the
            in-flight run BEFORE the control being interacted with applies
            its own change, not after — a bubble-phase handler would see the
            new value already committed. Covers clicks (pills, dropdown
            options, TeamSelect/PlayerSelect rows — several of those use
            onMouseDown rather than onClick, hence both) and onChange
            (the quarter time inputs, the score-diff slider). Killing on a
            click that lands on inert space inside this column is harmless
            — it only stops a request that was about to be thrown away by
            the input change anyway. */}
        <div
          style={{ display: "flex", flexDirection: "column", gap: 28 }}
          onClickCapture={() => { if (loading) killRun(); }}
          onMouseDownCapture={() => { if (loading) killRun(); }}
          onChangeCapture={() => { if (loading) killRun(); }}
        >
          <InputSection label="MATCHUP SOURCE">
            <div style={{ display: "flex", gap: 8 }}>
              <Pill active={matchupSource === "search"} onClick={() => setMatchupSource("search")}>SEARCH</Pill>
              <Pill active={matchupSource === "team"} onClick={() => setMatchupSource("team")}>TEAM VS TEAM</Pill>
            </div>
          </InputSection>

          {matchupSource === "team" && (
            <>
              <InputSection label="OFFENSE TEAM">
                <TeamSelect
                  teams={allTeams}
                  selected={offenseTeam}
                  onSelect={setOffenseTeam}
                  excludeTeamId={defenseTeam?.teamId}
                  accent="#C9A84C"
                />
              </InputSection>
              <InputSection label="DEFENSE TEAM">
                <TeamSelect
                  teams={allTeams}
                  selected={defenseTeam}
                  onSelect={setDefenseTeam}
                  excludeTeamId={offenseTeam?.teamId}
                  accent="#DC2626"
                />
              </InputSection>

              {(offenseTeam || defenseTeam) && (
                <button
                  onClick={() => { setOffenseTeam(defenseTeam); setDefenseTeam(offenseTeam); }}
                  style={{
                    background: "transparent",
                    border: "1px solid rgba(255,255,255,0.14)",
                    color: "#9aa7bd",
                    borderRadius: 3,
                    padding: "10px 16px",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11,
                    letterSpacing: "0.15em",
                    cursor: "pointer",
                    alignSelf: "flex-start",
                    transition: "color 200ms, border-color 200ms",
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = "#C9A84C"; e.currentTarget.style.borderColor = "rgba(201,168,76,0.4)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = "#9aa7bd"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.14)"; }}
                >
                  ⇅ SWAP OFFENSE / DEFENSE
                </button>
              )}
            </>
          )}

          <InputSection label="ATTACKER">
            <PlayerSelect selected={attacker} onSelect={setAttacker} season={season} accent={offenseAccent} statLabels={["FG%", "3P%", "RIM%"]} statValues={(p) => [p.fg, p.tp, p.rim]} ratingValue={(p) => p.offRtg} excludeIds={[primaryDef?.id, secondaryDef?.id]} excludeTeamId={primaryDef?.teamId} roster={matchupSource === "team" ? offenseRoster : undefined} />
          </InputSection>

          <InputSection label="DEFENSE TYPE">
            <div style={{ display: "flex", gap: 8 }}>
              <Pill active={!doubleTeam} onClick={() => { setDoubleTeam(false); setSecondaryDef(null); }}>SINGLE COVERAGE</Pill>
              <Pill active={doubleTeam} onClick={() => setDoubleTeam(true)}>DOUBLE TEAM</Pill>
            </div>
          </InputSection>

          <InputSection label="PRIMARY DEFENDER">
            <PlayerSelect selected={primaryDef} onSelect={setPrimaryDef} season={season} accent={defenseAccent} statLabels={["OPP FG%", "DFG DIFF%", "WINGSPAN"]} statValues={(p) => [`${p.defRtg}%`, p.contest, p.wingspanIn != null ? `${p.wingspanIn}"` : "—"]} ratingValue={(p) => p.defRtgOvr} excludeIds={[attacker?.id, secondaryDef?.id]} excludeTeamId={attacker?.teamId} roster={matchupSource === "team" ? defenseRoster : undefined} />
          </InputSection>

          <div style={{ maxHeight: doubleTeam ? 400 : 0, opacity: doubleTeam ? 1 : 0, overflow: doubleTeam ? "visible" : "hidden", transition: "all 400ms ease" }}>
            <InputSection label="SECONDARY DEFENDER">
              <PlayerSelect selected={secondaryDef} onSelect={setSecondaryDef} season={season} accent={defenseAccentSecondary} statLabels={["OPP FG%", "DFG DIFF%", "WINGSPAN"]} statValues={(p) => [`${p.defRtg}%`, p.contest, p.wingspanIn != null ? `${p.wingspanIn}"` : "—"]} ratingValue={(p) => p.defRtgOvr} excludeIds={[attacker?.id, primaryDef?.id]} excludeTeamId={attacker?.teamId} roster={matchupSource === "team" ? defenseRoster : undefined} />
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
                  {/* AWAY gets the league blue rather than the default gold —
                      an actual broadcast convention (the road team wears its
                      colour, not white) rather than a decorative choice. */}
                  <Pill active={!home} onClick={() => setHome(false)} activeColor={NBA_BLUE} activeTextColor="#F0F0F0">AWAY</Pill>
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
              // Gold stays the dominant glow (it's the primary CTA colour);
              // a faint red/blue halo rides just outside it — visible as a
              // hint of colour at the button's edge without competing with
              // the gold for attention.
              boxShadow: canRun && !loading ? `0 0 32px rgba(201,168,76,0.2), 0 0 60px ${NBA_RED}33, 0 0 60px ${NBA_BLUE}3D` : "none",
              transition: "all 250ms ease",
            }}
            onMouseEnter={(e) => { if (canRun && !loading) { e.currentTarget.style.transform = "scale(1.015)"; e.currentTarget.style.boxShadow = `0 0 56px rgba(201,168,76,0.4), 0 0 84px ${NBA_RED}47, 0 0 84px ${NBA_BLUE}52`; } }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = "scale(1)"; if (canRun && !loading) e.currentTarget.style.boxShadow = `0 0 32px rgba(201,168,76,0.2), 0 0 60px ${NBA_RED}33, 0 0 60px ${NBA_BLUE}3D`; }}
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
