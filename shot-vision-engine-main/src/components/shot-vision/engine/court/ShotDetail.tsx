import { useEffect, useState } from "react";
import type { AttainabilityExplanation, HeatmapPoint, MatchupExplanation, Player } from "@/lib/shot-vision-data";
import { getAttainabilityExplanationAPI, getMatchupExplanationAPI } from "@/lib/api";
import { AttainabilityWhy } from "../explain/AttainabilityWhy";
import { MatchupWhy } from "../explain/MatchupWhy";
import { formatAttainability } from "../explain/format";

// One grid point, read out in full. The heat field can only show magnitude;
// this is where the shot the model is actually scoring becomes legible.
export function ShotDetail({ point, attacker, defender, showProb, season, onClose }: { point: HeatmapPoint; attacker: Player; defender: Player; showProb: boolean; season: string; onClose: () => void }) {
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
  }, [point.zone, point.loc_x, point.loc_y, attacker.id, defender.id, season]);

  useEffect(() => {
    if (!whyOpen || why || whyError) return;
    let cancelled = false;
    // Passing the named defender adds the attacker/defender comparison
    // attainability's own model doesn't make on its own — see
    // AttainabilityDefenderTendency.
    getAttainabilityExplanationAPI(attacker.id, point.zone, season, point.loc_x, point.loc_y, defender.id)
      .then((res) => { if (!cancelled) setWhy(res); })
      .catch((e) => { if (!cancelled) setWhyError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [whyOpen, why, whyError, attacker.id, defender.id, point.zone, point.loc_x, point.loc_y, season]);

  // The make-probability breakdown — offense vs defense, including who else
  // is on the floor (team_creation/help_defense). Same on-demand pattern as
  // the attainability breakdown above: a real TreeSHAP decomposition, not a
  // cheap lookup.
  const [matchupWhyOpen, setMatchupWhyOpen] = useState(false);
  const [matchupWhy, setMatchupWhy] = useState<MatchupExplanation | null>(null);
  const [matchupWhyError, setMatchupWhyError] = useState<string | null>(null);

  useEffect(() => {
    setMatchupWhyOpen(false);
    setMatchupWhy(null);
    setMatchupWhyError(null);
  }, [point.zone, point.loc_x, point.loc_y, attacker.id, defender.id, season]);

  useEffect(() => {
    if (!matchupWhyOpen || matchupWhy || matchupWhyError) return;
    let cancelled = false;
    getMatchupExplanationAPI(attacker.id, point.zone, point.loc_x, point.loc_y, season, defender.id)
      .then((res) => { if (!cancelled) setMatchupWhy(res); })
      .catch((e) => { if (!cancelled) setMatchupWhyError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [matchupWhyOpen, matchupWhy, matchupWhyError, attacker.id, defender.id, point.zone, point.loc_x, point.loc_y, season]);

  const band =
    point.ep_low != null && point.ep_high != null
      ? `${point.ep_low.toFixed(2)}–${point.ep_high.toFixed(2)}`
      : null;

  const rows: Array<{ label: string; value: string; accent?: string; note?: string; expandable?: boolean; expandKey?: "attainability" | "matchup" }> = [
    {
      label: "MAKE PROB",
      value: `${(point.make_probability * 100).toFixed(1)}%`,
      accent: showProb ? "#C9A84C" : undefined,
      note: "tap for why",
      expandable: true,
      expandKey: "matchup",
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
      expandKey: "attainability",
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
          const isOpen = r.expandKey === "matchup" ? matchupWhyOpen : whyOpen;
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
                    {isOpen ? "▾" : "▸"}
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
              onClick={() => {
                if (r.expandKey === "matchup") setMatchupWhyOpen((v) => !v);
                else setWhyOpen((v) => !v);
              }}
              aria-expanded={isOpen}
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
      {matchupWhyOpen && (
        <MatchupWhy explanation={matchupWhy} error={matchupWhyError} />
      )}
    </div>
  );
}
