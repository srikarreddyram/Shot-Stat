import type { AttainabilityExplanation } from "@/lib/shot-vision-data";
import { formatAttainability, ordinal } from "./format";

// The attainability breakdown, opened from the shot detail panel.
//
// Reports the split the model actually makes: what any player would get in
// this zone, then what this player's own game adds or gives up. Collapsing
// those into one number is what makes a bare "4%" unreadable — it hides
// whether the shot is rare for everybody or rare for him.
export function AttainabilityWhy({ explanation, error }: { explanation: AttainabilityExplanation | null; error: string | null }) {
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

      {explanation.defender?.note && (
        <div style={{
          marginTop: 12,
          paddingTop: 10,
          borderTop: "1px solid rgba(255,255,255,0.07)",
        }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
            THIS MATCHUP SPECIFICALLY
          </div>
          {explanation.defender.defender_freq != null && explanation.defender.league_freq != null && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0 5px" }}>
              <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden", display: "flex" }}>
                <div style={{
                  width: `${Math.min(100, explanation.defender.defender_freq * 100)}%`,
                  background: "#DC2626",
                }} />
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#94A3B8", whiteSpace: "nowrap" }}>
                {Math.round(explanation.defender.defender_freq * 100)}% HIS EXPOSURE
                <span style={{ color: "#4a5568" }}>
                  {" / LG "}{Math.round(explanation.defender.league_freq * 100)}%
                </span>
              </div>
            </div>
          )}
          <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 11, color: "#CBD5E1", lineHeight: 1.5 }}>
            {explanation.defender.note}
          </div>
        </div>
      )}

      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#4a5568", marginTop: 10, lineHeight: 1.5 }}>
        CONTRIBUTIONS ARE EXACT (TREESHAP) AND SUM TO THE ESTIMATE. PERCENTILES ARE LEAGUE-WIDE.
      </div>
    </div>
  );
}
