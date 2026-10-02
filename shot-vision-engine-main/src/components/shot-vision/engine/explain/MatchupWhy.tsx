import type { MatchupExplanation, ShotQualityBreakdown, ShotQualityFactor } from "@/lib/shot-vision-data";

// One ranked list of factor bars — shared by the offense/defense sections
// of MatchupWhy. Bar width is relative to the largest factor IN THIS
// SECTION, same reasoning as AttainabilityWhy's factor bars: the ranking
// stays legible whether the spread within a side is wide or narrow.
function FactorSection({ title, factors }: { title: string; factors: ShotQualityFactor[] }) {
  if (factors.length === 0) return null;
  const max = Math.max(...factors.map((f) => Math.abs(f.impact)), 1e-9);
  const decimals = max * 100 < 1 ? 2 : 1;

  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#64748b", letterSpacing: "0.18em" }}>
        {title}
      </div>
      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 8 }}>
        {factors.map((f) => {
          const lowers = f.direction === "lowers";
          const width = `${Math.max(4, (Math.abs(f.impact) / max) * 100)}%`;
          return (
            <div key={f.feature}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 11, color: "#E2E8F0", minWidth: 0 }}>
                  {f.label}
                  <span style={{ color: "#64748b" }}>{" — "}{f.display_value}</span>
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
    </div>
  );
}

// Comparable shots — a nearest-neighbor sanity check on `make_probability`,
// not another TreeSHAP factor list, so it's styled deliberately differently
// (a bordered callout, not another bar section) to keep the two kinds of
// evidence visually distinct: one is exact model attribution, the other is
// "here's what actually happened to shots that looked like this."
function ComparableShots({ comparable, makeProbability }: {
  comparable: NonNullable<ShotQualityBreakdown["comparable_shots"]>;
  makeProbability: number;
}) {
  if (comparable.count < 5 || comparable.historical_make_rate == null) return null;
  const gapPp = Math.round(Math.abs(comparable.historical_make_rate - makeProbability) * 100);
  const diverges = gapPp > 8;

  return (
    <div style={{
      marginTop: 12, padding: "8px 10px", borderRadius: 6,
      border: `1px solid ${diverges ? "rgba(220,38,38,0.35)" : "rgba(22,163,74,0.3)"}`,
      background: diverges ? "rgba(220,38,38,0.06)" : "rgba(22,163,74,0.05)",
    }}>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace", fontSize: 8,
        color: diverges ? "#DC2626" : "#16A34A", letterSpacing: "0.18em",
        display: "flex", justifyContent: "space-between",
      }}>
        <span>COMPARABLE SHOTS ({comparable.count})</span>
        <span>{(comparable.historical_make_rate * 100).toFixed(0)}% ACTUAL</span>
      </div>
      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 3 }}>
        {comparable.examples.map((ex, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: "space-between", gap: 8,
            fontFamily: "'Inter', sans-serif", fontSize: 10.5, color: "#94A3B8",
          }}>
            <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {ex.player_name}{ex.opponent ? ` vs ${ex.opponent}` : ""}
            </span>
            <span style={{ color: ex.made ? "#16A34A" : "#DC2626", fontFamily: "'JetBrains Mono', monospace", flexShrink: 0 }}>
              {ex.made ? "MADE" : "MISS"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// The make-probability breakdown, opened from the shot detail panel.
//
// Splits the estimate by side of the matchup: the shooter's own profile —
// now including who else is on the floor with him, teammates' creation/
// gravity/rim-pressure/foul-drawing (team_creation) — versus the defense,
// the named defender's own numbers plus the four help defenders' shot-
// blocking/steals/deflections/disruption (help_defense). Both sides were
// computed all along; only the offense side (and the help-defense half of
// the defense side) was never actually shown before this panel existed.
//
// Unlike AttainabilityWhy's factors, `percentile`/`detail` are almost
// always null here — the shot-quality model carries no league percentile
// grid to compare against — so each line states the labeled value and its
// exact TreeSHAP contribution rather than a "high/low" trait clause.
export function MatchupWhy({ explanation, error }: { explanation: MatchupExplanation | null; error: string | null }) {
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

  return (
    <div style={wrap}>
      <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 12, color: "#CBD5E1", lineHeight: 1.5 }}>
        {explanation.narrative}
      </div>

      <FactorSection title="OFFENSE — HIS PROFILE & TEAMMATES" factors={explanation.shot_quality.offense_factors} />
      <FactorSection
        title={explanation.defender_name ? `DEFENSE — ${explanation.defender_name.toUpperCase()} & HELP` : "DEFENSE"}
        factors={explanation.shot_quality.defense_factors}
      />
      <FactorSection title="GAME SITUATION — SCORE, CLOCK & CLUTCH" factors={explanation.shot_quality.context_factors} />

      {explanation.shot_quality.comparable_shots && (
        <ComparableShots
          comparable={explanation.shot_quality.comparable_shots}
          makeProbability={explanation.shot_quality.make_probability}
        />
      )}

      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: "#4a5568", marginTop: 10, lineHeight: 1.5 }}>
        CONTRIBUTIONS ARE EXACT (TREESHAP) AND SUM TO THE ESTIMATE.
      </div>
    </div>
  );
}
