import { heightToFeet, type MatchupResponse, type Player } from "@/lib/shot-vision-data";
import { Disclosure } from "./explain/Disclosure";
import { ProjectedBadge } from "./badges";

export function MatchupEdge({ attacker, defender, data, loading, error }: {
  attacker: Player; defender: Player;
  // Fetched once by ResultsOutput (the common parent) and shared with the
  // ranked zone cards above, which now show the same attacker-vs-defender
  // exploit-zone numbers instead of a bare EP figure — this used to be its
  // own independent fetch here, duplicating the request.
  data: MatchupResponse | null; loading: boolean; error: string | null;
}) {
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
