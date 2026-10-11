// Flags a player whose numbers come from src/training/position_priors.py's
// position-bucket fallback, not real recorded stats — shown whenever a
// player's/response's stats_source is "prior" (no NBA shot history yet,
// e.g. this year's draft class before their first real game).
export function ProjectedBadge({ compact = false }: { compact?: boolean }) {
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

// A rating sourced from NBA 2K rather than our own measured data — a gap-
// filler for players our own model has nothing at all for (a true rookie,
// or too few real minutes to clear the eligibility bar), never a silent
// override of a real measured rating. See
// src/inference/player_ratings.two_k_fallback_ratings for the boundary.
export function TwoKBadge({ compact = false }: { compact?: boolean }) {
  return (
    <span
      title="No measured rating yet for this player — this number comes from NBA 2K, not our own model."
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: compact ? "1px 6px" : "2px 8px",
        borderRadius: 2,
        border: "1px solid rgba(148,110,230,0.35)",
        background: "rgba(148,110,230,0.08)",
        color: "#946EE6",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: compact ? 8 : 9,
        letterSpacing: "0.15em",
        whiteSpace: "nowrap",
      }}
    >
      2K
    </span>
  );
}
