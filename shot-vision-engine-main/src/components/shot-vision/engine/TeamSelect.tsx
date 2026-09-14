import { useEffect, useState } from "react";
import type { Team } from "@/lib/shot-vision-data";
import { teamColor, teamLogoUrl } from "@/components/stat-engine/ui";
import { foldAccents } from "./textUtils";

// A team's crest, served by our own API (see teamLogoUrl) which proxies and
// caches NBA's CDN. Silently collapses to nothing on a 404/network error
// rather than showing a broken-image glyph in a dropdown row.
function TeamLogo({ teamId, size }: { teamId: string; size: number }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => setBroken(false), [teamId]);
  const src = teamLogoUrl(teamId);
  if (!src || broken) return <div style={{ width: size, height: size, flexShrink: 0 }} />;
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      onError={() => setBroken(true)}
      style={{ width: size, height: size, objectFit: "contain", flexShrink: 0 }}
    />
  );
}

// Team-vs-team mode's team picker. Deliberately not a text search like
// PlayerSelect — there are only 30 franchises, a fixed list small enough to
// just browse — but built the same way (a bordered input that opens a
// dropdown, a filled-in card once picked) so the two pickers read as one
// consistent control language rather than two different widgets bolted
// together.
export function TeamSelect({ teams, selected, onSelect, excludeTeamId, accent }: {
  teams: Team[];
  selected: Team | null;
  onSelect: (t: Team | null) => void;
  // The other side's team — a team can't play itself, so it's dropped from
  // this list entirely rather than shown disabled.
  excludeTeamId?: string | null;
  accent: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  // Once a team is picked, its own colour replaces the generic offense/
  // defense accent passed in — this is what makes the box read as "Lakers"
  // rather than as "the gold slot". Before anything is picked there's no
  // team colour to borrow yet, so it falls back to that generic accent.
  const theme = selected ? teamColor(selected.abbreviation) : accent;

  const visible = teams
    .filter(t => t.teamId !== excludeTeamId)
    .filter(t => query.length === 0
      || foldAccents(t.name.toLowerCase()).includes(foldAccents(query.toLowerCase()))
      || foldAccents(t.abbreviation.toLowerCase()).includes(foldAccents(query.toLowerCase())));

  if (selected) {
    return (
      <div style={{ background: "#16161f", borderLeft: `3px solid ${theme}`, padding: "12px 18px", borderRadius: 3, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <TeamLogo teamId={selected.teamId} size={36} />
          <div>
            <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 15, fontWeight: 600, color: "#F0F0F0" }}>{selected.name}</div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: theme, letterSpacing: "0.1em" }}>{selected.abbreviation}</div>
          </div>
        </div>
        <button onClick={() => onSelect(null)} style={{ background: "transparent", border: "none", color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, cursor: "pointer" }}>CHANGE</button>
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <input
        placeholder="Select team..."
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
      />
      {open && visible.length > 0 && (
        <div style={{ position: "absolute", top: "100%", left: 0, right: 0, marginTop: 4, background: "#16161f", border: "1px solid rgba(201,168,76,0.15)", borderRadius: 3, maxHeight: 280, overflowY: "auto", zIndex: 20 }}>
          {visible.map((t) => {
            const rowColor = teamColor(t.abbreviation);
            return (
              <div key={t.teamId} onMouseDown={() => { onSelect(t); setQuery(""); setOpen(false); }} style={{ padding: "10px 16px", cursor: "pointer", display: "flex", alignItems: "center", gap: 12, borderBottom: "1px solid rgba(255,255,255,0.03)", transition: "background 150ms" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = `${rowColor}14`)}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <TeamLogo teamId={t.teamId} size={22} />
                <div style={{ flex: 1, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ fontFamily: "'Inter', sans-serif", fontSize: 14, color: "#F0F0F0" }}>{t.name}</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: rowColor, letterSpacing: "0.05em" }}>{t.abbreviation}</div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
