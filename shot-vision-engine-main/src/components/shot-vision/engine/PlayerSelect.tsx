import { useEffect, useState } from "react";
import { heightToFeet, type Player } from "@/lib/shot-vision-data";
import { searchPlayersAPI } from "@/lib/api";
import { foldAccents } from "./textUtils";
import { ProjectedBadge, TwoKBadge } from "./badges";
import { PlayerAvatar } from "./PlayerAvatar";

export function PlayerSelect({ selected, onSelect, season, accent, statLabels, statValues, ratingValue, excludeIds, excludeTeamId, roster }: {
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
  // The OTHER side's team — a teammate can't guard (or be guarded by)
  // another teammate, so a player on this team never shows up as a
  // candidate here. Undefined/null (unrostered, or the other slot is
  // still empty) applies no filter.
  excludeTeamId?: string | null;
  // Team-vs-team mode: a fixed candidate list (one team's roster) instead
  // of a league-wide search. When set, typing filters this list by name
  // client-side rather than hitting /players/search — a ~15-player roster
  // needs no server round-trip to filter.
  roster?: Player[];
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [searched, setSearched] = useState<Player[]>([]);

  useEffect(() => {
    if (roster || query.length < 2 || !season) {
      setSearched([]);
      return;
    }
    const timer = setTimeout(() => {
      searchPlayersAPI(query, season).then(res => setSearched(res)).catch(console.error);
    }, 250);
    return () => clearTimeout(timer);
  }, [query, season, roster]);

  const filtered = roster
    ? roster.filter(p => query.length === 0 || foldAccents(p.name.toLowerCase()).includes(foldAccents(query.toLowerCase())))
    : searched;

  // Filtered at render time (not inside the fetch) so excluding a player
  // stays correct even if excludeIds/excludeTeamId changes without the user
  // retyping — e.g. picking the attacker after the defender dropdown
  // already has cached results.
  const withoutExcluded = filtered.filter(p => !excludeIds?.includes(p.id));
  const visible = withoutExcluded
    .filter(p => !excludeTeamId || p.teamId !== excludeTeamId)
    .slice(0, roster ? 20 : 8);
  // A search can come back non-empty yet show nothing once the team filter
  // applies — every match is a teammate of the other slot. Silence there
  // reads as a broken search, not as the rule actually working.
  const allSameTeam = withoutExcluded.length > 0 && visible.length === 0 && !!excludeTeamId;

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
              {selected.ratingSource === "2k_fallback" && <TwoKBadge />}
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
        placeholder={roster ? "Filter roster..." : "Search player..."}
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
      {open && allSameTeam && (
        <div style={{ position: "absolute", top: "100%", left: 0, right: 0, marginTop: 4, background: "#16161f", border: "1px solid rgba(220,38,38,0.25)", borderRadius: 3, padding: "10px 16px", zIndex: 20 }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#DC2626" }}>
            Every match plays for the other side already — teammates can't guard each other.
          </div>
        </div>
      )}
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
                    {p.ratingSource === "2k_fallback" && <TwoKBadge compact />}
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
