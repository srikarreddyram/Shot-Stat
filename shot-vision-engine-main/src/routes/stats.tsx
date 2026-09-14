import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { CareerPanel } from "../components/stat-engine/CareerPanel";
import { ComparePlayers } from "../components/stat-engine/ComparePlayers";
import { ArchetypeCard, DefenseCategoryChart, PlayTypeBars, ShotZoneCourt } from "../components/stat-engine/Visuals";
import { RadarChart, RADAR_W } from "../components/stat-engine/ComparePlayers";
import {
  C, F, NUM, Bar, Card, CourtWatermark, DEFAULT_ACCENT, EmptyState,
  ErrorState, GhostButton, NavLink, RankPill, SearchInput, SectionLabel,
  PlayerAvatar, Segmented, TableSkeleton, TeamBackdrop, TeamChip, TeamPageTint,
  accentVars, teamColor, tierColor, NBAEdgeBezel, NBADefaultWash, LeagueBrandMark, NBAHeaderStripe,
} from "../components/stat-engine/ui";
import {
  ArchetypeCatalogueEntry, ArchetypeDetail, ArchetypesResponse, Column,
  PlayerArchetype, Row, Stat, PlayerProfile, TeamProfile,
  buildDistributions, formatStat, isRankable, matchesQuery, numericValue,
  percentileOf, rankOf, foldAccents, sortRows, meetsQualifier, LOWER_IS_BETTER,
  RADAR_AXES, axisScore,
} from "../lib/stat-engine";

// Stat Engine — browse every stat this project has collected, for every
// player and every team, basic through niche. This answers "what does this
// PLAYER (or team) actually do"; the model-introspection page that used to
// live here now lives at /model-features.
//
// Three shapes, because they answer different questions:
//   • leaderboard — sortable, "who leads the league in X, and where does
//     this player sit in that list" (the rank column survives a name search,
//     so searching a player inside a sorted stat shows their real rank)
//   • profile     — one entity, every stat, sectioned basic-to-advanced
//   • compare     — two or three players side by side, radar plus every stat
//
// Column names, labels, formats and section grouping all come from the API
// (src/inference/stat_engine.py's STAT_GROUPS). This page deliberately does
// not keep its own copy of that list: a duplicated column list is how a
// renamed stat quietly becomes a permanently-empty column.
//
// Percentiles and ranks are computed here rather than server-side because
// the leaderboard payload is already the whole league — ranking a value
// against it is free on the client and would cost a round trip per profile.
// The logic lives in src/lib/stat-engine.ts; the visual language, shared with
// EnginePage, lives in src/components/stat-engine/ui.tsx.

export const Route = createFileRoute("/stats")({
  head: () => ({
    meta: [
      { title: "SHOT VISION — Stat Engine" },
      { name: "description", content: "Browse, rank and compare every player and team stat the engine tracks, from box-score basics to tracking-derived creation and defensive-activity metrics." },
    ],
  }),
  component: StatEnginePage,
});

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

// Table behaviour that inline styles cannot express: sticky row headers that
// stay put under horizontal scroll, hover that covers a whole row, and zebra
// banding. Sticky cells need an OPAQUE background or the columns they are
// meant to occlude show straight through them, so each state sets a solid
// colour rather than a translucent overlay.
const TABLE_CSS = `
.se-table { border-collapse: separate; border-spacing: 0; width: 100%; }
.se-table th {
  position: sticky; top: 0; z-index: 2; background: ${C.surface};
  padding: 11px 12px; white-space: nowrap; cursor: pointer; user-select: none;
  font-family: ${F.mono}; font-size: 9px; letter-spacing: 0.14em;
  text-transform: uppercase; color: ${C.muted}; text-align: right;
  border-bottom: 1px solid ${C.edge}; transition: color 200ms;
}
.se-table th:hover { color: ${C.gold}; }
.se-table th.se-active { color: ${C.gold}; }
.se-table th.se-left, .se-table td.se-left { text-align: left; }
.se-table td {
  padding: 9px 12px; white-space: nowrap; text-align: right;
  font-family: ${F.mono}; font-variant-numeric: tabular-nums; font-size: 12px;
  color: rgba(240,240,240,0.8); border-bottom: 1px solid ${C.rule};
  transition: background 150ms;
}
.se-table td.se-name { font-family: ${F.body}; font-size: 13px; color: ${C.text}; }
.se-table th.se-stick, .se-table td.se-stick { position: sticky; }
.se-table th.se-stick { z-index: 4; }
.se-table td.se-stick { z-index: 1; background: ${C.surface}; }
/* The pinned offsets must match the rendered widths EXACTLY. A table cell
   sizes itself from its content, so a bare width is only a hint — if the
   rank column renders wider than the offset the name column is pinned at,
   a strip of the scrolling columns shows through the gap between them. */
.se-table th.se-col-rank, .se-table td.se-col-rank {
  left: 0; box-sizing: border-box; width: 58px; min-width: 58px; max-width: 58px;
}
.se-table th.se-col-name, .se-table td.se-col-name { left: 58px; }
.se-table tbody tr:nth-child(even) td,
.se-table tbody tr:nth-child(even) td.se-stick { background: #101018; }
.se-table tbody tr:hover { cursor: pointer; }
.se-table tbody tr:hover td,
.se-table tbody tr:hover td.se-stick { background: #1c1c27; }
/* The pinned region needs a deliberate edge, or the seam where the sticky
   columns occlude the scrolling ones reads as a rendering artefact. */
.se-table th.se-col-name, .se-table td.se-col-name { border-right: 1px solid rgba(201,168,76,0.13); }
.se-scroll { overflow: auto; max-height: 70vh; border-radius: 6px; }
.se-scroll::-webkit-scrollbar { height: 9px; width: 9px; }
.se-scroll::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.22); border-radius: 5px; }
.se-scroll::-webkit-scrollbar-track { background: transparent; }
`;

type Mode = "players" | "teams" | "compare";

function StatEnginePage() {
  const [mode, setMode] = useState<Mode>("players");
  const [playerCols, setPlayerCols] = useState<Column[]>([]);
  const [playerRows, setPlayerRows] = useState<Row[]>([]);
  const [teamCols, setTeamCols] = useState<Column[]>([]);
  const [teamRows, setTeamRows] = useState<Row[]>([]);
  const [archetypeCatalogue, setArchetypeCatalogue] = useState<ArchetypeCatalogueEntry[]>([]);
  const [archetypeByPlayer, setArchetypeByPlayer] = useState<Map<string, PlayerArchetype>>(new Map());
  const [season, setSeason] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [selected, setSelected] = useState<{ kind: "player" | "team"; id: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const get = (path: string) =>
      fetch(`${API_BASE}${path}`).then((r) => {
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      });

    Promise.all([get("/stats/players"), get("/stats/teams"), get("/stats/archetypes")])
      .then(([p, t, a]: [any, any, ArchetypesResponse]) => {
        if (cancelled) return;
        setPlayerCols(p.columns ?? []);
        setPlayerRows(p.players ?? []);
        setTeamCols(t.columns ?? []);
        setTeamRows(t.teams ?? []);
        setSeason(p.season ?? t.season ?? null);
        setArchetypeCatalogue(a.catalogue ?? []);
        setArchetypeByPlayer(new Map((a.players ?? []).map((pa) => [pa.player_id, pa])));
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Could not load");
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const teamNames = useMemo(() => {
    const map = new Map<string, string>();
    for (const t of teamRows) {
      const id = t.team_id == null ? null : String(t.team_id);
      if (id) map.set(id, String(t.team_abbrev ?? t.team_name ?? id));
    }
    return map;
  }, [teamRows]);

  // Which team, if any, themes the page. A team profile is its own team; a
  // player profile takes the team they play for. The leaderboards stay on
  // SHOT VISION gold, since no single team owns a league-wide table.
  const themeTeamId = useMemo(() => {
    if (!selected) return null;
    if (selected.kind === "team") return selected.id;
    const row = playerRows.find((r) => String(r.player_id) === selected.id);
    return row?.team_id == null ? null : String(row.team_id);
  }, [selected, playerRows]);
  const themeAbbrev = themeTeamId ? teamNames.get(themeTeamId) ?? null : null;
  const accent = themeAbbrev ? teamColor(themeAbbrev) : DEFAULT_ACCENT;

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text, fontFamily: F.body,
      // 400ms, so switching between two players on different teams reads as
      // the page re-theming rather than as a flash.
      transition: "background 400ms",
      ...accentVars(accent),
    }}>
      <style>{TABLE_CSS}</style>
      <NBAEdgeBezel />
      {themeAbbrev ? <TeamPageTint abbrev={themeAbbrev} /> : <NBADefaultWash />}

      <header style={{
        position: "sticky", top: 0, zIndex: 30, background: "rgba(10,10,15,0.92)",
        backdropFilter: "blur(10px)", borderBottom: `1px solid ${C.edge}`,
        padding: "16px 28px", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 20,
        overflow: "hidden",
      }}>
        <CourtWatermark />
        <NBAHeaderStripe />
        <div style={{ display: "flex", alignItems: "center", gap: 14, position: "relative" }}>
          <LeagueBrandMark height={30} />
          <div style={{ width: 1, height: 26, background: "rgba(255,255,255,0.12)" }} />
          <div>
            <div style={{ fontFamily: F.display, fontSize: 24, letterSpacing: "0.05em" }}>STAT ENGINE</div>
            <div style={{ fontFamily: F.mono, fontSize: 9, color: C.gold, letterSpacing: "0.4em", marginTop: 2 }}>
              {season ? `LEAGUE-WIDE · ${season}` : "LEAGUE-WIDE"}
            </div>
          </div>
        </div>
        <nav style={{ display: "flex", gap: 22, alignItems: "center", position: "relative" }}>
          <NavLink href="/#engine">← ENGINE</NavLink>
        </nav>
      </header>

      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "26px 28px 64px", position: "relative", zIndex: 1 }}>
        {error && <ErrorState message={error} />}

        {loading && !error && (
          <>
            <div style={{ height: 34, width: 340, background: "rgba(255,255,255,0.04)", borderRadius: 4, marginBottom: 18 }} />
            <TableSkeleton />
          </>
        )}

        {!loading && !error && (
          selected ? (
            selected.kind === "player" ? (
              <PlayerProfileView
                playerId={selected.id}
                leagueRows={playerRows}
                leagueColumns={playerCols}
                teamNames={teamNames}
                onBack={() => setSelected(null)}
                onOpenTeam={(id) => setSelected({ kind: "team", id })}
              />
            ) : (
              <TeamProfileView
                teamId={selected.id}
                teamRows={teamRows}
                teamCols={teamCols}
                onBack={() => setSelected(null)}
                onOpenPlayer={(id) => setSelected({ kind: "player", id })}
              />
            )
          ) : (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap", marginBottom: 18 }}>
                <Segmented
                  value={mode}
                  onChange={(v) => setMode(v as Mode)}
                  options={[
                    { value: "players", label: "PLAYERS" },
                    { value: "teams", label: "TEAMS" },
                    { value: "compare", label: "COMPARE" },
                  ]}
                />
                <div style={{ ...NUM, fontSize: 10, color: C.muted, letterSpacing: "0.2em" }}>
                  {mode === "teams" ? `${teamRows.length} TEAMS` : `${playerRows.length} PLAYERS`}
                </div>
              </div>

              {mode === "players" && (
                <PlayerLeaderboard
                  columns={playerCols}
                  rows={playerRows}
                  teamNames={teamNames}
                  archetypeCatalogue={archetypeCatalogue}
                  archetypeByPlayer={archetypeByPlayer}
                  onOpen={(id) => setSelected({ kind: "player", id })}
                />
              )}
              {mode === "teams" && (
                <TeamLeaderboard
                  columns={teamCols}
                  rows={teamRows}
                  onOpen={(id) => setSelected({ kind: "team", id })}
                />
              )}
              {mode === "compare" && (
                <ComparePlayers
                  columns={playerCols}
                  rows={playerRows}
                  teamNames={teamNames}
                  onOpenPlayer={(id) => setSelected({ kind: "player", id })}
                />
              )}
            </>
          )
        )}
      </div>
    </div>
  );
}

function useSort(defaultKey: string, defaultDir: "asc" | "desc" = "asc") {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" }>({ key: defaultKey, dir: defaultDir });
  const toggle = (key: string) =>
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
        // A fresh column opens on "who leads the league", which is the
        // question being asked ~every time someone clicks a stat — so a
        // lower-is-better stat opens ASCENDING (best defensive rating on
        // top), not descending, which would lead with the worst team.
        : { key, dir: key === "name" || key === "team_name" || LOWER_IS_BETTER.has(key) ? "asc" : "desc" }
    );
  return { sort, toggle };
}

function Th({ label, active, dir, onClick, className = "" }: {
  label: string; active: boolean; dir: "asc" | "desc"; onClick: () => void; className?: string;
}) {
  return (
    <th onClick={onClick} title={label} className={`${className}${active ? " se-active" : ""}`}>
      {label}
      <span style={{ opacity: active ? 1 : 0, marginLeft: 4 }}>{dir === "asc" ? "▲" : "▼"}</span>
    </th>
  );
}

function useGroupChooser(columns: Column[]) {
  const groups = useMemo(() => {
    const seen: { value: string; label: string }[] = [];
    for (const c of columns) {
      if (c.group && !seen.some((g) => g.value === c.group)) {
        seen.push({ value: c.group, label: c.group_label ?? c.group });
      }
    }
    return seen;
  }, [columns]);
  const [group, setGroup] = useState<string | null>(null);
  const active = group ?? groups[0]?.value ?? "";
  const shown = useMemo(() => columns.filter((c) => c.group === active), [columns, active]);
  return { groups, active, setGroup, shown };
}

/** The column being sorted on, when it is one that can be ranked at all. */
function useRanking(columns: Column[], rows: Row[], sortKey: string) {
  const sorted = columns.find((c) => c.key === sortKey);
  const rankingBy = sorted && isRankable(sorted.key, sorted.fmt) ? sorted : null;
  const distribution = useMemo(
    () => (rankingBy ? buildDistributions(rows, [rankingBy.key], columns).get(rankingBy.key) ?? [] : []),
    [rows, rankingBy, columns]
  );
  return { rankingBy, distribution };
}

// The column-group chooser (Overview, Shooting, Hustle, ...) used to render
// as a Chips row, but at 9 groups it wrapped onto its own line on anything
// narrower than a wide desktop. A native <select> holds all of them in one
// compact control, same idea as the archetype filter below.
function SectionSelect({ options, value, onChange }: {
  options: { value: string; label: string }[]; value: string; onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        flex: "0 1 240px", background: C.surface, border: `1px solid ${C.edgeStrong}`,
        borderRadius: 3, padding: "9px 13px", color: C.gold,
        fontFamily: F.mono, fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase",
        outline: "none",
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

/** A small labelled number input — "MIN GP" / "MIN MPG" — for filtering out
 *  small-sample players (a bench guy who went 4-for-5 all season) from the
 *  leaderboard regardless of which column is currently sorted. Empty means
 *  no floor at all, not zero, so an unset filter never silently excludes
 *  anyone. */
function MinFilterInput({ label, value, onChange }: {
  label: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8, flex: "0 0 auto",
      background: C.surface, border: `1px solid ${value ? C.edgeStrong : C.edge}`,
      borderRadius: 3, padding: "6px 10px 6px 13px",
    }}>
      <span style={{ fontFamily: F.mono, fontSize: 9.5, letterSpacing: "0.15em", color: value ? C.gold : C.muted, whiteSpace: "nowrap" }}>
        {label}
      </span>
      <input
        type="number"
        min={0}
        inputMode="numeric"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="0"
        style={{
          width: 44, background: "transparent", border: "none", outline: "none",
          color: C.text, fontFamily: F.body, fontSize: 13,
        }}
      />
    </div>
  );
}

// A native <select>, grouped by category via <optgroup> — the archetype
// catalogue is 27 entries, too many for the Chips row every other filter on
// this page uses without wrapping into its own multi-line mess.
function ArchetypeFilterSelect({ catalogue, value, onChange }: {
  catalogue: ArchetypeCatalogueEntry[]; value: string; onChange: (v: string) => void;
}) {
  const byCategory = useMemo(() => {
    const groups: { category: string; entries: ArchetypeCatalogueEntry[] }[] = [];
    for (const entry of catalogue) {
      let g = groups.find((x) => x.category === entry.category);
      if (!g) { g = { category: entry.category, entries: [] }; groups.push(g); }
      g.entries.push(entry);
    }
    return groups;
  }, [catalogue]);

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        flex: "0 1 240px", background: C.surface, border: `1px solid ${value ? C.edgeStrong : C.edge}`,
        borderRadius: 3, padding: "9px 13px", color: value ? C.gold : C.text,
        fontFamily: F.body, fontSize: 13, outline: "none",
      }}
    >
      <option value="">All archetypes</option>
      {byCategory.map((g) => (
        <optgroup key={g.category} label={g.category}>
          {g.entries.map((e) => (
            <option key={e.key} value={e.key}>{e.label}</option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

function PlayerLeaderboard({ columns, rows, teamNames, archetypeCatalogue, archetypeByPlayer, onOpen }: {
  columns: Column[]; rows: Row[]; teamNames: Map<string, string>;
  archetypeCatalogue: ArchetypeCatalogueEntry[];
  archetypeByPlayer: Map<string, PlayerArchetype>;
  onOpen: (id: string) => void;
}) {
  // 44 columns is far too many for one readable table, so the section tabs
  // from the profile page double as the leaderboard's column chooser — the
  // same basic-to-advanced grouping, used to slice width instead of depth.
  const { groups, active, setGroup, shown } = useGroupChooser(columns);
  const [query, setQuery] = useState("");
  const [archetypeFilter, setArchetypeFilter] = useState("");
  // A user-set volume floor, independent of whichever column is being
  // ranked. The per-column qualify_key/qualify_min pair below already keeps
  // a 2-for-3 shooter off the top of a PERCENTAGE ranking, but it only ever
  // applies to the one column actually being sorted by — browsing the table
  // unsorted, or sorted by a column with no qualifier of its own, still
  // surfaces a five-shot-all-season case as a "leader". Empty string means
  // no floor, not zero — most columns' minimums are left to the user rather
  // than guessed at.
  const [minGames, setMinGames] = useState("");
  const [minMpg, setMinMpg] = useState("");
  const { sort, toggle } = useSort("name");
  const { rankingBy, distribution } = useRanking(columns, rows, sort.key);

  const filtered = useMemo(() => {
    const q = foldAccents(query.trim().toLowerCase());
    let base = q ? rows.filter((r) => matchesQuery(String(r.name ?? ""), q)) : rows;
    if (archetypeFilter) {
      base = base.filter((r) => archetypeByPlayer.get(String(r.player_id))?.archetype_key === archetypeFilter);
    }
    const gpFloor = minGames.trim() === "" ? null : Number(minGames);
    if (gpFloor != null && !Number.isNaN(gpFloor)) {
      base = base.filter((r) => { const v = numericValue(r.games_played); return v != null && v >= gpFloor; });
    }
    const mpgFloor = minMpg.trim() === "" ? null : Number(minMpg);
    if (mpgFloor != null && !Number.isNaN(mpgFloor)) {
      base = base.filter((r) => { const v = numericValue(r.min_per_game); return v != null && v >= mpgFloor; });
    }
    const sorted = sortRows(base, sort.key, sort.dir);
    if (!rankingBy?.qualify_key) return sorted;
    // Unqualified players keep their row but sink below everyone who cleared
    // the threshold — otherwise a 1-for-2 sample still owns the top of the
    // table, just without a rank badge next to it.
    const ok = sorted.filter((r) => meetsQualifier(r, rankingBy));
    return [...ok, ...sorted.filter((r) => !meetsQualifier(r, rankingBy))];
  }, [rows, query, archetypeFilter, archetypeByPlayer, minGames, minMpg, sort, rankingBy]);

  return (
    <>
      <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <SearchInput value={query} onChange={setQuery} placeholder="Search players…" />
        <SectionSelect options={groups} value={active} onChange={setGroup} />
        <ArchetypeFilterSelect catalogue={archetypeCatalogue} value={archetypeFilter} onChange={setArchetypeFilter} />
        <MinFilterInput label="MIN GP" value={minGames} onChange={setMinGames} />
        <MinFilterInput label="MIN MPG" value={minMpg} onChange={setMinMpg} />
        <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.5 }}>
          {rankingBy ? (
            <>
              Ranked by <span style={{ color: C.gold }}>{rankingBy.label}</span> across{" "}
              {distribution.length} qualified players — a name search keeps the league rank
              instead of renumbering.
              {rankingBy.qualify_key && (
                <> Needs at least {rankingBy.qualify_min}{" "}
                  {(columns.find((c) => c.key === rankingBy.qualify_key)?.label ?? "attempts").toLowerCase()}
                  ; below that the number is shown without a rank.</>
              )}
            </>
          ) : (
            <>Click any stat column to rank the league by it.</>
          )}
        </div>
      </div>

      <Card style={{ overflow: "hidden" }}>
        <div className="se-scroll">
          <table className="se-table">
            <thead>
              <tr>
                <th className="se-stick se-col-rank" style={{ cursor: "default" }}>#</th>
                <Th className="se-stick se-col-name se-left" label="Player" active={sort.key === "name"} dir={sort.dir} onClick={() => toggle("name")} />
                <Th className="se-left" label="Team" active={sort.key === "team_id"} dir={sort.dir} onClick={() => toggle("team_id")} />
                <th className="se-left" style={{ cursor: "default" }}>Archetype</th>
                {shown.map((c) => (
                  <Th key={c.key} label={c.label} active={sort.key === c.key} dir={sort.dir} onClick={() => toggle(c.key)} />
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => {
                const qualified = rankingBy ? meetsQualifier(r, rankingBy) : true;
                const value = rankingBy && qualified ? numericValue(r[rankingBy.key]) : null;
                const rank = value != null ? rankOf(value, distribution, rankingBy!.key) : null;
                const pct = value != null ? percentileOf(value, distribution, rankingBy!.key) : null;
                return (
                  <tr key={String(r.player_id)} onClick={() => onOpen(String(r.player_id))}>
                    <td className="se-stick se-col-rank" style={{ color: rank ? tierColor(pct) : C.muted, fontSize: 11 }}>
                      {rankingBy ? (rank ? rank.rank : "—") : i + 1}
                    </td>
                    <td className="se-stick se-col-name se-name se-left">
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 9 }}>
                        <PlayerAvatar
                          playerId={String(r.player_id)}
                          teamAbbrev={r.team_id != null ? teamNames.get(String(r.team_id)) : null}
                          size={26}
                        />
                        {String(r.name ?? "—")}
                      </span>
                    </td>
                    <td className="se-left">
                      <TeamChip abbrev={r.team_id != null ? teamNames.get(String(r.team_id)) : null} />
                    </td>
                    <td className="se-left" style={{ fontSize: 11, color: C.dim }}>
                      {archetypeByPlayer.get(String(r.player_id))?.archetype_label ?? (
                        <span style={{ color: C.muted }}>—</span>
                      )}
                    </td>
                    {shown.map((c) => {
                      // Only the ranked column is tinted. Colouring every cell
                      // would turn the table into confetti and stop the tint
                      // meaning anything.
                      const isRanked = rankingBy?.key === c.key;
                      return (
                        <td key={c.key} style={isRanked ? { color: tierColor(pct), fontWeight: 600 } : undefined}>
                          {formatStat(r[c.key], c.fmt)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {filtered.length === 0 && <EmptyState>NO PLAYERS MATCH THAT SEARCH</EmptyState>}
        </div>
      </Card>
    </>
  );
}

function TeamLeaderboard({ columns, rows, onOpen }: { columns: Column[]; rows: Row[]; onOpen: (id: string) => void }) {
  const [query, setQuery] = useState("");
  const { sort, toggle } = useSort("team_name");
  const { rankingBy, distribution } = useRanking(columns, rows, sort.key);

  const filtered = useMemo(() => {
    const q = foldAccents(query.trim().toLowerCase());
    const base = q ? rows.filter((r) => matchesQuery(`${r.team_name ?? ""} ${r.team_abbrev ?? ""}`, q)) : rows;
    return sortRows(base, sort.key, sort.dir);
  }, [rows, query, sort]);

  return (
    <>
      <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <SearchInput value={query} onChange={setQuery} placeholder="Search teams…" />
        <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.5, maxWidth: 620 }}>
          Columns prefixed <span style={{ color: C.gold }}>Avg</span> are roster averages of
          player-level stats, not separately-measured team numbers — defensive rating is the only
          genuinely team-level stat here.
        </div>
      </div>

      <Card style={{ overflow: "hidden" }}>
        <div className="se-scroll">
          <table className="se-table">
            <thead>
              <tr>
                <th className="se-stick se-col-rank" style={{ cursor: "default" }}>#</th>
                <Th className="se-stick se-col-name se-left" label="Team" active={sort.key === "team_name"} dir={sort.dir} onClick={() => toggle("team_name")} />
                {columns.map((c) => (
                  <Th key={c.key} label={c.label} active={sort.key === c.key} dir={sort.dir} onClick={() => toggle(c.key)} />
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => {
                const qualified = rankingBy ? meetsQualifier(r, rankingBy) : true;
                const value = rankingBy && qualified ? numericValue(r[rankingBy.key]) : null;
                const rank = value != null ? rankOf(value, distribution, rankingBy!.key) : null;
                const pct = value != null ? percentileOf(value, distribution, rankingBy!.key) : null;
                return (
                  <tr key={String(r.team_id)} onClick={() => onOpen(String(r.team_id))}>
                    <td className="se-stick se-col-rank" style={{ color: rank ? tierColor(pct) : C.muted, fontSize: 11 }}>
                      {rankingBy ? (rank ? rank.rank : "—") : i + 1}
                    </td>
                    <td className="se-stick se-col-name se-name se-left">
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 9 }}>
                        <span style={{
                          width: 3, height: 15, borderRadius: 1, flexShrink: 0,
                          background: teamColor(r.team_abbrev == null ? null : String(r.team_abbrev)),
                        }} />
                        {String(r.team_name ?? r.team_abbrev ?? "—")}
                      </span>
                    </td>
                    {columns.map((c) => {
                      const isRanked = rankingBy?.key === c.key;
                      return (
                        <td
                          key={c.key}
                          style={
                            isRanked
                              ? { color: tierColor(pct), fontWeight: 600 }
                              : c.derived
                                ? { color: "rgba(240,240,240,0.55)" }
                                : undefined
                          }
                        >
                          {formatStat(r[c.key], c.fmt)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
          {filtered.length === 0 && <EmptyState>NO TEAMS MATCH THAT SEARCH</EmptyState>}
        </div>
      </Card>
    </>
  );
}

// ── Profiles ────────────────────────────────────────────────────────────────

function findStat(profile: PlayerProfile, key: string): Stat | null {
  for (const section of profile.sections) {
    const hit = section.stats.find((s) => s.key === key);
    if (hit) return hit;
  }
  return null;
}

function PlayerProfileView({ playerId, leagueRows, leagueColumns, teamNames, onBack, onOpenTeam }: {
  playerId: string;
  leagueRows: Row[];
  leagueColumns: Column[];
  teamNames: Map<string, string>;
  onBack: () => void;
  onOpenTeam: (id: string) => void;
}) {
  const [profile, setProfile] = useState<PlayerProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"season" | "career">("season");
  const [side, setSide] = useState<"offense" | "defense">("offense");
  // VISUAL (default): a profile-shape radar up top and dense sections
  // rendered as compact bar charts instead of a tall list. LIST: every stat
  // exactly as before, one row apiece — the toggle the user asked for, so
  // switching back costs nothing and loses no information either way.
  const [density, setDensity] = useState<"visual" | "list">("visual");

  useEffect(() => {
    let cancelled = false;
    setProfile(null);
    setError(null);
    fetch(`${API_BASE}/stats/player/${encodeURIComponent(playerId)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: PlayerProfile) => { if (!cancelled) setProfile(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [playerId]);

  // Sorted values per column, once, so each stat row can locate itself in the
  // league without re-scanning the whole table. Also covers every stat
  // RADAR_AXES needs, since those are all already members of some section.
  const distributions = useMemo(
    () => buildDistributions(
      leagueRows,
      profile ? profile.sections.flatMap((s) => s.stats.map((x) => x.key)) : [],
      leagueColumns
    ),
    [profile, leagueRows, leagueColumns]
  );

  // This player's own leaderboard row, for the profile-shape radar — the
  // same RADAR_AXES/axisScore ComparePlayers.tsx already uses, just with one
  // series instead of two or three.
  const playerRow = useMemo(
    () => leagueRows.find((r) => String(r.player_id) === String(playerId)) ?? null,
    [leagueRows, playerId]
  );
  const radarAxes = useMemo(() => {
    if (!playerRow) return [];
    return RADAR_AXES
      .map((axis) => ({ axis, value: axisScore(playerRow, axis, distributions, leagueColumns) }))
      .filter((a): a is { axis: typeof RADAR_AXES[number]; value: number } => a.value != null);
  }, [playerRow, distributions, leagueColumns]);

  if (error) return <><GhostButton onClick={onBack}>← ALL STATS</GhostButton><div style={{ marginTop: 16 }}><ErrorState message={error} /></div></>;
  if (!profile) {
    return (
      <>
        <GhostButton onClick={onBack}>← ALL STATS</GhostButton>
        <div style={{ marginTop: 16 }}><TableSkeleton rows={8} /></div>
      </>
    );
  }

  const teamLabel = profile.team_id ? teamNames.get(String(profile.team_id)) : null;
  const accent = teamLabel ? teamColor(teamLabel) : C.gold;
  const physique = ["position", "height", "weight"]
    .map((k) => {
      const s = findStat(profile, k);
      if (!s || s.value == null) return null;
      return k === "weight" ? `${formatStat(s.value, s.fmt)} lb` : formatStat(s.value, s.fmt);
    })
    .filter(Boolean)
    .join("  ·  ");

  const headline = [
    { key: "off_rating", label: "Offense" },
    { key: "def_rating_ours", label: "Defense" },
    { key: "two_k_overall", label: "NBA 2K" },
  ]
    .map(({ key, label }) => {
      const s = findStat(profile, key);
      const v = numericValue(s?.value);
      if (v == null) return null;
      const sorted = distributions.get(key) ?? [];
      return { label, value: v, pct: percentileOf(v, sorted, key), rank: rankOf(v, sorted, key) };
    })
    .filter((x): x is NonNullable<typeof x> => x != null);

  const infoSections = profile.sections.filter((s) => s.side === "info");
  const sideSections = profile.sections.filter((s) => s.side === side);

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <GhostButton onClick={onBack}>← ALL STATS</GhostButton>
        <span style={{ ...NUM, fontSize: 10, color: C.muted, letterSpacing: "0.25em" }}>{profile.season}</span>
      </div>

      <Card accent={accent} style={{ padding: "22px 26px", marginBottom: 16, position: "relative", overflow: "hidden" }}>
        <TeamBackdrop teamId={profile.team_id} abbrev={teamLabel} height={260} />
        <div style={{ display: "flex", justifyContent: "space-between", gap: 28, flexWrap: "wrap", alignItems: "flex-end", position: "relative", zIndex: 1 }}>
          <div style={{ minWidth: 220, display: "flex", alignItems: "center", gap: 18 }}>
            <PlayerAvatar playerId={playerId} teamAbbrev={teamLabel} size={86} variant="large" />
            <div>
            <div style={{ fontFamily: F.display, fontSize: 42, lineHeight: 1.05, letterSpacing: "0.02em" }}>
              {profile.name}
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
              {profile.team_id && (
                <button
                  onClick={() => onOpenTeam(String(profile.team_id))}
                  style={{
                    background: `${accent}1F`, border: `1px solid ${accent}55`,
                    color: accent, cursor: "pointer", borderRadius: 2, padding: "3px 9px",
                    fontFamily: F.mono, fontSize: 10, letterSpacing: "0.2em", transition: "all 200ms",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = `${accent}38`)}
                  onMouseLeave={(e) => (e.currentTarget.style.background = `${accent}1F`)}
                >
                  {teamLabel ?? "TEAM"} →
                </button>
              )}
              {physique && (
                <span style={{ ...NUM, fontSize: 11, color: C.muted, letterSpacing: "0.06em" }}>{physique}</span>
              )}
            </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 30, flexWrap: "wrap" }}>
            {headline.map((h) => (
              <div key={h.label} style={{ minWidth: 78 }}>
                <div style={{ fontFamily: F.mono, fontSize: 9, color: C.muted, letterSpacing: "0.25em", textTransform: "uppercase" }}>
                  {h.label}
                </div>
                <div style={{ fontFamily: F.display, fontSize: 44, lineHeight: 1.05, color: tierColor(h.pct) }}>
                  {h.value.toFixed(0)}
                </div>
                {h.rank && (
                  <div style={{ ...NUM, fontSize: 9.5, color: C.muted }}>
                    #{h.rank.rank} / {h.rank.outOf}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </Card>

      {density === "visual" && radarAxes.length >= 3 && (
        <Card style={{ padding: "18px 22px 20px", marginBottom: 16 }}>
          <SectionLabel>Profile shape</SectionLabel>
          <div style={{ fontSize: 11.5, color: C.faint, margin: "8px 0 14px", lineHeight: 1.55, maxWidth: 720 }}>
            Each axis is the league percentile of this player's measured stats in that area — a
            summary of the sections below, not a separate rating. Further out is better on every axis.
          </div>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <div style={{ maxWidth: RADAR_W, width: "100%" }}>
              <RadarChart
                axes={radarAxes.map((a) => a.axis.label)}
                series={[{ color: accent, values: radarAxes.map((a) => a.value) }]}
              />
            </div>
          </div>
        </Card>
      )}

      {/* "Info" sections (Overview, Ratings) belong to neither side of the
          ball — shown once here, above the split, rather than duplicated
          into both tabs or hidden behind either one. */}
      <div style={{ columnWidth: 320, columnGap: 16, marginBottom: 16 }}>
        <div style={{ breakInside: "avoid" }}>
          <ArchetypeCard playerId={playerId} />
        </div>
        {infoSections.map((section) => (
            <div key={section.group} style={{ breakInside: "avoid", marginBottom: 16 }}>
              <Card style={{ padding: "15px 19px 9px" }}>
                <div style={{ marginBottom: 12 }}><SectionLabel>{section.label}</SectionLabel></div>
                {density === "visual" && section.stats.length > COMPACT_THRESHOLD ? (
                  <CompactStatBars stats={section.stats} distributions={distributions} />
                ) : (
                  section.stats.map((stat) => (
                    <StatRow key={stat.key} stat={stat} sorted={distributions.get(stat.key) ?? []} />
                  ))
                )}
              </Card>
            </div>
          ))}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
        <Segmented
          value={side}
          onChange={(v) => setSide(v as "offense" | "defense")}
          options={[
            { value: "offense", label: "OFFENSE" },
            { value: "defense", label: "DEFENSE" },
          ]}
        />
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Segmented
            value={density}
            onChange={(v) => setDensity(v as "visual" | "list")}
            options={[
              { value: "visual", label: "VISUAL" },
              { value: "list", label: "LIST" },
            ]}
          />
          <Segmented
            value={view}
            onChange={(v) => setView(v as "season" | "career")}
            options={[
              { value: "season", label: "THIS SEASON" },
              { value: "career", label: "CAREER" },
            ]}
          />
        </div>
      </div>

      {view === "career" ? (
        <CareerPanel playerId={playerId} side={side} />
      ) : (
        <>
          <div style={{ fontSize: 11.5, color: C.faint, marginBottom: 16, lineHeight: 1.6, maxWidth: 780 }}>
            Bars and ranks place this player against everyone the stat is measured for, and are coloured
            by tier — green is top of the league, red is bottom. A dash means no measurement, not a zero.
            Stats with no better-or-worse direction (height, shot-diet shares) are shown without a rank.
          </div>

          {sideSections.length === 0 ? (
            <EmptyState>NO {side.toUpperCase()} STATS FOR THIS PLAYER</EmptyState>
          ) : (
            <>
              {side === "offense" && <ShotZoneCourt profile={profile} />}
              {side === "offense" && <PlayTypeBars profile={profile} />}
              {side === "defense" && <DefenseCategoryChart profile={profile} />}

              {/* Columns, not grid: the sections are wildly different heights
                  (Shooting has 16 rows, Overview 6) and a grid leaves a tall
                  hole under the short ones. CSS columns pack them. */}
              <div style={{ columnWidth: 350, columnGap: 16 }}>
                {sideSections.map((section) => (
                  <div key={section.group} style={{ breakInside: "avoid", marginBottom: 16 }}>
                    <Card style={{ padding: "15px 19px 9px" }}>
                      <div style={{ marginBottom: 12 }}><SectionLabel>{section.label}</SectionLabel></div>
                      {density === "visual" && section.stats.length > COMPACT_THRESHOLD ? (
                        <CompactStatBars stats={section.stats} distributions={distributions} />
                      ) : (
                        section.stats.map((stat) => (
                          <StatRow key={stat.key} stat={stat} sorted={distributions.get(stat.key) ?? []} />
                        ))
                      )}
                    </Card>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}

function StatRow({ stat, sorted }: { stat: Stat; sorted: number[] }) {
  const numeric = numericValue(stat.value);
  const rankable = isRankable(stat.key, stat.fmt);
  const pct = rankable && numeric != null ? percentileOf(numeric, sorted, stat.key) : null;
  const rank = rankable && numeric != null ? rankOf(numeric, sorted, stat.key) : null;
  const missing = stat.value == null || stat.value === "";

  return (
    <div style={{ padding: "8px 0", borderBottom: `1px solid ${C.rule}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
        <div style={{ fontSize: 12.5, color: missing ? C.faint : C.dim }}>{stat.label}</div>
        <div style={{ ...NUM, fontSize: 13, color: missing ? C.muted : C.text, whiteSpace: "nowrap", fontWeight: missing ? 400 : 500 }}>
          {formatStat(stat.value, stat.fmt)}
        </div>
      </div>
      {pct != null && (
        <div style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 6 }}>
          <Bar pct={pct} />
          {rank && <RankPill rank={rank.rank} outOf={rank.outOf} pct={pct} />}
        </div>
      )}
    </div>
  );
}

// Sections past this many stats get the compact treatment in VISUAL mode —
// short sections (Overview, Ratings, most Hustle groups) are already
// scannable at a glance and gain nothing from being squeezed further; the
// long ones (Shooting ~20 rows, Creation ~17, Defense ~24) are exactly what
// made the page feel like an endless scroll.
const COMPACT_THRESHOLD = 8;

// One line per stat instead of StatRow's two (label+value, then bar+rank on
// its own row below) — same bar, same rank pill, same every underlying
// number, just laid out so a 20-stat section reads as one glance down a
// column of thin bars rather than a page and a half of scrolling.
function CompactStatBars({ stats, distributions }: { stats: Stat[]; distributions: Map<string, number[]> }) {
  return (
    <div>
      {stats.map((stat) => {
        const numeric = numericValue(stat.value);
        const rankable = isRankable(stat.key, stat.fmt);
        const sorted = distributions.get(stat.key) ?? [];
        const pct = rankable && numeric != null ? percentileOf(numeric, sorted, stat.key) : null;
        const rank = rankable && numeric != null ? rankOf(numeric, sorted, stat.key) : null;
        const missing = stat.value == null || stat.value === "";

        return (
          <div
            key={stat.key}
            style={{ display: "flex", alignItems: "center", gap: 10, padding: "5px 0", borderBottom: `1px solid ${C.rule}` }}
          >
            <div style={{ flex: "1.3 1 0", minWidth: 0, fontSize: 11, color: missing ? C.faint : C.dim, lineHeight: 1.3 }}>
              {stat.label}
            </div>
            <div style={{ flex: "1 1 0", minWidth: 36 }}>
              {pct != null && <Bar pct={pct} height={5} />}
            </div>
            <div style={{
              ...NUM, fontSize: 11.5, width: 52, flexShrink: 0, textAlign: "right",
              color: missing ? C.muted : C.text, fontWeight: missing ? 400 : 500,
            }}>
              {formatStat(stat.value, stat.fmt)}
            </div>
            <div style={{ width: 46, flexShrink: 0 }}>
              {rank && pct != null && <RankPill rank={rank.rank} outOf={rank.outOf} pct={pct} />}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TeamProfileView({ teamId, teamRows, teamCols, onBack, onOpenPlayer }: {
  teamId: string;
  teamRows: Row[];
  teamCols: Column[];
  onBack: () => void;
  onOpenPlayer: (id: string) => void;
}) {
  const [profile, setProfile] = useState<TeamProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setProfile(null);
    setError(null);
    fetch(`${API_BASE}/stats/team/${encodeURIComponent(teamId)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: TeamProfile) => { if (!cancelled) setProfile(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, [teamId]);

  const leagueRow = useMemo(
    () => teamRows.find((r) => String(r.team_id) === String(teamId)) ?? null,
    [teamRows, teamId]
  );
  const distributions = useMemo(
    () => buildDistributions(teamRows, teamCols.map((c) => c.key)),
    [teamCols, teamRows]
  );
  const rosterDistributions = useMemo(() => {
    const players = (profile?.roster ?? []).map((p) => ({
      off_rating: p.off_rating, def_rating_ours: p.def_rating_ours,
    })) as Row[];
    return buildDistributions(players, ["off_rating", "def_rating_ours"]);
  }, [profile]);

  if (error) return <><GhostButton onClick={onBack}>← ALL STATS</GhostButton><div style={{ marginTop: 16 }}><ErrorState message={error} /></div></>;
  if (!profile) {
    return (
      <>
        <GhostButton onClick={onBack}>← ALL STATS</GhostButton>
        <div style={{ marginTop: 16 }}><TableSkeleton rows={8} /></div>
      </>
    );
  }

  const defRating = numericValue(leagueRow?.def_rating);
  const defPct = defRating != null ? percentileOf(defRating, distributions.get("def_rating") ?? [], "def_rating") : null;
  const defRank = defRating != null ? rankOf(defRating, distributions.get("def_rating") ?? [], "def_rating") : null;

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <GhostButton onClick={onBack}>← ALL STATS</GhostButton>
        <span style={{ ...NUM, fontSize: 10, color: C.muted, letterSpacing: "0.25em" }}>{profile.season}</span>
      </div>

      <Card accent={teamColor(profile.team_abbrev)} style={{ padding: "26px 26px", marginBottom: 16, position: "relative", overflow: "hidden" }}>
        <TeamBackdrop teamId={profile.team_id} abbrev={profile.team_abbrev} height={300} />
        <div style={{ display: "flex", justifyContent: "space-between", gap: 28, flexWrap: "wrap", alignItems: "flex-end", position: "relative", zIndex: 1 }}>
          <div>
            <div style={{ fontFamily: F.display, fontSize: 42, lineHeight: 1.05, letterSpacing: "0.02em" }}>
              {profile.team_name ?? profile.team_id}
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 8 }}>
              <TeamChip abbrev={profile.team_abbrev} size="lg" />
              <span style={{ ...NUM, fontSize: 10, color: C.muted, letterSpacing: "0.25em" }}>
                {profile.roster.length} PLAYERS
              </span>
            </div>
          </div>
          {defRating != null && (
            <div style={{ minWidth: 96 }}>
              <div style={{ fontFamily: F.mono, fontSize: 9, color: C.muted, letterSpacing: "0.25em", textTransform: "uppercase" }}>
                Def rating
              </div>
              <div style={{ fontFamily: F.display, fontSize: 44, lineHeight: 1.05, color: tierColor(defPct) }}>
                {defRating.toFixed(1)}
              </div>
              {defRank && <div style={{ ...NUM, fontSize: 9.5, color: C.muted }}>#{defRank.rank} / {defRank.outOf}</div>}
            </div>
          )}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(290px, 1fr) minmax(330px, 1.25fr)", gap: 16, alignItems: "start" }}>
        <div>
          {(["offense", "defense"] as const).map((side) => {
            const cols = teamCols.filter((c) => c.side === side);
            if (cols.length === 0) return null;
            return (
              <Card key={side} style={{ padding: "15px 19px 9px", marginBottom: 14 }}>
                <div style={{ marginBottom: 12 }}><SectionLabel>Team {side}</SectionLabel></div>
                {cols.map((c) => (
                  <StatRow
                    key={c.key}
                    stat={{
                      key: c.key,
                      label: c.derived ? `${c.label} (roster avg)` : c.label,
                      fmt: c.fmt,
                      value: leagueRow ? (leagueRow[c.key] as number | string | null) : null,
                    }}
                    sorted={distributions.get(c.key) ?? []}
                  />
                ))}
              </Card>
            );
          })}
        </div>

        <Card style={{ padding: "15px 19px 12px" }}>
          <div style={{ marginBottom: 12 }}><SectionLabel>Roster</SectionLabel></div>
          <table className="se-table" style={{ marginTop: -2 }}>
            <thead>
              <tr>
                <th className="se-left" style={{ cursor: "default", position: "static" }}>Player</th>
                <th style={{ cursor: "default", position: "static" }}>Pos</th>
                <th style={{ cursor: "default", position: "static" }}>Off</th>
                <th style={{ cursor: "default", position: "static" }}>Def</th>
              </tr>
            </thead>
            <tbody>
              {profile.roster.map((p) => (
                <tr key={p.player_id} onClick={() => onOpenPlayer(p.player_id)}>
                  <td className="se-name se-left" style={{ padding: "7px 12px" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 9 }}>
                      <PlayerAvatar playerId={p.player_id} teamAbbrev={profile.team_abbrev} size={24} />
                      {p.name}
                    </span>
                  </td>
                  <td style={{ color: C.muted, fontSize: 11, padding: "7px 12px" }}>{p.position ?? "—"}</td>
                  <RosterRating value={p.off_rating} sorted={rosterDistributions.get("off_rating") ?? []} statKey="off_rating" />
                  <RosterRating value={p.def_rating_ours} sorted={rosterDistributions.get("def_rating_ours") ?? []} statKey="def_rating_ours" />
                </tr>
              ))}
            </tbody>
          </table>
          {profile.roster.length === 0 && <EmptyState>NO ROSTERED PLAYERS THIS SEASON</EmptyState>}
        </Card>
      </div>
    </>
  );
}

/** A roster rating cell, tinted against THIS ROSTER rather than the league —
 *  the question a roster list answers is "who on this team", so a league tint
 *  would paint a whole rebuilding roster uniformly red and say nothing. */
function RosterRating({ value, sorted, statKey }: { value: number | null; sorted: number[]; statKey: string }) {
  const pct = value != null ? percentileOf(value, sorted, statKey) : null;
  return (
    <td style={{ padding: "7px 12px", color: pct != null ? tierColor(pct) : C.muted, fontSize: 11.5 }}>
      {formatStat(value, "num")}
    </td>
  );
}
