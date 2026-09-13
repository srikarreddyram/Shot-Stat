import { useMemo, useState } from "react";
import {
  Column, Row, RADAR_AXES, RADAR_STAT_KEYS, COMPARE_COLORS,
  axisScore, buildDistributions, formatStat, isRankable, matchesQuery,
  numericValue, percentileOf, rankOf, foldAccents,
} from "../../lib/stat-engine";
import { C, F, NUM, Bar, Card, SectionLabel, TeamChip } from "./ui";

// Head-to-head comparison for up to three players: a radar summary over
// six measured dimensions, then every stat side by side with each player's
// league percentile and rank.
//
// The radar is a summary, not a verdict — see RADAR_AXES' note in
// lib/stat-engine.ts on why each axis is an unweighted mean of percentile
// ranks rather than a weighted formula. An axis only appears when EVERY
// selected player has data for it: a radar is a comparison device, and a
// polygon drawn through a hole compares a player against nothing.
//
// Bars here are coloured by PLAYER, not by quality tier as elsewhere on the
// page. In a comparison the question is "which of these three", so the
// colour has to answer "who" — a tier ramp would make all three bars the
// same colour whenever the players are of similar quality, which is exactly
// when the comparison matters most.

const SLOT_PLACEHOLDERS = ["Search a player…", "Compare with…", "And… (optional)"];

export function ComparePlayers({ columns, rows, teamNames, onOpenPlayer }: {
  columns: Column[];
  rows: Row[];
  teamNames: Map<string, string>;
  onOpenPlayer: (id: string) => void;
}) {
  const [ids, setIds] = useState<(string | null)[]>([null, null, null]);

  const byId = useMemo(() => {
    const map = new Map<string, Row>();
    for (const r of rows) map.set(String(r.player_id), r);
    return map;
  }, [rows]);

  const picked = ids
    .map((id, slot) => (id ? { slot, row: byId.get(id) } : null))
    .filter((p): p is { slot: number; row: Row } => p != null && p.row != null);

  const distributions = useMemo(
    () => buildDistributions(rows, [...columns.map((c) => c.key), ...RADAR_STAT_KEYS]),
    [rows, columns]
  );

  const setSlot = (slot: number, id: string | null) =>
    setIds((prev) => prev.map((v, i) => (i === slot ? id : v)));

  return (
    <>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 18 }}>
        {ids.map((id, slot) => (
          <PlayerPicker
            key={slot}
            rows={rows}
            teamNames={teamNames}
            color={COMPARE_COLORS[slot]}
            selectedId={id}
            disabledIds={ids.filter((x, i) => x && i !== slot) as string[]}
            onSelect={(chosen) => setSlot(slot, chosen)}
            placeholder={SLOT_PLACEHOLDERS[slot]}
          />
        ))}
      </div>

      {picked.length < 2 ? (
        <Card style={{ padding: "56px 24px", textAlign: "center" }}>
          <div style={{ fontFamily: F.display, fontSize: 30, color: C.muted, letterSpacing: "0.04em" }}>
            PICK TWO OR THREE PLAYERS
          </div>
          <div style={{ fontSize: 12.5, color: C.faint, marginTop: 8, lineHeight: 1.6, maxWidth: 440, margin: "8px auto 0" }}>
            Every stat is shown side by side with each player's league percentile and rank, and the
            best value in each row is highlighted.
          </div>
        </Card>
      ) : (
        <>
          <RadarPanel picked={picked} distributions={distributions} />
          <HeadToHead
            picked={picked}
            columns={columns}
            distributions={distributions}
            teamNames={teamNames}
            onOpenPlayer={onOpenPlayer}
          />
        </>
      )}
    </>
  );
}

// ── Radar ───────────────────────────────────────────────────────────────────

function RadarPanel({ picked, distributions }: {
  picked: { slot: number; row: Row }[];
  distributions: Map<string, number[]>;
}) {
  const scored = RADAR_AXES.map((axis) => ({
    axis,
    values: picked.map((p) => axisScore(p.row, axis, distributions)),
  }));
  const comparable = scored.filter((s) => s.values.every((v) => v != null));
  const dropped = scored.filter((s) => !s.values.every((v) => v != null));

  return (
    <Card style={{ padding: "18px 22px 20px", marginBottom: 16 }}>
      <SectionLabel>Profile shape</SectionLabel>
      <div style={{ fontSize: 11.5, color: C.faint, margin: "8px 0 10px", lineHeight: 1.55, maxWidth: 720 }}>
        Each axis is the average league percentile of the measured stats behind it — a summary of
        the rows below, not a rating. Further out is better on every axis.
      </div>

      {comparable.length < 3 ? (
        <div style={{ color: C.muted, fontSize: 12.5, padding: "24px 0" }}>
          Not enough shared dimensions to draw a shape — these players have measured data for fewer
          than three of the same areas.
        </div>
      ) : (
        <div style={{ display: "flex", gap: 30, flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ flex: "1 1 330px", maxWidth: RADAR_W }}>
            <RadarChart
              axes={comparable.map((c) => c.axis.label)}
              series={picked.map((p, i) => ({
                color: COMPARE_COLORS[p.slot],
                values: comparable.map((c) => c.values[i] as number),
              }))}
            />
          </div>
          <div style={{ flex: "1 1 250px", minWidth: 230 }}>
            {comparable.map((c) => {
              const best = Math.max(...(c.values as number[]));
              return (
                <div key={c.axis.key} style={{ padding: "9px 0", borderBottom: `1px solid ${C.rule}` }}>
                  <div style={{ fontSize: 12, color: C.dim, marginBottom: 6 }}>{c.axis.label}</div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    {picked.map((p, i) => {
                      const v = c.values[i] as number;
                      return (
                        <div key={p.slot} style={{ flex: 1, display: "flex", alignItems: "center", gap: 6 }}>
                          <Bar pct={v} color={COMPARE_COLORS[p.slot]} height={4} />
                          <span style={{
                            ...NUM, fontSize: 11, width: 20, textAlign: "right",
                            color: COMPARE_COLORS[p.slot], fontWeight: v === best ? 700 : 400,
                          }}>
                            {Math.round(v * 100)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {dropped.length > 0 && (
        <div style={{ fontSize: 11, color: C.muted, marginTop: 16, lineHeight: 1.55 }}>
          Not shown, because at least one of these players has no measured data for it:{" "}
          <span style={{ color: C.faint }}>{dropped.map((d) => d.axis.label).join(", ")}</span>.
        </div>
      )}
    </Card>
  );
}

// Wider than it is tall on purpose: the left and right axis labels sit
// outside the shape and need horizontal room, or they clip at the viewBox
// edge ("Rim pressure" becoming "Rim pre").
const RADAR_W = 440;
const RADAR_H = 320;
const RADAR_R = 104;

function RadarChart({ axes, series }: {
  axes: string[];
  series: { color: string; values: number[] }[];
}) {
  const cx = RADAR_W / 2;
  const cy = RADAR_H / 2;
  const n = axes.length;
  const angle = (i: number) => (-Math.PI / 2) + (i * 2 * Math.PI) / n;
  const point = (i: number, r: number) => [cx + r * Math.cos(angle(i)), cy + r * Math.sin(angle(i))];
  const ring = (r: number) => axes.map((_, i) => point(i, r).join(",")).join(" ");

  return (
    <svg
      viewBox={`0 0 ${RADAR_W} ${RADAR_H}`}
      style={{ width: "100%", maxWidth: RADAR_W, height: "auto", overflow: "visible" }}
      role="img"
      aria-label="Percentile profile comparison"
    >
      {[0.25, 0.5, 0.75, 1].map((f, i) => (
        <polygon
          key={f}
          points={ring(RADAR_R * f)}
          fill={i === 3 ? "rgba(255,255,255,0.012)" : "none"}
          stroke={i === 3 ? "rgba(201,168,76,0.18)" : "rgba(255,255,255,0.06)"}
          strokeWidth={1}
        />
      ))}
      {axes.map((_, i) => {
        const [x, y] = point(i, RADAR_R);
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(255,255,255,0.06)" strokeWidth={1} />;
      })}
      {series.map((s, si) => (
        <polygon
          key={si}
          points={s.values.map((v, i) => point(i, RADAR_R * v).join(",")).join(" ")}
          fill={s.color}
          fillOpacity={0.13}
          stroke={s.color}
          strokeWidth={2}
          strokeLinejoin="round"
        />
      ))}
      {series.map((s, si) =>
        s.values.map((v, i) => {
          const [x, y] = point(i, RADAR_R * v);
          return <circle key={`${si}-${i}`} cx={x} cy={y} r={3} fill={C.bg} stroke={s.color} strokeWidth={2} />;
        })
      )}
      {axes.map((label, i) => {
        const [x, y] = point(i, RADAR_R + 20);
        // Nudge the anchor so labels on the left/right sides don't overlap
        // the shape, and the top/bottom ones stay centred.
        const dx = x - cx;
        const anchor = Math.abs(dx) < 12 ? "middle" : dx > 0 ? "start" : "end";
        return (
          <text
            key={label}
            x={x}
            y={y}
            textAnchor={anchor}
            dominantBaseline="middle"
            fill={C.muted}
            style={{ fontFamily: F.mono, fontSize: 8.5, letterSpacing: "0.14em", textTransform: "uppercase" }}
          >
            {label}
          </text>
        );
      })}
    </svg>
  );
}

// ── Side-by-side stat rows ──────────────────────────────────────────────────

function HeadToHead({ picked, columns, distributions, teamNames, onOpenPlayer }: {
  picked: { slot: number; row: Row }[];
  columns: Column[];
  distributions: Map<string, number[]>;
  teamNames: Map<string, string>;
  onOpenPlayer: (id: string) => void;
}) {
  // Group by the section metadata the API already sent, so the comparison
  // reads basic-to-advanced in the same order as a single player's profile.
  const sections = useMemo(() => {
    const out: { group: string; label: string; cols: Column[] }[] = [];
    for (const c of columns) {
      const group = c.group ?? "other";
      let section = out.find((s) => s.group === group);
      if (!section) {
        section = { group, label: c.group_label ?? group, cols: [] };
        out.push(section);
      }
      section.cols.push(c);
    }
    return out;
  }, [columns]);

  const grid = `minmax(158px, 1.15fr) repeat(${picked.length}, minmax(128px, 1fr))`;

  return (
    <>
      {/* Sticky under the page header, so the reader never loses track of
          which colour is which player while scrolling 44 rows of stats. */}
      <div style={{
        position: "sticky", top: 72, zIndex: 5,
        background: "rgba(10,10,15,0.94)", backdropFilter: "blur(10px)",
        display: "grid", gridTemplateColumns: grid, gap: 12,
        padding: "10px 18px", marginBottom: 10,
        border: `1px solid ${C.edge}`, borderRadius: 6,
      }}>
        <div style={{ fontFamily: F.mono, fontSize: 9, color: C.muted, letterSpacing: "0.25em", alignSelf: "center" }}>
          STAT
        </div>
        {picked.map((p) => (
          <button
            key={p.slot}
            onClick={() => onOpenPlayer(String(p.row.player_id))}
            title="Open full profile"
            style={{
              background: "transparent", border: "none", padding: 0, cursor: "pointer",
              textAlign: "left", minWidth: 0,
            }}
          >
            <div style={{
              display: "flex", alignItems: "center", gap: 7,
              color: COMPARE_COLORS[p.slot], fontSize: 13.5, fontWeight: 600,
            }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: COMPARE_COLORS[p.slot], flexShrink: 0 }} />
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {String(p.row.name)}
              </span>
            </div>
            <div style={{ marginLeft: 15, marginTop: 1 }}>
              <TeamChip abbrev={p.row.team_id != null ? teamNames.get(String(p.row.team_id)) : null} />
            </div>
          </button>
        ))}
      </div>

      {sections.map((section) => (
        <Card key={section.group} style={{ padding: "15px 18px 9px", marginBottom: 12 }}>
          <div style={{ marginBottom: 12 }}><SectionLabel>{section.label}</SectionLabel></div>
          {section.cols.map((col) => (
            <CompareRow
              key={col.key}
              col={col}
              picked={picked}
              sorted={distributions.get(col.key) ?? []}
              grid={grid}
            />
          ))}
        </Card>
      ))}
    </>
  );
}

function CompareRow({ col, picked, sorted, grid }: {
  col: Column;
  picked: { slot: number; row: Row }[];
  sorted: number[];
  grid: string;
}) {
  const rankable = isRankable(col.key, col.fmt);
  const cells = picked.map((p) => {
    const value = p.row[col.key];
    const n = numericValue(value);
    return {
      slot: p.slot,
      value,
      pct: rankable && n != null ? percentileOf(n, sorted, col.key) : null,
      rank: rankable && n != null ? rankOf(n, sorted, col.key) : null,
    };
  });

  // "Best" follows the same direction rule as the percentile, and only where
  // a direction exists — highlighting the taller of two players as the winner
  // would be asserting something the number does not say.
  const bestPct = rankable ? Math.max(...cells.map((c) => c.pct ?? -1)) : -1;

  return (
    <div style={{
      display: "grid", gridTemplateColumns: grid, gap: 12,
      alignItems: "center", padding: "9px 0", borderBottom: `1px solid ${C.rule}`,
    }}>
      <div style={{ fontSize: 12.5, color: C.dim }}>{col.label}</div>
      {cells.map((c) => {
        const isBest = bestPct > -1 && c.pct != null && c.pct === bestPct;
        const color = COMPARE_COLORS[c.slot];
        return (
          <div key={c.slot} style={{ minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
              <span style={{
                ...NUM, fontSize: 13, whiteSpace: "nowrap",
                color: isBest ? color : C.text,
                fontWeight: isBest ? 700 : 400,
              }}>
                {formatStat(c.value, col.fmt)}
              </span>
              {c.rank && (
                <span style={{ ...NUM, fontSize: 9.5, color: C.muted }}>#{c.rank.rank}</span>
              )}
              {isBest && (
                <span style={{ color, fontSize: 9, lineHeight: 1 }} title="Best of the selected players">▲</span>
              )}
            </div>
            {c.pct != null && (
              <div style={{ display: "flex", marginTop: 5 }}>
                <Bar pct={c.pct} color={color} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Picker ──────────────────────────────────────────────────────────────────

function PlayerPicker({ rows, teamNames, color, selectedId, disabledIds, onSelect, placeholder }: {
  rows: Row[];
  teamNames: Map<string, string>;
  color: string;
  selectedId: string | null;
  disabledIds: string[];
  onSelect: (id: string | null) => void;
  placeholder: string;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const selected = selectedId ? rows.find((r) => String(r.player_id) === selectedId) : null;

  const matches = useMemo(() => {
    const q = foldAccents(query.trim().toLowerCase());
    if (!q) return [];
    return rows
      .filter((r) => !disabledIds.includes(String(r.player_id)) && matchesQuery(String(r.name ?? ""), q))
      .slice(0, 8);
  }, [rows, query, disabledIds]);

  if (selected) {
    return (
      <div style={{
        display: "flex", alignItems: "center", gap: 11, flex: "0 1 288px",
        background: C.surface, border: `1px solid ${C.edge}`, borderLeft: `3px solid ${color}`,
        borderRadius: 6, padding: "11px 13px",
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {String(selected.name)}
          </div>
          <div style={{ marginTop: 3 }}>
            <TeamChip abbrev={selected.team_id != null ? teamNames.get(String(selected.team_id)) : null} />
          </div>
        </div>
        <button
          onClick={() => { onSelect(null); setQuery(""); }}
          aria-label={`Remove ${String(selected.name)}`}
          style={{
            background: "transparent", border: "none", color: C.muted, cursor: "pointer",
            fontSize: 17, lineHeight: 1, padding: "0 2px", transition: "color 200ms",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = C.red)}
          onMouseLeave={(e) => (e.currentTarget.style.color = C.muted)}
        >
          ×
        </button>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", flex: "0 1 288px" }}>
      <input
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        // Blur is delayed so a click on a result lands before the list closes.
        onBlur={() => window.setTimeout(() => setOpen(false), 150)}
        placeholder={placeholder}
        style={{
          width: "100%", background: C.surface, border: `1px dashed ${C.edge}`,
          borderLeft: `3px solid ${C.edge}`, borderRadius: 6,
          padding: "15px 13px", color: C.text, fontFamily: F.body, fontSize: 13,
          outline: "none", transition: "border-color 200ms",
        }}
        onFocusCapture={(e) => (e.currentTarget.style.borderLeftColor = color)}
        onBlurCapture={(e) => (e.currentTarget.style.borderLeftColor = C.edge)}
      />
      {open && matches.length > 0 && (
        <div style={{
          position: "absolute", top: "calc(100% + 5px)", left: 0, right: 0, zIndex: 20,
          background: C.raised, border: `1px solid ${C.edgeStrong}`, borderRadius: 4,
          overflow: "hidden", boxShadow: "0 12px 32px rgba(0,0,0,0.6)",
        }}>
          {matches.map((r) => (
            <button
              key={String(r.player_id)}
              onMouseDown={() => { onSelect(String(r.player_id)); setQuery(""); setOpen(false); }}
              style={{
                display: "flex", width: "100%", justifyContent: "space-between", gap: 10,
                background: "transparent", border: "none", padding: "9px 13px", cursor: "pointer",
                color: C.text, fontSize: 13, textAlign: "left", transition: "background 150ms",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = C.hover)}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <span>{String(r.name)}</span>
              <TeamChip abbrev={r.team_id != null ? teamNames.get(String(r.team_id)) : null} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
