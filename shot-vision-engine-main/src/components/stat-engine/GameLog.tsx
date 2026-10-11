import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  GameAverages,
  GameLog,
  GameRow,
  GameZone,
  PlayerGameDetail,
  SeasonType,
  TeamGameDetail,
  TeamGamePlayer,
} from "../../lib/game-logs";
import { ZONE_SHORT, useShotMap } from "../../lib/shot-maps";
import { formatStat } from "../../lib/stat-engine";
import { COURT_AREA_H, COURT_CW, LEGEND_H } from "../shot-vision/engine/court/geometry";
import { ShotFilter, ShotReplay } from "./ShotReplay";
import {
  C,
  Card,
  Chips,
  F,
  MiniTable,
  NUM,
  PlayerAvatar,
  SectionLabel,
  Segmented,
  teamColor,
} from "./ui";

// The game log on the Stat Engine profiles: every game of a season with its
// box score, newest first, and the chosen game opened up above the list —
// the box score set against the season average and every shot of the game
// replayed on the court. A team's game also gets both teams' totals side by
// side and its players' lines. Data: src/inference/game_logs.py (box scores
// from 2016-17 on; a game's shots are exactly its box score's field goals).

const FIRST_LOGGED_SEASON = "2016-17";
/** Below this many games "season high" means little: in a four-game playoff
 *  run most numbers are one. */
const MIN_GAMES_FOR_HIGHS = 5;
const WIN = "#16A34A";
const LOSS = "#DC2626";
/** A game's shots launch this far apart, slow enough to follow one by one: a
 *  player's 14 shots take 3.5 s; a team's 90 hit ShotReplay's 7 s cap (a
 *  season's thousands launch 7 ms apart). */
const GAME_SHOT_MS = 250;
/** The sticky site header's height, plus a little air. */
const HEADER_CLEARANCE = 84;

type Side = "offense" | "defense";
type Detail = PlayerGameDetail | TeamGameDetail;

// ── Formatting ──────────────────────────────────────────────────────────────
function onDay(iso: string, opts: Intl.DateTimeFormatOptions) {
  // Noon, so no time zone can move the date.
  return new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", opts);
}
const shortDate = (iso: string) => onDay(iso, { month: "short", day: "numeric" });
const longDate = (iso: string) =>
  onDay(iso, { weekday: "short", month: "short", day: "numeric", year: "numeric" });

function signed(v: number | null | undefined, digits = 0) {
  if (v == null) return "—";
  const s = Math.abs(v).toFixed(digits);
  if (Number(s) === 0) return s;
  return v > 0 ? `+${s}` : `−${s}`;
}

function madeOf(made: number | null | undefined, att: number | null | undefined) {
  return made == null || att == null ? "—" : `${made}-${att}`;
}

const pct = (v: number | null | undefined) => formatStat(v ?? null, "pct");

function Opponent({ g }: { g: GameRow }) {
  return (
    <span style={{ whiteSpace: "nowrap" }}>
      <span style={{ color: C.muted }}>{g.home ? "vs " : "@ "}</span>
      <span style={{ color: teamColor(g.opponent), letterSpacing: "0.06em" }}>{g.opponent}</span>
    </span>
  );
}

function Result({ g }: { g: GameRow }) {
  if (g.win == null) return <span style={{ color: C.muted }}>—</span>;
  return (
    <span style={{ whiteSpace: "nowrap" }}>
      <span style={{ color: g.win ? WIN : LOSS, fontWeight: 700 }}>{g.win ? "W" : "L"}</span>
      {g.team_score != null && g.opponent_score != null && (
        <span style={{ color: C.dim }}>
          {" "}
          {g.team_score}–{g.opponent_score}
        </span>
      )}
    </span>
  );
}

// A row is the click target; the button in it makes the row reachable by
// keyboard (its click bubbles up to the row).
const ROW_BUTTON: React.CSSProperties = {
  background: "none",
  border: "none",
  padding: 0,
  margin: 0,
  color: C.text,
  font: "inherit",
  cursor: "pointer",
  textAlign: "left",
};

// ── The season's games ──────────────────────────────────────────────────────
type LogCol = { label: string; cell: (g: GameRow) => ReactNode; high?: string };

const count = (key: "pts" | "reb" | "ast" | "stl" | "blk" | "tov" | "pf") => (g: GameRow) =>
  g[key] ?? "—";

const PLAYER_COLS: LogCol[] = [
  { label: "MIN", cell: (g) => (g.min == null ? "—" : Math.round(g.min)) },
  { label: "PTS", cell: count("pts"), high: "pts" },
  { label: "REB", cell: count("reb"), high: "reb" },
  { label: "AST", cell: count("ast"), high: "ast" },
  { label: "STL", cell: count("stl"), high: "stl" },
  { label: "BLK", cell: count("blk"), high: "blk" },
  { label: "TOV", cell: count("tov") },
  { label: "PF", cell: count("pf") },
  { label: "FG", cell: (g) => madeOf(g.fgm, g.fga) },
  { label: "FG%", cell: (g) => pct(g.fg_pct) },
  { label: "3P", cell: (g) => madeOf(g.fg3m, g.fg3a), high: "fg3m" },
  { label: "3P%", cell: (g) => pct(g.fg3_pct) },
  { label: "FT", cell: (g) => madeOf(g.ftm, g.fta) },
  { label: "FT%", cell: (g) => pct(g.ft_pct) },
  { label: "+/−", cell: (g) => signed(g.plus_minus) },
];
// A team's +/− is its winning margin, already in the result.
const TEAM_COLS = PLAYER_COLS.filter((c) => c.label !== "MIN" && c.label !== "+/−");

function LogTable({
  games,
  highs,
  cols,
  picked,
  onPick,
}: {
  games: GameRow[];
  highs: GameLog["highs"];
  cols: LogCol[];
  picked: string;
  onPick: (gameId: string) => void;
}) {
  const box = useRef<HTMLDivElement>(null);

  // Keep the chosen game's row in sight inside the list (it can be stepped to
  // with the arrows above) without scrolling the page itself.
  useEffect(() => {
    const el = box.current;
    const row = el?.querySelector<HTMLElement>("tr.se-picked");
    if (!el || !row) return;
    const head = el.querySelector("thead")?.getBoundingClientRect().height ?? 0;
    if (row.offsetTop - head < el.scrollTop) el.scrollTop = row.offsetTop - head;
    else if (row.offsetTop + row.offsetHeight > el.scrollTop + el.clientHeight)
      el.scrollTop = row.offsetTop + row.offsetHeight - el.clientHeight;
  }, [picked]);

  return (
    <div
      ref={box}
      className="se-scroll"
      style={{ maxHeight: 380, border: `1px solid ${C.edge}`, borderRadius: 4 }}
    >
      <table className="se-table" style={{ minWidth: 0 }}>
        <thead>
          <tr>
            {["Date", "Opp", "Result"].map((h) => (
              <th key={h} className="se-left" style={{ cursor: "default" }}>
                {h}
              </th>
            ))}
            {cols.map((c) => (
              <th key={c.label} style={{ cursor: "default" }}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {games.map((g) => {
            const on = g.game_id === picked;
            return (
              <tr
                key={g.game_id}
                className={on ? "se-picked" : undefined}
                onClick={() => onPick(g.game_id)}
              >
                <td className="se-left">
                  <button
                    type="button"
                    aria-pressed={on}
                    aria-label={`${longDate(g.date)}, ${g.home ? "home to" : "away at"} ${g.opponent}`}
                    style={ROW_BUTTON}
                  >
                    {shortDate(g.date)}
                  </button>
                </td>
                <td className="se-left">
                  <Opponent g={g} />
                </td>
                <td className="se-left">
                  <Result g={g} />
                </td>
                {cols.map((c) => {
                  const high =
                    c.high != null && (highs[c.high]?.games.includes(g.game_id) ?? false);
                  return (
                    <td key={c.label} style={high ? { color: C.gold, fontWeight: 700 } : undefined}>
                      {c.cell(g)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── One game ────────────────────────────────────────────────────────────────
function StepButton({
  label,
  glyph,
  onClick,
}: {
  label: string;
  glyph: string;
  onClick: (() => void) | null;
}) {
  return (
    <button
      type="button"
      onClick={onClick ?? undefined}
      disabled={!onClick}
      aria-label={label}
      title={label}
      style={{
        width: 34,
        height: 34,
        flexShrink: 0,
        padding: 0,
        borderRadius: 4,
        border: `1px solid ${C.edge}`,
        background: "transparent",
        color: onClick ? C.dim : "rgba(240,240,240,0.15)",
        cursor: onClick ? "pointer" : "default",
        fontSize: 20,
        lineHeight: "30px",
      }}
    >
      {glyph}
    </button>
  );
}

function GameHeader({
  g,
  onOlder,
  onNewer,
}: {
  g: GameRow;
  onOlder: (() => void) | null;
  onNewer: (() => void) | null;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 10,
        marginBottom: 14,
      }}
    >
      <StepButton label="Older game" glyph="‹" onClick={onOlder} />
      <div style={{ textAlign: "center", minWidth: 0 }}>
        <div
          style={{
            fontFamily: F.mono,
            fontSize: 9.5,
            letterSpacing: "0.2em",
            color: C.muted,
            textTransform: "uppercase",
          }}
        >
          {longDate(g.date)}
        </div>
        <div
          style={{
            fontFamily: F.display,
            fontSize: 28,
            lineHeight: 1.1,
            letterSpacing: "0.04em",
            display: "flex",
            gap: "0 14px",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <span style={{ whiteSpace: "nowrap" }}>
            <span style={{ color: teamColor(g.team) }}>{g.team}</span>
            <span style={{ color: C.muted, fontSize: 18 }}>{g.home ? " VS " : " @ "}</span>
            <span style={{ color: teamColor(g.opponent) }}>{g.opponent}</span>
          </span>
          <Result g={g} />
        </div>
      </div>
      <StepButton label="Newer game" glyph="›" onClick={onNewer} />
    </div>
  );
}

function StatTile({
  label,
  value,
  note,
  high = false,
}: {
  label: string;
  value: ReactNode;
  note: ReactNode;
  high?: boolean;
}) {
  return (
    <div
      style={{
        background: C.raised,
        border: `1px solid ${high ? C.gold : C.edge}`,
        borderRadius: 4,
        padding: "8px 10px 7px",
        minWidth: 0,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
        <span style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.22em", color: C.muted }}>
          {label}
        </span>
        {high && (
          <span
            title="Season high"
            style={{ fontFamily: F.mono, fontSize: 8, letterSpacing: "0.16em", color: C.gold }}
          >
            HIGH
          </span>
        )}
      </div>
      <div
        style={{
          fontFamily: F.display,
          fontSize: 30,
          lineHeight: 1.05,
          marginTop: 2,
          color: high ? C.gold : C.text,
        }}
      >
        {value}
      </div>
      <div
        style={{
          ...NUM,
          fontSize: 9.5,
          color: C.muted,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {note}
      </div>
    </div>
  );
}

/** "−2.6 vs 26.6 avg": the game against the season average, green where it
 *  beat it and red where it fell short. */
function VsAvg({
  value,
  avg,
  lowerBetter = false,
  neutral = false,
  signedAvg = false,
}: {
  value: number | null | undefined;
  avg: number | null | undefined;
  lowerBetter?: boolean;
  /** Minutes: more or fewer is neither good nor bad. */
  neutral?: boolean;
  signedAvg?: boolean;
}) {
  if (value == null || avg == null) return <>—</>;
  const d = value - avg;
  const color = neutral || Math.abs(d) < 0.05 ? C.muted : d > 0 !== lowerBetter ? WIN : LOSS;
  return (
    <>
      <span style={{ color }}>{signed(d, 1)}</span> vs {signedAvg ? signed(avg, 1) : avg.toFixed(1)}{" "}
      avg
    </>
  );
}

function PctVsAvg({ value, avg }: { value: number | null; avg: number | null }) {
  const color =
    value == null || avg == null || Math.abs(value - avg) < 0.0005
      ? C.muted
      : value > avg
        ? WIN
        : LOSS;
  return (
    <>
      <span style={{ color }}>{pct(value)}</span> · {pct(avg)} avg
    </>
  );
}

function PlayerTiles({
  g,
  avg,
  highs,
}: {
  g: GameRow;
  avg: GameAverages;
  highs: GameLog["highs"];
}) {
  const high = (key: string) => highs[key]?.games.includes(g.game_id) ?? false;
  const stat = (label: string, key: "pts" | "reb" | "ast" | "stl" | "blk" | "tov" | "pf") => (
    <StatTile
      key={label}
      label={label}
      value={g[key] ?? "—"}
      high={high(key)}
      note={<VsAvg value={g[key]} avg={avg[key]} lowerBetter={key === "tov" || key === "pf"} />}
    />
  );
  const shooting = (
    label: string,
    made: number | null,
    att: number | null,
    value: number | null,
    average: number | null,
    isHigh = false,
  ) => (
    <StatTile
      key={label}
      label={label}
      value={madeOf(made, att)}
      high={isHigh}
      note={<PctVsAvg value={value} avg={average} />}
    />
  );
  // 140 px tiles give 2, 3 or 4 to a row at every width the column takes
  // (it tops out near 700 px), and twelve tiles fill each of those evenly.
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 8,
      }}
    >
      {stat("PTS", "pts")}
      {stat("REB", "reb")}
      {stat("AST", "ast")}
      {shooting("FG", g.fgm, g.fga, g.fg_pct, avg.fg_pct)}
      {shooting("3P", g.fg3m, g.fg3a, g.fg3_pct, avg.fg3_pct, high("fg3m"))}
      {shooting("FT", g.ftm, g.fta, g.ft_pct, avg.ft_pct)}
      {stat("STL", "stl")}
      {stat("BLK", "blk")}
      {stat("TOV", "tov")}
      <StatTile
        label="MIN"
        value={g.min == null ? "—" : Math.round(g.min)}
        note={<VsAvg value={g.min} avg={avg.min} neutral />}
      />
      {stat("PF", "pf")}
      <StatTile
        label="+/−"
        value={signed(g.plus_minus)}
        note={<VsAvg value={g.plus_minus} avg={avg.plus_minus} signedAvg />}
      />
    </div>
  );
}

/** The game zone by zone, against the player's own season in each zone. */
function GameZones({ zones }: { zones: GameZone[] }) {
  return (
    <MiniTable
      headers={["Zone", "FG", "FG%", "Season FG%", "+/− pts"]}
      rows={zones.map((z) => {
        const gap =
          z.fg_pct != null && z.season_fg_pct != null ? (z.fg_pct - z.season_fg_pct) * 100 : null;
        return [
          ZONE_SHORT[z.zone] ?? z.zone,
          madeOf(z.fgm, z.fga),
          pct(z.fg_pct),
          pct(z.season_fg_pct),
          <span
            style={{ color: gap == null || Math.abs(gap) < 0.05 ? C.muted : gap > 0 ? WIN : LOSS }}
          >
            {signed(gap, 1)}
          </span>,
        ];
      })}
    />
  );
}

// Both teams' totals, side by side: each bar is that team's share of the
// two teams' combined figure, so the longer bar is the bigger number.
type CompareRow = {
  label: string;
  value: (g: GameRow) => number | null;
  show: (g: GameRow) => string;
  lowerBetter?: boolean;
};

const total = (
  label: string,
  key: "pts" | "reb" | "oreb" | "ast" | "stl" | "blk" | "tov" | "pf",
  lowerBetter = false,
): CompareRow => ({
  label,
  value: (g) => g[key],
  show: (g) => String(g[key] ?? "—"),
  lowerBetter,
});

type ShotCol = "fgm" | "fga" | "fg3m" | "fg3a" | "ftm" | "fta" | "fg_pct" | "fg3_pct" | "ft_pct";

const shootingRow = (label: string, m: ShotCol, a: ShotCol, p: ShotCol): CompareRow => ({
  label,
  value: (g) => g[p],
  show: (g) => `${madeOf(g[m], g[a])} · ${pct(g[p])}`,
});

const COMPARE: CompareRow[] = [
  total("Points", "pts"),
  shootingRow("Field goals", "fgm", "fga", "fg_pct"),
  shootingRow("Threes", "fg3m", "fg3a", "fg3_pct"),
  shootingRow("Free throws", "ftm", "fta", "ft_pct"),
  total("Rebounds", "reb"),
  total("Offensive rebounds", "oreb"),
  total("Assists", "ast"),
  total("Steals", "stl"),
  total("Blocks", "blk"),
  total("Turnovers", "tov", true),
  total("Fouls", "pf", true),
];

function TeamComparison({ g, o }: { g: GameRow; o: GameRow }) {
  const ends = { fontFamily: F.display, fontSize: 18, letterSpacing: "0.06em" } as const;
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 4,
        }}
      >
        <span style={{ ...ends, color: teamColor(g.team) }}>{g.team}</span>
        <span style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.3em", color: C.muted }}>
          TEAM STATS
        </span>
        <span style={{ ...ends, color: teamColor(o.team) }}>{o.team}</span>
      </div>
      {COMPARE.map((row) => {
        const a = row.value(g) ?? 0;
        const b = row.value(o) ?? 0;
        const sum = a + b;
        const better = a === b ? null : a > b !== !!row.lowerBetter ? "team" : "opponent";
        const side = (who: "team" | "opponent"): React.CSSProperties => ({
          ...NUM,
          fontSize: 12,
          color: better === who ? C.text : C.dim,
          fontWeight: better === who ? 700 : 400,
        });
        return (
          <div key={row.label} style={{ padding: "5px 0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
              <span style={side("team")}>{row.show(g)}</span>
              <span style={{ fontSize: 11, color: C.muted, textAlign: "center" }}>{row.label}</span>
              <span style={side("opponent")}>{row.show(o)}</span>
            </div>
            <div style={{ display: "flex", gap: 3, marginTop: 4 }}>
              {(["team", "opponent"] as const).map((who) => (
                <div
                  key={who}
                  style={{
                    flex: 1,
                    height: 4,
                    borderRadius: 2,
                    overflow: "hidden",
                    display: "flex",
                    justifyContent: who === "team" ? "flex-end" : "flex-start",
                    background: "rgba(255,255,255,0.06)",
                  }}
                >
                  <div
                    style={{
                      width: `${sum > 0 ? ((who === "team" ? a : b) / sum) * 100 : 0}%`,
                      background: teamColor(who === "team" ? g.team : o.team),
                      opacity: better === who ? 1 : 0.45,
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

const BOX_HEAD = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG", "3P", "FT", "+/−"];

function PlayersBox({
  players,
  team,
  onOpenPlayer,
}: {
  players: TeamGamePlayer[];
  team: string;
  onOpenPlayer?: (id: string) => void;
}) {
  const th: React.CSSProperties = { cursor: "default", position: "static" };
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="se-table" style={{ minWidth: 0 }}>
        <thead>
          <tr>
            <th className="se-left" style={th}>
              Player
            </th>
            {BOX_HEAD.map((h) => (
              <th key={h} style={th}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {players.map((p) => (
            <tr
              key={p.player_id}
              onClick={onOpenPlayer ? () => onOpenPlayer(p.player_id) : undefined}
              style={onOpenPlayer ? undefined : { cursor: "default" }}
            >
              <td className="se-name se-left" style={{ padding: "6px 12px" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                  <PlayerAvatar playerId={p.player_id} teamAbbrev={team} size={22} />
                  {onOpenPlayer ? (
                    <button type="button" style={ROW_BUTTON}>
                      {p.name}
                    </button>
                  ) : (
                    p.name
                  )}
                </span>
              </td>
              <td>{p.min == null ? "—" : Math.round(p.min)}</td>
              <td style={{ color: C.text }}>{p.pts ?? "—"}</td>
              <td>{p.reb ?? "—"}</td>
              <td>{p.ast ?? "—"}</td>
              <td>{p.stl ?? "—"}</td>
              <td>{p.blk ?? "—"}</td>
              <td>{p.tov ?? "—"}</td>
              <td>{madeOf(p.fgm, p.fga)}</td>
              <td>{madeOf(p.fg3m, p.fg3a)}</td>
              <td>{madeOf(p.ftm, p.fta)}</td>
              <td>{signed(p.plus_minus)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const quiet: React.CSSProperties = {
  fontFamily: F.mono,
  fontSize: 10.5,
  color: C.muted,
  letterSpacing: "0.2em",
  padding: "22px 0",
  textAlign: "center",
};

// ── The card ────────────────────────────────────────────────────────────────
export function GameLogCard({
  kind,
  id,
  season,
  onOpenPlayer,
}: {
  kind: "player" | "team";
  id: string;
  season: string;
  /** Team profiles: open a player from the game's box score. */
  onOpenPlayer?: (id: string) => void;
}) {
  const [type, setType] = useState<SeasonType>("regular");
  const [picked, setPicked] = useState<string | null>(null);
  const [side, setSide] = useState<Side>("offense");
  const [filter, setFilter] = useState<ShotFilter>("all");
  const detailRef = useRef<HTMLDivElement>(null);
  const base = `/stats/${kind}/${encodeURIComponent(id)}`;

  const log = useShotMap<GameLog>(
    `${base}/games?season=${encodeURIComponent(season)}&season_type=${type}`,
  );
  // The fetch hook still holds the last request's answer for a render after
  // the request changes: only trust a log that says it is this one.
  const owner = kind === "player" ? log.data?.player_id : log.data?.team_id;
  const current =
    log.data != null && owner === id && log.data.season === season && log.data.season_type === type
      ? log.data
      : null;
  const games = current?.games ?? [];
  const at = Math.max(
    0,
    games.findIndex((g) => g.game_id === picked),
  );
  const game: GameRow | null = games[at] ?? null;
  const highs = current && games.length >= MIN_GAMES_FOR_HIGHS ? current.highs : {};

  const fetched = useShotMap<Detail>(
    game ? `${base}/games/${game.game_id}${kind === "team" ? `?side=${side}` : ""}` : null,
  );
  // While the next game loads, the last one stays up, dimmed, rather than the
  // panel collapsing to a loading line and jolting the page.
  const [shown, setShown] = useState<Detail | null>(null);
  useEffect(() => {
    if (fetched.data) setShown(fetched.data);
  }, [fetched.data]);
  useEffect(() => {
    setPicked(null);
    setShown(null);
  }, [kind, id, season, type]);

  const loading =
    !shown ||
    !game ||
    shown.game.game_id !== game.game_id ||
    (kind === "team" && (shown as TeamGameDetail).side !== side);
  const replay = useMemo(
    () => shown && { shots: shown.shots, attempts: shown.attempts, games: shown.periods },
    [shown],
  );

  const pick = (gameId: string) => {
    setPicked(gameId);
    // Chosen from far down the list: bring the game up into view.
    const el = detailRef.current;
    if (el && el.getBoundingClientRect().top < HEADER_CLEARANCE) {
      const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      el.scrollIntoView({ block: "start", behavior: still ? "auto" : "smooth" });
    }
  };

  let body: ReactNode;
  if (log.error) {
    body = <div style={{ color: C.red, fontSize: 12 }}>{log.error}</div>;
  } else if (log.missing) {
    body = (
      <div style={quiet}>
        {season < FIRST_LOGGED_SEASON
          ? "GAME LOGS START IN 2016-17"
          : type === "playoffs"
            ? "NO PLAYOFF GAMES THIS SEASON"
            : "NO GAMES THIS SEASON"}
      </div>
    );
  } else if (!current || !game) {
    body = <div style={{ fontSize: 12, color: C.muted, padding: "18px 0" }}>Loading game log…</div>;
  } else {
    const team = kind === "team" ? (shown as TeamGameDetail | null) : null;
    const dim: React.CSSProperties = { opacity: loading ? 0.5 : 1, transition: "opacity 150ms" };
    const shooter = team?.side === "defense" ? game.opponent : game.team;
    body = (
      <>
        <div ref={detailRef} style={{ scrollMarginTop: HEADER_CLEARANCE }}>
          <GameHeader
            g={game}
            onOlder={at < games.length - 1 ? () => setPicked(games[at + 1].game_id) : null}
            onNewer={at > 0 ? () => setPicked(games[at - 1].game_id) : null}
          />
          <div
            style={{
              display: "flex",
              gap: 20,
              flexWrap: "wrap",
              alignItems: "flex-start",
              marginBottom: 16,
            }}
          >
            <div style={{ flex: "1 1 340px", maxWidth: 460, minWidth: 0 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 10,
                  flexWrap: "wrap",
                  marginBottom: 10,
                }}
              >
                {kind === "team" ? (
                  <Segmented
                    value={side}
                    onChange={(v) => setSide(v as Side)}
                    options={[
                      { value: "offense", label: "OFFENSE" },
                      { value: "defense", label: "DEFENSE" },
                    ]}
                  />
                ) : (
                  <span
                    style={{
                      fontFamily: F.mono,
                      fontSize: 9,
                      letterSpacing: "0.22em",
                      color: C.muted,
                    }}
                  >
                    EVERY SHOT, IN ORDER
                  </span>
                )}
                <Chips
                  value={filter}
                  onChange={(v) => setFilter(v as ShotFilter)}
                  options={[
                    { value: "all", label: "All" },
                    { value: "made", label: "Makes" },
                    { value: "missed", label: "Misses" },
                  ]}
                />
              </div>
              <div style={dim}>
                {replay ? (
                  <ShotReplay
                    data={replay}
                    filter={filter}
                    unit="period"
                    perShotMs={GAME_SHOT_MS}
                    label={`${kind === "team" ? `${shooter}'s shots` : "Every shot"}, ${longDate(game.date)} ${game.home ? "against" : "at"} ${game.opponent}, replayed in order`}
                  />
                ) : (
                  <div
                    style={{
                      ...quiet,
                      aspectRatio: `${COURT_CW} / ${COURT_AREA_H + LEGEND_H}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      padding: 0,
                      background: C.raised,
                      borderRadius: 4,
                    }}
                  >
                    {fetched.error ??
                      (fetched.missing ? "NO SHOTS FOR THIS GAME" : "LOADING SHOTS…")}
                  </div>
                )}
              </div>
              {shown && !loading && shown.attempts === 0 && (
                <div style={{ fontSize: 11, color: C.faint, marginTop: 8, lineHeight: 1.5 }}>
                  {game.fga
                    ? "This game's shot locations aren't in the data."
                    : "No field goal attempts this game."}
                </div>
              )}
              {team && !loading && (
                <div style={{ ...NUM, fontSize: 10.5, color: C.muted, marginTop: 8 }}>
                  {shooter} shots{team.side === "defense" ? `, against ${game.team}` : ""}
                </div>
              )}
            </div>
            <div style={{ flex: "1 1 300px", minWidth: 0 }}>
              {kind === "player" ? (
                <>
                  <PlayerTiles g={game} avg={current.averages} highs={highs} />
                  {shown && (
                    <div style={{ ...dim, marginTop: 14 }}>
                      <GameZones zones={shown.zones} />
                    </div>
                  )}
                </>
              ) : team?.opponent_box ? (
                <div style={dim}>
                  <TeamComparison g={team.game} o={team.opponent_box} />
                </div>
              ) : (
                <div style={{ fontSize: 12, color: C.muted, padding: "18px 0" }}>
                  Loading box score…
                </div>
              )}
            </div>
          </div>
          {team && team.players.length > 0 && (
            <div style={{ ...dim, marginBottom: 18 }}>
              <div
                style={{
                  fontFamily: F.mono,
                  fontSize: 9,
                  letterSpacing: "0.3em",
                  color: C.muted,
                  marginBottom: 4,
                }}
              >
                BOX SCORE · {team.game.team}
              </div>
              <PlayersBox
                players={team.players}
                team={team.game.team}
                onOpenPlayer={onOpenPlayer}
              />
            </div>
          )}
        </div>
        <LogTable
          games={games}
          highs={highs}
          cols={kind === "player" ? PLAYER_COLS : TEAM_COLS}
          picked={game.game_id}
          onPick={pick}
        />
      </>
    );
  }

  return (
    <Card style={{ padding: "16px 20px 18px", marginBottom: 16 }}>
      <SectionLabel>Game log</SectionLabel>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          margin: "10px 0 14px",
        }}
      >
        <div style={{ fontSize: 11, color: C.faint, lineHeight: 1.5, maxWidth: 640 }}>
          Every game of {season}, newest first. Pick one to see{" "}
          {kind === "player"
            ? "the box score against the season average and every shot, replayed in order"
            : "both teams' totals, the box score and every shot replayed — theirs, or their opponent's"}
          . <span style={{ color: C.gold }}>Gold</span> marks a season high.
        </div>
        <Segmented
          value={type}
          onChange={(v) => setType(v as SeasonType)}
          options={[
            { value: "regular", label: "REGULAR SEASON" },
            { value: "playoffs", label: "PLAYOFFS" },
          ]}
        />
      </div>
      {body}
    </Card>
  );
}
