// Game-by-game box scores for the Stat Engine profiles, and any one game
// opened up with every shot of it (src/inference/game_logs.py — box scores
// from 2016-17 on). Drawn by components/stat-engine/GameLog.tsx.

export type SeasonType = "regular" | "playoffs";

/** One game's box score line, for a player or a team. */
export type GameRow = {
  game_id: string;
  /** YYYY-MM-DD */
  date: string;
  team: string;
  opponent: string;
  home: boolean;
  win: boolean | null;
  /** The final score, from the team's side. */
  team_score?: number;
  opponent_score?: number;
  /** Players only. */
  min?: number | null;
  pts: number | null;
  reb: number | null;
  oreb: number | null;
  dreb: number | null;
  ast: number | null;
  stl: number | null;
  blk: number | null;
  tov: number | null;
  pf: number | null;
  fgm: number | null;
  fga: number | null;
  fg3m: number | null;
  fg3a: number | null;
  ftm: number | null;
  fta: number | null;
  plus_minus: number | null;
  fg_pct: number | null;
  fg3_pct: number | null;
  ft_pct: number | null;
};

/** Per-game averages over a season's games; shooting is total makes over
 *  total attempts. */
export type GameAverages = {
  gp: number;
  min?: number | null;
  pts: number | null;
  reb: number | null;
  ast: number | null;
  stl: number | null;
  blk: number | null;
  tov: number | null;
  pf: number | null;
  fg3m: number | null;
  fg_pct: number | null;
  fg3_pct: number | null;
  ft_pct: number | null;
  plus_minus: number | null;
};

export type GameLog = {
  player_id?: string;
  team_id?: string;
  season: string;
  season_type: SeasonType;
  /** Newest first. */
  games: GameRow[];
  averages: GameAverages;
  /** The season high in each stat, and the game(s) that reached it. */
  highs: Record<string, { value: number; games: string[] }>;
};

export type GameZone = {
  zone: string;
  fga: number;
  fgm: number;
  fg_pct: number | null;
  /** Players: the zone over the whole season (same season type). */
  season_fga?: number;
  season_fgm?: number;
  season_fg_pct?: number | null;
};

type GameShots = {
  season: string;
  game: GameRow;
  season_averages: GameAverages;
  /** [loc_x, loc_y, made, period] for every shot, in the order taken. */
  shots: [number, number, number, number][];
  attempts: number;
  makes: number;
  /** 4, or more with overtime. */
  periods: number;
  zones: GameZone[];
};

export type PlayerGameDetail = GameShots;

export type TeamGamePlayer = {
  player_id: string;
  name: string;
  min: number | null;
  pts: number | null;
  reb: number | null;
  ast: number | null;
  stl: number | null;
  blk: number | null;
  tov: number | null;
  fgm: number | null;
  fga: number | null;
  fg3m: number | null;
  fg3a: number | null;
  ftm: number | null;
  fta: number | null;
  plus_minus: number | null;
};

export type TeamGameDetail = GameShots & {
  /** Whose shots: the team's (offense) or its opponent's (defense). */
  side: "offense" | "defense";
  opponent_box: GameRow | null;
  players: TeamGamePlayer[];
};
