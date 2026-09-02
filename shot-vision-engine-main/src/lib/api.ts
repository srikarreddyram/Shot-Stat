import { Player, HeatmapResponse, MatchupResponse, HealthStatus, AttainabilityExplanation } from "./shot-vision-data";

// Ratings are computed by the backend (src/inference/player_ratings.py) as
// percentile ranks over the whole league, from measured data.
//
// They used to be invented here, from a stack of magic numbers:
//
//     let score = 65;
//     score += Math.min(12, Math.max(-5, (fg - 43) * 0.8));
//     if (p.ast != null) score += p.ast * 2.0;
//     ...
//     return Math.max(68, Math.min(99, Math.round(score)));
//
// Three problems, all of them visible on screen. `ast`, `tov` and `ft_pct` are
// NULL for every player in the database — never ingested — so the entire
// playmaking term evaluated to zero for everyone, which is why the best passer
// in the league rated 75 on offence. What remained was dominated by raw FG%,
// so the top-rated offensive players were end-of-bench centres who only shoot
// dunks. And both scales saturated against their clamps: 41% of the league sat
// exactly on the offensive floor of 68, and 69 players were tied at the
// defensive cap of 87 — so an "87" never meant a rating, it meant the ceiling.
//
// A missing rating stays null rather than falling back to a made-up number;
// the UI shows a dash.
const ratingOr = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

function mapBackendPlayer(backendPlayer: any): Player {
  const fg = backendPlayer.career_fg_pct != null ? Math.round(backendPlayer.career_fg_pct * 100) : 45;
  const tp = backendPlayer.career_3p_pct != null ? Math.round(backendPlayer.career_3p_pct * 100) : 25;
  
  const offRtg = ratingOr(backendPlayer.rating_off_rating);
  const defRtgOvr = ratingOr(backendPlayer.rating_def_rating);

  let contestStr = "0.0%";
  if (backendPlayer.def_plus_minus != null) {
    const diff = backendPlayer.def_plus_minus * 100;
    contestStr = (diff > 0 ? "+" : "") + diff.toFixed(1) + "%";
  }

  return {
    id: backendPlayer.player_id,
    name: backendPlayer.name || backendPlayer.player_name,
    pos: backendPlayer.position || "N/A",
    heightIn: backendPlayer.height ?? null,
    weightLbs: backendPlayer.weight ?? null,
    wingspanIn: backendPlayer.wingspan ?? null,
    offRtg,
    defRtgOvr,
    ratingComponents: {
      scoring: ratingOr(backendPlayer.rating_scoring),
      playmaking: ratingOr(backendPlayer.rating_playmaking),
      volume: ratingOr(backendPlayer.rating_volume),
      creation: ratingOr(backendPlayer.rating_creation),
      defence: ratingOr(backendPlayer.rating_defence),
    },
    fg: fg,
    tp: tp,
    rim: fg + 10, // heuristic for rim finishing
    defRtg: backendPlayer.def_fg_pct_allowed != null ? Math.round(backendPlayer.def_fg_pct_allowed * 100) : 46,
    contest: contestStr,
    ast: backendPlayer.ast,
    tov: backendPlayer.tov,
    ftPct: backendPlayer.ft_pct,
    rimPct: backendPlayer.rim_pct,
    midPct: backendPlayer.mid_pct,
    headshotUrl: headshotUrl(backendPlayer.player_id),
    statsSource: backendPlayer.stats_source,
    resolvedSeason: backendPlayer.resolved_season,
  };
}

// Backend origin. Override with VITE_API_BASE (e.g. in .env.local) when the
// API is not on its default port — a mismatch here reads as "engine offline"
// in the UI, since every call including /health simply fails to connect.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

// Headshots go through the backend rather than straight to cdn.nba.com.
// Chrome fails every direct CDN request on some networks with
// ERR_HTTP2_PROTOCOL_ERROR — curl fetches the identical URL without
// complaint — and since the avatar falls back to initials on error, the
// result looked deliberate instead of broken. The backend proxies and caches
// the PNG, so this works wherever the API itself works.
export function headshotUrl(playerId: string | undefined) {
  return playerId ? `${API_BASE}/headshot/${playerId}` : undefined;
}

// Backend errors (FastAPI HTTPException) come back as {"detail": "..."}.
// Surface that message to the UI instead of a generic "request failed".
async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    // response wasn't JSON — fall through to the generic message
  }
  return fallback;
}

export async function checkHealthAPI(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await extractErrorMessage(res, "Health check failed"));
  return res.json();
}

export async function searchPlayersAPI(query: string, season: string): Promise<Player[]> {
  if (!query || query.length < 2) return [];
  const res = await fetch(`${API_BASE}/players/search?q=${encodeURIComponent(query)}&season=${season}`);
  if (!res.ok) throw new Error(await extractErrorMessage(res, "Failed to search players"));
  const data = await res.json();
  return data.map(mapBackendPlayer);
}

export async function getPlayerAPI(id: string, season: string): Promise<Player> {
  const res = await fetch(`${API_BASE}/player/${id}?season=${season}`);
  if (!res.ok) throw new Error(await extractErrorMessage(res, "Failed to fetch player"));
  const data = await res.json();
  return mapBackendPlayer(data);
}

// Why a player can or cannot get a shot in a given zone. Fetched on demand —
// only when a user actually opens the attainability breakdown — rather than
// bundled into the heatmap response, which would mean computing six
// explanations per request that are usually never read.
export async function getAttainabilityExplanationAPI(
  playerId: string,
  zone: string,
  season: string,
  // Coordinates resolve the angle sub-zone: "Above the Break 3" splits into a
  // dead-centre and a wing half, which have materially different attainability.
  // Omitting them collapses back to the blended parent zone.
  locX?: number,
  locY?: number
): Promise<AttainabilityExplanation> {
  const loc = locX != null && locY != null ? `&loc_x=${locX}&loc_y=${locY}` : "";
  const res = await fetch(
    `${API_BASE}/explain/attainability/${playerId}?zone=${encodeURIComponent(zone)}&season=${season}${loc}`
  );
  if (!res.ok) throw new Error(await extractErrorMessage(res, "Failed to explain attainability"));
  return res.json();
}

export async function getMatchupAPI(
  attackerId: string,
  defenderId: string,
  season: string
): Promise<MatchupResponse> {
  const res = await fetch(`${API_BASE}/matchup/${attackerId}/${defenderId}?season=${season}`);
  if (!res.ok) throw new Error(await extractErrorMessage(res, "Failed to fetch matchup data"));
  return res.json();
}

// Fetches the full dense shot-location grid (~180 points) plus zone rollups
// and shot-volume-by-zone. Powers both the real court heatmap and the
// zone summary cards.
export async function getRecommendationHeatmapAPI(
  attackerId: string,
  defenderId: string,
  season: string,
  quarter: number,
  timeRemaining: number,
  scoreDiff: number,
  homeAway: number,
  secondaryDefenderId?: string | null
): Promise<HeatmapResponse> {
  const body = {
    player_id: attackerId,
    defender_id: defenderId,
    secondary_defender_id: secondaryDefenderId ?? null,
    season: season,
    quarter: quarter,
    time_remaining: timeRemaining,
    score_diff: scoreDiff,
    home_away: homeAway,
    playoff_flag: 0,
  };

  const res = await fetch(`${API_BASE}/recommend/heatmap`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(await extractErrorMessage(res, "Failed to get shot recommendations"));
  return res.json();
}
