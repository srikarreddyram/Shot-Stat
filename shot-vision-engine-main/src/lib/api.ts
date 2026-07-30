import { Player, ZoneResult } from "./shot-vision-data";

function calculateOffRtg(p: any, fg: number, tp: number): number {
  let score = 65; // Base score
  
  // Shooting
  score += Math.min(12, Math.max(-5, (fg - 43) * 0.8));
  if (tp > 0) score += Math.min(10, Math.max(-3, (tp - 32) * 0.6));
  
  // Playmaking (Assists - Turnovers)
  if (p.ast != null) score += (p.ast * 2.0);
  if (p.tov != null) score -= (p.tov * 1.2);
  
  // Free Throws
  if (p.ft_pct != null) {
    score += Math.min(6, Math.max(-3, ((p.ft_pct * 100) - 75) * 0.2));
  }
  
  // Rim Finishing
  if (p.rim_pct != null) {
    score += Math.min(6, Math.max(-3, ((p.rim_pct * 100) - 60) * 0.2));
  }
  
  // Mid-Range
  if (p.mid_pct != null) {
    score += Math.min(5, Math.max(-2, ((p.mid_pct * 100) - 40) * 0.15));
  }
  
  return Math.max(68, Math.min(99, Math.round(score)));
}

function calculateDefRtgOvr(p: any): number {
  const name = (p.name || p.player_name || "").toLowerCase();
  if (name.includes("wembanyama") || name.includes("gobert") || name.includes("davis") || name.includes("adebayo") || name.includes("antetokounmpo")) return 97;
  if (name.includes("leonard") || name.includes("caruso") || name.includes("holiday") || name.includes("smart") || name.includes("jones")) return 94;
  if (name.includes("doncic") || name.includes("trae") || name.includes("brunson")) return 65; // User wanted Luka low on defense
  
  let score = 75;
  if (p.contest_rate != null) {
    // pct_plusminus logic. Negative is better!
    score += Math.min(12, Math.max(-8, (p.contest_rate * 100) * -1.5));
  }
  return Math.max(60, Math.min(89, Math.round(score)));
}

function mapBackendPlayer(backendPlayer: any): Player {
  const fg = backendPlayer.career_fg_pct != null ? Math.round(backendPlayer.career_fg_pct * 100) : 45;
  const tp = backendPlayer.career_3p_pct != null ? Math.round(backendPlayer.career_3p_pct * 100) : 25;
  
  const offRtg = calculateOffRtg(backendPlayer, fg, tp);
  const defRtgOvr = calculateDefRtgOvr(backendPlayer);

  let contestStr = "0.0%";
  if (backendPlayer.contest_rate != null) {
    const diff = backendPlayer.contest_rate * 100;
    contestStr = (diff > 0 ? "+" : "") + diff.toFixed(1) + "%";
  }

  return {
    id: backendPlayer.player_id,
    name: backendPlayer.name || backendPlayer.player_name,
    pos: backendPlayer.position || "N/A",
    heightIn: backendPlayer.height || 78,
    weightLbs: backendPlayer.weight || 200,
    wingspanIn: backendPlayer.wingspan || (backendPlayer.height ? backendPlayer.height + 4 : 82),
    offRtg,
    defRtgOvr,
    fg: fg,
    tp: tp,
    rim: fg + 10, // heuristic for rim finishing
    defRtg: backendPlayer.def_rating != null ? Math.round(backendPlayer.def_rating * 100) : 46,
    contest: contestStr,
    ast: backendPlayer.ast,
    tov: backendPlayer.tov,
    ftPct: backendPlayer.ft_pct,
    rimPct: backendPlayer.rim_pct,
    midPct: backendPlayer.mid_pct,
  };
}

const API_BASE = "http://127.0.0.1:8000";

export async function searchPlayersAPI(query: string, season = "2024-25"): Promise<Player[]> {
  if (!query || query.length < 2) return [];
  const res = await fetch(`${API_BASE}/players/search?q=${encodeURIComponent(query)}&season=${season}`);
  if (!res.ok) throw new Error("Failed to search players");
  const data = await res.json();
  return data.map(mapBackendPlayer);
}

export async function getPlayerAPI(id: string, season = "2023-24"): Promise<Player> {
  const res = await fetch(`${API_BASE}/player/${id}?season=${season}`);
  if (!res.ok) throw new Error("Failed to fetch player");
  const data = await res.json();
  return mapBackendPlayer(data);
}

export async function getRecommendationHeatmapAPI(
  attackerId: string,
  defenderId: string,
  quarter: number,
  timeRemaining: number,
  scoreDiff: number,
  homeAway: number
) {
  const body = {
    player_id: attackerId,
    defender_id: defenderId,
    season: "2023-24",
    quarter: quarter,
    time_remaining: timeRemaining,
    score_diff: scoreDiff,
    home_away: homeAway,
    playoff_flag: 0,
  };
  
  const res = await fetch(`${API_BASE}/recommend/summary`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error("Failed to get heatmap recommendations");
  return await res.json();
}
