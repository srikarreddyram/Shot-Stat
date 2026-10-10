"""Builds the one-row-per-team stat table and one team's full profile
(its real stat plus its roster). See build_team_stat_table's docstring for
why the roster-derived columns are prefixed roster_avg_ rather than
presented as separately-measured team numbers."""
from __future__ import annotations

import pandas as pd

from src.inference.player_ratings import resolve_rating_season
from .formatting import _clean
from .players import build_player_stat_table


def build_team_stat_table(engine, season: str | None = None) -> pd.DataFrame:
    """
    One row per team: the measured team-level stats this project tracks —
    def_rating from team_stats, and the NBA's own team advanced stats
    (adv_*, from team_advanced_stats) — plus honest ROSTER-DERIVED
    aggregates: the mean of each roster's own player-level stats, labelled
    roster_avg_ because they are not separately-collected team numbers.
    Fabricating "team stats" that are secretly just player averages under a
    different name would be the same mistake the old frontend rating
    formula made — presenting an invented number as if it were measured.
    """
    season = resolve_rating_season(engine, season or "2026-27")

    teams = pd.read_sql(
        "SELECT team_id, team_name, team_abbrev, def_rating FROM team_stats WHERE season = ?",
        engine, params=(season,),
    )

    players = build_player_stat_table(engine, season)
    if players.empty or teams.empty:
        return teams

    roster_avg_cols = [
        "season_fg_pct", "off_rating", "def_rating_ours",
        "self_creation_index", "playmaking_gravity", "rim_pressure",
        "def_fg_pct_allowed", "blk_per_min", "stl_per_min", "deflections_per_min",
    ]
    present = [c for c in roster_avg_cols if c in players.columns]
    agg = players.groupby("team_id")[present].mean().add_prefix("roster_avg_")
    agg["roster_size"] = players.groupby("team_id").size()

    teams = teams.merge(agg, left_on="team_id", right_index=True, how="left")

    advanced = pd.read_sql(
        "SELECT team_id, off_rating AS adv_off_rating, net_rating AS adv_net_rating, pace AS adv_pace, "
        "pie AS adv_pie, ts_pct AS adv_ts_pct, efg_pct AS adv_efg_pct, ast_pct AS adv_ast_pct, "
        "tov_pct AS adv_tov_pct, oreb_pct AS adv_oreb_pct, dreb_pct AS adv_dreb_pct "
        "FROM team_advanced_stats WHERE season = ? AND season_type = 'Regular Season'",
        engine, params=(season,),
    )
    teams = teams.merge(advanced, on="team_id", how="left")
    teams["season"] = season
    return teams


def team_full_profile(engine, team_id: str, season: str | None = None) -> dict | None:
    """A team's real stat plus its roster, each player carrying their own
    full stat row — a team profile IS its roster here, not a separate
    invented team-level number set. See build_team_stat_table's docstring."""
    season = resolve_rating_season(engine, season or "2026-27")

    team_row = pd.read_sql(
        "SELECT team_id, team_name, team_abbrev, def_rating FROM team_stats "
        "WHERE season = ? AND team_id = ?",
        engine, params=(season, str(team_id)),
    )

    players = build_player_stat_table(engine, season)
    roster = players[players["team_id"] == str(team_id)] if not players.empty else players

    if team_row.empty and roster.empty:
        return None

    return {
        "team_id": str(team_id),
        "team_name": team_row.iloc[0]["team_name"] if not team_row.empty else None,
        "team_abbrev": team_row.iloc[0]["team_abbrev"] if not team_row.empty else None,
        "def_rating": _clean(team_row.iloc[0]["def_rating"]) if not team_row.empty else None,
        "season": season,
        "roster": [
            {"player_id": r["player_id"], "name": r["name"], "position": _clean(r.get("position")),
             "off_rating": _clean(r.get("off_rating")), "def_rating_ours": _clean(r.get("def_rating_ours"))}
            for _, r in roster.sort_values("name").iterrows()
        ] if not roster.empty else [],
    }
