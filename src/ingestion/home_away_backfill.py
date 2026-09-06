"""
Home/Away Backfill — Fixes the `home_away` column for all existing shots.

The original shot_ingestor.py had a bug where it compared full team names
("Los Angeles Lakers") against HTM/VTM abbreviations ("LAL"/"PHX"), so
home_away was always NULL.

Why the first fix at this file was incomplete
-----------------------------------------------
The original backfill built one team per (player, season) from
LeagueDashPlayerStats — a SEASON-LEVEL aggregate. A player traded mid-season
has games for two different teams within one season, so a single season-level
team assignment cannot resolve either stint correctly; the old code detected
the mismatch and left those shots NULL rather than guess.

That was not a rare edge case. It reproduced every season: roughly 12-18k
shots per year stayed NULL, evenly spread across all ten seasons in the
training window, which is exactly the volume trades produce every year, not a
one-off data gap. Measured directly: 146,623 of 2,253,739 shots from 2016-17
onward, about 6.5% of the training window.

The fix
-------
LeagueGameLog with player_or_team_abbreviation='P' (rather than the team-level
'T' game_ingestor.py already uses) returns one row per (player, GAME) with
that player's team abbreviation for THAT specific game. A trade is not
ambiguous at this granularity — the player has different rows, with different
teams, for games before and after it. One API call per season per season-type,
same cost pattern as the team-level version, and it eliminates the entire
class of "traded mid-season" failure rather than working around it.

Usage:
    python -m src.ingestion.home_away_backfill
"""
import sys
import time
from pathlib import Path

from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine


def _pull_player_game_teams(seasons: list[str]) -> dict:
    """
    (player_id, game_id) -> team_abbreviation, from LeagueGameLog at the
    PLAYER level. One row per player per game actually played, so a trade
    simply produces different rows with different teams rather than an
    ambiguous season-level answer.
    """
    from nba_api.stats.endpoints import leaguegamelog

    mapping: dict[tuple[str, str], str] = {}
    for season in tqdm(seasons, desc="  Pulling player game logs"):
        for season_type in ("Regular Season", "Playoffs"):
            try:
                lg = leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation="P",
                    season_type_all_star=season_type,
                )
                df = lg.get_data_frames()[0]
                for _, row in df.iterrows():
                    key = (str(row["PLAYER_ID"]), str(row["GAME_ID"]))
                    mapping[key] = row["TEAM_ABBREVIATION"]
            except Exception as e:
                print(f"\n  ⚠ Failed for {season} {season_type}: {e}")
            time.sleep(config.REQUEST_DELAY)
    return mapping


def backfill_home_away():
    engine = get_engine()

    print(f"\n{'='*60}")
    print("  HOME/AWAY BACKFILL (per-game, trade-safe)")
    print(f"{'='*60}")

    with engine.connect() as conn:
        null_count = conn.execute(
            text("SELECT COUNT(*) FROM shots WHERE home_away IS NULL")
        ).scalar()
        seasons = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT season FROM shots WHERE home_away IS NULL ORDER BY season"
        ))]
    print(f"  → {null_count:,} shots need home_away, across {len(seasons)} seasons")
    if null_count == 0:
        print("  ✓ Nothing to backfill!")
        return 0

    player_game_team = _pull_player_game_teams(seasons)
    print(f"  ✓ {len(player_game_team):,} (player, game) team assignments pulled")

    game_map: dict[str, tuple[str, str]] = {}
    with engine.connect() as conn:
        for row in conn.execute(text("SELECT game_id, home_team, away_team FROM games")):
            game_map[row[0]] = (row[1], row[2])
    print(f"  ✓ Loaded {len(game_map):,} games")

    total_updated = 0
    total_unresolved = 0

    with engine.begin() as conn:
        result = conn.execute(text(
            "SELECT shot_id, game_id, player_id FROM shots WHERE home_away IS NULL"
        ))
        batch = []
        for shot_id, game_id, player_id in tqdm(
            result, total=null_count, desc="  Computing home_away"
        ):
            game_info = game_map.get(game_id)
            team_abbr = player_game_team.get((player_id, game_id))

            home_away = None
            if game_info and team_abbr:
                home_team, away_team = game_info
                if team_abbr == home_team:
                    home_away = 1
                elif team_abbr == away_team:
                    home_away = 0

            if home_away is not None:
                batch.append({"sid": shot_id, "ha": home_away})
            else:
                total_unresolved += 1

            if len(batch) >= 50_000:
                for item in batch:
                    conn.execute(
                        text("UPDATE shots SET home_away = :ha WHERE shot_id = :sid"), item
                    )
                total_updated += len(batch)
                batch = []

        if batch:
            for item in batch:
                conn.execute(
                    text("UPDATE shots SET home_away = :ha WHERE shot_id = :sid"), item
                )
            total_updated += len(batch)

    print(f"\n{'='*60}")
    print("  HOME/AWAY BACKFILL COMPLETE")
    print(f"  Updated:    {total_updated:,}")
    print(f"  Unresolved: {total_unresolved:,} (no matching game-log row — "
          f"pre-2016 game-log coverage or a data gap, not a trade)")
    print(f"{'='*60}\n")

    return total_updated


if __name__ == "__main__":
    backfill_home_away()
