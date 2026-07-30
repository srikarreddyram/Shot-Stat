"""
Score Differential Backfill — Populates `score_diff` for every shot.

Uses PlayByPlayV3 to get the running score at the time of each shot.
For each game, we:
  1. Pull play-by-play data (scoreHome, scoreAway per event)
  2. Forward-fill scores to get the running score at every event
  3. Match shots by (game_id, period, clock) to get the score at shot time
  4. Compute score_diff = shooter's team score - opponent score

This requires ~20,000+ API calls (one per game), so it's designed to be
run overnight. It's fully resumable — tracks completed games.

Usage:
    python -m src.ingestion.score_diff_backfill                  # All games
    python -m src.ingestion.score_diff_backfill --season 2023-24 # One season
"""
import sys
import time
import re
from pathlib import Path
from collections import defaultdict

from nba_api.stats.endpoints import playbyplayv3
from nba_api.stats.static import teams as nba_teams_static
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine


# Build team_id -> abbreviation map
_TEAM_ID_TO_ABBR = {t['id']: t['abbreviation'] for t in nba_teams_static.get_teams()}


def _parse_clock(clock_str: str) -> float:
    """Convert 'PT11M42.00S' to seconds remaining (e.g. 702.0)."""
    if not clock_str:
        return 0.0
    match = re.match(r'PT(\d+)M([\d.]+)S', clock_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return 0.0


def _get_completed_games(engine):
    """Get set of game_ids that already have score_diff populated."""
    with engine.connect() as conn:
        # A game is "done" if ANY shot in it has a non-null score_diff
        result = conn.execute(text(
            "SELECT DISTINCT game_id FROM shots WHERE score_diff IS NOT NULL"
        ))
        return set(row[0] for row in result)


def _get_games_to_process(engine, season=None):
    """Get list of game_ids that need score_diff, optionally filtered by season."""
    with engine.connect() as conn:
        if season:
            result = conn.execute(text(
                "SELECT DISTINCT s.game_id FROM shots s "
                "WHERE s.score_diff IS NULL AND s.season = :season"
            ), {"season": season})
        else:
            result = conn.execute(text(
                "SELECT DISTINCT s.game_id FROM shots s WHERE s.score_diff IS NULL"
            ))
        return [row[0] for row in result]


def _get_player_team_for_game(engine, game_id):
    """
    For a given game, figure out which team each player is on.
    Uses the shots table + games table: if a player's shot is in a game,
    and we know home_away, we can derive their team.
    Fallback: use the play-by-play data's teamId field.
    """
    # We'll populate this from PBP data instead (more reliable)
    return {}


def _process_game(engine, game_id):
    """
    Process a single game: pull PBP, compute score_diff for each shot.
    Returns number of shots updated.
    """
    # Pull play-by-play
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        df = pbp.get_data_frames()[0]
    except Exception as e:
        return -1, str(e)

    if df.empty:
        return 0, None

    # Get home/away teams from games table
    with engine.connect() as conn:
        game_row = conn.execute(text(
            "SELECT home_team, away_team FROM games WHERE game_id = :gid"
        ), {"gid": game_id}).fetchone()

    if not game_row:
        return 0, "Game not in games table"

    home_team, away_team = game_row

    # Build running score from PBP
    # scoreHome/scoreAway are only populated on scoring events
    # Forward-fill to get running score at every event
    current_home = 0
    current_away = 0

    # Build: (period, clock_seconds) -> (home_score, away_score) at that moment
    # Also build: personId -> teamTricode mapping from PBP events
    player_team_map = {}  # personId -> team abbreviation
    event_scores = []  # list of (period, clock_secs, home_score, away_score)

    for _, row in df.iterrows():
        # Update player -> team mapping from any event with a teamTricode
        person_id = row.get("personId")
        team_tri = row.get("teamTricode")
        if person_id and team_tri and str(person_id) != "0":
            player_team_map[str(int(person_id))] = str(team_tri)

        # Update running score
        sh = row.get("scoreHome")
        sa = row.get("scoreAway")
        if sh is not None and sh != "" and sh != 0:
            try:
                current_home = int(sh)
            except (ValueError, TypeError):
                pass
        if sa is not None and sa != "" and sa != 0:
            try:
                current_away = int(sa)
            except (ValueError, TypeError):
                pass

        period = row.get("period")
        clock = row.get("clock", "")
        clock_secs = _parse_clock(str(clock))

        event_scores.append({
            "period": period,
            "clock": clock_secs,
            "home": current_home,
            "away": current_away,
        })

    # Build lookup: (period, clock_rounded) -> (home_score, away_score)
    # Use the LAST event at each (period, clock) for the most up-to-date score
    score_lookup = {}
    for ev in event_scores:
        key = (ev["period"], round(ev["clock"], 1))
        score_lookup[key] = (ev["home"], ev["away"])

    # Now update shots for this game
    with engine.begin() as conn:
        shots = conn.execute(text(
            "SELECT shot_id, player_id, quarter, time_remaining "
            "FROM shots WHERE game_id = :gid"
        ), {"gid": game_id}).fetchall()

        updated = 0
        for shot in shots:
            shot_id, player_id, quarter, time_remaining = shot

            if quarter is None or time_remaining is None:
                continue

            # Find the score at this moment
            # Try exact match first, then nearest
            key = (quarter, round(time_remaining, 1))
            score = score_lookup.get(key)

            if score is None:
                # Find closest event in same period
                best_diff = float('inf')
                for (p, c), (h, a) in score_lookup.items():
                    if p == quarter:
                        diff = abs(c - time_remaining)
                        if diff < best_diff:
                            best_diff = diff
                            score = (h, a)

            if score is None:
                continue

            home_score, away_score = score

            # Determine if this player is home or away
            player_team = player_team_map.get(str(player_id))

            if player_team:
                if player_team == home_team:
                    score_diff = home_score - away_score
                elif player_team == away_team:
                    score_diff = away_score - home_score
                else:
                    continue  # Can't determine
            else:
                continue

            conn.execute(
                text("UPDATE shots SET score_diff = :sd WHERE shot_id = :sid"),
                {"sd": score_diff, "sid": shot_id}
            )
            updated += 1

    return updated, None


def backfill_score_diff(season=None):
    """Main entry point for score_diff backfill."""
    engine = get_engine()

    print(f"\n{'='*60}")
    print(f"  SCORE DIFFERENTIAL BACKFILL")
    if season:
        print(f"  Season: {season}")
    else:
        print(f"  All seasons")
    print(f"{'='*60}")

    # Get games needing processing
    completed = _get_completed_games(engine)
    all_games = _get_games_to_process(engine, season)

    # Filter out already completed
    games_to_do = [g for g in all_games if g not in completed]

    print(f"  → {len(all_games):,} games have NULL score_diff")
    print(f"  → {len(completed):,} games already completed")
    print(f"  → {len(games_to_do):,} games to process")
    print(f"  → Estimated time: ~{len(games_to_do) * 0.8 / 60:.0f} minutes")
    print()

    if not games_to_do:
        print("  ✓ Nothing to do!")
        return

    total_updated = 0
    total_errors = 0

    for game_id in tqdm(games_to_do, desc="  Processing games"):
        updated, error = _process_game(engine, game_id)

        if updated == -1:
            total_errors += 1
            if total_errors <= 5:
                tqdm.write(f"  ⚠ {game_id}: {error}")
        else:
            total_updated += updated

        time.sleep(config.REQUEST_DELAY)

    print(f"\n{'='*60}")
    print(f"  SCORE DIFF BACKFILL COMPLETE")
    print(f"  Shots updated:  {total_updated:,}")
    print(f"  Games failed:   {total_errors:,}")
    print(f"{'='*60}\n")

    return total_updated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill score_diff from play-by-play data.")
    parser.add_argument("--season", type=str, default=None,
                        help="Process a specific season only (e.g. '2023-24')")
    args = parser.parse_args()

    backfill_score_diff(season=args.season)
