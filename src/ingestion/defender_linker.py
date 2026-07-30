"""
Defender Linker — Populates defender_id on the shots table.

For each shot, finds the primary defender (most matchup minutes) from the
matchups table and writes it to shots.defender_id.

Pure SQL — no API calls needed. Runs after matchup_ingestor.py.

Usage:
    python -m src.ingestion.defender_linker
"""
import sys
from pathlib import Path

from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine


def link_defenders():
    """Populate defender_id on shots from matchup data."""
    engine = get_engine()

    print(f"\n{'='*60}")
    print(f"  DEFENDER LINKER")
    print(f"{'='*60}")

    # Check current state
    with engine.connect() as conn:
        total_shots = conn.execute(text("SELECT COUNT(*) FROM shots")).scalar()
        already_linked = conn.execute(text(
            "SELECT COUNT(*) FROM shots WHERE defender_id IS NOT NULL"
        )).scalar()
        matchup_games = conn.execute(text(
            "SELECT COUNT(DISTINCT game_id) FROM matchups"
        )).scalar()
        eligible_shots = conn.execute(text("""
            SELECT COUNT(*) FROM shots s
            WHERE s.defender_id IS NULL
              AND EXISTS (SELECT 1 FROM matchups m WHERE m.game_id = s.game_id)
        """)).scalar()

    print(f"  → Total shots:       {total_shots:,}")
    print(f"  → Already linked:    {already_linked:,}")
    print(f"  → Games w/ matchups: {matchup_games:,}")
    print(f"  → Eligible to link:  {eligible_shots:,}")

    if eligible_shots == 0:
        print("  ✓ Nothing to link!")
        return 0

    # Process in chunks by game_id to avoid massive single UPDATE
    print(f"\n  → Linking defenders to shots...")

    with engine.connect() as conn:
        game_ids = conn.execute(text(
            "SELECT DISTINCT game_id FROM matchups ORDER BY game_id"
        )).fetchall()
        game_ids = [row[0] for row in game_ids]

    total_linked = 0

    with engine.begin() as conn:
        for game_id in tqdm(game_ids, desc="  Linking"):
            # For each shot in this game without a defender_id,
            # find the defender with the most matchup minutes
            result = conn.execute(text("""
                UPDATE shots
                SET defender_id = (
                    SELECT m.defense_player_id
                    FROM matchups m
                    WHERE m.game_id = shots.game_id
                      AND m.offense_player_id = shots.player_id
                    ORDER BY m.matchup_minutes DESC
                    LIMIT 1
                )
                WHERE shots.game_id = :gid
                  AND shots.defender_id IS NULL
                  AND EXISTS (
                    SELECT 1 FROM matchups m
                    WHERE m.game_id = shots.game_id
                      AND m.offense_player_id = shots.player_id
                  )
            """), {"gid": game_id})
            total_linked += result.rowcount

    # Final stats
    with engine.connect() as conn:
        final_linked = conn.execute(text(
            "SELECT COUNT(*) FROM shots WHERE defender_id IS NOT NULL"
        )).scalar()
        still_null = total_shots - final_linked

    print(f"\n{'='*60}")
    print(f"  DEFENDER LINKER COMPLETE")
    print(f"  Newly linked:  {total_linked:,}")
    print(f"  Total linked:  {final_linked:,} / {total_shots:,} ({final_linked/total_shots*100:.1f}%)")
    print(f"  Still NULL:    {still_null:,} (pre-2016 seasons or missing matchup data)")
    print(f"{'='*60}\n")

    return total_linked


if __name__ == "__main__":
    link_defenders()
