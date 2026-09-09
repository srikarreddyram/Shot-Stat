"""
Shot-event simulator — the producer half of the Kafka demo.

There is no live NBA play-by-play feed this project is authorized to
consume in real time. This replays REAL historical shots from the existing
database, in their real order, at a configurable pace, onto the
`shot-events` topic. It is a stand-in for a live feed, not one — see
docs/streaming.md.

Usage:
    python -m src.streaming.producer                       # auto-picks a game
    python -m src.streaming.producer --game-id 0042500404
    python -m src.streaming.producer --game-id 0042500404 --pace 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine
from src.streaming.config import (
    DEFAULT_PACE_SECONDS,
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_SHOT_EVENTS,
)


def _pick_demo_game(engine, season: str | None) -> str:
    """
    A real game with full defender coverage and enough shots to be worth
    watching — picked automatically so the demo has a sane default with no
    arguments.
    """
    season_clause = "AND s.season = :season" if season else ""
    query = text(f"""
        SELECT s.game_id, COUNT(*) AS n
        FROM shots s
        WHERE s.defender_id IS NOT NULL {season_clause}
        GROUP BY s.game_id
        HAVING n > 150
        ORDER BY s.game_id DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        params = {"season": season} if season else {}
        row = conn.execute(query, params).fetchone()
    if row is None:
        raise RuntimeError(
            "No game with full defender coverage found — pass --game-id explicitly."
        )
    return row[0]


def load_game_shots(engine, game_id: str) -> pd.DataFrame:
    """
    Real shots for one game, in the order they actually happened: descending
    quarter time (the game clock counts down), then quarter ascending.
    """
    query = text("""
        SELECT s.shot_id, s.game_id, s.season, s.player_id, s.defender_id,
               s.shot_made, s.loc_x, s.loc_y, s.shot_distance, s.zone,
               s.quarter, s.time_remaining, s.score_diff, s.home_away,
               s.playoff_flag, g.date AS game_date
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.game_id = :game_id
        ORDER BY s.quarter ASC, s.time_remaining DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"game_id": game_id})
    if df.empty:
        raise RuntimeError(f"No shots found for game_id={game_id}")
    return df


def run(game_id: str | None, season: str | None, pace: float, loop: bool):
    engine = get_engine()
    if game_id is None:
        game_id = _pick_demo_game(engine, season)
        print(f"No --game-id given; auto-picked {game_id}")

    shots = load_game_shots(engine, game_id)
    print(f"Replaying {len(shots)} shots from game {game_id} "
          f"({shots['game_date'].iloc[0]}) at {pace}s/event")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    try:
        while True:
            for _, row in shots.iterrows():
                event = {
                    "shot_id": row["shot_id"],
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "player_id": row["player_id"],
                    "defender_id": row["defender_id"],
                    "shot_made": int(row["shot_made"]),
                    "loc_x": row["loc_x"],
                    "loc_y": row["loc_y"],
                    "shot_distance": row["shot_distance"],
                    "zone": row["zone"],
                    "quarter": int(row["quarter"]),
                    "time_remaining": row["time_remaining"],
                    "score_diff": int(row["score_diff"]),
                    "home_away": int(row["home_away"]),
                    "playoff_flag": int(row["playoff_flag"]),
                    "game_date": str(row["game_date"]),
                    "emitted_at": time.time(),
                }
                # Keyed by game_id: Kafka only guarantees order within a
                # partition, so every event for one game must land on the
                # same partition to be read back in the order it happened.
                producer.send(TOPIC_SHOT_EVENTS, key=event["game_id"], value=event)
                print(f"  → sent {event['shot_id']} "
                      f"(Q{event['quarter']} {event['time_remaining']:.0f}s left, "
                      f"zone={event['zone']})")
                time.sleep(pace)
            producer.flush()
            if not loop:
                break
            print("— replay finished, looping —")
    finally:
        producer.flush()
        producer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", default=None,
                        help="Specific game to replay (default: auto-pick a recent one)")
    parser.add_argument("--season", default=None,
                        help="Restrict auto-pick to this season")
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE_SECONDS,
                        help="Seconds to sleep between events")
    parser.add_argument("--loop", action="store_true",
                        help="Replay the game repeatedly instead of stopping once")
    args = parser.parse_args()
    run(args.game_id, args.season, args.pace, args.loop)


if __name__ == "__main__":
    main()
