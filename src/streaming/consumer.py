"""
Shot-scoring consumer — the other half of the Kafka demo.

Subscribes to `shot-events`, looks up each event's precomputed point-in-time
features (see feature_store.py for why they are precomputed rather than
recomputed per event), scores it with the SAME trained model the batch
recommender serves from, and appends the result to `live_shot_scores`.

This does not modify `src.inference.recommender` — it instantiates the real
`ShotRecommender` and calls its existing `_predict`, so a live score and a
batch prediction for the identical feature row are the identical number,
by construction.

Usage:
    python -m src.streaming.consumer
    python -m src.streaming.consumer --model-name shot-quality-v13
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kafka import KafkaConsumer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine, get_session_factory, init_db
from src.db.models import LiveShotScore
from src.features.spec import as_model_matrix
from src.inference.recommender import ShotRecommender
from src.streaming.config import KAFKA_BOOTSTRAP_SERVERS, TOPIC_SHOT_EVENTS
from src.streaming.feature_store import load_feature_store


def score_event(event: dict, store, feature_cols: list[str],
                recommender: ShotRecommender) -> float | None:
    """
    Look up `event["shot_id"]` in the precomputed feature store and score it.

    Returns None if the shot isn't in the store (e.g. it was replayed from a
    season outside the store's build range) — the caller logs and skips
    rather than guessing a feature row for it.
    """
    shot_id = event["shot_id"]
    if shot_id not in store.index:
        return None
    row = store.loc[[shot_id]]
    X = as_model_matrix(row, feature_cols)
    return float(recommender._predict(X)[0])


def run(model_name: str, seasons: list[str] | None, prior_through_season: str | None):
    engine = get_engine()
    init_db(engine)  # idempotent — creates live_shot_scores if not present
    Session = get_session_factory(engine)

    print("Loading model + feature store (one-time startup cost) ...")
    recommender = ShotRecommender(model_name=model_name)
    store, feature_cols = load_feature_store(seasons, prior_through_season)

    consumer = KafkaConsumer(
        TOPIC_SHOT_EVENTS,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        group_id="shot-scoring-consumer",
    )

    print(f"✓ consumer ready — listening on '{TOPIC_SHOT_EVENTS}'")
    with Session() as session:
        for message in consumer:
            event = message.value
            received_at = time.time()

            prob = score_event(event, store, feature_cols, recommender)
            if prob is None:
                print(f"  ⚠ {event['shot_id']} not in feature store, skipping")
                continue

            latency_ms = (received_at - event.get("emitted_at", received_at)) * 1000.0
            row = LiveShotScore(
                shot_id=event["shot_id"],
                game_id=event["game_id"],
                player_id=event["player_id"],
                defender_id=event.get("defender_id"),
                zone=event.get("zone"),
                predicted_make_probability=prob,
                actual_shot_made=event.get("shot_made"),
                game_date=event.get("game_date"),
                scored_at=received_at,
                latency_ms=latency_ms,
            )
            session.add(row)
            session.commit()

            outcome = "MADE" if event.get("shot_made") else "missed"
            print(f"  scored {event['shot_id']}: p={prob:.3f}  "
                  f"(actual: {outcome}, zone={event.get('zone')}, "
                  f"latency={latency_ms:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="shot-quality-v13")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Seasons to build the feature store over "
                             "(default: everything from 2016-17)")
    parser.add_argument("--prior-through-season", default=None)
    args = parser.parse_args()
    run(args.model_name, args.seasons, args.prior_through_season)


if __name__ == "__main__":
    main()
