# Streaming demo (Kafka)

A live-event scoring path that runs parallel to the batch training/serving
pipeline. It does not replace anything — `src/features`, `src/training`, and
`src/inference/recommender.py` are unmodified by this component.

## This is a simulated feed, not a live one

**There is no live NBA play-by-play API this project is authorized to
consume in real time.** `src/streaming/producer.py` replays real historical
shots from the existing database — in their real order, with their real
outcomes — onto a Kafka topic at a configurable pace. It is a stand-in for
a live feed, built so the transport and real-time scoring path can be
demonstrated and measured honestly. If a real upstream ever existed, this
producer is the only file that would change.

## Architecture

```
producer.py  →  Kafka ("shot-events", partitioned by game_id)  →  consumer.py  →  live_shot_scores (SQLite)  →  /live/* (FastAPI)
```

- **Producer**: reads one real game's shots from `shots`/`games`, in the
  order they were actually taken, and emits each as a JSON event keyed by
  `game_id`. Kafka only guarantees ordering *within* a partition, so keying
  by `game_id` is what keeps one game's shots in order.
- **Broker**: real Apache Kafka (KRaft mode — the upstream `apache/kafka`
  Docker image, no Zookeeper, no third-party vendor image), via
  `docker-compose.yml`.
- **Consumer**: on startup, loads the trained model exactly as the batch
  recommender does (`ShotRecommender`) and a precomputed feature store (see
  below). For each event, looks up its features, scores with
  `recommender._predict` — the identical serving-time code path
  `/recommend` uses — and appends the result to `live_shot_scores`.
- **API**: `GET /live/games`, `GET /live/scores` read that table.

## Why the features are precomputed, not computed per event

The model's point-in-time shooter/defender features are cumulative-to-date
aggregates that require season-long joins across shots, matchups, and team
data — exactly what `src.features.build.build_matrix` already does for
training. Re-deriving a second, streaming-native version of those joins
would duplicate real pipeline logic and risk it drifting from what the
model was actually trained on.

Instead, `src/streaming/feature_store.py` calls `build_matrix` once, up
front, and caches the result keyed by `shot_id`; `consumer.py` does an O(1)
lookup per event. This is the standard real-world split between an
offline/batch feature store and a real-time scoring service (the pattern
tools like Feast or Tecton formalize) — **the honest description is
"real-time scoring against a precomputed feature store," not "real-time
feature computation."** Measured end-to-end scoring latency (lookup +
predict, once the store is warm) is 2-3ms per event; the one-time feature
store build for the full 2.1M-shot history takes on the order of a minute.

## Running it

```bash
docker compose up -d                              # starts Kafka
python -m src.streaming.consumer                  # one-time feature-store build, then listens
python -m src.streaming.producer                  # auto-picks a recent game, replays it
python -m src.streaming.producer --game-id 0042500404 --pace 0.5
```

Scored events land in `live_shot_scores` and are queryable via
`GET /live/scores?game_id=...` or `GET /live/games`.
