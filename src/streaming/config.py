"""
Constants for the Kafka demo path. Kept separate from the top-level
config.py because nothing outside src/streaming reads these — the training
and inference pipelines have no Kafka dependency at all.
"""
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_SHOT_EVENTS = "shot-events"

# Partitioned by game_id at produce time (see producer.py) so that one game's
# events are always read back in the order they were sent — Kafka only
# guarantees ordering within a partition, not across a topic.
PARTITION_KEY_FIELD = "game_id"

DEFAULT_PACE_SECONDS = 1.0  # gap between successive replayed events
