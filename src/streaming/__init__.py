"""
Live-scoring demo path, parallel to the batch training/serving pipeline.

This is a simulated feed, not a live upstream integration — see
docs/streaming.md for exactly what that means and why. Nothing under
src/features, src/training, or src/inference is modified to support it;
`producer.py` and `consumer.py` are read-only clients of the existing
database and the existing trained model.
"""
