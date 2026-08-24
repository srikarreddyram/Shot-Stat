"""
Shared feature layer.

Every feature the model consumes is defined exactly once in this package and
computed by the same code at training time and at serving time. Before this
package existed, `src/training/feature_engineering.py` built features in SQL
and `src/inference/recommender.py` rebuilt the same features by hand in a
Python dict — two implementations that had to be kept in lockstep by eye, and
whose divergence would have been a silent accuracy bug no test could catch.

The split of responsibility:

  raw assembly   — differs by path. Training reads strictly-prior cumulative
                   counts for millions of shots at once; serving looks up one
                   player's current counts. Both produce the SAME raw columns.
  derivation     — identical. `spec.derive_features` is a pure, vectorized
                   function over those raw columns, and it is the only place
                   a derived feature is ever computed.

tests/test_train_serve_parity.py asserts the two paths agree.
"""
