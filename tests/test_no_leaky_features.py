"""
Guard against the leaky play-by-play columns reaching the model.

`shot_context.is_assisted` is a perfect predictor of the target — an assist is
credited only on a made basket, so field-goal percentage is exactly 1.000 when
the flag is set and 0.230 when it is not. A model handed this feature scores
beautifully offline and is worthless in production, because at prediction time
the shot has not been taken and nobody knows whether it will be assisted.

It is a genuinely useful column for descriptive work, so it is stored rather
than discarded. This test is what keeps "stored" from drifting into "used".
"""
from src.features.spec import FEATURE_GROUPS, all_feature_columns

# Columns that exist in the database and must never appear in a feature list.
# `action_type` is "Made Shot" / "Missed Shot" — the label, spelled differently.
LEAKY_COLUMNS = {"is_assisted", "action_type", "shot_made", "shot_result"}


def test_leaky_columns_are_not_in_any_feature_group():
    for group, columns in FEATURE_GROUPS.items():
        overlap = LEAKY_COLUMNS.intersection(columns)
        assert not overlap, (
            f"FEATURE_GROUPS['{group}'] contains target-leaking column(s) "
            f"{sorted(overlap)}. See the warning in src/ingestion/pbp_ingestor.py."
        )


def test_leaky_columns_are_not_in_the_assembled_feature_list():
    columns = set(all_feature_columns())
    overlap = LEAKY_COLUMNS.intersection(columns)
    assert not overlap, f"assembled feature list leaks: {sorted(overlap)}"


def test_pbp_context_features_that_ARE_safe_stay_available():
    """
    The counterpart assertion: the non-leaky play-by-play signals should be
    reachable, so a future cleanup does not throw the useful columns out along
    with the dangerous one.
    """
    safe = {"is_putback", "seconds_since_prev_event"}
    columns = set(all_feature_columns())
    missing = safe - columns
    assert not missing, (
        f"safe play-by-play features missing from the feature list: {sorted(missing)}"
    )


def test_mechanic_names_round_trip_through_the_classifier():
    """
    The recommender scores hypothetical shots by handing each grid row a
    mechanic NAME and running it through `classify_mechanic`, so that the
    serving path reproduces the training encoding rather than reimplementing
    it. That only holds if every bucket name classifies back to itself.

    Two did not when this was written — "alley_oop" and "stepback", whose rules
    matched only the spaced spellings — which would have silently filed those
    shots as "other" at serving time while training saw them correctly.
    """
    from src.features.spec import SHOT_MECHANICS, classify_mechanic

    for name in SHOT_MECHANICS:
        if name == "other":
            continue
        assert classify_mechanic(name) == name, (
            f"mechanic '{name}' classifies as '{classify_mechanic(name)}' — the "
            "serving path would encode it differently from training"
        )
