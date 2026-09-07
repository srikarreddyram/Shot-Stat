"""
Tests for src/inference/recommender.py.

Rewritten for the feature-layer redesign. The old suite asserted against
`_get_player_data(...)["zone_stats"]` — raw whole-season shooting splits read
straight off `player_zone_stats`. That contract is gone on purpose: those
splits were computed over the full season including the shot being predicted,
and they do not exist mid-season at serving time. Their replacement is a
shrunk, point-in-time rate assembled from strictly prior games, so the tests
here assert the properties that quantity is supposed to have.

Model artifacts are the real ones under models/ (a few MB), so the inference
path is exercised end to end. The engine is monkeypatched to the seeded
in-memory database so nothing depends on the 1.1GB data file.
"""
from pathlib import Path

import pytest

import src.inference.recommender as rec_mod
from src.features.point_in_time import DEFENSE_CATEGORIES
from src.features.shrinkage import BetaPrior

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_NAME = "shot-quality-v13"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / f"metadata_{MODEL_NAME}.json").exists(),
    reason=f"{MODEL_NAME} artifacts not present in models/",
)


@pytest.fixture()
def recommender(seeded_db, monkeypatch):
    monkeypatch.setattr(rec_mod, "get_engine", lambda: seeded_db)
    rec = rec_mod.ShotRecommender(model_name=MODEL_NAME, model_dir=str(MODELS_DIR))
    # `models/metadata_shot-quality-v9.json` predates point-in-time defender
    # quality, so it carries no `category_priors` and every lookup would
    # short-circuit to empty. strength=0 priors make `shrink()` return the raw
    # observed rate unshrunk, which is also what makes the fixture's expected
    # values in the tests below exact, hand-checkable fractions rather than
    # shrinkage-dependent ones.
    rec.category_priors = {cat: BetaPrior(mean=0.0, strength=0.0) for cat in DEFENSE_CATEGORIES}
    return rec


def test_zone_to_def_category_mapping_covers_all_six_zones():
    expected_zones = {
        "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
        "Left Corner 3", "Right Corner 3", "Above the Break 3",
    }
    assert set(rec_mod.ZONE_TO_DEF_CATEGORY) == expected_zones
    assert rec_mod.ZONE_TO_DEF_CATEGORY["Restricted Area"] == "Less Than 6Ft"
    # Both corners and above-the-break collapse to one "3 Pointers" category.
    assert rec_mod.ZONE_TO_DEF_CATEGORY["Left Corner 3"] == "3 Pointers"
    assert rec_mod.ZONE_TO_DEF_CATEGORY["Right Corner 3"] == "3 Pointers"
    assert rec_mod.ZONE_TO_DEF_CATEGORY["Above the Break 3"] == "3 Pointers"


def test_court_grid_stays_within_observed_shot_distances():
    """
    Every candidate location must sit inside the distance range where shots of
    that zone are actually attempted.

    The grid previously emitted restricted-area candidates half a foot from the
    basket and nearly five feet out — both outside the real 0-3ft envelope — and
    the model, having never seen such shots, extrapolated badly.
    """
    for point in rec_mod.SHOT_GRID:
        low, high = rec_mod.ZONE_DISTANCE_RANGE[point["zone"]]
        assert low <= point["shot_distance"] <= high, (
            f"{point['zone']} candidate at {point['shot_distance']}ft is outside "
            f"the observed range [{low}, {high}]"
        )


def test_grid_covers_every_zone():
    zones = {p["zone"] for p in rec_mod.SHOT_GRID}
    assert zones == set(rec_mod.ZONE_POINTS)


def test_player_row_returns_attributes(recommender):
    row = recommender._player_row("P_TALL", "2023-24")
    assert row["name"] == "Tall Center"
    assert row["height"] == pytest.approx(84.0)
    assert row["wingspan"] == pytest.approx(88.0)


def test_player_row_populates_point_in_time_counts(recommender):
    """
    The serving path must return prior-shot counts under the same keys the
    training builder emits, since both are fed to the shared
    `apply_hierarchy`.
    """
    row = recommender._player_row("P_TALL", "2023-24")
    for key in ("pit_car_mk_restricted_area", "pit_car_att_restricted_area",
                "pit_car_mk_all", "pit_car_att_all"):
        assert key in row, f"serving path missing point-in-time key {key}"


def test_player_row_populates_recent_form(recommender):
    """
    Regression test. `recent_10_fg`/`recent_20_fg` are set for effectively every
    training row, but the serving path never set them at all — so every served
    shot looked to the model like a player's first career game, which dragged
    rim probabilities from roughly 0.70 down to 0.17.
    """
    row = recommender._player_row("P_TALL", "2023-24")
    assert "recent_10_fg" in row
    assert "recent_20_fg" in row


def test_player_row_raises_for_unknown_player(recommender):
    with pytest.raises(ValueError, match="not found"):
        recommender._player_row("NOBODY", "2023-24")


def test_defender_row_returns_zone_level_stats_not_just_overall(recommender):
    """
    Zone-level defender quality must differ from the pooled "Overall" figure,
    not collapse to it — the bug this fixture was originally built around.

    P_DEF is every seeded shooter's only matchup in G1, so his point-in-time
    rates are exact fractions of the seeded shots: 2 makes allowed at the rim
    on 2 attempts (S1 off P_TALL, S4 off P_SHORT — both Restricted Area), 0
    of 1 at mid-range (S2), 0 of 2 on threes (S5, S3), giving Overall 2/5.
    """
    defender = recommender._defender_row("P_DEF", "2023-24")
    by_category = defender["_by_category"]

    assert by_category["Overall"]["d_fg_pct"] == pytest.approx(0.4)
    assert by_category["Less Than 6Ft"]["d_fg_pct"] == pytest.approx(1.0)
    assert by_category["3 Pointers"]["d_fg_pct"] == pytest.approx(0.0)
    assert by_category["Greater Than 15Ft"]["d_fg_pct"] == pytest.approx(0.0)
    # No seeded shot maps to this category — with strength=0 test priors,
    # zero attempts and zero prior pull both come out as 0/0, which
    # `lookup_defender_category_rates` reports as None (JSON-safe "unknown").
    assert by_category["Less Than 10Ft"]["d_fg_pct"] is None


def test_defender_row_returns_empty_for_unknown_defender(recommender):
    assert recommender._defender_row("NOBODY", "2023-24") == {}


def test_recommend_runs_end_to_end_with_defender(recommender):
    out = recommender.recommend(
        player_id="P_TALL", season="2023-24", defender_id="P_DEF", top_n=5
    )
    assert len(out) == 5
    for col in ("zone", "make_probability", "expected_points",
                "ep_low", "ep_high", "attainability", "score"):
        assert col in out.columns
    assert out["make_probability"].between(0, 1).all()
    # Sorted by the ranking score, descending.
    assert out["score"].is_monotonic_decreasing


def test_recommend_without_defender_uses_league_average_not_nulls(recommender):
    """
    Omitting a defender means "against a typical defender". Leaving the columns
    NaN would instead say "this game has no matchup data", which is a different
    and much rarer thing in the training distribution.
    """
    out = recommender.recommend(player_id="P_TALL", season="2023-24", top_n=3)
    assert len(out) == 3
    assert out["make_probability"].notna().all()


def test_intervals_bracket_the_estimate(recommender):
    out = recommender.recommend(player_id="P_TALL", season="2023-24", top_n=10)
    assert (out["ep_low"] <= out["expected_points"]).all()
    assert (out["expected_points"] <= out["ep_high"]).all()
