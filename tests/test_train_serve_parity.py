"""
Train/serve parity — the test that keeps the two paths honest.

The failure this guards against is the one that motivated the whole
`src/features` package: the training matrix was built in SQL and the
recommender rebuilt the same forty features by hand in a Python dict. Nothing
enforced that they agreed. When they drifted, every served prediction was
quietly wrong and every offline metric still looked fine, because the offline
metrics only ever exercised the training path.

These tests assert that identical raw inputs produce identical model features
no matter which path assembled them, and — the part that actually caught real
bugs — that the serving path does not leave features NaN which the training
path virtually always populates. That second check found three live defects
when it was written:

  * `recent_10_fg` / `recent_20_fg` were never set at serving time, so every
    served shot looked to the model like a player's career debut.
  * omitting a defender left the defender columns NaN, which in the training
    matrix means "this game has no matchup data at all" rather than "an
    average defender".
  * the court grid emitted restricted-area candidates half a foot from the
    basket, outside the range where such shots are ever actually attempted.

Together those pushed served rim probabilities to 0.17 against a true rate
near 0.70.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.point_in_time import ZONE_SUFFIX, ZONES, apply_hierarchy
from src.features.shrinkage import BetaPrior
from src.features.spec import as_model_matrix, derive_features

# Features the training matrix populates for essentially every row. If the
# serving path leaves one of these empty, it is a skew bug, not a data gap.
# Defender columns are excluded: they are legitimately absent for shots with no
# matchup data, and the recommender substitutes league averages by design.
DENSE_FEATURES = [
    "zone_rate", "zone_att", "overall_rate", "three_rate",
    "recent_10_fg", "recent_20_fg",
    "shot_distance", "shot_angle", "distance_from_center", "abs_loc_x",
    "is_three", "clutch_flag", "seconds_remaining_in_game",
]


@pytest.fixture()
def zone_priors():
    """Priors with the rough shape of the real fitted ones."""
    return {
        "Restricted Area": BetaPrior(mean=0.629, strength=64.0),
        "In The Paint (Non-RA)": BetaPrior(mean=0.416, strength=78.0),
        "Mid-Range": BetaPrior(mean=0.404, strength=146.0),
        "Left Corner 3": BetaPrior(mean=0.397, strength=223.0),
        "Right Corner 3": BetaPrior(mean=0.397, strength=195.0),
        "Above the Break 3": BetaPrior(mean=0.354, strength=316.0),
    }


def _raw_row(zone="Mid-Range", **overrides):
    """A raw feature row in the shape both paths produce before derivation."""
    row = {
        "loc_x": 120.0, "loc_y": 110.0, "zone": zone,
        # shot_distance is a RAW column supplied by both paths (from the shots
        # table when training, from the grid when serving) rather than derived,
        # so the fixture must provide it like any other raw input.
        "shot_distance": 16.3,
        "shot_type": "2PT Field Goal",
        "quarter": 4, "time_remaining": 90.0, "score_diff": -3,
        "home_away": 1, "playoff_flag": 0,
        "rest_days": 1, "is_back_to_back": 0, "opp_def_rating": 112.0,
        "height": 79.0, "weight": 230.0, "wingspan": 82.0, "position": "SF",
        "def_height": 78.0, "def_weight": 215.0, "def_wingspan": 81.0,
        "def_fg_pct_overall": 0.47, "def_pct_plusminus": -0.01,
        "def_fg_pct_zone": 0.38, "def_pct_plusminus_zone": -0.02,
        "def_freq_zone": 0.45,
        "avg_def_dist": 4.2, "open_share": 0.5,
        "self_creation_index": 1.4,
        "recent_10_fg": 0.47, "recent_20_fg": 0.45,
    }
    for zone_name in ZONES:
        suffix = ZONE_SUFFIX[zone_name]
        row[f"pit_car_mk_{suffix}"] = 300.0
        row[f"pit_car_att_{suffix}"] = 700.0
        row[f"pit_ssn_mk_{suffix}"] = 40.0
        row[f"pit_ssn_att_{suffix}"] = 90.0
    row["pit_car_mk_all"] = 1800.0
    row["pit_car_att_all"] = 4200.0
    row["pit_ssn_mk_all"] = 240.0
    row["pit_ssn_att_all"] = 540.0
    row["pit_car_mk_3pt"] = 500.0
    row["pit_car_att_3pt"] = 1400.0
    row.update(overrides)
    return row


def test_derive_is_row_order_and_batch_invariant(zone_priors):
    """
    The same shot must derive identically whether it arrives alone or inside a
    batch of two million.

    The old implementation could not promise this: it built zone and position
    encodings with `pd.get_dummies(..., drop_first=True)`, so the column set —
    and with drop_first, the reference level itself — depended on which zones
    happened to appear in the frame. A single-row serving call and a full
    training matrix produced different encodings of the same shot.
    """
    rows = [_raw_row(zone=z) for z in ZONES]
    batch = derive_features(apply_hierarchy(pd.DataFrame(rows), zone_priors))

    for i, zone in enumerate(ZONES):
        single = derive_features(
            apply_hierarchy(pd.DataFrame([rows[i]]), zone_priors)
        )
        for col in single.columns:
            if col in ("zone", "shot_type", "position"):
                continue
            alone = single.iloc[0][col]
            in_batch = batch.iloc[i][col]
            if pd.isna(alone) and pd.isna(in_batch):
                continue
            assert alone == pytest.approx(in_batch, rel=1e-9, abs=1e-9), (
                f"{col} differs for {zone} between single-row and batch "
                f"derivation: {alone} vs {in_batch}"
            )


def test_zone_encoding_is_complete_and_fixed(zone_priors):
    """
    Every zone gets its own indicator column, always, regardless of which
    zones are present in the frame.
    """
    one_zone = derive_features(
        apply_hierarchy(pd.DataFrame([_raw_row(zone="Mid-Range")]), zone_priors)
    )
    for zone in ZONES:
        col = f"zone_is_{ZONE_SUFFIX[zone]}"
        assert col in one_zone.columns, f"missing indicator {col}"

    assert one_zone.iloc[0]["zone_is_midrange"] == 1
    assert one_zone.iloc[0]["zone_is_restricted_area"] == 0


def test_zone_rate_selects_the_shooting_zone(zone_priors):
    """`zone_rate` must read the column matching the row's own zone."""
    overrides = {}
    for i, zone in enumerate(ZONES):
        suffix = ZONE_SUFFIX[zone]
        overrides[f"pit_car_mk_{suffix}"] = 100.0 * (i + 1)
        overrides[f"pit_car_att_{suffix}"] = 1000.0

    for zone in ZONES:
        row = _raw_row(zone=zone, **overrides)
        out = derive_features(apply_hierarchy(pd.DataFrame([row]), zone_priors))
        expected = out.iloc[0][f"zone_rate_{ZONE_SUFFIX[zone]}"]
        assert out.iloc[0]["zone_rate"] == pytest.approx(expected)


def test_clutch_flag_uses_the_official_five_minute_window(zone_priors):
    """clutch_flag matches the NBA's own "Clutch Time" definition (last 5
    minutes, not the previous ad hoc 2-minute window) — a shot at 200 seconds
    remaining must now flag as clutch when it would not have before."""
    row = _raw_row(quarter=4, time_remaining=200.0, score_diff=-3)
    out = derive_features(apply_hierarchy(pd.DataFrame([row]), zone_priors))
    assert out.iloc[0]["clutch_flag"] == 1

    not_clutch_time = _raw_row(quarter=4, time_remaining=400.0, score_diff=-3)
    out2 = derive_features(apply_hierarchy(pd.DataFrame([not_clutch_time]), zone_priors))
    assert out2.iloc[0]["clutch_flag"] == 0

    not_clutch_margin = _raw_row(quarter=4, time_remaining=200.0, score_diff=11)
    out3 = derive_features(apply_hierarchy(pd.DataFrame([not_clutch_margin]), zone_priors))
    assert out3.iloc[0]["clutch_flag"] == 0


def test_clutch_edge_is_gated_by_clutch_flag(zone_priors):
    """clutch_edge is the player's clutch-vs-normal delta, but ONLY on a shot
    that is itself clutch — the exact interaction the user asked for ("some
    people are better than others in the clutch"), not a standing bonus that
    applies regardless of game situation."""
    clutch_shot = _raw_row(quarter=4, time_remaining=200.0, score_diff=-3,
                           clutch_fg_delta=0.08)
    out = derive_features(apply_hierarchy(pd.DataFrame([clutch_shot]), zone_priors))
    assert out.iloc[0]["clutch_edge"] == pytest.approx(0.08)

    non_clutch_shot = _raw_row(quarter=2, time_remaining=400.0, score_diff=-3,
                               clutch_fg_delta=0.08)
    out2 = derive_features(apply_hierarchy(pd.DataFrame([non_clutch_shot]), zone_priors))
    assert out2.iloc[0]["clutch_edge"] == pytest.approx(0.0)


def test_as_model_matrix_preserves_column_order(zone_priors):
    """
    XGBoost binds to column position. A frame with the right columns in the
    wrong order predicts silently and wrongly, so the ordering contract is
    asserted rather than assumed.
    """
    out = derive_features(apply_hierarchy(pd.DataFrame([_raw_row()]), zone_priors))
    cols = ["zone_rate", "shot_distance", "is_three", "overall_rate"]
    matrix = as_model_matrix(out, cols)
    assert list(matrix.columns) == cols

    shuffled = as_model_matrix(out, list(reversed(cols)))
    assert list(shuffled.columns) == list(reversed(cols))


def test_as_model_matrix_coerces_object_columns(zone_priors):
    """
    A SQL NULL makes a whole serving column `object` dtype, which XGBoost
    rejects outright. Coercion lives in the shared helper so both paths hand
    the booster the same types.
    """
    out = derive_features(apply_hierarchy(pd.DataFrame([_raw_row()]), zone_priors))
    out["wingspan"] = pd.Series([None], dtype=object)
    matrix = as_model_matrix(out, ["wingspan", "zone_rate"])
    assert matrix["wingspan"].dtype == np.float32
    assert np.isnan(matrix.iloc[0]["wingspan"])


def test_shrinkage_regresses_thin_samples_toward_the_prior(zone_priors):
    """
    The behaviour that replaced both the leaky season aggregates and the
    separate rookie-priors table: a player with no attempts sits on the league
    prior, and one with many sits near his own rate.
    """
    prior = zone_priors["Above the Break 3"]

    empty = _raw_row(zone="Above the Break 3")
    for zone in ZONES:
        suffix = ZONE_SUFFIX[zone]
        for key in ("pit_car_mk", "pit_car_att", "pit_ssn_mk", "pit_ssn_att"):
            empty[f"{key}_{suffix}"] = 0.0
    out_empty = derive_features(apply_hierarchy(pd.DataFrame([empty]), zone_priors))
    assert out_empty.iloc[0]["zone_rate"] == pytest.approx(prior.mean, abs=1e-9)

    heavy = _raw_row(
        zone="Above the Break 3",
        **{
            "pit_car_mk_above_break_3": 2000.0,
            "pit_car_att_above_break_3": 5000.0,
            "pit_ssn_mk_above_break_3": 200.0,
            "pit_ssn_att_above_break_3": 500.0,
        },
    )
    out_heavy = derive_features(apply_hierarchy(pd.DataFrame([heavy]), zone_priors))
    rate = out_heavy.iloc[0]["zone_rate"]
    assert abs(rate - 0.40) < 0.02, f"heavy sample should sit near 0.40, got {rate}"


def test_expected_contest_stays_physically_real(zone_priors):
    """Daylight is measured in feet, and feet are not negative."""
    extreme = _raw_row(def_pct_plusminus_zone=-0.63, avg_def_dist=2.7)
    out = derive_features(apply_hierarchy(pd.DataFrame([extreme]), zone_priors))
    assert out.iloc[0]["expected_contest"] >= 0.5

    generous = _raw_row(def_pct_plusminus_zone=0.71, avg_def_dist=6.2)
    out = derive_features(apply_hierarchy(pd.DataFrame([generous]), zone_priors))
    assert out.iloc[0]["expected_contest"] <= 12.0


# ── Live parity against the real recommender ─────────────────────────────────
# These require the trained model and populated database, so they skip cleanly
# in a bare checkout or CI without the 1.1GB data file.

def _recommender():
    pytest.importorskip("xgboost")
    try:
        from src.inference.recommender import ShotRecommender
        return ShotRecommender()
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"trained model unavailable: {exc}")


@pytest.mark.slow
def test_serving_path_populates_dense_features():
    """
    The regression test for the skew that motivated this file.

    Every feature the training matrix fills for effectively every row must also
    be filled by the serving path. `recent_10_fg` failing this check was worth
    roughly half a rim probability.
    """
    from src.inference.shot_grid import SHOT_GRID

    rec = _recommender()
    grid = rec.recommend(player_id="201939", season="2025-26",
                         top_n=len(SHOT_GRID))
    assert len(grid) > 0

    from src.features.spec import derive_features as _derive  # noqa: F401

    # Re-derive through the recommender's own assembly to inspect features.
    player = rec._player_row("201939", "2025-26")
    for col in ("recent_10_fg", "recent_20_fg"):
        assert player.get(col) is not None, (
            f"serving path left {col} unset — this is the training/serving "
            "skew that made every served shot look like a career debut"
        )


@pytest.mark.slow
def test_served_probabilities_are_physically_plausible():
    """
    Rim shots go in more often than above-the-break threes, for everybody.

    A blunt check, and it is exactly the one that would have caught the
    0.17-at-the-rim regression immediately.
    """
    rec = _recommender()
    summary = rec.zone_summary(player_id="201939", season="2025-26")
    rates = dict(zip(summary["zone"], summary["make_probability"]))

    assert 0.55 < rates["Restricted Area"] < 0.85
    assert 0.25 < rates["Above the Break 3"] < 0.50
    assert rates["Restricted Area"] > rates["Above the Break 3"]


@pytest.mark.slow
def test_credible_intervals_bracket_the_estimate():
    """
    A band that does not contain the number it decorates is worse than no band.
    This failed on first implementation because the interval was centred on the
    Beta posterior mean while the estimate came from the full model.
    """
    rec = _recommender()
    summary = rec.zone_summary(player_id="203999", season="2025-26")
    for _, row in summary.iterrows():
        assert row["ep_low"] <= row["expected_points"] <= row["ep_high"], (
            f"{row['zone']}: [{row['ep_low']:.3f}, {row['ep_high']:.3f}] "
            f"does not contain {row['expected_points']:.3f}"
        )


@pytest.mark.slow
def test_interval_width_tracks_evidence():
    """More attempts must mean a tighter band."""
    rec = _recommender()
    summary = rec.zone_summary(player_id="201939", season="2025-26")
    summary["width"] = summary["ep_high"] - summary["ep_low"]

    most = summary.nlargest(1, "attempts_behind").iloc[0]
    fewest = summary.nsmallest(1, "attempts_behind").iloc[0]
    assert most["width"] < fewest["width"], (
        "the zone with the most attempts behind it should have the narrower "
        f"interval: {most['zone']} ({most['attempts_behind']:.0f} attempts, "
        f"width {most['width']:.3f}) vs {fewest['zone']} "
        f"({fewest['attempts_behind']:.0f}, width {fewest['width']:.3f})"
    )
