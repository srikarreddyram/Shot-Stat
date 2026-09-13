"""
The attainability explanation.

The property that matters is completeness: the reported decomposition must
account for the whole prediction. TreeSHAP guarantees the contributions sum to
it, so these tests assert that the grouping into baseline-plus-player does not
lose or double-count anything on the way to the UI — a decomposition that
silently drops a term would read as a confident, wrong explanation.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.inference.explain import (
    _percentile, defender_zone_tendency_note, explain_attainability,
    explain_shot_quality,
)
from src.training.attainability import (
    POSITION_BUCKETS, SUB_ZONE_SUFFIX, SUB_ZONES, encode_zone_and_position,
)


@pytest.fixture()
def toy_model():
    """
    A small regressor over the real feature shape, fit on synthetic rows where
    the answer is known: pull-up shooters get fewer rim shots, spot-up shooters
    get more corner threes.
    """
    rng = np.random.default_rng(0)
    n = 800
    rows = []
    for _ in range(n):
        zone = SUB_ZONES[rng.integers(len(SUB_ZONES))]
        pullup = float(rng.uniform(0, 1))
        catch = 1.0 - pullup
        share = 0.30 - 0.25 * pullup if zone == "Restricted Area" else 0.05 + 0.05 * catch
        rows.append({
            "zone": zone, "position": "G",
            "height": 76.0, "weight": 200.0, "wingspan": 80.0,
            "pullup_share": pullup, "catch_shoot_share": catch,
            "zone_share": share,
        })
    df = encode_zone_and_position(pd.DataFrame(rows))

    cols = (["height", "weight", "wingspan", "pullup_share", "catch_shoot_share"]
            + [f"zone_is_{s}" for s in SUB_ZONE_SUFFIX.values()]
            + ["is_three"]
            + [f"pos_is_{b}" for b in POSITION_BUCKETS])

    model = xgb.XGBRegressor(n_estimators=60, max_depth=4, random_state=0)
    model.fit(df[cols], df["zone_share"])

    metadata = {
        "feature_cols": cols,
        "position_buckets": POSITION_BUCKETS,
        "league_zone_shares": {z: 0.1 for z in SUB_ZONES},
        "feature_reference": {
            "pullup_share": {"mean": 0.5, "quantiles": {"0.1": 0.1, "0.5": 0.5, "0.9": 0.9}},
            "catch_shoot_share": {"mean": 0.5, "quantiles": {"0.1": 0.1, "0.5": 0.5, "0.9": 0.9}},
        },
    }
    return model, metadata, cols


def _row(zone, cols, **overrides):
    base = {
        "zone": zone, "position": "G",
        "height": 76.0, "weight": 200.0, "wingspan": 80.0,
        "pullup_share": 0.9, "catch_shoot_share": 0.1,
    }
    base.update(overrides)
    frame = encode_zone_and_position(pd.DataFrame([base]))
    for c in cols:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame


def test_baseline_plus_player_effect_reconstructs_the_estimate(toy_model):
    """
    The two halves the UI shows side by side must add up to the number beside
    them. If they drift, the panel is telling the reader a sum that is false.
    """
    model, metadata, cols = toy_model
    out = explain_attainability(model, metadata, _row("Restricted Area", cols),
                                "Restricted Area")
    assert out["baseline"] + out["player_effect"] == pytest.approx(
        out["attainability"], abs=1e-3
    )


def test_a_pullup_heavy_guard_is_marked_down_at_the_rim(toy_model):
    """
    The synthetic data was built so pull-up share suppresses rim attempts. The
    explanation must recover that rather than merely reporting the total.
    """
    model, metadata, cols = toy_model
    out = explain_attainability(model, metadata, _row("Restricted Area", cols),
                                "Restricted Area")

    pullup = next((f for f in out["factors"] if f["feature"] == "pullup_share"), None)
    assert pullup is not None, "pull-up share should surface as a factor at the rim"
    assert pullup["direction"] == "lowers"


def test_zone_and_position_terms_stay_out_of_the_player_factors(toy_model):
    """
    Zone identity belongs to the baseline half. Leaking it into the player
    factors would attribute the floor's own geometry to the player's game —
    the exact conflation this decomposition exists to prevent.
    """
    model, metadata, cols = toy_model
    out = explain_attainability(model, metadata, _row("Left Corner 3", cols),
                                "Left Corner 3")
    for factor in out["factors"]:
        assert not factor["feature"].startswith("zone_is_")
        assert not factor["feature"].startswith("pos_is_")
        assert factor["feature"] != "is_three"


def test_factors_are_ranked_by_absolute_impact(toy_model):
    model, metadata, cols = toy_model
    out = explain_attainability(model, metadata, _row("Restricted Area", cols),
                                "Restricted Area")
    impacts = [abs(f["impact"]) for f in out["factors"]]
    assert impacts == sorted(impacts, reverse=True)


def test_percentile_interpolates_and_clamps():
    reference = {"quantiles": {"0.1": 1.0, "0.5": 5.0, "0.9": 9.0}}
    assert _percentile(5.0, reference) == pytest.approx(50, abs=1e-6)
    # Outside the stored grid the percentile clamps rather than extrapolating
    # to an impossible value.
    assert _percentile(-100.0, reference) == pytest.approx(10)
    assert _percentile(100.0, reference) == pytest.approx(90)


def test_percentile_is_none_without_a_reference():
    assert _percentile(1.0, {}) is None
    assert _percentile(np.nan, {"quantiles": {"0.5": 1.0}}) is None


def test_every_model_feature_has_display_copy():
    """
    A feature with no entry in FEATURE_COPY is silently invisible in the
    explanation — it can move the estimate several points and never appear in
    the reasons, which is worse than an ugly label because the panel then
    claims to explain a number it has only partly accounted for.

    Zone and position terms are exempt: they belong to the baseline half of
    the decomposition and are deliberately never rendered as player traits.
    """
    from src.inference.explain import FEATURE_COPY, POSITION_PREFIX, ZONE_PREFIXES
    from src.training.attainability import (
        ATTAINABILITY_FEATURE_COLS, CAST_FEATURE_COLS,
    )

    missing = [
        col for col in ATTAINABILITY_FEATURE_COLS + CAST_FEATURE_COLS
        if not col.startswith(ZONE_PREFIXES)
        and not col.startswith(POSITION_PREFIX)
        and col not in FEATURE_COPY
    ]
    assert not missing, f"features with no reader-facing copy: {missing}"


def test_supporting_cast_copy_describes_teammates_not_the_team():
    """
    The leave-one-out construction is the whole point of these features — a
    star's raw team numbers are mostly the star. Copy that says "his team"
    would describe the quantity the model deliberately does not use.
    """
    from src.inference.explain import FEATURE_COPY
    from src.training.attainability import CAST_FEATURE_COLS

    for col in CAST_FEATURE_COLS:
        copy = FEATURE_COPY[col]
        blob = " ".join([copy["label"], copy["high"], copy["low"]]).lower()
        assert "teammate" in blob or "around him" in blob, (
            f"{col} copy should describe teammates, not the team: {copy}"
        )


def test_encoding_is_identical_for_a_single_row_and_a_batch():
    """
    Same train/serve parity rule the shot-quality features follow: a one-row
    serving call must encode exactly as the full training matrix does.
    """
    rows = [{"zone": z, "position": "F"} for z in SUB_ZONES]
    batch = encode_zone_and_position(pd.DataFrame(rows))
    for i, zone in enumerate(SUB_ZONES):
        single = encode_zone_and_position(pd.DataFrame([rows[i]]))
        for col in single.columns:
            if col in ("zone", "position"):
                continue
            assert single.iloc[0][col] == batch.iloc[i][col], f"{col} differs for {zone}"


def test_angle_split_separates_dead_centre_from_the_wing():
    """
    The distinction the sub-zone taxonomy exists for. Above the Break 3 spans
    the whole arc, and a dead-centre pull-up and a wing spot-up are different
    shots — measured self-creation falls from 27% within ten degrees of centre
    to 9.5% beyond sixty. Before the split both received a byte-identical
    attainability because they were one one-hot column.
    """
    from src.training.attainability import attach_sub_zone

    df = pd.DataFrame([
        # dead centre: straight out from the rim
        {"zone": "Above the Break 3", "loc_x": 0.0, "loc_y": 260.0},
        # 25 degrees off — still inside the centre boundary
        {"zone": "Above the Break 3", "loc_x": 110.0, "loc_y": 236.0},
        # 55 degrees off — a wing three
        {"zone": "Above the Break 3", "loc_x": 213.0, "loc_y": 149.0},
        # corners are never split: already angle-specific by construction
        {"zone": "Left Corner 3", "loc_x": -230.0, "loc_y": 40.0},
    ])
    out = attach_sub_zone(df)
    assert list(out["sub_zone"]) == [
        "Above the Break 3 (centre)",
        "Above the Break 3 (centre)",
        "Above the Break 3 (wing)",
        "Left Corner 3",
    ]


def test_sub_zone_falls_back_to_the_parent_without_coordinates():
    """
    A row with no coordinates must not be guessed into a half. It keeps the
    parent name, whose one-hot columns then read all-zero — an honest
    "unknown" rather than a fabricated centre or wing.
    """
    from src.training.attainability import attach_sub_zone

    out = attach_sub_zone(pd.DataFrame([{"zone": "Above the Break 3"}]))
    assert out["sub_zone"].iloc[0] == "Above the Break 3"


def test_diet_history_has_a_bare_zone_entry_for_angle_split_zones():
    """
    Regression test for a real bug: `attach_sub_zone` resolves a coordinate-
    less shot to the bare zone name ("Above the Break 3"), but
    `lookup_diet_history`'s output used to be keyed ONLY by the split names
    ("Above the Break 3 (centre)"/"(wing)") for the two angle-split zones.
    A caller without exact coordinates — a real, reachable path, since
    loc_x/loc_y are optional query params on /explain/attainability — looked
    up a key that never existed and got None back, regardless of how much
    real data the player had.

    Concretely, this returned a 2% attainability for Stephen Curry at
    Above the Break 3 (his single most common shot, ~55% of his 2024-25
    attempts) instead of anything close to the real number, because the
    missing diet-history features let the model fall back to whatever it
    learned for that gap — and the explanation shown to the user never even
    mentioned history as the cause, because the summary only narrates it
    when a share is known.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from src.db.database import register_unaccent
    from src.db.models import Base, Game, Shot
    from src.training.attainability import lookup_diet_history

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        session.add(Game(game_id="G1", date=date(2024, 11, 1),
                         home_team="GSW", away_team="LAL"))
        # 2 centre threes, 1 wing three this season — a real, sizeable share
        # (3 of 3 shots are Above the Break 3) that a fixed bug reported as
        # unknown.
        session.add_all([
            Shot(shot_id="s1", game_id="G1", player_id="CURRY", season="2024-25",
                shot_made=1, loc_x=0.0, loc_y=260.0, zone="Above the Break 3"),
            Shot(shot_id="s2", game_id="G1", player_id="CURRY", season="2024-25",
                shot_made=0, loc_x=20.0, loc_y=255.0, zone="Above the Break 3"),
            Shot(shot_id="s3", game_id="G1", player_id="CURRY", season="2024-25",
                shot_made=1, loc_x=213.0, loc_y=149.0, zone="Above the Break 3"),
        ])
        session.commit()

        with engine.connect() as conn:
            out = lookup_diet_history(conn, "CURRY", "2024-25", sub_zone_priors={})

    assert "Above the Break 3" in out["zones"], (
        "no bare-zone entry — a coordinate-less lookup for this zone would "
        "silently get None regardless of real data"
    )
    bare = out["zones"]["Above the Break 3"]
    assert bare["diet_to_date"] == pytest.approx(1.0)
    assert bare["diet_att_to_date"] == 3


def test_creation_priors_have_a_bare_zone_entry_for_angle_split_zones():
    """
    Regression test for the same class of bug as the diet-history one above,
    found in a different function: `fit_league_creation_priors` was keyed
    ONLY by split sub-zone names for the two angle-split zones, so
    `lookup_zone_creation` — which resolves to the bare zone name whenever
    it isn't given exact coordinates — found no prior at all for either one.
    `share` and `league_self_created_share` came back None and
    `creation_note` produced nothing.

    Concretely: asking for Jamal Murray's self-created share at Above the
    Break 3 returned nothing, reading as "we don't know how he generates
    this shot" for a real, established NBA starter with 1,157 makes there —
    not because the data was thin, but because the prior dict had no key a
    coordinate-less lookup could ever match.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from src.db.database import register_unaccent
    from src.db.models import Base, Game, Shot, ShotContext
    from src.features.point_in_time import fit_league_creation_priors

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        session.add(Game(game_id="G1", date=date(2024, 11, 1),
                         home_team="DEN", away_team="LAL"))
        # A handful of synthetic players, split across centre/wing, with a
        # mix of self-created and assisted makes — enough for fit_beta_prior
        # to fit something real rather than degenerate.
        for i in range(5):
            pid = f"P{i}"
            for j in range(4):
                shot_id = f"s{i}_{j}"
                # Alternate centre/wing and self-created/assisted.
                loc_y = 260.0 if j % 2 == 0 else 149.0
                loc_x = 0.0 if j % 2 == 0 else 213.0
                session.add(Shot(
                    shot_id=shot_id, game_id="G1", player_id=pid, season="2023-24",
                    shot_made=1, loc_x=loc_x, loc_y=loc_y, zone="Above the Break 3",
                ))
                session.add(ShotContext(
                    shot_id=shot_id, game_id="G1", is_assisted=1 if j < 2 else 0,
                ))
        session.commit()

        with engine.connect() as conn:
            priors = fit_league_creation_priors(conn, through_season="2024-25")

    assert "Above the Break 3" in priors, (
        "no bare-zone prior — a coordinate-less creation-share lookup for "
        "this zone would silently get None regardless of real data"
    )
    assert 0.0 <= priors["Above the Break 3"].mean <= 1.0


def test_history_is_reported_separately_from_traits():
    """
    The player's own prior-season share is not a trait and must not appear in
    the factor list. Shot diet repeats at r=0.94 season over season, so left in
    the list it would be the top "reason" for every shot on the floor and
    nothing about his actual game would ever surface.
    """
    from src.inference.explain import HISTORY_FEATURE

    model, metadata, cols = _toy_with_history()
    frame = _row("Restricted Area", cols, **{HISTORY_FEATURE: 0.40})
    out = explain_attainability(model, metadata, frame, "Restricted Area")

    assert all(f["feature"] != HISTORY_FEATURE for f in out["factors"])
    assert out["history"]["prior_share"] == pytest.approx(0.40)


def test_the_summary_attributes_the_history_term_to_history():
    """
    An earlier draft read "...and his game pulls him off it" while quoting the
    prior-season contribution — naming the player's game as the cause of a
    number that came from his own history. Wrong cause, confidently stated.
    """
    model, metadata, cols = _toy_with_history()
    from src.inference.explain import HISTORY_FEATURE

    out = explain_attainability(
        model, metadata, _row("Restricted Area", cols, **{HISTORY_FEATURE: 0.40}),
        "Restricted Area",
    )
    if abs(out["history"]["effect"]) >= 0.01:
        # Names his own record as the cause, in whichever window it came from.
        assert out["history"]["window"] in out["summary"]
        assert "his game keeps him there" not in out["summary"]
        assert "his game pulls him off it" not in out["summary"]


def test_a_tiny_share_does_not_render_as_zero():
    """
    Giannis takes 0.2% of his shots from the left corner. Printed as "0%" in a
    sentence that also quotes a non-zero projection, the sentence reads as
    broken arithmetic.
    """
    from src.inference.explain import _share_text

    assert _share_text(0.002) == "0.2%"
    assert _share_text(0.0) == "0%"
    assert _share_text(0.28) == "28%"


def _toy_with_history():
    """The toy model, refit with the prior-season share included."""
    from src.inference.explain import HISTORY_FEATURE

    rng = np.random.default_rng(1)
    rows = []
    for _ in range(800):
        zone = SUB_ZONES[rng.integers(len(SUB_ZONES))]
        prior = float(rng.uniform(0, 0.5))
        rows.append({
            "zone": zone, "position": "G",
            "height": 76.0, "weight": 200.0, "wingspan": 80.0,
            "pullup_share": 0.5, "catch_shoot_share": 0.5,
            HISTORY_FEATURE: prior,
            # target tracks history closely, as it genuinely does in the data
            "zone_share": 0.9 * prior + 0.02,
        })
    df = encode_zone_and_position(pd.DataFrame(rows))
    cols = (["height", "weight", "wingspan", "pullup_share", "catch_shoot_share"]
            + [f"zone_is_{s}" for s in SUB_ZONE_SUFFIX.values()]
            + ["is_three"]
            + [f"pos_is_{b}" for b in POSITION_BUCKETS]
            + [HISTORY_FEATURE])
    model = xgb.XGBRegressor(n_estimators=60, max_depth=4, random_state=0)
    model.fit(df[cols], df["zone_share"])
    metadata = {
        "feature_cols": cols,
        "position_buckets": POSITION_BUCKETS,
        "league_zone_shares": {z: 0.1 for z in SUB_ZONES},
        "feature_reference": {},
    }
    return model, metadata, cols


def test_the_history_window_is_named_correctly():
    """
    The reported figure comes from this season when there is evidence and last
    season when there is not. An earlier draft quoted a season-to-date rate and
    called it "last season" — a confident mislabel of the exact number the
    reader is looking at.
    """
    from src.inference.explain import HISTORY_FALLBACK, HISTORY_FEATURE

    model, metadata, cols = _toy_with_history()

    current = _row("Restricted Area", cols,
                   **{HISTORY_FEATURE: 0.40, HISTORY_FALLBACK: 0.10})
    out = explain_attainability(model, metadata, current, "Restricted Area")
    assert out["history"]["window"] == "this season"
    assert out["history"]["prior_share"] == pytest.approx(0.40)
    if abs(out["history"]["effect"]) >= 0.01:
        assert "this season" in out["summary"]
        assert "last season" not in out["summary"]

    # No current-season evidence: fall back and say so.
    lapsed = _row("Restricted Area", cols,
                  **{HISTORY_FEATURE: np.nan, HISTORY_FALLBACK: 0.10})
    out = explain_attainability(model, metadata, lapsed, "Restricted Area")
    assert out["history"]["window"] == "last season"
    assert out["history"]["prior_share"] == pytest.approx(0.10)


def test_summary_never_quotes_a_factor_whose_value_is_unknown():
    """
    A factor with no value is not evidence. Quoting one produced sentences like
    "mostly teammates' ball movement (unknown)" — citing a quantity in the same
    breath as admitting it is not known, which reads as a rendering fault and
    undercuts the reasons that ARE backed by data.

    It still belongs in the ranked factor list, where its contribution is shown
    with an explicit "unknown" value; it just must not be narrated.
    """
    from src.inference.explain import _driver_clause

    known = {"label": "Pull-up share", "display_value": "50%",
             "value": 0.5, "impact": 0.03, "direction": "raises"}
    unknown = {"label": "Teammates' ball movement", "display_value": "unknown",
               "value": None, "impact": 0.04, "direction": "raises"}

    clause = _driver_clause([unknown, known])
    assert "unknown" not in clause
    assert "pull-up share" in clause.lower()

    # Nothing quotable at all is silence, not a sentence naming an unknown.
    assert _driver_clause([unknown]) == ""


def test_league_zone_shares_has_a_bare_zone_entry_for_angle_split_zones():
    """
    Regression test for the same class of bug as the diet-history and
    creation-priors ones above, found in a THIRD function:
    `fit_attainability`'s `league_zone_shares` was keyed ONLY by split
    sub-zone names for the two angle-split zones, so `explain_attainability`
    — which is called with the bare zone name whenever the caller hasn't
    resolved exact court coordinates, the common case for a zone-level
    attainability display — got a silent `None` for `league_zone_share` on
    exactly the two highest-volume zones (Above the Break 3, Mid-Range).
    The attainability number itself was still fine; only the "compared to a
    league-average X%" figure alongside it silently disappeared.
    """
    from src.training.attainability import with_bare_zone_entries

    zone_means = pd.Series({
        "Above the Break 3 (centre)": 0.10,
        "Above the Break 3 (wing)": 0.18,
        "Mid-Range (centre)": 0.05,
        "Mid-Range (wing)": 0.07,
        "Restricted Area": 0.30,
    })
    out = with_bare_zone_entries(zone_means)

    assert out["Above the Break 3"] == pytest.approx(0.28)
    assert out["Mid-Range"] == pytest.approx(0.12)
    # Untouched zones (no split, already a real key) pass through unchanged.
    assert out["Restricted Area"] == pytest.approx(0.30)
    # The original sub-zone entries still exist — a caller with exact
    # coordinates can still ask for the more precise number.
    assert out["Above the Break 3 (wing)"] == pytest.approx(0.18)


# ═══════════════════════════════════════════════════════════════════════════
# Shot-quality (make-probability) explanation
#
# Regression coverage for a real bug: explain_shot_quality summed TreeSHAP
# contributions and reported the raw sum as `make_probability` directly.
# For a `binary:logistic` model those contributions are additive in
# LOG-ODDS, not probability — the sum has to go through a sigmoid first.
# Nothing here caught that before: explain_shot_quality had zero test
# coverage prior to this file. The bug was invisible whenever the true
# probability happened to be near 0.5 (margin near zero, so skipping the
# sigmoid barely mattered) and produced nonsense otherwise — a real query
# (Stephen Curry, above the break three) reported 0% when the model's own
# predict_proba said 43%, because the summed margin was -0.30 and got
# clipped straight to zero instead of passed through a sigmoid.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def toy_classifier():
    """
    A small binary:logistic classifier fit so its own decision boundary is
    non-trivial (not everything clustered near p=0.5) — the bug this guards
    against is invisible near p=0.5, where skipping the sigmoid changes
    little, and only obvious away from it.
    """
    rng = np.random.default_rng(0)
    n = 600
    rows = []
    for _ in range(n):
        zone_rate = float(rng.uniform(0.2, 0.7))
        def_fg_pct_zone = float(rng.uniform(0.2, 0.7))
        matchup_advantage = zone_rate - def_fg_pct_zone
        score_diff = float(rng.integers(-15, 15))
        made = rng.uniform(0, 1) < np.clip(0.5 + 1.2 * matchup_advantage, 0.02, 0.98)
        rows.append({
            "zone_rate": zone_rate,
            "def_fg_pct_zone": def_fg_pct_zone,
            "matchup_advantage": matchup_advantage,
            "score_diff": score_diff,
            "made": int(made),
        })
    df = pd.DataFrame(rows)
    cols = ["zone_rate", "def_fg_pct_zone", "matchup_advantage", "score_diff"]

    model = xgb.XGBClassifier(n_estimators=60, max_depth=4, random_state=0)
    model.fit(df[cols], df["made"])
    return model, cols


def test_make_probability_matches_predict_proba_not_the_raw_shap_sum(toy_classifier):
    model, cols = toy_classifier
    # A row chosen to sit well away from p=0.5, on the low side — exactly
    # the regime where summing log-odds contributions as if they were
    # probabilities clips to 0 instead of reporting the real number.
    row = pd.DataFrame([{
        "zone_rate": 0.25, "def_fg_pct_zone": 0.68,
        "matchup_advantage": -0.43, "score_diff": 3,
    }])

    truth = float(model.predict_proba(row[cols])[0, 1])
    out = explain_shot_quality(model, cols, row)

    assert out["make_probability"] == pytest.approx(truth, abs=1e-4)
    # The failure mode this guards against: reporting the un-sigmoided
    # log-odds sum clipped into [0, 1] instead of the real probability.
    assert out["make_probability"] != pytest.approx(0.0, abs=1e-6)


def test_offense_only_probability_is_a_valid_probability(toy_classifier):
    model, cols = toy_classifier
    row = pd.DataFrame([{
        "zone_rate": 0.65, "def_fg_pct_zone": 0.22,
        "matchup_advantage": 0.43, "score_diff": -8,
    }])
    out = explain_shot_quality(model, cols, row)
    assert 0.0 <= out["offense_only_probability"] <= 1.0
    assert 0.0 <= out["make_probability"] <= 1.0


def test_per_feature_impact_is_a_probability_delta_not_a_log_odds_contribution(toy_classifier):
    """
    A factor's `impact` is printed in the UI as percentage points of make
    probability (e.g. "+4.2pp"). It must live on that scale — the log-odds
    contribution underneath it can be arbitrarily larger or smaller than
    the probability move it actually corresponds to once the model is away
    from p=0.5.
    """
    model, cols = toy_classifier
    row = pd.DataFrame([{
        "zone_rate": 0.25, "def_fg_pct_zone": 0.68,
        "matchup_advantage": -0.43, "score_diff": 3,
    }])
    out = explain_shot_quality(model, cols, row, top_n=10)
    all_factors = out["offense_factors"] + out["defense_factors"]
    assert all_factors  # the toy model's features must actually show up
    for f in all_factors:
        assert -1.0 <= f["impact"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# defender_zone_tendency_note — the attacker/defender comparison
# attainability was missing entirely: the model itself has no defender in
# it by design (a season-long shot-diet frequency, not a single matchup),
# but a named-matchup explanation can and should say whether THIS defender's
# opponents attack a zone more or less than a typical defender's do.
# ═══════════════════════════════════════════════════════════════════════════

def test_defender_tendency_note_is_none_without_frequency_data():
    assert defender_zone_tendency_note("Luka", 0.3, "Wemby", None, 0.25, "Restricted Area") is None
    assert defender_zone_tendency_note("Luka", 0.3, "Wemby", 0.25, None, "Restricted Area") is None


def test_defender_tendency_note_flags_above_average_exposure():
    note = defender_zone_tendency_note(
        "Luka", 0.30, "Wemby", defender_freq=0.45, league_freq=0.25,
        zone_label="Restricted Area",
    )
    assert note is not None
    assert "MORE" in note
    assert "Wemby" in note and "Luka" in note
    assert "45%" in note and "25%" in note and "30%" in note


def test_defender_tendency_note_flags_below_average_exposure():
    note = defender_zone_tendency_note(
        "Luka", 0.30, "Wemby", defender_freq=0.10, league_freq=0.25,
        zone_label="Restricted Area",
    )
    assert note is not None
    assert "LESS" in note


def test_defender_tendency_note_is_neutral_within_the_negligible_band():
    note = defender_zone_tendency_note(
        "Luka", 0.30, "Wemby", defender_freq=0.26, league_freq=0.25,
        zone_label="Restricted Area",
    )
    assert note is not None
    assert "MORE" not in note and "LESS" not in note
