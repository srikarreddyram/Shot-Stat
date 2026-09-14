"""
FEATURE_GROUPS — every model feature, grouped by what it describes. The
grouping is not cosmetic: `backtest.py` ablates whole groups to report what
each one is actually worth, `train.py` uses it to build a player-identity-
free model for the hierarchical residual stage, and the API surfaces it
directly at /stats/features.
"""
from src.features.creation import CREATION_FEATURE_COLS
from src.features.point_in_time import ZONE_SUFFIX
from .classification import FINISH_TYPES, POSSESSION_ORIGINS
from .spatial import SPATIAL_FEATURES

FEATURE_GROUPS: dict[str, list[str]] = {
    "spatial": [
        "loc_x", "loc_y", "shot_distance", "shot_angle",
        "distance_from_center", "abs_loc_x", "is_three",
    ],
    # Smooth radial basis over the floor. Held out by default: measured, it
    # makes the model slightly WORSE (log-loss 0.6327 -> 0.6335, AUC 0.6847 ->
    # 0.6836, ECE 0.0078 -> 0.0096 on the 2025-26 hold-out). Thirty-six extra
    # columns is a lot of surface for the trees to overfit, and distance plus
    # the zone indicators already carry the spatial signal that exists.
    #
    # It was added to smooth the fitted surface, since axis-aligned splits make
    # a blocky one — but the court render interpolates between grid points
    # anyway, so the visual benefit was already being delivered downstream at
    # no cost to accuracy. Enable with `--spatial-basis` to re-measure.
    "spatial_basis": SPATIAL_FEATURES,
    "context": [
        "quarter", "time_remaining", "score_diff", "home_away",
        "playoff_flag", "clutch_flag", "seconds_remaining_in_game",
        "rest_days", "is_back_to_back", "opp_def_rating",
        # This player's own clutch-shooting delta vs. his normal baseline
        # (career-to-date, shrunk), engaged only on shots that are
        # themselves clutch — see point_in_time.clutch_performance. Zero on
        # every non-clutch shot by construction, so it can only ever move a
        # clutch-situation prediction.
        "clutch_edge",
    ],
    "shooter_physical": [
        "height", "weight", "wingspan",
    ],
    "shooter_skill": [
        "zone_rate", "zone_att", "overall_rate", "three_rate",
        "recent_10_fg", "recent_20_fg", "zone_rate_vs_league",
    ] + [f"zone_rate_{s}" for s in ZONE_SUFFIX.values()],
    "creation": CREATION_FEATURE_COLS + ["creation_is_prior"],
    "defender": [
        "def_fg_pct_overall", "def_pct_plusminus",
        "def_freq_zone", "def_fg_pct_zone", "def_pct_plusminus_zone",
        "def_matchup_share", "def_matchup_dispersion",
    ],
    # Defender size, held out of the model by default.
    #
    # These measure nothing. Actual FG% by attacker-minus-defender height is
    # flat across every populated band — 0.361 at -6ft to -3ft against 0.351 at
    # +3 to +6 above the break, about one point over the whole range — because
    # `height_diff` is computed against the possession-weighted AVERAGE
    # defender, which sits near the league mean almost always. The extreme
    # bands hold 4 to 25 rows out of two million.
    #
    # The model fitted those handful of rows anyway, and the recommender drove
    # straight into them: naming a tall defender against a guard moved the
    # prediction by 18 points, of which 17.8 came from these features and 0.9
    # from every genuine defensive-quality feature combined. A spurious effect
    # twenty times the size of the real one, triggered by the single most
    # obvious thing a user can do in the UI.
    #
    # Enable with `--defender-physicals` to reproduce the old behaviour.
    "defender_physical": [
        "def_height", "def_weight", "def_wingspan",
        "height_diff", "weight_diff", "wingspan_diff", "size_mismatch",
    ],
    "interaction": [
        "matchup_advantage", "expected_contest", "creation_edge",
        "openness_vs_defender",
    ],
    # Per-shot play-by-play context. `is_assisted` is deliberately ABSENT —
    # assists are credited only on made baskets, so it predicts the label
    # perfectly and is useless at prediction time. See pbp_ingestor.py and
    # tests/test_no_leaky_features.py.
    "shot_context": [
        "is_putback", "seconds_since_prev_event",
    ],
    # Possession origin indicators, grouped separately so `--no-origin` style
    # ablations can measure them on their own.
    "possession_origin": [f"origin_{o}" for o in POSSESSION_ORIGINS],
    # What the shot physically was — dunk, layup, hook, floater, jumper —
    # kept on its own axis from how it was created. See FINISH_RULES.
    "finish": [f"finish_{f}" for f in FINISH_TYPES],
    # Point-in-time opponent defence per zone. The model previously knew about
    # opponent defence only through one season-level `def_rating`, and residuals
    # aggregated by defending team showed it explaining essentially none of it
    # (z-score sd 2.82 at the rim against 1.0 for a correct model).
    "opponent_defence": [
        "opp_zone_def_rate", "opp_zone_def_att",
    ] + [f"opp_def_rate_{s}" for s in ZONE_SUFFIX.values()],
    # How contested this shooter's looks have actually been, accumulated over
    # strictly prior games (see point_in_time.build_contest_history).
    #
    # Contest level is the largest single thing the model cannot see. Per-SHOT
    # closest-defender distance is not published anywhere — `shotchartdetail`
    # accepts a CloseDefDistRange parameter and silently ignores it — and that
    # remains the hard noise floor on whether a given shot goes in.
    #
    # What these carry is the tendency one level up: LeagueDashPlayerPtShot
    # does honour a date filter, so the four defender-distance bands are
    # available per GAME and can therefore be accumulated point-in-time. The
    # creation group already has `avg_def_dist` / `open_share` / `tight_share`
    # from the same bands, but only as a season aggregate lagged a full year —
    # it cannot say a player has been run off the line for the past month.
    "contest": [
        "contest_car_very_tight", "contest_car_tight",
        "contest_car_open", "contest_car_wide_open",
        "contest_ssn_very_tight", "contest_ssn_tight",
        "contest_ssn_open", "contest_ssn_wide_open",
        "contest_car_sep", "contest_ssn_sep", "contest_att",
    ],
    # No longer reserved — src/ingestion/lineup_ingestor.py reconstructs the
    # on-court five from play-by-play substitution events, and
    # point_in_time.build_lineup_context computes these from it: the OTHER
    # four offensive teammates' creation/gravity profile, excluding the
    # shooter (who already has his own creation features) and unrelated to
    # cast_ast_rate/cast_efg (a whole-SEASON roster aggregate, not who was
    # literally on the floor for this possession).
    "team_creation": [
        "oncourt_off_creation", "oncourt_off_gravity",
        "oncourt_off_rim_pressure", "oncourt_off_n",
        # Peak-threat versions — see point_in_time.build_lineup_context for
        # why the mean alone dilutes a single elite teammate's effect to a
        # quarter strength.
        "oncourt_off_creation_max", "oncourt_off_gravity_max",
        "oncourt_off_rim_pressure_max", "oncourt_off_foul_rate_max",
    ],
    # The defensive mirror of team_creation: the other four defenders'
    # point-in-time quality, excluding the primary `defender_id` (already
    # its own feature group). Kept separate from "team_creation" so an
    # ablation can tell whether the offense-side or defense-side half of
    # "who else is on the floor" is doing any work, rather than one number
    # for both.
    "help_defense": [
        "oncourt_def_fg_pct", "oncourt_def_n", "oncourt_def_fg_pct_min",
        # Peak-threat defensive activity (blocks/steals/deflections and the
        # composite built from them) — a floor-presence deterrent effect on
        # WHETHER a shot is attempted, distinct from d_fg_pct's "how well was
        # a shot that was taken contested". See defensive_activity.py.
        "oncourt_def_blk_max", "oncourt_def_stl_max",
        "oncourt_def_deflections_max", "oncourt_def_gravity_max",
    ],
}
