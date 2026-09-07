"""
Recommendation engine — ranks court locations for a player/defender matchup.

Rewritten around three changes.

1. It shares the training code path
------------------------------------
Every feature is now produced by `src.features.spec.derive_features`, the same
function that builds the training matrix. The previous version rebuilt forty
features by hand in a Python dict that had to be kept in lockstep with a large
SQL query by eye; any drift between them was a silent accuracy bug that no
test could catch. `tests/test_train_serve_parity.py` now asserts the two paths
agree on real historical shots.

2. It answers the question the product is actually asking
----------------------------------------------------------
The model estimates P(make | a shot was taken here). Ranking by that alone
recommends the restricted area to every player on earth, because the rim is
the most efficient spot on the floor for everyone and the entire difficulty is
getting there. So expected points are now paired with an attainability
estimate — how readily this specific player can generate this specific look,
which is largely a question of handle (see `src/training/attainability.py`).
The two are reported separately and combined into the ranking score, so the
interface can distinguish "this would be a great shot for you" from "this is a
shot you can get".

3. It admits what it does not know
-----------------------------------
Every projection carries a credible interval derived from how many attempts
actually back it. A corner-three number built on nine attempts and one built
on nine hundred used to render identically. They no longer do.

Usage:
    from src.inference.recommender import ShotRecommender
    rec = ShotRecommender()
    out = rec.recommend(player_id="1629029", defender_id="203999")
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.features.creation import CREATION_FEATURE_COLS
from src.features.mechanics import (
    expand_grid_over_mechanics,
    league_zone_mix,
    marginalize,
    player_zone_mix,
)
from src.features.point_in_time import (
    ZONE_SUFFIX,
    apply_hierarchy,
    league_average_defender,
    lookup_defender_category_rates,
    lookup_prior_counts,
    lookup_recent_form,
    lookup_supporting_cast,
    lookup_zone_creation,
)
from src.features.shrinkage import BetaPrior, posterior_interval
from src.inference.explain import creation_note, explain_attainability
from src.training.attainability import (
    attach_sub_zone,
    encode_zone_and_position,
    lookup_diet_history,
)
from src.features.spec import (
    ZONE_TO_DEF_CATEGORY,
    as_model_matrix,
    derive_features,
)
from src.db.database import get_engine
from src.training.calibration import Calibrator

MODEL_DIR = Path(config.PROJECT_ROOT) / "models"

# How much of a shooter's possessions a specifically-named primary defender is
# taken to account for. Chosen against the training distribution of
# `def_matchup_share` (mean 0.38, p99 0.71): naming a defender means a
# concentrated assignment, but not an exclusive one, because exclusive
# assignments essentially do not occur.
PRIMARY_DEFENDER_SHARE = 0.60
PRIMARY_DEFENDER_DISPERSION = 0.55

ZONE_POINTS = {
    "Restricted Area": 2, "In The Paint (Non-RA)": 2, "Mid-Range": 2,
    "Left Corner 3": 3, "Right Corner 3": 3, "Above the Break 3": 3,
}


# Observed shot-distance range (feet) per zone, from every shot 2016-17
# onward. The analytic grid below can otherwise place candidates outside the
# region where shots of that type are actually attempted — restricted-area
# points nearly five feet out, for instance — and the model has no training
# signal there, so it extrapolates. Filtering to the observed envelope keeps
# every scored location one the model has genuinely seen.
ZONE_DISTANCE_RANGE = {
    "Restricted Area": (0.0, 3.0),
    "In The Paint (Non-RA)": (4.0, 15.0),
    "Mid-Range": (8.0, 23.0),
    "Left Corner 3": (22.0, 26.0),
    "Right Corner 3": (22.0, 26.0),
    "Above the Break 3": (23.0, 32.0),
}


def _generate_court_grid() -> list[dict]:
    """
    Candidate shot locations across all six zones.

    Coordinates are NBA shot-chart units: the basket is (0, 0), ten units to
    the foot, y increasing toward half court. Zone membership is assigned by
    construction here rather than recomputed from coordinates, so a grid point
    can never disagree with the zone whose statistics it is scored against.
    """
    grid: list[dict] = []

    def add(loc_x, loc_y, zone, shot_type):
        distance = round(float(np.hypot(loc_x, loc_y)) / 10.0, 1)
        low, high = ZONE_DISTANCE_RANGE[zone]
        if not (low <= distance <= high):
            return
        grid.append({
            "loc_x": float(loc_x),
            "loc_y": float(loc_y),
            "shot_distance": distance,
            "zone": zone,
            "shot_type": shot_type,
        })

    for x in np.linspace(-35, 35, 8):
        for y in np.linspace(2, 35, 4):
            if np.hypot(x, y) <= 42:
                add(x, y, "Restricted Area", "2PT Field Goal")

    for x in np.linspace(-75, 75, 7):
        for y in np.linspace(40, 90, 5):
            if 42 < np.hypot(x, y) <= 100 and abs(x) <= 80:
                add(x, y, "In The Paint (Non-RA)", "2PT Field Goal")

    for angle in np.linspace(10, 170, 12):
        for r in (100, 130, 160, 190):
            x, y = r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))
            d = np.hypot(x, y) / 10.0
            if 9.0 < d < 22.0 and y > 0:
                add(x, y, "Mid-Range", "2PT Field Goal")

    for x in np.linspace(-235, -220, 4):
        for y in np.linspace(5, 85, 5):
            if np.hypot(x, y) / 10.0 >= 22.0 and y <= 93:
                add(x, y, "Left Corner 3", "3PT Field Goal")

    for x in np.linspace(220, 235, 4):
        for y in np.linspace(5, 85, 5):
            if np.hypot(x, y) / 10.0 >= 22.0 and y <= 93:
                add(x, y, "Right Corner 3", "3PT Field Goal")

    for angle in np.linspace(15, 165, 14):
        for r in (237, 250, 270):
            x, y = r * np.cos(np.radians(angle)), r * np.sin(np.radians(angle))
            if np.hypot(x, y) / 10.0 >= 23.0 and y > 93:
                add(x, y, "Above the Break 3", "3PT Field Goal")

    return grid


SHOT_GRID = _generate_court_grid()


class ShotRecommender:
    """Loads the trained models and scores court locations for a matchup."""

    def __init__(self, model_name: str = "shot-quality-v13",
                 attainability_name: str = "attainability-pit",
                 model_dir: str | Path = MODEL_DIR):
        model_dir = Path(model_dir)

        metadata_path = model_dir / f"metadata_{model_name}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata at {metadata_path}. Train first: "
                f"python -m src.training.train --name {model_name}"
            )
        self.metadata = json.loads(metadata_path.read_text())
        self.feature_cols: list[str] = self.metadata["feature_cols"]
        self.league_zone_rates: dict = self.metadata["league_zone_rates"]

        # Priors travel with the model rather than being refit here. Refitting
        # at serving time from whatever the database currently holds would
        # apply a different shrinkage than the model was trained under, and the
        # drift would be invisible.
        self.zone_priors = {
            zone: BetaPrior(mean=p["mean"], strength=p["strength"],
                            n_players=p.get("n_players", 0),
                            n_attempts=p.get("n_attempts", 0))
            for zone, p in self.metadata["zone_priors"].items()
        }
        # Same discipline for the point-in-time defender-quality priors
        # (`lookup_defender_category_rates`). Older metadata files predate
        # this field, hence the empty-dict fallback.
        self.category_priors = {
            cat: BetaPrior(mean=p["mean"], strength=p["strength"],
                           n_players=p.get("n_players", 0),
                           n_attempts=p.get("n_attempts", 0))
            for cat, p in self.metadata.get("category_priors", {}).items()
        }
        # Populated from the attainability metadata below, once it is loaded.
        self.creation_priors: dict[str, BetaPrior] = {}
        self.sub_zone_priors: dict[str, BetaPrior] = {}
        self.max_season_progress: float = 0.7

        # Per-feature training range, used to stop the serving path from
        # extrapolating. See `_clip_to_training_range`.
        self.feature_bounds = self.metadata.get("feature_bounds", {})

        # Whether this model consumes per-shot mechanics. If it does, the
        # serving path has to supply one per row — a hypothetical shot has no
        # mechanic of its own, so the prediction is marginalised over the
        # player's mix. See src/features/mechanics.py.
        self.uses_mechanics = any(
            c.startswith("mech_") for c in self.feature_cols
        )

        self.hierarchical = self.metadata.get("hierarchical_split", False)
        if self.hierarchical:
            self.models = {}
            for label in ("interior", "perimeter"):
                m = xgb.XGBClassifier()
                m.load_model(str(model_dir / f"xgb_{label}_{model_name}.json"))
                self.models[label] = m
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_dir / f"xgb_{model_name}.json"))

        self.calibration_adopted = self.metadata.get("calibration_adopted", False)
        self.calibrators = {
            k: Calibrator.from_dict(v)
            for k, v in self.metadata.get("calibrators", {}).items()
        } if self.calibration_adopted else {}

        # Attainability is optional: the engine still works without it, it just
        # cannot separate "good shot" from "gettable shot" and says so.
        self.attainability = None
        att_meta_path = model_dir / f"metadata_{attainability_name}.json"
        if att_meta_path.exists():
            self.att_metadata = json.loads(att_meta_path.read_text())
            self.attainability = xgb.XGBRegressor()
            self.attainability.load_model(
                str(model_dir / f"xgb_{attainability_name}.json")
            )
            self.creation_priors = {
                zone: BetaPrior(mean=p["mean"], strength=p["strength"],
                                n_players=p.get("n_players", 0),
                                n_attempts=p.get("n_attempts", 0))
                for zone, p in self.att_metadata.get("creation_priors", {}).items()
            }
            # Shrinkage for a partial season, and the furthest point through a
            # season the model was actually trained at.
            self.sub_zone_priors = {
                zone: BetaPrior(mean=p["mean"], strength=p["strength"])
                for zone, p in self.att_metadata.get("sub_zone_priors", {}).items()
            }
            self.max_season_progress = float(
                self.att_metadata.get("max_season_progress", 0.7)
            )

        self.engine = get_engine()
        self.model_name = model_name

        self._league_mix = None

        print(f"✓ ShotRecommender ready — model={model_name}, "
              f"features={len(self.feature_cols)}, "
              f"calibrated={self.calibration_adopted}, "
              f"attainability={'yes' if self.attainability is not None else 'no'}, "
              f"mechanics={'marginalised' if self.uses_mechanics else 'n/a'}")

    # ── Player state lookup ──────────────────────────────────────────────

    def _player_row(self, player_id: str, season: str, as_of_date=None) -> dict:
        """
        Everything known about the shooter as of `as_of_date`, in exactly the
        raw-column form the training builder produces.

        The point-in-time counts come from `lookup_prior_counts`, whose output
        keys match the training path's, so `apply_hierarchy` — shared with
        training — turns them into rates identically.
        """
        with self.engine.connect() as conn:
            attrs = conn.execute(text("""
                SELECT name, height, weight, wingspan, position
                FROM players WHERE player_id = :pid AND season <= :season
                ORDER BY season DESC LIMIT 1
            """), {"pid": str(player_id), "season": season}).fetchone()

            if attrs is None:
                raise ValueError(f"Player {player_id} not found at or before {season}")

            counts = lookup_prior_counts(conn, player_id, as_of_date=as_of_date)
            recent = lookup_recent_form(conn, player_id, as_of_date=as_of_date)

        row = {
            "player_id": str(player_id),
            "name": attrs[0],
            "height": attrs[1],
            "weight": attrs[2],
            "wingspan": attrs[3],
            "position": attrs[4],
            "season": season,
        }
        row.update({k: v for k, v in counts.items() if not k.startswith("_")})
        row.update(recent)
        row["_latest_season"] = counts.get("_latest_season")
        return row

    def _creation_row(self, player_id: str, season: str) -> dict:
        """
        The player's most recent creation profile strictly before `season`.

        Lagged for the same reason training lags it: the current season's
        tracking totals do not exist mid-season, and using them would make the
        serving feature a different object than the training feature.
        """
        from src.features.creation import load_creation_profiles

        if not hasattr(self, "_creation_cache"):
            self._creation_cache = load_creation_profiles(self.engine)

        profiles = self._creation_cache
        rows = profiles[
            (profiles["player_id"] == str(player_id)) & (profiles["season"] < season)
        ]
        if rows.empty:
            return {}
        latest = rows.sort_values("season").iloc[-1]
        return {c: latest[c] for c in CREATION_FEATURE_COLS if c in latest.index}

    def _defender_row(self, defender_id: str, season: str, as_of_date=None) -> dict:
        """
        Defender physicals and per-category defensive quality, as of
        `as_of_date` — point-in-time, via `lookup_defender_category_rates`,
        rather than reading the whole-season `defender_stats` aggregate the
        way this used to. See `point_in_time.build_defender_category_rates`
        for why that was a leak: a shot contested by a defender used to
        contribute to that same defender's own season FG%-allowed, which was
        then fed back in as a feature describing that shot.
        """
        with self.engine.connect() as conn:
            attrs = conn.execute(text("""
                SELECT height, weight, wingspan, name, position FROM players
                WHERE player_id = :pid AND season <= :season
                ORDER BY season DESC LIMIT 1
            """), {"pid": str(defender_id), "season": season}).fetchone()

            if attrs is None:
                return {}

            rates = lookup_defender_category_rates(
                conn, defender_id, self.category_priors, as_of_date=as_of_date
            )

        by_category = rates["by_category"]
        # Physicals are exposed under BOTH naming conventions: `def_*` is what
        # the feature assembly reads, while the bare names are what
        # `_combine_defenders` operates on. Keeping one dict with both avoids a
        # translation step that could silently drop an attribute.
        return {
            "def_height": attrs[0],
            "def_weight": attrs[1],
            "def_wingspan": attrs[2],
            "height": attrs[0],
            "weight": attrs[1],
            "wingspan": attrs[2],
            "name": attrs[3],
            "position": attrs[4],
            "_by_category": by_category,
            "def_stats": by_category,
        }

    @staticmethod
    def _combine_defenders(primary: dict, secondary: dict) -> dict:
        """
        Merge two defenders into one effective defender for a double team.

        There is no genuinely double-teamed shot data to learn a real effect
        from — no row in this database records two defenders on one shot — so
        this stays a deliberate, clearly-labelled heuristic floor rather than a
        learned interaction:

          - Physicals take the larger of the two, since a shooter facing a
            double team has to account for whichever defender presents the
            longer contest.
          - Defensive quality takes, per category, whichever defender allows
            the LOWER FG%: the shooter faces at least his toughest individual
            defender's resistance. A conservative floor, since nothing here
            models genuine double-team suppression beyond that.

        A missing physical is never filled from the other defender. Doing so
        would substitute a different, likely smaller player's real measurement
        for this one's unmeasured reach — which once made a double team look
        EASIER to score against than facing the tougher defender alone. Same
        "exact data or nothing" rule the rest of the pipeline follows.
        """
        combined = dict(primary)
        for attr in ("height", "weight", "wingspan"):
            p_val, s_val = primary.get(attr), secondary.get(attr)
            if p_val is not None and s_val is not None:
                combined[attr] = max(p_val, s_val)

        # Accepts either shape: the internal `_by_category` used by
        # `_defender_row`, or the `def_stats` key the public callers pass.
        key = "_by_category" if "_by_category" in primary else "def_stats"
        merged = dict(primary.get(key, {}))
        for category, s_stats in secondary.get(key, {}).items():
            p_stats = merged.get(category)
            if p_stats is None or p_stats.get("d_fg_pct") is None:
                merged[category] = s_stats
            elif (s_stats.get("d_fg_pct") is not None
                  and s_stats["d_fg_pct"] < p_stats["d_fg_pct"]):
                merged[category] = s_stats
        combined[key] = merged

        return combined

    # ── Scoring ──────────────────────────────────────────────────────────

    def _clip_to_training_range(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clamp every feature to the 1st-99th percentile range it occupied during
        training.

        The recommender is a hypothetical generator: it invents feature vectors
        that never occurred, by design — this player, from this spot, against
        that defender. Nothing stops it inventing one outside the region the
        model was fitted on, and a gradient-boosted tree does not degrade
        gracefully there. It holds whatever value the outermost leaf learned and
        applies it with full confidence, so an out-of-range input does not
        produce a slightly-worse estimate, it produces an arbitrary one.

        That is what turned "Curry guarded by Wembanyama" into a 21-point
        mid-range collapse. Clipping makes the worst case "the most extreme
        matchup the data actually contains", which is a defensible answer, and
        it costs nothing on real historical rows because 98% of them are inside
        the bounds by construction.
        """
        if not self.feature_bounds:
            return X
        clipped = X.copy()
        for col, (lo, hi) in self.feature_bounds.items():
            if col in clipped.columns:
                clipped[col] = clipped[col].clip(lo, hi)
        return clipped

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        X = self._clip_to_training_range(X)
        if self.hierarchical:
            out = np.zeros(len(X))
            interior_cols = [c for c in ("zone_is_restricted_area", "zone_is_paint")
                             if c in X.columns]
            m = (X[interior_cols].sum(axis=1).values > 0
                 if interior_cols else np.zeros(len(X), dtype=bool))
            if m.any():
                out[m] = self.models["interior"].predict_proba(X[m])[:, 1]
            if (~m).any():
                out[~m] = self.models["perimeter"].predict_proba(X[~m])[:, 1]
            if self.calibration_adopted:
                out[m] = self.calibrators["interior"].predict(out[m])
                out[~m] = self.calibrators["perimeter"].predict(out[~m])
            return out

        raw = self.model.predict_proba(X)[:, 1]
        if self.calibration_adopted and "all" in self.calibrators:
            return self.calibrators["all"].predict(raw)
        return raw

    def _cast_row(self, player_id: str, season: str, as_of_date=None) -> dict:
        """
        The player's supporting cast, leave-one-out and season-to-date.

        See `point_in_time.build_supporting_cast` for why these exclude the
        player himself: a star's raw team numbers are mostly the star.
        """
        with self.engine.connect() as conn:
            return lookup_supporting_cast(
                conn, player_id, season, as_of_date=as_of_date
            )

    def _diet_row(self, player_id: str, season: str, as_of_date=None) -> dict:
        """
        Everything known about the player's shot diet right now: this season to
        date, his career before it, and his last completed season.

        Season-to-date is the strongest of the three by a distance — the first
        25% of a player's own season predicts his full-season diet better than
        all of the previous one — which is why this is a point-in-time lookup
        rather than a season-level join. See `attainability.PRIOR_DIET_COLS`.
        """
        with self.engine.connect() as conn:
            return lookup_diet_history(
                conn, player_id, season, self.sub_zone_priors,
                as_of_date=as_of_date, max_progress=self.max_season_progress,
            )

    def _attainability_frame(self, player_row: dict, creation: dict,
                             zones: pd.Series, cast: dict | None = None,
                             loc_x=None, loc_y=None,
                             prior_diet: dict | None = None) -> pd.DataFrame:
        """
        Assemble the attainability model's input rows.

        Split out from `_attainability` so the explainer scores the exact frame
        the prediction came from, rather than rebuilding it and risking an
        explanation of a slightly different row than the one served.

        `loc_x`/`loc_y` are what let `attach_sub_zone` split the two wide zones
        by angle. Without them every above-the-break three collapses to one
        bucket and the model cannot tell a dead-centre pull-up from a wing
        spot-up — which was the whole point of the sub-zone taxonomy.
        """
        cols = self.att_metadata["feature_cols"]

        frame = pd.DataFrame({"zone": zones.values})
        if loc_x is not None and loc_y is not None:
            frame["loc_x"] = np.asarray(loc_x, dtype=float)
            frame["loc_y"] = np.asarray(loc_y, dtype=float)
        frame["position"] = player_row.get("position")
        for key in ("height", "weight", "wingspan"):
            frame[key] = player_row.get(key)
        for key in CREATION_FEATURE_COLS:
            frame[key] = creation.get(key, np.nan)
        # Supporting cast. Constant across the grid — who a player's teammates
        # are does not depend on where he shoots from.
        for key, value in (cast or {}).items():
            frame[key] = value if value is not None else np.nan

        # Shared with the training matrix builder — the encoding is defined
        # once, in one place, for the same train/serve parity reason the
        # shot-quality features are.
        frame = encode_zone_and_position(frame)

        # Diet history is per SUB-zone, so it is applied after the sub-zone is
        # resolved rather than broadcast like the cast figures.
        if prior_diet:
            per_zone = prior_diet.get("zones", {})
            for col in ("diet_to_date", "diet_att_to_date", "career_diet",
                        "career_diet_att", "prior_zone_share"):
                frame[col] = frame["sub_zone"].map(
                    lambda z: (per_zone.get(z) or {}).get(col)
                ).astype(float)
            frame["season_progress"] = prior_diet.get("season_progress", 0.0)

        for col in cols:
            if col not in frame.columns:
                frame[col] = np.nan
        return frame

    def _attainability(self, player_row: dict, creation: dict,
                       zones: pd.Series, cast: dict | None = None,
                       loc_x=None, loc_y=None,
                       prior_diet: dict | None = None) -> np.ndarray:
        """Predicted share of the player's shot diet coming from each sub-zone."""
        if self.attainability is None:
            return np.full(len(zones), np.nan)

        cols = self.att_metadata["feature_cols"]
        frame = self._attainability_frame(
            player_row, creation, zones, cast=cast, loc_x=loc_x, loc_y=loc_y,
            prior_diet=prior_diet,
        )
        return np.clip(
            self.attainability.predict(as_model_matrix(frame, cols)), 0.0, 1.0
        )

    def explain_attainability(self, player_id: str, zone: str,
                              season: str | None = None,
                              as_of_date=None, top_n: int = 4,
                              loc_x=None, loc_y=None) -> dict:
        """
        Why this player can or cannot get a shot in this zone.

        Returns the decomposition described in `src/inference/explain.py`:
        the zone's baseline share for any player, this player's deviation from
        it, and the ranked traits responsible.
        """
        if self.attainability is None:
            raise ValueError("No attainability model loaded")

        if season is None:
            with self.engine.connect() as conn:
                season = conn.execute(
                    text("SELECT MAX(season) FROM players")
                ).fetchone()[0]

        player = self._player_row(player_id, season, as_of_date=as_of_date)
        creation = self._creation_row(player_id, season)
        cast = self._cast_row(player_id, season, as_of_date=as_of_date)
        frame = self._attainability_frame(
            player, creation, pd.Series([zone]), cast=cast,
            loc_x=None if loc_x is None else [loc_x],
            loc_y=None if loc_y is None else [loc_y],
            prior_diet=self._diet_row(player_id, season, as_of_date=as_of_date),
        )
        # The sub-zone the shot actually resolved to, so the creation lookup
        # and the reported label describe the same half of the arc the model
        # scored rather than the blended parent zone.
        resolved_zone = str(frame.iloc[0].get("sub_zone", zone))

        out = explain_attainability(
            self.attainability, self.att_metadata, frame, zone, top_n=top_n
        )

        # Who generates shots here — the question attainability itself cannot
        # answer. A 26% attainability means very different things when the
        # player manufactures those looks himself and when they only exist
        # because a teammate found him.
        with self.engine.connect() as conn:
            out["creation"] = lookup_zone_creation(
                conn, player_id, resolved_zone, self.creation_priors,
                season=season, as_of_date=as_of_date,
            )
        out["creation"]["note"] = creation_note(out["creation"])

        out["player_id"] = player_id
        out["player_name"] = player.get("name")
        out["season"] = season
        out["sub_zone"] = resolved_zone
        return out

    def recommend(
        self,
        player_id: str,
        season: str | None = None,
        defender_id: str | None = None,
        secondary_defender_id: str | None = None,
        quarter: int = 1,
        time_remaining: float = 600.0,
        score_diff: int = 0,
        home_away: int = 1,
        playoff_flag: int = 0,
        rest_days: int = 1,
        is_back_to_back: int = 0,
        opp_def_rating: float = 112.0,
        as_of_date=None,
        top_n: int = 10,
        interval_level: float = 0.90,
    ) -> pd.DataFrame:
        """
        Score every grid location and return the top-N by attainability-weighted
        expected points.

        Columns returned:
            zone, loc_x, loc_y, make_probability, expected_points,
            ep_low / ep_high      credible interval from the attempts behind it
            attempts_behind       prior attempts supporting this player's rate
            attainability         predicted share of shot diet from this zone
            score                 the ranking quantity
            ep_vs_own_average     expected points relative to this player's
                                  own overall average — the number worth
                                  showing a user, since raw EP mostly restates
                                  that threes are worth more than twos
        """
        if season is None:
            with self.engine.connect() as conn:
                season = conn.execute(
                    text("SELECT MAX(season) FROM players")
                ).fetchone()[0]

        player = self._player_row(player_id, season, as_of_date=as_of_date)
        creation = self._creation_row(player_id, season)
        defender = (
            self._defender_row(defender_id, season, as_of_date=as_of_date)
            if defender_id else {}
        )
        if defender and secondary_defender_id:
            secondary = self._defender_row(
                secondary_defender_id, season, as_of_date=as_of_date
            )
            if secondary:
                defender = self._combine_defenders(defender, secondary)
                for attr in ("height", "weight", "wingspan"):
                    defender[f"def_{attr}"] = defender.get(attr)

        # League-average FG% allowed per defence category, the "everyone else"
        # half of the mixture blend below.
        with self.engine.connect() as conn:
            league_defence = league_average_defender(conn, season)
        league_zone_fg = {
            cat: stats.get("d_fg_pct")
            for cat, stats in league_defence["by_category"].items()
        }

        grid = pd.DataFrame(SHOT_GRID)

        # One row per (location, mechanic), weighted by how often this player
        # actually takes that kind of shot from that zone. Without this the
        # mechanic indicators would all be zero — a combination that appears in
        # no training row, since every real shot has exactly one mechanic.
        if self.uses_mechanics:
            if self._league_mix is None:
                self._league_mix = league_zone_mix(self.engine)
            mix = player_zone_mix(self.engine, str(player_id), season,
                                  league=self._league_mix)
            grid = expand_grid_over_mechanics(grid, mix)

        # ── Assemble raw columns, exactly as the training builder does ────
        raw = grid.copy()
        if self.uses_mechanics:
            # derive_features classifies `shot_subtype` into mech_* indicators.
            # The mechanic names ARE the classifier's own output vocabulary, so
            # round-tripping them through it reproduces the training encoding
            # exactly rather than setting the indicators by hand here.
            raw["shot_subtype"] = grid["_mechanic"]
        raw["player_id"] = player["player_id"]
        raw["season"] = season
        for key in ("height", "weight", "wingspan", "position"):
            raw[key] = player.get(key)

        game_state = {
            "quarter": quarter,
            "time_remaining": time_remaining,
            "score_diff": score_diff,
            "home_away": home_away,
            "playoff_flag": playoff_flag,
            "rest_days": rest_days,
            "is_back_to_back": is_back_to_back,
            "opp_def_rating": opp_def_rating,
        }
        for key, value in game_state.items():
            raw[key] = value

        # Point-in-time counts are identical for every grid point — the player's
        # history does not depend on where the hypothetical shot is taken.
        for key, value in player.items():
            if key.startswith("pit_"):
                raw[key] = value
        for key in ("recent_10_fg", "recent_20_fg"):
            raw[key] = player.get(key)

        for key in CREATION_FEATURE_COLS:
            raw[key] = creation.get(key, np.nan)
        raw["creation_is_prior"] = int(not creation)

        if not defender:
            # No defender named means "against a typical defender", which is
            # not the same statement as the NaNs the training matrix carries
            # for shots with no matchup data at all. Fill with league means so
            # the served prediction answers the question that was asked.
            with self.engine.connect() as conn:
                avg = league_average_defender(conn, season)
            defender = {
                "def_height": avg["height"],
                "def_weight": avg["weight"],
                "def_wingspan": avg["wingspan"],
                "_by_category": avg["by_category"],
                "_is_league_average": True,
            }

        def _mix_overall(value, league=0.0):
            """Blend one defender's figure toward the rest of the assignment."""
            if value is None:
                return None
            if league is None:
                league = 0.0
            blend = 0.37 if defender.get("_is_league_average") else PRIMARY_DEFENDER_SHARE
            return blend * value + (1.0 - blend) * league

        if defender:
            # Assignment concentration. These must describe the matchup the way
            # the TRAINING features do, which is the subtlety that made naming
            # an elite defender produce nonsense.
            #
            # Training builds defender features as a possession-weighted mixture
            # over everyone who guarded the shooter that game. Nobody guards one
            # player for a whole game: `def_matchup_share` averages 0.38 and
            # reaches only 0.71 at the 99th percentile. Setting it to 1.0 with
            # dispersion 0.0 — as this did — described a matchup that occurs in
            # 0.004% of training rows, and the model extrapolated wildly:
            # Curry's above-the-break probability against Wembanyama came out at
            # 0.141, less than half his unguarded 0.309, which no defender in
            # basketball has ever done to anyone.
            share = 0.37 if defender.get("_is_league_average") else PRIMARY_DEFENDER_SHARE
            raw["def_matchup_share"] = share
            raw["def_matchup_dispersion"] = (
                0.75 if defender.get("_is_league_average") else PRIMARY_DEFENDER_DISPERSION
            )
            # Physicals get the same mixture blend as the quality stats, and
            # for the same reason — this was the single largest error in the
            # served vector.
            #
            # `def_height` in training is the possession-weighted average height
            # of everyone who guarded the shooter, so it clusters near the
            # league mean and `height_diff` is almost always within +/-6 inches.
            # Feeding one defender's raw height produced Curry-vs-Wembanyama at
            # height_diff = -14, a value with 25 supporting rows in two million.
            # The model had fitted a ~20 point drop there off pure noise (the
            # measured effect across every populated band is ~1 point, and
            # flat), and the recommender drove straight into it every time a
            # tall defender was named against a guard.
            for attr in ("height", "weight", "wingspan"):
                raw[f"def_{attr}"] = _mix_overall(
                    defender.get(f"def_{attr}"), league_defence.get(attr)
                )

            by_category = defender.get("_by_category", {})
            overall = by_category.get("Overall", {})
            # Blended on the same terms as the zone-level figures below — these
            # were left raw in the first pass at this fix, which is why naming
            # an elite defender still moved mid-range by 21 points.
            raw["def_fg_pct_overall"] = _mix_overall(
                overall.get("d_fg_pct"), league_defence["by_category"]
                .get("Overall", {}).get("d_fg_pct")
            )
            raw["def_pct_plusminus"] = _mix_overall(overall.get("pct_plusminus"))

            categories = raw["zone"].map(ZONE_TO_DEF_CATEGORY)

            # The same mixture logic applied to the defender's quality. A
            # training row's `def_pct_plusminus` is `share * this defender +
            # (1 - share) * everyone else who guarded him`, and "everyone else"
            # averages out to roughly league-normal. Feeding one defender's raw
            # figure instead compares an individual against a distribution of
            # averages: Wembanyama's -0.099 sits past the 1st percentile of the
            # training feature, not because he is that much of an outlier as a
            # defender, but because averages are narrower than individuals.
            def _mix(value, league=0.0):
                if value is None:
                    return None
                if league is None:
                    league = 0.0
                return share * value + (1.0 - share) * league

            raw["def_fg_pct_zone"] = [
                _mix(by_category.get(c, {}).get("d_fg_pct"), league_zone_fg.get(c))
                for c in categories
            ]
            raw["def_pct_plusminus_zone"] = [
                _mix(by_category.get(c, {}).get("pct_plusminus")) for c in categories
            ]
            raw["def_freq_zone"] = [
                by_category.get(c, {}).get("freq") for c in categories
            ]

        # ── Shared transforms: identical to the training path ─────────────
        raw = apply_hierarchy(raw, self.zone_priors)
        features = derive_features(raw, league_zone_rates=self.league_zone_rates)

        for col in self.feature_cols:
            if col not in features.columns:
                features[col] = np.nan

        make_prob = self._predict(as_model_matrix(features, self.feature_cols))

        # ── Assemble output ──────────────────────────────────────────────
        out = grid[["zone", "loc_x", "loc_y", "shot_distance"]].copy()
        out["make_probability"] = make_prob

        if self.uses_mechanics:
            # Collapse the (location x mechanic) expansion back to one row per
            # location: the reported probability is the expectation over the
            # player's mechanic mix, and `best_mechanic` names the shot type
            # that scored highest there — which is the actually actionable half
            # of the answer.
            out["_mechanic"] = grid["_mechanic"].values
            out["_mech_weight"] = grid["_mech_weight"].values
            out = marginalize(
                out, key_cols=("loc_x", "loc_y", "zone", "shot_distance")
            )
        out["points"] = out["zone"].map(ZONE_POINTS).astype(float)
        out["expected_points"] = out["make_probability"] * out["points"]

        # Credible interval, driven by how much evidence backs this player in
        # this zone.
        #
        # The interval is derived as a WIDTH around the Beta posterior mean and
        # then applied around the model's prediction, rather than being used
        # directly. Those are two different estimators: the posterior mean is
        # career attempts regressed to the league zone prior, while the model's
        # prediction also reflects the season-over-career hierarchy, the
        # defender, and the game state. Using the raw posterior bounds as the
        # band produced intervals that did not contain the number they were
        # drawn under — a left-corner projection of 0.86 expected points with a
        # lower bound of 0.97.
        #
        # What this band represents is therefore specific and worth stating in
        # the UI: uncertainty about the player's SHOOTING RATE from that zone
        # given how many attempts back it. It is not a confidence interval on
        # the model, and it does not cover defender or context uncertainty.
        lows, highs, attempts = [], [], []
        for zone in out["zone"]:
            suffix = ZONE_SUFFIX[zone]
            makes = float(player.get(f"pit_car_mk_{suffix}", 0.0) or 0.0)
            att = float(player.get(f"pit_car_att_{suffix}", 0.0) or 0.0)
            prior = self.zone_priors.get(zone)
            if prior is None:
                lows.append(np.nan)
                highs.append(np.nan)
                attempts.append(att)
                continue
            lo, hi = posterior_interval(makes, att, prior, level=interval_level)
            posterior_mean = (makes + prior.alpha) / (att + prior.strength)
            lows.append(float(posterior_mean - lo))
            highs.append(float(hi - posterior_mean))
            attempts.append(att)

        out["attempts_behind"] = attempts
        out["ep_low"] = (
            (out["make_probability"].values - np.array(lows)) * out["points"].values
        ).clip(0.0)
        out["ep_high"] = (
            (out["make_probability"].values + np.array(highs)) * out["points"].values
        )

        out["attainability"] = self._attainability(
            player, creation, out["zone"],
            cast=self._cast_row(player_id, season, as_of_date=as_of_date),
            loc_x=out["loc_x"], loc_y=out["loc_y"],
            prior_diet=self._diet_row(player_id, season, as_of_date=as_of_date),
        )

        # Ranking score. Attainability enters as a square root rather than
        # linearly: weighting it fully would collapse every recommendation onto
        # whatever the player already does most, which is not advice. The
        # square root keeps a genuinely better shot competitive when it is
        # somewhat harder to generate, while still preventing the engine from
        # telling a rim-running centre to shoot step-back threes.
        att = out["attainability"].fillna(out["attainability"].mean())
        if att.notna().any() and att.max() > 0:
            out["score"] = out["expected_points"] * np.sqrt(att / att.max())
        else:
            out["score"] = out["expected_points"]

        # Against the player's own average, which is what makes the number
        # actionable. Raw EP mostly restates that a three is worth more than a
        # two; the useful question is where THIS player beats his own baseline.
        own_average = float(
            np.average(out["expected_points"], weights=att.fillna(0) + 1e-9)
        )
        out["ep_vs_own_average"] = out["expected_points"] - own_average

        # ── Backwards-compatible fields ──────────────────────────────────
        # The React client (shot-vision-engine-main) reads these names. They
        # are kept as aliases over the new quantities rather than dropped, so
        # the redesigned engine is a drop-in for the existing UI; the client
        # can adopt `attainability`, `ep_low`/`ep_high` and `ep_vs_own_average`
        # whenever it is ready, without a coordinated deploy.
        out["shot_type"] = np.where(
            out["points"] == 3, "3PT Field Goal", "2PT Field Goal"
        )
        out["shot_quality_score"] = out["score"]
        out["difficulty_score"] = 1.0 - out["make_probability"]

        out["player_name"] = player.get("name")
        out["model"] = self.model_name

        return out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    def recommend_heatmap(self, **kwargs) -> dict:
        """
        The full scored grid, for the court heat map.

        Thin wrapper over `recommend` with the top-N cap lifted, so the map and
        the ranked list can never disagree — they are literally the same
        scored frame.
        """
        full = self.recommend(**{**kwargs, "top_n": len(SHOT_GRID)})
        records = full.to_dict("records")

        # Key names are the client's contract (HeatmapResponse in
        # shot-vision-data.ts). Renaming `heatmap` to `grid` during the rewrite
        # silently emptied the court — the page read `res.heatmap`, got
        # undefined, and rendered every zone with no field at all while the
        # court lines drew fine, so it looked like a styling problem rather
        # than a missing payload.
        return {
            "heatmap": records,
            "recommendations": full.nlargest(10, "score").to_dict("records"),
            "zone_summary": self._summarize(full).to_dict("records"),
            "shot_volume": {},
            "grid_size": len(records),
        }

    @staticmethod
    def _summarize(scored: pd.DataFrame) -> pd.DataFrame:
        """Per-zone aggregate of an already-scored grid."""
        summary = scored.groupby("zone").agg(
            make_probability=("make_probability", "mean"),
            expected_points=("expected_points", "mean"),
            ep_low=("ep_low", "mean"),
            ep_high=("ep_high", "mean"),
            attainability=("attainability", "mean"),
            attempts_behind=("attempts_behind", "max"),
            score=("score", "mean"),
            # Aliases the existing React client reads. See the note in
            # `recommend` — kept so the redesign drops into the current UI.
            best_make_prob=("make_probability", "max"),
            avg_make_prob=("make_probability", "mean"),
            best_ep=("expected_points", "max"),
            avg_ep=("expected_points", "mean"),
            best_quality=("score", "max"),
        ).reset_index()
        # Carry the best mechanic through: for each zone, the shot type at that
        # zone's highest-scoring location. Aggregating a categorical needs an
        # explicit choice, and "what to do at the best spot" is the one that
        # matches how a reader will use it.
        if "best_mechanic" in scored.columns:
            best_rows = scored.loc[scored.groupby("zone")["score"].idxmax()]
            summary = summary.merge(
                best_rows[["zone", "best_mechanic", "best_mechanic_prob"]],
                on="zone", how="left",
            )

        summary["shot_type"] = np.where(
            summary["zone"].map(ZONE_POINTS) == 3,
            "3PT Field Goal", "2PT Field Goal",
        )
        return summary.sort_values("score", ascending=False).reset_index(drop=True)

    def zone_summary(self, **kwargs) -> pd.DataFrame:
        """Per-zone aggregate of the full grid, for the court heat map."""
        full = self.recommend(**{**kwargs, "top_n": len(SHOT_GRID)})
        return self._summarize(full)
