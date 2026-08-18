"""
Recommendation Engine — Scores all viable court locations for a player+defender matchup.

Given an attacker, defender, and game state, this module:
1. Generates a grid of candidate shot locations across all 6 court zones
2. Constructs the full feature vector for each candidate
3. Runs the XGBoost model over all candidates in a single batch
4. Returns the top-N shot recommendations ranked by Expected Points (EP)

Usage:
    from src.inference.recommender import ShotRecommender
    rec = ShotRecommender(model_version="v1")
    recs = rec.recommend(player_id="2544", season="2023-24", ...)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine


# ── Court grid: generate dense shot locations per zone ────────────────────────
# Coordinates use NBA StaticShotChart system: (0,0) = basket, y = towards halfcourt
# Each zone gets ~25-35 points for smooth heatmap rendering.

def _generate_court_grid():
    """Generate ~180 candidate shot locations evenly distributed across all 6 zones."""
    grid = []

    def _add(loc_x, loc_y, zone, shot_type):
        dist = np.sqrt(loc_x**2 + loc_y**2) / 10.0  # convert to feet
        grid.append({
            "loc_x": float(loc_x), "loc_y": float(loc_y),
            "shot_distance": round(dist, 1), "zone": zone, "shot_type": shot_type,
        })

    # ── Restricted Area (radius ~40 units from basket, ~25 points) ──
    for x in np.linspace(-35, 35, 8):
        for y in np.linspace(2, 35, 4):
            if np.sqrt(x**2 + y**2) <= 42:
                _add(x, y, "Restricted Area", "2PT Field Goal")

    # ── In The Paint (Non-RA): inside paint but outside RA (~30 points) ──
    for x in np.linspace(-75, 75, 7):
        for y in np.linspace(40, 90, 5):
            d = np.sqrt(x**2 + y**2)
            if 42 < d <= 100 and abs(x) <= 80:
                _add(x, y, "In The Paint (Non-RA)", "2PT Field Goal")

    # ── Mid-Range: inside 3pt line but outside paint (~35 points) ──
    for angle in np.linspace(10, 170, 12):
        for r in [100, 130, 160, 190]:
            x = r * np.cos(np.radians(angle))
            y = r * np.sin(np.radians(angle))
            d = np.sqrt(x**2 + y**2) / 10.0
            # Inside 3pt line (varies: ~22ft at corners, ~23.75ft at top)
            # but outside paint
            if d < 22.0 and d > 9.0 and y > 0:
                _add(x, y, "Mid-Range", "2PT Field Goal")

    # ── Left Corner 3 (y < 93, x < -220, ~10 points) ──
    for x in np.linspace(-235, -220, 4):
        for y in np.linspace(5, 85, 5):
            d = np.sqrt(x**2 + y**2) / 10.0
            if d >= 22.0 and y <= 93:
                _add(x, y, "Left Corner 3", "3PT Field Goal")

    # ── Right Corner 3 (y < 93, x > 220, ~10 points) ──
    for x in np.linspace(220, 235, 4):
        for y in np.linspace(5, 85, 5):
            d = np.sqrt(x**2 + y**2) / 10.0
            if d >= 22.0 and y <= 93:
                _add(x, y, "Right Corner 3", "3PT Field Goal")

    # ── Above the Break 3 (~35 points) ──
    for angle in np.linspace(15, 165, 14):
        for r in [237, 250, 270]:
            x = r * np.cos(np.radians(angle))
            y = r * np.sin(np.radians(angle))
            d = np.sqrt(x**2 + y**2) / 10.0
            if d >= 23.0 and y > 93:
                _add(x, y, "Above the Break 3", "3PT Field Goal")

    return grid


SHOT_GRID = _generate_court_grid()
# Also keep the sparse grid for quick recommendations
SHOT_GRID_SPARSE = [
    {"loc_x":   0.0, "loc_y":   5.0, "shot_distance": 1.0,  "zone": "Restricted Area",        "shot_type": "2PT Field Goal"},
    {"loc_x":  30.0, "loc_y":   8.0, "shot_distance": 3.0,  "zone": "Restricted Area",        "shot_type": "2PT Field Goal"},
    {"loc_x": -30.0, "loc_y":   8.0, "shot_distance": 3.0,  "zone": "Restricted Area",        "shot_type": "2PT Field Goal"},
    {"loc_x":   0.0, "loc_y":  60.0, "shot_distance": 6.0,  "zone": "In The Paint (Non-RA)",  "shot_type": "2PT Field Goal"},
    {"loc_x":  50.0, "loc_y":  50.0, "shot_distance": 7.0,  "zone": "In The Paint (Non-RA)",  "shot_type": "2PT Field Goal"},
    {"loc_x": -50.0, "loc_y":  50.0, "shot_distance": 7.0,  "zone": "In The Paint (Non-RA)",  "shot_type": "2PT Field Goal"},
    {"loc_x":   0.0, "loc_y": 140.0, "shot_distance": 14.0, "zone": "Mid-Range",              "shot_type": "2PT Field Goal"},
    {"loc_x": 100.0, "loc_y": 100.0, "shot_distance": 14.0, "zone": "Mid-Range",              "shot_type": "2PT Field Goal"},
    {"loc_x":-100.0, "loc_y": 100.0, "shot_distance": 14.0, "zone": "Mid-Range",              "shot_type": "2PT Field Goal"},
    {"loc_x":-220.0, "loc_y":  10.0, "shot_distance": 22.0, "zone": "Left Corner 3",          "shot_type": "3PT Field Goal"},
    {"loc_x":-230.0, "loc_y":  20.0, "shot_distance": 23.0, "zone": "Left Corner 3",          "shot_type": "3PT Field Goal"},
    {"loc_x": 220.0, "loc_y":  10.0, "shot_distance": 22.0, "zone": "Right Corner 3",         "shot_type": "3PT Field Goal"},
    {"loc_x": 230.0, "loc_y":  20.0, "shot_distance": 23.0, "zone": "Right Corner 3",         "shot_type": "3PT Field Goal"},
    {"loc_x":   0.0, "loc_y": 240.0, "shot_distance": 24.0, "zone": "Above the Break 3",      "shot_type": "3PT Field Goal"},
    {"loc_x":  80.0, "loc_y": 230.0, "shot_distance": 24.5, "zone": "Above the Break 3",      "shot_type": "3PT Field Goal"},
    {"loc_x": -80.0, "loc_y": 230.0, "shot_distance": 24.5, "zone": "Above the Break 3",      "shot_type": "3PT Field Goal"},
    {"loc_x": 160.0, "loc_y": 190.0, "shot_distance": 25.0, "zone": "Above the Break 3",      "shot_type": "3PT Field Goal"},
    {"loc_x":-160.0, "loc_y": 190.0, "shot_distance": 25.0, "zone": "Above the Break 3",      "shot_type": "3PT Field Goal"},
]


class ShotRecommender:
    """
    Scores all viable court locations and returns ranked shot recommendations.

    Loads a trained XGBoost model and metadata (feature list, train medians).
    Constructs feature vectors for every grid point, runs batch inference,
    and returns recommendations sorted by Expected Points.
    """

    def __init__(self, model_version: str = "v1", model_dir: str = "models"):
        model_path = Path(model_dir)

        # Load metadata (feature columns, train medians, zone averages)
        self.metadata = joblib.load(model_path / f"metadata_{model_version}.joblib")
        self.feature_cols = self.metadata["feature_cols"]
        self.train_medians = self.metadata["train_medians"]
        self.zone_avg = self.metadata["zone_avg"]
        self.is_hierarchical = self.metadata.get("hierarchical", False)

        if self.is_hierarchical:
            # Load Interior Model
            self.model_interior = xgb.XGBClassifier()
            self.model_interior.load_model(str(model_path / f"xgb_interior_{model_version}.json"))
            
            # Load Perimeter Model
            self.model_perimeter = xgb.XGBClassifier()
            self.model_perimeter.load_model(str(model_path / f"xgb_perimeter_{model_version}.json"))
            
            # Load Calibrators
            cal_int_path = model_path / f"calibrator_interior_{model_version}.joblib"
            cal_per_path = model_path / f"calibrator_perimeter_{model_version}.joblib"
            self.calibrator_int = joblib.load(cal_int_path) if cal_int_path.exists() else None
            self.calibrator_per = joblib.load(cal_per_path) if cal_per_path.exists() else None
            print(f"  ✓ Hierarchical Models loaded")
        else:
            # Load Monolithic Model
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path / f"xgb_{model_version}.json"))
            
            # Load Calibrator
            self.calibrator = None
            cal_path = model_path / f"calibrator_{model_version}.joblib"
            if cal_path.exists():
                self.calibrator = joblib.load(cal_path)
                print(f"  ✓ Monolithic Calibrator loaded")

        # DB engine for player lookups
        self.engine = get_engine()

        self.version = model_version
        is_cal = (self.calibrator_int is not None) if self.is_hierarchical else (self.calibrator is not None)
        print(f"✓ ShotRecommender loaded (model {model_version}, {len(self.feature_cols)} features, hierarchical={self.is_hierarchical}, calibrated={is_cal})")

    def _get_player_data(self, player_id: str, season: str) -> dict:
        """Fetch a player's attributes and zone stats from the DB."""
        with self.engine.connect() as conn:
            # Player attributes
            row = conn.execute(text("""
                SELECT name, height, weight, wingspan, position,
                       career_fg_pct, career_3p_pct, season_fg_pct
                FROM players
                WHERE player_id = :pid AND season = :season
            """), {"pid": player_id, "season": season}).fetchone()

            if row is None:
                raise ValueError(f"Player {player_id} not found in season {season}")

            player = dict(row._mapping)

            # Zone stats
            zone_rows = conn.execute(text("""
                SELECT zone, fg_pct
                FROM player_zone_stats
                WHERE player_id = :pid AND season = :season
            """), {"pid": player_id, "season": season}).fetchall()

            zone_stats = {r[0]: r[1] for r in zone_rows}

        return {**player, "zone_stats": zone_stats}

    def _get_defender_data(self, defender_id: str, season: str) -> dict:
        """Fetch a defender's attributes and defensive stats from the DB."""
        with self.engine.connect() as conn:
            # Defender physical attributes
            row = conn.execute(text("""
                SELECT name, height, weight, wingspan, position
                FROM players
                WHERE player_id = :pid AND season = :season
            """), {"pid": defender_id, "season": season}).fetchone()

            if row is None:
                return None

            defender = dict(row._mapping)

            # Defender stats by category (now includes zone-level d_fg_pct)
            def_rows = conn.execute(text("""
                SELECT defense_category, d_fg_pct, pct_plusminus, freq
                FROM defender_stats
                WHERE player_id = :pid AND season = :season
            """), {"pid": defender_id, "season": season}).fetchall()

            defender["def_stats"] = {
                r[0]: {"d_fg_pct": r[1], "pct_plusminus": r[2], "freq": r[3]}
                for r in def_rows
            }

        return defender

    def recommend(
        self,
        player_id: str,
        season: str,
        quarter: int = 1,
        time_remaining: float = 600.0,
        score_diff: int = 0,
        home_away: int = 1,
        playoff_flag: int = 0,
        defender_id: str = None,
        top_n: int = 10,
        rest_days: int = 1,
        is_back_to_back: int = 0,
        opp_def_rating: float = 112.0,
    ) -> pd.DataFrame:
        """
        Score all court locations and return top-N recommendations.

        Returns DataFrame with columns:
            zone, loc_x, loc_y, make_probability, expected_points,
            shot_quality_score, difficulty_score
        """
        # Fetch player data
        player = self._get_player_data(player_id, season)

        # Fetch defender data (optional)
        defender = None
        if defender_id:
            defender = self._get_defender_data(defender_id, season)

        # Zone-to-defense-category mapping
        zone_to_def_cat = {
            "Restricted Area": "Less Than 6Ft",
            "In The Paint (Non-RA)": "Less Than 10Ft",
            "Mid-Range": "Greater Than 15Ft",
            "Left Corner 3": "3 Pointers",
            "Right Corner 3": "3 Pointers",
            "Above the Break 3": "3 Pointers",
        }

        # Zone-to-fg_pct column mapping (matches feature_engineering.py)
        zone_to_col = {
            "Restricted Area": "fg_pct_restricted_area",
            "In The Paint (Non-RA)": "fg_pct_paint",
            "Mid-Range": "fg_pct_midrange",
            "Left Corner 3": "fg_pct_left_corner_3",
            "Right Corner 3": "fg_pct_right_corner_3",
            "Above the Break 3": "fg_pct_above_break_3",
        }

        # Clutch flag
        clutch_flag = int(quarter >= 4 and time_remaining <= 120 and abs(score_diff) <= 5)

        # Build feature rows for every grid point
        rows = []
        for pt in SHOT_GRID:
            loc_x = pt["loc_x"]
            loc_y = pt["loc_y"]
            zone = pt["zone"]

            # Shot-level zone efficiency for this player
            zone_eff = player["zone_stats"].get(zone)

            row = {
                # Spatial
                "loc_x": loc_x,
                "loc_y": loc_y,
                "shot_distance": pt["shot_distance"],
                "shot_angle": np.degrees(np.arctan2(loc_y, loc_x)) if loc_x != 0 else 90.0,
                "distance_from_center": np.sqrt(loc_x**2 + loc_y**2),
                "abs_loc_x": abs(loc_x),
                "is_three": 1 if pt["shot_type"] == "3PT Field Goal" else 0,

                # Game context
                "quarter": quarter,
                "time_remaining": time_remaining,
                "score_diff": score_diff,
                "home_away": home_away,
                "playoff_flag": playoff_flag,
                "clutch_flag": clutch_flag,

                # Team / schedule context (Phase D)
                "rest_days": rest_days,
                "is_back_to_back": is_back_to_back,
                "opp_def_rating": opp_def_rating,

                # Player ability
                "height": player.get("height"),
                "weight": player.get("weight"),
                "wingspan": player.get("wingspan"),
                "career_fg_pct": player.get("career_fg_pct"),
                "career_3p_pct": player.get("career_3p_pct"),
                "season_fg_pct": player.get("season_fg_pct"),
                "zone_efficiency": zone_eff,

                # Zone-level shooting (all 6 zones for this player)
                "fg_pct_restricted_area": player["zone_stats"].get("Restricted Area"),
                "fg_pct_paint": player["zone_stats"].get("In The Paint (Non-RA)"),
                "fg_pct_midrange": player["zone_stats"].get("Mid-Range"),
                "fg_pct_left_corner_3": player["zone_stats"].get("Left Corner 3"),
                "fg_pct_right_corner_3": player["zone_stats"].get("Right Corner 3"),
                "fg_pct_above_break_3": player["zone_stats"].get("Above the Break 3"),

                # Metadata (not features, but needed for output)
                "_zone": zone,
                "_shot_type": pt["shot_type"],
            }

            # Defender features (if defender provided and data exists)
            if defender:
                row["def_height"] = defender.get("height")
                row["def_weight"] = defender.get("weight")
                row["def_wingspan"] = defender.get("wingspan")

                h = player.get("height")
                dh = defender.get("height")
                row["height_diff"] = (h - dh) if (h and dh) else None
                w = player.get("weight")
                dw = defender.get("weight")
                row["weight_diff"] = (w - dw) if (w and dw) else None
                ws = player.get("wingspan")
                dws = defender.get("wingspan")
                row["wingspan_diff"] = (ws - dws) if (ws and dws) else None
                row["size_mismatch"] = int(abs(row["height_diff"]) >= 4) if row["height_diff"] is not None else 0

                # Defender quality stats — overall
                overall = defender.get("def_stats", {}).get("Overall", {})
                row["def_fg_pct_overall"] = overall.get("d_fg_pct")
                row["def_pct_plusminus"] = overall.get("pct_plusminus")

                # Defender zone-specific stats (now populated with correct column mapping)
                def_cat = zone_to_def_cat.get(zone)
                zone_def = defender.get("def_stats", {}).get(def_cat, {})
                row["def_freq_zone"] = zone_def.get("freq")
                row["def_fg_pct_zone"] = zone_def.get("d_fg_pct")
                row["def_pct_plusminus_zone"] = zone_def.get("pct_plusminus")

                # Composite: attacker zone FG% minus defender ZONE-level FG% allowed
                # Falls back to overall if zone-level is unavailable
                def_fg = row["def_fg_pct_zone"] or row["def_fg_pct_overall"]
                if zone_eff is not None and def_fg is not None:
                    row["matchup_advantage"] = zone_eff - def_fg
                else:
                    row["matchup_advantage"] = None

            rows.append(row)

        # Convert to DataFrame
        candidates_df = pd.DataFrame(rows)

        # Add one-hot encoded dummies (matching what the model was trained on)
        # We need zone dummies and position dummies
        zone_dummies = pd.get_dummies(candidates_df["_zone"], prefix="zone", drop_first=True)
        candidates_df = pd.concat([candidates_df, zone_dummies], axis=1)

        # Position dummies — attacker's position
        pos = player.get("position")
        if pos:
            # Create all possible position columns that may exist in the model
            for col in self.feature_cols:
                if col.startswith("pos_"):
                    candidates_df[col] = 1 if col == f"pos_{pos}" else 0

        # Select only the features the model expects
        X = candidates_df.reindex(columns=self.feature_cols)

        # Fill missing columns with NaN (XGBoost handles NaN)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_cols].astype(float)

        # Run inference
        if self.is_hierarchical:
            raw_probs = np.zeros(len(X))
            make_probs = np.zeros(len(X))
            
            # Identify interior vs perimeter rows
            interior_zones = ["Restricted Area", "In The Paint (Non-RA)"]
            mask_int = candidates_df["_zone"].isin(interior_zones).values
            mask_per = ~mask_int
            
            # Predict interior
            if mask_int.sum() > 0:
                X_int = X[mask_int]
                raw_int = self.model_interior.predict_proba(X_int)[:, 1]
                raw_probs[mask_int] = raw_int
                if self.calibrator_int:
                    make_probs[mask_int] = self.calibrator_int.predict(raw_int)
                else:
                    make_probs[mask_int] = raw_int
                    
            # Predict perimeter
            if mask_per.sum() > 0:
                X_per = X[mask_per]
                raw_per = self.model_perimeter.predict_proba(X_per)[:, 1]
                raw_probs[mask_per] = raw_per
                if self.calibrator_per:
                    make_probs[mask_per] = self.calibrator_per.predict(raw_per)
                else:
                    make_probs[mask_per] = raw_per
        else:
            raw_probs = self.model.predict_proba(X)[:, 1]
            if self.calibrator:
                make_probs = self.calibrator.predict(raw_probs)
            else:
                make_probs = raw_probs

        # Build results
        shot_value = np.where(candidates_df["is_three"] == 1, 3.0, 2.0)
        expected_points = make_probs * shot_value
        shot_quality = (make_probs * 100).round(1)

        results = pd.DataFrame({
            "zone": candidates_df["_zone"],
            "loc_x": candidates_df["loc_x"],
            "loc_y": candidates_df["loc_y"],
            "shot_type": candidates_df["_shot_type"],
            "shot_distance": candidates_df["shot_distance"].round(1),
            "make_probability": make_probs.round(4),
            "expected_points": expected_points.round(4),
            "shot_quality_score": shot_quality,
            "difficulty_score": (100 - shot_quality).round(1),
        })

        # Sort by expected points (descending)
        results = results.sort_values("expected_points", ascending=False).reset_index(drop=True)

        return results.head(top_n) if top_n else results

    def recommend_summary(
        self,
        player_id: str,
        season: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Zone-level summary: best EP per zone, sorted by recommendation strength.
        Collapses the grid points into one row per zone.
        """
        all_recs = self.recommend(player_id=player_id, season=season, top_n=None, **kwargs)

        summary = all_recs.groupby("zone").agg(
            best_make_prob=("make_probability", "max"),
            best_ep=("expected_points", "max"),
            avg_make_prob=("make_probability", "mean"),
            avg_ep=("expected_points", "mean"),
            best_quality=("shot_quality_score", "max"),
            shot_type=("shot_type", "first"),
        ).sort_values("best_ep", ascending=False).reset_index()

        return summary

    def _get_shot_volume(self, player_id: str, season: str) -> dict:
        """Get shot attempt counts per zone for a player-season."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT zone, COUNT(*) as attempts, 
                       SUM(shot_made) as makes,
                       ROUND(AVG(shot_made) * 100, 1) as fg_pct
                FROM shots
                WHERE player_id = :pid AND season = :season
                  AND zone IS NOT NULL AND zone != 'Backcourt'
                GROUP BY zone
            """), {"pid": player_id, "season": season}).fetchall()

        return {r[0]: {"attempts": r[1], "makes": r[2], "fg_pct": r[3]} for r in rows}

    def recommend_heatmap(
        self,
        player_id: str,
        season: str,
        quarter: int = 1,
        time_remaining: float = 600.0,
        score_diff: int = 0,
        home_away: int = 1,
        playoff_flag: int = 0,
        defender_id: str = None,
        rest_days: int = 1,
        is_back_to_back: int = 0,
        opp_def_rating: float = 112.0,
    ) -> dict:
        """
        Return the full dense grid scored for heatmap rendering.

        Returns a dict with:
            - heatmap: list of {loc_x, loc_y, zone, make_prob, ep, quality} for every grid point
            - zone_summary: per-zone aggregates
            - shot_volume: shot attempt counts per zone for this player
            - recommendations: top 10 by EP
        """
        # Get full predictions over the dense grid
        all_recs = self.recommend(
            player_id=player_id, season=season,
            quarter=quarter, time_remaining=time_remaining,
            score_diff=score_diff, home_away=home_away,
            playoff_flag=playoff_flag, defender_id=defender_id,
            top_n=None,
            rest_days=rest_days, is_back_to_back=is_back_to_back,
            opp_def_rating=opp_def_rating,
        )

        # Shot volume from DB
        volume = self._get_shot_volume(player_id, season)

        # Add volume + confidence to each recommendation
        all_recs["shot_volume"] = all_recs["zone"].map(
            lambda z: volume.get(z, {}).get("attempts", 0)
        )
        # Confidence: penalize zones with low volume
        all_recs["confidence"] = (
            all_recs["make_probability"] * np.log1p(all_recs["shot_volume"])
        ).round(4)

        # Zone summary with volume
        zone_summary = all_recs.groupby("zone").agg(
            best_make_prob=("make_probability", "max"),
            avg_make_prob=("make_probability", "mean"),
            best_ep=("expected_points", "max"),
            avg_ep=("expected_points", "mean"),
            point_count=("loc_x", "count"),
            shot_type=("shot_type", "first"),
        ).reset_index()

        # Merge volume data
        zone_summary["attempts"] = zone_summary["zone"].map(
            lambda z: volume.get(z, {}).get("attempts", 0)
        )
        zone_summary["actual_fg_pct"] = zone_summary["zone"].map(
            lambda z: volume.get(z, {}).get("fg_pct", 0)
        )
        zone_summary = zone_summary.sort_values("best_ep", ascending=False)

        return {
            "heatmap": all_recs.to_dict("records"),
            "zone_summary": zone_summary.to_dict("records"),
            "shot_volume": volume,
            "recommendations": all_recs.nlargest(10, "expected_points").to_dict("records"),
            "grid_size": len(all_recs),
        }


if __name__ == "__main__":
    # Quick demo
    rec = ShotRecommender(model_version="v2")

    # LeBron James in 2023-24, Q3, 5 min left, tied game, home
    results = rec.recommend(
        player_id="2544",
        season="2023-24",
        quarter=3,
        time_remaining=300,
        score_diff=0,
        home_away=1,
    )

    print(f"\n{'='*80}")
    print(f"  SHOT RECOMMENDATIONS — LeBron James (2023-24)")
    print(f"  Q3 | 5:00 left | Score tied | Home")
    print(f"{'='*80}")
    print(results.to_string(index=False))

    # Test heatmap
    heatmap = rec.recommend_heatmap(
        player_id="2544",
        season="2023-24",
        quarter=3,
        time_remaining=300,
        score_diff=0,
        home_away=1,
    )
    print(f"\n  Heatmap grid: {heatmap['grid_size']} points")
    print(f"  Shot volume: {heatmap['shot_volume']}")

