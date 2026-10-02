"""
"Comparable shots" retrieval — a sanity check on the shot-quality
explanation, not another explanation of it.

`shot_quality.explain_shot_quality` decomposes a probability into feature
contributions: fully rigorous, but every claim it makes is internal to the
model ("defender FG% allowed here is at the 83rd percentile"). Nothing in
that module can say whether this exact combination of inputs is well
supported by real outcomes or a corner of feature space the model rarely
saw. This module answers that by finding the k real historical shots whose
FEATURE VECTORS are closest to the one being explained — in the exact same
120-column space `shot-quality-v20` was trained on, not a separately
invented notion of "similar" — and reporting how often those actually went
in. When that historical rate and the model's predicted probability agree,
that is corroboration. When they diverge, that is a real signal the
TreeSHAP breakdown alone cannot surface: either a genuinely unusual matchup,
or a thin region of training data the model is extrapolating through.

This is deliberately NOT an embedding-model/vector-database feature. The
model already builds the one feature vector that matters for every shot in
the database (`src.features.build.build_matrix`); reusing it means the
"similarity" here is provably the same similarity the model itself learned
from, with zero new data-ingestion or embedding infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class ComparableShotsIndex:
    feature_cols: list[str]
    scale: np.ndarray       # per-feature normalizer (train-set 1st-99th pctile width)
    meta: pd.DataFrame      # shot_id, player_id, player_name, game_date, zone, def_team, shot_made
    nn: NearestNeighbors    # fit on the scaled matrix; matrix itself isn't kept separately

    def query(self, feature_row: pd.DataFrame, k: int = 8, exclude_shot_id=None) -> dict:
        """
        `feature_row` must be the same post-`derive_features` single row
        `explain_shot_quality` was given — same discipline as that function,
        so "comparable" is computed in the space the shot was actually scored
        in, not re-derived independently.
        """
        x = feature_row[self.feature_cols].astype(float).to_numpy()[0]
        x = np.nan_to_num(x, nan=0.0)
        x_scaled = (x / self.scale)[None, :]

        n_query = min(k + (1 if exclude_shot_id is not None else 0), len(self.meta))
        if n_query == 0:
            return {"count": 0, "historical_make_rate": None, "examples": []}
        _, idx = self.nn.kneighbors(x_scaled, n_neighbors=n_query)
        rows = self.meta.iloc[idx[0]]
        if exclude_shot_id is not None:
            rows = rows[rows["shot_id"] != exclude_shot_id]
        rows = rows.head(k)
        if rows.empty:
            return {"count": 0, "historical_make_rate": None, "examples": []}

        made = rows["shot_made"].astype(float)
        examples = [
            {
                "player_name": r["player_name"],
                "game_date": None if pd.isna(r["game_date"]) else str(pd.Timestamp(r["game_date"]).date()),
                "zone": r["zone"],
                "opponent": r["def_team"] if pd.notna(r["def_team"]) else None,
                "made": bool(r["shot_made"]),
            }
            for _, r in rows.head(5).iterrows()
        ]
        return {
            "count": int(len(rows)),
            "historical_make_rate": round(float(made.mean()), 4),
            "examples": examples,
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ComparableShotsIndex":
        return joblib.load(path)


def _load_player_names() -> dict:
    from sqlalchemy import text
    from src.db.database import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.player_id, p.name FROM players p
            INNER JOIN (
                SELECT player_id, MAX(season) AS season FROM players GROUP BY player_id
            ) latest ON p.player_id = latest.player_id AND p.season = latest.season
        """)).fetchall()
    return {r[0]: r[1] for r in rows}


def build_comparable_shots_index(
    seasons: list[str],
    feature_cols: list[str],
    feature_bounds: dict,
    prior_through_season: str | None = None,
    max_rows: int | None = 250_000,
    verbose: bool = True,
) -> ComparableShotsIndex:
    """
    Build the index off `build_matrix` — the SAME function `train.py` calls —
    so a "comparable" shot is comparable in precisely the feature space the
    model being explained was trained on. Meant to be run once, offline (see
    scripts/build_comparable_index.py) and persisted; rebuilding per-request
    would be needlessly slow and would silently drift if the DB changes
    between requests.
    """
    from src.features.build import build_matrix
    from src.features.spec import as_model_matrix

    df, _, _ = build_matrix(seasons, prior_through_season=prior_through_season, verbose=verbose)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0).reset_index(drop=True)

    X = as_model_matrix(df, feature_cols)

    # feature_bounds is the model's own [1st, 99th] percentile per column
    # (models/metadata_<name>.json) — a scale robust to outliers, used only
    # to make euclidean distance meaningful across columns with unrelated
    # units (a percentage, a distance in feet, a one-hot flag). This never
    # touches what the model itself sees; it only shapes the neighbor search.
    scale = np.ones(len(feature_cols), dtype="float32")
    for i, col in enumerate(feature_cols):
        bounds = feature_bounds.get(col)
        if bounds:
            width = float(bounds[1]) - float(bounds[0])
            if width > 1e-6:
                scale[i] = width

    X_filled = X.fillna(X.median()).to_numpy(dtype="float32")
    X_scaled = X_filled / scale

    nn = NearestNeighbors(metric="euclidean")
    nn.fit(X_scaled)

    player_names = _load_player_names()
    meta = pd.DataFrame({
        "shot_id": df["shot_id"].to_numpy(),
        "player_id": df["player_id"].to_numpy(),
        "game_date": df["game_date"].to_numpy(),
        "zone": df["zone"].to_numpy(),
        "def_team": df["def_team"].to_numpy() if "def_team" in df.columns else pd.NA,
        "shot_made": df["shot_made"].to_numpy(),
    })
    meta["player_name"] = meta["player_id"].map(player_names).fillna(meta["player_id"].astype(str))

    return ComparableShotsIndex(feature_cols=feature_cols, scale=scale, meta=meta, nn=nn)
