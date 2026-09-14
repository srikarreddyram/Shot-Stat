"""
Shot Quality API — FastAPI serving layer for the recommendation engine.

This package used to be a single 1,261-line api.py. It is now this
__init__.py (app setup, shared state, lifespan, and the one /health
endpoint that reports that state directly) plus one router module per
domain — recommend.py, explanations.py, players.py, teams.py, matchup.py,
live.py, shot_archetypes.py, model_features.py, stat_engine_routes.py,
media.py, player_archetypes.py — each a few hundred lines at most, findable
by name instead of by scrolling.

THE SHARED-STATE PATTERN, AND WHY: `recommender`, `db_engine` and
`latest_season` start as None below and are only ever filled in once, at
startup, inside `lifespan()`. Every router module needs to read them at
request time. The tempting way — `from src.inference.api import
recommender` inside a router file — is wrong: that copies the `None` that
was live at IMPORT time into the router module's own namespace, and never
sees the real value `lifespan()` assigns later, because Python name
imports bind a value, not a live reference to the origin module's
attribute slot. Every router module instead does `from src.inference
import api as _state` and reads `_state.recommender` at CALL time —
attribute access on the module object, which always reflects whatever
`lifespan()` most recently assigned. `tests/test_api.py` relies on this
too (it pokes `api_mod.recommender.category_priors` directly after
constructing a TestClient), which is also why this file keeps `app`,
`lifespan` and the three state variables here rather than moving them to
some inner `state.py` the tests would need to learn about instead.

Usage:
    uvicorn src.inference.api:app --reload --port 8000
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import requests  # noqa: F401  (re-exported: tests monkeypatch api_mod.requests.get)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Load-bearing for `uvicorn src.inference.api:app` invoked directly (not via
# `python -m uvicorn`, which would add the repo root to sys.path itself) —
# see scripts/dev.sh. One `.parent` deeper than when this was a flat
# api.py, since this file now lives at src/inference/api/__init__.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.inference.recommender import ShotRecommender
from src.db.database import get_engine

# ── Shared app state ─────────────────────────────────────────────────────────
# See the shared-state pattern note in this file's docstring before reading
# or writing these from a router module.
recommender: ShotRecommender = None
db_engine = None
latest_season: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, db_engine, latest_season
    db_engine = get_engine()
    # Model names are now descriptive rather than a v1/v2/v3 counter, and each
    # one has a runs/<stamp>__<name>/run.json recording exactly what it scored.
    # The list is ordered newest-first; the first that loads wins.
    for model_name in ["shot-quality-v20", "shot-quality-v19", "shot-quality-v13", "shot-quality-v12", "shot-quality-v11", "shot-quality-v10", "shot-quality-v9", "shot-quality-v8", "shot-quality-v5", "shot-quality-v4", "shot-quality", "v3", "v2", "v1"]:
        try:
            recommender = ShotRecommender(model_name=model_name)
            break
        except Exception as e:
            print(f"⚠ Could not load model {model_name}: {e}")
    if recommender is None:
        print("  API will start but /recommend endpoints won't work until a model is trained.")

    # The most recent season with ingested player data. Clients (the UI) should
    # use this instead of hardcoding a season string, which silently goes stale
    # every year and hides that season's rookies/trades from search results.
    with db_engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(season) FROM players")).fetchone()
        latest_season = row[0] if row else None
    print(f"  ✓ Latest ingested season: {latest_season}")

    yield


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NBA Shot Quality Engine",
    description="Predicts make probability and recommends optimal shot locations based on player + defender + game state.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": recommender is not None,
        "model_name": recommender.model_name if recommender else None,
        # Alias for the existing React client, which reads `model_version`.
        "model_version": recommender.model_name if recommender else None,
        "features": len(recommender.feature_cols) if recommender else 0,
        "latest_season": latest_season,
        # Whether a calibration mapping was adopted at training time. It is
        # deliberately not always on: the trainer adopts it only when it
        # improves held-out log-loss, and on the current data it does not.
        "calibrated": recommender.calibration_adopted if recommender else None,
        "attainability_model": (
            recommender.attainability is not None if recommender else None
        ),
    }


# ── Routers ──────────────────────────────────────────────────────────────────
# Imported down here, after `app`/`recommender`/`db_engine`/`latest_season`
# exist: each router module does `from src.inference import api as _state`
# at its own import time, which — because this __init__.py is mid-execution
# and already registered in sys.modules — resolves to this very module
# object rather than re-entering it. That's safe specifically because no
# router module reads `_state.recommender` etc. at IMPORT time, only from
# inside its request-handler functions, by which point this file has long
# finished running.
from .recommend import router as _recommend_router
from .explanations import router as _explanations_router
from .players import router as _players_router
from .teams import router as _teams_router
from .matchup import router as _matchup_router
from .live import router as _live_router
from .shot_archetypes import router as _shot_archetypes_router
from .model_features import router as _model_features_router
from .stat_engine_routes import router as _stat_engine_router
from .media import router as _media_router
from .player_archetypes import router as _player_archetypes_router

for _router in (
    _recommend_router,
    _explanations_router,
    _players_router,
    _teams_router,
    _matchup_router,
    _live_router,
    _shot_archetypes_router,
    _model_features_router,
    _stat_engine_router,
    _media_router,
    _player_archetypes_router,
):
    app.include_router(_router)
