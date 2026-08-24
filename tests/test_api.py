"""
Tests for src/inference/api.py using FastAPI's TestClient.

The model artifacts under models/ are small (a few MB), so the app's startup
event loads the real ShotRecommender rather than a mock — this gives genuine
end-to-end coverage of request validation -> DB lookup -> inference. The DB
engine is monkeypatched (in both api.py and recommender.py, since each imported
get_engine into its own module namespace via `from ... import`) to point at the
seeded in-memory DB so no test touches data/nba_shots.db.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.inference.api as api_mod
import src.inference.recommender as rec_mod

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_NAME = "shot-quality-v9"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / f"metadata_{MODEL_NAME}.json").exists(),
    reason=f"{MODEL_NAME} artifacts not present in models/",
)


@pytest.fixture()
def client(seeded_db, monkeypatch):
    monkeypatch.setattr(api_mod, "get_engine", lambda: seeded_db)
    monkeypatch.setattr(rec_mod, "get_engine", lambda: seeded_db)
    with TestClient(api_mod.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_name"] == MODEL_NAME
    assert body["features"] > 0


def test_health_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(api_mod, "recommender", None)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert body["model_name"] is None
    assert body["features"] == 0


def test_recommend_success(client):
    resp = client.post("/recommend", json={
        "player_id": "P_TALL", "season": "2023-24", "top_n": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["player_name"] == "Tall Center"
    assert len(body["recommendations"]) <= 5


def test_recommend_unknown_player_returns_404(client):
    resp = client.post("/recommend", json={
        "player_id": "NOT_A_REAL_ID", "season": "2023-24",
    })
    assert resp.status_code == 404


def test_recommend_validation_error_on_bad_quarter(client):
    # quarter is constrained to 1..8 by RecommendRequest
    resp = client.post("/recommend", json={
        "player_id": "P_TALL", "season": "2023-24", "quarter": 99,
    })
    assert resp.status_code == 422


def test_recommend_validation_error_missing_required_field(client):
    # player_id is required
    resp = client.post("/recommend", json={"season": "2023-24"})
    assert resp.status_code == 422


def test_recommend_returns_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(api_mod, "recommender", None)
    resp = client.post("/recommend", json={
        "player_id": "P_TALL", "season": "2023-24",
    })
    assert resp.status_code == 503


def test_recommend_summary_success(client):
    resp = client.post("/recommend/summary", json={
        "player_id": "P_TALL", "season": "2023-24",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["player_name"] == "Tall Center"
    assert len(body["zone_summary"]) > 0


def test_get_player_success(client):
    resp = client.get("/player/P_SHORT", params={"season": "2023-24"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Short Guard"


def test_get_player_not_found(client):
    resp = client.get("/player/NOT_A_REAL_ID", params={"season": "2023-24"})
    assert resp.status_code == 404


def test_players_search_requires_min_length_query(client):
    resp = client.get("/players/search", params={"q": "a", "season": "2023-24"})
    assert resp.status_code == 422


def test_players_search_finds_seeded_player(client):
    resp = client.get("/players/search", params={"q": "Tall", "season": "2023-24"})
    assert resp.status_code == 200
    body = resp.json()
    assert any(p["name"] == "Tall Center" for p in body)


def test_matchup_uses_zone_level_defender_fg_pct(client):
    """
    /matchup should use zone-level (not overall) defender FG% for the
    Restricted Area exploit analysis — the same fix ZONE_TO_DEF_CATEGORY
    documents in recommender.py.
    """
    resp = client.get("/matchup/P_TALL/P_DEF", params={"season": "2023-24"})
    assert resp.status_code == 200
    body = resp.json()

    rim_zone = next(z for z in body["exploit_zones"] if z["zone"] == "Restricted Area")
    assert rim_zone["defender_fg_pct_allowed"] == pytest.approx(0.357)
    assert rim_zone["defender_fg_pct_allowed"] != body["defender_quality"]["fg_pct_allowed"]


def test_matchup_unknown_defender_returns_404(client):
    resp = client.get("/matchup/P_TALL/NOT_A_REAL_ID", params={"season": "2023-24"})
    assert resp.status_code == 404
