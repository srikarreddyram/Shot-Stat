"""
End-to-end latency of the engine's main endpoints, measured over HTTP against
a running server (start one with ./scripts/dev.sh first).

    python scripts/benchmark_latency.py --api http://127.0.0.1:PORT

For each endpoint it makes one cold request per matchup (the first time that
player pair is asked for), then repeats the same requests to measure the warm
path, and prints median and 90th-percentile times. "Cold" here means a new
matchup on an already-running server, not a server restart.
"""
from __future__ import annotations

import argparse
import statistics
import time

import requests

# Shooter / defender pairs spread across positions and shot profiles.
MATCHUPS = [
    ("2544", "203497"),     # LeBron James vs Rudy Gobert
    ("201939", "1628369"),  # Stephen Curry vs Jayson Tatum
    ("1629029", "1627832"), # Luka Doncic vs Fred VanVleet
    ("203999", "1641705"),  # Nikola Jokic vs Victor Wembanyama
    ("1628983", "1626157"), # Shai Gilgeous-Alexander vs Karl-Anthony Towns
    ("203507", "1630169"),  # Giannis Antetokounmpo vs Tyrese Haliburton
]
GAME = {"quarter": 4, "time_remaining": 134, "score_diff": -2, "home_away": 1}


def _timed(fn) -> float:
    t = time.perf_counter()
    r = fn()
    r.raise_for_status()
    return (time.perf_counter() - t) * 1000


def _summary(ms: list[float]) -> str:
    q = statistics.quantiles(ms, n=10) if len(ms) >= 2 else ms
    return f"median {statistics.median(ms):6.0f} ms   p90 {q[-1]:6.0f} ms   (n={len(ms)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", required=True, help="Backend base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    api = args.api.rstrip("/")
    season = requests.get(f"{api}/health", timeout=30).json().get("latest_season")

    def heatmap(a, d):
        return lambda: requests.post(f"{api}/recommend/heatmap", timeout=120, json={
            "player_id": a, "defender_id": d, "season": season, "playoff_flag": 0, **GAME})

    def explain(a, d):
        return lambda: requests.get(f"{api}/explain/matchup/{a}", timeout=120, params={
            "zone": "Restricted Area", "loc_x": 0, "loc_y": 10, "season": season, "defender_id": d, **GAME})

    results: dict[str, dict[str, list[float]]] = {}
    for name, make in (("POST /recommend/heatmap  (score the whole court)", heatmap),
                       ("GET  /explain/matchup    (one shot, full explanation)", explain)):
        cold = [_timed(make(a, d)) for a, d in MATCHUPS]
        warm = [_timed(make(a, d)) for _ in range(args.repeats) for a, d in MATCHUPS]
        results[name] = {"cold": cold, "warm": warm}

    print(f"\nLatency over HTTP, {len(MATCHUPS)} matchups, season {season}\n")
    for name, r in results.items():
        print(name)
        print(f"  new matchup   {_summary(r['cold'])}")
        print(f"  repeat        {_summary(r['warm'])}\n")


if __name__ == "__main__":
    main()
