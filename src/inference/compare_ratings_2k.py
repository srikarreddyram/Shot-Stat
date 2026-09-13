"""
Compare our computed player ratings against NBA 2K's.

Why this exists
----------------
src/inference/player_ratings.py turns real box-score/tracking data into
off_rating/def_rating on a 40-99 scale. That is a defensible methodology,
but "defensible" and "correct" are not the same claim, and the only way to
tell them apart is to check the output against an independent opinion.
NBA 2K's ratings team is exactly that: thousands of hours of professional
scouting judgment, refreshed continuously, on the same 0-99-ish scale (see
src/ingestion/two_k_ratings_ingestor.py).

This is a comparison, not a validation — 2K's ratings are themselves
subjective and not ground truth. Where the two agree, that is a real signal
each is probably tracking something true. Where they disagree sharply, it is
worth reading the specific player's component breakdown before concluding
either side is wrong.

What "agreement" means here
----------------------------
Our ratings are PERCENTILE RANKS by construction (RATING_FLOOR..RATING_CEIL
mapped from a rank), so raw-value correlation against 2K's un-ranked
attribute averages is comparing two different kinds of number. Two numbers
are reported for each side:

  Spearman rank correlation  — robust to the scale difference, the primary
                                comparison.
  2K's averages re-percentile-ranked onto the SAME 40-99 band our own
  ratings use, purely so the biggest-mismatch table below can show two
  numbers on the same footing.

Usage:
    python -m src.inference.compare_ratings_2k
    python -m src.inference.compare_ratings_2k --season 2025-26
    python -m src.inference.compare_ratings_2k --top 20
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db.database import get_engine
from src.inference.player_ratings import (
    compute_ratings, resolve_rating_season, to_rating_band,
)


def build_comparison(engine, season: str) -> pd.DataFrame:
    anchor = resolve_rating_season(engine, season)
    ours = compute_ratings(engine, anchor)
    if ours.empty:
        return pd.DataFrame()

    two_k = pd.read_sql(
        "SELECT player_id, edition, overall, offense_avg, defense_avg FROM player_two_k_ratings",
        engine,
    )
    names = pd.read_sql(
        "SELECT DISTINCT player_id, name FROM players", engine
    ).drop_duplicates("player_id")

    df = ours.merge(two_k, on="player_id", how="inner").merge(names, on="player_id", how="left")
    if df.empty:
        return df

    df["two_k_off_rating"] = to_rating_band(df["offense_avg"])
    df["two_k_def_rating"] = to_rating_band(df["defense_avg"])
    df["off_diff"] = df["off_rating"] - df["two_k_off_rating"]
    df["def_diff"] = df["def_rating"] - df["two_k_def_rating"]
    return df


def report(season: str | None = None, top: int = 15):
    engine = get_engine()
    season = season or resolve_rating_season(engine, "2026-27")
    df = build_comparison(engine, season)

    if df.empty:
        print("\n  No overlap between computed ratings and player_two_k_ratings — "
              "has src/ingestion/two_k_ratings_ingestor.py been run yet?\n")
        return

    print(f"\n{'='*70}")
    print(f"  OUR RATINGS vs NBA 2K — {len(df)} players compared")
    print(f"{'='*70}")

    # nan_policy="omit": a player missing def_rating (didn't meet
    # MIN_MINUTES_FOR_RATING) or a 2K attribute subset shouldn't propagate a
    # single NaN into the whole correlation.
    off_rho, off_p = spearmanr(df["off_rating"], df["overall"], nan_policy="omit")
    def_rho, def_p = spearmanr(df["def_rating"], df["overall"], nan_policy="omit")
    off_vs_off_rho, _ = spearmanr(df["off_rating"], df["offense_avg"], nan_policy="omit")
    def_vs_def_rho, _ = spearmanr(df["def_rating"], df["defense_avg"], nan_policy="omit")

    print(f"\n  off_rating  vs 2K Overall        rho={off_rho:+.3f}  (p={off_p:.1e})")
    print(f"  def_rating  vs 2K Overall        rho={def_rho:+.3f}  (p={def_p:.1e})")
    print(f"  off_rating  vs 2K offense_avg     rho={off_vs_off_rho:+.3f}")
    print(f"  def_rating  vs 2K defense_avg     rho={def_vs_def_rho:+.3f}")
    print(f"\n  mean |off_rating - 2K off band|  {df['off_diff'].abs().mean():.1f}")
    print(f"  mean |def_rating - 2K def band|  {df['def_diff'].abs().mean():.1f}")

    print(f"\n  ── Biggest offense disagreements (ours higher) ──")
    for _, r in df.nlargest(top, "off_diff").iterrows():
        print(f"    {str(r['name'])[:22]:<22} ours={int(r.off_rating):>3}  2K={int(r.two_k_off_rating):>3}  "
              f"(2K overall {int(r.overall)})")
    print(f"\n  ── Biggest offense disagreements (2K higher) ──")
    for _, r in df.nsmallest(top, "off_diff").iterrows():
        print(f"    {str(r['name'])[:22]:<22} ours={int(r.off_rating):>3}  2K={int(r.two_k_off_rating):>3}  "
              f"(2K overall {int(r.overall)})")

    print(f"\n  ── Biggest defense disagreements (ours higher) ──")
    for _, r in df.nlargest(top, "def_diff").iterrows():
        print(f"    {str(r['name'])[:22]:<22} ours={int(r.def_rating):>3}  2K={int(r.two_k_def_rating):>3}  "
              f"(2K overall {int(r.overall)})")
    print(f"\n  ── Biggest defense disagreements (2K higher) ──")
    for _, r in df.nsmallest(top, "def_diff").iterrows():
        print(f"    {str(r['name'])[:22]:<22} ours={int(r.def_rating):>3}  2K={int(r.two_k_def_rating):>3}  "
              f"(2K overall {int(r.overall)})")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare our ratings against NBA 2K's.")
    parser.add_argument("--season", type=str, default=None)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    report(season=args.season, top=args.top)
