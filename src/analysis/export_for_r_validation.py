"""
One-time export for r/validate_defender_signal.R.

Why an export rather than R reading the pipeline directly
-----------------------------------------------------------
The defender features being validated (`def_fg_pct_zone`,
`def_pct_plusminus_zone`, `def_matchup_share`) are point-in-time aggregates
produced by `src.features.build.build_matrix` — season-long joins across
shots, matchups, and team data. Re-deriving them in R would duplicate real
pipeline logic in a second language, which is exactly the kind of scope
creep this validation is meant to avoid. Instead this script calls the
existing, unmodified `build_matrix` once and writes a plain CSV of the
columns R needs. R's job is the independent statistical test on that export,
not feature engineering.

Usage:
    python -m src.analysis.export_for_r_validation
    python -m src.analysis.export_for_r_validation --season 2024-25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.features.build import build_matrix

EXPORT_COLS = [
    "shot_made", "zone", "shot_distance",
    "def_fg_pct_zone", "def_pct_plusminus_zone", "def_matchup_share",
    "season",
]

OUTPUT_PATH = config.DATA_DIR / "exports" / "shots_for_r_validation.csv"


def export(season: str | None = None):
    seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    matrix, _feature_cols, _artifacts = build_matrix(
        seasons, prior_through_season=seasons[-2], verbose=True,
    )

    if season:
        matrix = matrix[matrix["season"] == season]
        if matrix.empty:
            raise ValueError(f"No rows for season={season!r}")

    out = matrix[EXPORT_COLS].copy()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ wrote {len(out):,} rows to {OUTPUT_PATH}")
    print(f"  defender feature coverage: "
          f"{out['def_fg_pct_zone'].notna().mean():.1%} of rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=None,
                        help="Restrict the export to one season (default: all)")
    args = parser.parse_args()
    export(args.season)
