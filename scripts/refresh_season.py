"""
Refresh one season's data in a single command — every ingestion step the
engine and the Stat Engine read from, in dependency order.

    python scripts/refresh_season.py                   # the season in progress
    python scripts/refresh_season.py --season 2026-27
    python scripts/refresh_season.py --dry-run         # print the plan only
    python scripts/refresh_season.py --from pbp        # resume after a failure
    python scripts/refresh_season.py --only game_logs comparable

Every step is safe to re-run: the per-game steps (play-by-play, lineups,
matchups, score margin, contest) skip games they already have, and the rest
upsert. Shots and zone splits run with --refresh, because their usual resume
flags would otherwise mark a player done after his first week.

A first run of a new season makes a few thousand API calls and takes hours;
the per-game steps dominate. Later runs only fetch games played since.

Each step runs as its own `python -m` process, so a step behaves exactly as
it does when run by hand (see COMMANDS.md). If one fails the script stops and
prints the --from command that resumes there.

Restart the API afterwards: the comparable-shots index is loaded at startup.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

DEFAULT_MODEL = "shot-quality-v20"


@dataclass(frozen=True)
class Step:
    name: str
    what: str
    module: str
    args: tuple[str, ...] = ()


def plan(season: str, model: str = DEFAULT_MODEL) -> list[Step]:
    """The refresh, in dependency order. Each step lists what it needs."""
    s = season
    return [
        Step("schema", "create any missing tables and indexes",
             "src.db.database"),
        Step("games", "game list and results",
             "src.ingestion.game_ingestor", ("--seasons", s)),
        Step("player_stats", "players who have played, with season shooting",
             "src.ingestion.roster_ingestor", ("--seasons", s)),
        # After player_stats so the signed roster has the final word on team.
        Step("rosters", "signed rosters: trades, signings, rookies",
             "src.ingestion.current_roster_ingestor", ("--seasons", s)),
        Step("physicals", "height, weight, wingspan for new players",
             "src.ingestion.physical_ingestor_nba", ("--season", s)),
        Step("shots", "every shot (needs games + players)",
             "src.ingestion.shot_ingestor", ("--seasons", s, "--refresh")),
        Step("home_away", "home/away for shots missing it",
             "src.ingestion.home_away_backfill"),
        Step("score_diff", "score margin at each shot, per new game",
             "src.ingestion.score_diff_backfill", ("--season", s)),
        Step("pbp", "shot mechanics and putbacks, per new game",
             "src.ingestion.pbp_ingestor", ("--season", s)),
        Step("lineups", "the ten players on court for each shot, per new game",
             "src.ingestion.lineup_ingestor", ("--season", s)),
        Step("matchups", "defender matchup minutes, per new game",
             "src.ingestion.matchup_ingestor", ("--season", s)),
        Step("defenders", "primary defender on each shot (needs matchups)",
             "src.ingestion.defender_linker"),
        Step("zone_stats", "per-zone shooting splits",
             "src.ingestion.zone_stats_ingestor", ("--seasons", s, "--refresh")),
        Step("defender_stats", "defensive FG% allowed",
             "src.ingestion.defender_stats_ingestor", ("--seasons", s)),
        Step("tracking", "drives, touches, passing",
             "src.ingestion.tracking_ingestor", ("--season", s)),
        Step("shot_profile", "pull-up vs catch-and-shoot profile",
             "src.ingestion.shot_profile_ingestor", ("--season", s)),
        Step("defensive_activity", "steals, blocks, deflections, hustle",
             "src.ingestion.defensive_activity_ingestor", ("--season", s)),
        Step("playtypes", "Synergy play types",
             "src.ingestion.playtype_ingestor", ("--season", s)),
        Step("contest", "per-game shot contest bands, per new date",
             "src.ingestion.contest_ingestor", ("--seasons", s)),
        Step("team_stats", "team defensive ratings",
             "src.ingestion.team_stats_ingestor", ("--seasons", s)),
        Step("schedule", "rest days and back-to-backs (from games)",
             "src.ingestion.schedule_ingestor"),
        Step("career_stats", "career shooting (from shots)",
             "src.ingestion.career_stats_backfill"),
        Step("game_logs", "box scores, for season averages",
             "src.ingestion.game_log_ingestor", ("--season", s)),
        Step("comparable", "comparable-shots index, now including this season",
             "src.training.build_comparable_index", ("--model", model, "--add-season", s)),
    ]


def select(steps: list[Step], start: str | None, only: list[str] | None,
           skip: list[str]) -> list[Step]:
    names = [st.name for st in steps]
    for n in [start, *(only or []), *skip]:
        if n is not None and n not in names:
            raise SystemExit(f"Unknown step {n!r}. Steps: {', '.join(names)}")
    if start is not None:
        steps = steps[names.index(start):]
    if only:
        steps = [st for st in steps if st.name in only]
    return [st for st in steps if st.name not in skip]


def _command(step: Step) -> list[str]:
    return [sys.executable, "-m", step.module, *step.args]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default=config.current_season(),
                    help="Season to refresh. Default: the one in progress (%(default)s).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model the comparable index is built for.")
    ap.add_argument("--from", dest="start", metavar="STEP", help="Start at this step (resume).")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="Run only these steps.")
    ap.add_argument("--skip", nargs="+", default=[], metavar="STEP", help="Leave these steps out.")
    ap.add_argument("--dry-run", action="store_true", help="Print the steps and commands, run nothing.")
    args = ap.parse_args()

    steps = select(plan(args.season, args.model), args.start, args.only, args.skip)
    width = max(len(st.name) for st in steps)
    print(f"Refreshing {args.season}: {len(steps)} step(s)\n")
    for i, st in enumerate(steps, 1):
        print(f"  {i:2d}. {st.name:<{width}}  {st.what}")
        if args.dry_run:
            print(f"      {' '.join(_command(st)[1:])}")
    if args.dry_run:
        return

    timings = []
    for i, st in enumerate(steps, 1):
        print(f"\n{'━' * 70}\n  [{i}/{len(steps)}] {st.name} — {st.what}\n{'━' * 70}", flush=True)
        t = time.time()
        code = subprocess.call(_command(st), cwd=ROOT)
        timings.append((st.name, time.time() - t))
        if code != 0:
            print(f"\n✗ {st.name} failed (exit {code}). Fix it, then resume with:\n"
                  f"    python scripts/refresh_season.py --season {args.season} --from {st.name}")
            sys.exit(code)

    print(f"\n✓ {args.season} refreshed.\n")
    for name, secs in timings:
        print(f"  {name:<{width}}  {secs / 60:6.1f} min")
    if any(st.name == "comparable" for st in steps):
        print("\nRestart the API (or ./scripts/dev.sh) so it loads the new comparable index.")


if __name__ == "__main__":
    main()
