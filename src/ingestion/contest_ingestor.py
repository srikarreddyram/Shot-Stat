"""
Per-game contest level — how open a player's shots actually were, by night.

What this is, and what it is not
--------------------------------
Contest level is the biggest missing input to shot quality. Per-SHOT closest-
defender distance is not published anywhere public: `shotchartdetail` accepts a
`CloseDefDistRange` parameter and then SILENTLY IGNORES IT, returning the
unfiltered set for every band, so per-shot bands cannot be recovered from it.
That wall is real and this ingestor does not break it.

What it does get is one level up. `LeagueDashPlayerPtShot` honours DateFrom /
DateTo, and one call returns every player in the league for that range. Setting
DateFrom == DateTo therefore yields, for a single game date, each player's shot
distribution across the four defender-distance bands.

Verified by reconciliation rather than assumption: Doncic on 2024-01-26 returns
2 + 8 + 23 + 0 = 33 FGA, against exactly 33 rows for that game in `shots`.

How closely it reconciles, measured
------------------------------------
Across 543 player-games it agrees exactly 97.4% of the time. Every disagreement
runs the same direction — the tracking totals are LOWER than the shots table,
typically by one to three attempts — so a minority of attempts carry no
closest-defender assignment and are simply absent from these bands rather than
misfiled between them.

The consequence is that these counts are not a second source of truth for FGA
and must not be used as one. They are the denominator for a SHARE: what
fraction of a player's TRACKED attempts came against each band. Treating the
sum as his true attempt count would quietly understate volume for the players
the tracker drops most.

Why per-game matters when season-level already exists
-----------------------------------------------------
`player_shot_profile` stores the same four bands as a SEASON aggregate. A
season average describes a player, not a night, and — the operative part — it
cannot be made point-in-time. Accumulated over strictly prior games, these
per-game rows become a "how contested has he been getting lately" feature built
the same way as every other quantity in `point_in_time.py`.

Cost: 4 bands x ~205 dates per season, about 100 minutes for 2016-17 onward.
Resumable — dates already present in `player_game_contest` are skipped, so an
interrupted run is re-run rather than restarted.

Usage:
    python -m src.ingestion.contest_ingestor                  # recent season
    python -m src.ingestion.contest_ingestor --seasons 2023-24 2024-25
    python -m src.ingestion.contest_ingestor --full
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine
from src.db.models import PlayerGameContest

# The endpoint's own band labels, used verbatim as the stored key so a rename
# upstream surfaces as absent data rather than as a silently new category.
DEF_DIST_BANDS = [
    "0-2 Feet - Very Tight",
    "2-4 Feet - Tight",
    "4-6 Feet - Open",
    "6+ Feet - Wide Open",
]


def _game_dates(engine, seasons: list[str]) -> list[tuple[str, str, tuple]]:
    """
    (season, date, season_types) for every date with shots, oldest first.

    The season type is derived from the game-id prefix rather than queried for
    both: NBA ids encode it (002 regular, 004 playoffs, 005 play-in). Asking
    for both types on every date doubled the call count, and the wrong one
    returns an empty frame — pure latency for no data.
    """
    season_list = ", ".join(f"'{s}'" for s in seasons)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT s.season, g.date, substr(g.game_id, 1, 3) AS prefix
            FROM shots s JOIN games g ON g.game_id = s.game_id
            WHERE s.season IN ({season_list})
            ORDER BY g.date
        """)).fetchall()

    by_date: dict[tuple[str, str], set] = {}
    for season, date, prefix in rows:
        label = "Playoffs" if prefix in ("004", "005") else "Regular Season"
        by_date.setdefault((season, str(date)), set()).add(label)
    return [(s, d, tuple(sorted(t))) for (s, d), t in sorted(by_date.items(), key=lambda kv: kv[0][1])]


def _fetch_with_retry(endpoint_mod, season, season_type, stamp, band, date,
                      attempts: int = 4):
    """
    One (date, band) pull, retried through transient network failure.

    stats.nba.com read-timeouts and transient DNS failures are routine on a run
    this long. Without a retry a single blip drops a whole band, and on the
    first run a short DNS outage cost 53 consecutive dates before it recovered.
    Backoff is exponential so a sustained outage backs off rather than hammering.

    Returns the frame, or None if every attempt failed — the caller treats None
    as "this date is not complete" rather than as "this band had no shots".
    """
    delay = config.REQUEST_DELAY
    for i in range(attempts):
        try:
            df = endpoint_mod.LeagueDashPlayerPtShot(
                season=season,
                season_type_all_star=season_type,
                date_from_nullable=stamp,
                date_to_nullable=stamp,
                close_def_dist_range_nullable=band,
            ).get_data_frames()[0]
            time.sleep(config.REQUEST_DELAY)
            return df
        except Exception as e:
            if i == attempts - 1:
                print(f"\n  ⚠ {date} {band} {season_type} gave up after "
                      f"{attempts} tries: {type(e).__name__}")
                return None
            time.sleep(delay)
            delay *= 2
    return None


def _already_done(engine) -> set:
    """
    Dates stored with ALL four bands.

    Keying resume on mere presence let a partially-written date count as done.
    Requiring the full set means an interrupted or degraded date is picked up
    again on the next run instead of being silently frozen half-complete.
    """
    with engine.connect() as conn:
        return {
            str(r[0]) for r in conn.execute(text(f"""
                SELECT game_date FROM player_game_contest
                GROUP BY game_date
                HAVING COUNT(DISTINCT def_dist_range) = {len(DEF_DIST_BANDS)}
            """))
        }


def ingest_contest(seasons: list[str], resume: bool = True) -> int:
    from nba_api.stats.endpoints import leaguedashplayerptshot

    engine = get_engine()
    dates = _game_dates(engine, seasons)
    done = _already_done(engine) if resume else set()
    todo = [(s, d, t) for s, d, t in dates if d not in done]

    print(f"\n{'='*62}")
    print("  PER-GAME CONTEST INGEST")
    print(f"  {len(dates):,} game dates, {len(done):,} already stored, "
          f"{len(todo):,} to pull")
    calls = sum(len(DEF_DIST_BANDS) * len(t) for _, _, t in todo)
    print(f"  {calls:,} calls, ~{calls * 0.75 / 60:.0f} min at "
          f"{config.REQUEST_DELAY}s/call")
    print(f"{'='*62}")
    if not todo:
        print("  ✓ Nothing to do.")
        return 0

    written = 0
    skipped = 0
    for season, date, season_types in tqdm(todo, desc="  Dates"):
        stamp = pd.to_datetime(date).strftime("%m/%d/%Y")
        batch = []
        complete = True
        for band in DEF_DIST_BANDS:
            for season_type in season_types:
                df = _fetch_with_retry(
                    leaguedashplayerptshot, season, season_type, stamp, band, date
                )
                if df is None:
                    # A band that never arrived must not be stored as a band
                    # with no shots — see the atomicity note below.
                    complete = False
                    continue

                for _, row in df.iterrows():
                    fga = int(row.get("FGA") or 0)
                    if fga == 0:
                        # A zero-attempt row carries no information about how
                        # contested he was; storing it would inflate the table
                        # by every bench player in the league, every night.
                        continue
                    batch.append({
                        "player_id": str(row["PLAYER_ID"]),
                        "game_date": pd.to_datetime(date).date(),
                        "def_dist_range": band,
                        "season": season,
                        "fga": fga,
                        "fgm": int(row.get("FGM") or 0),
                        "fg_pct": float(row["FG_PCT"]) if pd.notna(row.get("FG_PCT")) else None,
                        "fga_frequency": (
                            float(row["FGA_FREQUENCY"])
                            if pd.notna(row.get("FGA_FREQUENCY")) else None
                        ),
                    })
                time.sleep(config.REQUEST_DELAY)

        # Atomic per date. Resume keys on "does this date exist at all", so a
        # date written with two of its four bands would be skipped forever and
        # silently read as a night with no wide-open looks. A transient DNS
        # failure did exactly that to three dates on the first run. Writing
        # nothing is recoverable; writing half is not.
        if not complete:
            skipped += 1
            continue

        if batch:
            with engine.begin() as conn:
                for item in batch:
                    conn.execute(
                        sqlite_upsert(PlayerGameContest)
                        .values(**item)
                        .on_conflict_do_update(
                            index_elements=["player_id", "game_date", "def_dist_range"],
                            set_={k: item[k] for k in
                                  ("season", "fga", "fgm", "fg_pct", "fga_frequency")},
                        )
                    )
            written += len(batch)

    print(f"\n{'='*62}")
    print(f"  CONTEST INGEST COMPLETE — {written:,} rows written")
    if skipped:
        print(f"  {skipped} date(s) left unwritten after retries — re-run to pick them up")
    print(f"{'='*62}\n")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest per-game contest levels.")
    parser.add_argument("--seasons", nargs="+", default=None)
    parser.add_argument("--full", action="store_true",
                        help="All seasons with matchup-era coverage (2016-17 on)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-pull dates already stored")
    args = parser.parse_args()

    if args.seasons:
        seasons = args.seasons
    elif args.full:
        seasons = [s for s in config.ALL_SEASONS if s >= "2016-17"]
    else:
        seasons = [config.ALL_SEASONS[-1]]

    ingest_contest(seasons, resume=not args.no_resume)
