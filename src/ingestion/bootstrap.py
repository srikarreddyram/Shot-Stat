"""
Bootstrap orchestrator — runs the full historical data ingestion pipeline.

Usage:
    python -m src.ingestion.bootstrap              # Single season test (2023-24)
    python -m src.ingestion.bootstrap --full        # All 15 seasons
    python -m src.ingestion.bootstrap --validate    # Run validation queries only
"""
import sys
import time
from pathlib import Path
from datetime import datetime

from sqlalchemy import select, func, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, init_db, get_session_factory
from src.db.models import Game, Player, Shot
from src.ingestion.game_ingestor import ingest_games
from src.ingestion.roster_ingestor import ingest_rosters
from src.ingestion.physical_ingestor_nba import ingest_physicals_nba
from src.ingestion.physical_ingestor_bref import ingest_physicals_bref
from src.ingestion.physical_ingestor_2k import ingest_physicals_2k
from src.ingestion.shot_ingestor import ingest_shots
from src.ingestion.export_csv import export_all as export_csvs


def validate_database():
    """Run validation queries on the database and print a report."""
    engine = get_engine()
    Session = get_session_factory(engine)

    print("\n" + "=" * 60)
    print("  DATABASE VALIDATION REPORT")
    print("=" * 60)

    with Session() as session:
        # ── Row counts ──
        game_count = session.execute(select(func.count()).select_from(Game)).scalar()
        player_count = session.execute(select(func.count()).select_from(Player)).scalar()
        shot_count = session.execute(select(func.count()).select_from(Shot)).scalar()

        print(f"\n  📊 Row Counts:")
        print(f"     Games:   {game_count:>8,}")
        print(f"     Players: {player_count:>8,}")
        print(f"     Shots:   {shot_count:>8,}")

        # ── Shots per season ──
        print(f"\n  📊 Shots per Season:")
        result = session.execute(
            select(Shot.season, func.count()).group_by(Shot.season).order_by(Shot.season)
        )
        for season, count in result:
            print(f"     {season}: {count:>8,}")

        # ── Null checks ──
        print(f"\n  🔍 Null Checks:")
        null_player = session.execute(
            select(func.count()).select_from(Shot).where(Shot.player_id.is_(None))
        ).scalar()
        null_game = session.execute(
            select(func.count()).select_from(Shot).where(Shot.game_id.is_(None))
        ).scalar()
        null_coords = session.execute(
            select(func.count()).select_from(Shot).where(
                (Shot.loc_x.is_(None)) | (Shot.loc_y.is_(None))
            )
        ).scalar()
        print(f"     Shots with null player_id: {null_player}")
        print(f"     Shots with null game_id:   {null_game}")
        print(f"     Shots with null coords:    {null_coords}")

        # ── Player physical data coverage ──
        print(f"\n  🏀 Player Physical Data:")
        total_players = session.execute(select(func.count(Player.player_id.distinct()))).scalar()
        has_height = session.execute(
            select(func.count(Player.player_id.distinct())).where(Player.height.isnot(None))
        ).scalar()
        has_weight = session.execute(
            select(func.count(Player.player_id.distinct())).where(Player.weight.isnot(None))
        ).scalar()
        has_wingspan = session.execute(
            select(func.count(Player.player_id.distinct())).where(Player.wingspan.isnot(None))
        ).scalar()
        
        nba_wingspan = session.execute(
            select(func.count(Player.player_id.distinct())).where(Player.wingspan_source == "NBA_API")
        ).scalar()
        bref_wingspan = session.execute(
            select(func.count(Player.player_id.distinct())).where(Player.wingspan_source == "BREF")
        ).scalar()
        twok_wingspan = session.execute(
            select(func.count(Player.player_id.distinct())).where(Player.wingspan_source == "2K")
        ).scalar()

        print(f"     Total unique players:     {total_players}")
        print(f"     With height:              {has_height} ({has_height/max(total_players,1)*100:.1f}%)")
        print(f"     With weight:              {has_weight} ({has_weight/max(total_players,1)*100:.1f}%)")
        print(f"     With wingspan:            {has_wingspan} ({has_wingspan/max(total_players,1)*100:.1f}%)")
        print(f"     Wingspan source NBA API:  {nba_wingspan} ({nba_wingspan/max(total_players,1)*100:.1f}%)")
        print(f"     Wingspan source BRef:     {bref_wingspan} ({bref_wingspan/max(total_players,1)*100:.1f}%)")
        print(f"     Wingspan source 2K:       {twok_wingspan} ({twok_wingspan/max(total_players,1)*100:.1f}%)")

        # ── Spot check: a well-known player ──
        print(f"\n  🔎 Spot Check (LeBron James, player_id=2544):")
        lebron = session.execute(
            select(Player).where(Player.player_id == "2544").order_by(Player.season.desc())
        ).scalars().first()
        if lebron:
            print(f"     Name:     {lebron.name}")
            print(f"     Season:   {lebron.season}")
            print(f"     Height:   {lebron.height}\" ({lebron.height/12:.0f}'{lebron.height%12:.0f}\")" if lebron.height else "     Height:   None")
            print(f"     Weight:   {lebron.weight} lbs" if lebron.weight else "     Weight:   None")
            print(f"     Wingspan: {lebron.wingspan}\" ({lebron.wingspan_source})" if lebron.wingspan else "     Wingspan: None")
            print(f"     Position: {lebron.position}")
            print(f"     FG%:      {lebron.season_fg_pct:.3f}" if lebron.season_fg_pct else "     FG%:      None")

            # Count his shots
            lebron_shots = session.execute(
                select(func.count()).select_from(Shot).where(Shot.player_id == "2544")
            ).scalar()
            print(f"     Shots:    {lebron_shots:,}")
        else:
            print("     ✗ Not found in DB")

        # ── Zone distribution ──
        print(f"\n  📊 Shot Zone Distribution:")
        result = session.execute(
            select(Shot.zone, func.count(), func.avg(Shot.shot_made))
            .group_by(Shot.zone)
            .order_by(func.count().desc())
        )
        print(f"     {'Zone':<30} {'Count':>8} {'FG%':>8}")
        print(f"     {'-'*30} {'-'*8} {'-'*8}")
        for zone, count, avg_made in result:
            zone_name = zone or "(null)"
            fg_pct = f"{avg_made*100:.1f}%" if avg_made is not None else "N/A"
            print(f"     {zone_name:<30} {count:>8,} {fg_pct:>8}")

    print(f"\n{'='*60}")
    print("  ✓ Validation complete")
    print(f"{'='*60}\n")


def remove_duplicates():
    """Remove any duplicate rows from the database."""
    engine = get_engine()
    
    print("\n" + "━" * 40)
    print("  Step 5/5: Deduplication Check")
    print("━" * 40)
    
    queries = [
        ("Games", "DELETE FROM games WHERE rowid NOT IN (SELECT MIN(rowid) FROM games GROUP BY game_id);"),
        ("Players", "DELETE FROM players WHERE rowid NOT IN (SELECT MIN(rowid) FROM players GROUP BY player_id, season);"),
        ("Shots", "DELETE FROM shots WHERE rowid NOT IN (SELECT MIN(rowid) FROM shots GROUP BY shot_id);")
    ]
    
    with engine.begin() as conn:
        for table, query in queries:
            result = conn.execute(text(query))
            print(f"  ✓ {table}: Removed {result.rowcount} duplicates")

def _check_games(seasons, Session):
    print("\n  🔍 Per-Season Game Check:")
    issues = []
    with Session() as session:
        result = session.execute(text(
            "SELECT substr(game_id, 1, 3) as prefix, COUNT(*) as cnt "
            "FROM games GROUP BY prefix"
        ))
        for prefix, cnt in result:
            game_type = "Regular" if prefix == "002" else "Playoff" if prefix == "004" else "Play-In"
            print(f"     {game_type} ({prefix}): {cnt:,} games")

    with Session() as session:
        total = session.execute(select(func.count()).select_from(Game)).scalar()
        has_win = session.execute(
            select(func.count()).select_from(Game).where(Game.home_team_win.isnot(None))
        ).scalar()
        pct = has_win / max(total, 1) * 100
        if pct < 90:
            print(f"     ⚠️  WARNING: Only {pct:.0f}% of games have home_team_win data!")
            issues.append("home_team_win coverage low")
        else:
            print(f"     ✓ home_team_win coverage: {pct:.0f}%")
    return issues


def _check_players(seasons, Session):
    print("\n  🔍 Per-Season Player Physical Data Check:")
    issues = []
    with Session() as session:
        for season in seasons:
            total = session.execute(
                select(func.count()).select_from(Player).where(Player.season == season)
            ).scalar()
            has_height = session.execute(
                select(func.count()).select_from(Player)
                .where(Player.season == season)
                .where(Player.height.isnot(None))
            ).scalar()
            has_weight = session.execute(
                select(func.count()).select_from(Player)
                .where(Player.season == season)
                .where(Player.weight.isnot(None))
            ).scalar()

            if total == 0:
                print(f"     ⚠️  {season}: NO PLAYERS FOUND!")
                issues.append(f"{season}: no players")
                continue

            h_pct = has_height / total * 100
            w_pct = has_weight / total * 100

            if h_pct < 80 or w_pct < 80:
                print(f"     ⚠️  {season}: {total} players — height {h_pct:.0f}%, weight {w_pct:.0f}% ← BELOW 80% THRESHOLD")
                issues.append(f"{season}: height={h_pct:.0f}% weight={w_pct:.0f}%")
            else:
                print(f"     ✓ {season}: {total} players — height {h_pct:.0f}%, weight {w_pct:.0f}%")

    if issues:
        print(f"\n     🚨 {len(issues)} season(s) have incomplete physical data!")
        print(f"     Re-run bootstrap to backfill missing height/weight.")
    else:
        print(f"\n     ✅ All seasons have 80%+ height/weight coverage.")
    return issues


def _check_shots(seasons, Session):
    MIN_SHOTS_PER_SEASON = 100_000
    print("\n  🔍 Per-Season Shot Count Check:")
    issues = []
    with Session() as session:
        for season in seasons:
            count = session.execute(
                select(func.count()).select_from(Shot).where(Shot.season == season)
            ).scalar()

            has_playoff = session.execute(
                select(func.count()).select_from(Shot)
                .where(Shot.season == season)
                .where(Shot.playoff_flag == 1)
            ).scalar()

            if count == 0:
                print(f"     ⚠️  {season}: NO SHOTS!")
                issues.append(f"{season}: 0 shots")
            elif count < MIN_SHOTS_PER_SEASON:
                print(f"     ⚠️  {season}: {count:,} shots (below {MIN_SHOTS_PER_SEASON:,} threshold) — playoff: {has_playoff:,}")
                issues.append(f"{season}: only {count:,} shots")
            else:
                print(f"     ✓ {season}: {count:,} shots (reg: {count - has_playoff:,}, playoff: {has_playoff:,})")

    if issues:
        print(f"\n     🚨 {len(issues)} season(s) have missing or low shot counts!")
    else:
        print(f"\n     ✅ All seasons have {MIN_SHOTS_PER_SEASON:,}+ shots.")
    return issues


def run_bootstrap(seasons: list[str]):
    """Run the full bootstrap pipeline."""
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  NBA SHOT QUALITY ENGINE — BOOTSTRAP")
    print(f"  Seasons: {seasons[0]} → {seasons[-1]} ({len(seasons)} seasons)")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    engine = get_engine()
    Session = get_session_factory(engine)
    all_issues = []

    # Step 1: Initialize database
    print("━" * 40)
    print("  Step 1/5: Initialize Database")
    print("━" * 40)
    engine = init_db()

    # Step 2: Ingest games (Skipped per user request)
    # print("\n" + "━" * 40)
    # print("  Step 2/5: Game Metadata")
    # print("━" * 40)
    # game_count = ingest_games(seasons)
    all_issues += _check_games(seasons, Session)

    # Step 3: Phase 1 (Roster) & Phase 2 (Physicals: NBA API → BRef → 2K)
    print("\n" + "━" * 40)
    print("  Step 3/5: Roster & Physical Attributes")
    print("━" * 40)
    ingest_rosters(seasons)
    ingest_physicals_nba()     # Phase 2A: NBA API (CommonPlayerInfo + Draft Combine)
    ingest_physicals_bref()    # Phase 2B: Basketball Reference fallback
    ingest_physicals_2k()      # Phase 2C: 2K Ratings final fallback
    player_issues = _check_players(seasons, Session)
    all_issues += player_issues

    # Step 4: Ingest shots (heaviest step)
    print("\n" + "━" * 40)
    print("  Step 4/5: Shot Chart Data (this takes a while...)")
    print("━" * 40)
    shot_count = ingest_shots(seasons)
    all_issues += _check_shots(seasons, Session)

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"  BOOTSTRAP COMPLETE")
    print(f"  Elapsed: {elapsed}")
    print(f"{'='*60}\n")

    # Step 5: Deduplication (Safety check)
    remove_duplicates()

    # Final health summary
    if all_issues:
        print(f"\n{'='*60}")
        print(f"  🚨 DATA QUALITY ISSUES FOUND ({len(all_issues)} total)")
        print(f"{'='*60}")
        for issue in all_issues:
            print(f"     • {issue}")
        print(f"\n  Re-run bootstrap to attempt backfill.")
        print(f"{'='*60}\n")
    else:
        print(f"\n  ✅ All per-season sanity checks passed!\n")

    # Run full validation
    validate_database()

    # Export to CSV so data is human-readable
    export_csvs()


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_database()
    elif "--full" in sys.argv:
        run_bootstrap(config.ALL_SEASONS)
    else:
        run_bootstrap(config.TEST_SEASONS)

