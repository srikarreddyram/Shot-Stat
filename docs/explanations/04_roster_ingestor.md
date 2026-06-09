# Roster Ingestor Explanation

## What does this script do?
The `roster_ingestor.py` script is Phase 1 of our strict data ingestion pipeline. It has only one job: get the list of players who played in a given season and insert their base information (ID, Name, Position, Season FG%) into the database. It explicitly **does not** fetch physical attributes (height, weight, wingspan).

## Code Walkthrough

```python
def ingest_rosters(seasons: list[str]):
    engine = get_engine()
    Session = get_session_factory(engine)
    total_players = 0

    for season in tqdm(seasons, desc="Seasons (Roster)"):
        print(f"\n  Pulling roster for {season}...")
        try:
            # We call the NBA API for the season dashboard
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
            )
            df = stats.get_data_frames()[0]
        except Exception as e:
            ...
```
**Explanation:** For each season, we ask the NBA API's `LeagueDashPlayerStats` endpoint for the season summary. This returns a DataFrame containing a row for every single player who played that season, including their Name, ID, Position, and Field Goal Percentage.

```python
        players_to_upsert = []
        for _, row in df.iterrows():
            players_to_upsert.append({
                "player_id": str(row["PLAYER_ID"]),
                "season": season,
                "name": row["PLAYER_NAME"],
                "position": _normalize_position(str(row.get("PLAYER_POSITION", ""))),
                "season_fg_pct": _safe_float(row.get("FG_PCT")),
                
                # We explicitly set these to None so Phase 2 can fill them in!
                "height": None,
                "weight": None,
                "wingspan": None,
                "wingspan_source": None,
            })
```
**Explanation:** We loop over every player in the API response. Notice that we set `height`, `weight`, and `wingspan` to `None`. This is crucial because it ensures that when this script inserts the rows into the database, those physical columns are left empty and ready for the next phase to find them.

```python
        if players_to_upsert:
            with Session() as session:
                for p_data in players_to_upsert:
                    stmt = sqlite_upsert(Player.__table__).values(**p_data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["player_id", "season"],
                        set_={
                            "name": stmt.excluded.name,
                            "position": stmt.excluded.position,
                            "season_fg_pct": stmt.excluded.season_fg_pct,
                        }
                    )
                    session.execute(stmt)
                session.commit()
```
**Explanation:** This block inserts the data into the SQLite database. The `on_conflict_do_update` part is important: it means if a row for this `player_id` + `season` already exists, we simply update the name, position, and FG% instead of crashing with an error.

## Potential Pitfalls
1. **API Rate Limiting:** The `leaguedashplayerstats` endpoint is heavily rate-limited by the NBA API. We sleep after each call to prevent being IP-banned.
2. **Missing Players:** A player who only played in the Playoffs but not the Regular Season wouldn't be caught here. However, this is exceptionally rare (almost impossible in modern NBA rules).
