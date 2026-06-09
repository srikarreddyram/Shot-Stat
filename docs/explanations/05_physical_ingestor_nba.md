# NBA API Physical Ingestor Explanation

## What does this script do?
The `physical_ingestor_nba.py` script is Phase 2A of our pipeline. It looks at the database, finds all unique players who are missing height, weight, or wingspan, and makes a dedicated call to the NBA API for each of those players to fetch their exact physical attributes. 

## Code Walkthrough

```python
def ingest_physicals_nba():
    engine = get_engine()
    Session = get_session_factory(engine)
    
    # 1. First, we load all draft combine wingspans into a memory dictionary.
    wingspan_map = _get_wingspan_lookup()
```
**Explanation:** Before querying players, we call `_get_wingspan_lookup()`. This function downloads the Draft Combine results for every year from 2000 to present and stores them in a Python dictionary. This is much faster than looking up wingspan player-by-player.

```python
    with Session() as session:
        # 2. Query only players who are missing data
        query = select(Player.player_id).where(
            (Player.height.is_(None)) | 
            (Player.weight.is_(None)) | 
            (Player.wingspan.is_(None))
        ).distinct()
        missing_players = session.execute(query).scalars().all()
```
**Explanation:** This is the core "resume" logic. We ask SQLite: "Give me the IDs of any player who has a NULL value for height, weight, OR wingspan." We use `.distinct()` so we only get one ID per player, even if they played 15 seasons. This completely eliminates redundant API calls.

```python
        for pid in tqdm(missing_players, desc="NBA API Physicals"):
            # 3. Call the NBA API for this specific player
            info = _fetch_player_info_with_retry(pid)
            
            height = None
            weight = None
            if info:
                height = _height_to_inches(info.get("HEIGHT"))
                weight = _safe_float(info.get("WEIGHT"))
                
            wingspan = wingspan_map.get(str(pid))
            wingspan_source = "NBA_API" if wingspan else None
```
**Explanation:** For every player missing data, we fetch their `CommonPlayerInfo` (which has exact height/weight). Then, we check our `wingspan_map` from the Draft Combine to see if we have their exact wingspan. We set `wingspan_source` to `"NBA_API"` so we know where this data came from.

```python
            if height or weight or wingspan:
                # 4. Update the database
                stmt = update(Player).where(Player.player_id == pid).values(
                    height=height,
                    weight=weight,
                    wingspan=wingspan,
                    wingspan_source=wingspan_source
                )
                session.execute(stmt)
```
**Explanation:** Finally, if we found *any* data for this player, we use an `UPDATE` statement. Because we do `where(Player.player_id == pid)`, this updates the physical stats for **all seasons** this player played simultaneously.

## Potential Pitfalls
1. **Old Players:** The Draft Combine data only goes back to 2000. Players drafted before 2000 (like Michael Jordan or Shaquille O'Neal) won't have wingspans here, which means they will be passed on to Phase 2B (the Basketball Reference scraper).
2. **API Reliability:** `CommonPlayerInfo` is notoriously flaky. If the API returns a timeout, the player is skipped, and they will simply be checked again the next time the script is run because their data will still be `NULL`.
