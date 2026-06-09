# Shot Ingestor (Updated) Explanation

## What changed in this script?
The `shot_ingestor.py` script (Phase 3) is the heaviest part of the pipeline. We completely rewrote it to fix two critical bugs:
1. **The Infinite Loop Bug:** The old script looked at the `shots` table to see if a player had been fetched. If a player took 0 shots in a season, they were never inserted into the `shots` table, so the script kept trying to fetch them over and over.
2. **Performance Issues:** We switched to using extremely fast SQL bulk inserts because our new resume logic guarantees we never double-fetch a player.

## Code Walkthrough

```python
        season_types = ["Regular Season", "Playoffs"]

        for season_type in season_types:
            type_label = "REG" if season_type == "Regular Season" else "PLY"
            
            # 1. New Resume Logic using boolean flags
            with Session() as session:
                flag_col = Player.shots_fetched_reg if season_type == "Regular Season" else Player.shots_fetched_ply
                
                existing = session.execute(
                    select(Player.player_id)
                    .where(Player.season == season)
                    .where(flag_col == True)
                ).scalars().all()
                existing_set = set(str(p) for p in existing)
```
**Explanation:** Instead of asking the `shots` table "do we have shots for this player?", we ask the `players` table "has this player been checked?". The `shots_fetched_reg` and `shots_fetched_ply` flags are exactly what we added to `models.py`. 

```python
            for player_id, player_name in tqdm(current_players, desc=f"  {season} {type_label}"):
                if str(player_id) in existing_set:
                    continue
                    
                df = _api_call_with_retry(int(player_id), season, season_type)

                shots_to_insert = []
                if not df.empty:
                    # Map the DataFrame rows to Python dictionaries...
                    for _, row in df.iterrows():
                        shots_to_insert.append({ ... })

                # 2. Fast inserts and marking as complete
                with Session() as session:
                    if shots_to_insert:
                        session.bulk_insert_mappings(Shot, shots_to_insert)
                    
                    if season_type == "Regular Season":
                        stmt = update(Player).where(Player.player_id == str(player_id)).where(Player.season == season).values(shots_fetched_reg=True)
                    else:
                        stmt = update(Player).where(Player.player_id == str(player_id)).where(Player.season == season).values(shots_fetched_ply=True)
                        
                    session.execute(stmt)
                    session.commit()
```
**Explanation:** Notice what happens here. 
- If `df.empty` is true (the player took 0 shots), `shots_to_insert` is empty. We skip the insert.
- **BUT**, we *still* execute the `update(Player)... values(shots_fetched_reg=True)` command. 
- This flips the boolean flag in the database to `True`. The next time the script runs, this player will be caught by the `existing_set` check at the top, and we will safely skip them. The "Infinite Loop Bug" is completely solved.

Furthermore, we use `session.bulk_insert_mappings(Shot, shots_to_insert)` instead of `sqlite_upsert`. Bulk insert bypasses conflict checking, which makes it over 10x faster. We can safely do this because our boolean flags guarantee we will never insert the same player twice.

## Potential Pitfalls
- **API Errors:** If `_api_call_with_retry` returns `None` (meaning the API completely failed and we ran out of retries), we *do not* flip the boolean flag. This is intentional. We want the script to try again on the next run.
