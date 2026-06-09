# Shot Ingestor Explanation (`src/ingestion/shot_ingestor.py`)

**Goal:** Fetch every single shot taken by every player (coordinates, shot type, make/miss, time remaining).

This is the heaviest script. It makes over 15,000 API calls and downloads nearly 2 million rows of data. Efficiency and Resume Logic are the absolute top priorities here.

---

## 1. The Playoff Roster Filter (Optimizing API Calls)

The `shotchartdetail` API endpoint requires a `player_id`. If we want shots for the 2013-14 Regular Season, we must loop through all ~500 players who played that season.

But what about the 2013-14 Playoffs? Only ~200 players made the playoffs. If we loop through all 500 players, we will make 300 useless API calls that just return empty data!

```python
        # Get actual playoff roster for this season to avoid checking players who didn't make it
        try:
            playoff_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Playoffs"
            )
            playoff_df = playoff_stats.get_data_frames()[0]
            
            # Put all playoff player IDs in a fast 'set'
            playoff_player_ids = set(str(pid) for pid in playoff_df["PLAYER_ID"])
            
            # Filter the regular season list to only those who made the playoffs
            ply_players = [p for p in reg_players if str(p.player_id) in playoff_player_ids]
        except Exception:
            # Fallback if API fails: try everyone (slow but safe)
            ply_players = reg_players
```

### 🧠 Python Concepts Used:
- **`set()`:** A Python set is like a list, but it cannot contain duplicates, and looking up an item (`if pid in set:`) is mathematically instantaneous ($O(1)$ time). Searching through a list is slow ($O(n)$ time).
- **List Comprehension Filtering:** `[p for p in reg_players if str(p.player_id) in playoff_player_ids]` rapidly filters down our master roster to just the 200 players we need.

---

## 2. The Resume Logic (And the 0-Shot Loop)

When you kill the terminal, the script needs to know exactly where to pick up.

```python
            # Find players we've already ingested by checking the Shots table
            with Session() as session:
                existing = session.execute(
                    select(Shot.player_id)
                    .where(Shot.season == season)
                    .where(Shot.playoff_flag == playoff_val)
                    .distinct()
                ).scalars().all()
                existing_set = set(str(p) for p in existing)

            # Loop through roster
            for player_id, player_name in tqdm(current_players):
                # If they are in the DB, skip them instantly!
                if str(player_id) in existing_set:
                    continue
                
                # Otherwise, call the API
                df = _api_call_with_retry(int(player_id), season, season_type)
```

### 🚨 The "0-Shot Loop" Vulnerability Explained:
This logic looks at the `Shots` table to see if a player is "done". 

But what if a player (like Udonis Haslem) was on the active roster but **never took a shot** all season? 
1. The script hits the API for Haslem.
2. API returns 0 shots.
3. Script saves 0 shots to the DB.
4. Next time you restart the script, it queries the `Shots` table. Haslem isn't there!
5. Script hits the API for Haslem *again*.

Because it never records the fact that it checked him, it gets stuck re-checking every single 0-shot player every time you run `bootstrap --full`.

**The Fix (To Be Applied):**
We must completely decouple the resume logic from the `Shots` table. We will add a boolean flag directly to the `Players` table called `shots_fetched`. When we check Haslem, we set `shots_fetched = True`. The resume logic will only check that flag, breaking the infinite loop.

---

## 3. Bulk Inserts (Speed over Upserts)

In `game_ingestor` and `player_stats_ingestor`, we used SQLite UPSERTs (`on_conflict_do_update`). 
But for the Shots table, we are inserting 200,000 rows per season. UPSERTing 200,000 rows one-by-one is incredibly slow.

```python
                    with Session() as session:
                        if shots_to_insert:
                            session.bulk_insert_mappings(Shot, shots_to_insert)
                            session.commit()
```

### 🧠 Python & SQLAlchemy Concepts Used:
- **`bulk_insert_mappings`:** This is an extremely fast SQLAlchemy method. Instead of inserting rows one at a time, it bundles all 1,500 shots for a player into a single massive SQL statement and blasts them into the database in milliseconds. 
- **The Tradeoff:** Bulk inserts *cannot* do UPSERTs. If there is a primary key collision (duplicate shot), the entire batch crashes. But because our resume logic strictly skips players who are already in the database, we guarantee we will never insert duplicates, allowing us to safely use this lightning-fast method.
