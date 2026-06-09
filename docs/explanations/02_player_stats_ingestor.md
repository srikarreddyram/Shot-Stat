# Player Stats Ingestor Explanation (`src/ingestion/player_stats_ingestor.py`)

**Goal:** Fetch player rosters, season stats, height, weight, and combine wingspan to populate the `Players` table.

This script is much more complex than `game_ingestor.py` because it has to merge data from three completely different API endpoints, and handle the fact that some endpoints strictly limit how fast you can call them.

---

## 1. Exponential Backoff (Handling Strict APIs)

The NBA's `CommonPlayerInfo` endpoint (which gives us height and weight) is heavily protected. If you ask for 100 players too quickly, it will block your IP address.

```python
def _fetch_player_info_with_retry(player_id: str, max_retries=5) -> dict | None:
    """Fetch CommonPlayerInfo with exponential backoff for rate limits."""
    backoff = 10  # Initial wait time
    for attempt in range(max_retries):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id), timeout=15)
            df = info.get_data_frames()[0]
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch info for player {player_id}")
                return None
            
            # Exponential Backoff
            time.sleep(backoff)
            backoff *= 2  # Double the wait time for the next attempt
    return None
```

### 🧠 Python Concepts Used:
- **Exponential Backoff:** A classic computer science technique. If a server says "too many requests", you wait 10 seconds. If it fails again, you wait 20 seconds. Then 40 seconds. This is the polite way to scrape data without getting permanently banned.
- **`range(max_retries)`:** Creates a loop that will run exactly 5 times (0, 1, 2, 3, 4).
- **Early Return:** Notice how if `df.empty` is false, it immediately does `return df.iloc[0].to_dict()`. Returning out of a function instantly stops the loop!

---

## 2. The Resume Logic (And the Infamous Erase Bug)

Because fetching physical data takes hours, we want to skip players we've already fetched if you kill the terminal and restart.

```python
        # Step A: Check which players we already have physical data for in the database
        existing_physical_data = {}
        with Session() as session:
            result = session.execute(
                select(Player.player_id, Player.height, Player.weight, Player.position)
                .where(Player.height.isnot(None), Player.season == season)
            ).all()
            
            # Store their existing data in a dictionary
            for row in result:
                existing_physical_data[str(row[0])] = {
                    "height": row[1],
                    "weight": row[2],
                    "position_from_info": row[3],
                }

        # Step B: Identify who is missing
        player_ids = [str(row["PLAYER_ID"]) for _, row in stats_df.iterrows()]
        players_needing_physical = [pid for pid in player_ids if pid not in existing_physical_data]
```

### 🚨 The "Erase Bug" Explained:
In the older version of the code, I successfully identified who was missing data (`players_needing_physical`). BUT, I didn't actually load the *existing* data into `existing_physical_data`. I just stored their IDs.

So, when the script went to save the players to the database, it said: *"I didn't fetch height for LeBron this run, so his height must be `None`!"* It then used an UPSERT to update LeBron's row in the database, overwriting his perfectly good height with `NULL`. 

**The Fix:** I updated the code above to fully pull LeBron's existing height/weight from the database into the `existing_physical_data` dictionary. Now, when it UPSERTs, it injects his existing data back into the row, preserving it perfectly!

### 🧠 Python Concepts Used:
- **List Comprehensions:** `[pid for pid in player_ids if pid not in existing_physical_data]` is a highly optimized, one-line `for` loop. It creates a new list containing only the IDs that need data.

---

## 3. The Wingspan Overwrite Vulnerability

We have a separate script (`fix_wingspans.py`) that gets highly accurate wingspans. We use `wingspan_imputed` to flag the quality:
- `0`: Real measurement
- `2`: Accurate height-based estimate
- `1`: Terrible position-based average

If `player_stats_ingestor.py` runs, it blindly recalculates the terrible position average and sets `wingspan_imputed = 1`. 

### 🚨 The Vulnerability:
If you run `fix_wingspans.py`, and *then* run `bootstrap --full`, `player_stats_ingestor.py` will hit the UPSERT statement:

```python
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id", "season"],
        set_={
            "wingspan": stmt.excluded.wingspan,
            "wingspan_imputed": stmt.excluded.wingspan_imputed,
        },
    )
```

It will overwrite our good `0` or `2` data with its bad `1` data!

**The Fix (To Be Applied):**
We need to use an SQL `CASE` statement to conditionally update the column. We tell SQLite: *"Only update the wingspan IF the incoming data is better than what's already in the database."* I will apply this fix shortly.
