# Game Ingestor Explanation (`src/ingestion/game_ingestor.py`)

**Goal:** Query the NBA API for a list of every game played in a season, figure out who was the home team and who won, and save it to the database.

---

## 1. Deduplicating Games

When we ask the NBA API for the game log, it returns **two rows for every single game**. 
For example, when the Nuggets play the Lakers in Denver, the API gives us:
1. A row for Denver that says `MATCHUP: DEN vs. LAL`
2. A row for the Lakers that says `MATCHUP: LAL @ DEN`

If we save both rows to our database, we would have duplicates of every game! Here is the helper function that prevents that:

```python
def _parse_matchup(matchup: str, team_abbr: str, wl: str):
    """
    Parse 'DEN vs. LAL' or 'LAL @ DEN' to determine home/away teams.
    Returns (home_team, away_team, home_team_win) or None if this row is the away team's entry.
    """
    if " vs. " in matchup:
        # This is the home team's row — "HOME vs. AWAY"
        parts = matchup.split(" vs. ")
        home_won = 1 if wl == "W" else 0
        return parts[0].strip(), parts[1].strip(), home_won
    else:
        # This is the away team's row — "AWAY @ HOME" — skip to avoid dupes
        return None
```

### 🧠 Python Concepts Used:
- **`in` keyword:** `if " vs. " in matchup` checks if a smaller string exists inside a larger string.
- **`split()` method:** `matchup.split(" vs. ")` takes a string like `"DEN vs. LAL"` and chops it into a list: `["DEN", "LAL"]`.
- **`strip()` method:** Removes any accidental spaces from the beginning or end of the string.
- **Ternary Operator:** `1 if wl == "W" else 0` is a one-line `if/else` statement. It sets `home_won` to 1 (True) or 0 (False).
- **`None` keyword:** `None` is Python's way of saying "nothing" or "null". By returning `None` for the away team's row, we can easily tell our main loop to skip it.

---

## 2. Identifying Playoff Games

Every NBA game has a unique ID like `0021900001`. The NBA encodes what type of game it is in the first three digits.

```python
def _is_playoff_game(game_id: str) -> int:
    """
    NBA game IDs encode game type in the leading digits:
    - 002xxxxx = Regular Season
    - 004xxxxx = Playoffs
    - 005xxxxx = Play-In
    """
    if game_id.startswith("004") or game_id.startswith("005"):
        return 1
    return 0
```

### 🧠 Python Concepts Used:
- **`startswith()` method:** A built-in string method that checks the first few characters. Extremely fast and readable.
- **Type Hinting:** `-> int` tells anyone reading the code that this function will return an integer. It doesn't actually force the code to return an integer, but it's a great habit for readability.

---

## 3. The Main Loop & Error Handling

This is where the actual API call and database save happens.

```python
    for season in tqdm(seasons, desc="Seasons"):
        for season_type in ["Regular Season", "Playoffs"]:
            try:
                # 1. Ask the NBA API for the game log
                gl = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation="T", 
                )
                df = gl.get_data_frames()[0]
            except Exception as e:
                time.sleep(config.REQUEST_DELAY * 2)
                continue
```

### 🧠 Python Concepts Used:
- **`tqdm()`:** A third-party library that wraps around any list (like our `seasons` list) and draws a progress bar in the terminal.
- **`try / except`:** This is crucial for web scraping. The NBA API will randomly drop connections or block you for hitting it too fast. Without `try / except`, an error would instantly crash the whole script. 
  - `try` attempts the API call.
  - If the NBA API throws an error, it immediately jumps to `except Exception as e:`.
  - Inside the `except` block, `time.sleep()` pauses the script for a second, and `continue` tells the loop to skip to the next season instead of crashing.

---

## 4. The Upsert (Safe Database Saving)

After we clean the data, we need to save it to SQLite. 

```python
            if games_to_upsert:
                with Session() as session:
                    for game_data in games_to_upsert:
                        # Create the insert statement
                        stmt = sqlite_upsert(Game.__table__).values(**game_data)
                        
                        # Add the "ON CONFLICT" rule (The Upsert)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["game_id"],
                            set_={
                                "date": stmt.excluded.date,
                                "home_team": stmt.excluded.home_team,
                                "away_team": stmt.excluded.away_team,
                                "playoff_flag": stmt.excluded.playoff_flag,
                                "home_team_win": stmt.excluded.home_team_win,
                            },
                        )
                        session.execute(stmt)
                    session.commit()
```

### 🧠 Python & SQL Concepts Used:
- **`with Session() as session:`** The `with` statement is a "context manager". It opens a connection to the database, and when the indented block of code finishes, it *automatically closes* the connection. You never have to remember to do `session.close()`.
- **`**game_data`:** The double-asterisk is called "dictionary unpacking". It takes a Python dictionary and expands it into keyword arguments.
- **UPSERT (`on_conflict_do_update`):** This is the core of our "resume" logic. It tells the database: *"Try to insert this game. But if a game with this `game_id` already exists (`index_elements=["game_id"]`), don't throw an error! Instead, just update its values."* This guarantees we never get duplicate games even if you restart the script 100 times.
