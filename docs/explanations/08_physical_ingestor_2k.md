# NBA 2K Ratings Physical Ingestor Explanation

## What does this script do?
The `physical_ingestor_2k.py` script is Phase 2C — the **final fallback** in our physical attributes pipeline. After the NBA API (Phase 2A) and Basketball Reference (Phase 2B) have done their best, there are still hundreds of players missing wingspan data. This script scrapes those exact measurements from **2kratings.com**, which sources its data directly from the NBA 2K game files.

## Why 2K Ratings?
NBA 2K models every single player with accurate body proportions. Their developers use real measurements when available (from the Draft Combine, team records, etc.) and create extremely accurate visual estimates for players who never got officially measured. This means:
- **Luka Dončić** (who skipped the NBA Draft Combine) has a wingspan listed as 6'11" (83 inches)
- **Every single NBA player** in the 2K franchise has a wingspan value

## How the URL slugs work

```python
def _name_to_slug(name: str) -> str:
    # 1. Strip diacritics:  Dončić  →  Doncic
    nfkd = unicodedata.normalize("NFD", name)
    ascii_name = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    
    # 2. Lowercase + remove dots/apostrophes
    slug = ascii_name.lower().strip()
    slug = re.sub(r"[.']", "", slug)
    
    # 3. Replace non-alphanumeric chars with dashes
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug
```
**Explanation:** 2kratings.com uses URL slugs like `https://www.2kratings.com/luka-doncic`. We need to convert the player's full name from our database into this format. The trickiest part is handling special characters:
- `Luka Dončić` → strip the háčky (accents) → `luka-doncic`
- `P.J. Tucker` → remove dots → `pj-tucker`
- `Shai Gilgeous-Alexander` → keep the hyphen → `shai-gilgeous-alexander`

## How we parse the profile page

```python
soup = BeautifulSoup(resp.content, "lxml")
info_div = soup.find("div", class_="player-info")

for p_tag in info_div.find_all("p"):
    text = p_tag.get_text(strip=True)
    
    if text.startswith("Height:"):
        physicals["height"] = _parse_feet_inches(text)
    elif text.startswith("Weight:"):
        physicals["weight"] = _parse_weight(text)
    elif text.startswith("Wingspan:"):
        physicals["wingspan"] = _parse_feet_inches(text)
```
**Explanation:** On every 2kratings player page, there is a `<div class="player-info">` containing `<p>` tags like:
- `Height:6'8" (203cm)`
- `Weight:230lbs (104kg)` 
- `Wingspan:6'11" (211cm)`

We loop through all the `<p>` tags, check what they start with, and parse the feet-inches format into total inches (e.g., 6'11" = 83.0 inches).

## How we avoid overwriting good data

```python
existing = session.execute(
    select(Player.height, Player.weight)
    .where(Player.player_id == pid)
    .limit(1)
).first()

if existing and existing.height is None and phys["height"]:
    update_vals["height"] = phys["height"]
if existing and existing.weight is None and phys["weight"]:
    update_vals["weight"] = phys["weight"]

# Always set wingspan (that's why we're here)
update_vals["wingspan"] = phys["wingspan"]
update_vals["wingspan_source"] = "2K"
```
**Explanation:** Before updating, we check the player's current row in the database. If their height already came from the NBA API (which is the most official source), we do NOT overwrite it with the 2K value. We only fill in columns that are still `NULL`. The wingspan is always set because that's specifically what this Phase 2C is trying to fill.

## Potential Pitfalls
1. **Rate limiting (403s):** 2kratings.com will ban you if you send requests faster than ~3 seconds apart. We use a 4-second delay to be safe.
2. **Name mismatches:** If a player's name in the NBA API doesn't match the 2kratings slug format, the scraper will get a 404. Example: a player listed as "Nene" in the API might be "nene-hilario" on 2kratings. These players will be reported in the "not found" count.
3. **Historic players:** Very old players (pre-NBA 2K era, before ~2000) likely won't have a 2kratings page at all.
