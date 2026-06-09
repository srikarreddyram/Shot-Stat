# Basketball Reference Physical Ingestor Explanation

## What does this script do?
The `physical_ingestor_bref.py` script is Phase 2B (the Fallback Phase). It takes over where the NBA API fails. It queries the database for players who are *still* missing physical data, and uses a web scraper to fetch their exact measurements from Basketball-Reference.com.

## Code Walkthrough

```python
def scrape_bref_physicals(player_name: str) -> dict | None:
    # 1. Formulate the search URL
    url = "https://www.basketball-reference.com/search/search.fcgi?search=" + urllib.parse.quote(player_name)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=10)
```
**Explanation:** Basketball-Reference has a very handy search endpoint. By sending an HTTP GET request to this URL, we can search for a player by their name. We use a User-Agent header so the server thinks we are a normal browser, not a robot.

```python
    soup = BeautifulSoup(resp.content, "lxml")
    
    # 2. Handle Search Results pages
    if "Search Results" in soup.title.text:
        results = soup.find_all("div", class_="search-item")
        if not results: return None
        link = results[0].find("a")
        ...
        resp = requests.get(player_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, "lxml")
```
**Explanation:** Sometimes two players have the same name (e.g., "Seth Curry"). If BRef returns a search results page instead of a direct profile, we find the first valid profile link, wait 2 seconds (to be polite to the server), and fetch that profile page.

```python
    # 3. Extract the data
    info_div = soup.find("div", id="info")
    
    # Height
    h_span = info_div.find("span", itemprop="height")
    if h_span:
        h_text = h_span.text.strip()
        match = re.match(r"(\d+)-(\d+)", h_text)
        if match:
            physicals["height"] = int(match.group(1)) * 12.0 + int(match.group(2))
```
**Explanation:** BRef uses standard schema.org HTML tags. We find the span with `itemprop="height"` (which looks like "6-9"), parse it using a Regular Expression (`\d+-\d+`), and convert it to total inches (6*12 + 9 = 81 inches). We do the exact same thing for weight.

```python
    # Wingspan
    text_content = info_div.get_text()
    ws_match = re.search(r"Wingspan:\s+(\d+)-(\d+)", text_content)
    if ws_match:
        physicals["wingspan"] = int(ws_match.group(1)) * 12.0 + int(ws_match.group(2))
```
**Explanation:** Wingspan isn't explicitly tagged on BRef, it's just raw text on the page. We pull all the text from the info box and use a Regex to search for the word "Wingspan:" followed by a measurement.

```python
        # 4. Update the DB conditionally
        for pid, name in missing_players:
            phys = scrape_bref_physicals(name)
            if phys and ...:
                update_vals = {}
                if phys["height"]: update_vals["height"] = phys["height"]
                if phys["weight"]: update_vals["weight"] = phys["weight"]
                if phys["wingspan"]: 
                    update_vals["wingspan"] = phys["wingspan"]
                    update_vals["wingspan_source"] = "BREF"
                
                stmt = update(Player).where(Player.player_id == pid).values(**update_vals)
                session.execute(stmt)
```
**Explanation:** We iterate over missing players, run the scraper, and then dynamically build an `update_vals` dictionary. This is important: if BRef has a wingspan but *doesn't* have a height, we only update the wingspan. We don't want to accidentally overwrite a good height we got from the NBA API with a `None` value from BRef.

## Potential Pitfalls
1. **IP Bans:** BRef is incredibly strict about bot traffic. We *must* use `time.sleep(3.5)` between calls. If we go too fast, they will return a `429 Too Many Requests` status code and ban the IP address for 24 hours.
2. **Name Clashes:** If a random player from the 1950s has the exact same name as a modern player, the search scraper might click the wrong profile. This is rare, but possible.
