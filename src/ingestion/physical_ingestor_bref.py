"""
Phase 2B: Basketball Reference Physical Data Collection (Fallback)
Iterates through unique players STILL missing physicals after the NBA API check
and scrapes exact measurements from Basketball Reference.
"""
import sys
import time
import re
from pathlib import Path
import urllib.parse

import requests
from bs4 import BeautifulSoup
from sqlalchemy import select, update
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.db.database import get_engine, get_session_factory
from src.db.models import Player

BREF_SEARCH_URL = "https://www.basketball-reference.com/search/search.fcgi?search="

def _safe_float(val, default=None):
    if val is None: return default
    try: return float(val)
    except (ValueError, TypeError): return default

def scrape_bref_physicals(player_name: str) -> dict | None:
    """Scrapes height, weight, and wingspan from BRef. Returns dict or None."""
    try:
        url = BREF_SEARCH_URL + urllib.parse.quote(player_name)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return None
            
        soup = BeautifulSoup(resp.content, "lxml")
        
        # If it's a search results page (multiple players with same name), we take the first active one
        if "Search Results" in soup.title.text:
            results = soup.find_all("div", class_="search-item")
            if not results: return None
            link = results[0].find("a")
            if not link: return None
            player_url = "https://www.basketball-reference.com" + link["href"]
            time.sleep(2)  # Delay before following link
            resp = requests.get(player_url, headers=headers, timeout=10)
            if resp.status_code != 200: return None
            soup = BeautifulSoup(resp.content, "lxml")

        # Now we are on the player profile page
        physicals = {"height": None, "weight": None, "wingspan": None}
        
        # Height and Weight usually in an info block
        info_div = soup.find("div", id="info")
        if not info_div: return None
        
        # Height: looks like <span itemprop="height">6-9</span>
        h_span = info_div.find("span", itemprop="height")
        if h_span:
            h_text = h_span.text.strip()
            match = re.match(r"(\d+)-(\d+)", h_text)
            if match:
                physicals["height"] = int(match.group(1)) * 12.0 + int(match.group(2))
                
        # Weight: looks like <span itemprop="weight">250lb</span>
        w_span = info_div.find("span", itemprop="weight")
        if w_span:
            w_text = w_span.text.replace("lb", "").strip()
            physicals["weight"] = _safe_float(w_text)
            
        # Wingspan is harder. Sometimes in the text: "Wingspan:  7-0"
        text_content = info_div.get_text()
        ws_match = re.search(r"Wingspan:\s+(\d+)-(\d+)", text_content)
        if ws_match:
            physicals["wingspan"] = int(ws_match.group(1)) * 12.0 + int(ws_match.group(2))
            
        return physicals
    except Exception as e:
        print(f"\n  ✗ BRef scrape error for {player_name}: {e}")
        return None

def ingest_physicals_bref():
    engine = get_engine()
    Session = get_session_factory(engine)
    
    with Session() as session:
        # Get unique players STILL missing height, weight, or wingspan
        query = select(Player.player_id, Player.name).where(
            (Player.height.is_(None)) | 
            (Player.weight.is_(None)) | 
            (Player.wingspan.is_(None))
        ).distinct()
        missing_players = session.execute(query).all()
        
        print(f"\nFound {len(missing_players)} unique players needing BRef fallback scrape...")
        
        updated_count = 0
        for pid, name in tqdm(missing_players, desc="BRef Physicals"):
            phys = scrape_bref_physicals(name)
            
            if phys and (phys["height"] or phys["weight"] or phys["wingspan"]):
                # We only update wingspan_source if we actually found a wingspan on BRef
                ws_source = "BREF" if phys["wingspan"] else None
                
                # Fetch existing row to avoid overwriting existing NBA_API data with None
                # Actually, the update statement can just update everything. Wait, if height is None on BRef, 
                # we don't want to overwrite a valid height from NBA API. 
                # Better to selectively update.
                update_vals = {}
                if phys["height"]: update_vals["height"] = phys["height"]
                if phys["weight"]: update_vals["weight"] = phys["weight"]
                if phys["wingspan"]: 
                    update_vals["wingspan"] = phys["wingspan"]
                    update_vals["wingspan_source"] = ws_source
                
                if update_vals:
                    stmt = update(Player).where(Player.player_id == pid).values(**update_vals)
                    session.execute(stmt)
                    updated_count += 1
            
            # Be very polite to BRef to avoid 403s
            time.sleep(3.5)
            
        session.commit()
        print(f"\n✓ Checked {len(missing_players)} players, applied BRef data to {updated_count}.")

if __name__ == "__main__":
    ingest_physicals_bref()
