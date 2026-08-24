"""
Tests for diacritic-insensitive player search (src/db/database.py's
UNACCENT SQL function).

Regression context: /players/search used plain `LOWER(name) LIKE LOWER(:q)`,
so searching "Doncic" (how most people type it) found nothing for a player
named "Dončić" in the database — SQLite's LIKE has no built-in diacritic
folding. Common for this dataset specifically: Dončić, Jokić, Porziņģis,
Šarić, and other international players all have real diacritics in their
official NBA name field.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from src.db.database import _strip_diacritics, register_unaccent


def test_strip_diacritics_removes_accents():
    assert _strip_diacritics("Dončić") == "Doncic"
    assert _strip_diacritics("Jokić") == "Jokic"
    assert _strip_diacritics("Porziņģis") == "Porzingis"


def test_strip_diacritics_leaves_plain_ascii_unchanged():
    assert _strip_diacritics("Stephen Curry") == "Stephen Curry"


def test_strip_diacritics_handles_none():
    assert _strip_diacritics(None) is None


def test_unaccent_sql_function_matches_unaccented_query():
    """End-to-end: UNACCENT() registered on a plain SQLite connection lets
    an unaccented search term match an accented stored name."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    register_unaccent(engine)

    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE players (name TEXT)"))
        conn.execute(text("INSERT INTO players (name) VALUES ('Luka Dončić')"))
        conn.commit()

        rows = conn.execute(text(
            "SELECT name FROM players WHERE LOWER(UNACCENT(name)) LIKE LOWER(UNACCENT(:q))"
        ), {"q": "%doncic%"}).fetchall()

    assert [r[0] for r in rows] == ["Luka Dončić"]
