"""
Database engine, session management, and initialization.
"""
import sys
import unicodedata
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

# Add project root to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from src.db.models import Base


def _strip_diacritics(value: str | None) -> str | None:
    """'Dončić' -> 'Doncic'. Lets player search match on unaccented input,
    which is how most people type names like Dončić, Jokić, or Porziņģis."""
    if value is None:
        return None
    return "".join(
        c for c in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(c)
    )


def register_unaccent(engine):
    """
    Registers UNACCENT(x) as a SQL function on every new DBAPI connection —
    SQLite has no built-in diacritic folding, and this is the one place
    search needs it (player names like "Dončić", "Jokić", "Porziņģis").

    get_engine() calls this automatically. Any OTHER engine pointed at a
    SQLite DB and queried with UNACCENT() — e.g. a test's isolated
    in-memory fixture — must call this itself, since the function lives on
    the DBAPI connection, not the database file.
    """
    @event.listens_for(engine, "connect")
    def _register_unaccent(dbapi_connection, connection_record):
        dbapi_connection.create_function("UNACCENT", 1, _strip_diacritics)


def get_engine(echo=False):
    """Create and return a SQLAlchemy engine."""
    engine = create_engine(
        config.SQLALCHEMY_DATABASE_URL,
        echo=echo,
        # SQLite-specific:
        #   check_same_thread=False  → allow cross-thread access
        #   timeout=30               → wait up to 30s if DB is locked instead of crashing
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    register_unaccent(engine)
    return engine


def get_session_factory(engine=None):
    """Return a session factory bound to the given engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def init_db(engine=None, echo=False):
    """
    Create all tables. Idempotent — safe to call multiple times.
    Returns the engine used.
    """
    if engine is None:
        engine = get_engine(echo=echo)

    # Enable WAL mode for SQLite (better concurrency)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()

    Base.metadata.create_all(engine)
    print(f"✓ Database initialized at {config.DB_PATH}")
    return engine


if __name__ == "__main__":
    init_db(echo=True)
