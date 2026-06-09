"""
Database engine, session management, and initialization.
"""
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from src.db.models import Base


def get_engine(echo=False):
    """Create and return a SQLAlchemy engine."""
    return create_engine(
        config.SQLALCHEMY_DATABASE_URL,
        echo=echo,
        # SQLite-specific: enable WAL mode for better concurrent read performance
        connect_args={"check_same_thread": False},
    )


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
