"""The shared SQLAlchemy declarative base every table in this package
registers against. Every model module imports `Base` from here rather
than each defining its own — a second `DeclarativeBase` subclass would
start its own, disconnected mapper registry, and string-based
`relationship(...)` references (e.g. Game <-> Shot) would silently fail
to resolve across the two.
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
