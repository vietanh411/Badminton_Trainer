"""Engine / session factory and migration-based initialisation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings

BACKEND_DIR = Path(__file__).resolve().parents[2]

_engine: Engine | None = None
_factory: sessionmaker | None = None


def _make_engine(url: str) -> Engine:
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, pool_pre_ping=True, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def init_engine(settings: Settings | None = None) -> Engine:
    global _engine, _factory
    settings = settings or get_settings()
    _engine = _make_engine(settings.db_url)
    _factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    return _engine or init_engine()


def session_factory() -> sessionmaker:
    if _factory is None:
        init_engine()
    return _factory  # type: ignore[return-value]


def get_db() -> Iterator[Session]:
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()


def run_migrations(settings: Settings | None = None) -> None:
    """Apply Alembic migrations up to head (creates the schema on first run)."""
    from alembic import command
    from alembic.config import Config

    settings = settings or get_settings()
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", settings.db_url)
    command.upgrade(cfg, "head")
