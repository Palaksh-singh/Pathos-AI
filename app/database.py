"""
Pathos AI — Database Engine & Session Management
====================================================
Async SQLAlchemy 2.0 engine, defaulting to SQLite (aiosqlite) for local dev
and trivially pointed at Postgres in staging/production via DATABASE_URL
(e.g. postgresql+asyncpg://user:pass@host/db).
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger("pathos_ai.database")


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Creates tables on startup. In production this is replaced by Alembic
    migrations (see README's 'Trade-offs' section) — kept as create_all here
    so the project boots with zero external migration tooling for local dev."""
    from app.models import db_models  # noqa: F401 — ensures models are registered on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized", extra={"url_scheme": settings.database_url.split(":")[0]})


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session and guarantees close/rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Context-manager variant for use outside of FastAPI's DI system
    (e.g. background jobs, scripts)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
