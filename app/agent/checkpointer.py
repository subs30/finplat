"""The real, durable checkpoint store — LangGraph's Postgres checkpointer,
pointed at finplat's existing database. Not an in-memory stub: killing
the process mid-investigation and reconnecting with the same thread_id
resumes from the last completed step, because the state genuinely lives
in Postgres, not in this process's memory. See tests/test_agent_checkpoint.py
for the explicit resume proof.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import get_settings


def _psycopg_conn_string() -> str:
    # settings.DATABASE_URL is SQLAlchemy-flavored (postgresql+psycopg://),
    # normalized that way in app/config.py for SQLAlchemy/alembic's
    # benefit. langgraph-checkpoint-postgres talks to psycopg directly and
    # expects a plain postgresql:// string — strip the "+psycopg" driver
    # marker rather than introduce a second settings field for what is
    # the same database.
    return get_settings().DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Yields a ready-to-use AsyncPostgresSaver, with .setup() applied.

    .setup() creates langgraph's own checkpoint tables (checkpoints,
    checkpoint_writes, checkpoint_blobs, checkpoint_migrations) if they
    don't exist yet — idempotent, safe to call on every connection rather
    than requiring a separate one-time migration step.
    """
    async with AsyncPostgresSaver.from_conn_string(_psycopg_conn_string()) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
