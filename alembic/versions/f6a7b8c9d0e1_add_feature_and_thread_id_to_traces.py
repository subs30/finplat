"""add feature and thread_id to traces

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-09-19 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: str | Sequence[str] | None = 'e5f6a7b8c9d0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("traces", sa.Column("feature", sa.String(length=50), nullable=True))
    op.add_column("traces", sa.Column("thread_id", sa.String(length=64), nullable=True))
    op.create_index("ix_traces_thread_id", "traces", ["thread_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_traces_thread_id", table_name="traces")
    op.drop_column("traces", "thread_id")
    op.drop_column("traces", "feature")
