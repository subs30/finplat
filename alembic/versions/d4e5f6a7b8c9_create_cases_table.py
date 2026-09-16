"""create cases table

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-09-17 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: str | Sequence[str] | None = 'c3d4e5f6a7b8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    case_status = postgresql.ENUM("open", "in_review", "closed", name="case_status", create_type=False)
    case_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column(
            "account_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("accounts.id"), nullable=False
        ),
        sa.Column(
            "opened_by_user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column("status", case_status, nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("findings_summary", sa.Text(), nullable=True),
        sa.Column("pending_approval_action", sa.Text(), nullable=True),
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cases_organization_id", "cases", ["organization_id"])
    op.create_index("ix_cases_account_id", "cases", ["account_id"])
    op.create_index("ix_cases_thread_id", "cases", ["thread_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_cases_thread_id", table_name="cases")
    op.drop_index("ix_cases_account_id", table_name="cases")
    op.drop_index("ix_cases_organization_id", table_name="cases")
    op.drop_table("cases")

    postgresql.ENUM(name="case_status").drop(op.get_bind(), checkfirst=True)
