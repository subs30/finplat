"""create case_approvals table

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-18 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: str | Sequence[str] | None = 'd4e5f6a7b8c9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    case_approval_status = postgresql.ENUM(
        "pending", "approved", "rejected", name="case_approval_status", create_type=False
    )
    case_approval_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "case_approvals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("action_description", sa.Text(), nullable=False),
        sa.Column("status", case_approval_status, nullable=False),
        sa.Column(
            "decided_by_user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_case_approvals_organization_id", "case_approvals", ["organization_id"])
    op.create_index("ix_case_approvals_case_id", "case_approvals", ["case_id"])
    op.create_index("ix_case_approvals_thread_id", "case_approvals", ["thread_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_case_approvals_thread_id", table_name="case_approvals")
    op.drop_index("ix_case_approvals_case_id", table_name="case_approvals")
    op.drop_index("ix_case_approvals_organization_id", table_name="case_approvals")
    op.drop_table("case_approvals")

    postgresql.ENUM(name="case_approval_status").drop(op.get_bind(), checkfirst=True)
