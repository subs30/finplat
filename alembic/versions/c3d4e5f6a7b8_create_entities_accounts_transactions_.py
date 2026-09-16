"""create entities, accounts, transactions tables

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-16 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: str | Sequence[str] | None = 'b2c3d4e5f6a7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    entity_type = postgresql.ENUM("individual", "business", name="entity_type", create_type=False)
    entity_type.create(op.get_bind(), checkfirst=True)

    risk_rating = postgresql.ENUM(
        "standard", "elevated", "high_risk", name="risk_rating", create_type=False
    )
    risk_rating.create(op.get_bind(), checkfirst=True)

    kyc_status = postgresql.ENUM(
        "pending", "verified", "rejected", name="kyc_status", create_type=False
    )
    kyc_status.create(op.get_bind(), checkfirst=True)

    account_type = postgresql.ENUM(
        "checking", "savings", "business", name="account_type", create_type=False
    )
    account_type.create(op.get_bind(), checkfirst=True)

    account_status = postgresql.ENUM(
        "active", "frozen", "closed", name="account_status", create_type=False
    )
    account_status.create(op.get_bind(), checkfirst=True)

    transaction_type = postgresql.ENUM(
        "deposit", "withdrawal", "transfer", "wire", name="transaction_type", create_type=False
    )
    transaction_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column("entity_type", entity_type, nullable=False),
        sa.Column("legal_name", sa.String(length=255), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("registration_number", sa.String(length=100), nullable=True),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("address_line", sa.String(length=255), nullable=True),
        sa.Column("city", sa.String(length=100), nullable=True),
        sa.Column("country", sa.String(length=100), nullable=True),
        sa.Column("risk_rating", risk_rating, nullable=False),
        sa.Column("kyc_status", kyc_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_entities_organization_id", "entities", ["organization_id"])

    op.create_table(
        "accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column(
            "entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id"), nullable=False
        ),
        sa.Column("account_number", sa.String(length=50), nullable=False),
        sa.Column("account_type", account_type, nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("status", account_status, nullable=False),
        sa.Column("open_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_accounts_organization_id", "accounts", ["organization_id"])
    op.create_index("ix_accounts_entity_id", "accounts", ["entity_id"])

    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column(
            "sender_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id"),
            nullable=True,
        ),
        sa.Column(
            "receiver_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id"),
            nullable=True,
        ),
        sa.Column("amount", sa.Numeric(18, 2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("transaction_type", transaction_type, nullable=False),
        sa.Column("channel", sa.String(length=50), nullable=True),
        sa.Column("location_city", sa.String(length=100), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_transactions_organization_id", "transactions", ["organization_id"])
    op.create_index("ix_transactions_sender_account_id", "transactions", ["sender_account_id"])
    op.create_index("ix_transactions_receiver_account_id", "transactions", ["receiver_account_id"])
    op.create_index("ix_transactions_occurred_at", "transactions", ["occurred_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_transactions_occurred_at", table_name="transactions")
    op.drop_index("ix_transactions_receiver_account_id", table_name="transactions")
    op.drop_index("ix_transactions_sender_account_id", table_name="transactions")
    op.drop_index("ix_transactions_organization_id", table_name="transactions")
    op.drop_table("transactions")

    op.drop_index("ix_accounts_entity_id", table_name="accounts")
    op.drop_index("ix_accounts_organization_id", table_name="accounts")
    op.drop_table("accounts")

    op.drop_index("ix_entities_organization_id", table_name="entities")
    op.drop_table("entities")

    postgresql.ENUM(name="transaction_type").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="account_status").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="account_type").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="kyc_status").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="risk_rating").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="entity_type").drop(op.get_bind(), checkfirst=True)
