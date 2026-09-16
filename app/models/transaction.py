import enum
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.account import Account


class TransactionType(str, enum.Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    WIRE = "wire"


class Transaction(Base):
    """A tenant-scoped table: one money movement. Either side of
    sender/receiver may be null — that represents money crossing the
    bank's boundary (an external deposit source or withdrawal
    destination), not a data-quality gap. See app/rag or the V0.3 data
    model report for why only transactions with BOTH sides internal
    become Neo4j graph edges: deposits/withdrawals are a rules/ML
    concern (amount + velocity), not a graph-topology one.
    """

    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    sender_account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=True, index=True
    )
    receiver_account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=True, index=True
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    transaction_type: Mapped[TransactionType] = mapped_column(
        SQLEnum(
            TransactionType, name="transaction_type", values_callable=lambda e: [m.value for m in e]
        ),
        nullable=False,
    )
    channel: Mapped[str | None] = mapped_column(String(50), nullable=True)
    location_city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    sender_account: Mapped["Account | None"] = relationship(foreign_keys=[sender_account_id])
    receiver_account: Mapped["Account | None"] = relationship(foreign_keys=[receiver_account_id])
