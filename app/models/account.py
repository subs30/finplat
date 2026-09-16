import enum
import uuid
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, ForeignKey, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.entity import Entity


class AccountType(str, enum.Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    BUSINESS = "business"


class AccountStatus(str, enum.Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"


class Account(Base):
    """A tenant-scoped table: one bank account, owned by one Entity.

    organization_id is denormalized here (also reachable via
    entities.organization_id) for the same reason as every other
    tenant-scoped table in this project — see app/repositories/base.py.
    """

    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True
    )
    account_number: Mapped[str] = mapped_column(String(50), nullable=False)
    account_type: Mapped[AccountType] = mapped_column(
        SQLEnum(AccountType, name="account_type", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    status: Mapped[AccountStatus] = mapped_column(
        SQLEnum(AccountStatus, name="account_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=AccountStatus.ACTIVE,
    )
    open_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    entity: Mapped["Entity"] = relationship(back_populates="accounts")
