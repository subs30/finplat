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
    from app.models.account import Account


class EntityType(str, enum.Enum):
    INDIVIDUAL = "individual"
    BUSINESS = "business"


class RiskRating(str, enum.Enum):
    """Same three tiers as the V0.2 RAG corpus's Customer Risk Rating
    Policy document, so detection findings and policy citations speak a
    consistent vocabulary.
    """

    STANDARD = "standard"
    ELEVATED = "elevated"
    HIGH_RISK = "high_risk"


class KycStatus(str, enum.Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class Entity(Base):
    """A tenant-scoped table: a real-world person or company — the subject
    of investigation, not an app login (see app.models.user.User for that).

    phone/address are deliberately plain, unvalidated strings: for
    synthetic fraud-pattern generation (mule networks, shell companies)
    what matters is that two Entity rows can share the *same* value, not
    that the value is well-formed.
    """

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    entity_type: Mapped[EntityType] = mapped_column(
        SQLEnum(EntityType, name="entity_type", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    legal_name: Mapped[str] = mapped_column(String(255), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    registration_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    address_line: Mapped[str | None] = mapped_column(String(255), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    risk_rating: Mapped[RiskRating] = mapped_column(
        SQLEnum(RiskRating, name="risk_rating", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=RiskRating.STANDARD,
    )
    kyc_status: Mapped[KycStatus] = mapped_column(
        SQLEnum(KycStatus, name="kyc_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=KycStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    accounts: Mapped[list["Account"]] = relationship(back_populates="entity")
