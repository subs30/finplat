import enum
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.account import Account
    from app.models.user import User


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    CLOSED = "closed"


class Case(Base):
    """A tenant-scoped table: one investigation, attached to the account it
    concerns. Created by V0.4's agent (via the create_case MCP tool), not
    directly by a user.

    `thread_id` is the link to the agent's full findings/conversation
    trace — not duplicated here as a second findings table. The complete
    tool-call history lives in LangGraph's own Postgres checkpoint tables
    (langgraph-checkpoint-postgres, pointed at this same database), keyed
    by thread_id; `findings_summary` below is only the human-readable
    rollup, refreshed by the agent's update_case tool as it works.

    `pending_approval_action` is deliberately just one nullable text field
    for now — V0.4's request_human_approval tool is an explicit stub (see
    app/agent/tools.py) that only records this field. The real
    human-in-the-loop approval workflow (a decision record, an
    approve/reject endpoint, notification) is V0.5's job; this column
    exists so that stub has somewhere to write without a schema change
    later forcing a rework of this table.
    """

    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False, index=True
    )
    opened_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    status: Mapped[CaseStatus] = mapped_column(
        SQLEnum(CaseStatus, name="case_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=CaseStatus.OPEN,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    findings_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    pending_approval_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    account: Mapped["Account"] = relationship()
    opened_by: Mapped["User | None"] = relationship()
