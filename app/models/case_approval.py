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
    from app.models.case import Case
    from app.models.user import User


class CaseApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CaseApproval(Base):
    """A tenant-scoped table: one row per human-approval request raised by
    the investigation agent's `request_human_approval` tool.

    This is the real decision record V0.4's `Case.pending_approval_action`
    stub couldn't be (it's a single nullable text field — no room for who
    decided, when, or a history of past requests). It stays deliberately
    smaller than aiplat's AgentRun/AgentStep/AgentApproval trio: the agent's
    full conversation/tool-call state at the pause point already lives in
    LangGraph's own Postgres checkpoint tables (keyed by thread_id), so this
    table only needs to record the human-decision side of the story.

    `thread_id` is denormalized from `Case.thread_id` — it's the resume key
    (`app.agent.investigation.resume_investigation_with_decision` needs it
    to rebuild the graph's config), and denormalizing it means listing or
    deciding an approval never needs a join back through `cases`.
    """

    __tablename__ = "case_approvals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False, index=True
    )
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action_description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[CaseApprovalStatus] = mapped_column(
        SQLEnum(
            CaseApprovalStatus, name="case_approval_status", values_callable=lambda e: [m.value for m in e]
        ),
        nullable=False,
        default=CaseApprovalStatus.PENDING,
    )
    decided_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    case: Mapped["Case"] = relationship()
    decided_by: Mapped["User | None"] = relationship()
