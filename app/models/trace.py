import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TraceFeature(str, enum.Enum):
    """Which part of the app produced this trace. A plain Python enum, not
    a DB-level constraint (same reasoning as provider/model already being
    plain strings below): this is a display/aggregation label, not an
    enforced business rule, and new features will be added over time
    without wanting an ALTER TYPE migration each time.

    Deliberately no DETECTION value — rules/ML/graph detection never calls
    an LLM, so it never produces a trace at all; forcing one in would
    misrepresent it as an AI cost when it isn't.
    """

    AI_ASK = "ai_ask"
    RAG_ASK = "rag_ask"
    INVESTIGATION_AGENT = "investigation_agent"


class Trace(Base):
    """A tenant-scoped table: every AI gateway call is logged here, scoped
    to the caller's organization_id (same tenant isolation rule as every
    other table — see app/repositories/base.py).

    `feature` and `thread_id` are V0.7 additions — nullable, so existing
    rows from V0.1-V0.6 (written before either existed) stay valid without
    a backfill. `thread_id` is only populated for INVESTIGATION_AGENT
    traces (see app/agent/investigation.py) — it's the same LangGraph
    checkpoint thread_id Case.thread_id already uses, letting every trace
    from one investigation (its fresh start, plus one more per
    human-approval resume) be grouped into a single cost/latency rollup
    even though the wall-clock time between them may span a real human's
    review, not processing time — see the V0.7 design report for why that
    doesn't need special-casing here.
    """

    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_summary: Mapped[str] = mapped_column(String(500), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    feature: Mapped[str | None] = mapped_column(String(50), nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
