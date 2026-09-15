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
    from app.models.document_chunk import DocumentChunk


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentType(str, enum.Enum):
    """Category tag for retrieval/citations to distinguish corpus documents
    by kind. Nullable: an arbitrary ad hoc upload (not part of a tagged
    corpus) has no category and that's fine — this is metadata for
    filtering/display, not a required classification.
    """

    POLICY = "policy"
    TYPOLOGY = "typology"
    CASE_WRITEUP = "case_writeup"


class Document(Base):
    """A tenant-scoped table: an uploaded source document for RAG. Chunking
    and embedding happen synchronously in the request that creates this row
    (see app/rag/ingestion.py) — `status` tracks that pipeline so a client
    can tell a still-processing (or failed) document apart from one whose
    chunks are actually searchable yet.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    uploaded_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[DocumentType | None] = mapped_column(
        SQLEnum(
            DocumentType,
            name="document_type",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=True,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        # values_callable: store the enum's lowercase .value, matching the
        # Postgres "document_status" type created in the migration — see the
        # identical pattern (and reasoning) for User.role / "user_role".
        SQLEnum(
            DocumentStatus,
            name="document_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=DocumentStatus.PENDING,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
