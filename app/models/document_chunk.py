import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.document import Document

# sentence-transformers/all-MiniLM-L6-v2 (see app/rag/embeddings.py) outputs
# 384-dimensional vectors. This is the hard schema constraint the migration's
# `embedding` column is fixed to — pgvector requires a fixed dimension per
# column, so swapping embedding models later means a new migration (and
# re-embedding every existing chunk), not just a config change.
EMBEDDING_DIMENSION = 384


class DocumentChunk(Base):
    """A tenant-scoped table: one chunk of a Document, with its embedding.

    organization_id is denormalized here (also present on the parent
    Document) so retrieval's similarity search can filter by tenant with a
    plain WHERE clause on this table alone — no join required — keeping the
    same "every tenant-scoped table carries organization_id" rule from
    app/repositories/base.py. This is the load-bearing tenant-isolation
    guarantee for RAG retrieval: see app/repositories/document_chunk.py.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
