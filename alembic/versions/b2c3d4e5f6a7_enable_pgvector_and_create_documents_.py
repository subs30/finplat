"""enable pgvector and create documents/document_chunks tables

Revision ID: b2c3d4e5f6a7
Revises: a7b8c9d0e1f2
Create Date: 2026-09-15 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: str | Sequence[str] | None = 'a7b8c9d0e1f2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must match app.models.document_chunk.EMBEDDING_DIMENSION (sentence-transformers/
# all-MiniLM-L6-v2's output size) — pgvector requires a fixed dimension per column.
EMBEDDING_DIMENSION = 384


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # create_type=False: created/dropped explicitly below, same reasoning as
    # "user_role" in the V0.1 migration (avoids a double CREATE TYPE via
    # create_table/drop_table's own enum handling).
    document_status = postgresql.ENUM(
        "pending", "processing", "ready", "failed", name="document_status", create_type=False
    )
    document_status.create(op.get_bind(), checkfirst=True)

    document_type = postgresql.ENUM(
        "policy", "typology", "case_writeup", name="document_type", create_type=False
    )
    document_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column(
            "uploaded_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("doc_type", document_type, nullable=True),
        sa.Column("status", document_status, nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_organization_id", "documents", ["organization_id"])

    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id"),
            nullable=False,
        ),
        # Denormalized organization_id (also reachable via documents.organization_id)
        # so retrieval's similarity search can filter by tenant with a plain WHERE
        # on this table alone, no join required — see app/models/document_chunk.py.
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_organization_id", "document_chunks", ["organization_id"])

    # HNSW over IVFFlat: HNSW needs no "lists" parameter tuned to table size
    # and gives good recall starting from the very first rows — IVFFlat's
    # clustering is only accurate once trained on a representative amount of
    # data, so a small/fresh table gets poor recall until it grows. Tradeoff:
    # HNSW's index build time and memory use scale worse for very large
    # tables. Acceptable here — correctness from day one matters more than
    # build cost at this project's scale.
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.drop_index("ix_document_chunks_organization_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")

    op.drop_index("ix_documents_organization_id", table_name="documents")
    op.drop_table("documents")

    postgresql.ENUM(name="document_type").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="document_status").drop(op.get_bind(), checkfirst=True)

    # The "vector" extension is intentionally left installed on downgrade —
    # it's a database-wide object other schemas/objects could depend on, and
    # DROP EXTENSION is a much bigger blast radius than this migration's own
    # tables. Standard practice for extension-enabling migrations.
