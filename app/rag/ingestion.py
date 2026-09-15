import uuid

from sqlalchemy.orm import Session

from app.models.document import Document, DocumentType
from app.models.document_chunk import DocumentChunk
from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository


def ingest_document(
    db: Session,
    *,
    organization_id: uuid.UUID,
    uploaded_by_user_id: uuid.UUID | None,
    filename: str,
    content: str,
    embedding_provider: EmbeddingProvider,
    doc_type: DocumentType | None = None,
) -> Document:
    """Create, chunk, and embed a document, synchronously within one call.

    Never raises: any failure in chunking/embedding/storing chunks is
    caught and turns into `status=failed` with the exception message in
    `error` — the document row itself is never lost or left stuck at
    `processing` (same "never leave the caller guessing" contract as the
    gateway's GatewayResponse). The caller (a router, or the corpus seed
    script) is responsible for `db.commit()` — this function only flushes.
    """
    doc_repo = DocumentRepository(db, organization_id)
    document = doc_repo.create_pending(
        filename=filename, uploaded_by_user_id=uploaded_by_user_id, doc_type=doc_type
    )
    doc_repo.mark_processing(document)

    # A SAVEPOINT, not the outer transaction: if chunking/embedding/insertion
    # fails partway through, only ITS work rolls back — the document row
    # created above (now `processing`) survives so we can mark it `failed`
    # with detail, rather than losing the row or leaving it stuck.
    savepoint = db.begin_nested()
    try:
        texts = chunk_text(content)
        if not texts:
            raise ValueError("Document produced no chunks (empty or whitespace-only content)")

        vectors = embedding_provider.embed(texts)
        chunk_repo = DocumentChunkRepository(db, organization_id)
        chunk_repo.bulk_create(
            [
                DocumentChunk(
                    document_id=document.id,
                    organization_id=organization_id,
                    chunk_index=index,
                    content=chunk,
                    embedding=vector,
                )
                for index, (chunk, vector) in enumerate(zip(texts, vectors, strict=True))
            ]
        )
    except Exception as exc:  # noqa: BLE001 - see docstring: never raise, always mark failed.
        savepoint.rollback()
        doc_repo.mark_failed(document, str(exc))
    else:
        savepoint.commit()
        doc_repo.mark_ready(document)

    return document
