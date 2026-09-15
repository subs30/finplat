from sqlalchemy.orm import joinedload

from app.models.document_chunk import DocumentChunk
from app.repositories.base import TenantScopedRepository


class DocumentChunkRepository(TenantScopedRepository[DocumentChunk]):
    model = DocumentChunk

    def bulk_create(self, chunks: list[DocumentChunk]) -> None:
        self.db.add_all(chunks)
        self.db.flush()

    def search_similar(self, query_embedding: list[float], *, k: int) -> list[DocumentChunk]:
        """Top-k nearest chunks to `query_embedding`, by cosine distance.

        Tenant isolation: `self._scoped()` (inherited from
        TenantScopedRepository) filters on this repository's organization_id
        BEFORE the similarity ordering/limit is applied — so a query can
        never rank, and therefore never return, another organization's
        chunks, no matter how similar their embeddings are. This is the
        single most important correctness property in the RAG pipeline; see
        tests/test_rag_tenant_isolation.py.
        """
        query = (
            self._scoped()
            .options(joinedload(DocumentChunk.document))
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        return list(self.db.execute(query).scalars().all())
