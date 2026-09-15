import uuid

from app.models.document import Document, DocumentStatus, DocumentType
from app.repositories.base import TenantScopedRepository


class DocumentRepository(TenantScopedRepository[Document]):
    model = Document

    def create_pending(
        self,
        *,
        filename: str,
        uploaded_by_user_id: uuid.UUID | None,
        doc_type: DocumentType | None = None,
    ) -> Document:
        # organization_id is stamped from the repository's own scope, not a
        # caller-supplied argument — same rule as TraceRepository.create.
        document = Document(
            organization_id=self.organization_id,
            uploaded_by_user_id=uploaded_by_user_id,
            filename=filename,
            doc_type=doc_type,
            status=DocumentStatus.PENDING,
        )
        self.db.add(document)
        self.db.flush()
        return document

    def mark_processing(self, document: Document) -> None:
        document.status = DocumentStatus.PROCESSING
        self.db.flush()

    def mark_ready(self, document: Document) -> None:
        document.status = DocumentStatus.READY
        document.error = None
        self.db.flush()

    def mark_failed(self, document: Document, error: str) -> None:
        document.status = DocumentStatus.FAILED
        document.error = error
        self.db.flush()

    def delete(self, document: Document) -> None:
        # ORM-level delete (not a bulk DELETE statement) so SQLAlchemy's
        # unit-of-work honors Document.chunks' cascade="all, delete-orphan"
        # and removes every chunk along with the document.
        self.db.delete(document)
        self.db.flush()
