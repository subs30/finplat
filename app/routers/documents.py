from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.rag.dependency import get_embedding_provider
from app.rag.embeddings import EmbeddingProvider
from app.rag.ingestion import ingest_document
from app.repositories.document import DocumentRepository
from app.schemas.documents import DocumentResponse, DocumentUploadRequest

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    payload: DocumentUploadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentResponse:
    # ingest_document never raises — a pipeline failure comes back as a
    # `status=failed` document (with `error` set), not an HTTP error, since
    # the resource (the document row) really was created. See its docstring.
    document = ingest_document(
        db,
        organization_id=current_user.organization_id,
        uploaded_by_user_id=current_user.id,
        filename=payload.filename,
        content=payload.content,
        embedding_provider=embedding_provider,
        doc_type=payload.doc_type,
    )
    db.commit()
    db.refresh(document)
    return DocumentResponse.model_validate(document)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DocumentResponse]:
    repo = DocumentRepository(db, current_user.organization_id)
    return [DocumentResponse.model_validate(document) for document in repo.list()]
