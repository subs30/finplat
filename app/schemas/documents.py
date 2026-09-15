import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.document import DocumentStatus, DocumentType

# A plain-text JSON body (not a multipart file upload) — consistent with the
# rest of this API's JSON style, and avoids adding a file-upload dependency
# (python-multipart) for a pipeline whose scope is "chunk + embed + retrieve
# correctly", not file-format handling. 200_000 chars (~40k words) is a
# generous but bounded cap to keep ingestion requests reasonably sized.
_MAX_CONTENT_LENGTH = 200_000


class DocumentUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=_MAX_CONTENT_LENGTH)
    doc_type: DocumentType | None = None


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    doc_type: DocumentType | None
    status: DocumentStatus
    error: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
