import uuid

from pydantic import BaseModel, Field

from app.models.document import DocumentType


class RagAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    """One retrieved chunk that fed into the answer — the source document
    (by id, filename, and category) and which chunk of it was used.
    """

    document_id: uuid.UUID
    filename: str
    doc_type: DocumentType | None
    chunk_index: int


class RagAskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace_id: uuid.UUID
