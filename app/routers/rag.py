from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user, validate_body_before_gateway
from app.gateway.base import GatewayResponse, ModelProvider
from app.gateway.dependency import get_gateway
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.rag.dependency import get_embedding_provider
from app.rag.embeddings import EmbeddingProvider
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.trace import TraceRepository
from app.schemas.rag import Citation, RagAskRequest, RagAskResponse

router = APIRouter(prefix="/rag", tags=["rag"])

# 3-5 is the usual sweet spot for a simple top-k RAG setup: enough chunks to
# cover a question whose answer spans more than one passage, without
# diluting the prompt with marginally-relevant context. No reranking —
# retrieval order IS final order.
_TOP_K = 4
_PROMPT_SUMMARY_LENGTH = 200


def _summarize(text: str) -> str:
    if len(text) <= _PROMPT_SUMMARY_LENGTH:
        return text
    return text[:_PROMPT_SUMMARY_LENGTH] + "…"


def _build_prompt(question: str, chunks: list[DocumentChunk]) -> str:
    # Citations come from what retrieval actually returned, not from parsing
    # the model's output — so the model's only job is answering from the
    # given context, never inventing which source it used.
    if not chunks:
        return (
            "No relevant context was found in the knowledge base for this "
            "question. Say you don't have enough information to answer it — "
            "do not guess or use outside knowledge.\n\n"
            f"Question: {question}"
        )

    context = "\n\n".join(
        f"[{index + 1}] (from {chunk.document.filename}): {chunk.content}"
        for index, chunk in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say you don't have enough information "
        "to answer it — do not guess or use outside knowledge.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


@router.post("/ask", response_model=RagAskResponse)
def rag_ask(
    # See validate_body_before_gateway's docstring: this must be its own
    # Depends(), listed first, so a malformed body 422s before
    # Depends(get_gateway) below (which 503s when GROQ_API_KEY is unset).
    payload: RagAskRequest = Depends(validate_body_before_gateway(RagAskRequest)),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    gateway: ModelProvider = Depends(get_gateway),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> RagAskResponse:
    query_embedding = embedding_provider.embed([payload.question])[0]

    # Tenant isolation happens here: DocumentChunkRepository is constructed
    # scoped to the caller's organization_id, and search_similar() filters on
    # it before ranking/limiting — see that method's docstring and
    # tests/test_rag_tenant_isolation.py.
    chunk_repo = DocumentChunkRepository(db, current_user.organization_id)
    chunks = chunk_repo.search_similar(query_embedding, k=_TOP_K)

    prompt = _build_prompt(payload.question, chunks)
    result: GatewayResponse[BaseModel] = gateway.generate(prompt)

    traces = TraceRepository(db, current_user.organization_id)
    trace = traces.create(
        user_id=current_user.id,
        provider=result.provider,
        model=result.model,
        prompt_summary=_summarize(payload.question),
        success=result.success,
        error=result.error,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
    db.commit()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.error or "Model provider call failed",
        )

    citations = [
        Citation(
            document_id=chunk.document_id,
            filename=chunk.document.filename,
            doc_type=chunk.document.doc_type,
            chunk_index=chunk.chunk_index,
        )
        for chunk in chunks
    ]
    return RagAskResponse(answer=result.text or "", citations=citations, trace_id=trace.id)
