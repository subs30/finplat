import uuid

from app.gateway.base import GatewayResponse
from app.gateway.dependency import get_gateway
from app.main import app
from app.models.document_chunk import EMBEDDING_DIMENSION
from app.models.trace import Trace
from app.rag.dependency import get_embedding_provider


class _FixedEmbeddingProvider:
    """Every text maps to the same fixed vector — deterministic similarity
    ranking without depending on any real semantic behavior.
    """

    dimension = EMBEDDING_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]


class _FakeRagGateway:
    name = "groq"

    def generate(self, prompt, *, response_schema=None):
        return GatewayResponse(
            success=True,
            provider="groq",
            model="openai/gpt-oss-20b",
            latency_ms=10,
            text="Employees get 20 days of PTO per year.",
        )


def _register(client, unique_email):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": "RAG Co", "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def _upload_document(client, headers, filename, content, doc_type=None):
    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        payload = {"filename": filename, "content": content}
        if doc_type is not None:
            payload["doc_type"] = doc_type
        response = client.post("/documents", json=payload, headers=headers)
    finally:
        del app.dependency_overrides[get_embedding_provider]
    assert response.status_code == 201
    assert response.json()["status"] == "ready"
    return response.json()


def test_rag_ask_without_token_returns_401(client):
    response = client.post("/rag/ask", json={"question": "hi"})
    assert response.status_code == 401


def test_rag_ask_returns_answer_and_citations(client, db_session, unique_email):
    headers, _ = _register(client, unique_email)
    document = _upload_document(
        client, headers, "policy.txt", "Employees get 20 days of PTO per year.", doc_type="policy"
    )

    app.dependency_overrides[get_gateway] = lambda: _FakeRagGateway()
    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        response = client.post(
            "/rag/ask", json={"question": "How much PTO do employees get?"}, headers=headers
        )
    finally:
        del app.dependency_overrides[get_gateway]
        del app.dependency_overrides[get_embedding_provider]

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Employees get 20 days of PTO per year."
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["document_id"] == document["id"]
    assert body["citations"][0]["filename"] == "policy.txt"
    assert body["citations"][0]["doc_type"] == "policy"
    assert "trace_id" in body

    # V0.7: distinguishes this from a plain /ai/ask trace, which is
    # otherwise identical in shape (same provider/model) — see
    # TraceFeature.
    trace = db_session.get(Trace, uuid.UUID(body["trace_id"]))
    assert trace is not None
    assert trace.feature == "rag_ask"


def test_rag_ask_with_no_documents_returns_empty_citations(client, unique_email):
    headers, _ = _register(client, unique_email)

    app.dependency_overrides[get_gateway] = lambda: _FakeRagGateway()
    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        response = client.post(
            "/rag/ask", json={"question": "How much PTO do employees get?"}, headers=headers
        )
    finally:
        del app.dependency_overrides[get_gateway]
        del app.dependency_overrides[get_embedding_provider]

    assert response.status_code == 200
    assert response.json()["citations"] == []


def test_rag_ask_with_failing_gateway_returns_502(client, unique_email):
    headers, _ = _register(client, unique_email)

    class _FailingGateway:
        name = "groq"

        def generate(self, prompt, *, response_schema=None):
            return GatewayResponse(
                success=False,
                provider="groq",
                model="openai/gpt-oss-20b",
                latency_ms=5,
                error="simulated failure",
            )

    app.dependency_overrides[get_gateway] = lambda: _FailingGateway()
    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        response = client.post("/rag/ask", json={"question": "hi"}, headers=headers)
    finally:
        del app.dependency_overrides[get_gateway]
        del app.dependency_overrides[get_embedding_provider]

    assert response.status_code == 502
