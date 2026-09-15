import uuid

from app.main import app
from app.models.document_chunk import EMBEDDING_DIMENSION, DocumentChunk
from app.rag.dependency import get_embedding_provider


class _FakeEmbeddingProvider:
    dimension = EMBEDDING_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]


def _register(client, unique_email):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": "Docs Co", "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def test_upload_without_token_returns_401(client):
    response = client.post("/documents", json={"filename": "a.txt", "content": "hello"})
    assert response.status_code == 401


def test_upload_document_creates_ready_document_with_chunks(client, db_session, unique_email):
    headers, user = _register(client, unique_email)
    app.dependency_overrides[get_embedding_provider] = lambda: _FakeEmbeddingProvider()
    try:
        response = client.post(
            "/documents",
            json={
                "filename": "handbook.txt",
                "content": "Our vacation policy grants 20 days of PTO per year. "
                "Remote work is allowed up to 3 days per week.",
            },
            headers=headers,
        )
    finally:
        del app.dependency_overrides[get_embedding_provider]

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["error"] is None
    assert body["filename"] == "handbook.txt"
    assert body["doc_type"] is None

    document_id = uuid.UUID(body["id"])
    chunks = (
        db_session.query(DocumentChunk)
        .filter_by(document_id=document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    assert len(chunks) >= 1
    assert all(str(c.organization_id) == user["organization_id"] for c in chunks)
    assert chunks[0].content  # non-empty


def test_upload_document_with_doc_type_is_persisted_and_returned(client, unique_email):
    headers, _ = _register(client, unique_email)
    app.dependency_overrides[get_embedding_provider] = lambda: _FakeEmbeddingProvider()
    try:
        response = client.post(
            "/documents",
            json={
                "filename": "kyc_policy.md",
                "content": "Customers must be identified before onboarding.",
                "doc_type": "policy",
            },
            headers=headers,
        )
    finally:
        del app.dependency_overrides[get_embedding_provider]

    assert response.status_code == 201
    assert response.json()["doc_type"] == "policy"


def test_upload_empty_whitespace_content_marks_document_failed(client, unique_email):
    headers, _ = _register(client, unique_email)
    app.dependency_overrides[get_embedding_provider] = lambda: _FakeEmbeddingProvider()
    try:
        response = client.post(
            "/documents",
            json={"filename": "blank.txt", "content": "   "},
            headers=headers,
        )
    finally:
        del app.dependency_overrides[get_embedding_provider]

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] is not None


def test_list_documents_returns_only_callers_org_documents(client, unique_email):
    headers_a, _ = _register(client, unique_email)
    headers_b, _ = _register(client, unique_email)

    app.dependency_overrides[get_embedding_provider] = lambda: _FakeEmbeddingProvider()
    try:
        client.post(
            "/documents", json={"filename": "org-a-doc.txt", "content": "Org A content."}, headers=headers_a
        )
    finally:
        del app.dependency_overrides[get_embedding_provider]

    response_a = client.get("/documents", headers=headers_a)
    response_b = client.get("/documents", headers=headers_b)

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert any(d["filename"] == "org-a-doc.txt" for d in response_a.json())
    assert all(d["filename"] != "org-a-doc.txt" for d in response_b.json())
