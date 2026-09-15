from app.gateway.base import GatewayResponse
from app.gateway.dependency import get_gateway
from app.main import app
from app.models.document_chunk import EMBEDDING_DIMENSION
from app.rag.dependency import get_embedding_provider
from app.repositories.document_chunk import DocumentChunkRepository


class _FixedEmbeddingProvider:
    """Every text — org A's document content AND org B's query — maps to the
    exact same vector. This is deliberate: it makes org A's chunks the
    *maximally* similar match to any query, from any organization, so the
    only thing that can possibly keep them out of org B's results is the
    organization_id filter in DocumentChunkRepository.search_similar()
    itself. If tenant isolation were broken (or merely "usually right by
    accident because embeddings differ"), this test would catch it.
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
            text="(model answer, irrelevant to this test)",
        )


def _register(client, unique_email, org_name):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def test_org_a_document_never_retrievable_via_rag_ask_as_org_b(client, unique_email):
    headers_a, user_a = _register(client, unique_email, "Org A")
    headers_b, user_b = _register(client, unique_email, "Org B")
    assert user_a["organization_id"] != user_b["organization_id"]

    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        upload = client.post(
            "/documents",
            json={
                "filename": "org-a-secret.txt",
                "content": "Org A's confidential Q3 revenue was $42 million.",
            },
            headers=headers_a,
        )
    finally:
        del app.dependency_overrides[get_embedding_provider]
    assert upload.status_code == 201
    assert upload.json()["status"] == "ready"
    org_a_document_id = upload.json()["id"]

    # Same question an org-A user might ask, asked instead as org B — with
    # every embedding identical, org A's chunk would rank #1 for org B too
    # if the organization_id filter were missing or broken.
    app.dependency_overrides[get_gateway] = lambda: _FakeRagGateway()
    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        response_b = client.post(
            "/rag/ask",
            json={"question": "Org A's confidential Q3 revenue was $42 million."},
            headers=headers_b,
        )
    finally:
        del app.dependency_overrides[get_gateway]
        del app.dependency_overrides[get_embedding_provider]

    assert response_b.status_code == 200
    body_b = response_b.json()
    assert body_b["citations"] == []
    assert all(c["document_id"] != org_a_document_id for c in body_b["citations"])

    # Sanity check: the exact same question, asked as org A, DOES retrieve
    # org A's own document — proving the empty result above is because of
    # tenant isolation, not because retrieval is broken outright.
    app.dependency_overrides[get_gateway] = lambda: _FakeRagGateway()
    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        response_a = client.post(
            "/rag/ask",
            json={"question": "Org A's confidential Q3 revenue was $42 million."},
            headers=headers_a,
        )
    finally:
        del app.dependency_overrides[get_gateway]
        del app.dependency_overrides[get_embedding_provider]

    assert response_a.status_code == 200
    body_a = response_a.json()
    assert len(body_a["citations"]) >= 1
    assert body_a["citations"][0]["document_id"] == org_a_document_id


def test_document_chunk_repository_search_similar_is_tenant_scoped(db_session, unique_email, client):
    """Same property, exercised one layer down — directly against the
    repository, bypassing the router/HTTP layer entirely.
    """
    headers_a, user_a = _register(client, unique_email, "Org C")
    _headers_b, user_b = _register(client, unique_email, "Org D")

    app.dependency_overrides[get_embedding_provider] = lambda: _FixedEmbeddingProvider()
    try:
        client.post(
            "/documents",
            json={"filename": "c-doc.txt", "content": "Some content only Org C uploaded."},
            headers=headers_a,
        )
    finally:
        del app.dependency_overrides[get_embedding_provider]

    query_vector = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)

    repo_scoped_to_b = DocumentChunkRepository(db_session, organization_id=user_b["organization_id"])
    assert repo_scoped_to_b.search_similar(query_vector, k=5) == []

    repo_scoped_to_a = DocumentChunkRepository(db_session, organization_id=user_a["organization_id"])
    assert len(repo_scoped_to_a.search_similar(query_vector, k=5)) >= 1
