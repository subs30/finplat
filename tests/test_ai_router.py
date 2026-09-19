import uuid

from app.gateway.base import GatewayResponse
from app.gateway.dependency import get_gateway
from app.main import app
from app.models.trace import Trace


class _FakeAskProvider:
    name = "groq"

    def generate(self, prompt, *, response_schema=None):
        return GatewayResponse(
            success=True,
            provider="groq",
            model="openai/gpt-oss-20b",
            latency_ms=42,
            text="42 is the answer.",
        )


class _FakeFailingProvider:
    name = "groq"

    def generate(self, prompt, *, response_schema=None):
        return GatewayResponse(
            success=False,
            provider="groq",
            model="openai/gpt-oss-20b",
            latency_ms=5,
            error="simulated failure",
        )


def _register(client, unique_email):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": "AI Co", "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def test_ask_without_token_returns_401(client):
    response = client.post("/ai/ask", json={"question": "hi"})
    assert response.status_code == 401


def test_ask_returns_answer_and_writes_trace(client, db_session, unique_email):
    headers, user = _register(client, unique_email)
    app.dependency_overrides[get_gateway] = lambda: _FakeAskProvider()
    try:
        response = client.post("/ai/ask", json={"question": "What is 6*7?"}, headers=headers)
    finally:
        del app.dependency_overrides[get_gateway]

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "42 is the answer."

    trace = db_session.get(Trace, uuid.UUID(body["trace_id"]))
    assert trace is not None
    assert trace.success is True
    assert trace.provider == "groq"
    assert str(trace.organization_id) == user["organization_id"]
    # V0.7: distinguishes this from a /rag/ask trace, which is otherwise
    # identical in shape (same provider/model) — see TraceFeature.
    assert trace.feature == "ai_ask"


def test_ask_with_failing_gateway_returns_502_and_still_traces_failure(client, db_session, unique_email):
    headers, user = _register(client, unique_email)
    app.dependency_overrides[get_gateway] = lambda: _FakeFailingProvider()
    try:
        response = client.post("/ai/ask", json={"question": "hi"}, headers=headers)
    finally:
        del app.dependency_overrides[get_gateway]

    assert response.status_code == 502

    traces = (
        db_session.query(Trace)
        .filter_by(organization_id=uuid.UUID(user["organization_id"]), success=False)
        .all()
    )
    assert any(t.error == "simulated failure" for t in traces)


def test_gateway_not_configured_returns_503(client, unique_email):
    # No override installed: hits the real get_gateway dependency, which
    # requires GROQ_API_KEY (the active provider). The test environment
    # doesn't set one, so this exercises the "missing key" failure path.
    headers, _ = _register(client, unique_email)
    from app.config import get_settings

    if get_settings().GROQ_API_KEY:
        return  # skip: a real key is configured in this environment
    response = client.post("/ai/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 503
