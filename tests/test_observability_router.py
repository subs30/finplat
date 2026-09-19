"""Integration tests for GET /observability/usage and
GET /observability/investigations/{thread_id}/usage (V0.7) — real HTTP
stack, real Postgres, no mocked provider/gateway calls needed since these
endpoints only read already-recorded Trace rows.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.trace import Trace


def _register(client, unique_email, org_name):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def _seed_trace(
    db_session,
    organization_id,
    *,
    provider: str = "groq",
    model: str = "openai/gpt-oss-20b",
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: int = 200,
    success: bool = True,
    created_at: datetime,
    feature: str | None = None,
    thread_id: str | None = None,
) -> None:
    db_session.add(
        Trace(
            id=uuid.uuid4(),
            organization_id=organization_id,
            user_id=None,
            provider=provider,
            model=model,
            prompt_summary="test",
            success=success,
            error=None if success else "boom",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            created_at=created_at,
            feature=feature,
            thread_id=thread_id,
        )
    )
    db_session.commit()


def test_usage_requires_auth(client):
    resp = client.get("/observability/usage")
    assert resp.status_code == 401


def test_usage_endpoint_returns_correct_aggregates(client, db_session, unique_email):
    headers, user = _register(client, unique_email, "Usage Endpoint Org")
    now = datetime.now(UTC)

    _seed_trace(
        db_session, user["organization_id"], feature="rag_ask",
        input_tokens=100, output_tokens=50, latency_ms=200, success=True, created_at=now,
    )
    _seed_trace(
        db_session, user["organization_id"], feature="investigation_agent", thread_id=str(uuid.uuid4()),
        input_tokens=300, output_tokens=150, latency_ms=400, success=True, created_at=now,
    )

    resp = client.get("/observability/usage", headers=headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_calls"] == 2
    assert body["total_input_tokens"] == 400
    assert body["total_output_tokens"] == 200
    assert body["estimated_cost_usd"] == round(600 / 1000 * 0.05, 6)
    assert body["success_rate"] == 100.0

    feature_names = {row["feature"] for row in body["breakdown_by_feature"]}
    assert feature_names == {"rag_ask", "investigation_agent"}


def test_usage_endpoint_is_tenant_isolated(client, db_session, unique_email):
    headers_a, user_a = _register(client, unique_email, "Usage Isolation Org A")
    headers_b, _ = _register(client, unique_email, "Usage Isolation Org B")
    now = datetime.now(UTC)

    _seed_trace(db_session, user_a["organization_id"], created_at=now)

    resp_a = client.get("/observability/usage", headers=headers_a)
    resp_b = client.get("/observability/usage", headers=headers_b)

    assert resp_a.json()["total_calls"] == 1
    assert resp_b.json()["total_calls"] == 0


def test_usage_endpoint_respects_date_range(client, db_session, unique_email):
    headers, user = _register(client, unique_email, "Usage Date Range Org")
    now = datetime.now(UTC)

    _seed_trace(db_session, user["organization_id"], created_at=now - timedelta(days=60))
    _seed_trace(db_session, user["organization_id"], created_at=now)

    resp = client.get("/observability/usage", headers=headers)
    assert resp.json()["total_calls"] == 1  # default 30-day window excludes the 60-day-old trace

    resp = client.get(
        "/observability/usage",
        params={"start_date": (now - timedelta(days=90)).date().isoformat()},
        headers=headers,
    )
    assert resp.json()["total_calls"] == 2


def test_usage_endpoint_rejects_start_after_end(client, unique_email):
    headers, _ = _register(client, unique_email, "Usage Bad Range Org")

    resp = client.get(
        "/observability/usage",
        params={"start_date": "2026-01-10", "end_date": "2026-01-01"},
        headers=headers,
    )
    assert resp.status_code == 422


def test_investigation_usage_requires_auth(client):
    resp = client.get(f"/observability/investigations/{uuid.uuid4()}/usage")
    assert resp.status_code == 401


def test_investigation_usage_rolls_up_multiple_calls_for_one_thread(client, db_session, unique_email):
    headers, user = _register(client, unique_email, "Investigation Usage Org")
    now = datetime.now(UTC)
    thread_id = str(uuid.uuid4())

    _seed_trace(
        db_session, user["organization_id"], feature="investigation_agent", thread_id=thread_id,
        input_tokens=300, output_tokens=60, latency_ms=4000, created_at=now,
    )
    _seed_trace(
        db_session, user["organization_id"], feature="investigation_agent", thread_id=thread_id,
        input_tokens=60, output_tokens=15, latency_ms=1500, created_at=now + timedelta(hours=3),
    )
    # A different investigation's trace must not leak into this rollup.
    _seed_trace(
        db_session, user["organization_id"], feature="investigation_agent", thread_id=str(uuid.uuid4()),
        input_tokens=9999, output_tokens=9999, latency_ms=9999, created_at=now,
    )

    resp = client.get(f"/observability/investigations/{thread_id}/usage", headers=headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["thread_id"] == thread_id
    assert body["total_calls"] == 2
    assert body["total_input_tokens"] == 360
    assert body["total_output_tokens"] == 75
    assert body["total_latency_ms"] == 5500
    assert body["success_rate"] == 100.0


def test_investigation_usage_for_unknown_thread_is_all_zero_not_an_error(client, unique_email):
    headers, _ = _register(client, unique_email, "Investigation Usage Unknown Org")

    resp = client.get(f"/observability/investigations/{uuid.uuid4()}/usage", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["total_calls"] == 0


def test_investigation_usage_is_tenant_isolated(client, db_session, unique_email):
    headers_a, user_a = _register(client, unique_email, "Investigation Isolation Org A")
    headers_b, _ = _register(client, unique_email, "Investigation Isolation Org B")
    now = datetime.now(UTC)
    thread_id = str(uuid.uuid4())

    _seed_trace(
        db_session, user_a["organization_id"], feature="investigation_agent", thread_id=thread_id,
        input_tokens=100, output_tokens=50, latency_ms=1000, created_at=now,
    )

    resp_a = client.get(f"/observability/investigations/{thread_id}/usage", headers=headers_a)
    resp_b = client.get(f"/observability/investigations/{thread_id}/usage", headers=headers_b)

    assert resp_a.json()["total_calls"] == 1
    assert resp_b.json()["total_calls"] == 0
