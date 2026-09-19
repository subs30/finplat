"""Unit tests for TraceRepository's aggregation queries (V0.7, forked
from aiplat's Step 10) — seed trace rows with known values directly, and
check the SQL GROUP BY/aggregate results are exactly what those values
imply, at the repository layer (no HTTP, no router, no real model call).
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.trace import Trace
from app.repositories.trace import TraceRepository


def _make_trace(
    db_session,
    organization_id,
    *,
    provider: str = "groq",
    model: str = "openai/gpt-oss-20b",
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    success: bool,
    created_at: datetime,
    feature: str | None = None,
    thread_id: str | None = None,
) -> Trace:
    trace = Trace(
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
    db_session.add(trace)
    db_session.flush()
    return trace


def _register(client, unique_email, org_name):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": email, "password": "supersecret1"},
    )
    return resp.json()["user"]["organization_id"]


def test_usage_overview_aggregates_known_values(client, db_session, unique_email):
    org_id = _register(client, unique_email, "Usage Overview Org")
    now = datetime.now(UTC)

    _make_trace(
        db_session, org_id,
        input_tokens=100, output_tokens=50, latency_ms=200, success=True, created_at=now,
    )
    _make_trace(
        db_session, org_id,
        input_tokens=300, output_tokens=150, latency_ms=400, success=True, created_at=now,
    )
    _make_trace(
        db_session, org_id,
        input_tokens=10, output_tokens=5, latency_ms=600, success=False, created_at=now,
    )
    db_session.commit()

    traces = TraceRepository(db_session, organization_id=org_id)
    overview = traces.usage_overview(now - timedelta(days=1), now + timedelta(days=1))

    assert overview.total_calls == 3
    assert overview.total_input_tokens == 410
    assert overview.total_output_tokens == 205
    assert float(overview.average_latency_ms) == (200 + 400 + 600) / 3
    assert overview.success_count == 2


def test_usage_overview_with_no_matching_traces_is_all_zero(client, db_session, unique_email):
    org_id = _register(client, unique_email, "Empty Usage Org")
    now = datetime.now(UTC)

    traces = TraceRepository(db_session, organization_id=org_id)
    overview = traces.usage_overview(now - timedelta(days=1), now + timedelta(days=1))

    assert overview.total_calls == 0
    assert overview.total_input_tokens == 0
    assert overview.total_output_tokens == 0
    assert float(overview.average_latency_ms) == 0
    assert overview.success_count == 0


def test_usage_breakdown_groups_by_provider_and_model(client, db_session, unique_email):
    org_id = _register(client, unique_email, "Usage Breakdown Org")
    now = datetime.now(UTC)

    _make_trace(
        db_session, org_id, provider="groq", model="openai/gpt-oss-20b",
        input_tokens=100, output_tokens=50, latency_ms=100, success=True, created_at=now,
    )
    _make_trace(
        db_session, org_id, provider="groq", model="openai/gpt-oss-20b",
        input_tokens=100, output_tokens=50, latency_ms=300, success=True, created_at=now,
    )
    # A different provider/model — proving the aggregation groups by both
    # fields together, not just provider.
    _make_trace(
        db_session, org_id, provider="gemini", model="gemini-flash-lite-latest",
        input_tokens=20, output_tokens=10, latency_ms=50, success=False, created_at=now,
    )
    db_session.commit()

    traces = TraceRepository(db_session, organization_id=org_id)
    rows = {
        (row.provider, row.model): row
        for row in traces.usage_breakdown(now - timedelta(days=1), now + timedelta(days=1))
    }

    assert len(rows) == 2
    groq_row = rows[("groq", "openai/gpt-oss-20b")]
    assert groq_row.calls == 2
    assert groq_row.input_tokens == 200
    assert groq_row.output_tokens == 100
    assert float(groq_row.average_latency_ms) == 200
    assert groq_row.success_count == 2

    gemini_row = rows[("gemini", "gemini-flash-lite-latest")]
    assert gemini_row.calls == 1
    assert gemini_row.success_count == 0


def test_usage_breakdown_by_feature_groups_by_feature(client, db_session, unique_email):
    org_id = _register(client, unique_email, "Feature Breakdown Org")
    now = datetime.now(UTC)

    _make_trace(
        db_session, org_id, feature="rag_ask",
        input_tokens=50, output_tokens=20, latency_ms=100, success=True, created_at=now,
    )
    _make_trace(
        db_session, org_id, feature="investigation_agent",
        input_tokens=500, output_tokens=80, latency_ms=3000, success=True, created_at=now,
    )
    # No feature set — as every trace written before V0.7 has — groups
    # into its own row rather than being dropped or crashing.
    _make_trace(
        db_session, org_id, feature=None,
        input_tokens=10, output_tokens=5, latency_ms=50, success=True, created_at=now,
    )
    db_session.commit()

    traces = TraceRepository(db_session, organization_id=org_id)
    rows = {
        row.feature: row
        for row in traces.usage_breakdown_by_feature(now - timedelta(days=1), now + timedelta(days=1))
    }

    assert set(rows) == {"rag_ask", "investigation_agent", None}
    assert rows["investigation_agent"].input_tokens == 500
    assert rows["rag_ask"].output_tokens == 20
    assert rows[None].calls == 1


def test_usage_by_thread_sums_across_multiple_calls_ignoring_other_threads(client, db_session, unique_email):
    org_id = _register(client, unique_email, "Thread Usage Org")
    now = datetime.now(UTC)
    thread_id = str(uuid.uuid4())

    # Fresh-start call.
    _make_trace(
        db_session, org_id, feature="investigation_agent", thread_id=thread_id,
        input_tokens=300, output_tokens=60, latency_ms=4000, success=True, created_at=now,
    )
    # Resumed call, hours later — the wall-clock gap between them (human
    # review time) is never part of either row's latency_ms, but both
    # still roll up together by thread_id.
    _make_trace(
        db_session, org_id, feature="investigation_agent", thread_id=thread_id,
        input_tokens=60, output_tokens=15, latency_ms=1500, success=True,
        created_at=now + timedelta(hours=3),
    )
    # A different investigation entirely — must not leak into the rollup.
    _make_trace(
        db_session, org_id, feature="investigation_agent", thread_id=str(uuid.uuid4()),
        input_tokens=9999, output_tokens=9999, latency_ms=9999, success=True, created_at=now,
    )
    db_session.commit()

    traces = TraceRepository(db_session, organization_id=org_id)
    row = traces.usage_by_thread(thread_id)

    assert row.total_calls == 2
    assert row.total_input_tokens == 360
    assert row.total_output_tokens == 75
    assert row.total_latency_ms == 5500
    assert row.success_count == 2


def test_usage_by_thread_is_tenant_isolated(client, db_session, unique_email):
    org_a = _register(client, unique_email, "Thread Isolation Org A")
    org_b = _register(client, unique_email, "Thread Isolation Org B")
    now = datetime.now(UTC)
    thread_id = str(uuid.uuid4())

    _make_trace(
        db_session, org_a, feature="investigation_agent", thread_id=thread_id,
        input_tokens=100, output_tokens=50, latency_ms=1000, success=True, created_at=now,
    )
    db_session.commit()

    other_org_view = TraceRepository(db_session, organization_id=org_b).usage_by_thread(thread_id)

    assert other_org_view.total_calls == 0
    assert other_org_view.total_input_tokens == 0
