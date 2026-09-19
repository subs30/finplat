import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from app.agent.investigation import (
    _content_to_text,
    resume_investigation_with_decision,
    run_investigation,
)
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.trace import TraceFeature
from app.models.transaction import Transaction, TransactionType
from app.repositories.case_approval import CaseApprovalRepository
from app.repositories.trace import TraceRepository

# V0.7: does app.agent.investigation actually write Trace rows correctly?
# Mocked (no real Groq) — this tests the *instrumentation mechanics*
# (one trace per call, correct summing, no double-counting on resume,
# failure path), not model behavior. See evals/run_agent_evals.py and the
# V0.7 report for the real-Groq, real-endpoint verification of the same
# code path.


def _usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _tool_call(name: str, args: dict, call_id: str, usage: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}], usage_metadata=usage)


class _ScriptedTracingLLM(BaseChatModel):
    """Reacts to the message history so far (case_id isn't known until
    create_case's own ToolMessage comes back), and stamps a distinct,
    known usage_metadata on every AIMessage it returns so tests can
    assert exact summed totals.
    """

    account_id: str
    raise_on_call_number: int | None = None
    call_count: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.call_count += 1
        if self.raise_on_call_number == self.call_count:
            raise RuntimeError("simulated transient provider failure")

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        by_name = {m.name: m for m in tool_messages}

        if "create_case" not in by_name:
            response: AIMessage = _tool_call(
                "create_case",
                {"account_id": self.account_id, "title": "Trace test case", "summary": "s"},
                "c1",
                _usage(100, 20),
            )
        elif "update_case" not in by_name:
            create_case_text = _content_to_text(by_name["create_case"].content)
            assert create_case_text is not None
            case_id = json.loads(create_case_text)["case_id"]
            response = _tool_call(
                "update_case",
                {"case_id": case_id, "status": "closed", "findings_summary": "Nothing suspicious."},
                "c2",
                _usage(150, 30),
            )
        else:
            response = AIMessage(content="Investigation closed.", usage_metadata=_usage(50, 10))

        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-tracing"


class _ScriptedTracingApprovalLLM(BaseChatModel):
    """Same idea, but pauses at request_human_approval instead of closing
    directly — for the resume/double-counting test.
    """

    account_id: str

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        by_name = {m.name: m for m in tool_messages}

        if "create_case" not in by_name:
            response: AIMessage = _tool_call(
                "create_case",
                {"account_id": self.account_id, "title": "Resume trace test", "summary": "s"},
                "c1",
                _usage(100, 20),
            )
        elif "request_human_approval" not in by_name:
            create_case_text = _content_to_text(by_name["create_case"].content)
            assert create_case_text is not None
            case_id = json.loads(create_case_text)["case_id"]
            response = _tool_call(
                "request_human_approval",
                {"case_id": case_id, "action_description": "Freeze pending review."},
                "c2",
                _usage(200, 40),
            )
        else:
            response = AIMessage(content="Resumed and closed.", usage_metadata=_usage(60, 15))

        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-tracing-approval"


def _register_org(client, unique_email, org_name="Trace Writing Test Co"):
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": unique_email(), "password": "supersecret1"},
    )
    body = resp.json()["user"]
    return uuid.UUID(body["organization_id"]), uuid.UUID(body["id"])


def _make_account(db_session, organization_id: uuid.UUID) -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Trace Writing Test Subject",
        risk_rating=RiskRating.STANDARD,
        kyc_status=KycStatus.VERIFIED,
    )
    account = Account(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_id=entity.id,
        account_number=f"ACC-{uuid.uuid4().hex[:8]}",
        account_type=AccountType.CHECKING,
        currency="USD",
        status=AccountStatus.ACTIVE,
        open_date=date(2025, 1, 1),
    )
    db_session.add_all([entity, account])
    db_session.flush()
    now = datetime.now(UTC)
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=organization_id,
            sender_account_id=None,
            receiver_account_id=account.id,
            amount=Decimal("500.00"),
            currency="USD",
            transaction_type=TransactionType.DEPOSIT,
            occurred_at=now - timedelta(days=1),
        )
    )
    db_session.flush()
    return account


async def test_run_investigation_writes_one_trace_with_summed_tokens(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())
    llm = _ScriptedTracingLLM(account_id=str(account.id))

    result = await run_investigation(db_session, org_id, account.id, thread_id=thread_id, llm=llm)
    assert result.final_message == "Investigation closed."

    traces = TraceRepository(db_session, org_id).list()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.feature == TraceFeature.INVESTIGATION_AGENT.value
    assert trace.thread_id == thread_id
    assert trace.success is True
    # Three scripted turns: (100,20) + (150,30) + (50,10)
    assert trace.input_tokens == 300
    assert trace.output_tokens == 60
    assert trace.latency_ms > 0


async def test_resume_writes_a_second_trace_without_double_counting(client, db_session, unique_email):
    org_id, admin_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())
    llm = _ScriptedTracingApprovalLLM(account_id=str(account.id))

    first = await run_investigation(db_session, org_id, account.id, thread_id=thread_id, llm=llm)
    assert first.awaiting_approval is True

    traces_after_pause = TraceRepository(db_session, org_id).list()
    assert len(traces_after_pause) == 1
    # Two pre-pause turns: (100,20) + (200,40)
    assert traces_after_pause[0].input_tokens == 300
    assert traces_after_pause[0].output_tokens == 60

    approval_repo = CaseApprovalRepository(db_session, org_id)
    pending = approval_repo.list_pending()[0]
    approval_repo.resolve(pending, approved=True, decided_by_user_id=admin_id)
    db_session.commit()

    resumed = await resume_investigation_with_decision(
        db_session, org_id, thread_id, {"approved": True, "note": "ok"}, llm=llm
    )
    assert resumed.final_message == "Resumed and closed."

    traces_after_resume = TraceRepository(db_session, org_id).list()
    assert len(traces_after_resume) == 2
    resume_trace = max(traces_after_resume, key=lambda t: t.created_at)
    # Only the ONE post-resume turn: (60,15) — not re-summed with the
    # two pre-pause turns already traced by the first call.
    assert resume_trace.input_tokens == 60
    assert resume_trace.output_tokens == 15
    assert resume_trace.thread_id == thread_id


async def test_failed_investigation_still_writes_a_trace(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())
    llm = _ScriptedTracingLLM(account_id=str(account.id), raise_on_call_number=1)

    # Surfaces wrapped in an anyio ExceptionGroup (the MCP session's task
    # group), not a bare RuntimeError — str() on it is just "unhandled
    # errors in a TaskGroup", so check repr() instead, which nests the
    # real exception.
    with pytest.raises(BaseException) as exc_info:
        await run_investigation(db_session, org_id, account.id, thread_id=thread_id, llm=llm)
    assert "simulated transient provider failure" in repr(exc_info.value)

    traces = TraceRepository(db_session, org_id).list()
    assert len(traces) == 1
    assert traces[0].success is False
    assert traces[0].error is not None
    assert "RuntimeError: simulated transient provider failure" in traces[0].error
    assert traces[0].feature == TraceFeature.INVESTIGATION_AGENT.value
    assert traces[0].input_tokens is None
    assert traces[0].output_tokens is None
