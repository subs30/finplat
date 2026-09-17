import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from app.agent.checkpointer import get_checkpointer
from app.agent.graph import build_graph
from app.agent.investigation import run_investigation
from app.mcp.server import build_mcp_server
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType

# These tests verify checkpoint *mechanics* (does interrupt/resume
# genuinely persist and continue from Postgres) — not model behavior, so
# they script the LLM's responses via FakeMessagesListChatModel instead
# of making real Groq calls. Real model behavior is exercised separately
# in tests/test_investigations_router.py's category-specific tests. See
# the V0.4 follow-up discussion on why this split, not "mock everything"
# or "mock nothing", is the right call: these mechanics don't need real
# reasoning to verify, and scripting them makes the tests fast and
# deterministic; the tests that DO need real model behavior stay real
# because that's exactly the class of bug (see this file's git history)
# a script can't spontaneously reproduce.


class _FakeToolCallingChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel ignores its input and always returns the
    next scripted response, so it has no real use for a tool schema — but
    BaseChatModel.bind_tools() is abstract (NotImplementedError) unless a
    subclass provides it, since real providers use it to format tools for
    their own wire protocol. This override is a no-op: app/agent/graph.py
    calls .bind_tools(tools) once per graph build, and this fake simply
    doesn't need whatever it would have done with them.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self


def _register_org(client, unique_email, org_name="Checkpoint Test Co"):
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": unique_email(), "password": "supersecret1"},
    )
    return uuid.UUID(resp.json()["user"]["organization_id"])


def _make_account(db_session, organization_id: uuid.UUID) -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Checkpoint Test Subject",
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


def _tool_call_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def test_investigation_resumes_from_checkpoint_after_simulated_process_restart(
    client, db_session, unique_email
):
    """The explicit proof Step 4 asks for: killing/restarting the process
    mid-investigation must resume from the last completed step, not
    restart from scratch.

    interrupt_after=["tools"] forces a deterministic pause right after
    the FIRST tool call executes. Everything about the run is then torn
    down (the compiled graph, the MCP session, the checkpointer's
    Postgres connection — all local Python objects, including the fake
    LLM itself) and a completely fresh set — with a SECOND, independently
    scripted fake LLM — is built for the resume call, connected only by
    the SAME thread_id. If state genuinely lives in Postgres rather than
    in this process's memory, the resumed run must have the interrupted
    run's exact messages already present, not regenerate them.
    """
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())

    # --- "before the crash": one scripted tool call, then the interrupt ---
    fake_llm_before_crash = _FakeToolCallingChatModel(
        responses=[_tool_call_message("run_fraud_model", {"account_id": str(account.id)}, "call_1")]
    )
    interrupted = await run_investigation(
        db_session,
        org_id,
        account.id,
        thread_id=thread_id,
        interrupt_after=["tools"],
        llm=fake_llm_before_crash,
    )
    assert len(interrupted.tool_calls) == 1, "expected exactly the one scripted tool call before the interrupt"
    assert interrupted.tool_calls[0].tool == "run_fraud_model"
    # The investigation must NOT have reached a conclusion yet — proves
    # this really is a mid-flight pause, not a completed run.
    assert interrupted.case_id is None
    assert interrupted.final_message == ""

    # --- "process restart": a brand new fake LLM (fresh Python object,
    # own script) stands in for the model the resumed process would talk
    # to — nothing about it is shared with fake_llm_before_crash. ---
    fake_llm_after_restart = _FakeToolCallingChatModel(
        responses=[
            _tool_call_message(
                "create_case",
                {
                    "account_id": str(account.id),
                    "title": "Checkpoint resume test case",
                    "summary": "Scripted investigation for checkpoint mechanics testing.",
                },
                "call_2",
            ),
            AIMessage(content="Investigation complete after resume."),
        ]
    )
    resumed = await run_investigation(
        db_session, org_id, account.id, thread_id=thread_id, llm=fake_llm_after_restart
    )

    # The interrupted run's first tool call must still be the first tool
    # call after resuming — not re-run, not lost, not replaced. If the
    # checkpoint didn't genuinely persist, this would either be empty
    # (state lost, started over) or a duplicate run_fraud_model call
    # (state lost, silently redone) instead of moving straight to
    # create_case as scripted.
    assert resumed.tool_calls[0].tool == "run_fraud_model"
    assert resumed.tool_calls[0].args == {"account_id": str(account.id)}
    assert [tc.tool for tc in resumed.tool_calls] == ["run_fraud_model", "create_case"]

    # The resumed run actually reached the scripted conclusion.
    assert resumed.case_id is not None
    assert resumed.final_message == "Investigation complete after resume."

    # Independent confirmation straight from the checkpointer's own
    # Postgres tables (not just through run_investigation's return
    # value): the persisted message list contains a ToolMessage from the
    # pre-interrupt tool call. Building this graph needs SOME llm to bind
    # tools to, but .aget_state() never invokes it — a throwaway fake is
    # enough, keeping this check fully independent of Groq/network.
    server = build_mcp_server(db_session, org_id, thread_id)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = await load_mcp_tools(session)
        async with get_checkpointer() as checkpointer:
            graph = build_graph(
                tools, checkpointer, llm=_FakeToolCallingChatModel(responses=[AIMessage(content="unused")])
            )
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await graph.aget_state(config)

    tool_messages = [m for m in final_state.values["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == "run_fraud_model" for tm in tool_messages)


async def test_resuming_a_thread_with_no_prior_state_starts_fresh(client, db_session, unique_email):
    """Sanity check for the other branch of the same logic: a never-seen
    thread_id must start a brand new investigation, not error out or
    return empty, confirming the "existing_state.values.get('messages')"
    check correctly distinguishes new from resumed.
    """
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()

    fake_llm = _FakeToolCallingChatModel(
        responses=[
            _tool_call_message("run_fraud_model", {"account_id": str(account.id)}, "call_1"),
            AIMessage(content="No signs of financial crime."),
        ]
    )
    result = await run_investigation(
        db_session, org_id, account.id, thread_id=str(uuid.uuid4()), llm=fake_llm
    )

    assert [tc.tool for tc in result.tool_calls] == ["run_fraud_model"]
    assert result.final_message == "No signs of financial crime."
