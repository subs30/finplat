import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from app.agent.checkpointer import get_checkpointer
from app.agent.graph import build_graph
from app.agent.investigation import run_investigation
from app.mcp.server import build_mcp_server
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register_org(client, unique_email, org_name="Checkpoint Test Co"):
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": unique_email(), "password": "supersecret1"},
    )
    return uuid.UUID(resp.json()["user"]["organization_id"])


def _make_structuring_account(db_session, organization_id: uuid.UUID) -> Account:
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
    for i in range(3):
        db_session.add(
            Transaction(
                id=uuid.uuid4(),
                organization_id=organization_id,
                sender_account_id=None,
                receiver_account_id=account.id,
                amount=Decimal("9300.00"),
                currency="USD",
                transaction_type=TransactionType.DEPOSIT,
                occurred_at=now - timedelta(days=i),
            )
        )
    db_session.flush()
    return account


async def test_investigation_resumes_from_checkpoint_after_simulated_process_restart(
    client, db_session, unique_email
):
    """The explicit proof Step 4 asks for: killing/restarting the process
    mid-investigation must resume from the last completed step, not
    restart from scratch.

    interrupt_after=["tools"] forces a deterministic pause right after
    the FIRST tool call executes — no dependency on real LLM call timing
    to "catch" a run mid-flight. Everything about the run is then torn
    down (the compiled graph, the MCP session, the checkpointer's
    Postgres connection — all local Python objects) and a completely
    fresh set is built for the resume call, with only the SAME thread_id
    connecting them. If state genuinely lives in Postgres rather than in
    this process's memory, the resumed run must have the interrupted
    run's exact messages already present, not regenerate them.
    """
    org_id = _register_org(client, unique_email)
    account = _make_structuring_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())

    # --- "before the crash": run until right after the first tool call ---
    interrupted = await run_investigation(
        db_session, org_id, account.id, thread_id=thread_id, interrupt_after=["tools"]
    )
    messages_at_interrupt = interrupted.tool_calls
    assert len(messages_at_interrupt) >= 1, "expected at least one tool call before the interrupt"
    # The investigation must NOT have reached a conclusion yet — proves
    # this really is a mid-flight pause, not a completed run.
    assert interrupted.case_id is None
    assert interrupted.final_message == ""

    first_tool_call = messages_at_interrupt[0]

    # --- "process restart": brand new db.commit()'d state is all that
    # carries over; run_investigation builds a fresh MCP server, fresh
    # checkpointer connection, and fresh compiled graph internally on
    # every call, so calling it again here with no interrupt_after is a
    # faithful stand-in for a fresh process reconnecting to Postgres. ---
    resumed = await run_investigation(db_session, org_id, account.id, thread_id=thread_id)

    # The interrupted run's first tool call must still be the first tool
    # call after resuming — not re-run, not lost, not replaced. If the
    # checkpoint didn't genuinely persist, this would either be empty
    # (state lost, started over) or a NEW/duplicate first tool call
    # (state lost, silently redone).
    assert resumed.tool_calls[0].tool == first_tool_call.tool
    assert resumed.tool_calls[0].args == first_tool_call.args

    # The resumed run actually reached a real conclusion this time.
    assert resumed.case_id is not None
    assert resumed.final_message != ""

    # Independent confirmation straight from the checkpointer's own
    # Postgres tables (not just through run_investigation's return
    # value): the persisted message list contains a ToolMessage from the
    # pre-interrupt tool call.
    server = build_mcp_server(db_session, org_id, thread_id)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = await load_mcp_tools(session)
        async with get_checkpointer() as checkpointer:
            graph = build_graph(tools, checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await graph.aget_state(config)

    tool_messages = [m for m in final_state.values["messages"] if isinstance(m, ToolMessage)]
    assert any(tm.name == first_tool_call.tool for tm in tool_messages)


async def test_resuming_a_thread_with_no_prior_state_starts_fresh(client, db_session, unique_email):
    """Sanity check for the other branch of the same logic: a never-seen
    thread_id must start a brand new investigation, not error out or
    return empty, confirming the "existing_state.values.get('messages')"
    check correctly distinguishes new from resumed.
    """
    org_id = _register_org(client, unique_email)
    account = _make_structuring_account(db_session, org_id)
    db_session.commit()

    result = await run_investigation(db_session, org_id, account.id, thread_id=str(uuid.uuid4()))

    assert len(result.tool_calls) >= 1
    assert result.case_id is not None
