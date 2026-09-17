import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp.server import build_mcp_server
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register_org(client, unique_email, org_name="MCP Server Test Co"):
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
        legal_name="MCP Server Test Subject",
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
    return account


async def test_list_tools_exposes_all_seven_mcp_tools_with_no_organization_id_parameter(
    client, db_session, unique_email
):
    org_id = _register_org(client, unique_email)
    server = build_mcp_server(db_session, org_id, str(uuid.uuid4()))

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.list_tools()

    # request_human_approval is deliberately NOT here — as of V0.5 it's a
    # native LangChain tool bound alongside these MCP ones (see
    # app/agent/approval_tool.py and app/agent/investigation.py), not
    # served by this MCP server at all: LangGraph's interrupt() cannot
    # suspend across the MCP request/response boundary.
    tool_names = {t.name for t in result.tools}
    assert tool_names == {
        "get_transaction",
        "get_customer_history",
        "search_typology",
        "run_fraud_model",
        "query_relationship_graph",
        "create_case",
        "update_case",
    }

    # The load-bearing tenant-isolation property: organization_id must
    # never appear in any tool's declared schema — it's bound via Python
    # closure in app/mcp/server.py, not something the LLM can see or
    # supply. If this ever regresses (e.g. someone adds organization_id
    # as an explicit tool parameter), this test catches it immediately.
    for tool in result.tools:
        properties = tool.inputSchema.get("properties", {})
        assert "organization_id" not in properties, f"{tool.name} leaks organization_id in its schema"
        assert "db" not in properties, f"{tool.name} leaks db in its schema"

    # Same property, specifically for thread_id on create_case — this is
    # the regression test for a real bug found during Step 4: the LLM
    # supplied its own made-up thread_id, which silently broke
    # Case.thread_id's link to the actual LangGraph conversation. Fixed
    # by binding it via closure, same as organization_id.
    create_case_tool = next(t for t in result.tools if t.name == "create_case")
    assert "thread_id" not in create_case_tool.inputSchema.get("properties", {})


async def test_call_tool_get_transaction_round_trips_real_data(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    txn = Transaction(
        id=uuid.uuid4(),
        organization_id=org_id,
        sender_account_id=None,
        receiver_account_id=account.id,
        amount=Decimal("777.00"),
        currency="USD",
        transaction_type=TransactionType.DEPOSIT,
        occurred_at=datetime.now(UTC),
    )
    db_session.add(txn)
    db_session.flush()

    server = build_mcp_server(db_session, org_id, str(uuid.uuid4()))
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.call_tool("get_transaction", {"transaction_id": str(txn.id)})

    assert result.isError is False
    body = json.loads(result.content[0].text)
    assert body["id"] == str(txn.id)
    assert body["amount"] == "777.00"


async def test_call_tool_run_fraud_model_flags_structuring_over_mcp(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    now = datetime.now(UTC)
    for i in range(3):
        db_session.add(
            Transaction(
                id=uuid.uuid4(),
                organization_id=org_id,
                sender_account_id=None,
                receiver_account_id=account.id,
                amount=Decimal("9400.00"),
                currency="USD",
                transaction_type=TransactionType.DEPOSIT,
                occurred_at=now - timedelta(days=i),
            )
        )
    db_session.flush()

    server = build_mcp_server(db_session, org_id, str(uuid.uuid4()))
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.call_tool("run_fraud_model", {"account_id": str(account.id)})

    body = json.loads(result.content[0].text)
    assert "rule:structuring" in body["flagged_by"]


async def test_call_tool_create_and_update_case_over_mcp(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)

    bound_thread_id = str(uuid.uuid4())
    server = build_mcp_server(db_session, org_id, bound_thread_id)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        create_result = await session.call_tool(
            "create_case",
            {"account_id": str(account.id), "title": "MCP-created case"},
        )
        case = json.loads(create_result.content[0].text)
        assert case["status"] == "open"
        # thread_id came from the server's closure, not an LLM-supplied
        # argument — this is the regression test for the exact bug found
        # during Step 4: the model can't invent its own thread_id.
        assert case["thread_id"] == bound_thread_id

        update_result = await session.call_tool(
            "update_case",
            {"case_id": case["case_id"], "findings_summary": "Looks clean.", "status": "closed"},
        )
        updated = json.loads(update_result.content[0].text)

    assert updated["status"] == "closed"
    assert updated["findings_summary"] == "Looks clean."
