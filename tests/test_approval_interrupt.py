import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from app.agent.checkpointer import get_checkpointer
from app.agent.graph import build_graph
from app.agent.investigation import (
    _content_to_text,
    resume_investigation_with_decision,
    run_investigation,
)
from app.mcp.server import build_mcp_server
from app.models.account import Account, AccountStatus, AccountType
from app.models.case import CaseStatus
from app.models.case_approval import CaseApprovalStatus
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType
from app.repositories.case import CaseRepository
from app.repositories.case_approval import CaseApprovalRepository

# This is the critical proof for V0.5: request_human_approval must pause
# the graph via LangGraph's real interrupt() — not finish the run and
# leave something for a cron job or a manual re-trigger to pick up later.
# Same discipline as tests/test_agent_checkpoint.py: a scripted LLM (no
# real Groq calls — this tests interrupt/resume *mechanics*, not model
# reasoning) and a fully independent direct Postgres check, not just
# trusting run_investigation's own return value.


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _ScriptedApprovalFlowLLM(BaseChatModel):
    """Reacts to the message history so far, rather than a fixed response
    list (langchain_core.language_models.fake_chat_models.FakeMessagesListChatModel,
    used in test_agent_checkpoint.py) — needed here because
    request_human_approval's case_id argument isn't known until
    create_case's own ToolMessage comes back mid-run.
    """

    account_id: str
    after_resume_message: str = "unreachable unless genuinely resumed"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        by_name = {m.name: m for m in tool_messages}

        if "create_case" not in by_name:
            response: AIMessage = _tool_call(
                "create_case",
                {"account_id": self.account_id, "title": "Interrupt test case", "summary": "s"},
                "c1",
            )
        elif "request_human_approval" not in by_name:
            create_case_text = _content_to_text(by_name["create_case"].content)
            assert create_case_text is not None
            case_id = json.loads(create_case_text)["case_id"]
            response = _tool_call(
                "request_human_approval",
                {"case_id": case_id, "action_description": "Freeze pending review."},
                "c2",
            )
        else:
            response = AIMessage(content=self.after_resume_message)

        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-approval-flow"


def _register_org(client, unique_email, org_name="Approval Interrupt Test Co"):
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
        legal_name="Approval Interrupt Test Subject",
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


async def test_request_human_approval_genuinely_pauses_and_resumes_via_interrupt(
    client, db_session, unique_email
):
    org_id, admin_user_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())
    llm = _ScriptedApprovalFlowLLM(account_id=str(account.id))

    result = await run_investigation(db_session, org_id, account.id, thread_id=thread_id, llm=llm)

    # The run must have stopped exactly at request_human_approval — not
    # continued on to the scripted post-resume AIMessage. If interrupt()
    # were a no-op (e.g. the MCP-boundary bug this design avoided), the
    # graph would have sailed through to "unreachable unless genuinely
    # resumed" in this same call.
    assert result.awaiting_approval is True
    assert result.final_message == ""
    assert [tc.tool for tc in result.tool_calls] == ["create_case", "request_human_approval"]

    approval_repo = CaseApprovalRepository(db_session, org_id)
    pending = approval_repo.list_pending()
    assert len(pending) == 1
    assert pending[0].id == uuid.UUID(result.approval_id)
    assert pending[0].thread_id == thread_id
    assert pending[0].action_description == "Freeze pending review."

    case_repo = CaseRepository(db_session, org_id)
    case = case_repo.get(uuid.UUID(result.case_id))
    assert case is not None
    assert case.status == CaseStatus.IN_REVIEW
    assert case.pending_approval_action == "Freeze pending review."

    # Independent confirmation straight from Postgres, via a completely
    # fresh graph/checkpointer/MCP session built from nothing but the
    # thread_id — not the Python objects run_investigation just used.
    server = build_mcp_server(db_session, org_id, thread_id)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = await load_mcp_tools(session)
        async with get_checkpointer() as checkpointer:
            graph = build_graph(tools, checkpointer, llm=_ScriptedApprovalFlowLLM(account_id="unused"))
            config = {"configurable": {"thread_id": thread_id}}
            state = await graph.aget_state(config)

    assert state.next == ("tools",)
    assert len(state.tasks) == 1
    assert len(state.tasks[0].interrupts) == 1
    assert state.tasks[0].interrupts[0].value == {
        "case_id": result.case_id,
        "action_description": "Freeze pending review.",
    }

    # Resolve and resume — same fresh-objects discipline.
    approval_repo.resolve(pending[0], approved=True, decided_by_user_id=admin_user_id)
    case_repo.clear_pending_approval(case)
    db_session.commit()

    resumed = await resume_investigation_with_decision(
        db_session, org_id, thread_id, {"approved": True, "note": "looks right"}, llm=llm
    )

    assert resumed.final_message == "unreachable unless genuinely resumed"
    assert resumed.awaiting_approval is False
    # The pre-pause tool calls are still there, in order, not re-run or
    # duplicated — proof this continued the SAME run instead of starting
    # a fresh one.
    assert [tc.tool for tc in resumed.tool_calls] == ["create_case", "request_human_approval"]


async def test_rejected_decision_resumes_with_rejection_reflected(client, db_session, unique_email):
    org_id, admin_user_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    db_session.commit()
    thread_id = str(uuid.uuid4())
    llm = _ScriptedApprovalFlowLLM(account_id=str(account.id), after_resume_message="rejection observed")

    first = await run_investigation(db_session, org_id, account.id, thread_id=thread_id, llm=llm)
    assert first.awaiting_approval is True

    approval_repo = CaseApprovalRepository(db_session, org_id)
    pending = approval_repo.list_pending()[0]
    approval_repo.resolve(pending, approved=False, decided_by_user_id=admin_user_id)
    db_session.commit()

    resumed = await resume_investigation_with_decision(
        db_session, org_id, thread_id, {"approved": False, "note": "not warranted"}, llm=llm
    )

    assert resumed.final_message == "rejection observed"
    assert pending.status == CaseApprovalStatus.REJECTED
