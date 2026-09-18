import json
import uuid
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from app.agent.investigation import _content_to_text
from app.models.account import Account, AccountStatus, AccountType
from app.models.case_approval import CaseApprovalStatus
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.user import User, UserRole
from app.repositories.case import CaseRepository
from app.repositories.case_approval import CaseApprovalRepository
from app.security import create_access_token, hash_password

# Endpoint mechanics only (auth, tenant isolation, state transitions) — no
# real Groq calls. app.agent.graph.get_agent_llm is monkeypatched to a
# scripted model for every test that has to drive a real request through
# an investigation to reach a pause, same "mock mechanics" split as
# tests/test_agent_checkpoint.py and tests/test_approval_interrupt.py.
# Real model-behavior coverage lives in tests/test_investigations_router.py.


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _ScriptedApprovalFlowLLM(BaseChatModel):
    account_id: str

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        by_name = {m.name: m for m in tool_messages}

        if "create_case" not in by_name:
            response: AIMessage = _tool_call(
                "create_case",
                {"account_id": self.account_id, "title": "Router test case", "summary": "s"},
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
            response = AIMessage(content="resumed via the real endpoint")

        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-approval-flow"


def _register_org(client, unique_email, org_name="Approvals Router Test Co"):
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": unique_email(), "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def _add_member(db_session, organization_id: uuid.UUID, unique_email) -> dict:
    user = User(
        id=uuid.uuid4(),
        organization_id=organization_id,
        email=unique_email(),
        hashed_password=hash_password("supersecret1"),
        role=UserRole.MEMBER,
    )
    db_session.add(user)
    db_session.flush()
    token = create_access_token(user.id, organization_id, UserRole.MEMBER)
    return {"Authorization": f"Bearer {token}"}


def _make_account(db_session, organization_id: uuid.UUID, name="Approvals Router Subject") -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name=name,
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


def _trigger_pause(client, db_session, monkeypatch, headers, account) -> dict:
    monkeypatch.setattr(
        "app.agent.graph.get_agent_llm", lambda: _ScriptedApprovalFlowLLM(account_id=str(account.id))
    )
    resp = client.post("/investigations", json={"account_id": str(account.id)}, headers=headers)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["awaiting_approval"] is True
    return body


def test_pending_approvals_requires_auth(client):
    resp = client.get("/approvals/pending")
    assert resp.status_code == 401


def test_decide_requires_auth(client):
    resp = client.post(f"/approvals/{uuid.uuid4()}/decide", json={"approved": True})
    assert resp.status_code == 401


def test_decide_requires_admin_role(client, db_session, unique_email):
    _admin_headers, user = _register_org(client, unique_email)
    member_headers = _add_member(db_session, uuid.UUID(user["organization_id"]), unique_email)
    db_session.commit()

    resp = client.post(
        f"/approvals/{uuid.uuid4()}/decide", json={"approved": True}, headers=member_headers
    )

    assert resp.status_code == 403


def test_decide_nonexistent_approval_returns_404(client, unique_email):
    admin_headers, _ = _register_org(client, unique_email)

    resp = client.post(
        f"/approvals/{uuid.uuid4()}/decide", json={"approved": True}, headers=admin_headers
    )

    assert resp.status_code == 404


def test_pending_approvals_is_tenant_isolated(client, db_session, unique_email, monkeypatch):
    headers_a, user_a = _register_org(client, unique_email, "Approvals Isolation Org A")
    headers_b, _ = _register_org(client, unique_email, "Approvals Isolation Org B")
    account_a = _make_account(db_session, uuid.UUID(user_a["organization_id"]))
    db_session.commit()

    _trigger_pause(client, db_session, monkeypatch, headers_a, account_a)

    resp_a = client.get("/approvals/pending", headers=headers_a)
    resp_b = client.get("/approvals/pending", headers=headers_b)

    assert resp_a.status_code == 200
    assert len(resp_a.json()) == 1
    assert resp_b.status_code == 200
    assert resp_b.json() == []


def test_decide_on_another_orgs_approval_returns_404(client, db_session, unique_email, monkeypatch):
    headers_a, user_a = _register_org(client, unique_email, "Cross Org Decide A")
    headers_b, _ = _register_org(client, unique_email, "Cross Org Decide B")
    account_a = _make_account(db_session, uuid.UUID(user_a["organization_id"]))
    db_session.commit()

    paused = _trigger_pause(client, db_session, monkeypatch, headers_a, account_a)

    resp = client.post(
        f"/approvals/{paused['approval_id']}/decide", json={"approved": True}, headers=headers_b
    )

    assert resp.status_code == 404


def test_decide_approve_resumes_the_paused_agent(client, db_session, unique_email, monkeypatch):
    headers, user = _register_org(client, unique_email)
    org_id = uuid.UUID(user["organization_id"])
    account = _make_account(db_session, org_id)
    db_session.commit()

    paused = _trigger_pause(client, db_session, monkeypatch, headers, account)

    resp = client.post(
        f"/approvals/{paused['approval_id']}/decide",
        json={"approved": True, "note": "looks right"},
        headers=headers,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["approval"]["status"] == "approved"
    assert body["approval"]["decided_by_user_id"] == user["id"]
    assert body["investigation"]["final_message"] == "resumed via the real endpoint"
    assert body["investigation"]["awaiting_approval"] is False

    approval_repo = CaseApprovalRepository(db_session, org_id)
    approval = approval_repo.get(uuid.UUID(paused["approval_id"]))
    assert approval is not None
    assert approval.status == CaseApprovalStatus.APPROVED

    case_repo = CaseRepository(db_session, org_id)
    case = case_repo.get(uuid.UUID(paused["case_id"]))
    assert case is not None
    assert case.pending_approval_action is None


def test_decide_twice_returns_409(client, db_session, unique_email, monkeypatch):
    headers, user = _register_org(client, unique_email)
    account = _make_account(db_session, uuid.UUID(user["organization_id"]))
    db_session.commit()

    paused = _trigger_pause(client, db_session, monkeypatch, headers, account)

    first = client.post(
        f"/approvals/{paused['approval_id']}/decide", json={"approved": True}, headers=headers
    )
    second = client.post(
        f"/approvals/{paused['approval_id']}/decide", json={"approved": True}, headers=headers
    )

    assert first.status_code == 200
    assert second.status_code == 409
