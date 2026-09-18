import uuid
from datetime import date

from app.models.account import Account, AccountStatus, AccountType
from app.models.case import CaseStatus
from app.models.case_approval import CaseApprovalStatus
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.repositories.case import CaseRepository
from app.repositories.case_approval import CaseApprovalRepository


def _register_org(client, unique_email, org_name="Case Approval Test Co"):
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": unique_email(), "password": "supersecret1"},
    )
    body = resp.json()
    return uuid.UUID(body["user"]["organization_id"]), uuid.UUID(body["user"]["id"])


def _make_account(db_session, organization_id: uuid.UUID) -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Case Approval Test Subject",
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


def test_create_stamps_organization_and_defaults_to_pending(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    case = CaseRepository(db_session, org_id).create(
        account_id=account.id, title="Approval repo test", thread_id=str(uuid.uuid4())
    )
    repo = CaseApprovalRepository(db_session, org_id)

    approval = repo.create(
        case_id=case.id, thread_id=case.thread_id, action_description="Recommend freezing account."
    )

    assert approval.organization_id == org_id
    assert approval.case_id == case.id
    assert approval.thread_id == case.thread_id
    assert approval.status == CaseApprovalStatus.PENDING
    assert approval.decided_by_user_id is None
    assert approval.decided_at is None


def test_list_pending_excludes_decided_approvals(client, db_session, unique_email):
    org_id, user_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    case_repo = CaseRepository(db_session, org_id)
    approval_repo = CaseApprovalRepository(db_session, org_id)

    case_a = case_repo.create(account_id=account.id, title="Case A", thread_id=str(uuid.uuid4()))
    case_b = case_repo.create(account_id=account.id, title="Case B", thread_id=str(uuid.uuid4()))
    still_pending = approval_repo.create(
        case_id=case_a.id, thread_id=case_a.thread_id, action_description="Freeze case A."
    )
    already_decided = approval_repo.create(
        case_id=case_b.id, thread_id=case_b.thread_id, action_description="Freeze case B."
    )
    approval_repo.resolve(already_decided, approved=True, decided_by_user_id=user_id)

    pending = approval_repo.list_pending()

    assert [a.id for a in pending] == [still_pending.id]


def test_resolve_sets_status_decider_and_timestamp(client, db_session, unique_email):
    org_id, user_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    case = CaseRepository(db_session, org_id).create(
        account_id=account.id, title="Resolve test", thread_id=str(uuid.uuid4())
    )
    approval_repo = CaseApprovalRepository(db_session, org_id)
    approval = approval_repo.create(
        case_id=case.id, thread_id=case.thread_id, action_description="Freeze pending review."
    )

    approval_repo.resolve(approval, approved=False, decided_by_user_id=user_id)

    assert approval.status == CaseApprovalStatus.REJECTED
    assert approval.decided_by_user_id == user_id
    assert approval.decided_at is not None


def test_approval_from_org_a_not_visible_to_org_b(client, db_session, unique_email):
    org_a, _ = _register_org(client, unique_email, "Approval Isolation Org A")
    org_b, _ = _register_org(client, unique_email, "Approval Isolation Org B")
    account_a = _make_account(db_session, org_a)
    case_a = CaseRepository(db_session, org_a).create(
        account_id=account_a.id, title="Org A case", thread_id=str(uuid.uuid4())
    )
    approval = CaseApprovalRepository(db_session, org_a).create(
        case_id=case_a.id, thread_id=case_a.thread_id, action_description="Org A action."
    )

    repo_b = CaseApprovalRepository(db_session, org_b)

    assert repo_b.get(approval.id) is None
    assert repo_b.list_pending() == []


def test_clear_pending_approval_clears_field_but_leaves_status(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    case_repo = CaseRepository(db_session, org_id)
    case = case_repo.create(account_id=account.id, title="Clear test", thread_id=str(uuid.uuid4()))
    case_repo.set_pending_approval(case, "Recommend freezing account.")

    case_repo.clear_pending_approval(case)

    assert case.pending_approval_action is None
    # Deliberately untouched — the agent's own update_case call after
    # resume decides the case's real final state, not this method.
    assert case.status == CaseStatus.IN_REVIEW
