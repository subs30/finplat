import uuid
from datetime import date

from app.models.account import Account, AccountStatus, AccountType
from app.models.case import CaseStatus
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.repositories.case import CaseRepository


def _register_org(client, unique_email, org_name="Case Test Co"):
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
        legal_name="Case Test Subject",
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


def test_create_case_stamps_organization_and_defaults_to_open(client, db_session, unique_email):
    org_id, user_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    repo = CaseRepository(db_session, org_id)

    case = repo.create(
        account_id=account.id,
        title="Investigation of suspicious activity",
        thread_id=str(uuid.uuid4()),
        opened_by_user_id=user_id,
    )

    assert case.organization_id == org_id
    assert case.account_id == account.id
    assert case.status == CaseStatus.OPEN
    assert case.findings_summary is None
    assert case.pending_approval_action is None


def test_get_by_thread_id_finds_the_right_case(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    repo = CaseRepository(db_session, org_id)
    thread_id = str(uuid.uuid4())
    case = repo.create(account_id=account.id, title="Thread lookup test", thread_id=thread_id)

    found = repo.get_by_thread_id(thread_id)

    assert found is not None
    assert found.id == case.id


def test_update_findings_updates_summary_and_status(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    repo = CaseRepository(db_session, org_id)
    case = repo.create(account_id=account.id, title="Update test", thread_id=str(uuid.uuid4()))

    repo.update_findings(case, findings_summary="Confirmed structuring pattern.", status=CaseStatus.CLOSED)

    assert case.findings_summary == "Confirmed structuring pattern."
    assert case.status == CaseStatus.CLOSED


def test_set_pending_approval_sets_action_and_moves_to_in_review(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    repo = CaseRepository(db_session, org_id)
    case = repo.create(account_id=account.id, title="Approval test", thread_id=str(uuid.uuid4()))

    repo.set_pending_approval(case, "Recommend freezing account pending compliance review.")

    assert case.pending_approval_action == "Recommend freezing account pending compliance review."
    assert case.status == CaseStatus.IN_REVIEW


def test_case_from_org_a_not_visible_to_org_b(client, db_session, unique_email):
    org_a, _ = _register_org(client, unique_email, "Case Isolation Org A")
    org_b, _ = _register_org(client, unique_email, "Case Isolation Org B")
    account_a = _make_account(db_session, org_a)

    repo_a = CaseRepository(db_session, org_a)
    case = repo_a.create(account_id=account_a.id, title="Org A case", thread_id=str(uuid.uuid4()))

    repo_b = CaseRepository(db_session, org_b)
    assert repo_b.get(case.id) is None
    assert repo_b.get_by_thread_id(case.thread_id) is None
    assert all(c.id != case.id for c in repo_b.list())

    assert repo_a.get(case.id) is not None
