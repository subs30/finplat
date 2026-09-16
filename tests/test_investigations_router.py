import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.graph.ingestion import ingest_organization_to_graph
from app.models.account import Account, AccountStatus, AccountType
from app.models.case import CaseStatus
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType
from app.repositories.case import CaseRepository


def _register(client, unique_email, org_name="Investigations Router Test Co"):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def _make_account(db_session, organization_id: uuid.UUID, name: str) -> Account:
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


def _deposit(db_session, org_id, receiver, amount, occurred_at):
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=org_id,
            sender_account_id=None,
            receiver_account_id=receiver.id,
            amount=Decimal(str(amount)),
            currency="USD",
            transaction_type=TransactionType.DEPOSIT,
            occurred_at=occurred_at,
        )
    )


def _transfer(db_session, org_id, sender, receiver, amount, occurred_at):
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=org_id,
            sender_account_id=sender.id,
            receiver_account_id=receiver.id,
            amount=Decimal(str(amount)),
            currency="USD",
            transaction_type=TransactionType.TRANSFER,
            occurred_at=occurred_at,
        )
    )


def test_create_investigation_without_token_returns_401(client):
    response = client.post("/investigations", json={"account_id": str(uuid.uuid4())})
    assert response.status_code == 401


def test_create_investigation_for_nonexistent_account_returns_404(client, unique_email):
    headers, _ = _register(client, unique_email)
    response = client.post("/investigations", json={"account_id": str(uuid.uuid4())}, headers=headers)
    assert response.status_code == 404


def test_create_investigation_tenant_isolation(client, db_session, unique_email):
    _headers_a, user_a = _register(client, unique_email, "Investigations Isolation Org A")
    headers_b, _ = _register(client, unique_email, "Investigations Isolation Org B")
    account_a = _make_account(db_session, uuid.UUID(user_a["organization_id"]), "Isolation Subject")
    db_session.commit()

    response = client.post(
        "/investigations", json={"account_id": str(account_a.id)}, headers=headers_b
    )
    assert response.status_code == 404


def test_investigation_structuring_account(client, db_session, unique_email):
    headers, user = _register(client, unique_email, "Investigation Structuring Co")
    org_id = uuid.UUID(user["organization_id"])
    account = _make_account(db_session, org_id, "Structuring Subject")
    now = datetime.now(UTC)
    for i in range(3):
        _deposit(db_session, org_id, account, 9300 + i * 20, now - timedelta(days=i))
    db_session.commit()

    response = client.post("/investigations", json={"account_id": str(account.id)}, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] is not None
    assert any(tc["tool"] == "run_fraud_model" for tc in body["tool_calls"])
    assert any(tc["tool"] == "create_case" for tc in body["tool_calls"])

    case_repo = CaseRepository(db_session, org_id)
    case = case_repo.get(uuid.UUID(body["case_id"]))
    assert case is not None
    assert case.thread_id == body["thread_id"]
    assert case.status in (CaseStatus.IN_REVIEW, CaseStatus.CLOSED)


def test_investigation_mule_account(client, db_session, unique_email):
    headers, user = _register(client, unique_email, "Investigation Mule Co")
    org_id = uuid.UUID(user["organization_id"])
    hub = _make_account(db_session, org_id, "Mule Hub")
    sources = [_make_account(db_session, org_id, f"Mule Victim {i}") for i in range(5)]
    destination = _make_account(db_session, org_id, "Mule Destination")
    now = datetime.now(UTC)
    for i, source in enumerate(sources):
        _transfer(db_session, org_id, source, hub, 1000 + i * 50, now - timedelta(hours=i))
    _transfer(db_session, org_id, hub, destination, 4200, now + timedelta(hours=6))
    db_session.commit()

    response = client.post("/investigations", json={"account_id": str(hub.id)}, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] is not None
    tool_names = {tc["tool"] for tc in body["tool_calls"]}
    assert "run_fraud_model" in tool_names
    assert "create_case" in tool_names


def test_investigation_layering_account(client, db_session, unique_email, neo4j_driver, neo4j_cleanup):
    headers, user = _register(client, unique_email, "Investigation Layering Co")
    org_id = uuid.UUID(user["organization_id"])
    neo4j_cleanup(org_id)

    accounts = [_make_account(db_session, org_id, f"Layer Hop {i}") for i in range(5)]
    now = datetime.now(UTC)
    amount = 50000.0
    for i in range(4):
        _transfer(db_session, org_id, accounts[i], accounts[i + 1], amount, now - timedelta(hours=(4 - i) * 6))
        amount *= 0.93
    db_session.commit()
    ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    response = client.post(
        "/investigations", json={"account_id": str(accounts[0].id)}, headers=headers
    )

    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] is not None
    tool_names = {tc["tool"] for tc in body["tool_calls"]}
    assert "query_relationship_graph" in tool_names
    assert "create_case" in tool_names


def test_investigation_clean_account_closes_case_with_no_action(client, db_session, unique_email):
    headers, user = _register(client, unique_email, "Investigation Clean Co")
    org_id = uuid.UUID(user["organization_id"])
    account = _make_account(db_session, org_id, "Clean Subject")
    now = datetime.now(UTC)
    _deposit(db_session, org_id, account, 1500, now - timedelta(days=10))
    _deposit(db_session, org_id, account, 1450, now - timedelta(days=25))
    db_session.commit()

    response = client.post("/investigations", json={"account_id": str(account.id)}, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] is not None

    case_repo = CaseRepository(db_session, org_id)
    case = case_repo.get(uuid.UUID(body["case_id"]))
    assert case is not None
    # The agent must not narrate a status it never actually set — this is
    # the regression test for exactly that gap found during Step 4.
    assert case.status == CaseStatus.CLOSED
