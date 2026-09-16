import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from fastapi import HTTPException, status

import app.routers.detection as detection_router
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register(client, unique_email, org_name="Detection Router Test Co"):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": email, "password": "supersecret1"},
    )
    body = resp.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


def _make_structuring_account(db_session, organization_id: uuid.UUID) -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Router Test Structurer",
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


def test_detection_without_token_returns_401(client):
    response = client.get(f"/detection/accounts/{uuid.uuid4()}")
    assert response.status_code == 401


def test_detection_for_nonexistent_account_returns_404(client, unique_email):
    headers, _ = _register(client, unique_email)
    response = client.get(f"/detection/accounts/{uuid.uuid4()}", headers=headers)
    assert response.status_code == 404


def test_detection_flags_structuring_account_via_rules(client, db_session, unique_email):
    headers, user = _register(client, unique_email)
    account = _make_structuring_account(db_session, uuid.UUID(user["organization_id"]))

    response = client.get(f"/detection/accounts/{account.id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert "rule:structuring" in body["flagged_by"]
    struct_finding = next(f for f in body["findings"] if f["method"] == "rule:structuring")
    assert len(struct_finding["evidence_transaction_ids"]) == 3
    # This org's data was never ingested into Neo4j, so graph methods
    # correctly find nothing for it rather than erroring.
    assert "graph:layering_chain" not in body["flagged_by"]
    assert "graph:mule_community" not in body["flagged_by"]


def test_detection_tenant_isolation_returns_404_for_other_orgs_account(client, db_session, unique_email):
    headers_a, user_a = _register(client, unique_email, "Detection Isolation Org A")
    headers_b, _ = _register(client, unique_email, "Detection Isolation Org B")
    account = _make_structuring_account(db_session, uuid.UUID(user_a["organization_id"]))

    response_b = client.get(f"/detection/accounts/{account.id}", headers=headers_b)
    assert response_b.status_code == 404

    response_a = client.get(f"/detection/accounts/{account.id}", headers=headers_a)
    assert response_a.status_code == 200
    assert "rule:structuring" in response_a.json()["flagged_by"]


def test_detection_reports_ml_unavailable_when_model_not_trained(
    client, db_session, unique_email, monkeypatch
):
    def _raise_not_found():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(detection_router, "load_model", _raise_not_found)
    headers, user = _register(client, unique_email)
    account = _make_structuring_account(db_session, uuid.UUID(user["organization_id"]))

    response = client.get(f"/detection/accounts/{account.id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert "ml" in body["unavailable_methods"]
    assert not any(f["method"] == "ml:gradient_boosting" for f in body["findings"])
    # Rules still ran fine — one leg failing doesn't take down the others.
    assert "rule:structuring" in body["flagged_by"]


def test_detection_reports_graph_unavailable_when_neo4j_not_configured(
    client, db_session, unique_email, monkeypatch
):
    def _raise_503():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="not configured")

    monkeypatch.setattr(detection_router, "get_neo4j_driver", _raise_503)
    headers, user = _register(client, unique_email)
    account = _make_structuring_account(db_session, uuid.UUID(user["organization_id"]))

    response = client.get(f"/detection/accounts/{account.id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert "graph" in body["unavailable_methods"]
    assert "rule:structuring" in body["flagged_by"]


def test_detection_clean_account_returns_no_findings(client, db_session, unique_email):
    headers, user = _register(client, unique_email)
    org_id = uuid.UUID(user["organization_id"])
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=org_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Clean Person",
        risk_rating=RiskRating.STANDARD,
        kyc_status=KycStatus.VERIFIED,
    )
    account = Account(
        id=uuid.uuid4(),
        organization_id=org_id,
        entity_id=entity.id,
        account_number="ACC-CLEAN",
        account_type=AccountType.CHECKING,
        currency="USD",
        status=AccountStatus.ACTIVE,
        open_date=date(2025, 1, 1),
    )
    db_session.add_all([entity, account])
    db_session.flush()

    response = client.get(f"/detection/accounts/{account.id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["flagged_by"] == []
    assert body["findings"] == []
