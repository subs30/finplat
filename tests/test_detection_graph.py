import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.detection.graph import detect_layering_chains, detect_mule_communities
from app.graph.ingestion import ingest_organization_to_graph
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register_org(client, unique_email, name="Graph Detection Test Co") -> uuid.UUID:
    resp = client.post(
        "/auth/register",
        json={"organization_name": name, "email": unique_email(), "password": "supersecret1"},
    )
    return uuid.UUID(resp.json()["user"]["organization_id"])


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


def _transfer(db_session, organization_id, sender, receiver, amount, occurred_at) -> None:
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=organization_id,
            sender_account_id=sender.id,
            receiver_account_id=receiver.id,
            amount=Decimal(str(amount)),
            currency="USD",
            transaction_type=TransactionType.TRANSFER,
            occurred_at=occurred_at,
        )
    )


def test_detect_layering_chains_finds_embedded_chain(
    client, db_session, unique_email, neo4j_driver, neo4j_cleanup
):
    org_id = _register_org(client, unique_email)
    neo4j_cleanup(org_id)

    accounts = [_make_account(db_session, org_id, f"Layer {i}") for i in range(5)]
    now = datetime.now(UTC)
    amount = 50000.0
    for i in range(4):
        _transfer(db_session, org_id, accounts[i], accounts[i + 1], amount, now + timedelta(hours=i * 6))
        amount *= 0.93
    db_session.flush()
    ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    findings = detect_layering_chains(neo4j_driver, org_id)

    flagged_ids = {f.account_id for f in findings}
    assert flagged_ids == {a.id for a in accounts}


def test_detect_layering_chains_ignores_small_amounts_below_floor(
    client, db_session, unique_email, neo4j_driver, neo4j_cleanup
):
    org_id = _register_org(client, unique_email)
    neo4j_cleanup(org_id)

    accounts = [_make_account(db_session, org_id, f"SmallChain {i}") for i in range(4)]
    now = datetime.now(UTC)
    amount = 500.0
    for i in range(3):
        _transfer(db_session, org_id, accounts[i], accounts[i + 1], amount, now + timedelta(hours=i))
        amount *= 0.95
    db_session.flush()
    ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    findings = detect_layering_chains(neo4j_driver, org_id)

    assert findings == []


def test_detect_mule_communities_finds_isolated_fanin_cluster(
    client, db_session, unique_email, neo4j_driver, neo4j_cleanup
):
    org_id = _register_org(client, unique_email)
    neo4j_cleanup(org_id)

    hub = _make_account(db_session, org_id, "Mule Hub")
    sources = [_make_account(db_session, org_id, f"Victim {i}") for i in range(4)]
    destination = _make_account(db_session, org_id, "Consolidator")
    now = datetime.now(UTC)
    for i, source in enumerate(sources):
        _transfer(db_session, org_id, source, hub, 1000 + i * 50, now + timedelta(hours=i))
    _transfer(db_session, org_id, hub, destination, 3800, now + timedelta(hours=10))
    db_session.flush()
    ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    findings = detect_mule_communities(neo4j_driver, org_id, min_hub_indegree=4)

    flagged_ids = {f.account_id for f in findings}
    expected = {hub.id, destination.id, *(a.id for a in sources)}
    assert flagged_ids == expected


def test_graph_detection_is_tenant_isolated(client, db_session, unique_email, neo4j_driver, neo4j_cleanup):
    org_a = _register_org(client, unique_email, "Graph Isolation Org A")
    org_b = _register_org(client, unique_email, "Graph Isolation Org B")
    neo4j_cleanup(org_a)
    neo4j_cleanup(org_b)

    # Build the exact same isolated mule-shaped cluster in BOTH orgs.
    for org_id in (org_a, org_b):
        hub = _make_account(db_session, org_id, "Hub")
        sources = [_make_account(db_session, org_id, f"Source {i}") for i in range(4)]
        destination = _make_account(db_session, org_id, "Dest")
        now = datetime.now(UTC)
        for i, source in enumerate(sources):
            _transfer(db_session, org_id, source, hub, 1000, now + timedelta(hours=i))
        _transfer(db_session, org_id, hub, destination, 3500, now + timedelta(hours=10))
    db_session.flush()

    ingest_organization_to_graph(db_session, neo4j_driver, org_a)
    ingest_organization_to_graph(db_session, neo4j_driver, org_b)

    findings_a = detect_mule_communities(neo4j_driver, org_a, min_hub_indegree=4)
    findings_b = detect_mule_communities(neo4j_driver, org_b, min_hub_indegree=4)

    assert len(findings_a) == 6
    assert len(findings_b) == 6
    # No overlap whatsoever between the two orgs' flagged accounts.
    assert {f.account_id for f in findings_a}.isdisjoint({f.account_id for f in findings_b})
