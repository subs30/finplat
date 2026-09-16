import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.graph.ingestion import ingest_organization_to_graph
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register_org(client, unique_email) -> uuid.UUID:
    resp = client.post(
        "/auth/register",
        json={
            "organization_name": "Graph Ingestion Test Co",
            "email": unique_email(),
            "password": "supersecret1",
        },
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


def test_ingest_creates_expected_nodes_and_relationships(
    client, db_session, unique_email, neo4j_driver, neo4j_cleanup
):
    org_id = _register_org(client, unique_email)
    neo4j_cleanup(org_id)

    account_a = _make_account(db_session, org_id, "Sender Person")
    account_b = _make_account(db_session, org_id, "Receiver Person")
    now = datetime.now(UTC)

    transfer = Transaction(
        id=uuid.uuid4(),
        organization_id=org_id,
        sender_account_id=account_a.id,
        receiver_account_id=account_b.id,
        amount=Decimal("1500.00"),
        currency="USD",
        transaction_type=TransactionType.TRANSFER,
        occurred_at=now,
    )
    deposit = Transaction(
        id=uuid.uuid4(),
        organization_id=org_id,
        sender_account_id=None,
        receiver_account_id=account_a.id,
        amount=Decimal("500.00"),
        currency="USD",
        transaction_type=TransactionType.DEPOSIT,
        occurred_at=now,
    )
    db_session.add_all([transfer, deposit])
    db_session.flush()

    counts = ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    assert counts == {"entities": 2, "accounts": 2, "transactions": 1}  # deposit excluded, by design

    with neo4j_driver.session(database="neo4j") as session:
        node_count = session.run(
            "MATCH (a:Account {organization_id: $org_id}) RETURN count(a) AS c",
            org_id=str(org_id),
        ).single()["c"]
        edge = session.run(
            """
            MATCH (a:Account {id: $sender_id})-[t:TRANSACTED]->(b:Account {id: $receiver_id})
            RETURN t.amount AS amount, t.transaction_id AS transaction_id
            """,
            sender_id=str(account_a.id),
            receiver_id=str(account_b.id),
        ).single()

    assert node_count == 2
    assert edge is not None
    assert edge["amount"] == 1500.00
    assert edge["transaction_id"] == str(transfer.id)


def test_ingest_is_idempotent_on_rerun(client, db_session, unique_email, neo4j_driver, neo4j_cleanup):
    org_id = _register_org(client, unique_email)
    neo4j_cleanup(org_id)

    account_a = _make_account(db_session, org_id, "Idempotent Sender")
    account_b = _make_account(db_session, org_id, "Idempotent Receiver")
    txn = Transaction(
        id=uuid.uuid4(),
        organization_id=org_id,
        sender_account_id=account_a.id,
        receiver_account_id=account_b.id,
        amount=Decimal("750.00"),
        currency="USD",
        transaction_type=TransactionType.TRANSFER,
        occurred_at=datetime.now(UTC),
    )
    db_session.add(txn)
    db_session.flush()

    ingest_organization_to_graph(db_session, neo4j_driver, org_id)
    ingest_organization_to_graph(db_session, neo4j_driver, org_id)  # re-run, should not duplicate

    with neo4j_driver.session(database="neo4j") as session:
        edge_count = session.run(
            "MATCH (:Account {organization_id: $org_id})-[t:TRANSACTED]->() RETURN count(t) AS c",
            org_id=str(org_id),
        ).single()["c"]

    assert edge_count == 1


def test_layering_chain_is_traceable_as_a_graph_path(
    client, db_session, unique_email, neo4j_driver, neo4j_cleanup
):
    """The whole point of the graph representation: a multi-hop path
    query finds a chain that no single-account query could see.
    """
    org_id = _register_org(client, unique_email)
    neo4j_cleanup(org_id)

    accounts = [_make_account(db_session, org_id, f"Chain Hop {i}") for i in range(4)]
    now = datetime.now(UTC)
    for i in range(3):
        db_session.add(
            Transaction(
                id=uuid.uuid4(),
                organization_id=org_id,
                sender_account_id=accounts[i].id,
                receiver_account_id=accounts[i + 1].id,
                amount=Decimal("10000.00"),
                currency="USD",
                transaction_type=TransactionType.TRANSFER,
                occurred_at=now + timedelta(hours=i),
            )
        )
    db_session.flush()

    ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    with neo4j_driver.session(database="neo4j") as session:
        path = session.run(
            """
            MATCH path = (origin:Account {id: $origin_id})-[:TRANSACTED*3]->(dest:Account {id: $dest_id})
            RETURN length(path) AS hops
            """,
            origin_id=str(accounts[0].id),
            dest_id=str(accounts[3].id),
        ).single()

    assert path is not None
    assert path["hops"] == 3
