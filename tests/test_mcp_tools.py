import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.graph.ingestion import ingest_organization_to_graph
from app.mcp.tools import (
    create_case,
    get_customer_history,
    get_transaction,
    query_relationship_graph,
    run_fraud_model,
    search_typology,
    update_case,
)
from app.models.account import Account, AccountStatus, AccountType
from app.models.document import DocumentType
from app.models.document_chunk import EMBEDDING_DIMENSION
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType
from app.rag.ingestion import ingest_document


def _register_org(client, unique_email, org_name="MCP Tools Test Co"):
    resp = client.post(
        "/auth/register",
        json={"organization_name": org_name, "email": unique_email(), "password": "supersecret1"},
    )
    body = resp.json()
    return uuid.UUID(body["user"]["organization_id"]), uuid.UUID(body["user"]["id"])


def _make_account(db_session, organization_id: uuid.UUID, name="Tool Test Subject") -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name=name,
        risk_rating=RiskRating.STANDARD,
        kyc_status=KycStatus.VERIFIED,
        phone="555-000-0000",
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


class _FixedEmbeddingProvider:
    dimension = EMBEDDING_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]


# --- get_transaction ---------------------------------------------------------


def test_get_transaction_returns_expected_fields(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    txn = Transaction(
        id=uuid.uuid4(),
        organization_id=org_id,
        sender_account_id=None,
        receiver_account_id=account.id,
        amount=Decimal("1234.56"),
        currency="USD",
        transaction_type=TransactionType.DEPOSIT,
        occurred_at=datetime.now(UTC),
        channel="branch",
    )
    db_session.add(txn)
    db_session.flush()

    result = get_transaction(db_session, org_id, str(txn.id))

    assert result["id"] == str(txn.id)
    assert result["amount"] == "1234.56"
    assert result["transaction_type"] == "deposit"
    assert result["channel"] == "branch"


def test_get_transaction_returns_error_for_missing_id(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    result = get_transaction(db_session, org_id, str(uuid.uuid4()))
    assert "error" in result


def test_get_transaction_returns_error_for_malformed_id(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    result = get_transaction(db_session, org_id, "not-a-uuid")
    assert "error" in result


def test_get_transaction_is_tenant_isolated(client, db_session, unique_email):
    org_a, _ = _register_org(client, unique_email, "Tools Isolation Org A")
    org_b, _ = _register_org(client, unique_email, "Tools Isolation Org B")
    account_a = _make_account(db_session, org_a)
    txn = Transaction(
        id=uuid.uuid4(),
        organization_id=org_a,
        sender_account_id=None,
        receiver_account_id=account_a.id,
        amount=Decimal("500.00"),
        currency="USD",
        transaction_type=TransactionType.DEPOSIT,
        occurred_at=datetime.now(UTC),
    )
    db_session.add(txn)
    db_session.flush()

    result = get_transaction(db_session, org_b, str(txn.id))
    assert "error" in result


# --- get_customer_history ----------------------------------------------------


def test_get_customer_history_returns_account_entity_and_transactions(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id, "History Subject")
    other = _make_account(db_session, org_id, "Counterparty")
    now = datetime.now(UTC)
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=org_id,
            sender_account_id=None,
            receiver_account_id=account.id,
            amount=Decimal("2000.00"),
            currency="USD",
            transaction_type=TransactionType.DEPOSIT,
            occurred_at=now - timedelta(days=2),
        )
    )
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=org_id,
            sender_account_id=account.id,
            receiver_account_id=other.id,
            amount=Decimal("500.00"),
            currency="USD",
            transaction_type=TransactionType.TRANSFER,
            occurred_at=now - timedelta(days=1),
        )
    )
    db_session.flush()

    result = get_customer_history(db_session, org_id, str(account.id))

    assert result["account"]["id"] == str(account.id)
    assert result["entity"]["legal_name"] == "History Subject"
    assert result["transaction_count"] == 2
    assert result["transactions"][0]["direction"] == "inbound"
    assert result["transactions"][1]["direction"] == "outbound"
    assert result["transactions"][1]["counterparty_account_id"] == str(other.id)


def test_get_customer_history_returns_error_for_missing_account(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    result = get_customer_history(db_session, org_id, str(uuid.uuid4()))
    assert "error" in result


# --- search_typology ----------------------------------------------------------


def test_search_typology_returns_relevant_chunk_with_citation(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    ingest_document(
        db_session,
        organization_id=org_id,
        uploaded_by_user_id=None,
        filename="structuring.md",
        content="Structuring is the practice of breaking large sums into smaller deposits.",
        embedding_provider=_FixedEmbeddingProvider(),
        doc_type=DocumentType.TYPOLOGY,
    )
    db_session.commit()

    result = search_typology(db_session, org_id, _FixedEmbeddingProvider(), "structuring red flags")

    assert len(result["results"]) >= 1
    top = result["results"][0]
    assert top["filename"] == "structuring.md"
    assert top["doc_type"] == "typology"
    assert "Structuring" in top["content"]


# --- run_fraud_model ------------------------------------------------------------


def test_run_fraud_model_flags_structuring_account(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id, "Structuring Subject")
    now = datetime.now(UTC)
    for i in range(3):
        db_session.add(
            Transaction(
                id=uuid.uuid4(),
                organization_id=org_id,
                sender_account_id=None,
                receiver_account_id=account.id,
                amount=Decimal("9300.00"),
                currency="USD",
                transaction_type=TransactionType.DEPOSIT,
                occurred_at=now - timedelta(days=i),
            )
        )
    db_session.flush()

    result = run_fraud_model(db_session, org_id, str(account.id))

    assert "rule:structuring" in result["flagged_by"]


# --- query_relationship_graph ---------------------------------------------------


def test_query_relationship_graph_shows_direct_neighbors(
    client, db_session, unique_email, neo4j_driver, neo4j_cleanup
):
    org_id, _ = _register_org(client, unique_email)
    neo4j_cleanup(org_id)
    account_a = _make_account(db_session, org_id, "Graph Tool A")
    account_b = _make_account(db_session, org_id, "Graph Tool B")
    db_session.add(
        Transaction(
            id=uuid.uuid4(),
            organization_id=org_id,
            sender_account_id=account_a.id,
            receiver_account_id=account_b.id,
            amount=Decimal("1000.00"),
            currency="USD",
            transaction_type=TransactionType.TRANSFER,
            occurred_at=datetime.now(UTC),
        )
    )
    db_session.flush()
    ingest_organization_to_graph(db_session, neo4j_driver, org_id)

    result = query_relationship_graph(db_session, neo4j_driver, org_id, str(account_b.id))

    assert len(result["direct_inbound_senders"]) == 1
    assert result["direct_inbound_senders"][0]["account_id"] == str(account_a.id)
    assert result["in_layering_chain"] is None
    assert result["in_mule_community"] is None


# --- create_case / update_case ---------------------------------------------------
# request_human_approval moved out of app.mcp.tools in V0.5 — it's now a native
# LangChain tool (app.agent.approval_tool) so it can genuinely call LangGraph's
# interrupt(), which an MCP tool handler structurally cannot do. See
# tests/test_agent_checkpoint.py and tests/test_approvals_router.py for its
# coverage.


def test_create_case_then_update(client, db_session, unique_email):
    org_id, user_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id, "Case Tool Subject")

    created = create_case(
        db_session,
        org_id,
        str(account.id),
        title="Investigation of suspicious deposits",
        thread_id=str(uuid.uuid4()),
        opened_by_user_id=str(user_id),
        summary="Initial trigger: structuring rule.",
    )
    assert created["status"] == "open"

    updated = update_case(
        db_session,
        org_id,
        created["case_id"],
        findings_summary="Confirmed structuring across 3 deposits.",
        status="in_review",
    )
    assert updated["status"] == "in_review"
    assert updated["findings_summary"] == "Confirmed structuring across 3 deposits."


def test_update_case_returns_error_for_missing_case(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    result = update_case(db_session, org_id, str(uuid.uuid4()), findings_summary="x")
    assert "error" in result


def test_update_case_returns_error_for_invalid_status(client, db_session, unique_email):
    org_id, _ = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    created = create_case(
        db_session, org_id, str(account.id), title="Bad status test", thread_id=str(uuid.uuid4())
    )

    result = update_case(db_session, org_id, created["case_id"], status="not-a-real-status")

    assert "error" in result


def test_create_case_is_tenant_isolated(client, db_session, unique_email):
    org_a, _ = _register_org(client, unique_email, "Case Tool Isolation A")
    org_b, _ = _register_org(client, unique_email, "Case Tool Isolation B")
    account_a = _make_account(db_session, org_a)

    created = create_case(
        db_session, org_a, str(account_a.id), title="Org A case", thread_id=str(uuid.uuid4())
    )

    result = update_case(db_session, org_b, created["case_id"], findings_summary="tampering attempt")
    assert "error" in result
