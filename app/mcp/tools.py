"""Plain, directly-callable tool implementations — no MCP wiring here.

Each function takes a `db`/`organization_id` (and, for the graph-backed
ones, a Neo4j driver) plus string-typed arguments matching exactly what an
LLM tool call would supply over MCP (JSON has no UUID type, so IDs are
strings, parsed here). Every function is independently testable without
an agent, a checkpoint, or even MCP itself — see app/mcp/server.py for the
thin wrapper that exposes these through real MCP tool schemas, closing
over `db`/`organization_id`/`driver` so none of them are LLM-visible
arguments (see the V0.4 design report's tenant-isolation section for why
that matters).

Every lookup goes through the same TenantScopedRepository pattern as the
rest of this app — an id from another organization is indistinguishable
from a nonexistent one, same rule as every router.
"""

import uuid
from decimal import Decimal

from neo4j import Driver
from sqlalchemy.orm import Session

from app.detection.combined import get_account_detection
from app.detection.graph import detect_layering_chains, detect_mule_communities
from app.models.case import CaseStatus
from app.rag.embeddings import EmbeddingProvider
from app.repositories.account import AccountRepository
from app.repositories.case import CaseRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.transaction import TransactionRepository


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        return None


def _decimal_to_str(value: Decimal) -> str:
    return format(value, "f")


def get_transaction(db: Session, organization_id: uuid.UUID, transaction_id: str) -> dict:
    txn_id = _parse_uuid(transaction_id)
    if txn_id is None:
        return {"error": f"'{transaction_id}' is not a valid transaction id"}

    repo = TransactionRepository(db, organization_id)
    txn = repo.get(txn_id)
    if txn is None:
        return {"error": f"Transaction {transaction_id} not found"}

    return {
        "id": str(txn.id),
        "sender_account_id": str(txn.sender_account_id) if txn.sender_account_id else None,
        "receiver_account_id": str(txn.receiver_account_id) if txn.receiver_account_id else None,
        "amount": _decimal_to_str(txn.amount),
        "currency": txn.currency,
        "transaction_type": txn.transaction_type.value,
        "channel": txn.channel,
        "location_city": txn.location_city,
        "occurred_at": txn.occurred_at.isoformat(),
        "description": txn.description,
    }


def get_customer_history(db: Session, organization_id: uuid.UUID, account_id: str) -> dict:
    acc_id = _parse_uuid(account_id)
    if acc_id is None:
        return {"error": f"'{account_id}' is not a valid account id"}

    account_repo = AccountRepository(db, organization_id)
    account = account_repo.get(acc_id)
    if account is None:
        return {"error": f"Account {account_id} not found"}

    txn_repo = TransactionRepository(db, organization_id)
    transactions = txn_repo.list_for_account(acc_id)

    entity = account.entity
    return {
        "account": {
            "id": str(account.id),
            "account_number": account.account_number,
            "account_type": account.account_type.value,
            "status": account.status.value,
            "open_date": account.open_date.isoformat(),
        },
        "entity": {
            "id": str(entity.id),
            "legal_name": entity.legal_name,
            "entity_type": entity.entity_type.value,
            "risk_rating": entity.risk_rating.value,
            "kyc_status": entity.kyc_status.value,
            "phone": entity.phone,
            "email": entity.email,
            "address_line": entity.address_line,
            "city": entity.city,
        }
        if entity
        else None,
        "transaction_count": len(transactions),
        "transactions": [
            {
                "id": str(t.id),
                "direction": "outbound" if t.sender_account_id == acc_id else "inbound",
                "counterparty_account_id": str(
                    t.receiver_account_id if t.sender_account_id == acc_id else t.sender_account_id
                )
                if (t.receiver_account_id if t.sender_account_id == acc_id else t.sender_account_id)
                else None,
                "amount": _decimal_to_str(t.amount),
                "transaction_type": t.transaction_type.value,
                "occurred_at": t.occurred_at.isoformat(),
            }
            for t in transactions
        ],
    }


def search_typology(
    db: Session,
    organization_id: uuid.UUID,
    embedding_provider: EmbeddingProvider,
    query: str,
    k: int = 4,
) -> dict:
    """Retrieval-only — no second LLM call. Returns raw chunks + citations
    for the AGENT's own reasoning to read, same rationale as the V0.4
    design report: the agent's own LLM turn already synthesizes, so
    re-invoking /rag/ask's generation step here would just spend an extra
    Groq call restating what the agent can read directly.

    embedding_provider is injectable (rather than resolved internally via
    get_embedding_provider()) so tests can pass a fake, deterministic
    provider without loading the real sentence-transformers model — same
    pattern as scripts/seed_corpus.py's seed_corpus().
    """
    query_embedding = embedding_provider.embed([query])[0]

    chunk_repo = DocumentChunkRepository(db, organization_id)
    chunks = chunk_repo.search_similar(query_embedding, k=k)

    return {
        "query": query,
        "results": [
            {
                "document_id": str(c.document_id),
                "filename": c.document.filename if c.document else None,
                "doc_type": c.document.doc_type.value if c.document and c.document.doc_type else None,
                "chunk_index": c.chunk_index,
                "content": c.content,
            }
            for c in chunks
        ],
    }


def run_fraud_model(db: Session, organization_id: uuid.UUID, account_id: str) -> dict:
    acc_id = _parse_uuid(account_id)
    if acc_id is None:
        return {"error": f"'{account_id}' is not a valid account id"}

    result = get_account_detection(db, organization_id, acc_id)
    return {
        "account_id": account_id,
        "flagged_by": result.flagged_by,
        "unavailable_methods": result.unavailable_methods,
        "findings": [
            {
                "method": f.method,
                "explanation": f.explanation,
                "score": f.score,
                "evidence_transaction_ids": [str(t) for t in f.evidence_transaction_ids],
            }
            for f in result.findings
        ],
    }


def query_relationship_graph(
    db: Session, driver: Driver, organization_id: uuid.UUID, account_id: str
) -> dict:
    """Distinct from run_fraud_model: returns the account's direct 1-hop
    graph neighbors plus whether it participates in a detected layering
    chain or mule community — a graph-native investigative view for the
    agent to reason over directly, not just a pre-computed verdict.
    """
    acc_id = _parse_uuid(account_id)
    if acc_id is None:
        return {"error": f"'{account_id}' is not a valid account id"}

    with driver.session(database="neo4j") as session:
        neighbors = list(
            session.run(
                """
                MATCH (a:Account {id: $account_id, organization_id: $org_id})
                OPTIONAL MATCH (sender:Account)-[in_t:TRANSACTED]->(a)
                OPTIONAL MATCH (a)-[out_t:TRANSACTED]->(receiver:Account)
                RETURN
                    collect(DISTINCT {account_id: sender.id, amount: in_t.amount}) AS inbound,
                    collect(DISTINCT {account_id: receiver.id, amount: out_t.amount}) AS outbound
                """,
                account_id=account_id,
                org_id=str(organization_id),
            )
        )

    inbound = [n for n in (neighbors[0]["inbound"] if neighbors else []) if n["account_id"]]
    outbound = [n for n in (neighbors[0]["outbound"] if neighbors else []) if n["account_id"]]

    layering_findings = detect_layering_chains(driver, organization_id)
    mule_findings = detect_mule_communities(driver, organization_id)
    in_layering = next((f for f in layering_findings if str(f.account_id) == account_id), None)
    in_mule = next((f for f in mule_findings if str(f.account_id) == account_id), None)

    return {
        "account_id": account_id,
        "direct_inbound_senders": inbound,
        "direct_outbound_receivers": outbound,
        "in_layering_chain": in_layering.explanation if in_layering else None,
        "in_mule_community": in_mule.explanation if in_mule else None,
    }


def create_case(
    db: Session,
    organization_id: uuid.UUID,
    account_id: str,
    title: str,
    thread_id: str,
    opened_by_user_id: str | None = None,
    summary: str | None = None,
) -> dict:
    acc_id = _parse_uuid(account_id)
    if acc_id is None:
        return {"error": f"'{account_id}' is not a valid account id"}

    user_id = None
    if opened_by_user_id is not None:
        user_id = _parse_uuid(opened_by_user_id)

    case_repo = CaseRepository(db, organization_id)
    case = case_repo.create(
        account_id=acc_id,
        title=title,
        thread_id=thread_id,
        opened_by_user_id=user_id,
        findings_summary=summary,
    )
    db.commit()

    return {
        "case_id": str(case.id),
        "status": case.status.value,
        "title": case.title,
        "thread_id": case.thread_id,
    }


def update_case(
    db: Session,
    organization_id: uuid.UUID,
    case_id: str,
    findings_summary: str | None = None,
    status: str | None = None,
) -> dict:
    c_id = _parse_uuid(case_id)
    if c_id is None:
        return {"error": f"'{case_id}' is not a valid case id"}

    case_repo = CaseRepository(db, organization_id)
    case = case_repo.get(c_id)
    if case is None:
        return {"error": f"Case {case_id} not found"}

    parsed_status = None
    if status is not None:
        try:
            parsed_status = CaseStatus(status)
        except ValueError:
            return {"error": f"'{status}' is not a valid case status (open/in_review/closed)"}

    case_repo.update_findings(case, findings_summary=findings_summary, status=parsed_status)
    db.commit()

    return {
        "case_id": str(case.id),
        "status": case.status.value,
        "findings_summary": case.findings_summary,
    }


def request_human_approval(
    db: Session, organization_id: uuid.UUID, case_id: str, action_description: str
) -> dict:
    """Stub, per the V0.4 design: records a pending-approval state only.
    The real human-in-the-loop mechanics (a decision record, an
    approve/reject endpoint, notification) are V0.5's job.
    """
    c_id = _parse_uuid(case_id)
    if c_id is None:
        return {"error": f"'{case_id}' is not a valid case id"}

    case_repo = CaseRepository(db, organization_id)
    case = case_repo.get(c_id)
    if case is None:
        return {"error": f"Case {case_id} not found"}

    case_repo.set_pending_approval(case, action_description)
    db.commit()

    return {
        "case_id": str(case.id),
        "status": case.status.value,
        "pending_approval_action": case.pending_approval_action,
        "note": "This is a stub — no human has actually been notified. Real approval mechanics are V0.5.",
    }
