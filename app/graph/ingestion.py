import uuid
from datetime import UTC, datetime

from neo4j import Driver
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.account import Account
from app.models.entity import Entity
from app.models.transaction import Transaction, TransactionType

# Row dicts below pass raw Python date/datetime objects for open_date and
# occurred_at, not .isoformat() strings — the neo4j driver converts those
# to native Neo4j Date/DateTime property types automatically. That's not
# cosmetic: app/detection/graph.py's layering-chain query needs real
# temporal comparison (hop N+1 occurred at or after hop N) to enforce time
# ordering along a path, and lexicographic string comparison of ISO
# timestamps is the kind of thing that's "usually right" and wrong exactly
# when you'd least expect it (mixed offsets, missing zero-padding) — not a
# risk worth taking on data a detection query's correctness depends on.
#
# occurred_at specifically must be normalized to a fixed UTC offset first
# (`.astimezone(UTC)`), not left as whatever tzinfo psycopg attached —
# this Postgres session returns TIMESTAMPTZ values with a pytz-style
# Asia/Dhaka zone object rather than a fixed offset, and the neo4j
# driver's temporal dehydration path errors ("utcoffset must be a whole
# number of minutes") on pytz's dynamic offset computation even though
# the actual offset is a clean +06:00. Normalizing to UTC sidesteps that
# pytz/driver interaction entirely rather than fighting it.


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)

# Neo4j Community Edition supports one user database, not one-per-tenant —
# so every node/relationship carries organization_id as a plain property,
# and every query in this module (and every detection query built on top
# of it) MUST filter on it explicitly. This is the graph-world equivalent
# of app.repositories.base.TenantScopedRepository's organization_id
# column filter: same rule, enforced by convention here since there is no
# structural "repository" layer in Cypher to make it automatic. Do not add
# a query here (or in app/detection/graph.py) that reads Account/Entity
# nodes without an organization_id predicate.
_ENSURE_CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
    "CREATE INDEX entity_org_id IF NOT EXISTS FOR (e:Entity) ON (e.organization_id)",
    "CREATE INDEX account_org_id IF NOT EXISTS FOR (a:Account) ON (a.organization_id)",
]

_MERGE_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id})
SET e.organization_id = row.organization_id,
    e.entity_type = row.entity_type,
    e.legal_name = row.legal_name,
    e.phone = row.phone,
    e.email = row.email,
    e.address_line = row.address_line,
    e.city = row.city,
    e.country = row.country,
    e.risk_rating = row.risk_rating,
    e.kyc_status = row.kyc_status
"""

_MERGE_ACCOUNTS = """
UNWIND $rows AS row
MERGE (a:Account {id: row.id})
SET a.organization_id = row.organization_id,
    a.account_number = row.account_number,
    a.account_type = row.account_type,
    a.currency = row.currency,
    a.status = row.status,
    a.open_date = row.open_date
WITH a, row
MATCH (e:Entity {id: row.entity_id})
MERGE (e)-[:OWNS]->(a)
"""

# Only transactions where BOTH sides are internal accounts become graph
# edges — see the V0.3 data model report: deposits/withdrawals (money
# crossing the bank's boundary) are a rules/ML concern (amount + velocity),
# not a graph-topology one, and folding them in as edges to one shared
# "external" node would create an artificial hub that distorts exactly the
# degree/community-detection algorithms Step 6 needs to be genuine.
_MERGE_TRANSACTIONS = """
UNWIND $rows AS row
MATCH (sender:Account {id: row.sender_id})
MATCH (receiver:Account {id: row.receiver_id})
MERGE (sender)-[t:TRANSACTED {transaction_id: row.transaction_id}]->(receiver)
SET t.amount = row.amount,
    t.currency = row.currency,
    t.transaction_type = row.transaction_type,
    t.occurred_at = row.occurred_at,
    t.channel = row.channel
"""

_BATCH_SIZE = 500


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest_organization_to_graph(
    db: Session, driver: Driver, organization_id: uuid.UUID, *, database: str = "neo4j"
) -> dict[str, int]:
    """Loads one organization's entities/accounts/internal transactions from
    Postgres into Neo4j as the graph representation described in the V0.3
    data model report. Idempotent: every write is a MERGE keyed on the
    Postgres row's own id (or, for transactions, transaction_id), so
    re-running this after new Postgres rows are added only adds the new
    ones — it never duplicates nodes/relationships for rows already loaded.
    """
    entities = db.execute(select(Entity).where(Entity.organization_id == organization_id)).scalars().all()
    accounts = db.execute(select(Account).where(Account.organization_id == organization_id)).scalars().all()
    internal_txns = (
        db.execute(
            select(Transaction)
            .where(Transaction.organization_id == organization_id)
            .where(Transaction.sender_account_id.is_not(None))
            .where(Transaction.receiver_account_id.is_not(None))
        )
        .scalars()
        .all()
    )

    entity_rows = [
        {
            "id": str(e.id),
            "organization_id": str(e.organization_id),
            "entity_type": e.entity_type.value,
            "legal_name": e.legal_name,
            "phone": e.phone,
            "email": e.email,
            "address_line": e.address_line,
            "city": e.city,
            "country": e.country,
            "risk_rating": e.risk_rating.value,
            "kyc_status": e.kyc_status.value,
        }
        for e in entities
    ]
    account_rows = [
        {
            "id": str(a.id),
            "organization_id": str(a.organization_id),
            "entity_id": str(a.entity_id),
            "account_number": a.account_number,
            "account_type": a.account_type.value,
            "currency": a.currency,
            "status": a.status.value,
            "open_date": a.open_date,
        }
        for a in accounts
    ]
    txn_rows = [
        {
            "transaction_id": str(t.id),
            "sender_id": str(t.sender_account_id),
            "receiver_id": str(t.receiver_account_id),
            "amount": float(t.amount),
            "currency": t.currency,
            "transaction_type": t.transaction_type.value,
            "occurred_at": _to_utc(t.occurred_at),
            "channel": t.channel,
        }
        for t in internal_txns
        if t.transaction_type in (TransactionType.TRANSFER, TransactionType.WIRE)
    ]

    with driver.session(database=database) as session:
        for statement in _ENSURE_CONSTRAINTS:
            session.run(statement)
        for batch in _chunks(entity_rows, _BATCH_SIZE):
            session.run(_MERGE_ENTITIES, rows=batch)
        for batch in _chunks(account_rows, _BATCH_SIZE):
            session.run(_MERGE_ACCOUNTS, rows=batch)
        for batch in _chunks(txn_rows, _BATCH_SIZE):
            session.run(_MERGE_TRANSACTIONS, rows=batch)

    return {
        "entities": len(entity_rows),
        "accounts": len(account_rows),
        "transactions": len(txn_rows),
    }
