#!/usr/bin/env python3
"""Load one organization's accounts/entities/transactions from Postgres
into Neo4j as the graph representation described in the V0.3 data model
report — :Entity and :Account nodes, :OWNS and :TRANSACTED relationships.

Usage:
    python scripts/ingest_graph.py --org-id <organization-uuid>

Safe to re-run: every write is a MERGE keyed on the source row's own id,
so re-running after new Postgres data lands only adds what's new.
"""
import argparse
import uuid

from app.database import SessionLocal
from app.graph.dependency import get_neo4j_driver
from app.graph.ingestion import ingest_organization_to_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True, type=uuid.UUID, help="Target organization UUID")
    args = parser.parse_args()

    db = SessionLocal()
    driver = get_neo4j_driver()
    try:
        counts = ingest_organization_to_graph(db, driver, args.org_id)
    finally:
        db.close()
        driver.close()

    print(f"Ingested org {args.org_id} into Neo4j:")
    print(f"  entities:     {counts['entities']}")
    print(f"  accounts:     {counts['accounts']}")
    print(f"  transactions: {counts['transactions']} (internal transfer/wire only)")


if __name__ == "__main__":
    main()
