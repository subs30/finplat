#!/usr/bin/env python3
"""Ingest the synthetic AML/compliance corpus into one organization's RAG
knowledge base.

Documents in aiplat (and finplat) are always tenant-owned rows — there is
no shared/global document concept in the schema (every row carries a
required organization_id FK). So making this reference corpus available to
a tenant means ingesting a copy of it into that tenant's own document
table, the same way any other document reaches RAG: via
app.rag.ingestion.ingest_document(). This script just does that in bulk,
for every file under a corpus directory, tagging each with the
DocumentType inferred from its containing subdirectory (policy/ ->
DocumentType.POLICY, etc).

Usage:
    python scripts/seed_corpus.py --org-id <organization-uuid>
    python scripts/seed_corpus.py --org-id <organization-uuid> --corpus-dir /path/to/other/corpus

Re-running this for the same organization ingests a second copy of every
file (no de-duplication by filename) — intentional, since it's meant to
be a one-time seed step per tenant, and the resulting duplicate documents
are otherwise harmless (they'd just rank near-identically in retrieval to
their earlier twin).
"""
import argparse
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.document import DocumentType
from app.rag.dependency import get_embedding_provider
from app.rag.embeddings import EmbeddingProvider
from app.rag.ingestion import ingest_document

_DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "synthetic_corpus"

_DIR_NAME_TO_DOC_TYPE = {
    "policy": DocumentType.POLICY,
    "typology": DocumentType.TYPOLOGY,
    "case_writeup": DocumentType.CASE_WRITEUP,
}


def _iter_corpus_files(corpus_dir: Path) -> Iterator[tuple[Path, DocumentType]]:
    for dir_name, doc_type in _DIR_NAME_TO_DOC_TYPE.items():
        subdir = corpus_dir / dir_name
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.glob("*.md")):
            yield path, doc_type


def seed_corpus(
    organization_id: uuid.UUID,
    corpus_dir: Path,
    embedding_provider: EmbeddingProvider | None = None,
    db: Session | None = None,
) -> None:
    # embedding_provider and db are both injectable (rather than always
    # resolved/opened here) so tests can pass a fake embedding provider and
    # the test suite's own isolated session — same rationale as every
    # router in this app depending on the EmbeddingProvider interface
    # rather than constructing one directly, and the same db_session
    # fixture every router test already reuses.
    if embedding_provider is None:
        embedding_provider = get_embedding_provider()
    owns_db = db is None
    if db is None:
        db = SessionLocal()
    ready, failed, total_chunks = 0, 0, 0
    try:
        for path, doc_type in _iter_corpus_files(corpus_dir):
            content = path.read_text()
            document = ingest_document(
                db,
                organization_id=organization_id,
                uploaded_by_user_id=None,
                filename=path.name,
                content=content,
                embedding_provider=embedding_provider,
                doc_type=doc_type,
            )
            db.commit()
            db.refresh(document)
            chunk_count = len(document.chunks)
            total_chunks += chunk_count
            if document.status.value == "ready":
                ready += 1
                print(f"  [ready]  {doc_type.value:<13} {path.name} ({chunk_count} chunks)")
            else:
                failed += 1
                print(f"  [FAILED] {doc_type.value:<13} {path.name}: {document.error}")
    finally:
        if owns_db:
            db.close()

    print(f"\nSeeded org {organization_id}: {ready} ready, {failed} failed, {total_chunks} total chunks.")
    if failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True, type=uuid.UUID, help="Target organization UUID")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_DEFAULT_CORPUS_DIR,
        help=f"Corpus root directory (default: {_DEFAULT_CORPUS_DIR})",
    )
    args = parser.parse_args()
    seed_corpus(args.org_id, args.corpus_dir)


if __name__ == "__main__":
    main()
