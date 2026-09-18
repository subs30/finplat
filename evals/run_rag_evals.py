#!/usr/bin/env python
"""Golden-case eval for RAG retrieval quality — does a query about a given
AML/compliance topic actually surface the right document from the
synthetic corpus?

Reuses scripts/seed_corpus.py's real ingestion path (chunking + real
sentence-transformers embeddings + real pgvector) into a fresh, ephemeral
organization created just for this run and deleted afterward — same
pattern as aiplat's evals/run_agent_evals.py, adapted since finplat's
corpus lives on disk rather than being passed inline per case.

This makes ZERO real Groq calls — retrieval doesn't touch the LLM gateway
at all, only the local embedding model (a real ~80MB download on first
use, no API key) and a real Postgres+pgvector database. It is still kept
as a standalone script rather than a pytest test because it needs the
real embedding model loaded and a corpus actually ingested — heavier and
slower than pytest's established "mock the EmbeddingProvider" convention
(see tests/test_rag_router.py etc.) — consistency with that boundary
matters more than the fact this particular eval happens not to cost
quota. Run manually:

    source .venv/bin/activate
    python evals/run_rag_evals.py

Requires a reachable Postgres with pgvector enabled — the same one
`alembic upgrade head` manages. Runs against DATABASE_URL directly (not a
separate `_test` database), and cleans up everything it creates.
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.organization import Organization
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.repositories.document_chunk import DocumentChunkRepository
from evals.scoring import EvalOutcome, score_rag_case, summarize
from scripts.seed_corpus import _DEFAULT_CORPUS_DIR, seed_corpus

CASES_PATH = Path(__file__).parent / "rag_cases.json"


def _cleanup(db, organization_id: uuid.UUID) -> None:
    doc_ids = [row.id for row in db.query(Document.id).filter_by(organization_id=organization_id)]
    if doc_ids:
        db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(doc_ids)).delete(
            synchronize_session=False
        )
        db.query(Document).filter(Document.organization_id == organization_id).delete(
            synchronize_session=False
        )
    db.query(Organization).filter(Organization.id == organization_id).delete(synchronize_session=False)
    db.commit()


def main() -> int:
    data = json.loads(CASES_PATH.read_text())
    cases = data["cases"]
    top_k = data["top_k"]

    db = SessionLocal()
    organization = Organization(id=uuid.uuid4(), name=f"rag-eval-{uuid.uuid4().hex[:8]}")
    db.add(organization)
    db.commit()

    print("Loading embedding model and seeding the synthetic corpus (first run is slow)...")
    embedding_provider = SentenceTransformerEmbeddingProvider()
    seed_corpus(organization.id, _DEFAULT_CORPUS_DIR, embedding_provider=embedding_provider, db=db)

    try:
        chunk_repo = DocumentChunkRepository(db, organization.id)
        outcomes: list[EvalOutcome] = []
        for case in cases:
            query_embedding = embedding_provider.embed([case["query"]])[0]
            # Ask for more than top_k so a near-miss (found, but ranked
            # just past the threshold) is visible in the report instead
            # of looking identical to "never found at all".
            chunks = chunk_repo.search_similar(query_embedding, k=max(top_k * 2, 6))
            result_filenames = [c.document.filename for c in chunks if c.document]
            outcome = score_rag_case(case, result_filenames, top_k=top_k)
            outcomes.append(outcome)
            status = "PASS" if outcome.passed else "FAIL"
            print(f"{status}  {outcome.name}: {outcome.detail}")

        passed, total = summarize(outcomes)
        print(f"\n{passed}/{total} passed")
        return 1 if passed < total else 0
    finally:
        _cleanup(db, organization.id)
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
