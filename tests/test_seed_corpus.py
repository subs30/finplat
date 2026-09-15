import sys
from pathlib import Path

from app.models.document import Document, DocumentType
from app.models.document_chunk import EMBEDDING_DIMENSION, DocumentChunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from seed_corpus import _DEFAULT_CORPUS_DIR, _iter_corpus_files, seed_corpus


class _FixedEmbeddingProvider:
    dimension = EMBEDDING_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]


def test_iter_corpus_files_finds_all_18_synthetic_documents():
    files = list(_iter_corpus_files(_DEFAULT_CORPUS_DIR))
    assert len(files) == 18

    by_type: dict[DocumentType, int] = {}
    for _path, doc_type in files:
        by_type[doc_type] = by_type.get(doc_type, 0) + 1

    assert by_type[DocumentType.POLICY] == 6
    assert by_type[DocumentType.TYPOLOGY] == 6
    assert by_type[DocumentType.CASE_WRITEUP] == 6


def test_seed_corpus_ingests_all_documents_with_correct_doc_type(client, db_session, unique_email):
    email = unique_email()
    resp = client.post(
        "/auth/register",
        json={"organization_name": "Seed Test Co", "email": email, "password": "supersecret1"},
    )
    organization_id = resp.json()["user"]["organization_id"]

    seed_corpus(
        organization_id,
        _DEFAULT_CORPUS_DIR,
        embedding_provider=_FixedEmbeddingProvider(),
        db=db_session,
    )

    documents = db_session.query(Document).filter_by(organization_id=organization_id).all()
    assert len(documents) == 18
    assert all(d.status.value == "ready" for d in documents)
    assert all(d.doc_type is not None for d in documents)

    by_type: dict[str, int] = {}
    for d in documents:
        by_type[d.doc_type.value] = by_type.get(d.doc_type.value, 0) + 1
    assert by_type == {"policy": 6, "typology": 6, "case_writeup": 6}

    chunk_count = (
        db_session.query(DocumentChunk).filter_by(organization_id=organization_id).count()
    )
    assert chunk_count >= 18  # at least one chunk per document
