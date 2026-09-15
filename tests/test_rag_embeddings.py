from unittest.mock import MagicMock

from app.models.document_chunk import EMBEDDING_DIMENSION
from app.rag.embeddings import SentenceTransformerEmbeddingProvider


def test_dimension_matches_document_chunk_schema():
    provider = SentenceTransformerEmbeddingProvider()
    assert provider.dimension == EMBEDDING_DIMENSION


def test_embed_calls_model_encode_and_returns_list_of_lists():
    provider = SentenceTransformerEmbeddingProvider()

    # Never load the real (heavy, network-dependent) sentence-transformers
    # model in the pytest suite — stub the lazily-loaded model directly.
    fake_vectors = [[0.1] * EMBEDDING_DIMENSION, [0.2] * EMBEDDING_DIMENSION]
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_vectors
    provider._model = fake_model

    result = provider.embed(["hello", "world"])

    fake_model.encode.assert_called_once_with(["hello", "world"], normalize_embeddings=True)
    assert result == fake_vectors
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in result)


def test_embed_only_loads_model_once():
    provider = SentenceTransformerEmbeddingProvider()
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.0] * EMBEDDING_DIMENSION]
    provider._model = fake_model

    provider.embed(["one"])
    provider.embed(["two"])

    # _get_model() should short-circuit once _model is set — never re-import
    # or reconstruct the (expensive) real model.
    assert provider._model is fake_model
    assert fake_model.encode.call_count == 2
