from abc import ABC, abstractmethod
from typing import Any

from app.models.document_chunk import EMBEDDING_DIMENSION

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Generic interface every embedding backend implements.

    Business logic (ingestion, retrieval) depends only on this interface,
    never on a specific embedding library/model directly — same spirit as
    app.gateway.base.ModelProvider for the LLM gateway. This is what lets
    the embedding backend be swapped later with no change to
    ingestion/retrieval code.
    """

    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text, in
        the same order. Every returned vector has length `self.dimension`.
        """
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local, free, no-API-key embedding backend via `sentence-transformers`.

    Why local instead of a hosted API: this project's active LLM provider
    (Groq — see app/gateway/) does not offer an embeddings endpoint on its
    free tier — its model catalog lists chat/audio models only. Gemini and
    Cerebras (the inactive reference gateway adapters) were not evaluated
    for embeddings either, for the same access-limitation reasons that took
    them out of contention for the LLM gateway in the first place (see
    app/gateway/providers/gemini.py, cerebras.py).

    "all-MiniLM-L6-v2" is a small (~80MB), well-supported, CPU-friendly
    sentence-transformers model producing 384-dimensional embeddings —
    which is why app.models.document_chunk.EMBEDDING_DIMENSION is 384. Good
    general-purpose semantic-similarity quality at a size that runs fine on
    CPU for this project's scale, with no external API call, no API key,
    and no per-call cost.

    The underlying model is loaded lazily (on first `embed()` call, not at
    construction) so importing/instantiating this class is cheap and has no
    network dependency at app startup — and so tests can swap in a fake
    model without ever importing the real (heavy) sentence-transformers
    library.
    """

    dimension = EMBEDDING_DIMENSION

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._model: Any | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in vectors]

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model
