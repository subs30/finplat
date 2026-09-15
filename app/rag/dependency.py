from functools import lru_cache

from app.rag.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """FastAPI dependency returning the configured EmbeddingProvider
    singleton — mirrors app.gateway.dependency.get_gateway. Unlike the LLM
    gateway there's only one implementation today (no API key/config to
    check), but routers still depend on the EmbeddingProvider interface,
    never on SentenceTransformerEmbeddingProvider directly, so a future
    hosted-API backend is a drop-in swap here.
    """
    return SentenceTransformerEmbeddingProvider()
