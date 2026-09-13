from functools import lru_cache

from fastapi import HTTPException, status

from app.config import get_settings
from app.gateway.base import ModelProvider
from app.gateway.providers.groq import GroqProvider

# GeminiProvider and CerebrasProvider are kept as inactive reference
# implementations of the same ModelProvider interface — Groq is the active
# provider for this project's free tier.


@lru_cache
def get_gateway() -> ModelProvider:
    """FastAPI dependency returning the configured ModelProvider singleton.

    Swapping providers (or adding another one) later means changing only
    this function — routers depend on the ModelProvider interface, never on
    GroqProvider directly.
    """
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not configured on the server.",
        )
    return GroqProvider(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
