from functools import lru_cache

from fastapi import HTTPException, status
from neo4j import Driver, GraphDatabase

from app.config import get_settings


@lru_cache
def get_neo4j_driver() -> Driver:
    """Returns the configured Neo4j driver singleton — mirrors
    app.gateway.dependency.get_gateway and app.rag.dependency.get_embedding_provider:
    callers depend on this function, never construct a driver directly, and
    it fails clean (503, not a crash) when NEO4J_PASSWORD isn't configured,
    same "optional, not required for app startup" contract as the gateway's
    API keys.

    A neo4j.Driver is itself a connection-pooled client meant to be
    constructed once and reused — the @lru_cache here is that "once",
    exactly like the gateway/embedding singletons.
    """
    settings = get_settings()
    if not settings.NEO4J_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NEO4J_PASSWORD is not configured on the server.",
        )
    return GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
