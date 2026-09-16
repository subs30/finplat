from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ENVIRONMENT: str = "development"

    # Some deployment targets (e.g. Render/Heroku-style platforms) inject a
    # bare `postgres://` or driverless `postgresql://` connection string.
    # SQLAlchemy would then default to psycopg2, which isn't installed here
    # (this project uses psycopg 3 — see pyproject.toml). Rewriting the
    # scheme here, in the one place both the app (database.py) and alembic
    # (env.py) read DATABASE_URL from, keeps every environment working with
    # a single source of truth. A no-op for local dev, whose .env already
    # spells out `postgresql+psycopg://`.
    @field_validator("DATABASE_URL")
    @classmethod
    def _normalize_database_url_driver(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value.removeprefix("postgres://")
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value.removeprefix("postgresql://")
        return value

    # Optional deployed frontend origin for CORS, once a frontend exists.
    FRONTEND_ORIGIN: str | None = None

    # Optional: app startup must not fail without it — only /ai/* endpoints
    # need it, and they fail with a clear message rather than a crash if
    # it's unset.
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # Inactive reference adapter, kept configurable for a possible future
    # provider swap. Same "optional, fails clean" contract as GEMINI_API_KEY.
    CEREBRAS_API_KEY: str | None = None
    CEREBRAS_MODEL: str = "gpt-oss-120b"

    # Active gateway provider. Same "optional, fails clean rather than
    # crashing app startup" contract as GEMINI_API_KEY above.
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # Neo4j (V0.3 graph detection). Required only by the graph
    # ingestion/query paths, not app startup as a whole — same "optional,
    # fails clean" contract as the gateway keys above. NEO4J_URI defaults
    # to the local install's Bolt port; NEO4J_PASSWORD has no default since
    # there is no safe default password to ship.
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str | None = None
    NEO4J_DATABASE: str = "neo4j"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from env / .env
