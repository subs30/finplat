import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database import Base, get_db
from app.main import app

settings = get_settings()

# Tests run against a real Postgres instance, but in their OWN database —
# DATABASE_URL's database name with a "_test" suffix — never the dev
# database that `alembic upgrade head` manages. This fixture creates and
# drops tables in that database, so it can't collide with (or wipe)
# whatever schema/data lives in the dev database.
#
# Within the test database, each test gets an isolated, rolled-back
# transaction: we open one connection + outer transaction per test, bind the
# session to it, and roll everything back at teardown. This is fast (no
# per-test schema churn) and guarantees tests can't leak state into each
# other, while still exercising real Postgres behaviour (UUIDs, enums, unique
# constraints) that a lighter sqlite-in-memory setup would not catch.
_dev_url = make_url(settings.DATABASE_URL)
_test_url = _dev_url.set(database=f"{_dev_url.database}_test")


def _ensure_test_database_exists() -> None:
    maintenance_engine = create_engine(
        _dev_url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with maintenance_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": _test_url.database},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{_test_url.database}"'))
    finally:
        maintenance_engine.dispose()


_ensure_test_database_exists()
engine = create_engine(_test_url)


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(bind=connection, autoflush=False, autocommit=False)
    session = TestingSessionLocal()

    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, trans):
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def unique_email():
    def _make(prefix: str = "user") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"

    return _make
