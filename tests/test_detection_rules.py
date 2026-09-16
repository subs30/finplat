import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.detection.rules import detect_fanin_fanout, detect_structuring
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register_org(client, unique_email) -> uuid.UUID:
    resp = client.post(
        "/auth/register",
        json={
            "organization_name": "Rules Test Co",
            "email": unique_email(),
            "password": "supersecret1",
        },
    )
    return uuid.UUID(resp.json()["user"]["organization_id"])


def _make_account(db_session, organization_id: uuid.UUID) -> Account:
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Test Person",
        risk_rating=RiskRating.STANDARD,
        kyc_status=KycStatus.VERIFIED,
    )
    account = Account(
        id=uuid.uuid4(),
        organization_id=organization_id,
        entity_id=entity.id,
        account_number=f"ACC-{uuid.uuid4().hex[:8]}",
        account_type=AccountType.CHECKING,
        currency="USD",
        status=AccountStatus.ACTIVE,
        open_date=date(2025, 1, 1),
    )
    db_session.add_all([entity, account])
    db_session.flush()
    return account


def _deposit(db_session, organization_id, receiver, amount, occurred_at) -> Transaction:
    txn = Transaction(
        id=uuid.uuid4(),
        organization_id=organization_id,
        sender_account_id=None,
        receiver_account_id=receiver.id,
        amount=Decimal(str(amount)),
        currency="USD",
        transaction_type=TransactionType.DEPOSIT,
        occurred_at=occurred_at,
    )
    db_session.add(txn)
    db_session.flush()
    return txn


def _transfer(db_session, organization_id, sender, receiver, amount, occurred_at) -> Transaction:
    txn = Transaction(
        id=uuid.uuid4(),
        organization_id=organization_id,
        sender_account_id=sender.id,
        receiver_account_id=receiver.id,
        amount=Decimal(str(amount)),
        currency="USD",
        transaction_type=TransactionType.TRANSFER,
        occurred_at=occurred_at,
    )
    db_session.add(txn)
    db_session.flush()
    return txn


def test_detect_structuring_flags_deposits_near_threshold_within_window(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    now = datetime.now(UTC)
    for i in range(3):
        _deposit(db_session, org_id, account, 9200 + i * 50, now - timedelta(days=i))

    findings = detect_structuring(db_session, org_id)

    assert len(findings) == 1
    assert findings[0].account_id == account.id
    assert findings[0].method == "rule:structuring"
    assert len(findings[0].evidence_transaction_ids) == 3


def test_detect_structuring_ignores_ordinary_small_deposits(client, db_session, unique_email):
    """Regression test for the false-positive bug found during V0.3
    development: 3 small deposits (nowhere near the threshold) must not
    trigger structuring just because they're individually under $10,000 —
    almost every legitimate deposit is under $10,000.
    """
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    now = datetime.now(UTC)
    for i in range(3):
        _deposit(db_session, org_id, account, 500 + i * 20, now - timedelta(days=i))

    findings = detect_structuring(db_session, org_id)

    assert findings == []


def test_detect_structuring_ignores_deposits_spread_beyond_window(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    account = _make_account(db_session, org_id)
    now = datetime.now(UTC)
    for i in range(3):
        _deposit(db_session, org_id, account, 9200, now - timedelta(days=i * 20))

    findings = detect_structuring(db_session, org_id)

    assert findings == []


def test_detect_fanin_fanout_flags_mule_signature(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    hub = _make_account(db_session, org_id)
    sources = [_make_account(db_session, org_id) for _ in range(5)]
    destination = _make_account(db_session, org_id)

    now = datetime.now(UTC)
    for i, source in enumerate(sources):
        _transfer(db_session, org_id, source, hub, 1000 + i * 10, now - timedelta(hours=i * 2))
    _transfer(db_session, org_id, hub, destination, 4500, now + timedelta(hours=6))

    findings = detect_fanin_fanout(db_session, org_id)

    assert len(findings) == 1
    assert findings[0].account_id == hub.id
    assert findings[0].method == "rule:fanin_fanout"


def test_detect_fanin_fanout_ignores_insufficient_fanin(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    hub = _make_account(db_session, org_id)
    sources = [_make_account(db_session, org_id) for _ in range(2)]
    destination = _make_account(db_session, org_id)

    now = datetime.now(UTC)
    for i, source in enumerate(sources):
        _transfer(db_session, org_id, source, hub, 1000, now - timedelta(hours=i))
    _transfer(db_session, org_id, hub, destination, 1800, now + timedelta(hours=2))

    findings = detect_fanin_fanout(db_session, org_id)

    assert findings == []


def test_detect_fanin_fanout_ignores_fanin_with_wide_fanout(client, db_session, unique_email):
    """5+ distinct inbound senders alone isn't enough — the funds must also
    concentrate to a small number of destinations, not spread back out.
    """
    org_id = _register_org(client, unique_email)
    hub = _make_account(db_session, org_id)
    sources = [_make_account(db_session, org_id) for _ in range(5)]
    destinations = [_make_account(db_session, org_id) for _ in range(5)]

    now = datetime.now(UTC)
    for i, source in enumerate(sources):
        _transfer(db_session, org_id, source, hub, 1000, now - timedelta(hours=i))
    for i, dest in enumerate(destinations):
        _transfer(db_session, org_id, hub, dest, 900, now + timedelta(hours=i + 1))

    findings = detect_fanin_fanout(db_session, org_id)

    assert findings == []
