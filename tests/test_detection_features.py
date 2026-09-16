import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.detection.features import FEATURE_COLUMNS, build_feature_table
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType


def _register_org(client, unique_email) -> uuid.UUID:
    resp = client.post(
        "/auth/register",
        json={
            "organization_name": "Features Test Co",
            "email": unique_email(),
            "password": "supersecret1",
        },
    )
    return uuid.UUID(resp.json()["user"]["organization_id"])


def test_build_feature_table_computes_expected_columns_and_values(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)

    entity = Entity(
        id=uuid.uuid4(),
        organization_id=org_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Feature Test Person",
        risk_rating=RiskRating.ELEVATED,
        kyc_status=KycStatus.VERIFIED,
    )
    account = Account(
        id=uuid.uuid4(),
        organization_id=org_id,
        entity_id=entity.id,
        account_number="ACC-FEAT-1",
        account_type=AccountType.CHECKING,
        currency="USD",
        status=AccountStatus.ACTIVE,
        open_date=date(2025, 1, 1),
    )
    db_session.add_all([entity, account])
    db_session.flush()

    now = datetime.now(UTC)
    for i in range(3):
        txn = Transaction(
            id=uuid.uuid4(),
            organization_id=org_id,
            sender_account_id=None,
            receiver_account_id=account.id,
            amount=Decimal("9200.00"),
            currency="USD",
            transaction_type=TransactionType.DEPOSIT,
            occurred_at=now - timedelta(days=i),
        )
        db_session.add(txn)
    db_session.flush()

    features = build_feature_table(db_session, org_id, as_of=now)

    assert list(features.columns) == FEATURE_COLUMNS
    row = features.loc[str(account.id)]
    assert row["txn_count"] == 3
    assert row["deposit_count"] == 3
    assert row["near_threshold_deposit_count"] == 3  # all 3 are in [$7,500, $10,000)
    assert row["risk_rating_ordinal"] == 1  # elevated
    assert row["total_amount"] == 3 * 9200.00


def test_build_feature_table_handles_account_with_no_transactions(client, db_session, unique_email):
    org_id = _register_org(client, unique_email)
    entity = Entity(
        id=uuid.uuid4(),
        organization_id=org_id,
        entity_type=EntityType.INDIVIDUAL,
        legal_name="Dormant Person",
        risk_rating=RiskRating.STANDARD,
        kyc_status=KycStatus.VERIFIED,
    )
    account = Account(
        id=uuid.uuid4(),
        organization_id=org_id,
        entity_id=entity.id,
        account_number="ACC-DORMANT",
        account_type=AccountType.SAVINGS,
        currency="USD",
        status=AccountStatus.ACTIVE,
        open_date=date(2025, 1, 1),
    )
    db_session.add_all([entity, account])
    db_session.flush()

    features = build_feature_table(db_session, org_id, as_of=datetime.now(UTC))

    row = features.loc[str(account.id)]
    assert row["txn_count"] == 0
    assert row["avg_amount"] == 0.0
    assert row["max_txns_in_any_48h"] == 0
