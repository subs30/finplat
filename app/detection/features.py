import uuid
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.detection.rules import STRUCTURING_MIN_AMOUNT_RATIO, STRUCTURING_THRESHOLD
from app.models.account import Account
from app.models.entity import RiskRating
from app.models.transaction import Transaction, TransactionType

# Feature columns, in the fixed order the model is trained/scored on.
FEATURE_COLUMNS = [
    "account_age_days",
    "risk_rating_ordinal",
    "txn_count",
    "deposit_count",
    "withdrawal_count",
    "transfer_in_count",
    "transfer_out_count",
    "distinct_counterparties_in",
    "distinct_counterparties_out",
    "total_amount",
    "avg_amount",
    "max_amount",
    "amount_stddev",
    "near_threshold_deposit_count",
    "activity_span_days",
    "max_txns_in_any_48h",
]

_RISK_RATING_ORDINAL = {RiskRating.STANDARD: 0, RiskRating.ELEVATED: 1, RiskRating.HIGH_RISK: 2}


def _max_txns_in_any_window(timestamps: list[datetime], window_hours: int = 48) -> int:
    """Sliding-window peak transaction count — a velocity feature that's
    genuinely predictive of structuring/fan-in patterns without hardcoding
    any specific rule's threshold into the feature itself (the model
    learns how much this signal matters, rather than being told).
    """
    if not timestamps:
        return 0
    ts = sorted(timestamps)
    window = timedelta(hours=window_hours)
    best = 1
    left = 0
    for right in range(len(ts)):
        while ts[right] - ts[left] > window:
            left += 1
        best = max(best, right - left + 1)
    return best


def build_feature_table(db: Session, organization_id: uuid.UUID, *, as_of: datetime) -> pd.DataFrame:
    """One row per account in `organization_id`, engineered from that
    account's own transaction history only — see FEATURE_COLUMNS for the
    exact set. Returns a DataFrame indexed by account_id (string) with
    columns FEATURE_COLUMNS, in that fixed order, so it can be fed
    directly to a fitted model's .predict()/.predict_proba().
    """
    accounts = (
        db.execute(select(Account).where(Account.organization_id == organization_id)).scalars().all()
    )
    txns = (
        db.execute(select(Transaction).where(Transaction.organization_id == organization_id))
        .scalars()
        .all()
    )

    incoming: dict[uuid.UUID, list[Transaction]] = defaultdict(list)
    outgoing: dict[uuid.UUID, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.receiver_account_id is not None:
            incoming[t.receiver_account_id].append(t)
        if t.sender_account_id is not None:
            outgoing[t.sender_account_id].append(t)

    near_threshold_lower = STRUCTURING_THRESHOLD * STRUCTURING_MIN_AMOUNT_RATIO

    rows: dict[str, dict[str, float]] = {}
    for account in accounts:
        in_txns = incoming.get(account.id, [])
        out_txns = outgoing.get(account.id, [])
        all_txns = in_txns + out_txns
        amounts = [float(t.amount) for t in all_txns]
        all_timestamps = [t.occurred_at for t in all_txns]

        deposit_count = sum(1 for t in in_txns if t.transaction_type == TransactionType.DEPOSIT)
        withdrawal_count = sum(1 for t in out_txns if t.transaction_type == TransactionType.WITHDRAWAL)
        transfer_in_count = sum(
            1 for t in in_txns if t.transaction_type in (TransactionType.TRANSFER, TransactionType.WIRE)
        )
        transfer_out_count = sum(
            1 for t in out_txns if t.transaction_type in (TransactionType.TRANSFER, TransactionType.WIRE)
        )
        near_threshold_deposits = sum(
            1
            for t in in_txns
            if t.transaction_type == TransactionType.DEPOSIT
            and near_threshold_lower <= t.amount < STRUCTURING_THRESHOLD
        )

        amount_series = pd.Series(amounts, dtype="float64")
        age_days = (as_of - datetime.combine(account.open_date, datetime.min.time(), as_of.tzinfo)).days

        rows[str(account.id)] = {
            "account_age_days": float(age_days),
            "risk_rating_ordinal": float(_RISK_RATING_ORDINAL[account.entity.risk_rating])
            if account.entity
            else 0.0,
            "txn_count": float(len(all_txns)),
            "deposit_count": float(deposit_count),
            "withdrawal_count": float(withdrawal_count),
            "transfer_in_count": float(transfer_in_count),
            "transfer_out_count": float(transfer_out_count),
            "distinct_counterparties_in": float(len({t.sender_account_id for t in in_txns if t.sender_account_id})),
            "distinct_counterparties_out": float(
                len({t.receiver_account_id for t in out_txns if t.receiver_account_id})
            ),
            "total_amount": float(amount_series.sum()),
            "avg_amount": float(amount_series.mean()) if len(amount_series) else 0.0,
            "max_amount": float(amount_series.max()) if len(amount_series) else 0.0,
            "amount_stddev": float(amount_series.std(ddof=0)) if len(amount_series) else 0.0,
            "near_threshold_deposit_count": float(near_threshold_deposits),
            "activity_span_days": float(
                (max(all_timestamps) - min(all_timestamps)).days if len(all_timestamps) > 1 else 0
            ),
            "max_txns_in_any_48h": float(_max_txns_in_any_window(all_timestamps, window_hours=48)),
        }

    df = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_COLUMNS)
    df.index.name = "account_id"
    return df
