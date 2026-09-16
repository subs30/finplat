import uuid
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.detection.base import Finding
from app.models.transaction import Transaction, TransactionType

# --- structuring -------------------------------------------------------

STRUCTURING_THRESHOLD = Decimal(10000)
STRUCTURING_MIN_AMOUNT_RATIO = Decimal("0.75")
STRUCTURING_MIN_COUNT = 3
STRUCTURING_WINDOW_DAYS = 14


def detect_structuring(
    db: Session,
    organization_id: uuid.UUID,
    *,
    threshold: Decimal = STRUCTURING_THRESHOLD,
    min_amount_ratio: Decimal = STRUCTURING_MIN_AMOUNT_RATIO,
    min_count: int = STRUCTURING_MIN_COUNT,
    window_days: int = STRUCTURING_WINDOW_DAYS,
) -> list[Finding]:
    """Flags an account with `min_count`+ deposits, each in
    [`threshold` * `min_amount_ratio`, `threshold`), all falling within a
    `window_days`-wide span.

    The amount floor matters as much as the ceiling: an early version of
    this rule only checked "under threshold," which flagged ordinary small
    deposits (a background account with three $1,200 deposits in a month
    is not structuring) — 7 of 10 initial findings on the synthetic
    dataset were exactly that false-positive shape. Requiring amounts
    *close to* the threshold (default: at least 75% of it, i.e.
    $7,500-$9,999.99 for a $10,000 threshold) is what actually
    distinguishes "deliberately kept just under the reporting line" from
    "an unrelated small transaction that happens to be under it."

    The window is a rolling span, not "48 hours" as sometimes quoted for
    structuring: real structuring is often spaced over days-to-weeks
    specifically to look less suspicious than same-day deposits would — a
    narrow window would be trivially evaded by spacing deposits a few days
    apart. 14 days is deliberately wider than a naive 48h rule for that
    reason.
    """
    lower_bound = threshold * min_amount_ratio
    deposits = (
        db.execute(
            select(Transaction)
            .where(Transaction.organization_id == organization_id)
            .where(Transaction.transaction_type == TransactionType.DEPOSIT)
            .where(Transaction.amount >= lower_bound)
            .where(Transaction.amount < threshold)
            .where(Transaction.receiver_account_id.is_not(None))
            .order_by(Transaction.receiver_account_id, Transaction.occurred_at)
        )
        .scalars()
        .all()
    )

    by_account: dict[uuid.UUID, list[Transaction]] = defaultdict(list)
    for txn in deposits:
        assert txn.receiver_account_id is not None
        by_account[txn.receiver_account_id].append(txn)

    window = timedelta(days=window_days)
    findings: list[Finding] = []
    for account_id, txns in by_account.items():
        # txns already time-sorted by the query's ORDER BY. Slide a
        # min_count-wide window over consecutive deposits; the first
        # window found that fits within `window` is enough to flag the
        # account — no need to also report every later overlapping window.
        for i in range(len(txns) - min_count + 1):
            group = txns[i : i + min_count]
            span = group[-1].occurred_at - group[0].occurred_at
            if span <= window:
                total = sum(t.amount for t in group)
                findings.append(
                    Finding(
                        method="rule:structuring",
                        account_id=account_id,
                        triggered=True,
                        explanation=(
                            f"{len(group)} deposits of ${lower_bound:,.0f}-${threshold:,.0f} each "
                            f"(totaling ${total:,.2f}) within {span.days} day(s), "
                            f"well inside the {window_days}-day structuring window."
                        ),
                        evidence_transaction_ids=[t.id for t in group],
                        detected_at=group[-1].occurred_at,
                    )
                )
                break

    return findings


# --- fan-in / fan-out (mule) ---------------------------------------------

MULE_MIN_INBOUND_SOURCES = 5
MULE_INBOUND_WINDOW_HOURS = 72
MULE_MAX_OUTBOUND_DESTINATIONS = 2
MULE_OUTBOUND_WINDOW_HOURS = 48


def detect_fanin_fanout(
    db: Session,
    organization_id: uuid.UUID,
    *,
    min_inbound_sources: int = MULE_MIN_INBOUND_SOURCES,
    inbound_window_hours: int = MULE_INBOUND_WINDOW_HOURS,
    max_outbound_destinations: int = MULE_MAX_OUTBOUND_DESTINATIONS,
    outbound_window_hours: int = MULE_OUTBOUND_WINDOW_HOURS,
) -> list[Finding]:
    """Flags an account that receives transfers from `min_inbound_sources`+
    distinct accounts within `inbound_window_hours`, then sends to
    `max_outbound_destinations` or fewer distinct accounts within
    `outbound_window_hours` of the last qualifying inbound transfer — the
    fan-in-then-fan-out signature of a mule account.

    Single-hop and account-scoped, so unlike Step 6's graph-native
    detection this is a query any relational database could run — that's
    deliberate: this rule and the graph community/topology detection are
    meant to catch overlapping but not identical cases (see the V0.3
    report's Step 6 section for why fan-in-degree alone, with no time
    window, is too noisy on its own).
    """
    transfers = (
        db.execute(
            select(Transaction)
            .where(Transaction.organization_id == organization_id)
            .where(Transaction.transaction_type.in_([TransactionType.TRANSFER, TransactionType.WIRE]))
            .where(Transaction.sender_account_id.is_not(None))
            .where(Transaction.receiver_account_id.is_not(None))
            .order_by(Transaction.occurred_at)
        )
        .scalars()
        .all()
    )

    inbound_by_account: dict[uuid.UUID, list[Transaction]] = defaultdict(list)
    outbound_by_account: dict[uuid.UUID, list[Transaction]] = defaultdict(list)
    for txn in transfers:
        assert txn.sender_account_id is not None and txn.receiver_account_id is not None
        inbound_by_account[txn.receiver_account_id].append(txn)
        outbound_by_account[txn.sender_account_id].append(txn)

    inbound_window = timedelta(hours=inbound_window_hours)
    outbound_window = timedelta(hours=outbound_window_hours)
    findings: list[Finding] = []

    for account_id, inbound in inbound_by_account.items():
        # Find the earliest window (by end time) with enough distinct senders.
        fanin_window: list[Transaction] | None = None
        for i in range(len(inbound)):
            window_end = inbound[i].occurred_at
            window_start = window_end - inbound_window
            candidates = [t for t in inbound if window_start <= t.occurred_at <= window_end]
            distinct_senders = {t.sender_account_id for t in candidates}
            if len(distinct_senders) >= min_inbound_sources:
                fanin_window = candidates
                break
        if fanin_window is None:
            continue

        last_inbound_time = fanin_window[-1].occurred_at
        outbound = outbound_by_account.get(account_id, [])
        fanout_candidates = [
            t
            for t in outbound
            if last_inbound_time < t.occurred_at <= last_inbound_time + outbound_window
        ]
        distinct_destinations = {t.receiver_account_id for t in fanout_candidates}
        if fanout_candidates and 0 < len(distinct_destinations) <= max_outbound_destinations:
            distinct_sender_count = len({t.sender_account_id for t in fanin_window})
            findings.append(
                Finding(
                    method="rule:fanin_fanout",
                    account_id=account_id,
                    triggered=True,
                    explanation=(
                        f"Received transfers from {distinct_sender_count} distinct accounts "
                        f"within {inbound_window_hours}h, then sent to "
                        f"{len(distinct_destinations)} destination(s) within "
                        f"{outbound_window_hours}h of the last inbound transfer — "
                        "fan-in-then-fan-out consistent with a mule account."
                    ),
                    evidence_transaction_ids=[t.id for t in fanin_window]
                    + [t.id for t in fanout_candidates],
                    detected_at=fanout_candidates[-1].occurred_at,
                )
            )

    return findings


def run_all_rules(db: Session, organization_id: uuid.UUID) -> list[Finding]:
    return detect_structuring(db, organization_id) + detect_fanin_fanout(db, organization_id)
