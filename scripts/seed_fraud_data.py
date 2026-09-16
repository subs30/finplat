#!/usr/bin/env python3
"""Generate a synthetic accounts/entities/transactions dataset for one
organization, deliberately embedding known fraud patterns (structuring,
a mule account network, layering) plus a clean background population, and
write a ground-truth fixture recording exactly which rows belong to which
pattern.

The ground truth is written to a JSON file, not a database table — see
the V0.3 data model report: keeping it outside the runtime schema means
detection code (rules/ML/graph, built in later steps) has no accidental
path to reading it. Nothing in this script ever sets an "is_fraud" style
column on entities/accounts/transactions themselves.

Usage:
    python scripts/seed_fraud_data.py --org-id <organization-uuid>
    python scripts/seed_fraud_data.py --org-id <organization-uuid> --seed 7

Re-running this for the same organization adds a second full dataset
(no de-duplication) — like scripts/seed_corpus.py, this is meant to be a
one-time seed per tenant/demo run.
"""
import argparse
import json
import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.account import Account, AccountStatus, AccountType
from app.models.entity import Entity, EntityType, KycStatus, RiskRating
from app.models.transaction import Transaction, TransactionType
from app.repositories.account import AccountRepository
from app.repositories.entity import EntityRepository
from app.repositories.transaction import TransactionRepository

_DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "data" / "synthetic_fraud_data" / "ground_truth.json"
)

# --- Tunable dataset shape -------------------------------------------------
BACKGROUND_ENTITY_COUNT = 150
BACKGROUND_WINDOW_DAYS = 120
BACKGROUND_TXN_PER_ACCOUNT = (3, 10)
BACKGROUND_AMOUNT_RANGE = (50, 3000)
BACKGROUND_LARGE_AMOUNT_RANGE = (3000, 8000)  # occasional larger-but-legitimate txn
BACKGROUND_LARGE_AMOUNT_PROB = 0.08

CLEAN_CONTROL_COUNT = 5
CLEAN_CONTROL_PAYCHECK_RANGE = (2200, 4800)
CLEAN_CONTROL_SPEND_RANGE = (30, 400)
CLEAN_CONTROL_PAY_PERIODS = 6  # ~biweekly over ~3 months

STRUCTURING_CASES = 3
STRUCTURING_DEPOSIT_COUNT = (4, 6)
STRUCTURING_THRESHOLD = Decimal(10000)
STRUCTURING_AMOUNT_RANGE = (8500, 9800)
STRUCTURING_WINDOW_DAYS = 10
STRUCTURING_CHANNELS = ["branch", "atm"]
STRUCTURING_CITIES = ["Springfield", "Rivertown", "Fairview", "Oakland Heights", "Millbrook"]

MULE_NETWORKS = 2
MULE_VICTIMS_PER_NETWORK = 6
MULE_INBOUND_AMOUNT_RANGE = (500, 4000)
MULE_WINDOW_HOURS = 72
MULE_OUTBOUND_DELAY_HOURS = (2, 36)
MULE_RETENTION_RANGE = (0.85, 0.95)  # fraction of inbound the mule forwards onward

LAYERING_CHAINS = 2
LAYERING_HOPS = 4  # origin -> intermediary x3 -> destination = 5 accounts
LAYERING_START_AMOUNT_RANGE = (45000, 60000)
LAYERING_FORWARD_FRACTION_RANGE = (0.90, 0.97)
LAYERING_HOP_DELAY_HOURS = (6, 48)

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Daniel", "Nancy", "Matthew", "Lisa",
    "Anthony", "Margaret", "Mark", "Sandra", "Paul", "Ashley", "Steven", "Kimberly",
    "Andrew", "Emily", "Kenneth", "Donna", "Joshua", "Michelle", "Kevin", "Dorothy",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]
BUSINESS_NOUNS = ["Logistics", "Holdings", "Trading", "Consulting", "Imports", "Ventures", "Supply"]
BUSINESS_NAMES = [
    f"{a} {b} {c}"
    for a, b, c in zip(
        ["Meridian", "Oakstone", "Harborview", "Crestline", "Silverleaf", "Northgate", "Bluewater"],
        BUSINESS_NOUNS,
        ["LLC", "Inc.", "Co.", "Group"] * 2,
        strict=False,
    )
]
CITIES = ["Springfield", "Rivertown", "Fairview", "Oakland Heights", "Millbrook", "Lakeside", "Cedar Falls"]


@dataclass
class GroundTruthEntry:
    pattern_type: str
    case_id: str
    description: str
    entities: list[dict] = field(default_factory=list)
    accounts: list[dict] = field(default_factory=list)
    transactions: list[str] = field(default_factory=list)


class _Generator:
    def __init__(self, db: Session, organization_id: uuid.UUID, rng: random.Random):
        self.db = db
        self.organization_id = organization_id
        self.rng = rng
        self.entity_repo = EntityRepository(db, organization_id)
        self.account_repo = AccountRepository(db, organization_id)
        self.txn_repo = TransactionRepository(db, organization_id)
        self.now = datetime.now(UTC)
        self.ground_truth: list[GroundTruthEntry] = []

    # --- generic helpers ---------------------------------------------------

    def _random_name(self, entity_type: EntityType) -> str:
        if entity_type is EntityType.BUSINESS:
            return self.rng.choice(BUSINESS_NAMES) + f" {self.rng.randint(1, 999)}"
        return f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"

    def _make_entity(
        self,
        *,
        entity_type: EntityType = EntityType.INDIVIDUAL,
        risk_rating: RiskRating = RiskRating.STANDARD,
        kyc_status: KycStatus = KycStatus.VERIFIED,
        phone: str | None = None,
        city: str | None = None,
    ) -> Entity:
        city = city or self.rng.choice(CITIES)
        entity = Entity(
            id=uuid.uuid4(),
            organization_id=self.organization_id,
            entity_type=entity_type,
            legal_name=self._random_name(entity_type),
            date_of_birth=(
                date(self.rng.randint(1950, 2002), self.rng.randint(1, 12), self.rng.randint(1, 28))
                if entity_type is EntityType.INDIVIDUAL
                else None
            ),
            registration_number=(
                f"REG-{self.rng.randint(100000, 999999)}" if entity_type is EntityType.BUSINESS else None
            ),
            phone=phone or f"555-{self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}",
            email=None,
            address_line=f"{self.rng.randint(10, 9999)} {self.rng.choice(LAST_NAMES)} St",
            city=city,
            country="USA",
            risk_rating=risk_rating,
            kyc_status=kyc_status,
        )
        self.db.add(entity)
        return entity

    def _make_account(
        self,
        entity: Entity,
        *,
        account_type: AccountType = AccountType.CHECKING,
        open_date: date | None = None,
    ) -> Account:
        account = Account(
            id=uuid.uuid4(),
            organization_id=self.organization_id,
            entity_id=entity.id,
            account_number=f"ACC-{self.rng.randint(1000000, 9999999)}",
            account_type=account_type,
            currency="USD",
            status=AccountStatus.ACTIVE,
            open_date=open_date or self._random_past_date(days_min=60, days_max=900),
        )
        self.db.add(account)
        return account

    def _random_past_date(self, *, days_min: int, days_max: int) -> date:
        return (self.now - timedelta(days=self.rng.randint(days_min, days_max))).date()

    def _random_time_in_window(self, start: datetime, end: datetime) -> datetime:
        delta = end - start
        seconds = self.rng.uniform(0, delta.total_seconds())
        return start + timedelta(seconds=seconds)

    def _make_txn(
        self,
        *,
        sender: Account | None,
        receiver: Account | None,
        amount: float | Decimal,
        transaction_type: TransactionType,
        occurred_at: datetime,
        channel: str | None = None,
        location_city: str | None = None,
        description: str | None = None,
    ) -> Transaction:
        amt = amount if isinstance(amount, Decimal) else Decimal(str(round(amount, 2)))
        txn = Transaction(
            id=uuid.uuid4(),
            organization_id=self.organization_id,
            sender_account_id=sender.id if sender else None,
            receiver_account_id=receiver.id if receiver else None,
            amount=amt,
            currency="USD",
            transaction_type=transaction_type,
            channel=channel,
            location_city=location_city,
            occurred_at=occurred_at,
            description=description,
        )
        self.db.add(txn)
        return txn

    # --- background population ---------------------------------------------

    def generate_background(self) -> list[Account]:
        window_start = self.now - timedelta(days=BACKGROUND_WINDOW_DAYS)
        accounts: list[Account] = []
        for _ in range(BACKGROUND_ENTITY_COUNT):
            entity_type = EntityType.BUSINESS if self.rng.random() < 0.25 else EntityType.INDIVIDUAL
            risk_rating = self.rng.choices(
                [RiskRating.STANDARD, RiskRating.ELEVATED, RiskRating.HIGH_RISK],
                weights=[0.82, 0.14, 0.04],
            )[0]
            home_city = self.rng.choice(CITIES)
            entity = self._make_entity(
                entity_type=entity_type, risk_rating=risk_rating, city=home_city
            )
            account_type = (
                AccountType.BUSINESS
                if entity_type is EntityType.BUSINESS
                else self.rng.choice([AccountType.CHECKING, AccountType.SAVINGS])
            )
            account = self._make_account(entity, account_type=account_type)
            accounts.append(account)

        self.db.flush()

        # A second pass for transactions, so transfer counterparties can be
        # drawn from the full pool of background accounts already created.
        for account in accounts:
            txn_count = self.rng.randint(*BACKGROUND_TXN_PER_ACCOUNT)
            for _ in range(txn_count):
                occurred_at = self._random_time_in_window(window_start, self.now)
                kind = self.rng.choices(
                    [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL, TransactionType.TRANSFER],
                    weights=[0.4, 0.3, 0.3],
                )[0]
                amount = self._background_amount()
                if kind is TransactionType.DEPOSIT:
                    self._make_txn(
                        sender=None,
                        receiver=account,
                        amount=amount,
                        transaction_type=kind,
                        occurred_at=occurred_at,
                        channel=self.rng.choice(["branch", "atm", "online"]),
                        location_city=account.entity.city if account.entity else None,
                    )
                elif kind is TransactionType.WITHDRAWAL:
                    self._make_txn(
                        sender=account,
                        receiver=None,
                        amount=amount,
                        transaction_type=kind,
                        occurred_at=occurred_at,
                        channel=self.rng.choice(["branch", "atm"]),
                    )
                else:
                    counterparty = self.rng.choice(accounts)
                    if counterparty.id == account.id:
                        continue
                    self._make_txn(
                        sender=account,
                        receiver=counterparty,
                        amount=amount,
                        transaction_type=kind,
                        occurred_at=occurred_at,
                        channel="online",
                    )
        self.db.flush()
        return accounts

    def _background_amount(self) -> float:
        if self.rng.random() < BACKGROUND_LARGE_AMOUNT_PROB:
            return self.rng.uniform(*BACKGROUND_LARGE_AMOUNT_RANGE)
        return self.rng.uniform(*BACKGROUND_AMOUNT_RANGE)

    # --- clean control -------------------------------------------------------

    def generate_clean_controls(self) -> None:
        window_start = self.now - timedelta(days=BACKGROUND_WINDOW_DAYS)
        for i in range(CLEAN_CONTROL_COUNT):
            entity = self._make_entity(risk_rating=RiskRating.STANDARD)
            account = self._make_account(entity, account_type=AccountType.CHECKING)
            self.db.flush()

            paycheck = self.rng.uniform(*CLEAN_CONTROL_PAYCHECK_RANGE)
            txn_ids: list[str] = []
            # Biweekly paycheck deposits, same amount +/- a few dollars.
            for period in range(CLEAN_CONTROL_PAY_PERIODS):
                occurred_at = window_start + timedelta(days=14 * period + self.rng.randint(0, 1))
                if occurred_at > self.now:
                    break
                txn = self._make_txn(
                    sender=None,
                    receiver=account,
                    amount=paycheck + self.rng.uniform(-5, 5),
                    transaction_type=TransactionType.DEPOSIT,
                    occurred_at=occurred_at,
                    channel="online",
                    description="payroll deposit",
                )
                txn_ids.append(str(txn.id))
                # Steady, modest spending over the following days.
                for _ in range(self.rng.randint(3, 6)):
                    spend_at = occurred_at + timedelta(days=self.rng.randint(1, 12))
                    if spend_at > self.now:
                        continue
                    txn = self._make_txn(
                        sender=account,
                        receiver=None,
                        amount=self.rng.uniform(*CLEAN_CONTROL_SPEND_RANGE),
                        transaction_type=TransactionType.WITHDRAWAL,
                        occurred_at=spend_at,
                        channel=self.rng.choice(["online", "atm"]),
                    )
                    txn_ids.append(str(txn.id))

            self.db.flush()
            self.ground_truth.append(
                GroundTruthEntry(
                    pattern_type="clean",
                    case_id=f"clean-control-{i + 1}",
                    description=(
                        "Regular biweekly payroll deposit with steady, modest spending — "
                        "a deliberate negative control, not expected to be flagged by any "
                        "detection method."
                    ),
                    entities=[{"entity_id": str(entity.id), "role": "clean_control"}],
                    accounts=[{"account_id": str(account.id), "role": "clean_control"}],
                    transactions=txn_ids,
                )
            )

    # --- structuring ---------------------------------------------------------

    def generate_structuring(self) -> None:
        for i in range(STRUCTURING_CASES):
            entity = self._make_entity(risk_rating=RiskRating.STANDARD)
            account = self._make_account(entity, account_type=AccountType.CHECKING)
            self.db.flush()

            deposit_count = self.rng.randint(*STRUCTURING_DEPOSIT_COUNT)
            window_start = self.now - timedelta(days=STRUCTURING_WINDOW_DAYS)
            txn_ids = []
            for _ in range(deposit_count):
                occurred_at = self._random_time_in_window(window_start, self.now)
                amount = self.rng.uniform(*STRUCTURING_AMOUNT_RANGE)
                assert amount < float(STRUCTURING_THRESHOLD)
                txn = self._make_txn(
                    sender=None,
                    receiver=account,
                    amount=amount,
                    transaction_type=TransactionType.DEPOSIT,
                    occurred_at=occurred_at,
                    channel=self.rng.choice(STRUCTURING_CHANNELS),
                    location_city=self.rng.choice(STRUCTURING_CITIES),
                    description="cash deposit",
                )
                txn_ids.append(str(txn.id))

            self.db.flush()
            self.ground_truth.append(
                GroundTruthEntry(
                    pattern_type="structuring",
                    case_id=f"structuring-{i + 1}",
                    description=(
                        f"{deposit_count} cash deposits of $8,500-$9,800 each, all under the "
                        f"${STRUCTURING_THRESHOLD} reporting threshold, made across multiple "
                        f"branches/ATMs within {STRUCTURING_WINDOW_DAYS} days."
                    ),
                    entities=[{"entity_id": str(entity.id), "role": "structurer"}],
                    accounts=[{"account_id": str(account.id), "role": "structurer"}],
                    transactions=txn_ids,
                )
            )

    # --- mule network ----------------------------------------------------------

    def generate_mule_networks(self) -> None:
        for i in range(MULE_NETWORKS):
            shared_phone = f"555-{self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}"

            mule_entity = self._make_entity(risk_rating=RiskRating.STANDARD, phone=shared_phone)
            mule_account = self._make_account(
                mule_entity, open_date=self._random_past_date(days_min=30, days_max=90)
            )

            dest_entity = self._make_entity(risk_rating=RiskRating.STANDARD, phone=shared_phone)
            dest_account = self._make_account(dest_entity)

            self.db.flush()

            window_start = self.now - timedelta(hours=MULE_WINDOW_HOURS)
            inbound_txns: list[Transaction] = []
            victim_entities = []
            victim_accounts = []
            total_inbound = Decimal(0)
            for _ in range(MULE_VICTIMS_PER_NETWORK):
                victim_entity = self._make_entity(risk_rating=RiskRating.STANDARD)
                victim_account = self._make_account(victim_entity)
                self.db.flush()
                victim_entities.append(victim_entity)
                victim_accounts.append(victim_account)

                occurred_at = self._random_time_in_window(window_start, self.now)
                amount = round(self.rng.uniform(*MULE_INBOUND_AMOUNT_RANGE), 2)
                txn = self._make_txn(
                    sender=victim_account,
                    receiver=mule_account,
                    amount=amount,
                    transaction_type=TransactionType.TRANSFER,
                    occurred_at=occurred_at,
                    channel="online",
                    description="payment for online task",
                )
                inbound_txns.append(txn)
                total_inbound += Decimal(str(amount))

            self.db.flush()
            inbound_txn_ids = [str(t.id) for t in inbound_txns]

            # Mule fans the accumulated funds out to the single destination
            # account in one or two transfers shortly after the last inbound
            # payment — the fan-in-then-fan-out signature.
            last_inbound_time = max(t.occurred_at for t in inbound_txns)
            retained_fraction = self.rng.uniform(*MULE_RETENTION_RANGE)
            outbound_total = total_inbound * Decimal(str(retained_fraction))
            outbound_txn_ids = []
            splits = self.rng.choice([1, 2])
            remaining = outbound_total
            for split_index in range(splits):
                delay = timedelta(hours=self.rng.uniform(*MULE_OUTBOUND_DELAY_HOURS))
                occurred_at = last_inbound_time + delay
                portion = remaining if split_index == splits - 1 else outbound_total / splits
                txn = self._make_txn(
                    sender=mule_account,
                    receiver=dest_account,
                    amount=portion,
                    transaction_type=TransactionType.TRANSFER,
                    occurred_at=occurred_at,
                    channel="online",
                )
                outbound_txn_ids.append(str(txn.id))
                remaining -= portion

            self.db.flush()
            self.ground_truth.append(
                GroundTruthEntry(
                    pattern_type="mule_network",
                    case_id=f"mule-network-{i + 1}",
                    description=(
                        f"{MULE_VICTIMS_PER_NETWORK} distinct source accounts each send one "
                        f"transfer into a single mule account within {MULE_WINDOW_HOURS}h; the "
                        "mule forwards the bulk of the funds onward to one destination account "
                        "shortly after. Mule and destination entities share a phone number "
                        "(same controller, different identities). Victim accounts are NOT "
                        "expected to be flagged — only mule and destination are."
                    ),
                    entities=(
                        [{"entity_id": str(mule_entity.id), "role": "mule"}]
                        + [{"entity_id": str(dest_entity.id), "role": "destination"}]
                        + [{"entity_id": str(e.id), "role": "victim_source"} for e in victim_entities]
                    ),
                    accounts=(
                        [{"account_id": str(mule_account.id), "role": "mule"}]
                        + [{"account_id": str(dest_account.id), "role": "destination"}]
                        + [{"account_id": str(a.id), "role": "victim_source"} for a in victim_accounts]
                    ),
                    transactions=inbound_txn_ids + outbound_txn_ids,
                )
            )

    # --- layering ----------------------------------------------------------

    def generate_layering(self) -> None:
        for i in range(LAYERING_CHAINS):
            hop_count = LAYERING_HOPS + 1  # accounts in the chain
            entities = [self._make_entity(risk_rating=RiskRating.STANDARD) for _ in range(hop_count)]
            accounts = [
                self._make_account(
                    e,
                    account_type=AccountType.BUSINESS if self.rng.random() < 0.4 else AccountType.CHECKING,
                )
                for e in entities
            ]
            self.db.flush()

            amount = Decimal(str(round(self.rng.uniform(*LAYERING_START_AMOUNT_RANGE), 2)))
            occurred_at = self.now - timedelta(days=self.rng.randint(15, 45))
            txn_ids = []
            for hop in range(LAYERING_HOPS):
                sender = accounts[hop]
                receiver = accounts[hop + 1]
                txn = self._make_txn(
                    sender=sender,
                    receiver=receiver,
                    amount=amount,
                    transaction_type=TransactionType.WIRE if hop == 0 else TransactionType.TRANSFER,
                    occurred_at=occurred_at,
                    channel="wire" if hop == 0 else "online",
                    description="consulting services payment" if hop == 0 else None,
                )
                txn_ids.append(str(txn.id))
                forward_fraction = self.rng.uniform(*LAYERING_FORWARD_FRACTION_RANGE)
                amount = amount * Decimal(str(forward_fraction))
                occurred_at = occurred_at + timedelta(hours=self.rng.uniform(*LAYERING_HOP_DELAY_HOURS))

            self.db.flush()
            roles = ["origin"] + [f"intermediary_{n}" for n in range(1, LAYERING_HOPS)] + ["destination"]
            self.ground_truth.append(
                GroundTruthEntry(
                    pattern_type="layering",
                    case_id=f"layering-{i + 1}",
                    description=(
                        f"A chain of {hop_count} accounts, each forwarding ~90-97% of a large "
                        f"received sum (starting ~${amount:,.0f} range) to the next within "
                        f"{LAYERING_HOP_DELAY_HOURS[0]}-{LAYERING_HOP_DELAY_HOURS[1]}h of receipt — "
                        "rapid pass-through with minimal retained balance at each hop."
                    ),
                    entities=[
                        {"entity_id": str(e.id), "role": r} for e, r in zip(entities, roles, strict=True)
                    ],
                    accounts=[
                        {"account_id": str(a.id), "role": r} for a, r in zip(accounts, roles, strict=True)
                    ],
                    transactions=txn_ids,
                )
            )


def seed_fraud_data(
    organization_id: uuid.UUID, out_path: Path, seed: int, db: Session | None = None
) -> dict:
    rng = random.Random(seed)
    owns_db = db is None
    if db is None:
        db = SessionLocal()
    try:
        gen = _Generator(db, organization_id, rng)
        background_accounts = gen.generate_background()
        gen.generate_clean_controls()
        gen.generate_structuring()
        gen.generate_mule_networks()
        gen.generate_layering()
        db.commit()

        entity_count = db.query(Entity).filter_by(organization_id=organization_id).count()
        account_count = db.query(Account).filter_by(organization_id=organization_id).count()
        txn_count = db.query(Transaction).filter_by(organization_id=organization_id).count()

        summary = {
            "organization_id": str(organization_id),
            "generated_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "counts": {
                "entities": entity_count,
                "accounts": account_count,
                "transactions": txn_count,
                "background_accounts": len(background_accounts),
            },
            "patterns": [
                {
                    "pattern_type": g.pattern_type,
                    "case_id": g.case_id,
                    "description": g.description,
                    "entities": g.entities,
                    "accounts": g.accounts,
                    "transactions": g.transactions,
                }
                for g in gen.ground_truth
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if owns_db:
            db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True, type=uuid.UUID, help="Target organization UUID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT_PATH, help="Ground truth JSON output path")
    args = parser.parse_args()

    summary = seed_fraud_data(args.org_id, args.out, args.seed)
    counts = summary["counts"]
    print(f"Seeded org {args.org_id}:")
    print(f"  entities:     {counts['entities']}")
    print(f"  accounts:     {counts['accounts']}")
    print(f"  transactions: {counts['transactions']}")
    print(f"  patterns embedded: {len(summary['patterns'])}")
    for p in summary["patterns"]:
        print(f"    - [{p['pattern_type']}] {p['case_id']}: {len(p['accounts'])} accounts, "
              f"{len(p['transactions'])} transactions")
    print(f"\nGround truth written to {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
