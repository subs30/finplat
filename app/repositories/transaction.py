import uuid

from sqlalchemy import or_

from app.models.transaction import Transaction
from app.repositories.base import TenantScopedRepository


class TransactionRepository(TenantScopedRepository[Transaction]):
    model = Transaction

    def bulk_create(self, transactions: list[Transaction]) -> None:
        self.db.add_all(transactions)
        self.db.flush()

    def list_for_account(self, account_id: uuid.UUID) -> list[Transaction]:
        """Every transaction where `account_id` is either side — pushed
        into the WHERE clause rather than fetching the whole tenant's
        transactions and filtering in Python, since this backs
        get_customer_history, a per-account lookup a real deployment could
        call often.
        """
        query = self._scoped().where(
            or_(Transaction.sender_account_id == account_id, Transaction.receiver_account_id == account_id)
        ).order_by(Transaction.occurred_at)
        return list(self.db.execute(query).scalars().all())
