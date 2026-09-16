from app.models.transaction import Transaction
from app.repositories.base import TenantScopedRepository


class TransactionRepository(TenantScopedRepository[Transaction]):
    model = Transaction

    def bulk_create(self, transactions: list[Transaction]) -> None:
        self.db.add_all(transactions)
        self.db.flush()
