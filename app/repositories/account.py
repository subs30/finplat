from app.models.account import Account
from app.repositories.base import TenantScopedRepository


class AccountRepository(TenantScopedRepository[Account]):
    model = Account

    def bulk_create(self, accounts: list[Account]) -> None:
        self.db.add_all(accounts)
        self.db.flush()
