from app.models.user import User
from app.repositories.base import TenantScopedRepository


class UserRepository(TenantScopedRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        return self.db.execute(
            self._scoped().where(User.email == email)
        ).scalar_one_or_none()
