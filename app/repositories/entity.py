from app.models.entity import Entity
from app.repositories.base import TenantScopedRepository


class EntityRepository(TenantScopedRepository[Entity]):
    model = Entity

    def bulk_create(self, entities: list[Entity]) -> None:
        self.db.add_all(entities)
        self.db.flush()
