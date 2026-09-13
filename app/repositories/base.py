import uuid
from typing import Generic, TypeVar

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.database import Base

# Bound to Base rather than a Protocol declaring id/organization_id: SQLAlchemy's
# Mapped[...] descriptors aren't visible to plain mypy (no mypy plugin enabled),
# so a Protocol would only add false confidence. Subclasses are still required in
# practice to define both columns — enforced by every real tenant-scoped model.
ModelT = TypeVar("ModelT", bound=Base)


class TenantScopedRepository(Generic[ModelT]):
    """Base class for repositories over any table that has an organization_id
    column (i.e. every tenant-scoped table).

    A repository is constructed with the requester's organization_id once,
    and every query method it exposes filters on that organization_id under
    the hood. This means a developer adding a new tenant-scoped table can
    subclass this repository and get automatic tenant isolation "for free" —
    it is structurally hard to forget the organization_id filter, because
    there is no query path here that skips it.

    Do NOT add a method to this class (or a subclass) that returns rows
    without an organization_id filter. If you need a cross-tenant query
    (e.g. for an internal admin tool), it does not belong in this class —
    make it a separate, clearly-named function so it can't be reached by
    accident.
    """

    model: type[ModelT]

    def __init__(self, db: Session, organization_id: uuid.UUID):
        self.db = db
        self.organization_id = organization_id

    def _scoped(self) -> Select:
        return select(self.model).where(
            self.model.organization_id == self.organization_id  # type: ignore[attr-defined]
        )

    def get(self, id: uuid.UUID) -> ModelT | None:
        return self.db.execute(
            self._scoped().where(self.model.id == id)  # type: ignore[attr-defined]
        ).scalar_one_or_none()

    def list(self) -> list[ModelT]:
        return list(self.db.execute(self._scoped()).scalars().all())
