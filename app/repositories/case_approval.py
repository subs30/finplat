import uuid
from datetime import UTC, datetime

from app.models.case_approval import CaseApproval, CaseApprovalStatus
from app.repositories.base import TenantScopedRepository


class CaseApprovalRepository(TenantScopedRepository[CaseApproval]):
    model = CaseApproval

    def create(self, *, case_id: uuid.UUID, thread_id: str, action_description: str) -> CaseApproval:
        approval = CaseApproval(
            organization_id=self.organization_id,
            case_id=case_id,
            thread_id=thread_id,
            action_description=action_description,
            status=CaseApprovalStatus.PENDING,
        )
        self.db.add(approval)
        self.db.flush()
        return approval

    def list_pending(self) -> list[CaseApproval]:
        query = self._scoped().where(CaseApproval.status == CaseApprovalStatus.PENDING)
        return list(self.db.execute(query).scalars().all())

    def resolve(
        self, approval: CaseApproval, *, approved: bool, decided_by_user_id: uuid.UUID
    ) -> CaseApproval:
        approval.status = CaseApprovalStatus.APPROVED if approved else CaseApprovalStatus.REJECTED
        approval.decided_by_user_id = decided_by_user_id
        approval.decided_at = datetime.now(UTC)
        self.db.flush()
        return approval
