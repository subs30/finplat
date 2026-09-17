import uuid

from app.models.case import Case, CaseStatus
from app.repositories.base import TenantScopedRepository


class CaseRepository(TenantScopedRepository[Case]):
    model = Case

    def create(
        self,
        *,
        account_id: uuid.UUID,
        title: str,
        thread_id: str,
        opened_by_user_id: uuid.UUID | None = None,
        findings_summary: str | None = None,
    ) -> Case:
        # organization_id is stamped from the repository's own scope, not a
        # caller-supplied argument — same rule as every other repository's
        # create() in this project.
        case = Case(
            organization_id=self.organization_id,
            account_id=account_id,
            opened_by_user_id=opened_by_user_id,
            title=title,
            findings_summary=findings_summary,
            thread_id=thread_id,
            status=CaseStatus.OPEN,
        )
        self.db.add(case)
        self.db.flush()
        return case

    def get_by_thread_id(self, thread_id: str) -> Case | None:
        return self.db.execute(self._scoped().where(Case.thread_id == thread_id)).scalar_one_or_none()

    def update_findings(
        self,
        case: Case,
        *,
        findings_summary: str | None = None,
        status: CaseStatus | None = None,
    ) -> Case:
        if findings_summary is not None:
            case.findings_summary = findings_summary
        if status is not None:
            case.status = status
        self.db.flush()
        return case

    def set_pending_approval(self, case: Case, action_description: str) -> Case:
        case.pending_approval_action = action_description
        case.status = CaseStatus.IN_REVIEW
        self.db.flush()
        return case

    def clear_pending_approval(self, case: Case) -> Case:
        # Deliberately leaves case.status untouched: the agent's own
        # subsequent update_case call decides the case's real final state
        # (per SYSTEM_PROMPT, which requires a real tool call to match any
        # claimed status) — this just clears the now-resolved request.
        case.pending_approval_action = None
        self.db.flush()
        return case
