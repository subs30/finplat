import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.investigation import resume_investigation_with_decision
from app.database import get_db
from app.dependencies import get_current_user, require_role
from app.models.case_approval import CaseApprovalStatus
from app.models.user import User, UserRole
from app.repositories.case import CaseRepository
from app.repositories.case_approval import CaseApprovalRepository
from app.schemas.approval import CaseApprovalSchema, DecideApprovalRequest, DecideApprovalResponse
from app.schemas.investigation import InvestigationResponse, ToolCallSchema

router = APIRouter(prefix="/approvals", tags=["approvals"])


@router.get("/pending", response_model=list[CaseApprovalSchema])
def list_pending_approvals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[CaseApprovalSchema]:
    # Read access is deliberately broader than decide access — any
    # authenticated org member can see what's pending, mirroring aiplat's
    # Step 9 precedent (decide is ADMIN-only, read isn't).
    approvals = CaseApprovalRepository(db, current_user.organization_id).list_pending()
    return [CaseApprovalSchema.model_validate(a) for a in approvals]


@router.post("/{approval_id}/decide", response_model=DecideApprovalResponse)
async def decide_approval(
    approval_id: uuid.UUID,
    payload: DecideApprovalRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
) -> DecideApprovalResponse:
    approval_repo = CaseApprovalRepository(db, current_user.organization_id)
    approval = approval_repo.get(approval_id)
    if approval is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval not found")
    if approval.status != CaseApprovalStatus.PENDING:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Approval already decided")

    approval_repo.resolve(approval, approved=payload.approved, decided_by_user_id=current_user.id)
    case_repo = CaseRepository(db, current_user.organization_id)
    case = case_repo.get(approval.case_id)
    assert case is not None  # approval.case_id always points at a case in this same org
    case_repo.clear_pending_approval(case)
    db.commit()

    # The real resume: Command(resume=...) re-enters the exact interrupt()
    # call inside request_human_approval that paused this thread — not a
    # restart. See app/agent/investigation.py and the V0.5 design report.
    result = await resume_investigation_with_decision(
        db,
        current_user.organization_id,
        approval.thread_id,
        {
            "approved": payload.approved,
            "note": payload.note,
            "decided_by_user_id": str(current_user.id),
        },
    )

    return DecideApprovalResponse(
        approval=CaseApprovalSchema.model_validate(approval),
        investigation=InvestigationResponse(
            thread_id=result.thread_id,
            account_id=uuid.UUID(result.account_id),
            case_id=uuid.UUID(result.case_id) if result.case_id else None,
            final_message=result.final_message,
            tool_calls=[ToolCallSchema(tool=tc.tool, args=tc.args) for tc in result.tool_calls],
            awaiting_approval=result.awaiting_approval,
            approval_id=uuid.UUID(result.approval_id) if result.approval_id else None,
        ),
    )
