import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.detection.combined import get_account_detection
from app.models.user import User
from app.repositories.account import AccountRepository
from app.schemas.detection import AccountDetectionResponse, DetectionFinding

router = APIRouter(prefix="/detection", tags=["detection"])


@router.get("/accounts/{account_id}", response_model=AccountDetectionResponse)
def get_account_detection_endpoint(
    account_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AccountDetectionResponse:
    # Tenant isolation: AccountRepository.get() (TenantScopedRepository)
    # filters by this caller's organization_id AND the given id — an
    # account belonging to another organization is indistinguishable from
    # a nonexistent one, same 404-not-403 rule as every other resource
    # lookup in this app.
    account_repo = AccountRepository(db, current_user.organization_id)
    account = account_repo.get(account_id)
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")

    result = get_account_detection(db, current_user.organization_id, account_id)

    return AccountDetectionResponse(
        account_id=account_id,
        flagged_by=result.flagged_by,
        findings=[
            DetectionFinding(
                method=f.method,
                explanation=f.explanation,
                score=f.score,
                evidence_transaction_ids=f.evidence_transaction_ids,
            )
            for f in result.findings
        ],
        unavailable_methods=result.unavailable_methods,
    )
