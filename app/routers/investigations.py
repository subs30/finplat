import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.investigation import run_investigation
from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.repositories.account import AccountRepository
from app.schemas.investigation import InvestigationRequest, InvestigationResponse, ToolCallSchema

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.post("", response_model=InvestigationResponse, status_code=status.HTTP_201_CREATED)
async def create_investigation(
    payload: InvestigationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> InvestigationResponse:
    # Tenant isolation: same 404-not-403 rule as every other resource
    # lookup — an account belonging to another organization is
    # indistinguishable from a nonexistent one. This check is on top of
    # (not instead of) the deeper guarantee: every MCP tool call inside
    # run_investigation is scoped to current_user.organization_id via a
    # closure the LLM can't see or override (see app/mcp/server.py) — this
    # endpoint check just gives a clean 404 before spending any LLM calls
    # on an account the caller shouldn't even know exists.
    account_repo = AccountRepository(db, current_user.organization_id)
    account = account_repo.get(payload.account_id)
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")

    result = await run_investigation(
        db, current_user.organization_id, payload.account_id, thread_id=payload.thread_id
    )

    return InvestigationResponse(
        thread_id=result.thread_id,
        account_id=uuid.UUID(result.account_id),
        case_id=uuid.UUID(result.case_id) if result.case_id else None,
        final_message=result.final_message,
        tool_calls=[ToolCallSchema(tool=tc.tool, args=tc.args) for tc in result.tool_calls],
    )
