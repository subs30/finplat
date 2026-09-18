import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.case_approval import CaseApprovalStatus
from app.schemas.investigation import InvestigationResponse


class CaseApprovalSchema(BaseModel):
    id: uuid.UUID
    case_id: uuid.UUID
    action_description: str
    status: CaseApprovalStatus
    created_at: datetime
    decided_by_user_id: uuid.UUID | None
    decided_at: datetime | None

    model_config = {"from_attributes": True}


class DecideApprovalRequest(BaseModel):
    approved: bool
    note: str | None = None


class DecideApprovalResponse(BaseModel):
    approval: CaseApprovalSchema
    investigation: InvestigationResponse
