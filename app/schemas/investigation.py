import uuid

from pydantic import BaseModel


class InvestigationRequest(BaseModel):
    account_id: uuid.UUID
    # Supplying an existing thread_id resumes that investigation instead
    # of starting a new one — same automatic resume-or-start-fresh logic
    # as app.agent.investigation.run_investigation.
    thread_id: str | None = None


class ToolCallSchema(BaseModel):
    tool: str
    args: dict


class InvestigationResponse(BaseModel):
    thread_id: str
    account_id: uuid.UUID
    case_id: uuid.UUID | None
    final_message: str
    tool_calls: list[ToolCallSchema]
