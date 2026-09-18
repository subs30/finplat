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
    # True when the agent is genuinely paused inside request_human_approval
    # (a real LangGraph interrupt(), not finished) — see
    # app.agent.investigation and GET /approvals/pending /
    # POST /approvals/{id}/decide, which is how it resumes.
    awaiting_approval: bool = False
    approval_id: uuid.UUID | None = None
