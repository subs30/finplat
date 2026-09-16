import uuid

from pydantic import BaseModel


class DetectionFinding(BaseModel):
    """One detection method's verdict — mirrors app.detection.base.Finding,
    minus `account_id` (redundant on a per-account response) and
    `triggered` (only triggered findings are ever included; see
    AccountDetectionResponse's flagged_by for the "did anything fire" view).
    """

    method: str
    explanation: str
    score: float | None
    evidence_transaction_ids: list[uuid.UUID]


class AccountDetectionResponse(BaseModel):
    account_id: uuid.UUID
    flagged_by: list[str]
    findings: list[DetectionFinding]
    unavailable_methods: list[str]
