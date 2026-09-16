import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.detection.base import Finding
from app.detection.features import build_feature_table
from app.detection.graph import run_all_graph_detection
from app.detection.ml import load_model, predict
from app.detection.rules import run_all_rules
from app.graph.dependency import get_neo4j_driver
from app.models.user import User
from app.repositories.account import AccountRepository
from app.schemas.detection import AccountDetectionResponse, DetectionFinding

router = APIRouter(prefix="/detection", tags=["detection"])


def _run_rules(db: Session, organization_id: uuid.UUID, account_id: uuid.UUID) -> list[Finding]:
    # Whole-org scan filtered down to one account, not an account-scoped
    # query — the rules engine (app/detection/rules.py) is written to scan
    # a tenant at once (that's also what Step 8's verification needs), and
    # at this dataset's scale (a few hundred accounts/transactions) a full
    # rescan per lookup is well under a second. A production version
    # handling many more accounts would push the account filter into the
    # SQL query instead; not necessary at V0.3's scale.
    return [f for f in run_all_rules(db, organization_id) if f.account_id == account_id]


def _run_ml(db: Session, organization_id: uuid.UUID, account_id: uuid.UUID) -> list[Finding]:
    model = load_model()  # raises FileNotFoundError if untrained — caller handles it
    features = build_feature_table(db, organization_id, as_of=datetime.now(UTC))
    key = str(account_id)
    if key not in features.index:
        return []
    return predict(model, features.loc[[key]])


def _run_graph(organization_id: uuid.UUID, account_id: uuid.UUID) -> list[Finding]:
    # get_neo4j_driver() raises HTTPException(503) itself when
    # NEO4J_PASSWORD isn't configured — called directly (not as a
    # Depends()) so that failure can be caught here and turned into a
    # partial result instead of failing the whole combined endpoint.
    driver = get_neo4j_driver()
    return [f for f in run_all_graph_detection(driver, organization_id) if f.account_id == account_id]


@router.get("/accounts/{account_id}", response_model=AccountDetectionResponse)
def get_account_detection(
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

    findings: list[Finding] = []
    unavailable: list[str] = []

    findings += _run_rules(db, current_user.organization_id, account_id)

    try:
        findings += _run_ml(db, current_user.organization_id, account_id)
    except FileNotFoundError:
        unavailable.append("ml")

    try:
        findings += _run_graph(current_user.organization_id, account_id)
    except HTTPException:
        unavailable.append("graph")

    return AccountDetectionResponse(
        account_id=account_id,
        flagged_by=sorted({f.method for f in findings}),
        findings=[
            DetectionFinding(
                method=f.method,
                explanation=f.explanation,
                score=f.score,
                evidence_transaction_ids=f.evidence_transaction_ids,
            )
            for f in findings
        ],
        unavailable_methods=unavailable,
    )
