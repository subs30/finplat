import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.detection.base import Finding
from app.detection.features import build_feature_table
from app.detection.graph import run_all_graph_detection
from app.detection.ml import load_model, predict
from app.detection.rules import run_all_rules
from app.graph.dependency import get_neo4j_driver


@dataclass
class AccountDetectionResult:
    """Plain result type shared by the REST endpoint
    (app/routers/detection.py) and the run_fraud_model MCP tool
    (app/agent/tools.py) — neither knows about the other; both call
    get_account_detection() and format its result their own way (a
    Pydantic response model for the API, plain text for the agent).
    """

    account_id: uuid.UUID
    findings: list[Finding] = field(default_factory=list)
    unavailable_methods: list[str] = field(default_factory=list)

    @property
    def flagged_by(self) -> list[str]:
        return sorted({f.method for f in self.findings})


def _run_rules(db: Session, organization_id: uuid.UUID, account_id: uuid.UUID) -> list[Finding]:
    # Whole-org scan filtered down to one account, not an account-scoped
    # query — the rules engine (app/detection/rules.py) is written to scan
    # a tenant at once (that's also what Step 8's verification needs), and
    # at this dataset's scale (a few hundred accounts/transactions) a full
    # rescan per lookup is well under a second. A production version
    # handling many more accounts would push the account filter into the
    # SQL query instead; not necessary at this project's scale.
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
    # partial result instead of failing the whole combined lookup.
    driver = get_neo4j_driver()
    return [f for f in run_all_graph_detection(driver, organization_id) if f.account_id == account_id]


def get_account_detection(
    db: Session, organization_id: uuid.UUID, account_id: uuid.UUID
) -> AccountDetectionResult:
    """Runs all three detection methods (rules, ML, graph) against one
    account and combines the results. Each leg degrades independently —
    if Neo4j isn't configured or the model hasn't been trained, that leg
    is listed in `unavailable_methods` rather than failing the whole call.
    """
    findings: list[Finding] = []
    unavailable: list[str] = []

    findings += _run_rules(db, organization_id, account_id)

    try:
        findings += _run_ml(db, organization_id, account_id)
    except FileNotFoundError:
        unavailable.append("ml")

    try:
        findings += _run_graph(organization_id, account_id)
    except HTTPException:
        unavailable.append("graph")

    return AccountDetectionResult(account_id=account_id, findings=findings, unavailable_methods=unavailable)
