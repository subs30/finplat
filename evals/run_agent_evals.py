#!/usr/bin/env python
"""Golden-case eval for the investigation agent's TRAJECTORY and
human-in-the-loop decision quality — forks aiplat's evals/run_agent_evals.py
pattern (agent trajectory, not just final answer) onto finplat's LangGraph
agent, and adds a property aiplat's agent never had: whether a real
approve/reject decision correctly resumes the paused run to the right
outcome.

Reuses scripts/seed_fraud_data.py's real generator — the exact one
ground_truth.json comes from — to build one ephemeral organization
containing all pattern types (clean, structuring, mule_network, layering),
then for each case picks the account with the case's `account_role` out of
the matching pattern and runs app.agent.investigation.run_investigation()
against it with a REAL Groq call. If the run pauses at
request_human_approval, this makes the decision named by the case's
`decide_when_paused` field directly through the repositories +
resume_investigation_with_decision() — the same sequence
app.routers.approvals.decide_approval does, minus the HTTP/auth layer,
same rationale as aiplat calling run_agent() in-process rather than over
HTTP.

This makes REAL Groq API calls and costs quota — intentionally NOT part
of the pytest suite (see tests/test_approval_interrupt.py and
tests/test_investigations_router.py for the mocked-mechanics and
model-behavior pytest coverage this complements, not duplicates). Run
manually:

    source .venv/bin/activate
    python evals/run_agent_evals.py

Requires GROQ_API_KEY (.env) and a reachable Postgres with the app's full
schema (alembic upgrade head) — including Neo4j if you want
query_relationship_graph to actually resolve (it degrades to an "graph
unavailable" tool result otherwise, which is a normal, handled outcome,
not a crash). Runs against DATABASE_URL directly (not a separate `_test`
database) and deletes everything it creates afterward, in FK-safe order.
"""

import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.investigation import resume_investigation_with_decision, run_investigation
from app.config import get_settings
from app.database import SessionLocal
from app.models.account import Account
from app.models.case import Case
from app.models.case_approval import CaseApproval
from app.models.entity import Entity
from app.models.organization import Organization
from app.models.trace import Trace
from app.models.transaction import Transaction
from app.models.user import User, UserRole
from app.repositories.case import CaseRepository
from app.repositories.case_approval import CaseApprovalRepository
from app.security import hash_password
from evals.scoring import EvalOutcome, score_agent_case, summarize
from scripts.seed_fraud_data import seed_fraud_data

CASES_PATH = Path(__file__).parent / "agent_cases.json"


def _cleanup(db, organization_id: uuid.UUID) -> None:
    """Delete everything this run created, in FK-safe order. Deliberately
    does not touch LangGraph's own checkpoint tables (checkpoints,
    checkpoint_writes, checkpoint_blobs) — they're keyed by thread_id, not
    organization_id, and leaving a handful of finished/abandoned threads
    behind is the same accepted precedent as the pytest suite's own
    checkpoint tests (see the V0.5 report).

    Trace deletion is a V0.7 addition: run_investigation() /
    resume_investigation_with_decision() now write Trace rows (see
    app/agent/investigation.py), which weren't accounted for here before
    — without deleting them first, the Organization delete below would
    fail its foreign key check.
    """
    db.query(Trace).filter(Trace.organization_id == organization_id).delete(synchronize_session=False)
    db.query(CaseApproval).filter(CaseApproval.organization_id == organization_id).delete(
        synchronize_session=False
    )
    db.query(Case).filter(Case.organization_id == organization_id).delete(synchronize_session=False)
    db.query(Transaction).filter(Transaction.organization_id == organization_id).delete(
        synchronize_session=False
    )
    db.query(Account).filter(Account.organization_id == organization_id).delete(synchronize_session=False)
    db.query(Entity).filter(Entity.organization_id == organization_id).delete(synchronize_session=False)
    db.query(User).filter(User.organization_id == organization_id).delete(synchronize_session=False)
    db.query(Organization).filter(Organization.id == organization_id).delete(synchronize_session=False)
    db.commit()


def _find_account_id(summary: dict, pattern_type: str, role: str, index: int) -> uuid.UUID:
    matches = [
        account["account_id"]
        for pattern in summary["patterns"]
        if pattern["pattern_type"] == pattern_type
        for account in pattern["accounts"]
        if account["role"] == role
    ]
    if index >= len(matches):
        raise ValueError(
            f"no account with pattern_type={pattern_type!r} role={role!r} index={index} "
            f"(only {len(matches)} such accounts exist)"
        )
    return uuid.UUID(matches[index])


async def _run_case(
    case: dict, db, organization_id: uuid.UUID, admin_user_id: uuid.UUID, summary: dict
) -> EvalOutcome:
    account_id = _find_account_id(
        summary, case["pattern_type"], case["account_role"], case.get("account_index", 0)
    )
    thread_id = str(uuid.uuid4())

    result = await run_investigation(db, organization_id, account_id, thread_id=thread_id)

    final_result = result
    if result.awaiting_approval:
        decision_word = case.get("decide_when_paused")
        if decision_word is not None:
            approval_repo = CaseApprovalRepository(db, organization_id)
            case_repo = CaseRepository(db, organization_id)
            pending = approval_repo.list_pending()
            approval = next(a for a in pending if a.thread_id == thread_id)
            approved = decision_word == "approve"
            approval_repo.resolve(approval, approved=approved, decided_by_user_id=admin_user_id)
            db_case = case_repo.get(approval.case_id)
            assert db_case is not None  # approval.case_id always points at a case in this same org
            case_repo.clear_pending_approval(db_case)
            db.commit()

            final_result = await resume_investigation_with_decision(
                db,
                organization_id,
                thread_id,
                {"approved": approved, "note": "eval-driven decision", "decided_by_user_id": str(admin_user_id)},
            )

    case_repo = CaseRepository(db, organization_id)
    db_case = case_repo.get_by_thread_id(thread_id)
    final_case_status = db_case.status.value if db_case else None

    return score_agent_case(
        case,
        tool_calls=[tc.tool for tc in final_result.tool_calls],
        awaiting_approval_seen=result.awaiting_approval,
        final_case_status=final_case_status,
        final_message=final_result.final_message,
    )


async def _main_async() -> int:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Add it to your .env — get a free key (no credit card) at "
            "https://console.groq.com/keys — then re-run this script.",
            file=sys.stderr,
        )
        return 1

    cases = json.loads(CASES_PATH.read_text())["cases"]

    db = SessionLocal()
    organization = Organization(id=uuid.uuid4(), name=f"agent-eval-{uuid.uuid4().hex[:8]}")
    db.add(organization)
    db.flush()
    admin = User(
        id=uuid.uuid4(),
        organization_id=organization.id,
        email=f"agent-eval-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=hash_password("eval-only-not-a-real-login"),
        role=UserRole.ADMIN,
    )
    db.add(admin)
    db.commit()

    print("Seeding synthetic fraud dataset (clean/structuring/mule_network/layering)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = seed_fraud_data(organization.id, Path(tmp_dir) / "ground_truth.json", seed=42, db=db)

    try:
        outcomes: list[EvalOutcome] = []
        for case in cases:
            outcome = await _run_case(case, db, organization.id, admin.id, summary)
            outcomes.append(outcome)
            status = "PASS" if outcome.passed else "FAIL"
            print(f"{status}  {outcome.name}: {outcome.detail}")

        passed, total = summarize(outcomes)
        print(f"\n{passed}/{total} passed")
        return 1 if passed < total else 0
    finally:
        _cleanup(db, organization.id)
        db.close()


def main() -> int:
    import asyncio

    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
