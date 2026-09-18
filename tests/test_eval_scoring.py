from evals.scoring import score_agent_case, score_rag_case, summarize

# Pure-function tests for the eval harness's scoring logic — no real
# embedding model, no real Groq, no database. This is the "test the eval
# harness itself" coverage the V0.6 objective asked for; the runner
# scripts (run_rag_evals.py, run_agent_evals.py) that actually call real
# models/DBs stay out of pytest on purpose — see evals/scoring.py's
# module docstring and the V0.6 design report.


def test_score_rag_case_passes_when_found_within_top_k():
    case = {"name": "x", "expected_filename": "layering.md"}
    outcome = score_rag_case(case, ["layering.md", "other.md"], top_k=3)
    assert outcome.passed is True
    assert "rank 1" in outcome.detail


def test_score_rag_case_fails_when_found_but_past_threshold():
    case = {"name": "x", "expected_filename": "layering.md"}
    outcome = score_rag_case(case, ["a.md", "b.md", "c.md", "layering.md"], top_k=3)
    assert outcome.passed is False
    assert "rank 4" in outcome.detail


def test_score_rag_case_fails_when_not_found_at_all():
    case = {"name": "x", "expected_filename": "layering.md"}
    outcome = score_rag_case(case, ["a.md", "b.md"], top_k=3)
    assert outcome.passed is False
    assert "not found" in outcome.detail


def test_score_agent_case_passes_when_all_expectations_met():
    case = {
        "name": "clean",
        "expect_tools_called": ["run_fraud_model", "create_case"],
        "expect_tools_called_exact": False,
        "expect_approval_requested": False,
        "expect_final_case_status": "closed",
        "answer_must_contain_any": ["no signs"],
    }
    outcome = score_agent_case(
        case,
        tool_calls=["run_fraud_model", "get_customer_history", "create_case", "update_case"],
        awaiting_approval_seen=False,
        final_case_status="closed",
        final_message="Investigation concluded: no signs of financial crime.",
    )
    assert outcome.passed is True


def test_score_agent_case_fails_on_missing_required_tool_call():
    case = {"name": "x", "expect_tools_called": ["request_human_approval"], "expect_tools_called_exact": False}
    outcome = score_agent_case(
        case,
        tool_calls=["run_fraud_model", "create_case", "update_case"],
        awaiting_approval_seen=False,
        final_case_status="closed",
        final_message="Closed.",
    )
    assert outcome.passed is False
    assert "request_human_approval" in outcome.detail


def test_score_agent_case_exact_sequence_mismatch_fails():
    case = {
        "name": "x",
        "expect_tools_called": ["run_fraud_model", "create_case"],
        "expect_tools_called_exact": True,
    }
    outcome = score_agent_case(
        case,
        tool_calls=["create_case", "run_fraud_model"],
        awaiting_approval_seen=False,
        final_case_status="open",
        final_message="",
    )
    assert outcome.passed is False


def test_score_agent_case_fails_when_approval_requested_mismatches_expectation():
    case = {"name": "x", "expect_approval_requested": True}
    outcome = score_agent_case(
        case,
        tool_calls=["run_fraud_model", "create_case", "update_case"],
        awaiting_approval_seen=False,
        final_case_status="closed",
        final_message="Closed without escalation.",
    )
    assert outcome.passed is False
    assert "approval-requested" in outcome.detail


def test_score_agent_case_ignores_unset_expectations():
    """expect_approval_requested / expect_final_case_status absent (or
    None) means "don't assert" — used for finplat's deliberately
    observational cases (see agent_cases.json's layering case).
    """
    case = {"name": "x", "expect_tools_called": ["run_fraud_model"], "expect_tools_called_exact": False}
    outcome = score_agent_case(
        case,
        tool_calls=["run_fraud_model", "request_human_approval"],
        awaiting_approval_seen=True,
        final_case_status="in_review",
        final_message="",
    )
    assert outcome.passed is True


def test_score_agent_case_fails_when_answer_missing_required_content():
    case = {"name": "x", "answer_must_contain_any": ["structuring", "smurfing"]}
    outcome = score_agent_case(
        case,
        tool_calls=[],
        awaiting_approval_seen=False,
        final_case_status="closed",
        final_message="Nothing suspicious found.",
    )
    assert outcome.passed is False


def test_summarize_counts_passed_and_total():
    case_a = {"name": "a", "expect_tools_called": []}
    case_b = {"name": "b", "expect_tools_called": ["x"], "expect_tools_called_exact": False}
    outcomes = [
        score_agent_case(case_a, tool_calls=[], awaiting_approval_seen=False, final_case_status=None, final_message=""),
        score_agent_case(case_b, tool_calls=[], awaiting_approval_seen=False, final_case_status=None, final_message=""),
    ]
    passed, total = summarize(outcomes)
    assert passed == 1
    assert total == 2
