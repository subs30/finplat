"""Pure scoring functions for finplat's eval harness — no I/O, no model
calls, no database. Kept separate from the runner scripts (run_rag_evals.py,
run_agent_evals.py) specifically so tests/test_eval_scoring.py can exercise
this logic directly and cheaply, without loading a real embedding model or
spending Groq quota. See the V0.6 design report for why the runners
themselves stay standalone scripts instead of pytest tests.
"""

from dataclasses import dataclass


@dataclass
class EvalOutcome:
    name: str
    passed: bool
    detail: str


def score_rag_case(case: dict, result_filenames: list[str], *, top_k: int) -> EvalOutcome:
    """`result_filenames` is search_typology's results, in rank order
    (index 0 = best match). Scores whether `case["expected_filename"]`
    appears within the first `top_k` results — but reports the actual
    rank it was found at (or that it wasn't found at all) as a graded
    diagnostic, not just a hit/miss bit, since retrieval quality is a
    matter of degree even when the pass/fail threshold is binary.
    """
    expected = case["expected_filename"]
    if expected in result_filenames:
        rank = result_filenames.index(expected) + 1
        passed = rank <= top_k
        detail = f"{expected!r} found at rank {rank} (threshold: top-{top_k}); results={result_filenames}"
    else:
        passed = False
        detail = f"{expected!r} not found among {len(result_filenames)} results={result_filenames}"
    return EvalOutcome(name=case["name"], passed=passed, detail=detail)


def score_agent_case(
    case: dict,
    *,
    tool_calls: list[str],
    awaiting_approval_seen: bool,
    final_case_status: str | None,
    final_message: str,
) -> EvalOutcome:
    """`tool_calls` is the full, ordered list of tool names the agent
    invoked across the run (pre- and post-resume, if it paused).
    `awaiting_approval_seen` is whether request_human_approval was ever
    called. `final_case_status` is the Case's status after the run (and
    after resuming, if applicable) reached its terminal point.
    """
    failures: list[str] = []

    expected_tools = case.get("expect_tools_called", [])
    if case.get("expect_tools_called_exact", False):
        if tool_calls != expected_tools:
            failures.append(f"tool calls: expected exactly {expected_tools}, got {tool_calls}")
    elif not set(expected_tools) <= set(tool_calls):
        failures.append(f"tool calls: expected at least {expected_tools}, got {tool_calls}")

    expect_approval = case.get("expect_approval_requested")
    if expect_approval is not None and awaiting_approval_seen != expect_approval:
        failures.append(
            f"expected approval-requested={expect_approval}, got {awaiting_approval_seen}"
        )

    expected_status = case.get("expect_final_case_status")
    if expected_status is not None and final_case_status != expected_status:
        failures.append(f"expected final case status={expected_status!r}, got {final_case_status!r}")

    must_contain_any = case.get("answer_must_contain_any")
    if must_contain_any:
        lower = final_message.lower()
        if not any(needle.lower() in lower for needle in must_contain_any):
            failures.append(f"final message contains none of {must_contain_any}: {final_message!r}")

    detail = (
        f"tools={tool_calls} approval_requested={awaiting_approval_seen} "
        f"final_status={final_case_status} message={final_message!r}"
    )
    if failures:
        detail += " | FAILURES: " + "; ".join(failures)
    return EvalOutcome(name=case["name"], passed=not failures, detail=detail)


def summarize(outcomes: list[EvalOutcome]) -> tuple[int, int]:
    """Returns (passed_count, total_count) — the shared aggregate both
    runners print at the end, matching aiplat's "N/M passed" convention.
    """
    passed = sum(1 for o in outcomes if o.passed)
    return passed, len(outcomes)
