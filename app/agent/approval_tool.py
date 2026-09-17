"""The human-in-the-loop approval tool — deliberately a native LangChain
tool, not an MCP tool.

Per the V0.5 design report: LangGraph's `interrupt()` genuinely suspends
graph execution only when called from code LangGraph's own runner directly
awaits. An MCP tool handler runs inside FastMCP's own dispatch task instead
— verified empirically, calling interrupt() from there just raises "Called
get_config outside of a runnable context", which FastMCP turns into an
ordinary tool error instead of a pause. So this tool is built here and
added directly to the tools list in app.agent.investigation, alongside
(not through) the MCP-derived tools from app.mcp.server.

Closes over db/organization_id the same way every MCP tool does, for the
same reason: organization_id must never be an LLM-visible argument.

This tool does no database writes of its own. Verified empirically (see
the design report) that any code before an interrupt() call re-executes
in full when the graph resumes — so all persistence (recording the
pending CaseApproval when the pause is first observed, and resolving it
once a decision is made) is handled by the one-time call sites in
app.agent.investigation instead, which never re-run. Everything in this
function before interrupt() is a pure read, safe to repeat.
"""

import json
import uuid

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt
from sqlalchemy.orm import Session

from app.repositories.case import CaseRepository


def build_request_human_approval_tool(db: Session, organization_id: uuid.UUID) -> BaseTool:
    @tool
    def request_human_approval(case_id: str, action_description: str) -> str:
        """Pause the investigation and request a human reviewer's decision
        on a recommended action for a case (e.g. freezing an account,
        escalating to a human reviewer). Moves the case to in_review.
        Returns the reviewer's actual decision once made — never assume
        approval; act only on what this returns.
        """
        try:
            parsed_case_id = uuid.UUID(case_id)
        except ValueError:
            return json.dumps({"error": f"'{case_id}' is not a valid case id"})

        case_repo = CaseRepository(db, organization_id)
        case = case_repo.get(parsed_case_id)
        if case is None:
            return json.dumps({"error": f"Case {case_id} not found"})

        decision = interrupt({"case_id": case_id, "action_description": action_description})

        if decision.get("approved"):
            return json.dumps(
                {"case_id": case_id, "decision": "approved", "note": decision.get("note", "")}
            )
        return json.dumps(
            {"case_id": case_id, "decision": "rejected", "note": decision.get("note", "")}
        )

    return request_human_approval
