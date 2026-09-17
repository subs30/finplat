import json
import uuid
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.types import Command
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session

from app.agent.approval_tool import build_request_human_approval_tool
from app.agent.checkpointer import get_checkpointer
from app.agent.graph import build_graph
from app.mcp.server import build_mcp_server
from app.repositories.case import CaseRepository
from app.repositories.case_approval import CaseApprovalRepository

DEFAULT_MAX_STEPS = 10


@dataclass
class ToolCallRecord:
    tool: str
    args: dict


@dataclass
class InvestigationResult:
    thread_id: str
    account_id: str
    final_message: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    case_id: str | None = None
    awaiting_approval: bool = False
    approval_id: str | None = None


def _extract_tool_calls(messages: list) -> list[ToolCallRecord]:
    calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(ToolCallRecord(tool=tc["name"], args=tc["args"]))
    return calls


def _content_to_text(content: object) -> str | None:
    """LangChain message content is either a plain string, or (this is
    what langchain_mcp_adapters produces from MCP's TextContent, and what
    every ToolMessage in this graph actually has) a list of content
    blocks like [{"type": "text", "text": "..."}]. A naive
    isinstance(content, str) check misses the second shape entirely —
    found exactly that bug during Step 4 development: it silently made
    _extract_case_id never match any ToolMessage, ever.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"]
        if texts:
            return "".join(texts)
    return None


def _extract_case_id(messages: list) -> str | None:
    case_id = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name in ("create_case", "update_case"):
            text = _content_to_text(msg.content)
            if text is None:
                continue
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                continue
            found = parsed.get("case_id")
            if found:
                case_id = found
    return case_id


def _final_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = _content_to_text(msg.content)
            if text:
                return text
    return ""


def _extract_interrupt_payload(result_state: dict) -> dict | None:
    """`graph.ainvoke()`'s return dict carries a `__interrupt__` key (a
    tuple of `Interrupt` objects) exactly when the run just paused inside
    request_human_approval's interrupt() call — verified empirically, see
    the V0.5 design report. `None` means the run reached a normal end
    instead (finished, or errored out) without pausing.
    """
    interrupts = result_state.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def _record_pending_approval(db: Session, organization_id: uuid.UUID, thread_id: str, payload: dict) -> str:
    """Runs exactly once, right when a pause is first observed — never
    re-executed on resume, unlike code inside the tool itself — so this is
    where the actual persistence for a new approval request belongs.
    """
    case_repo = CaseRepository(db, organization_id)
    case = case_repo.get(uuid.UUID(payload["case_id"]))
    assert case is not None  # request_human_approval already validated this case exists
    case_repo.set_pending_approval(case, payload["action_description"])
    approval = CaseApprovalRepository(db, organization_id).create(
        case_id=case.id, thread_id=thread_id, action_description=payload["action_description"]
    )
    db.commit()
    return str(approval.id)


def _build_result(
    thread_id: str, account_id: str, result_state: dict, *, approval_id: str | None
) -> InvestigationResult:
    messages = result_state["messages"]
    return InvestigationResult(
        thread_id=thread_id,
        account_id=account_id,
        final_message=_final_text(messages),
        tool_calls=_extract_tool_calls(messages),
        case_id=_extract_case_id(messages),
        awaiting_approval=approval_id is not None,
        approval_id=approval_id,
    )


async def run_investigation(
    db: Session,
    organization_id: uuid.UUID,
    account_id: uuid.UUID,
    *,
    thread_id: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    interrupt_after: list[str] | None = None,
    llm: BaseChatModel | None = None,
) -> InvestigationResult:
    """Runs (or resumes) one investigation.

    Resume is automatic, not a separate function: if `thread_id` names a
    thread the checkpointer already has state for, this continues from
    the last completed step (`graph.ainvoke(None, config)` — LangGraph's
    convention for "continue, no new input"); otherwise it starts fresh
    with an initial HumanMessage. See tests/test_agent_checkpoint.py for
    the explicit proof this actually survives a full teardown of the
    graph/checkpointer objects, not just a re-invoke within the same
    Python objects.

    A fresh MCP server (bound to `organization_id` via closure — see
    app/mcp/server.py) and a fresh checkpointer connection are built on
    EVERY call, resume included — this is deliberate: it's what makes the
    resume test a genuine simulation of "the process restarted," since
    nothing here is held over in Python state between calls, only
    whatever actually made it into Postgres.
    """
    resolved_thread_id = thread_id or str(uuid.uuid4())
    server = build_mcp_server(db, organization_id, resolved_thread_id)

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        mcp_tools = await load_mcp_tools(session)
        # request_human_approval is a native tool, not MCP-derived — see
        # app/agent/approval_tool.py — but the LLM sees it as just another
        # entry in this same list, bound alongside the MCP tools below.
        tools = [*mcp_tools, build_request_human_approval_tool(db, organization_id)]
        async with get_checkpointer() as checkpointer:
            graph = build_graph(tools, checkpointer, interrupt_after=interrupt_after, llm=llm)
            config: RunnableConfig = {
                "configurable": {"thread_id": resolved_thread_id},
                "recursion_limit": max_steps * 2 + 2,
            }

            existing_state = await graph.aget_state(config)
            if existing_state.values.get("messages"):
                result_state = await graph.ainvoke(None, config)
            else:
                initial_state = {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"Investigate account {account_id} in organization "
                                f"{organization_id}. Determine whether it shows signs of "
                                "financial crime, and open a case with your conclusion."
                            )
                        )
                    ],
                    "organization_id": str(organization_id),
                    "account_id": str(account_id),
                    "case_id": None,
                }
                result_state = await graph.ainvoke(initial_state, config)

    approval_id = None
    interrupt_payload = _extract_interrupt_payload(result_state)
    if interrupt_payload is not None:
        approval_id = _record_pending_approval(db, organization_id, resolved_thread_id, interrupt_payload)

    return _build_result(resolved_thread_id, str(account_id), result_state, approval_id=approval_id)


async def resume_investigation_with_decision(
    db: Session,
    organization_id: uuid.UUID,
    thread_id: str,
    decision: dict,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    llm: BaseChatModel | None = None,
) -> InvestigationResult:
    """Resumes a genuinely paused investigation with a human reviewer's
    decision — the real mechanism, not a restart: `Command(resume=...)`
    re-enters request_human_approval's interrupt() call with `decision` as
    its return value, from the exact paused point LangGraph's checkpointer
    recorded. This is a different primitive from the `interrupt_after` /
    `ainvoke(None, ...)` static pause V0.4's tests used — that pauses
    *between* declared nodes; this resumes a dynamic, mid-node interrupt().

    `decision` is passed through verbatim to interrupt()'s caller inside
    the tool: `{"approved": bool, "note": str | None,
    "decided_by_user_id": str}` — see app/agent/approval_tool.py.

    Same fresh-graph-per-call discipline as run_investigation, for the
    same reason: this must work correctly even when the process handling
    the decide request isn't the one that paused.
    """
    case_repo = CaseRepository(db, organization_id)
    case = case_repo.get_by_thread_id(thread_id)
    assert case is not None  # the decide endpoint already resolved this via the approval row

    server = build_mcp_server(db, organization_id, thread_id)
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        mcp_tools = await load_mcp_tools(session)
        tools = [*mcp_tools, build_request_human_approval_tool(db, organization_id)]
        async with get_checkpointer() as checkpointer:
            graph = build_graph(tools, checkpointer, llm=llm)
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": max_steps * 2 + 2,
            }
            result_state = await graph.ainvoke(Command(resume=decision), config)

    approval_id = None
    interrupt_payload = _extract_interrupt_payload(result_state)
    if interrupt_payload is not None:
        approval_id = _record_pending_approval(db, organization_id, thread_id, interrupt_payload)

    return _build_result(thread_id, str(case.account_id), result_state, approval_id=approval_id)
