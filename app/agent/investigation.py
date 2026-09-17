import json
import uuid
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session

from app.agent.checkpointer import get_checkpointer
from app.agent.graph import build_graph
from app.mcp.server import build_mcp_server

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
        tools = await load_mcp_tools(session)
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

    messages = result_state["messages"]
    return InvestigationResult(
        thread_id=resolved_thread_id,
        account_id=str(account_id),
        final_message=_final_text(messages),
        tool_calls=_extract_tool_calls(messages),
        case_id=_extract_case_id(messages),
    )
