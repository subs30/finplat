from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InvestigationState(TypedDict):
    """LangGraph state for one investigation.

    organization_id/account_id/case_id are carried here for bookkeeping
    (so a node can reference "what am I investigating" without re-parsing
    the message history) — NOT as a tenant-isolation boundary. Tool calls
    are scoped to organization_id via the closures in
    app/mcp/server.py, built once per investigation before the graph
    runs; nothing here grants or checks access.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    organization_id: str
    account_id: str
    case_id: str | None
