from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_agent_llm
from app.agent.state import InvestigationState

SYSTEM_PROMPT = """You are a financial crime investigation assistant for a \
bank's compliance team. You have been asked to investigate one account. \
You have tools to: look up a specific transaction, pull an account's full \
customer history, search AML/compliance typology and policy documentation, \
run the bank's combined fraud/AML detection engine (rules + ML + graph) \
against an account, and query the transaction relationship graph for an \
account's direct counterparties and graph-level patterns (layering \
chains, mule-network clusters).

Investigate thoroughly but efficiently: prefer running the fraud model \
and relationship graph query early to see what's already been flagged, \
pull customer history to understand the account's profile, and use \
typology search when you want grounding for what a suspicious pattern \
you're seeing actually means. Do not call the same tool with the same \
arguments more than once.

Once you have enough information to reach a conclusion, create a case \
(create_case) summarizing what you found. Then you MUST bring the case \
to its correct final state before your last message — never just say \
what the outcome is without also making it true via a tool call:
- If you believe a specific action is warranted (e.g. freezing the \
account, escalating to a human reviewer), call request_human_approval \
with a clear description of the recommended action. This also moves the \
case to in_review — you do not need a separate update_case call for that.
- If you conclude the account shows no signs of financial crime and no \
action is needed, call update_case with status="closed" and a findings \
summary. Do not describe the case as closed, resolved, or requiring no \
further action unless you have actually called update_case to set that \
status — your final message must match the case's real status, not \
narrate a status you didn't set.

Give a final plain-language summary of your conclusion as your last \
message once the case is in its correct final state."""


def build_graph(
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph:
    """Assembles the investigation graph — a StateGraph, not the
    create_react_agent prebuilt helper, so InvestigationState's extra
    fields (organization_id/account_id/case_id) have somewhere to live
    and the structure stays inspectable rather than hidden inside a
    helper. See the V0.4 design report for why.

    `tools` here are LangChain BaseTool objects already converted from
    one investigation's MCP session (via langchain_mcp_adapters), and
    `checkpointer` is a real AsyncPostgresSaver — see
    app/agent/checkpointer.py. Both must be supplied by the caller
    (app/agent/investigation.py); this function only wires the graph
    shape, it doesn't know how either was constructed.

    `interrupt_after` is not used by real investigations — it exists so
    tests can force a deterministic pause after a specific node (e.g.
    `["tools"]`, to stop right after the first tool call) to verify
    checkpoint resume without depending on real LLM call timing to
    "catch" a run mid-flight. See tests/test_agent_checkpoint.py.
    """
    llm_with_tools = get_agent_llm().bind_tools(tools)

    async def agent_node(state: InvestigationState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(InvestigationState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge("__start__", "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
