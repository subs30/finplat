"""Builds a fresh, real MCP server for one investigation.

Per the V0.4 design report: one `FastMCP` instance per investigation, not
a shared long-lived server — every tool function registered here is a
closure over THIS investigation's `organization_id` (and its shared DB
session / lazily-resolved Neo4j driver / embedding provider), which is
structurally why the LLM can never see, supply, or override the tenant
scope: organization_id is never part of any tool's declared JSON schema,
only captured in the enclosing Python scope. The LangGraph agent connects
to this server over the MCP SDK's real in-memory transport (see
app/agent/graph.py) — genuine MCP protocol messages, no OS process
boundary.
"""

import uuid

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm import Session

from app.graph.dependency import get_neo4j_driver
from app.mcp import tools
from app.rag.dependency import get_embedding_provider


def build_mcp_server(db: Session, organization_id: uuid.UUID, thread_id: str) -> FastMCP:
    server = FastMCP("finplat-investigation")

    @server.tool(description="Fetch one transaction's full detail by id.")
    def get_transaction(transaction_id: str) -> dict:
        return tools.get_transaction(db, organization_id, transaction_id)

    @server.tool(
        description=(
            "Fetch an account's owning entity (name, risk rating, KYC status, "
            "contact info) plus its full transaction history."
        )
    )
    def get_customer_history(account_id: str) -> dict:
        return tools.get_customer_history(db, organization_id, account_id)

    @server.tool(
        description=(
            "Search the AML/compliance knowledge base for typology descriptions, "
            "policy guidance, or past case write-ups relevant to a query. "
            "Returns raw retrieved passages with citations — read and reason "
            "over them yourself, this does not summarize."
        )
    )
    def search_typology(query: str, k: int = 4) -> dict:
        try:
            embedding_provider = get_embedding_provider()
        except Exception as exc:  # noqa: BLE001 - report unavailability to the agent, don't crash
            return {"error": f"typology search unavailable: {exc}"}
        return tools.search_typology(db, organization_id, embedding_provider, query, k=k)

    @server.tool(
        description=(
            "Run the combined rules + ML + graph fraud/AML detection engine "
            "against one account. Returns which method(s) flagged it and why."
        )
    )
    def run_fraud_model(account_id: str) -> dict:
        return tools.run_fraud_model(db, organization_id, account_id)

    @server.tool(
        description=(
            "Query the transaction graph for one account's direct counterparties "
            "and whether it participates in a detected layering chain or mule "
            "community — a graph-native view distinct from run_fraud_model's "
            "pre-computed verdict."
        )
    )
    def query_relationship_graph(account_id: str) -> dict:
        try:
            driver = get_neo4j_driver()
        except HTTPException as exc:
            return {"error": f"relationship graph unavailable: {exc.detail}"}
        return tools.query_relationship_graph(db, driver, organization_id, account_id)

    @server.tool(description="Create a new investigation case for an account.")
    def create_case(
        account_id: str,
        title: str,
        opened_by_user_id: str | None = None,
        summary: str | None = None,
    ) -> dict:
        # thread_id is bound to THIS investigation's real LangGraph
        # checkpoint thread — never an LLM-supplied argument. It exists
        # so Case.thread_id genuinely links back to this conversation's
        # full trace; letting the model invent its own string here would
        # silently break that link (found exactly this bug during Step 4
        # development — the model passed a made-up thread_id that matched
        # no real checkpoint).
        return tools.create_case(
            db,
            organization_id,
            account_id,
            title,
            thread_id,
            opened_by_user_id=opened_by_user_id,
            summary=summary,
        )

    @server.tool(
        description="Update an existing case's findings summary and/or status (open/in_review/closed)."
    )
    def update_case(case_id: str, findings_summary: str | None = None, status: str | None = None) -> dict:
        return tools.update_case(db, organization_id, case_id, findings_summary=findings_summary, status=status)

    # request_human_approval is deliberately NOT registered here — see
    # app/agent/approval_tool.py for why. LangGraph's interrupt()
    # (verified empirically in the V0.5 design report) cannot suspend
    # across the MCP request/response boundary, even over this in-memory
    # transport: LangGraph tracks "am I inside a Pregel task" via a
    # contextvar that does not propagate into FastMCP's own dispatch task,
    # so interrupt() called from an MCP tool handler just raises a plain
    # error instead of pausing the graph. request_human_approval is built
    # as a native LangChain tool instead and added to the tools list
    # alongside this server's MCP-derived ones (see
    # app/agent/investigation.py).

    return server
