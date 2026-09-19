import uuid
from datetime import datetime

from sqlalchemy import Row, case, func, select

from app.models.trace import Trace
from app.repositories.base import TenantScopedRepository

_PROMPT_SUMMARY_MAX_LENGTH = 500


class TraceRepository(TenantScopedRepository[Trace]):
    model = Trace

    def create(
        self,
        *,
        provider: str,
        model: str,
        prompt_summary: str,
        success: bool,
        latency_ms: int,
        user_id: uuid.UUID | None = None,
        error: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        feature: str | None = None,
        thread_id: str | None = None,
    ) -> Trace:
        # organization_id is stamped from the repository's own scope, not a
        # caller-supplied argument — a caller can't accidentally (or
        # maliciously) write a trace into another tenant's organization.
        trace = Trace(
            organization_id=self.organization_id,
            user_id=user_id,
            provider=provider,
            model=model,
            prompt_summary=prompt_summary[:_PROMPT_SUMMARY_MAX_LENGTH],
            success=success,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            feature=feature,
            thread_id=thread_id,
        )
        self.db.add(trace)
        self.db.flush()
        return trace

    # --- V0.7: cost/latency observability, forked from aiplat's Step 10 --
    #
    # All three queries aggregate at the database level (SQLAlchemy
    # func.count/sum/avg + GROUP BY translate directly to Postgres
    # aggregate functions) rather than loading every matching trace row
    # into Python — stays correct and cheap regardless of row count.
    # Tenant isolation comes from the same organization_id filter every
    # other method on this repository uses.

    def usage_overview(self, start: datetime, end: datetime) -> Row:
        """Org-wide totals for [start, end) — one row, always present (SQL
        aggregates over zero matching rows still return one row, with
        COALESCE'd zeros rather than NULLs).
        """
        success_count = func.coalesce(func.sum(case((Trace.success.is_(True), 1), else_=0)), 0)
        stmt = (
            select(
                func.count(Trace.id).label("total_calls"),
                func.coalesce(func.sum(Trace.input_tokens), 0).label("total_input_tokens"),
                func.coalesce(func.sum(Trace.output_tokens), 0).label("total_output_tokens"),
                func.coalesce(func.avg(Trace.latency_ms), 0).label("average_latency_ms"),
                success_count.label("success_count"),
            )
            .where(Trace.organization_id == self.organization_id)
            .where(Trace.created_at >= start)
            .where(Trace.created_at < end)
        )
        return self.db.execute(stmt).one()

    def usage_breakdown(self, start: datetime, end: datetime) -> list[Row]:
        """Same shape as usage_overview(), grouped by (provider, model) —
        a future non-Groq provider/model combination shows up as its own
        row automatically.
        """
        success_count = func.coalesce(func.sum(case((Trace.success.is_(True), 1), else_=0)), 0)
        stmt = (
            select(
                Trace.provider,
                Trace.model,
                func.count(Trace.id).label("calls"),
                func.coalesce(func.sum(Trace.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(Trace.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.avg(Trace.latency_ms), 0).label("average_latency_ms"),
                success_count.label("success_count"),
            )
            .where(Trace.organization_id == self.organization_id)
            .where(Trace.created_at >= start)
            .where(Trace.created_at < end)
            .group_by(Trace.provider, Trace.model)
            .order_by(func.count(Trace.id).desc())
        )
        return list(self.db.execute(stmt).all())

    def usage_breakdown_by_feature(self, start: datetime, end: datetime) -> list[Row]:
        """Same shape again, grouped by `feature` instead — the dimension
        aiplat's Step 10 never needed (one feature, /ai/extract) but
        finplat does: this is what tells /ai/ask, /rag/ask, and the
        investigation agent apart, which raw provider/model grouping
        can't (they're all "groq" / the same model). Traces written
        before V0.7 have feature=NULL and group into their own row.
        """
        success_count = func.coalesce(func.sum(case((Trace.success.is_(True), 1), else_=0)), 0)
        stmt = (
            select(
                Trace.feature,
                func.count(Trace.id).label("calls"),
                func.coalesce(func.sum(Trace.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(Trace.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.avg(Trace.latency_ms), 0).label("average_latency_ms"),
                success_count.label("success_count"),
            )
            .where(Trace.organization_id == self.organization_id)
            .where(Trace.created_at >= start)
            .where(Trace.created_at < end)
            .group_by(Trace.feature)
            .order_by(func.count(Trace.id).desc())
        )
        return list(self.db.execute(stmt).all())

    def usage_by_thread(self, thread_id: str) -> Row:
        """One investigation's full cost/latency rollup — every trace
        sharing this thread_id, regardless of date (a fresh-start call
        plus one more per human-approval resume). No date range: an
        investigation's traces are already a small, bounded set, and
        "when" isn't the caller's question here, "how much did THIS
        investigation cost" is.
        """
        success_count = func.coalesce(func.sum(case((Trace.success.is_(True), 1), else_=0)), 0)
        stmt = select(
            func.count(Trace.id).label("total_calls"),
            func.coalesce(func.sum(Trace.input_tokens), 0).label("total_input_tokens"),
            func.coalesce(func.sum(Trace.output_tokens), 0).label("total_output_tokens"),
            func.coalesce(func.sum(Trace.latency_ms), 0).label("total_latency_ms"),
            success_count.label("success_count"),
        ).where(Trace.organization_id == self.organization_id, Trace.thread_id == thread_id)
        return self.db.execute(stmt).one()
