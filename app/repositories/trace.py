import uuid

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
        )
        self.db.add(trace)
        self.db.flush()
        return trace
