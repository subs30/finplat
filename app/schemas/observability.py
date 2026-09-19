from datetime import date

from pydantic import BaseModel


class ProviderModelUsage(BaseModel):
    """One (provider, model) group's totals within the requested date
    range — see app.repositories.trace.TraceRepository.usage_breakdown().
    """

    provider: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    average_latency_ms: float
    success_rate: float


class FeatureUsage(BaseModel):
    """One `feature`'s totals within the requested date range — the
    dimension that actually tells /ai/ask, /rag/ask, and the
    investigation agent apart (they're all "groq" / the same model, so
    ProviderModelUsage alone can't). `feature` is null for any trace
    written before V0.7, which groups into its own row rather than being
    dropped.
    """

    feature: str | None
    calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    average_latency_ms: float
    success_rate: float


class UsageResponse(BaseModel):
    start_date: date
    end_date: date
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    # Illustrative only — see app.config.Settings.ILLUSTRATIVE_COST_PER_1K_TOKENS_USD
    # and this field's rate below, which callers should display alongside
    # the figure so the caveat travels with the number, not just a code
    # comment.
    estimated_cost_usd: float
    illustrative_cost_per_1k_tokens_usd: float
    average_latency_ms: float
    success_rate: float
    breakdown: list[ProviderModelUsage]
    breakdown_by_feature: list[FeatureUsage]


class InvestigationUsageResponse(BaseModel):
    """One investigation's full cost/latency rollup — every trace sharing
    its thread_id, typically one for the fresh start plus one more per
    human-approval resume. See
    app.repositories.trace.TraceRepository.usage_by_thread().
    """

    thread_id: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    illustrative_cost_per_1k_tokens_usd: float
    total_latency_ms: int
    success_rate: float
