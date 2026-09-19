from datetime import UTC, date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.repositories.trace import TraceRepository
from app.schemas.observability import (
    FeatureUsage,
    InvestigationUsageResponse,
    ProviderModelUsage,
    UsageResponse,
)

router = APIRouter(prefix="/observability", tags=["observability"])

_DEFAULT_WINDOW_DAYS = 30


def _cost_usd(input_tokens: int, output_tokens: int, rate_per_1k: float) -> float:
    return round((input_tokens + output_tokens) / 1000 * rate_per_1k, 6)


def _success_rate(success_count: int, calls: int) -> float:
    return round(success_count / calls * 100, 2) if calls else 0.0


@router.get("/usage", response_model=UsageResponse)
def get_usage(
    start_date: date | None = Query(
        None, description="Inclusive start of the range (ISO date). Defaults to 30 days before end_date."
    ),
    end_date: date | None = Query(
        None, description="Inclusive end of the range (ISO date). Defaults to today (UTC)."
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UsageResponse:
    resolved_end = end_date or datetime.now(UTC).date()
    resolved_start = start_date or resolved_end - timedelta(days=_DEFAULT_WINDOW_DAYS)
    if resolved_start > resolved_end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="start_date must not be after end_date",
        )

    # Trace.created_at is timezone-aware UTC; both bounds are dates, so the
    # query range is the half-open [start_date 00:00 UTC, end_date+1 00:00
    # UTC) — i.e. end_date is inclusive from the caller's point of view.
    start_dt = datetime.combine(resolved_start, time.min, tzinfo=UTC)
    end_dt = datetime.combine(resolved_end + timedelta(days=1), time.min, tzinfo=UTC)

    traces = TraceRepository(db, current_user.organization_id)
    overview = traces.usage_overview(start_dt, end_dt)
    breakdown_rows = traces.usage_breakdown(start_dt, end_dt)
    feature_rows = traces.usage_breakdown_by_feature(start_dt, end_dt)

    cost_rate = get_settings().ILLUSTRATIVE_COST_PER_1K_TOKENS_USD

    breakdown = [
        ProviderModelUsage(
            provider=row.provider,
            model=row.model,
            calls=row.calls,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            estimated_cost_usd=_cost_usd(row.input_tokens, row.output_tokens, cost_rate),
            average_latency_ms=float(row.average_latency_ms),
            success_rate=_success_rate(row.success_count, row.calls),
        )
        for row in breakdown_rows
    ]

    breakdown_by_feature = [
        FeatureUsage(
            feature=row.feature,
            calls=row.calls,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            estimated_cost_usd=_cost_usd(row.input_tokens, row.output_tokens, cost_rate),
            average_latency_ms=float(row.average_latency_ms),
            success_rate=_success_rate(row.success_count, row.calls),
        )
        for row in feature_rows
    ]

    return UsageResponse(
        start_date=resolved_start,
        end_date=resolved_end,
        total_calls=overview.total_calls,
        total_input_tokens=overview.total_input_tokens,
        total_output_tokens=overview.total_output_tokens,
        estimated_cost_usd=_cost_usd(
            overview.total_input_tokens, overview.total_output_tokens, cost_rate
        ),
        illustrative_cost_per_1k_tokens_usd=cost_rate,
        average_latency_ms=float(overview.average_latency_ms),
        success_rate=_success_rate(overview.success_count, overview.total_calls),
        breakdown=breakdown,
        breakdown_by_feature=breakdown_by_feature,
    )


@router.get("/investigations/{thread_id}/usage", response_model=InvestigationUsageResponse)
def get_investigation_usage(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> InvestigationUsageResponse:
    # No existence check against Case/thread_id beyond the org filter
    # already inside usage_by_thread(): a thread_id belonging to another
    # organization (or one that never existed) simply has zero matching
    # traces in this org's scope — indistinguishable from "not found",
    # same 404-shaped-as-empty-result rule as everywhere else, without
    # needing a separate lookup.
    traces = TraceRepository(db, current_user.organization_id)
    row = traces.usage_by_thread(thread_id)

    cost_rate = get_settings().ILLUSTRATIVE_COST_PER_1K_TOKENS_USD
    return InvestigationUsageResponse(
        thread_id=thread_id,
        total_calls=row.total_calls,
        total_input_tokens=row.total_input_tokens,
        total_output_tokens=row.total_output_tokens,
        estimated_cost_usd=_cost_usd(row.total_input_tokens, row.total_output_tokens, cost_rate),
        illustrative_cost_per_1k_tokens_usd=cost_rate,
        total_latency_ms=row.total_latency_ms,
        success_rate=_success_rate(row.success_count, row.total_calls),
    )
