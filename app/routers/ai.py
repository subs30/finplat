from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user, validate_body_before_gateway
from app.gateway.base import GatewayResponse, ModelProvider
from app.gateway.dependency import get_gateway
from app.models.trace import TraceFeature
from app.models.user import User
from app.repositories.trace import TraceRepository
from app.schemas.ai import AskRequest, AskResponse

router = APIRouter(prefix="/ai", tags=["ai"])

_PROMPT_SUMMARY_LENGTH = 200


def _summarize(text: str) -> str:
    if len(text) <= _PROMPT_SUMMARY_LENGTH:
        return text
    return text[:_PROMPT_SUMMARY_LENGTH] + "…"


# The gateway itself never touches the database — it's a pure function of
# (prompt, schema) -> GatewayResponse. The calling code here writes the
# trace *after* the call returns, using the already-computed
# GatewayResponse, rather than threading a trace-writer callback into
# ModelProvider.generate() — a callback would tie the gateway's retry loop
# to a live DB session/transaction for no benefit, since we only ever want
# one trace row per logical call regardless of how many retries happened
# inside it.
@router.post("/ask", response_model=AskResponse)
def ask(
    # Validated as its own Depends(), listed first, so a malformed body is
    # rejected with 422 before Depends(get_gateway) below ever runs (which
    # would otherwise raise 503 first when GROQ_API_KEY is unset) — see
    # validate_body_before_gateway's docstring.
    payload: AskRequest = Depends(validate_body_before_gateway(AskRequest)),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    gateway: ModelProvider = Depends(get_gateway),
) -> AskResponse:
    result: GatewayResponse = gateway.generate(payload.question)

    traces = TraceRepository(db, current_user.organization_id)
    trace = traces.create(
        user_id=current_user.id,
        provider=result.provider,
        model=result.model,
        prompt_summary=_summarize(payload.question),
        success=result.success,
        error=result.error,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        feature=TraceFeature.AI_ASK.value,
    )
    db.commit()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.error or "Model provider call failed",
        )
    return AskResponse(answer=result.text or "", trace_id=trace.id)
