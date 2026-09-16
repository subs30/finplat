from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import get_settings
from app.gateway.providers.groq import GROQ_BASE_URL


def get_agent_llm() -> ChatOpenAI:
    """The investigation agent's chat model — LangChain's own client
    pointed at Groq, same base_url/model/key as app.gateway.GroqProvider
    (that class itself is untouched and still serves V0.1/V0.2's
    endpoints; LangGraph needs a langchain_core.BaseChatModel, which
    GroqProvider isn't, so this is a separate, parallel client
    construction reusing the same active provider and settings rather
    than forcing an adapter between two different library interfaces).
    """
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured — the investigation agent needs it.")
    return ChatOpenAI(
        model=settings.GROQ_MODEL,
        api_key=SecretStr(settings.GROQ_API_KEY),
        base_url=GROQ_BASE_URL,
        # ChatOpenAI's default max_retries (2) isn't enough headroom for
        # Groq's free tier: a multi-step investigation's later turns carry
        # growing context, and hitting the per-minute token cap mid-run
        # needs a real cool-down (Groq's own Retry-After has been
        # observed up to ~30s in this project), not two quick retries.
        # Observed directly: two agent investigations back-to-back in the
        # test suite exhausted the 8,000 TPM free-tier budget and failed
        # with only the default retry count.
        max_retries=6,
        timeout=60.0,
    )
