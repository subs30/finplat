from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class GatewayResponse(Generic[T]):
    """Provider-agnostic result of a single gateway call.

    Callers (routers, evals) only ever see this shape — never a provider
    SDK's own response type — so a second provider can be added later
    without touching any calling code.
    """

    success: bool
    provider: str
    model: str
    latency_ms: int
    text: str | None = None
    parsed: T | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class ModelProvider(ABC):
    """Generic interface every gateway provider adapter implements.

    Business logic (routers, evals) depends only on this interface, never
    on a provider SDK directly — that's what lets a second provider be
    added later as a new class implementing this interface, with zero
    changes to calling code.
    """

    name: str

    @abstractmethod
    def generate(
        self, prompt: str, *, response_schema: type[T] | None = None
    ) -> GatewayResponse[T]:
        """Generate a response to `prompt`.

        If `response_schema` (a Pydantic model type) is given, the provider
        must request structured output and return it parsed into that type
        via GatewayResponse.parsed. On any failure (network, provider error,
        or unparseable structured output), return success=False with a
        human-readable `error` instead of raising.
        """
        raise NotImplementedError
