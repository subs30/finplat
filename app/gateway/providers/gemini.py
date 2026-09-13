import time

import httpx
from google import genai
from google.genai import errors, types

from app.gateway.base import GatewayResponse, ModelProvider, T

_FIX_FORMAT_SUFFIX = (
    "\n\nYour previous response could not be parsed as valid JSON matching the "
    "required schema. Return ONLY valid JSON matching the schema, with no "
    "extra commentary or markdown fences."
)


class GeminiProvider(ModelProvider):
    """ModelProvider adapter for Google Gemini, via the google-genai SDK.
    Kept as an inactive reference implementation — not selected by
    app.gateway.dependency.get_gateway.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
    ):
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def generate(
        self, prompt: str, *, response_schema: type[T] | None = None
    ) -> GatewayResponse[T]:
        start = time.monotonic()
        config = None
        if response_schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            )

        response, error = self._call_with_retries(prompt, config)
        if error is not None:
            return self._error_response(error, start)
        assert response is not None  # guaranteed by _call_with_retries when error is None

        if response_schema is None:
            return self._success_response(response, start)

        if isinstance(response.parsed, response_schema):
            return self._success_response(response, start, parsed=response.parsed)

        # Structured output didn't parse/validate — retry once with an
        # explicit instruction to fix the format, per spec. Don't silently
        # swallow the bad output: if it still fails, the raw text goes in
        # `error` so the caller can see what the model actually returned.
        retry_response, retry_error = self._call_with_retries(
            prompt + _FIX_FORMAT_SUFFIX, config
        )
        if retry_error is not None:
            return self._error_response(
                retry_error, start, raw_text=response.text, prefix="Invalid structured output; retry failed"
            )
        assert retry_response is not None  # guaranteed by _call_with_retries when retry_error is None

        if isinstance(retry_response.parsed, response_schema):
            return self._success_response(retry_response, start, parsed=retry_response.parsed)

        latency_ms = self._elapsed_ms(start)
        return GatewayResponse(
            success=False,
            provider=self.name,
            model=self._model,
            latency_ms=latency_ms,
            text=retry_response.text,
            input_tokens=self._input_tokens(retry_response),
            output_tokens=self._output_tokens(retry_response),
            error=f"Model returned invalid structured output after retry. Raw output: {retry_response.text!r}",
        )

    def _call_with_retries(
        self, contents: str, config: "types.GenerateContentConfig | None"
    ) -> "tuple[types.GenerateContentResponse | None, Exception | None]":
        """Retries transient failures (5xx, rate limit, network/timeout) with
        exponential backoff. Does NOT retry other 4xx errors (bad request,
        auth, etc.) — those are not transient and won't succeed on retry.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except errors.ServerError as exc:
                last_error = exc
            except errors.ClientError as exc:
                if exc.code != 429:
                    return None, exc
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            else:
                return response, None

            if attempt < self._max_retries:
                time.sleep(self._backoff_seconds * (2**attempt))

        return None, last_error

    def _success_response(
        self,
        response: "types.GenerateContentResponse",
        start: float,
        parsed: T | None = None,
    ) -> GatewayResponse[T]:
        return GatewayResponse(
            success=True,
            provider=self.name,
            model=self._model,
            latency_ms=self._elapsed_ms(start),
            text=response.text,
            parsed=parsed,
            input_tokens=self._input_tokens(response),
            output_tokens=self._output_tokens(response),
        )

    def _error_response(
        self,
        error: Exception,
        start: float,
        raw_text: str | None = None,
        prefix: str = "Gemini call failed",
    ) -> GatewayResponse[T]:
        detail = f"{prefix}: {error}"
        if raw_text is not None:
            detail += f" (previous raw output: {raw_text!r})"
        return GatewayResponse(
            success=False,
            provider=self.name,
            model=self._model,
            latency_ms=self._elapsed_ms(start),
            error=detail,
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.monotonic() - start) * 1000)

    @staticmethod
    def _input_tokens(response: "types.GenerateContentResponse") -> int | None:
        usage = response.usage_metadata
        return usage.prompt_token_count if usage else None

    @staticmethod
    def _output_tokens(response: "types.GenerateContentResponse") -> int | None:
        usage = response.usage_metadata
        return usage.candidates_token_count if usage else None
