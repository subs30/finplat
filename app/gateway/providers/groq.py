import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from app.gateway.base import GatewayResponse, ModelProvider, T

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_FIX_FORMAT_SUFFIX = (
    "\n\nYour previous response could not be parsed as valid JSON matching the "
    "required schema. Return ONLY valid JSON matching the schema, with no "
    "extra commentary or markdown fences."
)


def _is_transient_tool_use_glitch(exc: APIStatusError) -> bool:
    """Identifies one specific, identifiable HTTP 400 from Groq that is
    actually transient, unlike 400s generally (see `_call_with_retries`'s
    docstring for why those aren't retried).

    This app never enables native tool-calling — no `tools`/`tool_choice`
    is ever sent, structured output is requested via `response_format`
    instead. But some Groq models occasionally emit a native tool-call
    anyway, a sampling-time quirk rather than something wrong with the
    request. Groq's API then rejects that response with an error body
    whose `code` is `"tool_use_failed"`. Since the identical request
    typically succeeds on a fresh sample, this specific error code is
    treated as retryable; every other 400 (bad schema, auth, etc.) is not.
    """
    return exc.code == "tool_use_failed"


def _force_additional_properties_false(node: Any) -> None:
    """Groq's strict structured-output mode requires every object in the
    JSON schema to set `additionalProperties: false` (see
    https://console.groq.com/docs/structured-outputs). Pydantic's
    `model_json_schema()` doesn't set this by default, so we patch it in —
    recursively, since a schema can nest objects via `$defs`/`$ref`.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _force_additional_properties_false(value)
    elif isinstance(node, list):
        for item in node:
            _force_additional_properties_false(item)


class GroqProvider(ModelProvider):
    """ModelProvider adapter for Groq, via its OpenAI-compatible API
    (accessed with the `openai` SDK pointed at Groq's base URL).
    """

    name = "groq"

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
        self._client = OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,  # we implement our own retry/backoff policy below
        )

    def generate(
        self, prompt: str, *, response_schema: type[T] | None = None
    ) -> GatewayResponse[T]:
        start = time.monotonic()
        response_format = None
        if response_schema is not None:
            response_format = self._response_format(response_schema)

        response, error = self._call_with_retries(prompt, response_format)
        if error is not None:
            return self._error_response(error, start)
        assert response is not None  # guaranteed by _call_with_retries when error is None

        if response_schema is None:
            return self._success_response(response, start)

        parsed, raw_text = self._parse(response, response_schema)
        if parsed is not None:
            return self._success_response(response, start, parsed=parsed)

        # Structured output didn't parse/validate — retry once with an
        # explicit instruction to fix the format, per spec. Don't silently
        # swallow the bad output: if it still fails, the raw text goes in
        # `error` so the caller can see what the model actually returned.
        retry_response, retry_error = self._call_with_retries(
            prompt + _FIX_FORMAT_SUFFIX, response_format
        )
        if retry_error is not None:
            return self._error_response(
                retry_error, start, raw_text=raw_text, prefix="Invalid structured output; retry failed"
            )
        assert retry_response is not None  # guaranteed by _call_with_retries when retry_error is None

        retry_parsed, retry_raw_text = self._parse(retry_response, response_schema)
        if retry_parsed is not None:
            return self._success_response(retry_response, start, parsed=retry_parsed)

        return GatewayResponse(
            success=False,
            provider=self.name,
            model=self._model,
            latency_ms=self._elapsed_ms(start),
            text=retry_raw_text,
            input_tokens=self._input_tokens(retry_response),
            output_tokens=self._output_tokens(retry_response),
            error=f"Model returned invalid structured output after retry. Raw output: {retry_raw_text!r}",
        )

    def _call_with_retries(
        self, prompt: str, response_format: dict[str, Any] | None
    ) -> "tuple[Any | None, Exception | None]":
        """Retries transient failures (5xx, rate limit, network/timeout,
        and the specific "tool_use_failed" 400 — see
        `_is_transient_tool_use_glitch`) with exponential backoff. Does NOT
        retry other 4xx errors (bad request, auth, etc.) — those are not
        transient and won't succeed on retry.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
            except (RateLimitError, InternalServerError, APIConnectionError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if _is_transient_tool_use_glitch(exc):
                    last_error = exc
                else:
                    return None, exc
            else:
                return response, None

            if attempt < self._max_retries:
                time.sleep(self._backoff_seconds * (2**attempt))

        return None, last_error

    def _response_format(self, response_schema: type[T]) -> dict[str, Any]:
        schema = response_schema.model_json_schema()
        _force_additional_properties_false(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    @staticmethod
    def _parse(response: Any, response_schema: type[T]) -> "tuple[T | None, str | None]":
        raw_text = response.choices[0].message.content
        if not raw_text:
            return None, raw_text
        try:
            return response_schema.model_validate_json(raw_text), raw_text
        except ValueError:
            # Covers both json.JSONDecodeError and pydantic.ValidationError,
            # which is itself a ValueError subclass.
            return None, raw_text

    def _success_response(
        self,
        response: Any,
        start: float,
        parsed: T | None = None,
    ) -> GatewayResponse[T]:
        return GatewayResponse(
            success=True,
            provider=self.name,
            model=self._model,
            latency_ms=self._elapsed_ms(start),
            text=response.choices[0].message.content,
            parsed=parsed,
            input_tokens=self._input_tokens(response),
            output_tokens=self._output_tokens(response),
        )

    def _error_response(
        self,
        error: Exception,
        start: float,
        raw_text: str | None = None,
        prefix: str = "Groq call failed",
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
    def _input_tokens(response: Any) -> int | None:
        usage = response.usage
        return usage.prompt_tokens if usage else None

    @staticmethod
    def _output_tokens(response: Any) -> int | None:
        usage = response.usage
        return usage.completion_tokens if usage else None
