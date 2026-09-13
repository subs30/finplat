from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import pytest
from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError
from pydantic import BaseModel

from app.gateway.providers.groq import GroqProvider, _is_transient_tool_use_glitch

_REQUEST = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


class _Sentiment(BaseModel):
    sentiment: str
    reason: str


def _fake_response(content, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _rate_limit_error():
    return RateLimitError("rate limited", response=httpx2.Response(429, request=_REQUEST), body=None)


def _server_error():
    return InternalServerError(
        "unavailable", response=httpx2.Response(503, request=_REQUEST), body=None
    )


def _bad_request_error():
    return BadRequestError(
        "bad request", response=httpx2.Response(400, request=_REQUEST), body=None
    )


def _tool_use_failed_error():
    """The real error shape Groq returns when a model spontaneously emits a
    native tool-call despite this app never enabling native tool-calling —
    see `_is_transient_tool_use_glitch`'s docstring. `body` here is the
    *unwrapped* inner error object — the `openai` SDK strips the wire-level
    `{"error": {...}}` envelope itself before this point.
    """
    body = {
        "message": "Tool choice is none, but model called a tool",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": '{"name": "some_tool", "arguments": {}}',
    }
    return BadRequestError(
        "tool use failed", response=httpx2.Response(400, request=_REQUEST), body=body
    )


@pytest.fixture
def provider():
    return GroqProvider(
        api_key="test-key", model="openai/gpt-oss-20b", max_retries=2, backoff_seconds=0
    )


def test_generate_plain_text_success(provider):
    provider._client.chat.completions.create = MagicMock(
        return_value=_fake_response("Paris is the capital of France.")
    )

    result = provider.generate("What is the capital of France?")

    assert result.success is True
    assert result.text == "Paris is the capital of France."
    assert result.provider == "groq"
    assert result.model == "openai/gpt-oss-20b"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0
    assert result.error is None


def test_generate_structured_output_success(provider):
    provider._client.chat.completions.create = MagicMock(
        return_value=_fake_response('{"sentiment": "positive", "reason": "Upbeat language."}')
    )

    result = provider.generate("I love this!", response_schema=_Sentiment)

    assert result.success is True
    assert result.parsed == _Sentiment(sentiment="positive", reason="Upbeat language.")


def test_generate_exhausts_retries_on_persistent_server_error(provider):
    provider._client.chat.completions.create = MagicMock(side_effect=_server_error())

    result = provider.generate("hello")

    assert result.success is False
    assert "Groq call failed" in result.error
    assert provider._client.chat.completions.create.call_count == provider._max_retries + 1


def test_generate_does_not_retry_non_rate_limit_client_error(provider):
    provider._client.chat.completions.create = MagicMock(side_effect=_bad_request_error())

    result = provider.generate("hello")

    assert result.success is False
    assert provider._client.chat.completions.create.call_count == 1


def test_generate_retries_rate_limit_error_then_succeeds(provider):
    provider._client.chat.completions.create = MagicMock(
        side_effect=[_rate_limit_error(), _fake_response("ok after retry")]
    )

    result = provider.generate("hello")

    assert result.success is True
    assert result.text == "ok after retry"
    assert provider._client.chat.completions.create.call_count == 2


def test_generate_retries_tool_use_failed_error_then_succeeds(provider):
    """The specific 400 "tool_use_failed" glitch IS retried (unlike other
    400s — see test_generate_does_not_retry_non_rate_limit_client_error)
    because it's driven by sampling randomness, not a malformed request.
    """
    provider._client.chat.completions.create = MagicMock(
        side_effect=[_tool_use_failed_error(), _fake_response("ok after retry")]
    )

    result = provider.generate("hello")

    assert result.success is True
    assert result.text == "ok after retry"
    assert provider._client.chat.completions.create.call_count == 2


def test_generate_exhausts_retries_on_persistent_tool_use_failed_error(provider):
    provider._client.chat.completions.create = MagicMock(side_effect=_tool_use_failed_error())

    result = provider.generate("hello")

    assert result.success is False
    assert "Groq call failed" in result.error
    assert provider._client.chat.completions.create.call_count == provider._max_retries + 1


def test_is_transient_tool_use_glitch_true_for_tool_use_failed_code():
    assert _is_transient_tool_use_glitch(_tool_use_failed_error()) is True


def test_is_transient_tool_use_glitch_false_for_plain_bad_request():
    assert _is_transient_tool_use_glitch(_bad_request_error()) is False


def test_generate_retries_on_network_timeout_then_succeeds(provider):
    provider._client.chat.completions.create = MagicMock(
        side_effect=[
            APIConnectionError(message="timed out", request=_REQUEST),
            _fake_response("ok"),
        ]
    )

    result = provider.generate("hello")

    assert result.success is True
    assert result.text == "ok"


def test_generate_invalid_structured_output_retries_once_then_fails(provider):
    provider._client.chat.completions.create = MagicMock(return_value=_fake_response("not valid json"))

    result = provider.generate("hello", response_schema=_Sentiment)

    assert result.success is False
    assert "invalid structured output" in result.error.lower()
    assert "not valid json" in result.error
    assert provider._client.chat.completions.create.call_count == 2


def test_generate_structured_output_recovers_on_format_fix_retry(provider):
    bad_response = _fake_response("not valid json")
    good_response = _fake_response('{"sentiment": "neutral", "reason": "Factual statement."}')
    provider._client.chat.completions.create = MagicMock(side_effect=[bad_response, good_response])

    result = provider.generate("hello", response_schema=_Sentiment)

    assert result.success is True
    assert result.parsed == _Sentiment(sentiment="neutral", reason="Factual statement.")


def test_response_format_forces_additional_properties_false(provider):
    response_format = provider._response_format(_Sentiment)

    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
