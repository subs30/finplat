from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import pytest
from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError
from pydantic import BaseModel

from app.gateway.providers.cerebras import CerebrasProvider

_REQUEST = httpx2.Request("POST", "https://api.cerebras.ai/v1/chat/completions")


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


@pytest.fixture
def provider():
    return CerebrasProvider(
        api_key="test-key", model="gpt-oss-120b", max_retries=2, backoff_seconds=0
    )


def test_generate_plain_text_success(provider):
    provider._client.chat.completions.create = MagicMock(
        return_value=_fake_response("Paris is the capital of France.")
    )

    result = provider.generate("What is the capital of France?")

    assert result.success is True
    assert result.text == "Paris is the capital of France."
    assert result.provider == "cerebras"
    assert result.model == "gpt-oss-120b"
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
    assert "Cerebras call failed" in result.error
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
