from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors
from pydantic import BaseModel

from app.gateway.providers.gemini import GeminiProvider


class _Sentiment(BaseModel):
    sentiment: str
    reason: str


def _fake_response(text, parsed=None, prompt_tokens=10, output_tokens=5):
    usage = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=output_tokens)
    return SimpleNamespace(text=text, parsed=parsed, usage_metadata=usage)


@pytest.fixture
def provider():
    return GeminiProvider(
        api_key="test-key", model="gemini-flash-lite-latest", max_retries=2, backoff_seconds=0
    )


def test_generate_plain_text_success(provider):
    provider._client.models.generate_content = MagicMock(
        return_value=_fake_response("Paris is the capital of France.")
    )

    result = provider.generate("What is the capital of France?")

    assert result.success is True
    assert result.text == "Paris is the capital of France."
    assert result.provider == "gemini"
    assert result.model == "gemini-flash-lite-latest"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0
    assert result.error is None


def test_generate_structured_output_success(provider):
    parsed = _Sentiment(sentiment="positive", reason="Upbeat language.")
    provider._client.models.generate_content = MagicMock(
        return_value=_fake_response(
            '{"sentiment": "positive", "reason": "Upbeat language."}', parsed=parsed
        )
    )

    result = provider.generate("I love this!", response_schema=_Sentiment)

    assert result.success is True
    assert result.parsed == parsed


def test_generate_exhausts_retries_on_persistent_server_error(provider):
    provider._client.models.generate_content = MagicMock(
        side_effect=errors.ServerError(503, {"message": "unavailable"})
    )

    result = provider.generate("hello")

    assert result.success is False
    assert "Gemini call failed" in result.error
    assert provider._client.models.generate_content.call_count == provider._max_retries + 1


def test_generate_does_not_retry_non_rate_limit_client_error(provider):
    provider._client.models.generate_content = MagicMock(
        side_effect=errors.ClientError(400, {"message": "bad request"})
    )

    result = provider.generate("hello")

    assert result.success is False
    assert provider._client.models.generate_content.call_count == 1


def test_generate_retries_rate_limit_error_then_succeeds(provider):
    provider._client.models.generate_content = MagicMock(
        side_effect=[
            errors.ClientError(429, {"message": "rate limited"}),
            _fake_response("ok after retry"),
        ]
    )

    result = provider.generate("hello")

    assert result.success is True
    assert result.text == "ok after retry"
    assert provider._client.models.generate_content.call_count == 2


def test_generate_retries_on_network_timeout_then_succeeds(provider):
    provider._client.models.generate_content = MagicMock(
        side_effect=[httpx.TimeoutException("timed out"), _fake_response("ok")]
    )

    result = provider.generate("hello")

    assert result.success is True
    assert result.text == "ok"


def test_generate_invalid_structured_output_retries_once_then_fails(provider):
    bad_response = _fake_response("not valid json", parsed=None)
    provider._client.models.generate_content = MagicMock(return_value=bad_response)

    result = provider.generate("hello", response_schema=_Sentiment)

    assert result.success is False
    assert "invalid structured output" in result.error.lower()
    assert "not valid json" in result.error
    assert provider._client.models.generate_content.call_count == 2


def test_generate_structured_output_recovers_on_format_fix_retry(provider):
    bad_response = _fake_response("not valid json", parsed=None)
    good_parsed = _Sentiment(sentiment="neutral", reason="Factual statement.")
    good_response = _fake_response('{"sentiment": "neutral", "reason": "..."}', parsed=good_parsed)
    provider._client.models.generate_content = MagicMock(side_effect=[bad_response, good_response])

    result = provider.generate("hello", response_schema=_Sentiment)

    assert result.success is True
    assert result.parsed == good_parsed
