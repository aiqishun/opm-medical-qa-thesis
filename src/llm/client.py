"""Thin, swappable interface to the language model.

Agents talk to an ``LLMClient``; they never import the OpenAI SDK. Tests use a
fake client and never touch the network. The real client wraps the OpenAI
Responses API with JSON-schema structured output.

Retry policy (thesis protocol): a call is retried at most once, and only after
a network error or a rate-limit error. A response whose content is unwanted or
unparseable is never retried. The SDK's own automatic retries are disabled so
that every attempt is visible in the record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    """Everything needed for one structured-output call."""

    model: str
    reasoning_effort: str
    max_output_tokens: int
    system_prompt: str
    user_prompt: str
    schema_name: str
    schema: dict[str, Any]
    strict_schema: bool = True

    def api_parameters(self) -> dict[str, Any]:
        """Non-content request parameters, as recorded in every call record."""
        return {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "text_format": {"type": "json_schema", "name": self.schema_name, "strict": self.strict_schema},
            "store": False,
        }


@dataclass
class LLMResponse:
    """What came back, plus the accounting the thesis reports."""

    ok: bool                         # False if the API call itself failed (after the retry budget)
    raw_text: str | None             # model output text, exactly as returned
    response_model: str | None       # model version reported by the API
    response_id: str | None
    status: str | None               # "completed", "incomplete", ... or None on API error
    incomplete_reason: str | None
    input_tokens: int | None
    output_tokens: int | None        # includes reasoning tokens (API convention)
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    latency_s: float                 # wall clock of the successful (or last) attempt
    total_latency_s: float           # wall clock over all attempts
    attempts: int
    error: dict[str, Any] | None = None
    attempt_errors: list[dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


def _error_dict(exc: BaseException) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    status = getattr(exc, "status_code", None)
    if status is not None:
        info["status_code"] = status
    return info


class OpenAIResponsesClient:
    """``LLMClient`` backed by the OpenAI Responses API.

    The API key is read by the SDK from the ``OPENAI_API_KEY`` environment
    variable only; this class never receives, stores or logs it.
    """

    MAX_ATTEMPTS = 2  # first try + at most one retry on network/rate-limit errors

    def __init__(self, *, timeout_s: float = 600.0, retry_wait_s: float = 20.0) -> None:
        import os

        import openai

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment")
        self._openai = openai
        self._client = openai.OpenAI(max_retries=0, timeout=timeout_s)
        self._retry_wait_s = retry_wait_s
        self._retryable = (openai.APIConnectionError, openai.RateLimitError)  # APITimeoutError ⊂ APIConnectionError

    def complete(self, request: LLMRequest) -> LLMResponse:
        attempt_errors: list[dict[str, Any]] = []
        started_all = time.monotonic()
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                response = self._client.responses.create(
                    model=request.model,
                    reasoning={"effort": request.reasoning_effort},
                    max_output_tokens=request.max_output_tokens,
                    input=[
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    text={"format": {"type": "json_schema", "name": request.schema_name,
                                     "schema": request.schema, "strict": request.strict_schema}},
                    store=False,
                )
            except self._retryable as exc:
                attempt_errors.append({"attempt": attempt, **_error_dict(exc)})
                if attempt < self.MAX_ATTEMPTS:
                    time.sleep(self._retry_wait_s)
                    continue
                return self._failed(exc, started, started_all, attempt, attempt_errors)
            except self._openai.APIError as exc:  # not retryable: bad request, auth, server error ...
                attempt_errors.append({"attempt": attempt, **_error_dict(exc)})
                return self._failed(exc, started, started_all, attempt, attempt_errors)
            latency = time.monotonic() - started
            return self._success(response, latency, time.monotonic() - started_all, attempt, attempt_errors)
        raise AssertionError("unreachable")

    @staticmethod
    def _failed(exc, started, started_all, attempt, attempt_errors) -> LLMResponse:
        now = time.monotonic()
        return LLMResponse(ok=False, raw_text=None, response_model=None, response_id=None, status=None,
                           incomplete_reason=None, input_tokens=None, output_tokens=None,
                           reasoning_tokens=None, cached_input_tokens=None, latency_s=now - started,
                           total_latency_s=now - started_all, attempts=attempt, error=_error_dict(exc),
                           attempt_errors=attempt_errors)

    @staticmethod
    def _success(response, latency, total_latency, attempt, attempt_errors) -> LLMResponse:
        usage = getattr(response, "usage", None)
        out_details = getattr(usage, "output_tokens_details", None) if usage else None
        in_details = getattr(usage, "input_tokens_details", None) if usage else None
        incomplete = getattr(response, "incomplete_details", None)
        return LLMResponse(
            ok=True,
            raw_text=response.output_text,
            response_model=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            incomplete_reason=getattr(incomplete, "reason", None) if incomplete else None,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            reasoning_tokens=getattr(out_details, "reasoning_tokens", None) if out_details else None,
            cached_input_tokens=getattr(in_details, "cached_tokens", None) if in_details else None,
            latency_s=latency,
            total_latency_s=total_latency,
            attempts=attempt,
            attempt_errors=attempt_errors,
        )
