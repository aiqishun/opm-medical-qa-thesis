"""Offline stand-ins for the language model client."""

from __future__ import annotations

import json
from typing import Callable

from llm.client import LLMRequest, LLMResponse


def make_response(raw_text: str | None, *, ok: bool = True, status: str | None = "completed",
                  incomplete_reason: str | None = None, error: dict | None = None) -> LLMResponse:
    return LLMResponse(ok=ok, raw_text=raw_text, response_model="fake-model-2026-01-01" if ok else None,
                       response_id="resp_fake" if ok else None, status=status if ok else None,
                       incomplete_reason=incomplete_reason, input_tokens=1000 if ok else None,
                       output_tokens=300 if ok else None, reasoning_tokens=200 if ok else None,
                       cached_input_tokens=0 if ok else None, latency_s=0.5, total_latency_s=0.5, attempts=1,
                       error=error)


class FakeClient:
    """Returns responses produced by ``responder(request)`` and records every request."""

    def __init__(self, responder: Callable[[LLMRequest], LLMResponse]) -> None:
        self.responder = responder
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.responder(request)


def c1_json(answer: str, citations: list[list[str]], groundings: list[str] | None = None,
            quotes: list[str | None] | None = None) -> str:
    """Default grounding: "passages" if the sentence cites something, else "none"."""
    groundings = groundings or ["passages" if ids else "none" for ids in citations]
    quotes = quotes or [None] * len(citations)
    return json.dumps({"answer": answer,
                       "explanation": [{"sentence": f"Sentence {i}.", "grounding": g, "question_quote": q,
                                        "passage_ids": ids}
                                       for i, (ids, g, q) in enumerate(zip(citations, groundings, quotes))]})
