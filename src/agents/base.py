"""Shared machinery for experimental conditions (C1, C2, C3, ...).

A condition receives one answer-free question with its fixed passage set and an
``LLMClient``. It performs one or more model calls, reports every call through
``on_call`` as soon as it finishes (so the runner can persist it immediately),
and returns its final prediction. Conditions never see reference answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from experiment.inputs import EvidencePassage, Question
from experiment.records import sha256_text
from llm.client import LLMClient, LLMRequest, LLMResponse

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
SCHEMAS_DIR = ROOT / "schemas"

_PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


@dataclass(frozen=True)
class PromptFile:
    path: Path
    text: str
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "PromptFile":
        text = path.read_text(encoding="utf-8")
        return cls(path=path, text=text, sha256=sha256_text(text))

    def render(self, values: dict[str, str]) -> str:
        """Substitute ``{{name}}`` placeholders in one pass; every placeholder must be supplied."""
        needed = set(_PLACEHOLDER.findall(self.text))
        missing, unused = needed - set(values), set(values) - needed
        if missing or unused:
            raise ValueError(f"{self.path.name}: missing {sorted(missing)}, unused {sorted(unused)}")
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], self.text)


@dataclass
class ModelConfig:
    model: str
    reasoning_effort: str
    max_output_tokens: int


@dataclass
class QuestionResult:
    question_id: str
    answer: str | None              # "A"-"D", "abstain", or None if invalid
    validity: str                   # "valid" or "invalid"
    invalid_reason: str | None
    output: dict[str, Any] | None   # parsed structured output (passage ids only, no passage text)
    n_calls: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OnCall = Callable[[dict[str, Any]], None]


def format_options(question: Question) -> str:
    return "\n".join(f"{label}. {text}" for label, text in sorted(question.options.items()))


def format_passages(passages: list[EvidencePassage]) -> str:
    return "\n\n".join(f"[{p.passage_id}]\n{p.text}" for p in passages)


def call_record(*, condition: str, question_id: str, round_: int, agent: str, request: LLMRequest,
                response: LLMResponse, prompt_files: dict[str, PromptFile], schema_sha256: str,
                parse: dict[str, Any]) -> dict[str, Any]:
    """The per-call log entry required by the thesis protocol.

    The rendered prompts contain copyrighted passage text, so only their hashes
    are stored; the passage identifiers are recorded by the runner.
    """
    return {
        "condition": condition,
        "question_id": question_id,
        "round": round_,
        "agent": agent,
        "model_requested": request.model,
        "model_returned": response.response_model,
        "reasoning_effort": request.reasoning_effort,
        "parameters": request.api_parameters(),
        "prompt_files": {role: {"path": str(pf.path.relative_to(ROOT)), "sha256": pf.sha256}
                         for role, pf in prompt_files.items()},
        "rendered_prompt_sha256": {"system": sha256_text(request.system_prompt),
                                   "user": sha256_text(request.user_prompt)},
        "schema_sha256": schema_sha256,
        "response_id": response.response_id,
        "api_ok": response.ok,
        "status": response.status,
        "incomplete_reason": response.incomplete_reason,
        "error": response.error,
        "attempts": response.attempts,
        "attempt_errors": response.attempt_errors,
        "usage": {
            "input_tokens": response.input_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "output_tokens": response.output_tokens,
            "reasoning_tokens": response.reasoning_tokens,
        },
        "latency_s": round(response.latency_s, 3),
        "total_latency_s": round(response.total_latency_s, 3),
        "raw_text": response.raw_text,
        "parse": parse,
    }


def load_schema(name: str) -> tuple[dict[str, Any], str]:
    path = SCHEMAS_DIR / name
    text = path.read_text(encoding="utf-8")
    return json.loads(text), sha256_text(text)


class Condition:
    """Base class; subclasses implement ``run_question``."""

    name: str = ""

    def __init__(self, config: ModelConfig, client: LLMClient) -> None:
        self.config = config
        self.client = client

    def provenance(self) -> dict[str, Any]:
        """Prompt and schema hashes for the run manifest."""
        raise NotImplementedError

    def run_question(self, question: Question, passages: list[EvidencePassage], on_call: OnCall) -> QuestionResult:
        raise NotImplementedError
