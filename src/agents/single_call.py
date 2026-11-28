"""Conditions that make exactly one structured-output call per question (C1, C2)."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from agents.base import (PROMPTS_DIR, Condition, ModelConfig, OnCall, PromptFile, QuestionResult, call_record,
                         format_options, format_passages, load_schema)
from experiment.inputs import EvidencePassage, Question
from llm.client import LLMClient, LLMRequest, LLMResponse


def parse_structured_output(response: LLMResponse, schema: dict[str, Any]) -> dict[str, Any]:
    """Classify a response as valid or invalid. Never repairs or retries content."""
    if not response.ok:
        return {"valid": False, "reason": "api_error", "output": None}
    if response.status != "completed":
        return {"valid": False, "reason": f"status_{response.status}:{response.incomplete_reason}", "output": None}
    if not response.raw_text:
        return {"valid": False, "reason": "empty_output", "output": None}
    try:
        output = json.loads(response.raw_text)
    except json.JSONDecodeError as exc:
        return {"valid": False, "reason": f"json_decode_error: {exc.msg}", "output": None}
    try:
        jsonschema.validate(output, schema)
    except jsonschema.ValidationError as exc:
        return {"valid": False, "reason": f"schema_violation: {exc.message[:300]}", "output": output}
    return {"valid": True, "reason": None, "output": output}


class SingleCallCondition(Condition):
    """One call; subclasses set the prompt directory, the schema and optional post-processing."""

    prompt_dir: str = ""
    schema_file: str = ""
    schema_name: str = ""
    agent_label: str = ""

    def __init__(self, config: ModelConfig, client: LLMClient) -> None:
        super().__init__(config, client)
        self.prompts = {
            "system": PromptFile.load(PROMPTS_DIR / self.prompt_dir / "system.md"),
            "user": PromptFile.load(PROMPTS_DIR / self.prompt_dir / "user.md"),
        }
        self.schema, self.schema_sha256 = load_schema(self.schema_file)

    def provenance(self) -> dict[str, Any]:
        return {
            "prompt_files": {role: {"path": f"prompts/{self.prompt_dir}/{pf.path.name}", "sha256": pf.sha256}
                             for role, pf in self.prompts.items()},
            "schema": {"path": f"schemas/{self.schema_file}", "name": self.schema_name, "sha256": self.schema_sha256},
        }

    def build_request(self, question: Question, passages: list[EvidencePassage]) -> LLMRequest:
        user = self.prompts["user"].render({
            "question": question.question,
            "options": format_options(question),
            "passages": format_passages(passages),
        })
        return LLMRequest(
            model=self.config.model,
            reasoning_effort=self.config.reasoning_effort,
            max_output_tokens=self.config.max_output_tokens,
            system_prompt=self.prompts["system"].render({}),
            user_prompt=user,
            schema_name=self.schema_name,
            schema=self.schema,
        )

    def postprocess(self, output: dict[str, Any], question: Question,
                    passages: list[EvidencePassage]) -> dict[str, Any]:
        """Extra, non-repairing analysis of a valid output (e.g. validators). Default: none."""
        return {}

    def run_question(self, question: Question, passages: list[EvidencePassage], on_call: OnCall) -> QuestionResult:
        request = self.build_request(question, passages)
        response = self.client.complete(request)
        parsed = parse_structured_output(response, self.schema)
        on_call(call_record(
            condition=self.name, question_id=question.id, round_=0, agent=self.agent_label,
            request=request, response=response, prompt_files=self.prompts,
            schema_sha256=self.schema_sha256,
            parse={"valid": parsed["valid"], "reason": parsed["reason"]},
        ))
        output = parsed["output"] if parsed["valid"] else None
        return QuestionResult(
            question_id=question.id,
            answer=output["answer"] if output else None,
            validity="valid" if output else "invalid",
            invalid_reason=parsed["reason"],
            output=output,
            n_calls=1,
            extra=self.postprocess(output, question, passages) if output else {},
        )
