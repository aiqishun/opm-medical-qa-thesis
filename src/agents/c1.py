"""C1: single LLM call — answer from the fixed passages with a cited textual explanation."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from agents.base import (PROMPTS_DIR, Condition, ModelConfig, OnCall, PromptFile, QuestionResult, call_record,
                         format_options, format_passages, load_schema)
from experiment.inputs import EvidencePassage, Question
from llm.client import LLMClient, LLMRequest, LLMResponse

SCHEMA_FILE = "c1_output.schema.json"
SCHEMA_NAME = "c1_answer_with_cited_explanation"


def parse_c1_output(response: LLMResponse, schema: dict[str, Any]) -> dict[str, Any]:
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


class C1Agent(Condition):
    name = "C1"

    def __init__(self, config: ModelConfig, client: LLMClient) -> None:
        super().__init__(config, client)
        self.prompts = {
            "system": PromptFile.load(PROMPTS_DIR / "c1/system.md"),
            "user": PromptFile.load(PROMPTS_DIR / "c1/user.md"),
        }
        self.schema, self.schema_sha256 = load_schema(SCHEMA_FILE)

    def provenance(self) -> dict[str, Any]:
        return {
            "prompt_files": {role: {"path": f"prompts/c1/{pf.path.name}", "sha256": pf.sha256}
                             for role, pf in self.prompts.items()},
            "schema": {"path": f"schemas/{SCHEMA_FILE}", "name": SCHEMA_NAME, "sha256": self.schema_sha256},
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
            schema_name=SCHEMA_NAME,
            schema=self.schema,
        )

    def run_question(self, question: Question, passages: list[EvidencePassage], on_call: OnCall) -> QuestionResult:
        request = self.build_request(question, passages)
        response = self.client.complete(request)
        parsed = parse_c1_output(response, self.schema)
        on_call(call_record(
            condition=self.name, question_id=question.id, round_=0, agent="c1_single",
            request=request, response=response, prompt_files=self.prompts,
            schema_sha256=self.schema_sha256,
            parse={"valid": parsed["valid"], "reason": parsed["reason"]},
        ))
        output = parsed["output"]
        return QuestionResult(
            question_id=question.id,
            answer=output["answer"] if parsed["valid"] else None,
            validity="valid" if parsed["valid"] else "invalid",
            invalid_reason=parsed["reason"],
            output=output if parsed["valid"] else None,
            n_calls=1,
        )
