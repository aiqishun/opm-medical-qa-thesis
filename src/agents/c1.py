"""C1: single LLM call — answer from the fixed passages with a cited textual explanation."""

from __future__ import annotations

from agents.single_call import SingleCallCondition, parse_structured_output

parse_c1_output = parse_structured_output  # backwards-compatible name


class C1Agent(SingleCallCondition):
    name = "C1"
    prompt_dir = "c1"
    schema_file = "c1_output.schema.json"
    schema_name = "c1_answer_with_cited_explanation"
    agent_label = "c1_single"
