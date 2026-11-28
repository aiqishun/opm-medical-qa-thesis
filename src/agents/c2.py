"""C2: single LLM call — answer, cited explanation and an evidence-linked OPM graph.

The deterministic validators (Table 4.2) run on the graph and their results are
stored with the prediction. There is no repair and no retry: a graph that fails
validation is still a valid output.

The support path is validated without regard to link direction, because OPM
link types point in different directions (consumption object -> process,
result process -> object, ...) and a reasoning path from a finding to the
answer may run against some of them. See ``evidence_graph.validate``.
"""

from __future__ import annotations

from typing import Any

from agents.single_call import SingleCallCondition
from evidence_graph.validate import validate_graph
from experiment.inputs import EvidencePassage, Question


class C2Agent(SingleCallCondition):
    name = "C2"
    prompt_dir = "c2"
    schema_file = "c2_output.schema.json"
    schema_name = "c2_answer_with_evidence_linked_opm_graph"
    agent_label = "c2_single_graph"

    def postprocess(self, output: dict[str, Any], question: Question,
                    passages: list[EvidencePassage]) -> dict[str, Any]:
        return {"validation": validate_graph(output["graph"], answer=output["answer"], stem=question.question,
                                             allowed_passages={p.passage_id for p in passages})}
