"""C2 condition, evidence-linked graph validators, scorer extensions and Mermaid export — offline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import render_graph
import run_condition
import score_run
from agents import get_condition
from agents.c2 import C2Agent
from agents.single_call import parse_structured_output
from evidence_graph.validate import CHECKS, RELATION_ENDPOINTS, validate_graph
from tests.fakes import FakeClient, make_response
from tests.test_c1_condition import CONFIG, SECRET_PASSAGE_TEXT, fake_data  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
STEM = "A 60-year-old man has a harsh systolic murmur at the right upper sternal border."
EQ = {"Book:1", "Book:2"}


def g(grounding="none", quote=None, ids=None):
    return {"grounding": grounding, "question_quote": quote, "passage_ids": ids or []}


def good_graph() -> dict:
    return {
        "objects": [
            {"id": "o1", "label": "systolic murmur", "is_question_finding": True,
             **g("question", "harsh systolic murmur")},
            {"id": "o2", "label": "aortic valve", "is_question_finding": False, **g("passages", ids=["Book:1"])},
            {"id": "o3", "label": "aortic stenosis", "is_question_finding": False, **g()},
        ],
        "processes": [{"id": "p1", "label": "valve calcification", **g("passages", ids=["Book:2"])}],
        "states": [
            {"id": "s1", "label": "normal", "owner_object_id": "o2", "is_question_finding": False, **g()},
            {"id": "s2", "label": "stenotic", "owner_object_id": "o2", "is_question_finding": False, **g()},
        ],
        "relations": [
            {"id": "r1", "type": "input_state", "source_id": "s1", "target_id": "p1", **g()},
            {"id": "r2", "type": "output_state", "source_id": "p1", "target_id": "s2", **g()},
            {"id": "r3", "type": "result", "source_id": "p1", "target_id": "o1", **g("passages", ids=["Book:1"])},
            {"id": "r4", "type": "exhibition", "source_id": "o3", "target_id": "p1", **g()},
        ],
        "answer_element_id": "o3",
        "support_path": ["o1", "r3", "p1", "r4", "o3"],
    }


def run(graph, answer="B", stem=STEM, allowed=EQ):
    return validate_graph(graph, answer=answer, stem=stem, allowed_passages=allowed)


def problems(result, check):
    return [(i["id"], i["problem"]) for i in result["checks"][check]["issues"]]


def rel(graph, rid):
    return next(r for r in graph["relations"] if r["id"] == rid)


# --------------------------------------------------------------------------- validators

def test_good_graph_passes_every_check():
    result = run(good_graph())
    assert result["all_passed"] and result["support_path_valid"] is True
    assert set(result["checks"]) == set(CHECKS)


def test_relation_vocabulary_matches_table_4_1():
    assert set(RELATION_ENDPOINTS) == {"consumption", "result", "effect", "instrument", "input_state", "output_state",
                                       "invocation", "aggregation", "exhibition", "generalization"}
    schema_types = json.loads((ROOT / "schemas/c2_output.schema.json").read_text())[
        "properties"]["graph"]["properties"]["relations"]["items"]["properties"]["type"]["enum"]
    assert set(schema_types) == set(RELATION_ENDPOINTS)


@pytest.mark.parametrize("rtype, source, target, ok", [
    ("consumption", "o2", "p1", True), ("consumption", "p1", "o2", False),
    ("result", "p1", "o3", True), ("result", "o3", "p1", False),
    ("effect", "p1", "o2", True), ("effect", "o2", "p1", False),
    ("instrument", "o2", "p1", True), ("instrument", "s1", "p1", False),
    ("invocation", "p1", "p1", False),
    ("aggregation", "o2", "o3", True), ("aggregation", "o2", "p1", False),
    ("exhibition", "o2", "o3", True), ("exhibition", "o2", "p1", True), ("exhibition", "p1", "o2", False),
    ("generalization", "o3", "o2", True), ("generalization", "s1", "o2", False),
    ("input_state", "o2", "p1", False), ("output_state", "p1", "o2", False),
])
def test_endpoint_types(rtype, source, target, ok):
    graph = good_graph()
    graph["relations"].append({"id": "r9", "type": rtype, "source_id": source, "target_id": target, **g()})
    result = run(graph)
    assert result["checks"]["endpoint_types"]["passed"] is ok
    if not ok:
        assert all(i == "r9" for i, _ in problems(result, "endpoint_types"))


def test_unknown_endpoint_and_self_loop():
    graph = good_graph()
    graph["relations"].append({"id": "r9", "type": "invocation", "source_id": "p1", "target_id": "p404", **g()})
    graph["processes"].append({"id": "p2", "label": "x", **g()})
    graph["relations"].append({"id": "r10", "type": "invocation", "source_id": "p2", "target_id": "p2", **g()})
    assert set(problems(run(graph), "endpoint_types")) == {("r9", "unknown_endpoint"), ("r10", "self_loop")}


def test_state_owner_must_exist_and_be_an_object():
    graph = good_graph()
    graph["states"][0]["owner_object_id"] = "p1"
    graph["states"][1]["owner_object_id"] = "o404"
    assert set(problems(run(graph), "state_ownership")) >= {("s1", "owner_not_object"), ("s2", "unknown_owner")}


def test_input_and_output_states_of_a_process_must_share_owner():
    graph = good_graph()
    graph["states"][1]["owner_object_id"] = "o3"  # p1 now turns a state of o2 into a state of o3
    result = run(graph)
    assert problems(result, "state_ownership") == [("p1", "input_output_state_owner_mismatch")]
    assert result["checks"]["state_ownership"]["issues"][0]["relations"] == ["r1", "r2"]


def test_process_with_only_output_state_passes():
    graph = good_graph()
    graph["relations"] = [r for r in graph["relations"] if r["id"] != "r1"]
    assert run(graph)["checks"]["state_ownership"]["passed"]


def test_isolated_elements_and_ownership_counts_as_connection():
    graph = good_graph()
    graph["objects"].append({"id": "o4", "label": "lonely", "is_question_finding": False, **g()})
    graph["objects"].append({"id": "o5", "label": "has a state", "is_question_finding": False, **g()})
    graph["states"].append({"id": "s3", "label": "st", "owner_object_id": "o5", "is_question_finding": False, **g()})
    assert problems(run(graph), "connectivity") == [("o4", "isolated_element")]


@pytest.mark.parametrize("path, expected", [
    (["o2", "r3", "p1", "r4", "o3"], "support_path_relation_does_not_connect_neighbours"),
    (["o3", "r4", "p1"], "support_path_does_not_start_at_question_finding"),
    (["o1", "r3", "p1"], "support_path_does_not_end_at_answer_element"),
    (["o1", "r3", "p1", "r4"], "support_path_even_length"),
    (["o1", "r3", "r4", "r4", "o3"], "support_path_expected_element"),
    (["o1", "p1", "o3"], "support_path_expected_relation"),
    ([], "empty_support_path"),
])
def test_support_path_errors(path, expected):
    graph = good_graph()
    graph["support_path"] = path
    result = run(graph)
    assert expected in {p for _, p in problems(result, "connectivity")}
    assert result["support_path_valid"] is False


def test_support_path_may_start_at_finding_state_and_runs_undirected():
    graph = good_graph()
    graph["objects"][0]["is_question_finding"] = False
    graph["states"][1]["is_question_finding"] = True
    graph["support_path"] = ["s2", "r2", "p1", "r4", "o3"]  # r2 is p1 -> s2, walked backwards
    assert run(graph)["support_path_valid"] is True


def test_answer_element_must_be_an_element():
    graph = good_graph()
    graph["answer_element_id"] = "r4"
    assert ("r4", "answer_element_not_an_element") in problems(run(graph), "connectivity")
    graph["answer_element_id"] = None
    assert (None, "missing_answer_element") in problems(run(graph), "connectivity")


def test_abstain_rules():
    graph = good_graph()
    assert ("o3", "abstain_with_answer_path") in problems(run(graph, answer="abstain"), "connectivity")
    graph["answer_element_id"], graph["support_path"] = None, []
    result = run(graph, answer="abstain")
    assert result["support_path_valid"] is None and result["checks"]["connectivity"]["passed"]


def test_citations_quotes_grounding_and_duplicate_ids():
    graph = good_graph()
    graph["processes"][0]["passage_ids"] = ["Book:1", "Book:404"]
    graph["objects"][0]["question_quote"] = "diastolic rumble"
    graph["objects"][2]["passage_ids"] = ["Book:1"]  # grounding none with a citation
    graph["relations"][3]["id"] = "o3"
    graph["support_path"] = ["o1", "r3", "p1"]
    graph["answer_element_id"] = "p1"
    result = run(graph)
    assert problems(result, "citation_validity") == [("p1", "passage_not_in_Eq")]
    assert problems(result, "question_quotes") == [("o1", "quote_not_in_stem")]
    assert problems(result, "grounding_rules") == [("o3", "none_with_citation")]
    assert problems(result, "unique_ids") == [("o3", "duplicate_id")]
    assert not result["all_passed"]


def test_question_quote_normalization_in_validator():
    graph = good_graph()
    graph["objects"][0]["question_quote"] = "Harsh  SYSTOLIC murmur."
    assert run(graph)["checks"]["question_quotes"]["passed"]


# --------------------------------------------------------------------------- schema and agent

def _walk_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _walk_objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_objects(value)


def test_c2_schema_is_strict_mode_compatible_and_shares_c1_explanation():
    schema = json.loads((ROOT / "schemas/c2_output.schema.json").read_text())
    for obj in _walk_objects(schema):
        assert obj["additionalProperties"] is False and set(obj["required"]) == set(obj["properties"])
    c1 = json.loads((ROOT / "schemas/c1_output.schema.json").read_text())
    assert schema["properties"]["explanation"] == c1["properties"]["explanation"]
    assert schema["properties"]["answer"] == c1["properties"]["answer"]
    graph = schema["properties"]["graph"]["properties"]
    for key in ("objects", "processes", "states", "relations"):
        assert graph[key]["items"]["properties"]["question_quote"]["maxLength"] == 300
    assert "is_question_finding" in graph["states"]["items"]["properties"]


def c2_output(graph=None, answer="B") -> dict:
    return {"answer": answer,
            "explanation": [{"sentence": "s", "grounding": "question", "question_quote": "systolic murmur",
                             "passage_ids": []}],
            "graph": graph if graph is not None else good_graph()}


def test_c2_parse_and_registry():
    agent = C2Agent(CONFIG, FakeClient(lambda r: None))
    assert get_condition("C2") is C2Agent
    assert parse_structured_output(make_response(json.dumps(c2_output())), agent.schema)["valid"]
    broken = c2_output()
    del broken["graph"]["support_path"]
    assert parse_structured_output(make_response(json.dumps(broken)), agent.schema)["reason"].startswith("schema_violation")
    bad_type = c2_output()
    bad_type["graph"]["relations"][0]["type"] = "causes"
    assert parse_structured_output(make_response(json.dumps(bad_type)), agent.schema)["reason"].startswith("schema_violation")


def test_c2_prompt_shares_c1_explanation_rules_and_inputs():
    c1_text = (ROOT / "prompts/c1/system.md").read_text()
    c2_text = (ROOT / "prompts/c2/system.md").read_text()
    c1_rules = c1_text[c1_text.index("For every sentence"):c1_text.index("Output only")]
    assert c1_rules in c2_text
    assert (ROOT / "prompts/c1/user.md").read_text() == (ROOT / "prompts/c2/user.md").read_text()
    c1_cfg = json.loads((ROOT / "configs/c1.json").read_text())
    c2_cfg = json.loads((ROOT / "configs/c2.json").read_text())
    assert {k: v for k, v in c1_cfg.items() if k != "condition"} == {k: v for k, v in c2_cfg.items() if k != "condition"}


# --------------------------------------------------------------------------- runner, scorer, render (fake data)

def _graph_for(stem_no: int) -> dict:
    """good_graph() adapted to the fake-data stems ("Stem <n> with {{braces}}?") and passage ids."""
    graph = copy.deepcopy(good_graph())
    graph["objects"][0]["question_quote"] = f"Stem {stem_no} with"
    graph["objects"][1]["passage_ids"] = [f"Book:{stem_no:06d}0"]
    graph["processes"][0]["passage_ids"] = [f"Book:{stem_no:06d}1"]
    rel(graph, "r3")["passage_ids"] = ["Made:up"]  # one illegal citation per graph
    return graph


def _c2_responder(request):
    if "Stem 0" in request.user_prompt:
        out = c2_output(_graph_for(0), answer="A")
        out["explanation"][0]["question_quote"] = "Stem 0"
        return make_response(json.dumps(out))
    if "Stem 1" in request.user_prompt:
        graph = _graph_for(1)
        graph["support_path"] = ["o3", "r4", "p1"]
        out = c2_output(graph, answer="B")
        out["explanation"][0]["question_quote"] = "Stem 1"
        return make_response(json.dumps(out))
    return make_response('{"answer": "C", "explanation": [], "graph": {}}')  # schema violation -> invalid


def _run_c2(fake_data):
    client = FakeClient(_c2_responder)
    runs = fake_data / "runs"
    code = run_condition.main(["--condition", "C2", "--split", "pilot", "--allow-dirty", "--runs-dir", str(runs)],
                              client_factory=lambda: client)
    assert code == 0
    [run_dir] = list(runs.iterdir())
    return run_dir, client


def test_c2_runner_records_validation_without_repair(fake_data):
    run_dir, client = _run_c2(fake_data)
    assert len(client.requests) == 3  # one call per question, never a retry for content
    assert "_C2_pilot_" in run_dir.name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema"]["path"] == "schemas/c2_output.schema.json"
    assert manifest["prompt_files"]["system"]["path"] == "prompts/c2/system.md"
    preds = {p.stem: json.loads(p.read_text()) for p in (run_dir / "predictions").glob("*.json")}
    v0 = preds["q0"]["extra"]["validation"]
    assert preds["q0"]["validity"] == "valid" and not v0["all_passed"]
    assert [i["passage_id"] for i in v0["checks"]["citation_validity"]["issues"]] == ["Made:up"]
    assert preds["q1"]["validity"] == "valid" and preds["q1"]["extra"]["validation"]["support_path_valid"] is False
    assert preds["q2"]["validity"] == "invalid" and preds["q2"]["extra"] == {}
    for path in run_dir.rglob("*.json"):
        assert SECRET_PASSAGE_TEXT not in path.read_text()


def test_score_c2(fake_data):
    run_dir, _ = _run_c2(fake_data)
    score = score_run.score_run(run_dir, dataset_dir=fake_data / "dataset")
    assert score["answers"]["correct"] == 2 and score["answers"]["invalid"] == 1
    graph = score["graph"]
    assert graph["graphs"] == 2 and graph["validated"] == 2
    assert graph["checks"]["citation_validity"] == {"passed": 0, "of": 2, "rate": 0.0, "issues": 2}
    assert graph["checks"]["endpoint_types"]["rate"] == 1.0
    assert graph["support_path_valid"] == 1 and graph["support_path_applicable"] == 2
    assert graph["size"]["objects"]["max"] == 3 and graph["size"]["relations"]["median"] == 4
    assert graph["element_grounding"]["counts"] == {"question": 2, "passages": 4, "none": 6}
    assert graph["relation_grounding"]["counts"] == {"question": 0, "passages": 2, "none": 6}
    assert graph["element_grounding"]["question_quote_verbatim_rate"] == 1.0
    assert graph["relation_grounding"]["passages_citation_validity_rate"] == 0.0
    assert graph["relation_types"]["result"] == 2
    text = score_run.format_summary(score)
    assert "check citation_validity" in text and "Element grounding" in text


def test_render_graph(fake_data, tmp_path):
    run_dir, _ = _run_c2(fake_data)
    out_dir = tmp_path / "rendered"
    assert render_graph.main([str(run_dir), "--out-dir", str(out_dir)]) == 0
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["q0.md", "q1.md"]  # invalid output skipped
    text = (out_dir / "q0.md").read_text()
    assert "```mermaid" in text and "flowchart LR" in text
    assert 'n_o1["systolic murmur' in text and "n_p1([" in text and "-. state .-" in text
    assert "==>" in text  # support-path relation drawn thick
    assert "class n_o3 answer" in text and "citation_validity: FAIL" in text
    assert SECRET_PASSAGE_TEXT not in text and "Stem 0 with" not in text  # no passage text, no quotes
    with pytest.raises(SystemExit):
        render_graph.main([str(run_dir), "--out-dir", str(run_dir / "graphs")])
