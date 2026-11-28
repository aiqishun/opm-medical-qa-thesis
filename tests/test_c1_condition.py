"""C1 condition, runner, scorer and client — all offline (fake client, temporary data)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
import pytest

import run_condition
import score_run
from agents import get_condition
from agents.base import ModelConfig, PromptFile
from agents.c1 import C1Agent, parse_c1_output
from experiment import inputs as data
from experiment.records import write_once_json
from llm.client import LLMRequest, OpenAIResponsesClient
from tests.fakes import FakeClient, c1_json, make_response

ROOT = Path(__file__).resolve().parents[1]
SECRET_PASSAGE_TEXT = "COPYRIGHTED-PASSAGE-TEXT"
CONFIG = ModelConfig(model="m", reasoning_effort="medium", max_output_tokens=1000)


# --------------------------------------------------------------------------- fixtures

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    """Three pilot questions, retrieval sets and a tiny corpus in a temp tree."""
    qs = [{"id": f"q{i}", "question": f"Stem {i} with {{{{braces}}}}?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}
          for i in range(3)]
    inputs_dir = tmp_path / "dataset/inputs"
    _write_jsonl(inputs_dir / "pilot_questions.jsonl", qs)
    _write_jsonl(inputs_dir / "test_questions.jsonl", qs)
    retrieval_dir = tmp_path / "retrieval"
    rows = [{"id": q["id"], "passages": [{"passage_id": f"Book:{i:06d}{j}", "rank": j + 1, "score": 1.0}
                                         for j in range(10)]} for i, q in enumerate(qs)]
    _write_jsonl(retrieval_dir / "pilot.jsonl", rows)
    _write_jsonl(retrieval_dir / "test.jsonl", rows)
    (retrieval_dir / "manifest.json").write_text("{}", encoding="utf-8")
    corpus = tmp_path / "corpus/passages.jsonl"
    _write_jsonl(corpus, [{"id": p["passage_id"], "book": "Book", "index": 0, "text": f"{SECRET_PASSAGE_TEXT} {p['passage_id']}"}
                          for row in rows for p in row["passages"]] + [{"id": "Other:1", "book": "Other", "index": 1, "text": "x"}])
    (corpus.parent / "manifest.json").write_text("{}", encoding="utf-8")
    gold_dir = tmp_path / "dataset/evaluator_only"
    _write_jsonl(gold_dir / "pilot_gold.jsonl", [{"id": "q0", "answer_idx": "A"}, {"id": "q1", "answer_idx": "B"},
                                                 {"id": "q2", "answer_idx": "C"}])
    monkeypatch.setattr(data, "INPUTS_DIR", inputs_dir)
    monkeypatch.setattr(data, "RETRIEVAL_DIR", retrieval_dir)
    monkeypatch.setattr(data, "CORPUS_PASSAGES", corpus)
    return tmp_path


def _scripted_responder():
    """q0: correct answer with one legal and one illegal citation; q1: abstain; q2: unparseable."""
    def respond(request: LLMRequest):
        if "Stem 0" in request.user_prompt:
            return make_response(c1_json("A", [[], ["Book:0000000"], [], ["Made:up"], [], ["Book:0000001"]],
                                         groundings=["question", "passages", "none", "passages", "question", "none"],
                                         quotes=["STEM  0 with", None, None, None, "not in the stem", None]))
        if "Stem 1" in request.user_prompt:
            return make_response(c1_json("abstain", [[]]))
        return make_response('{"answer": "C", "explanation": [')
    return respond


def _run(fake_data, extra: list[str] | None = None, responder=None):
    client = FakeClient(responder or _scripted_responder())
    runs = fake_data / "runs"
    code = run_condition.main(["--condition", "C1", "--split", "pilot", "--allow-dirty", "--runs-dir", str(runs),
                               *(extra or [])], client_factory=lambda: client)
    return code, client, runs


# --------------------------------------------------------------------------- answer isolation

def test_runner_and_agent_sources_never_mention_evaluator_only():
    sources = [ROOT / "scripts/run_condition.py", ROOT / "scripts/check_model_params.py",
               *sorted((ROOT / "src/agents").glob("*.py")), *sorted((ROOT / "src/llm").glob("*.py")),
               *sorted((ROOT / "src/experiment").glob("*.py")), *sorted((ROOT / "prompts").rglob("*"))]
    offenders = [str(p.relative_to(ROOT)) for p in sources if p.is_file() and "evaluator_only" in p.read_text()]
    assert len(sources) > 5
    assert offenders == []


def test_question_loader_rejects_answer_fields(tmp_path):
    path = tmp_path / "q.jsonl"
    _write_jsonl(path, [{"id": "x", "question": "q", "options": {"A": 1, "B": 2, "C": 3, "D": 4}, "answer_idx": "A"}])
    with pytest.raises(ValueError, match="unexpected fields"):
        data.load_questions(path)


# --------------------------------------------------------------------------- prompts and parsing

def test_prompt_render_requires_exact_placeholders(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("Q: {{question}}", encoding="utf-8")
    prompt = PromptFile.load(path)
    assert prompt.render({"question": "has {{options}} inside"}) == "Q: has {{options}} inside"
    with pytest.raises(ValueError):
        prompt.render({})
    with pytest.raises(ValueError):
        prompt.render({"question": "x", "extra": "y"})


def test_c1_request_contains_question_options_and_all_passages():
    agent = C1Agent(CONFIG, FakeClient(lambda r: None))
    question = data.Question(id="q", question="What murmur?", options={"A": "one", "B": "two", "C": "three", "D": "four"})
    passages = [data.EvidencePassage(f"Book:{i}", i + 1, f"text {i}") for i in range(10)]
    request = agent.build_request(question, passages)
    assert "What murmur?" in request.user_prompt and "D. four" in request.user_prompt
    assert all(f"[Book:{i}]\ntext {i}" in request.user_prompt for i in range(10))
    assert "{{" not in request.user_prompt and "{{" not in request.system_prompt
    assert request.model == "m" and request.reasoning_effort == "medium" and request.max_output_tokens == 1000


@pytest.mark.parametrize("response, reason", [
    (make_response("not json"), "json_decode_error"),
    (make_response('{"answer": "E", "explanation": []}'), "schema_violation"),
    (make_response('{"answer": "A"}'), "schema_violation"),
    (make_response(c1_json("A", [[]]), status="incomplete", incomplete_reason="max_output_tokens"), "status_incomplete"),
    (make_response(None, ok=False, error={"type": "BadRequestError"}), "api_error"),
    (make_response(""), "empty_output"),
])
def test_c1_parse_marks_invalid(response, reason):
    schema = C1Agent(CONFIG, FakeClient(lambda r: None)).schema
    parsed = parse_c1_output(response, schema)
    assert parsed["valid"] is False and parsed["reason"].startswith(reason)


def test_c1_parse_accepts_valid_output():
    schema = C1Agent(CONFIG, FakeClient(lambda r: None)).schema
    parsed = parse_c1_output(make_response(c1_json("abstain", [["X:1"], []])), schema)
    assert parsed["valid"] and parsed["output"]["answer"] == "abstain"


def test_c1_schema_requires_grounding_fields():
    schema = C1Agent(CONFIG, FakeClient(lambda r: None)).schema
    missing = '{"answer": "A", "explanation": [{"sentence": "s", "passage_ids": []}]}'
    bad_enum = '{"answer": "A", "explanation": [{"sentence": "s", "grounding": "x", "question_quote": null, "passage_ids": []}]}'
    for raw in (missing, bad_enum):
        assert parse_c1_output(make_response(raw), schema)["reason"].startswith("schema_violation")
    item = schema["properties"]["explanation"]["items"]
    assert set(item["required"]) == set(item["properties"])  # strict structured output: every field required


@pytest.mark.parametrize("sentence, expected", [
    ({"grounding": "question", "question_quote": "x", "passage_ids": []}, []),
    ({"grounding": "question", "question_quote": " ", "passage_ids": []}, ["question_without_quote"]),
    ({"grounding": "passages", "question_quote": None, "passage_ids": []}, ["passages_without_citation"]),
    ({"grounding": "passages", "question_quote": "q", "passage_ids": ["a"]}, ["quote_on_non_question_sentence"]),
    ({"grounding": "none", "question_quote": None, "passage_ids": ["a"]}, ["none_with_citation"]),
])
def test_grounding_rules(sentence, expected):
    assert score_run.sentence_violations(sentence) == expected


def test_quote_normalization():
    stem = "A 62\u2011year\u2010old man has  \u201ccrushing\u201d chest pain\u2014radiating to the jaw."
    assert score_run.quote_in_stem('"Crushing" chest pain-radiating', stem)
    assert score_run.quote_in_stem("chest   pain.", stem)
    assert not score_run.quote_in_stem("chest pain radiating", stem)
    assert not score_run.quote_in_stem("  ...  ", stem)


def test_condition_registry():
    assert get_condition("C1") is C1Agent
    with pytest.raises(KeyError):
        get_condition("C9")


# --------------------------------------------------------------------------- runner

def test_test_split_requires_confirmation_flag(fake_data, capsys):
    created = []
    code = run_condition.main(["--condition", "C1", "--split", "test", "--allow-dirty",
                               "--runs-dir", str(fake_data / "runs")], client_factory=lambda: created.append(1))
    assert code == 2 and created == []
    assert "--i-confirm-config-frozen" in capsys.readouterr().err
    assert not (fake_data / "runs").exists()


def test_runner_writes_complete_frozen_run(fake_data):
    code, client, runs = _run(fake_data)
    assert code == 0 and len(client.requests) == 3
    [run_dir] = list(runs.iterdir())
    assert run_dir.name.endswith("_C1_pilot_" + run_condition.git_info(ROOT)["short"])

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["question_ids"] == ["q0", "q1", "q2"]
    assert len(manifest["passage_ids"]["q0"]) == 10
    assert set(manifest["prompt_files"]) == {"system", "user"}
    assert len(manifest["prompt_files"]["system"]["sha256"]) == 64

    calls = [json.loads(p.read_text()) for p in sorted((run_dir / "calls").glob("*.json"))]
    assert len(calls) == 3
    for call in calls:
        for key in ("condition", "question_id", "round", "model_requested", "model_returned", "reasoning_effort",
                    "parameters", "prompt_files", "usage", "latency_s", "raw_text", "parse"):
            assert key in call
        assert call["usage"]["reasoning_tokens"] == 200
        assert "temperature" not in call["parameters"] and "top_p" not in call["parameters"]
    assert calls[2]["parse"]["valid"] is False and calls[2]["raw_text"].startswith('{"answer"')

    preds = {p.stem: json.loads(p.read_text()) for p in (run_dir / "predictions").glob("*.json")}
    assert preds["q0"]["answer"] == "A" and preds["q1"]["answer"] == "abstain"
    assert preds["q2"]["validity"] == "invalid" and preds["q2"]["answer"] is None
    assert json.loads((run_dir / "complete.json").read_text())["counts"]["invalid"] == 1

    for path in run_dir.rglob("*.json"):  # passage text must never be persisted
        assert SECRET_PASSAGE_TEXT not in path.read_text()


def test_runner_limit(fake_data):
    code, client, runs = _run(fake_data, ["--limit", "2"])
    assert code == 0 and len(client.requests) == 2


def test_runner_overrides_config_from_cli(fake_data):
    code, client, _ = _run(fake_data, ["--model", "other-model", "--max-output-tokens", "777"])
    assert code == 0
    assert {r.model for r in client.requests} == {"other-model"}
    assert {r.max_output_tokens for r in client.requests} == {777}
    assert {r.reasoning_effort for r in client.requests} == {"medium"}  # from configs/c1.json


def test_write_once(tmp_path):
    write_once_json(tmp_path / "a.json", {"x": 1})
    with pytest.raises(FileExistsError):
        write_once_json(tmp_path / "a.json", {"x": 2})


# --------------------------------------------------------------------------- scorer

def test_score_run(fake_data):
    _, _, runs = _run(fake_data)
    [run_dir] = list(runs.iterdir())
    score = score_run.score_run(run_dir, dataset_dir=fake_data / "dataset")
    answers, citations, cost = score["answers"], score["citations"], score["cost"]
    assert answers == {"n_questions": 3, "correct": 1, "accuracy": 1 / 3, "abstain": 1, "invalid": 1, "missing": 0}
    assert citations["cited_ids"] == 3 and citations["cited_ids_in_Eq"] == 2
    assert citations["citation_validity_rate"] == pytest.approx(2 / 3)
    g = score["grounding"]
    assert g["counts"] == {"question": 2, "passages": 2, "none": 3}  # q1 contributes one "none"
    assert g["question_quotes_checked"] == 2 and g["question_quotes_verbatim"] == 1
    assert g["passages_sentences"] == 2 and g["passages_sentences_all_ids_in_Eq"] == 1
    assert g["passages_citation_validity_rate"] == 0.5
    assert g["rule_violations"] == {"none_with_citation": 1}
    assert cost["reasoning_tokens_per_question"] == {"n": 3, "zero": 0, "min": 200, "max": 200, "mean": 200,
                                                     "median": 200}
    assert citations["outputs_all_citations_valid"] == 0 and citations["outputs_with_citations"] == 1
    assert cost["calls"] == 3 and cost["input_tokens"] == 3000 and cost["output_tokens_incl_reasoning"] == 900
    assert cost["reasoning_tokens"] == 600
    assert cost["estimated_cost_usd"] == pytest.approx(3000 * 2e-6 + 900 * 10e-6, abs=1e-4)
    assert "not a held-out" in score["note"]
    assert "Accuracy: 1/3" in score_run.format_summary(score)


def test_score_counts_missing_predictions_as_wrong(fake_data):
    _, _, runs = _run(fake_data)
    [run_dir] = list(runs.iterdir())
    (run_dir / "predictions/q0.json").unlink()
    score = score_run.score_run(run_dir, dataset_dir=fake_data / "dataset")
    assert score["answers"]["missing"] == 1 and score["answers"]["correct"] == 0


# --------------------------------------------------------------------------- OpenAI client (fake SDK, no network)

class _FakeUsage:
    input_tokens, output_tokens = 11, 22

    class output_tokens_details:  # noqa: N801
        reasoning_tokens = 7

    class input_tokens_details:  # noqa: N801
        cached_tokens = 0


class _FakeSDKResponse:
    output_text, model, id, status, incomplete_details, usage = '{"ok": true}', "m-2026", "resp_1", "completed", None, _FakeUsage


class _FakeResponses:
    def __init__(self, failures: list[Exception]):
        self.failures, self.kwargs = list(failures), []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return _FakeSDKResponse()


def _client_with(monkeypatch, failures):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = OpenAIResponsesClient(retry_wait_s=0)
    fake = _FakeResponses(failures)
    client._client = type("C", (), {"responses": fake})()
    return client, fake


_REQ = LLMRequest(model="m", reasoning_effort="medium", max_output_tokens=50, system_prompt="s", user_prompt="u",
                  schema_name="n", schema={"type": "object"})
_HTTP_REQ = httpx.Request("POST", "https://example.invalid/v1/responses")


def test_client_sends_no_sampling_parameters(monkeypatch):
    client, fake = _client_with(monkeypatch, [])
    response = client.complete(_REQ)
    sent = fake.kwargs[0]
    assert "temperature" not in sent and "top_p" not in sent
    assert sent["reasoning"] == {"effort": "medium"} and sent["max_output_tokens"] == 50
    assert sent["text"]["format"]["type"] == "json_schema" and sent["text"]["format"]["strict"] is True
    assert response.ok and response.reasoning_tokens == 7 and response.response_model == "m-2026"
    assert client._client is not None and "sk-test" not in repr(response)


def test_client_retries_once_on_network_error(monkeypatch):
    client, fake = _client_with(monkeypatch, [openai.APIConnectionError(request=_HTTP_REQ)])
    response = client.complete(_REQ)
    assert response.ok and response.attempts == 2 and len(fake.kwargs) == 2
    assert response.attempt_errors[0]["type"] == "APIConnectionError"


def test_client_gives_up_after_one_retry(monkeypatch):
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=_HTTP_REQ), body=None)
    client, fake = _client_with(monkeypatch, [rate, rate, rate])
    response = client.complete(_REQ)
    assert not response.ok and response.attempts == 2 and len(fake.kwargs) == 2


def test_client_does_not_retry_bad_request(monkeypatch):
    bad = openai.BadRequestError("unsupported parameter", response=httpx.Response(400, request=_HTTP_REQ), body=None)
    client, fake = _client_with(monkeypatch, [bad])
    response = client.complete(_REQ)
    assert not response.ok and response.attempts == 1 and response.error["status_code"] == 400


def test_client_requires_env_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIResponsesClient()
