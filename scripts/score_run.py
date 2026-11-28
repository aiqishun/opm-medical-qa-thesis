#!/usr/bin/env python3
"""Score a frozen run directory against the evaluator-only reference answers.

    python3 scripts/score_run.py experiments/runs/<run_id> [--out score.json]

This is the only experiment script that reads ``evaluator_only/``. It never
modifies the run directory. Counting rules (thesis protocol):

* accuracy denominator = every question in the run manifest; abstentions,
  invalid outputs and missing predictions all count as incorrect;
* citation validity = share of cited passage ids that belong to that
  question's fixed passage set E_q (computed over valid outputs);
* cost = input tokens x $2/M + output tokens x $10/M. Output tokens include
  reasoning tokens (OpenAI usage convention), which are also reported separately.

Pilot results are for development only and must not be reported as held-out results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/versions/thesis-v0.2"
GOLD_FILES = {"pilot": "evaluator_only/pilot_gold.jsonl", "test": "evaluator_only/test_answers.jsonl"}
PRICE_INPUT_PER_M = 2.0
PRICE_OUTPUT_PER_M = 10.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gold(split: str, dataset_dir: Path) -> dict[str, str]:
    gold = {}
    with (dataset_dir / GOLD_FILES[split]).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                gold[record["id"]] = record["answer_idx"]
    return gold


def _sum(values: list[int | None]) -> int:
    return sum(v for v in values if v is not None)


def score_run(run_dir: Path, dataset_dir: Path = DATASET_DIR) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    split = manifest["split"]
    gold = load_gold(split, dataset_dir)
    question_ids: list[str] = manifest["question_ids"]
    passage_sets = {qid: set(ids) for qid, ids in manifest["passage_ids"].items()}

    predictions = {}
    for path in sorted((run_dir / "predictions").glob("*.json")):
        record = read_json(path)
        predictions[record["question_id"]] = record
    calls = [read_json(path) for path in sorted((run_dir / "calls").glob("*.json"))]

    per_question = []
    correct = abstain = invalid = missing = 0
    cited_total = cited_legal = 0
    sentences_total = sentences_cited = 0
    outputs_all_legal = outputs_with_citations = 0
    for qid in question_ids:
        if qid not in gold:
            raise KeyError(f"question {qid} has no reference answer in {GOLD_FILES[split]}")
        pred = predictions.get(qid)
        entry: dict[str, Any] = {"question_id": qid, "gold": gold[qid]}
        if pred is None:
            missing += 1
            entry.update(status="missing", correct=False)
            per_question.append(entry)
            continue
        entry["answer"] = pred["answer"]
        if pred["validity"] != "valid":
            invalid += 1
            entry.update(status="invalid", reason=pred["invalid_reason"], correct=False)
            per_question.append(entry)
            continue
        is_correct = pred["answer"] == gold[qid]
        correct += is_correct
        abstain += pred["answer"] == "abstain"
        sentences = pred["output"]["explanation"]
        allowed = passage_sets[qid]
        cited = [pid for sentence in sentences for pid in sentence["passage_ids"]]
        illegal = [pid for pid in cited if pid not in allowed]
        sentences_total += len(sentences)
        sentences_cited += sum(1 for s in sentences if s["passage_ids"])
        cited_total += len(cited)
        cited_legal += len(cited) - len(illegal)
        if cited:
            outputs_with_citations += 1
            outputs_all_legal += not illegal
        entry.update(status="abstain" if pred["answer"] == "abstain" else "answered", correct=is_correct,
                     n_sentences=len(sentences), n_citations=len(cited), illegal_citations=illegal)
        per_question.append(entry)

    n = len(question_ids)
    usage = [c["usage"] for c in calls]
    input_tokens = _sum([u["input_tokens"] for u in usage])
    output_tokens = _sum([u["output_tokens"] for u in usage])
    reasoning_tokens = _sum([u["reasoning_tokens"] for u in usage])
    cached_tokens = _sum([u.get("cached_input_tokens") for u in usage])
    latencies = [c["total_latency_s"] for c in calls]
    cost = input_tokens / 1e6 * PRICE_INPUT_PER_M + output_tokens / 1e6 * PRICE_OUTPUT_PER_M

    return {
        "run_id": manifest["run_id"],
        "condition": manifest["condition"],
        "split": split,
        "note": ("DEVELOPMENT PILOT — not a held-out result." if split == "pilot" else "held-out evaluation split"),
        "complete": (run_dir / "complete.json").exists(),
        "models_returned": sorted({c["model_returned"] for c in calls if c["model_returned"]}),
        "answers": {
            "n_questions": n, "correct": correct, "accuracy": correct / n if n else None,
            "abstain": abstain, "invalid": invalid, "missing": missing,
        },
        "citations": {
            "sentences": sentences_total,
            "sentences_with_citation": sentences_cited,
            "cited_ids": cited_total,
            "cited_ids_in_Eq": cited_legal,
            "citation_validity_rate": cited_legal / cited_total if cited_total else None,
            "outputs_with_citations": outputs_with_citations,
            "outputs_all_citations_valid": outputs_all_legal,
        },
        "cost": {
            "calls": len(calls),
            "api_errors": sum(1 for c in calls if not c["api_ok"]),
            "retries": sum(c["attempts"] - 1 for c in calls),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens_incl_reasoning": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "mean_input_tokens_per_question": input_tokens / n if n else None,
            "mean_output_tokens_per_question": output_tokens / n if n else None,
            "mean_reasoning_tokens_per_question": reasoning_tokens / n if n else None,
            "latency_s_total": round(sum(latencies), 2),
            "latency_s_mean_per_call": round(statistics.mean(latencies), 2) if latencies else None,
            "latency_s_max": round(max(latencies), 2) if latencies else None,
            "prices_usd_per_million": {"input": PRICE_INPUT_PER_M, "output": PRICE_OUTPUT_PER_M},
            "estimated_cost_usd": round(cost, 4),
            "estimated_cost_usd_per_question": round(cost / n, 5) if n else None,
        },
        "per_question": per_question,
    }


def format_summary(score: dict[str, Any]) -> str:
    a, c, k = score["answers"], score["citations"], score["cost"]
    rate = c["citation_validity_rate"]
    lines = [
        f"Run {score['run_id']}  ({score['condition']}, {score['split']})  — {score['note']}",
        f"Model(s) returned: {', '.join(score['models_returned']) or '-'}; complete: {score['complete']}",
        f"Accuracy: {a['correct']}/{a['n_questions']} = {a['accuracy']:.3f}  "
        f"(abstain {a['abstain']}, invalid {a['invalid']}, missing {a['missing']}; all counted wrong)",
        f"Citations: {c['cited_ids_in_Eq']}/{c['cited_ids']} ids in E_q = "
        + (f"{rate:.3f}" if rate is not None else "n/a")
        + f"; {c['sentences_with_citation']}/{c['sentences']} sentences cite something; "
          f"{c['outputs_all_citations_valid']}/{c['outputs_with_citations']} outputs fully valid",
        f"Calls: {k['calls']} (api errors {k['api_errors']}, retries {k['retries']})",
        f"Tokens: input {k['input_tokens']}, output {k['output_tokens_incl_reasoning']} "
        f"(of which reasoning {k['reasoning_tokens']})",
        f"Per question: input {k['mean_input_tokens_per_question']:.0f}, output "
        f"{k['mean_output_tokens_per_question']:.0f}, reasoning {k['mean_reasoning_tokens_per_question']:.0f}",
        f"Latency: total {k['latency_s_total']} s, mean {k['latency_s_mean_per_call']} s/call, max {k['latency_s_max']} s",
        f"Estimated cost: ${k['estimated_cost_usd']:.4f} (${k['estimated_cost_usd_per_question']:.5f}/question)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="also write the full score as JSON (must not be inside the run dir)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args(argv)
    score = score_run(args.run_dir, args.dataset_dir)
    print(format_summary(score))
    if args.out:
        if args.run_dir.resolve() in args.out.resolve().parents:
            print("error: do not write into the frozen run directory", file=sys.stderr)
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
