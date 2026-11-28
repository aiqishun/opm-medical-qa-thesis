#!/usr/bin/env python3
"""Score a frozen run directory against the evaluator-only reference answers.

    python3 scripts/score_run.py experiments/runs/<run_id> [--out score.json]

This is the only experiment script that reads ``evaluator_only/``. It never
modifies the run directory. Counting rules (thesis protocol):

* accuracy denominator = every question in the run manifest; abstentions,
  invalid outputs and missing predictions all count as incorrect;
* citation validity = share of cited passage ids that belong to that
  question's fixed passage set E_q (computed over valid outputs);
* grounding (C1): every explanation sentence is labelled question / passages /
  none. ``question_quote`` must occur verbatim in the question stem after
  normalization (NFKC, case-fold, unified quotes and dashes, collapsed
  whitespace, outer punctuation stripped). Rule violations are reported, but they do
  not make an output invalid: the answer is still scored;
* cost = input tokens x $2/M + output tokens x $10/M. Output tokens include
  reasoning tokens (OpenAI usage convention), which are also reported separately.

Pilot results are for development only and must not be reported as held-out results.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
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


_TRANSLATE = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                            "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " "})
GROUNDINGS = ("question", "passages", "none")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).casefold()
    return re.sub(r"\s+", " ", text).strip()


def quote_in_stem(quote: str, stem: str) -> bool:
    core = normalize_text(quote).strip(" .,;:!?\"'()[]")
    return bool(core) and core in normalize_text(stem)


def sentence_violations(sentence: dict[str, Any]) -> list[str]:
    """Grounding rules of the C1 prompt that a well-formed sentence must follow."""
    grounding, quote, ids = sentence.get("grounding"), sentence.get("question_quote"), sentence["passage_ids"]
    problems = []
    if grounding == "question" and not (quote and quote.strip()):
        problems.append("question_without_quote")
    if grounding != "question" and quote is not None:
        problems.append("quote_on_non_question_sentence")
    if grounding == "passages" and not ids:
        problems.append("passages_without_citation")
    if grounding == "none" and ids:
        problems.append("none_with_citation")
    return problems


def load_stems(path: Path) -> dict[str, str]:
    stems = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                stems[record["id"]] = record["question"]
    return stems


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "zero": sum(1 for v in values if v == 0), "min": min(values), "max": max(values),
            "mean": round(statistics.mean(values), 1), "median": statistics.median(values)}


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

    stem_path = Path(manifest["inputs"]["questions"]["path"])
    stems = load_stems(stem_path if stem_path.is_absolute() else ROOT / stem_path)

    per_question = []
    g_counts = {g: 0 for g in GROUNDINGS}
    g_unlabelled = 0
    quotes_checked = quotes_verbatim = 0
    quote_lengths: list[int] = []
    p_cited_total = p_cited_legal = p_sentences = p_sentences_all_legal = 0
    violations: dict[str, int] = {}
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
        bad_quotes = []
        for sentence in sentences:
            grounding = sentence.get("grounding")
            if grounding in g_counts:
                g_counts[grounding] += 1
            else:
                g_unlabelled += 1
            for problem in sentence_violations(sentence):
                violations[problem] = violations.get(problem, 0) + 1
            if grounding == "question" and sentence.get("question_quote"):
                quotes_checked += 1
                quote_lengths.append(len(sentence["question_quote"]))
                if quote_in_stem(sentence["question_quote"], stems[qid]):
                    quotes_verbatim += 1
                else:
                    bad_quotes.append(sentence["question_quote"])
            if grounding == "passages":
                ids = sentence["passage_ids"]
                p_sentences += 1
                p_cited_total += len(ids)
                legal = sum(1 for pid in ids if pid in allowed)
                p_cited_legal += legal
                p_sentences_all_legal += bool(ids) and legal == len(ids)
        sentences_total += len(sentences)
        sentences_cited += sum(1 for s in sentences if s["passage_ids"])
        cited_total += len(cited)
        cited_legal += len(cited) - len(illegal)
        if cited:
            outputs_with_citations += 1
            outputs_all_legal += not illegal
        entry.update(status="abstain" if pred["answer"] == "abstain" else "answered", correct=is_correct,
                     n_sentences=len(sentences), n_citations=len(cited), illegal_citations=illegal,
                     grounding=[s_.get("grounding") for s_ in sentences], non_verbatim_quotes=bad_quotes)
        per_question.append(entry)

    n = len(question_ids)
    usage = [c["usage"] for c in calls]
    input_tokens = _sum([u["input_tokens"] for u in usage])
    output_tokens = _sum([u["output_tokens"] for u in usage])
    reasoning_tokens = _sum([u["reasoning_tokens"] for u in usage])
    cached_tokens = _sum([u.get("cached_input_tokens") for u in usage])
    latencies = [c["total_latency_s"] for c in calls]
    reasoning_per_question: dict[str, int] = {}
    for c in calls:
        if c["usage"]["reasoning_tokens"] is not None:
            reasoning_per_question[c["question_id"]] = (reasoning_per_question.get(c["question_id"], 0)
                                                        + c["usage"]["reasoning_tokens"])
    output_per_question: dict[str, int] = {}
    for c in calls:
        if c["usage"]["output_tokens"] is not None:
            output_per_question[c["question_id"]] = output_per_question.get(c["question_id"], 0) + c["usage"]["output_tokens"]
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
        "grounding": {
            "sentences": sentences_total,
            "counts": g_counts,
            "unlabelled": g_unlabelled,
            "shares": {g: (v / sentences_total if sentences_total else None) for g, v in g_counts.items()},
            "question_quotes_checked": quotes_checked,
            "question_quotes_verbatim": quotes_verbatim,
            "question_quote_verbatim_rate": quotes_verbatim / quotes_checked if quotes_checked else None,
            "question_quote_length_chars": _distribution(quote_lengths),
            "passages_sentences": p_sentences,
            "passages_sentences_all_ids_in_Eq": p_sentences_all_legal,
            "passages_cited_ids": p_cited_total,
            "passages_cited_ids_in_Eq": p_cited_legal,
            "passages_citation_validity_rate": p_cited_legal / p_cited_total if p_cited_total else None,
            "rule_violations": violations,
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
            "reasoning_tokens_per_question": _distribution(list(reasoning_per_question.values())),
            "output_tokens_per_question": _distribution(list(output_per_question.values())),
            "latency_s_total": round(sum(latencies), 2),
            "latency_s_mean_per_call": round(statistics.mean(latencies), 2) if latencies else None,
            "latency_s_max": round(max(latencies), 2) if latencies else None,
            "prices_usd_per_million": {"input": PRICE_INPUT_PER_M, "output": PRICE_OUTPUT_PER_M},
            "estimated_cost_usd": round(cost, 4),
            "estimated_cost_usd_per_question": round(cost / n, 5) if n else None,
        },
        "per_question": per_question,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _dist_text(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "n/a"
    return f"zero {d['zero']}/{d['n']}, min {d['min']}, median {d['median']}, mean {d['mean']}, max {d['max']}"


def _len_text(d: dict[str, Any]) -> str:
    return "n/a" if not d.get("n") else f"min {d['min']}, median {d['median']}, max {d['max']}"


def _grounding_line(g: dict[str, Any]) -> str:
    c, sh = g["counts"], g["shares"]
    return (f"Grounding ({g['sentences']} sentences): question {c['question']} ({_pct(sh['question'])}), "
            f"passages {c['passages']} ({_pct(sh['passages'])}), none {c['none']} ({_pct(sh['none'])})"
            + (f", unlabelled {g['unlabelled']}" if g["unlabelled"] else "")
            + f"\n  question_quote verbatim in stem: {g['question_quotes_verbatim']}/{g['question_quotes_checked']}"
              f" ({_pct(g['question_quote_verbatim_rate'])}); length chars: {_len_text(g['question_quote_length_chars'])}"
            + f"\n  passages sentences: {g['passages_sentences_all_ids_in_Eq']}/{g['passages_sentences']} with all ids in E_q;"
              f" ids {g['passages_cited_ids_in_Eq']}/{g['passages_cited_ids']} ({_pct(g['passages_citation_validity_rate'])})"
            + f"\n  rule violations: {g['rule_violations'] or 'none'}")


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
        _grounding_line(score["grounding"]),
        f"Calls: {k['calls']} (api errors {k['api_errors']}, retries {k['retries']})",
        f"Tokens: input {k['input_tokens']}, output {k['output_tokens_incl_reasoning']} "
        f"(of which reasoning {k['reasoning_tokens']})",
        f"Per question: input {k['mean_input_tokens_per_question']:.0f}, output "
        f"{k['mean_output_tokens_per_question']:.0f}, reasoning {k['mean_reasoning_tokens_per_question']:.0f}",
        f"Reasoning tokens per question: {_dist_text(k['reasoning_tokens_per_question'])}",
        f"Output tokens per question:    {_dist_text(k['output_tokens_per_question'])}",
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
