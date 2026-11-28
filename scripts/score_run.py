#!/usr/bin/env python3
"""Score a frozen run directory against the evaluator-only reference answers.

    python3 scripts/score_run.py experiments/runs/<run_id> [--out score.json]

This is the only experiment script that reads ``evaluator_only/``. It never
modifies the run directory. Counting rules (thesis protocol):

* accuracy denominator = every question in the run manifest; abstentions,
  invalid outputs and missing predictions all count as incorrect;
* citation validity = share of cited passage ids that belong to that
  question's fixed passage set E_q (computed over valid outputs);
* grounding: every explanation sentence (and, for C2, every graph element and
  relation) is labelled question / passages / none. ``question_quote`` must
  occur verbatim in the question stem after normalization (NFKC, case-fold,
  unified quotes and dashes, collapsed whitespace, outer punctuation stripped).
  Rule violations are reported, but they do not make an output invalid;
* graphs (C2): pass rates of the deterministic validator checks recorded at run
  time, graph sizes, and the share of valid support paths. A graph that fails
  validation is still a valid output; denominators are valid outputs with a graph;
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
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_graph.grounding import GROUNDINGS, grounding_violations, normalize_text, quote_in_stem  # noqa: E402,F401
from evidence_graph.validate import CHECKS  # noqa: E402

DATASET_DIR = ROOT / "data/versions/thesis-v0.2"
GOLD_FILES = {"pilot": "evaluator_only/pilot_gold.jsonl", "test": "evaluator_only/test_answers.jsonl"}
PRICE_INPUT_PER_M = 2.0
PRICE_OUTPUT_PER_M = 10.0

sentence_violations = grounding_violations  # name used by the C1 scorer


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


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


class GroundingTally:
    """Grounding statistics over a set of grounded items (sentences, elements or relations)."""

    def __init__(self) -> None:
        self.counts = {g: 0 for g in GROUNDINGS}
        self.unlabelled = 0
        self.quotes_checked = self.quotes_verbatim = 0
        self.quote_lengths: list[int] = []
        self.p_items = self.p_items_all_legal = self.p_cited = self.p_cited_legal = 0
        self.violations: dict[str, int] = {}

    def add(self, items: Iterable[dict[str, Any]], stem: str, allowed: set[str]) -> list[str]:
        """Tally items; returns the quotes that are not verbatim in the stem."""
        bad_quotes = []
        for item in items:
            grounding = item.get("grounding")
            if grounding in self.counts:
                self.counts[grounding] += 1
            else:
                self.unlabelled += 1
            for problem in grounding_violations(item):
                self.violations[problem] = self.violations.get(problem, 0) + 1
            if grounding == "question" and item.get("question_quote"):
                self.quotes_checked += 1
                self.quote_lengths.append(len(item["question_quote"]))
                if quote_in_stem(item["question_quote"], stem):
                    self.quotes_verbatim += 1
                else:
                    bad_quotes.append(item["question_quote"])
            if grounding == "passages":
                ids = item["passage_ids"]
                legal = sum(1 for pid in ids if pid in allowed)
                self.p_items += 1
                self.p_cited += len(ids)
                self.p_cited_legal += legal
                self.p_items_all_legal += bool(ids) and legal == len(ids)
        return bad_quotes

    def summary(self, unit: str) -> dict[str, Any]:
        total = sum(self.counts.values()) + self.unlabelled
        return {
            unit: total,
            "counts": self.counts,
            "unlabelled": self.unlabelled,
            "shares": {g: _ratio(v, total) for g, v in self.counts.items()},
            "question_quotes_checked": self.quotes_checked,
            "question_quotes_verbatim": self.quotes_verbatim,
            "question_quote_verbatim_rate": _ratio(self.quotes_verbatim, self.quotes_checked),
            "question_quote_length_chars": _distribution(self.quote_lengths),
            f"passages_{unit}": self.p_items,
            f"passages_{unit}_all_ids_in_Eq": self.p_items_all_legal,
            "passages_cited_ids": self.p_cited,
            "passages_cited_ids_in_Eq": self.p_cited_legal,
            "passages_citation_validity_rate": _ratio(self.p_cited_legal, self.p_cited),
            "rule_violations": self.violations,
        }


def _graph_items(graph: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return [item for key in keys for item in graph.get(key, [])]


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
    sentences_tally, elements_tally, relations_tally = GroundingTally(), GroundingTally(), GroundingTally()
    correct = abstain = invalid = missing = 0
    cited_total = cited_legal = 0
    sentences_total = sentences_cited = 0
    outputs_all_legal = outputs_with_citations = 0
    graphs: list[dict[str, Any]] = []
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
        output, allowed, stem = pred["output"], passage_sets[qid], stems[qid]
        sentences = output["explanation"]
        cited = [pid for sentence in sentences for pid in sentence["passage_ids"]]
        illegal = [pid for pid in cited if pid not in allowed]
        sentences_total += len(sentences)
        sentences_cited += sum(1 for s in sentences if s["passage_ids"])
        cited_total += len(cited)
        cited_legal += len(cited) - len(illegal)
        if cited:
            outputs_with_citations += 1
            outputs_all_legal += not illegal
        bad_quotes = sentences_tally.add(sentences, stem, allowed)
        entry.update(status="abstain" if pred["answer"] == "abstain" else "answered", correct=is_correct,
                     n_sentences=len(sentences), n_citations=len(cited), illegal_citations=illegal,
                     grounding=[s.get("grounding") for s in sentences], non_verbatim_quotes=bad_quotes)
        if "graph" in output:
            graph = output["graph"]
            validation = pred.get("extra", {}).get("validation")
            elements_tally.add(_graph_items(graph, ("objects", "processes", "states")), stem, allowed)
            relations_tally.add(graph["relations"], stem, allowed)
            graphs.append({"graph": graph, "validation": validation})
            entry["graph"] = {
                "size": {key: len(graph[key]) for key in ("objects", "processes", "states", "relations")},
                "all_checks_passed": validation["all_passed"] if validation else None,
                "failed_checks": [n for n, c in validation["checks"].items() if not c["passed"]] if validation else None,
                "support_path_valid": validation["support_path_valid"] if validation else None,
            }
        per_question.append(entry)

    n = len(question_ids)
    usage = [c["usage"] for c in calls]
    input_tokens = _sum([u["input_tokens"] for u in usage])
    output_tokens = _sum([u["output_tokens"] for u in usage])
    reasoning_tokens = _sum([u["reasoning_tokens"] for u in usage])
    cached_tokens = _sum([u.get("cached_input_tokens") for u in usage])
    latencies = [c["total_latency_s"] for c in calls]
    cost = input_tokens / 1e6 * PRICE_INPUT_PER_M + output_tokens / 1e6 * PRICE_OUTPUT_PER_M
    reasoning_per_question: dict[str, int] = {}
    output_per_question: dict[str, int] = {}
    for c in calls:
        u = c["usage"]
        if u["reasoning_tokens"] is not None:
            reasoning_per_question[c["question_id"]] = reasoning_per_question.get(c["question_id"], 0) + u["reasoning_tokens"]
        if u["output_tokens"] is not None:
            output_per_question[c["question_id"]] = output_per_question.get(c["question_id"], 0) + u["output_tokens"]

    score: dict[str, Any] = {
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
        "grounding": sentences_tally.summary("sentences"),
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
    if graphs:
        score["graph"] = _graph_summary(graphs, elements_tally, relations_tally)
    return score


def _graph_summary(graphs: list[dict[str, Any]], elements: GroundingTally,
                   relations: GroundingTally) -> dict[str, Any]:
    validated = [g["validation"] for g in graphs if g["validation"]]
    checks = {}
    for name in CHECKS:
        passed = sum(1 for v in validated if v["checks"][name]["passed"])
        checks[name] = {"passed": passed, "of": len(validated), "rate": _ratio(passed, len(validated)),
                        "issues": sum(len(v["checks"][name]["issues"]) for v in validated)}
    paths = [v["support_path_valid"] for v in validated]
    path_applicable = [p for p in paths if p is not None]
    return {
        "graphs": len(graphs),
        "validated": len(validated),
        "all_checks_passed": sum(1 for v in validated if v["all_passed"]),
        "checks": checks,
        "support_path_valid": sum(1 for p in path_applicable if p),
        "support_path_applicable": len(path_applicable),
        "support_path_valid_rate": _ratio(sum(1 for p in path_applicable if p), len(path_applicable)),
        "size": {key: _distribution([len(g["graph"][key]) for g in graphs])
                 for key in ("objects", "processes", "states", "relations")},
        "question_findings": _distribution([
            sum(1 for o in g["graph"]["objects"] if o["is_question_finding"])
            + sum(1 for s in g["graph"]["states"] if s["is_question_finding"]) for g in graphs]),
        "relation_types": _type_counts(graphs),
        "element_grounding": elements.summary("items"),
        "relation_grounding": relations.summary("items"),
    }


def _type_counts(graphs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for g in graphs:
        for rel in g["graph"]["relations"]:
            counts[rel["type"]] = counts.get(rel["type"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _dist_text(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "n/a"
    return f"zero {d['zero']}/{d['n']}, min {d['min']}, median {d['median']}, mean {d['mean']}, max {d['max']}"


def _len_text(d: dict[str, Any]) -> str:
    return "n/a" if not d.get("n") else f"min {d['min']}, median {d['median']}, max {d['max']}"


def _grounding_line(g: dict[str, Any], title: str, unit: str) -> str:
    c, sh = g["counts"], g["shares"]
    return (f"{title} ({g[unit]} {unit}): question {c['question']} ({_pct(sh['question'])}), "
            f"passages {c['passages']} ({_pct(sh['passages'])}), none {c['none']} ({_pct(sh['none'])})"
            + (f", unlabelled {g['unlabelled']}" if g["unlabelled"] else "")
            + f"\n  question_quote verbatim in stem: {g['question_quotes_verbatim']}/{g['question_quotes_checked']}"
              f" ({_pct(g['question_quote_verbatim_rate'])}); length chars: {_len_text(g['question_quote_length_chars'])}"
            + f"\n  passages {unit}: {g[f'passages_{unit}_all_ids_in_Eq']}/{g[f'passages_{unit}']} with all ids in E_q;"
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
        _grounding_line(score["grounding"], "Grounding", "sentences"),
    ]
    if "graph" in score:
        g = score["graph"]
        lines.append(f"Graphs: {g['graphs']} (validated {g['validated']}); all checks passed "
                     f"{g['all_checks_passed']}/{g['validated']}; support_path valid "
                     f"{g['support_path_valid']}/{g['support_path_applicable']} ({_pct(g['support_path_valid_rate'])})")
        for name, check in g["checks"].items():
            lines.append(f"  check {name:<18} {check['passed']}/{check['of']} ({_pct(check['rate'])}), issues {check['issues']}")
        for key, dist in g["size"].items():
            lines.append(f"  size {key:<10} {_dist_text(dist)}")
        lines.append(f"  question findings per graph: {_dist_text(g['question_findings'])}")
        lines.append(f"  relation types: {g['relation_types']}")
        lines.append(_grounding_line(g["element_grounding"], "Element grounding", "items"))
        lines.append(_grounding_line(g["relation_grounding"], "Relation grounding", "items"))
    lines += [
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
