#!/usr/bin/env python3
"""Freeze the thesis evaluation set from thesis-v0.1 and the author's review decisions.

The script never reads raw MedQA data. It takes candidate questions and their
reference answers from the parent dataset version, applies the recorded review
decisions, and writes a new self-contained dataset version in which model inputs
and evaluator references are stored separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION = "thesis-v0.2"
PARENT = "thesis-v0.1"
INPUT_FIELDS = ("id", "options", "question")
EXCLUSION_REASONS = {
    "out_of_scope",
    "essential_missing_exhibit",
    "near_duplicate_across_source_splits",
    "source_quality",
}
# Development files carried over unchanged from the parent version.
CARRIED = (
    "inputs/pilot_questions.jsonl",
    "inputs/reviewed_train_questions.jsonl",
    "evaluator_only/pilot_gold.jsonl",
    "evaluator_only/reviewed_train_gold.jsonl",
)


class FreezeError(ValueError):
    """A review decision or provenance invariant failed."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_parent(parent: Path) -> None:
    """Check every file listed in the parent MANIFEST.sha256."""
    listing = parent / "MANIFEST.sha256"
    if not listing.is_file():
        raise FreezeError(f"{listing} is missing")
    for line in listing.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if sha256(parent / relative) != digest:
            raise FreezeError(f"parent checksum mismatch: {relative}")


def validate_decisions(rows: list[dict[str, str]], candidate_ids: set[str]) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise FreezeError("duplicate ids in review decisions")
    if set(ids) != candidate_ids:
        missing, extra = candidate_ids - set(ids), set(ids) - candidate_ids
        raise FreezeError(f"decisions do not match candidates (missing={len(missing)}, extra={len(extra)})")
    for row in rows:
        where = f"decision {row['id'][:16]}"
        cardio, include, reason = row["is_cardiology"], row["include_for_evaluation"], row["exclusion_reason"]
        if cardio not in {"yes", "no"} or include not in {"yes", "no"}:
            raise FreezeError(f"{where}: labels must be resolved to yes/no before freezing")
        if include == "yes" and (cardio != "yes" or reason):
            raise FreezeError(f"{where}: an included item must be in scope and have no exclusion reason")
        if include == "no" and reason not in EXCLUSION_REASONS:
            raise FreezeError(f"{where}: excluded item needs one of {sorted(EXCLUSION_REASONS)}")
        if cardio == "no" and reason != "out_of_scope":
            raise FreezeError(f"{where}: out-of-scope items must use exclusion_reason=out_of_scope")
        for field in ("reason_and_evidence", "reviewer", "review_date"):
            if not row.get(field, "").strip():
                raise FreezeError(f"{where}: {field} is empty")


def freeze(parent: Path, decisions_dir: Path, output: Path, *, frozen_on: str) -> dict[str, Any]:
    if output.exists():
        raise FreezeError(f"{output} already exists; dataset versions are immutable")
    verify_parent(parent)

    questions = {row["id"]: row for row in read_jsonl(parent / "inputs/prospective_test_questions.jsonl")}
    answers = {row["id"]: row for row in read_jsonl(parent / "evaluator_only/prospective_test_answers.jsonl")}
    if set(questions) != set(answers):
        raise FreezeError("parent candidate questions and answers do not align")

    decisions = read_csv(decisions_dir / "prospective_test_decisions.csv")
    validate_decisions(decisions, set(questions))
    conflicts = read_csv(decisions_dir / "relevance_conflict_decisions.csv")
    near = read_csv(decisions_dir / "near_duplicate_decisions.csv")
    for row in conflicts:
        if row["adjudicated_label"] not in {"yes", "no"}:
            raise FreezeError("relevance conflicts must be adjudicated to yes/no")
    never_evaluate = set()
    for pair in near:
        if pair["same_underlying_item"] not in {"yes", "no"}:
            raise FreezeError("near-duplicate pairs must be resolved")
        if pair["same_underlying_item"] == "yes":
            never_evaluate.update({pair["left_id"], pair["right_id"]})

    included = sorted(row["id"] for row in decisions if row["include_for_evaluation"] == "yes")
    leaked = never_evaluate & set(included)
    if leaked:
        raise FreezeError(f"{len(leaked)} included item(s) belong to a duplicate pair")

    exclusion_counts = Counter(row["exclusion_reason"] for row in decisions if row["include_for_evaluation"] == "no")
    parent_manifest = parent / "manifest.json"
    counts = {
        "candidates_reviewed": len(decisions),
        "included_for_evaluation": len(included),
        "excluded": dict(sorted(exclusion_counts.items())),
        "in_scope_candidates": sum(row["is_cardiology"] == "yes" for row in decisions),
        "relevance_conflicts_adjudicated": len(conflicts),
        "near_duplicate_pairs_resolved": len(near),
        "never_evaluate_ids": len(never_evaluate),
    }
    manifest = {
        "dataset_version": VERSION,
        "parent_version": PARENT,
        "parent_manifest_sha256": sha256(parent_manifest),
        "status": "evaluation_set_frozen",
        "test_frozen": True,
        "frozen_on": frozen_on,
        "agent_input_fields": list(INPUT_FIELDS),
        "freeze_script_sha256": sha256(Path(__file__)),
        "review_decision_files": {path.name: sha256(path) for path in sorted(decisions_dir.glob("*")) if path.is_file()},
        "counts": counts,
        "evaluation_ids": included,
        "notes": [
            "Candidates were screened from the question stem only (see the parent manifest).",
            "Review used the question and options only; answer keys were not shown.",
            "Labels were drafted with LLM assistance (ChatGPT, Gemini, Claude) and finalized by the author under the rules in review/scope_rules.md.",
            "Selection did not depend on any system output or on whether any model answered correctly.",
            "Development files are carried over unchanged from the parent version.",
            "Records listed in quarantine/never_evaluate_ids.json must not enter any evaluation set.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".thesis-freeze-", dir=output.parent) as temporary:
        target = Path(temporary) / VERSION
        target.mkdir()

        def write(relative: str, text: str) -> None:
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        def jsonl(relative: str, rows: Any) -> None:
            write(relative, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))

        jsonl("inputs/test_questions.jsonl", ({key: questions[cid][key] for key in INPUT_FIELDS} for cid in included))
        jsonl("evaluator_only/test_answers.jsonl", ({"id": cid, "answer_idx": answers[cid]["answer_idx"],
                                                     "answer": answers[cid]["answer"],
                                                     "original_split": answers[cid]["original_split"]}
                                                    for cid in included))
        for relative in CARRIED:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent / relative, target / relative)
        (target / "review").mkdir()
        for path in sorted(decisions_dir.glob("*")):
            if path.is_file():
                shutil.copyfile(path, target / "review" / path.name)
        write("quarantine/never_evaluate_ids.json", json.dumps(sorted(never_evaluate), indent=2) + "\n")
        write("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        write("README.md", data_card(counts, frozen_on))
        listing = [sha256(path) + "  " + path.relative_to(target).as_posix()
                   for path in sorted(target.rglob("*")) if path.is_file()]
        write("MANIFEST.sha256", "\n".join(listing) + "\n")
        target.rename(output)
    return manifest


def data_card(counts: dict[str, Any], frozen_on: str) -> str:
    excluded = "\n".join(f"| Excluded: {reason.replace('_', ' ')} | {n} |" for reason, n in counts["excluded"].items())
    return f"""# Thesis dataset — {VERSION}

Status: **evaluation set frozen on {frozen_on}**. Parent version: `{PARENT}`.

| Item | Count |
| --- | ---: |
| Candidates reviewed | {counts['candidates_reviewed']} |
| Included for evaluation | {counts['included_for_evaluation']} |
{excluded}
| Relevance conflicts adjudicated | {counts['relevance_conflicts_adjudicated']} |
| Near-duplicate pairs resolved | {counts['near_duplicate_pairs_resolved']} |

## Files

- `inputs/test_questions.jsonl`: frozen evaluation questions. Each record has only `id`, `question`, `options`.
- `evaluator_only/test_answers.jsonl`: reference answers for scoring only. Never pass to an agent or index for retrieval.
- `inputs/pilot_questions.jsonl`, `evaluator_only/pilot_gold.jsonl`: unchanged 20-question development pilot. Not held-out.
- `inputs/reviewed_train_questions.jsonl`, `evaluator_only/reviewed_train_gold.jsonl`: unchanged development pool.
- `review/`: scope rules, labeling protocol and every review decision with its reason.
- `quarantine/never_evaluate_ids.json`: records that must not enter any evaluation set (near-duplicate pair).

## Rules of use

1. Prompts, retrieval settings and agent logic are tuned on the pilot only.
2. Freeze the system configuration (git tag) before the first run on `inputs/test_questions.jsonl`.
3. Report every test item, including abstentions and invalid outputs.
4. Do not change this version. Any later change to the evaluation set creates a new version.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "data/versions" / PARENT)
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/review" / VERSION)
    parser.add_argument("--output", type=Path, default=ROOT / "data/versions" / VERSION)
    parser.add_argument("--frozen-on", required=True, help="Freeze date, YYYY-MM-DD (recorded in the manifest)")
    args = parser.parse_args()
    manifest = freeze(args.parent, args.decisions, args.output, frozen_on=args.frozen_on)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
