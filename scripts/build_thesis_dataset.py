#!/usr/bin/env python3
"""Build an auditable development dataset and a pending evaluation review queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from prepare_medqa import HIGH_CONFIDENCE_DISEASE_TERMS, STRICT_CARDIOLOGY_KEYWORDS


ROOT = Path(__file__).resolve().parents[1]
VERSION = "thesis-v0.1"
TASK = "medqa_opm_task"
SOURCES = {
    "dev": f"{TASK}/data/processed/medqa_cardiology_llm_relevant.jsonl",
    "test": f"{TASK}/data/processed/medqa_cardiology_real_high_confidence.jsonl",
}
AUDITS = {
    "dev": "annotations/manual_audit_llm_supervised_v1.csv",
    "test": "annotations/manual_audit_baseline_v2.csv",
}
TERMS = tuple(sorted(set(HIGH_CONFIDENCE_DISEASE_TERMS) | set(STRICT_CARDIOLOGY_KEYWORDS)))
VISUAL = re.compile(r"\b(?:figure|image|photograph|diagram|radiograph)\b|\bshown (?:above|below)\b", re.I)


class DatasetError(ValueError):
    """An input or provenance invariant failed."""


def normalize(text: str) -> str:
    """Normalize canonical Unicode/whitespace; preserve case, signs, and numbers."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def case_id(row: dict[str, Any]) -> str:
    return "mq_" + digest([normalize(row["question"]),
                           {k: normalize(v) for k, v in row["options"].items()}])


def stem_id(question: str) -> str:
    return "stem_" + digest(normalize(question))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def validate_mcq(row: dict[str, Any], location: str) -> None:
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise DatasetError(f"{location}: missing question")
    options = row.get("options")
    if (not isinstance(options, dict) or len(options) < 2 or
            any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                for k, v in options.items())):
        raise DatasetError(f"{location}: invalid options")
    answer = row.get("answer_idx")
    if not isinstance(answer, str) or answer not in options:
        raise DatasetError(f"{location}: answer_idx does not identify an option")
    if not isinstance(row.get("answer"), str) or normalize(options[answer]) != normalize(row["answer"]):
        raise DatasetError(f"{location}: answer text and answer_idx disagree")
    if row.get("split") not in {"train", "dev", "test"}:
        raise DatasetError(f"{location}: unknown original split")


def label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"yes", "true"}:
        return True
    if lowered in {"no", "false"}:
        return False
    raise DatasetError(f"Unresolved/non-boolean legacy label: {value!r}")


def question_terms(row: dict[str, Any]) -> list[str]:
    """Screen on the question alone, never options, labels, or answer text."""
    text = normalize(row["question"]).casefold()
    return [term for term in TERMS if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)]


def agent_input(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": case_id(row), "question": row["question"], "options": row["options"]}


def choose_pilot(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or len(rows) < count:
        raise DatasetError(f"Requested {count} pilot records; only {len(rows)} eligible records")
    return sorted(rows, key=lambda row: digest([seed, case_id(row)]))[:count]


def near_pairs(rows: dict[str, dict[str, Any]], test_ids: set[str]) -> list[dict[str, Any]]:
    """Find lexical review candidates; do not merge or adjudicate them."""
    token_sets = {key: set(re.findall(r"\w+", normalize(row["question"]).casefold()))
                  for key, row in rows.items()}
    results = []
    for left in sorted(test_ids):
        a = normalize(rows[left]["question"]).casefold()
        for right in sorted(rows):
            if left == right or (right in test_ids and right < left):
                continue
            if stem_id(rows[left]["question"]) == stem_id(rows[right]["question"]):
                continue
            b = normalize(rows[right]["question"]).casefold()
            if min(len(a), len(b)) / max(len(a), len(b)) < 0.85:
                continue
            union = token_sets[left] | token_sets[right]
            overlap = len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0
            if overlap < 0.90:
                continue
            similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if similarity >= 0.92:
                results.append({"left_id": left, "right_id": right,
                                "token_jaccard": round(overlap, 6),
                                "sequence_similarity": round(similarity, 6),
                                "status": "manual_review_required_not_merged"})
    return results


def build(project: Path, raw_path: Path, output: Path, *, pilot_size: int = 20,
          seed: int = 42) -> dict[str, Any]:
    if output.exists():
        raise DatasetError(f"Output already exists; choose a new directory: {output}")
    source_inventory: list[dict[str, Any]] = []

    def inventory(path: Path, role: str, relative: str | None = None) -> None:
        source_inventory.append({"path": relative or path.relative_to(project).as_posix(),
                                 "role": role, "bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    raw = read_jsonl(raw_path)
    inventory(raw_path, "local_raw_source_not_bundled", "external_raw/" + raw_path.name)
    raw_index: dict[str, list[int]] = defaultdict(list)
    raw_stems: dict[str, set[str]] = defaultdict(set)
    representatives: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw):
        validate_mcq(row, f"raw:{index}")
        cid = case_id(row)
        if cid in representatives and representatives[cid]["answer_idx"] != row["answer_idx"]:
            raise DatasetError(f"Conflicting MCQ answers for {cid}")
        raw_index[cid].append(index)
        raw_stems[stem_id(row["question"])].add(cid)
        representatives.setdefault(cid, row)

    split_sets = {cid: {raw[i]["split"] for i in indexes} for cid, indexes in raw_index.items()}
    cross_split_stems = {sid for sid, ids in raw_stems.items()
                         if len(set().union(*(split_sets[cid] for cid in ids))) > 1}
    pool: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed = {}
    for benchmark_split, relative in SOURCES.items():
        path = project / relative
        current = read_jsonl(path)
        processed[benchmark_split] = current
        inventory(path, "inherited_processed_candidate_source")
        for index, row in enumerate(current):
            validate_mcq(row, f"{relative}:{index}")
            cid = case_id(row)
            if cid not in raw_index:
                raise DatasetError(f"{relative}:{index}: cannot trace record to raw source")
            if row["split"] not in split_sets[cid]:
                raise DatasetError(f"{relative}:{index}: original split mismatch")
            if row["answer_idx"] != representatives[cid]["answer_idx"]:
                raise DatasetError(f"{relative}:{index}: source answer disagrees with raw")
            pool[cid] = representatives[cid]
            occurrences[cid].append({"path": relative, "row_index": index,
                                     "original_split": row["split"]})

    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    packaged_counts = {}
    for benchmark_split, relative in AUDITS.items():
        path = project / relative
        with path.open(encoding="utf-8", newline="") as handle:
            audit_rows = list(csv.DictReader(handle))
        inventory(path, "inherited_manual_cardiology_annotations")
        public_relative = f"{TASK}/data/input/{benchmark_split}_questions.jsonl"
        public_rows = read_jsonl(project / public_relative)
        inventory(project / public_relative, "inherited_packaged_input")
        packaged = {row["id"]: row for row in public_rows}
        if len(packaged) != len(public_rows):
            raise DatasetError(f"{public_relative}: duplicate legacy IDs")
        gold_relative = (f"{TASK}/data/reference/dev_labels.jsonl" if benchmark_split == "dev"
                         else f"{TASK}/answer/cardio_gold_test.jsonl")
        gold_rows = read_jsonl(project / gold_relative)
        gold = {row["id"]: row for row in gold_rows}
        inventory(project / gold_relative, "inherited_packaged_labels")
        if len(gold) != len(gold_rows) or set(packaged) != set(gold):
            raise DatasetError(f"{gold_relative}: duplicate or mismatched IDs")
        seen = set()
        splits = Counter()
        for line_number, annotation in enumerate(audit_rows, 2):
            legacy_id = annotation["sample_id"]
            if legacy_id in seen or not re.fullmatch(r"q\d{4}", legacy_id):
                raise DatasetError(f"{relative}:{line_number}: duplicate/invalid sample_id")
            seen.add(legacy_id)
            index = int(legacy_id[1:])
            if index >= len(processed[benchmark_split]):
                raise DatasetError(f"{relative}:{line_number}: source index out of range")
            row = processed[benchmark_split][index]
            cid = case_id(row)
            if legacy_id not in packaged or case_id(packaged[legacy_id]) != cid:
                raise DatasetError(f"{relative}:{line_number}: packaged question mapping mismatch")
            keep = label(annotation["keep_for_cardiology_dataset"])
            if label(gold[legacy_id]["is_cardiology"]) != keep:
                raise DatasetError(f"{relative}:{line_number}: packaged label disagrees with annotation")
            splits[row["split"]] += 1
            annotations[cid].append({"id": cid, "benchmark_split": benchmark_split,
                                     "legacy_id": legacy_id, "annotation_file": relative,
                                     "csv_line": line_number, "is_cardiology": keep,
                                     "reason": annotation["reason"],
                                     "expected_topic": annotation["expected_topic"],
                                     "cardiology_relevance": annotation["cardiology_relevance"]})
        if seen != set(packaged):
            raise DatasetError(f"{relative}: audit IDs do not match packaged IDs")
        packaged_counts[benchmark_split] = dict(sorted(splits.items()))

    exposed_stems = {stem_id(row["question"]) for row in pool.values()}
    for relative in ["annotations/manual_sanity_check_20.csv", "annotations/opm_extraction_gold_20.csv"]:
        path = project / relative
        inventory(path, "previous_graph_annotation_exposure")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                exposed_stems.add(stem_id(row["question"]))

    conflicts = {cid for cid, ann in annotations.items()
                 if len({row["is_cardiology"] for row in ann}) > 1}
    agreed = {cid: ann[0]["is_cardiology"] for cid, ann in annotations.items() if cid not in conflicts}
    train_ids = {cid for cid in agreed if split_sets[cid] == {"train"}
                 and stem_id(pool[cid]["question"]) not in cross_split_stems}
    eligible = [pool[cid] for cid in sorted(train_ids) if agreed[cid]
                and len(raw_stems[stem_id(pool[cid]["question"])]) == 1]
    pilot = choose_pilot(eligible, pilot_size, seed)

    prospective: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for cid, row in sorted(representatives.items()):
        sid = stem_id(row["question"])
        if sid in exposed_stems or sid in cross_split_stems:
            continue
        if len(split_sets[cid]) != 1 or row["split"] not in prospective:
            continue
        if question_terms(row):
            prospective[row["split"]].append(row)
    test_ids = {case_id(row) for row in prospective["test"]}
    comparison = dict(pool)
    comparison.update({case_id(row): row for rows in prospective.values() for row in rows})
    near = near_pairs(comparison, test_ids)
    near_ids = {row[key] for row in near for key in ("left_id", "right_id")}

    def reference(row: dict[str, Any], cardio: bool | None = None) -> dict[str, Any]:
        cid = case_id(row)
        return {"id": cid, "answer_idx": row["answer_idx"], "answer": row["answer"],
                "original_split": row["split"], "is_cardiology": cardio,
                "raw_row_indexes": raw_index[cid]}

    duplicate_rows = []
    for cid, ann in sorted(annotations.items()):
        for left, right in itertools.combinations(ann, 2):
            if left["benchmark_split"] != right["benchmark_split"]:
                duplicate_rows.append({"id": cid, "left_split": left["benchmark_split"],
                                       "left_legacy_id": left["legacy_id"],
                                       "right_split": right["benchmark_split"],
                                       "right_legacy_id": right["legacy_id"],
                                       "label_conflict": cid in conflicts})

    counts = {"raw_rows": len(raw), "raw_unique_cases": len(raw_index),
              "raw_original_splits": dict(sorted(Counter(row["split"] for row in raw).items())),
              "raw_repeated_stem_groups": sum(len(ids) > 1 for ids in raw_stems.values()),
              "raw_cross_split_stem_groups": len(cross_split_stems),
              "legacy_source_occurrences": sum(len(rows) for rows in processed.values()),
              "legacy_unique_candidates": len(pool), "legacy_unique_reviewed_cases": len(annotations),
              "legacy_cross_split_duplicate_pairs": len(duplicate_rows),
              "legacy_packaged_original_splits": packaged_counts,
              "relevance_conflicts_quarantined": len(conflicts),
              "consensus_positive_cases": sum(agreed.values()),
              "consensus_negative_cases": sum(not value for value in agreed.values()),
              "reviewed_train_cases": len(train_ids),
              "eligible_positive_train_cases": len(eligible), "pilot_cases": len(pilot),
              "prospective_train_candidates": len(prospective["train"]),
              "prospective_test_candidates": len(prospective["test"]),
              "prospective_test_visual_reference_flags": sum(bool(VISUAL.search(row["question"])) for row in prospective["test"]),
              "near_duplicate_review_pairs": len(near)}
    inventory(project / "scripts/prepare_medqa.py", "inherited_answer_aware_filter_reference")
    inventory(project / f"{TASK}/reference/MedQA_LICENSE", "bundled_upstream_license")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"dataset_version": VERSION,
                "status": "development_ready_evaluation_review_required", "test_frozen": False,
                "id_policy": "SHA256 of NFC/whitespace-normalized question and key-sorted options; case/punctuation preserved",
                "agent_input_fields": ["id", "question", "options"],
                "pilot_seed": seed, "question_only_screening_terms": TERMS,
                "candidate_screening_fields": ["question"], "build_script_sha256": script_hash,
                "counts": counts, "sources": sorted(source_inventory, key=lambda row: row["path"]),
                "notes": ["No cardiology labels were adjudicated or overwritten.",
                          "Legacy benchmark dev/test are not original MedQA train/test.",
                          "The current inherited filter reads gold answer/explanation fields; historical filtering is not used as a blind evaluation protocol.",
                          "Candidate membership is not a clinical relevance judgment.",
                          "Source-reported train/test membership is traced to the local raw file; upstream release provenance is not independently certified.",
                          "Lexical near-duplicate checks do not establish absence of semantic duplicates or pretraining contamination."]}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".thesis-data-", dir=output.parent) as temporary:
        target = Path(temporary) / VERSION
        target.mkdir()

        def text(relative: str, value: str) -> None:
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")

        def jsonl(relative: str, rows: Any) -> None:
            text(relative, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))

        def csvfile(relative: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        jsonl("inputs/pilot_questions.jsonl", map(agent_input, pilot))
        jsonl("inputs/reviewed_train_questions.jsonl", (agent_input(pool[cid]) for cid in sorted(train_ids)))
        jsonl("inputs/prospective_test_questions.jsonl", map(agent_input, prospective["test"]))
        jsonl("evaluator_only/pilot_gold.jsonl", (reference(row, True) for row in pilot))
        jsonl("evaluator_only/reviewed_train_gold.jsonl", (reference(pool[cid], agreed[cid]) for cid in sorted(train_ids)))
        jsonl("evaluator_only/prospective_test_answers.jsonl", (reference(row) for row in prospective["test"]))
        jsonl("catalog/legacy_records.jsonl", ({"id": cid, "stem_id": stem_id(row["question"]),
              "original_splits": sorted(split_sets[cid]), "raw_row_indexes": raw_index[cid],
              "source_occurrences": occurrences[cid],
              "exposure_status": "previous_pipeline_candidate_excluded_from_prospective_evaluation"}
              for cid, row in sorted(pool.items())))
        jsonl("catalog/reviewed_annotations.jsonl", (row for cid in sorted(annotations) for row in annotations[cid]))
        jsonl("quarantine/relevance_conflicts.jsonl", ({**agent_input(pool[cid]),
              "original_split": pool[cid]["split"], "annotations": annotations[cid],
              "status": "unresolved_no_label_selected"} for cid in sorted(conflicts)))
        csvfile("audit/legacy_cross_split_duplicates.csv", duplicate_rows,
                ["id", "left_split", "left_legacy_id", "right_split", "right_legacy_id", "label_conflict"])
        jsonl("audit/near_duplicate_review.jsonl", near)
        jsonl("audit/raw_stem_variants.jsonl", ({"stem_id": sid, "case_ids": sorted(ids),
              "status": "option_variant_review_required"} for sid, ids in sorted(raw_stems.items()) if len(ids) > 1))
        conflict_queue = []
        for cid in sorted(conflicts):
            ann = annotations[cid]
            conflict_queue.append({"id": cid, "question": pool[cid]["question"],
                "options_json": json.dumps(pool[cid]["options"], ensure_ascii=False, sort_keys=True),
                "original_annotations_json": json.dumps(ann, ensure_ascii=False, sort_keys=True),
                "adjudicated_label": "", "reviewer": "", "review_date": "", "evidence_and_reason": ""})
        csvfile("review/relevance_conflicts.csv", conflict_queue,
                ["id", "question", "options_json", "original_annotations_json", "adjudicated_label", "reviewer", "review_date", "evidence_and_reason"])
        queue = [{"id": case_id(row), "question": row["question"],
                  "options_json": json.dumps(row["options"], ensure_ascii=False, sort_keys=True),
                  "question_only_terms": " | ".join(question_terms(row)),
                  "visual_reference_flag": bool(VISUAL.search(row["question"])),
                  "near_duplicate_flag": case_id(row) in near_ids,
                  "option_variant_flag": len(raw_stems[stem_id(row["question"])]) > 1,
                  "is_cardiology": "", "include_for_evaluation": "", "reviewer": "",
                  "review_date": "", "reason_and_evidence": ""} for row in prospective["test"]]
        csvfile("review/prospective_test_screening.csv", queue,
                ["id", "question", "options_json", "question_only_terms", "visual_reference_flag", "near_duplicate_flag", "option_variant_flag", "is_cardiology", "include_for_evaluation", "reviewer", "review_date", "reason_and_evidence"])
        near_queue = []
        for pair in near:
            left, right = comparison[pair["left_id"]], comparison[pair["right_id"]]
            near_queue.append({**{key: pair[key] for key in ("left_id", "right_id", "token_jaccard", "sequence_similarity")},
                               "left_original_split": left["split"], "right_original_split": right["split"],
                               "left_question": left["question"], "right_question": right["question"],
                               "left_options_json": json.dumps(left["options"], ensure_ascii=False, sort_keys=True),
                               "right_options_json": json.dumps(right["options"], ensure_ascii=False, sort_keys=True),
                               "same_underlying_item": "", "partition_decision": "", "reviewer": "",
                               "review_date": "", "reason_and_evidence": ""})
        csvfile("review/near_duplicate_pairs.csv", near_queue,
                ["left_id", "right_id", "token_jaccard", "sequence_similarity", "left_original_split", "right_original_split",
                 "left_question", "right_question", "left_options_json", "right_options_json", "same_underlying_item",
                 "partition_decision", "reviewer", "review_date", "reason_and_evidence"])
        text("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        text("README.md", data_card(counts))
        checksums = []
        for path in sorted(target.rglob("*")):
            if path.is_file():
                checksums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(target).as_posix())
        text("MANIFEST.sha256", "\n".join(checksums) + "\n")
        target.rename(output)
    return manifest


def data_card(counts: dict[str, Any]) -> str:
    return f'''# Thesis dataset preparation — {VERSION}

Status: **development-ready; evaluation review required; no final test set frozen**.

## What was prepared

| Item | Count |
| --- | ---: |
| Local raw records | {counts['raw_rows']} |
| Inherited processed occurrences | {counts['legacy_source_occurrences']} |
| Unique inherited candidates | {counts['legacy_unique_candidates']} |
| Unique manually reviewed cases | {counts['legacy_unique_reviewed_cases']} |
| Cross-package dev/test duplicate pairs | {counts['legacy_cross_split_duplicate_pairs']} |
| Conflicting cardiology labels, quarantined | {counts['relevance_conflicts_quarantined']} |
| Consistent positive / negative labels | {counts['consensus_positive_cases']} / {counts['consensus_negative_cases']} |
| Reviewed original-train cases with consistent labels | {counts['reviewed_train_cases']} |
| Eligible positive original-train cases | {counts['eligible_positive_train_cases']} |
| Fixed-seed development pilot | {counts['pilot_cases']} |
| Fresh question-only original-test candidates, pending review | {counts['prospective_test_candidates']} |
| Fresh original-train candidates, counted but not yet reviewed | {counts['prospective_train_candidates']} |
| Test candidate visual-reference flags | {counts['prospective_test_visual_reference_flags']} |
| Lexical near-duplicate pairs requiring review | {counts['near_duplicate_review_pairs']} |

## Inputs and evaluation references

- `inputs/pilot_questions.jsonl`: selected from consistent positive, source-reported train cases; suitable for engineering and prompt development, not a held-out result.
- `inputs/reviewed_train_questions.jsonl`: reviewed original-train records, including positive and negative relevance examples; use the intended task-specific subset.
- `inputs/prospective_test_questions.jsonl`: candidate pool only. Do not use for prompt tuning or call it a finalized cardiology benchmark.
- `evaluator_only/`: MCQ answer references and inherited relevance labels. The agent runner must load only the relevant `inputs/` file. Directory naming is not an access-control boundary; keep evaluator material out of agent tools and retrieval.

Each model-input record contains only `id`, `question`, and `options`. It does not carry answer keys, relevance labels, filter scores, matched terms, prior model decisions, or annotation reasons. Evaluation reference IDs match the corresponding input file exactly.

## Provenance and known limitations

The local raw source has source-reported splits {json.dumps(counts['raw_original_splits'], sort_keys=True)}. No separate upstream development file was present. Raw release provenance is not independently certified; its exact file checksum is pinned in `manifest.json`.

The inherited packaged development questions originated from {json.dumps(counts['legacy_packaged_original_splits']['dev'], sort_keys=True)}, and its packaged test questions from {json.dumps(counts['legacy_packaged_original_splits']['test'], sort_keys=True)}. Those legacy package names must not be mistaken for original MedQA partitions. All previous candidate question stems and previous graph annotations are excluded from the prospective evaluation pool, including option variants with an already exposed stem.

The current inherited keyword selector examines answer/explanation fields. The new screening function examines the question alone and records its vocabulary in the manifest. Keyword membership is not a human judgment of the main tested concept. The initial source-release split and duplicate checks do not prove semantic independence or absence of model pretraining exposure.

Canonical IDs hash the question plus sorted option keys after NFC/whitespace normalization. Case, punctuation, units, signs, numbers and option labels are preserved. Original row positions and `(benchmark_split, legacy_id)` annotations remain in `catalog/`. Different options can produce different case IDs; stem grouping identifies those variants. Lexical near matches are flagged for review and never automatically merged.

## Review before freezing an evaluation version

1. Review all rows in `review/prospective_test_screening.csv` against the prespecified criterion: the primary tested concept is cardiovascular, rather than incidental history, vital signs, or a distractor. Record yes/no/unclear, inclusion decision, reviewer/date, and reason. Reviewers should not use model success to choose questions.
2. Resolve visual-reference, option-variant, and near-duplicate flags; record whether essential source material is available. Compare both stems/options in `review/near_duplicate_pairs.csv` without exposing answer keys. Do not equate a visual-reference flag with a proven defective question.
3. Keep the two conflicting cases isolated until documented adjudication in `review/relevance_conflicts.csv`. No existing label has been changed.
4. Freeze a subsequent version with final IDs, selection criteria, reference answers, source/corpus hashes, and separate evaluator access. Do not overwrite this version or the inherited data.
5. Finish prompt/model selection on development data, then evaluate the frozen test once under the fixed protocol. Document any later protocol change.

The review spreadsheet intentionally omits MCQ answer keys. OPM mechanism annotations and independent expert judgments have not been created by this build. Existing relevance labels are not gold OPM graphs.

## Reproduce

From the repository root, with the original local raw JSONL available:

```bash
python scripts/build_thesis_dataset.py --raw-source /path/to/medqa_full.jsonl --output /tmp/thesis-data-rebuild --pilot-size 20 --seed 42
```

Use the raw checksum in `manifest.json`; the full raw file is not newly bundled here. The builder checks MCQ keys/text, legacy row mappings and labels, refuses an existing output directory, and writes only a new version. For a fixed input/code version it produces deterministic artifacts. `MANIFEST.sha256` covers every generated file except itself.

MedQA source notices are preserved at `../../../medqa_opm_task/reference/MedQA_LICENSE` and `MedQA_README.md`. This is research data preparation, not clinical validation.
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/versions" / VERSION)
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = build(args.project_root.resolve(), args.raw_source.resolve(),
                       args.output.resolve(), pilot_size=args.pilot_size, seed=args.seed)
    except (DatasetError, OSError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
