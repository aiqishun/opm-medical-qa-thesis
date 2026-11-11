#!/usr/bin/env python3
"""Build a clean, self-contained delivery package for medqa_opm_task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import jsonschema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_SOURCE = PROJECT_ROOT / "medqa_opm_task"
DEFAULT_DELIVERY_ROOT = PROJECT_ROOT / "delivery"
PUBLIC_FIELDS = (
    "question",
    "options",
    "metamap_phrases",
    "matched_terms",
    "filter_confidence",
)
ROOT_FILES = (
    "task.json",
    "prompt.md",
    "README.md",
    "rubrics.md",
    "judge_prompt.md",
    "requirements.txt",
)
STATIC_ANSWER_FILES = (
    "opm_gold.jsonl",
    "expected_output.md",
)
LEGACY_SCRIPTS = (
    "prepare_medqa.py",
    "llm_filter_medqa.py",
    "llm_route_audit.py",
    "run_qa.py",
    "run_batch_qa.py",
    "run_baseline_comparison.py",
    "audit_batch_results.py",
    "evaluate_opm_graphs_auto.py",
    "inspect_medqa_schema.py",
    "export_manual_eval_sample.py",
)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".git"),
    )


def parse_index(item_id: str, source_size: int) -> int:
    if len(item_id) != 5 or not item_id.startswith("q") or not item_id[1:].isdigit():
        raise ValueError(f"invalid sample ID: {item_id!r}")
    index = int(item_id[1:])
    if index >= source_size:
        raise ValueError(
            f"sample ID {item_id} maps to {index}, source has {source_size} records"
        )
    return index


def public_records(
    audit_rows: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for audit in audit_rows:
        item_id = audit["sample_id"]
        if item_id in seen:
            raise ValueError(f"duplicate {split} ID: {item_id}")
        seen.add(item_id)
        source_index = parse_index(item_id, len(source_records))
        source = source_records[source_index]
        if not isinstance(source.get("question"), str) or not source["question"].strip():
            raise ValueError(f"{split}/{item_id}: source question is missing")
        record: dict[str, Any] = {
            "id": item_id,
            "split": split,
            "source_record_index": source_index,
        }
        for field in PUBLIC_FIELDS:
            if field in source:
                record[field] = source[field]
        output.append(record)
    return output


def dev_labels(audit_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in audit_rows:
        raw_label = row["keep_for_cardiology_dataset"].strip().lower()
        if raw_label not in {"yes", "no"}:
            raise ValueError(
                f"dev/{row['sample_id']}: unsupported label {raw_label!r}"
            )
        output.append(
            {
                "id": row["sample_id"],
                "is_cardiology": raw_label == "yes",
                "reason": row["reason"],
            }
        )
    return output


def hidden_cardio_labels(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in audit_rows:
        raw_label = row["keep_for_cardiology_dataset"].strip().lower()
        if raw_label not in {"yes", "no"}:
            raise ValueError(
                f"test/{row['sample_id']}: unsupported label {raw_label!r}"
            )
        output.append(
            {
                "id": row["sample_id"],
                "is_cardiology": raw_label,
                "reason": row["reason"],
            }
        )
    return output


def hidden_topic_labels(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "id": row["sample_id"],
            "expected_topic": row["expected_topic"],
            "error_type": row["error_type"],
        }
        for row in audit_rows
    ]


def classification_metrics(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    gold = {
        row["id"]: str(row["is_cardiology"]).strip().lower()
        in {"true", "yes", "1", "y", "t"}
        for row in gold_rows
    }
    validator = jsonschema.Draft7Validator(schema)
    valid = [row for row in prediction_rows if not list(validator.iter_errors(row))]
    predictions = {row["id"]: row["is_cardiology"] for row in valid if row["id"] in gold}
    tp = sum(predictions.get(item_id) is True and label for item_id, label in gold.items())
    fp = sum(predictions.get(item_id) is True and not label for item_id, label in gold.items())
    fn = sum(predictions.get(item_id) is not True and label for item_id, label in gold.items())
    tn = sum(predictions.get(item_id) is False and not label for item_id, label in gold.items())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "rows": len(prediction_rows),
        "schema_valid_rows": len(valid),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
    }


def write_trace_summary(target: Path) -> None:
    gold = load_jsonl(TASK_SOURCE / "answer" / "cardio_gold_test.jsonl")
    schema = json.loads(
        (TASK_SOURCE / "data" / "contracts" / "filter_results.schema.json").read_text(
            encoding="utf-8"
        )
    )
    runs: list[dict[str, Any]] = []
    predictions_by_run: list[dict[str, bool]] = []
    for run_id in (1, 2):
        source = TASK_SOURCE / "answer" / f"filter_results_run{run_id}.jsonl"
        rows = load_jsonl(source)
        copy_file(source, target / source.name)
        predictions_by_run.append({row["id"]: row["is_cardiology"] for row in rows})
        runs.append(
            {
                "run_id": run_id,
                "file": source.name,
                "file_modified_at": datetime.fromtimestamp(
                    source.stat().st_mtime
                ).astimezone().isoformat(),
                "model": "gpt-5.6-sol",
                "temperature": 0.2,
                "seed": None,
                "seed_status": "unavailable; not supplied or returned by the runner",
                "provider_version": None,
                "endpoint_revision": None,
                "token_usage": None,
                "latency": None,
                "metrics": classification_metrics(gold, rows, schema),
            }
        )

    common_ids = set(predictions_by_run[0]) & set(predictions_by_run[1])
    agreements = sum(
        predictions_by_run[0][item_id] == predictions_by_run[1][item_id]
        for item_id in common_ids
    )
    summary = {
        "trace_version": "1.0",
        "gold_records": len(gold),
        "runs": runs,
        "cross_run": {
            "common_ids": len(common_ids),
            "agreement_count": agreements,
            "disagreement_count": len(common_ids) - agreements,
            "agreement_rate": agreements / len(common_ids) if common_ids else 0.0,
        },
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest(package_root: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(package_root).as_posix()}")
    (package_root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_deterministic_zip(package_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
            archive_path = Path(package_root.name) / path.relative_to(package_root)
            info = zipfile.ZipInfo(archive_path.as_posix(), date_time=ZIP_TIMESTAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes())


def build(delivery_root: Path) -> tuple[Path, Path]:
    delivery_root.mkdir(parents=True, exist_ok=True)
    package_root = delivery_root / "medqa_opm_task"
    temporary_root = delivery_root / ".medqa_opm_task.building"
    zip_path = delivery_root / "medqa_opm_task.zip"

    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir()

    for filename in ROOT_FILES:
        copy_file(TASK_SOURCE / filename, temporary_root / filename)
    for directory in ("env", "tools"):
        copy_tree(TASK_SOURCE / directory, temporary_root / directory)
    copy_tree(TASK_SOURCE / "data" / "contracts", temporary_root / "data" / "contracts")
    copy_file(TASK_SOURCE / "data" / "README.md", temporary_root / "data" / "README.md")
    copy_file(
        TASK_SOURCE / "data" / "processed" / "cardiology_knowledge.json",
        temporary_root / "data" / "processed" / "cardiology_knowledge.json",
    )

    dev_audit_path = PROJECT_ROOT / "annotations" / "manual_audit_llm_supervised_v1.csv"
    test_audit_path = PROJECT_ROOT / "annotations" / "manual_audit_baseline_v2.csv"
    dev_audit = load_csv(dev_audit_path)
    test_audit = load_csv(test_audit_path)
    dev_source = load_jsonl(
        TASK_SOURCE / "data" / "processed" / "medqa_cardiology_llm_relevant.jsonl"
    )
    test_source = load_jsonl(
        TASK_SOURCE
        / "data"
        / "processed"
        / "medqa_cardiology_real_high_confidence.jsonl"
    )
    write_jsonl(
        temporary_root / "data" / "input" / "dev_questions.jsonl",
        public_records(dev_audit, dev_source, "dev"),
    )
    write_jsonl(
        temporary_root / "data" / "input" / "test_questions.jsonl",
        public_records(test_audit, test_source, "test"),
    )
    write_jsonl(
        temporary_root / "data" / "reference" / "dev_labels.jsonl",
        dev_labels(dev_audit),
    )

    write_jsonl(
        temporary_root / "answer" / "cardio_gold_test.jsonl",
        hidden_cardio_labels(test_audit),
    )
    write_jsonl(
        temporary_root / "answer" / "topic_gold_test.jsonl",
        hidden_topic_labels(test_audit),
    )
    for filename in STATIC_ANSWER_FILES:
        copy_file(TASK_SOURCE / "answer" / filename, temporary_root / "answer" / filename)
    copy_tree(TASK_SOURCE / "grader", temporary_root / "grader")

    reference_target = temporary_root / "reference"
    copy_file(TASK_SOURCE / "reference" / "README.md", reference_target / "README.md")
    copy_file(
        TASK_SOURCE / "data" / "raw" / "external" / "MedQA" / "README.md",
        reference_target / "MedQA_README.md",
    )
    copy_file(
        TASK_SOURCE / "data" / "raw" / "external" / "MedQA" / "LICENSE",
        reference_target / "MedQA_LICENSE",
    )
    copy_file(
        PROJECT_ROOT / "annotations" / "manual_audit_labeling_guideline_v2.md",
        reference_target / "manual_audit_labeling_guideline_v2.md",
    )
    copy_file(dev_audit_path, reference_target / "dev_annotations.csv")
    copy_file(test_audit_path, reference_target / "test_annotations.csv")
    copy_file(
        PROJECT_ROOT / "annotations" / "opm_extraction_gold_20.csv",
        reference_target / "opm_extraction_gold_20.csv",
    )

    traces_target = temporary_root / "traces"
    copy_file(TASK_SOURCE / "traces" / "README.md", traces_target / "README.md")
    write_trace_summary(traces_target)

    legacy_target = temporary_root / "legacy_project"
    copy_file(PROJECT_ROOT / "README.md", legacy_target / "README.md")
    copy_file(PROJECT_ROOT / "requirements.txt", legacy_target / "requirements.txt")
    copy_tree(PROJECT_ROOT / "src", legacy_target / "src")
    copy_tree(PROJECT_ROOT / "tests", legacy_target / "tests")
    copy_tree(PROJECT_ROOT / "demos", legacy_target / "demos")
    for filename in LEGACY_SCRIPTS:
        copy_file(
            PROJECT_ROOT / "scripts" / filename,
            legacy_target / "scripts" / filename,
        )
    copy_file(
        PROJECT_ROOT / "data" / "processed" / "cardiology_knowledge.json",
        legacy_target / "data" / "processed" / "cardiology_knowledge.json",
    )
    copy_file(
        PROJECT_ROOT / "data" / "processed" / "medqa_cardiology_sample.jsonl",
        legacy_target / "data" / "processed" / "medqa_cardiology_sample.jsonl",
    )
    copy_file(
        PROJECT_ROOT / "data" / "raw" / "medqa_sample.jsonl",
        legacy_target / "data" / "raw" / "medqa_sample.jsonl",
    )

    for executable in (
        temporary_root / "grader" / "grade.py",
        temporary_root / "tools" / "validate_submission.py",
        temporary_root / "env" / "run_validate.sh",
        temporary_root / "env" / "run_grade.sh",
    ):
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    write_manifest(temporary_root)

    if package_root.exists():
        shutil.rmtree(package_root)
    temporary_root.replace(package_root)
    if zip_path.exists():
        zip_path.unlink()
    write_deterministic_zip(package_root, zip_path)
    return package_root, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-root", type=Path, default=DEFAULT_DELIVERY_ROOT)
    args = parser.parse_args()
    package_root, zip_path = build(args.delivery_root.resolve())
    print(f"package: {package_root}")
    print(f"zip: {zip_path}")


if __name__ == "__main__":
    main()
