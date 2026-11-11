#!/usr/bin/env python3
"""Deterministic grader for the MedQA cardiology/OPM task."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import jsonschema


TASK_DIR = Path(__file__).resolve().parents[1]
ANSWER_DIR = TASK_DIR / "answer"
CONTRACTS_DIR = TASK_DIR / "data" / "contracts"
TOOLS_DIR = TASK_DIR / "tools"
DEFAULT_SUBMISSION = TASK_DIR / "submission"
TRUTHY = {"true", "yes", "1", "y", "t"}

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from validate_submission import validate_submission  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return rows, [f"无法读取 {path}: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: JSON 解析失败: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: 每行必须是 JSON 对象")
            continue
        rows.append((line_number, row))
    return rows, errors


def normalize_gold_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUTHY


def load_gold(path: Path) -> dict[str, bool]:
    rows, errors = load_jsonl(path)
    if errors:
        raise ValueError("; ".join(errors))
    gold: dict[str, bool] = {}
    for line_number, row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str):
            raise ValueError(f"{path}: line {line_number}: gold id 必须是字符串")
        if item_id in gold:
            raise ValueError(f"{path}: 重复 gold id: {item_id}")
        gold[item_id] = normalize_gold_label(row.get("is_cardiology"))
    return gold


def f1_points(f1: float) -> float:
    if f1 >= 0.90:
        return 20.0
    if f1 >= 0.85:
        return 18.5
    if f1 >= 0.80:
        return 17.0
    if f1 >= 0.75:
        return 15.5
    if f1 >= 0.70:
        return 14.0
    if f1 >= 0.60:
        return 11.5
    if f1 >= 0.50:
        return 9.0
    if f1 >= 0.40:
        return 4.0
    return 0.0


def classification_metrics(
    gold: Mapping[str, bool],
    predictions_path: Path,
) -> dict[str, Any]:
    schema = load_json(CONTRACTS_DIR / "filter_results.schema.json")
    validator = jsonschema.Draft7Validator(schema)
    rows, parse_errors = load_jsonl(predictions_path)

    predictions: dict[str, bool] = {}
    duplicate_ids: set[str] = set()
    unknown_ids: set[str] = set()
    schema_errors: list[str] = []

    for line_number, row in rows:
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            schema_errors.extend(
                f"line {line_number}: {error.message}" for error in errors
            )
            continue

        item_id = row["id"]
        if item_id not in gold:
            unknown_ids.add(item_id)
            continue
        if item_id in predictions or item_id in duplicate_ids:
            duplicate_ids.add(item_id)
            predictions.pop(item_id, None)
            continue
        predictions[item_id] = row["is_cardiology"]

    missing_ids = sorted(set(gold) - set(predictions))
    tp = sum(predictions.get(item_id) is True and label for item_id, label in gold.items())
    fp = sum(predictions.get(item_id) is True and not label for item_id, label in gold.items())
    fn = sum(predictions.get(item_id) is not True and label for item_id, label in gold.items())
    tn = sum(predictions.get(item_id) is False and not label for item_id, label in gold.items())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / len(gold) if gold else 0.0

    return {
        "gold_records": len(gold),
        "submitted_nonempty_rows": len(rows),
        "valid_known_unique_predictions": len(predictions),
        "parse_error_count": len(parse_errors),
        "schema_error_count": len(schema_errors),
        "duplicate_id_count": len(duplicate_ids),
        "unknown_id_count": len(unknown_ids),
        "missing_prediction_count": len(missing_ids),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "f1_rubric_points": f1_points(f1),
        "errors": {
            "parse": parse_errors,
            "schema": schema_errors,
            "duplicate_ids": sorted(duplicate_ids),
            "unknown_ids": sorted(unknown_ids),
            "missing_ids": missing_ids,
        },
    }


def grade_submission(submission: Path) -> dict[str, Any]:
    public_validation = validate_submission(submission)
    dev_gold = load_gold(TASK_DIR / "data" / "reference" / "dev_labels.jsonl")
    test_gold = load_gold(ANSWER_DIR / "cardio_gold_test.jsonl")
    dev_metrics = classification_metrics(
        dev_gold, submission / "filter_results_dev.jsonl"
    )
    test_metrics = classification_metrics(
        test_gold, submission / "filter_results_test.jsonl"
    )

    return {
        "grader_version": "2.0",
        "mode": "submission",
        "submission": str(submission.resolve()),
        "public_validation": public_validation,
        "dev_classification": dev_metrics,
        "test_classification": test_metrics,
        "automatic_score": {
            "cardiology_f1_points": test_metrics["f1_rubric_points"],
            "cardiology_f1_maximum": 20.0,
            "total_task_maximum": 100.0,
        },
        "qualitative_review_required": True,
        "qualitative_review_files": ["rubrics.md", "judge_prompt.md"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submission", type=Path, default=None)
    group.add_argument(
        "--predictions",
        type=Path,
        help="只对一个测试集预测 JSONL 计算隐藏测试指标",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.predictions is not None:
        result: dict[str, Any] = {
            "grader_version": "2.0",
            "mode": "predictions_only",
            "predictions": str(args.predictions.resolve()),
            "test_classification": classification_metrics(
                load_gold(ANSWER_DIR / "cardio_gold_test.jsonl"),
                args.predictions,
            ),
        }
    else:
        result = grade_submission(args.submission or DEFAULT_SUBMISSION)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
