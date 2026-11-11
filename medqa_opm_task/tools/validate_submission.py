#!/usr/bin/env python3
"""Validate submission structure without reading hidden evaluator files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema


TASK_DIR = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = TASK_DIR / "data" / "contracts"
INPUT_DIR = TASK_DIR / "data" / "input"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
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
            errors.append(f"{path.name}:{line_number}: JSON 解析失败: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path.name}:{line_number}: 每行必须是 JSON 对象")
            continue
        rows.append(row)
    return rows, errors


def validate_rows(
    path: Path,
    schema_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = load_jsonl(path)
    schema = load_json(CONTRACTS_DIR / schema_name)
    validator = jsonschema.Draft7Validator(schema)
    for row_number, row in enumerate(rows, start=1):
        for error in sorted(validator.iter_errors(row), key=lambda item: list(item.path)):
            errors.append(f"{path.name}:{row_number}: {error.message}")
    return rows, errors


def question_index() -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for split in ("dev", "test"):
        rows, errors = load_jsonl(INPUT_DIR / f"{split}_questions.jsonl")
        if errors:
            raise ValueError("; ".join(errors))
        for row in rows:
            index[(split, row["id"])] = row["question"]
    return index


def validate_pair_coverage(
    rows: list[dict[str, Any]],
    filename: str,
    expected: set[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    keys = [(row.get("split"), row.get("id")) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    submitted = set(keys)
    missing = expected - submitted
    unknown = submitted - expected
    if duplicates:
        errors.append(f"{filename}: 存在 {len(duplicates)} 个重复 (split, id)")
    if missing:
        errors.append(f"{filename}: 缺少 {len(missing)} 个 (split, id)")
    if unknown:
        errors.append(f"{filename}: 包含 {len(unknown)} 个未知 (split, id)")
    return errors


def validate_single_json(path: Path, schema_name: str) -> list[str]:
    try:
        value = load_json(path)
        schema = load_json(CONTRACTS_DIR / schema_name)
        jsonschema.Draft7Validator(schema).validate(value)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return [f"{path.name}: {exc}"]
    return []


def validate_submission(submission: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    questions = question_index()

    for split in ("dev", "test"):
        filename = f"filter_results_{split}.jsonl"
        path = submission / filename
        if not path.is_file():
            errors.append(f"缺少 {filename}")
            continue
        rows, row_errors = validate_rows(path, "filter_results.schema.json")
        errors.extend(row_errors)
        ids = [row.get("id") for row in rows]
        duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicate_ids:
            errors.append(f"{filename}: 重复 ID: {duplicate_ids}")
        expected = {item_id for item_split, item_id in questions if item_split == split}
        submitted = {item for item in ids if isinstance(item, str)}
        missing = sorted(expected - submitted)
        unknown = sorted(submitted - expected)
        if missing:
            errors.append(f"{filename}: 缺少 {len(missing)} 个 ID")
        if unknown:
            errors.append(f"{filename}: 包含 {len(unknown)} 个未知 ID")
        counts[filename] = len(rows)

    semantic_path = submission / "semantic_annotations.jsonl"
    if not semantic_path.is_file():
        errors.append("缺少 semantic_annotations.jsonl")
    else:
        rows, row_errors = validate_rows(
            semantic_path, "semantic_annotations.schema.json"
        )
        errors.extend(row_errors)
        errors.extend(
            validate_pair_coverage(rows, semantic_path.name, set(questions))
        )
        counts[semantic_path.name] = len(rows)
        for row_number, row in enumerate(rows, start=1):
            key = (row.get("split"), row.get("id"))
            question = questions.get(key)
            if question is None:
                errors.append(
                    f"{semantic_path.name}:{row_number}: 未知的 (split, id): {key}"
                )
                continue
            for annotation in row.get("annotations", []):
                start = annotation.get("start")
                end = annotation.get("end")
                text = annotation.get("text")
                if isinstance(start, int) and isinstance(end, int):
                    if end <= start or question[start:end] != text:
                        errors.append(
                            f"{semantic_path.name}:{row_number}: 原文位置与 text 不一致"
                        )

    opm_path = submission / "opm_audit_results.jsonl"
    if not opm_path.is_file():
        errors.append("缺少 opm_audit_results.jsonl")
    else:
        rows, row_errors = validate_rows(
            opm_path, "opm_audit_results.schema.json"
        )
        errors.extend(row_errors)
        errors.extend(validate_pair_coverage(rows, opm_path.name, set(questions)))
        counts[opm_path.name] = len(rows)
        for row_number, row in enumerate(rows, start=1):
            key = (row.get("split"), row.get("id"))
            if key not in questions:
                errors.append(f"{opm_path.name}:{row_number}: 未知的 (split, id): {key}")

    required_json = {
        "data_audit.json": "data_audit.schema.json",
        "metrics.json": "metrics.schema.json",
    }
    for filename, schema_name in required_json.items():
        path = submission / filename
        if not path.is_file():
            errors.append(f"缺少 {filename}")
        else:
            errors.extend(validate_single_json(path, schema_name))

    report_path = submission / "research_report.md"
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        errors.append("缺少或空的 research_report.md")

    run_all_path = submission / "run_all.sh"
    if not run_all_path.is_file():
        errors.append("缺少 run_all.sh")
    elif not run_all_path.stat().st_mode & 0o111:
        warnings.append("run_all.sh 未设置可执行权限；可用 chmod +x submission/run_all.sh")

    return {
        "valid": not errors,
        "submission": str(submission.resolve()),
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=TASK_DIR / "submission")
    args = parser.parse_args()
    result = validate_submission(args.submission)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
