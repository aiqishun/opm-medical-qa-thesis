#!/usr/bin/env python3
"""Run cardiology relevance inference over the labelled test IDs.

The gold file is used only to obtain the IDs that belong to this evaluation
set. Labels and gold reasons are never included in the model prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTION_SOURCE = (
    TASK_DIR / "data" / "processed" / "medqa_cardiology_real_high_confidence.jsonl"
)
DEFAULT_ID_SOURCE = TASK_DIR / "answer" / "cardio_gold_test.jsonl"
DEFAULT_OUTPUT_DIR = TASK_DIR / "answer"
QUESTION_ID_PATTERN = re.compile(r"q(\d+)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            records.append(record)
    return records


def question_for_id(
    question_id: str,
    question_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Map qNNNN to its zero-based row in the high-confidence source file."""

    match = QUESTION_ID_PATTERN.fullmatch(question_id)
    if match is None:
        raise ValueError(f"无法识别题目 ID: {question_id!r}")

    source_index = int(match.group(1))
    if source_index >= len(question_records):
        raise ValueError(
            f"题目 {question_id} 映射到索引 {source_index}，"
            f"但题库只有 {len(question_records)} 条记录"
        )

    record = question_records[source_index]
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"题目 {question_id} 缺少 question 字段")
    return record


def record_for_prompt(question_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Select context fields without leaking the answer or gold label."""

    prompt_record: dict[str, Any] = {
        "id": question_id,
        "question": record["question"],
    }
    for key in ("options", "metamap_phrases", "matched_terms", "filter_confidence"):
        if key in record:
            prompt_record[key] = record[key]
    return prompt_record


def build_prompt(question_id: str, record: dict[str, Any]) -> str:
    prompt_record = record_for_prompt(question_id, record)
    record_json = json.dumps(prompt_record, ensure_ascii=False, indent=2)
    return f"""\
请判断下面 MedQA 题目的主要考点是否属于心血管系统疾病相关。

判断时请区分主要考点与既往史、危险因素、生命体征、偶然提及以及干扰选项；
不能仅因为出现了心血管关键词就判定为相关。

题目记录：
{record_json}

只输出一个 JSON 对象，不要包含 Markdown 标记或其他文字：
{{"id": "{question_id}", "is_cardiology": true, "reason": "简要判断理由"}}
"""


def parse_response(content: str, expected_id: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型没有返回有效 JSON: {content[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    if parsed.get("id") != expected_id:
        raise ValueError(
            f"模型返回的 id 是 {parsed.get('id')!r}，预期为 {expected_id!r}"
        )
    if not isinstance(parsed.get("is_cardiology"), bool):
        raise ValueError("is_cardiology 必须是 JSON 布尔值 true/false")
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        raise ValueError("reason 必须是非空字符串")

    return {
        "id": expected_id,
        "is_cardiology": parsed["is_cardiology"],
        "reason": parsed["reason"].strip(),
    }


def run_inference_batch(
    run_id: int,
    *,
    question_source: Path = DEFAULT_QUESTION_SOURCE,
    id_source: Path = DEFAULT_ID_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> Path | None:
    question_records = load_jsonl(question_source)
    id_records = load_jsonl(id_source)
    if limit is not None:
        id_records = id_records[:limit]

    selected_records: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for id_record in id_records:
        question_id = id_record.get("id")
        if not isinstance(question_id, str):
            raise ValueError("ID 文件中存在缺少字符串 id 的记录")
        if question_id in seen_ids:
            raise ValueError(f"ID 文件中存在重复题目: {question_id}")
        seen_ids.add(question_id)
        selected_records.append(
            (question_id, question_for_id(question_id, question_records))
        )

    if dry_run:
        print(f"已映射 {len(selected_records)} 道题；以下是第一道题的实际提示词：\n")
        if selected_records:
            question_id, record = selected_records[0]
            print(build_prompt(question_id, record))
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY 环境变量")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境未安装 openai；请先运行：python3 -m pip install openai"
        ) from exc
    base_url = os.environ.get("OPENAI_BASE_URL")
    selected_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"filter_results_run{run_id}.jsonl"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    print(f"开始第 {run_id} 次推理，共 {len(selected_records)} 道题，模型：{selected_model}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for position, (question_id, record) in enumerate(selected_records, start=1):
                print(f"[{position}/{len(selected_records)}] 正在处理 {question_id}")
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": build_prompt(question_id, record)}],
                    temperature=0.2,
                )
                content = response.choices[0].message.content
                if not isinstance(content, str):
                    raise ValueError(f"题目 {question_id} 的模型响应没有文本内容")
                parsed = parse_response(content, question_id)
                output_file.write(json.dumps(parsed, ensure_ascii=False) + "\n")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_path)
    print(f"第 {run_id} 次运行完成：{output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MedQA 心血管相关性分类")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--question-source", type=Path, default=DEFAULT_QUESTION_SOURCE)
    parser.add_argument("--id-source", type=Path, default=DEFAULT_ID_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=None, help="覆盖 OPENAI_MODEL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 ID 映射并显示第一条提示词，不调用模型或写结果",
    )
    args = parser.parse_args()
    run_inference_batch(
        args.run_id,
        question_source=args.question_source,
        id_source=args.id_source,
        output_dir=args.output_dir,
        model=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
