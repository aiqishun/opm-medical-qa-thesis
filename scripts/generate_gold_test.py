#!/usr/bin/env python3
"""Generate hidden classification/topic references from the manual audit CSV."""

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "annotations" / "manual_audit_baseline_v2.csv"
ANSWER_DIR = PROJECT_ROOT / "medqa_opm_task" / "answer"


def main() -> None:
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    with (ANSWER_DIR / "cardio_gold_test.jsonl").open(
        "w", encoding="utf-8"
    ) as cardio_file, (ANSWER_DIR / "topic_gold_test.jsonl").open(
        "w", encoding="utf-8"
    ) as topic_file:
        for row in rows:
            cardio_data = {
                "id": row["sample_id"],
                "is_cardiology": row["keep_for_cardiology_dataset"],
                "reason": row["reason"],
            }
            cardio_file.write(json.dumps(cardio_data, ensure_ascii=False) + "\n")

            topic_data = {
                "id": row["sample_id"],
                "expected_topic": row["expected_topic"],
                "error_type": row["error_type"],
            }
            topic_file.write(json.dumps(topic_data, ensure_ascii=False) + "\n")

    print(f"Generated {len(rows)} records in {ANSWER_DIR}")


if __name__ == "__main__":
    main()
