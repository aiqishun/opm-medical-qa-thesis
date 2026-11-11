#!/usr/bin/env python3
"""Evaluate OPM extraction accuracy against a small gold annotation CSV.

The gold CSV is expected to contain one row per sample with these fields:

sample_id,question,gold_objects,gold_processes,gold_states,gold_relations

Gold item fields may use ``|`` or ``;`` as separators. Relation annotations
can be written as readable triples, for example:

    source -> relationship -> target | source -> relationship -> target

The evaluation is a lightweight extraction audit. It compares predicted OPM
objects, processes, states, and relations from graph JSON files against gold
annotations using exact normalized matching plus a simple token-overlap match.
It is not a clinical validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = PROJECT_ROOT / "annotations" / "opm_extraction_gold_20.csv"
DEFAULT_GRAPHS_DIR = PROJECT_ROOT / "outputs" / "graphs" / "llm_relevant_batch"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "experiments" / "results" / "opm_extraction_accuracy_20.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "experiments" / "results" / "opm_extraction_accuracy_summary.md"
)

OBJECT_KEYS = ("objects", "opm_objects")
PROCESS_KEYS = ("processes", "opm_processes")
STATE_KEYS = ("states", "opm_states")
LINK_KEYS = ("links", "opm_links", "edges", "relations")
CATEGORIES = ("objects", "processes", "states", "relations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help=f"Gold annotation CSV (default: {DEFAULT_GOLD.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=DEFAULT_GRAPHS_DIR,
        help=f"Directory of graph JSON files (default: {DEFAULT_GRAPHS_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Per-sample metrics CSV (default: {DEFAULT_OUTPUT_CSV.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Markdown summary (default: {DEFAULT_SUMMARY.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=0.5,
        help="Minimum token overlap for partial semantic matching (default: 0.5).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_text(text: Any) -> str:
    raw = value_to_text(text).strip().lower()
    translation = str.maketrans({char: " " for char in string.punctuation})
    without_punctuation = raw.translate(translation)
    without_punctuation = re.sub(r"[^\w\s]", " ", without_punctuation)
    return " ".join(without_punctuation.split())


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(value_to_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(value_to_text(item) for item in value.values())
    return str(value)


def tokens(text: Any) -> set[str]:
    return set(normalize_text(text).split())


def item_matches(predicted: str, gold: str, threshold: float) -> bool:
    predicted_norm = normalize_text(predicted)
    gold_norm = normalize_text(gold)
    if not predicted_norm or not gold_norm:
        return False
    if predicted_norm == gold_norm:
        return True

    predicted_tokens = set(predicted_norm.split())
    gold_tokens = set(gold_norm.split())
    if not predicted_tokens or not gold_tokens:
        return False
    overlap = len(predicted_tokens & gold_tokens) / max(
        len(predicted_tokens), len(gold_tokens)
    )
    return overlap >= threshold


def dedupe_items(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        item_text = value_to_text(item).strip()
        item_norm = normalize_text(item_text)
        if item_norm and item_norm not in seen:
            seen.add(item_norm)
            deduped.append(item_text)
    return deduped


def parse_gold_items(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return dedupe_items(value_to_text(item) for item in parsed)

    separator = "|" if "|" in raw else ";"
    parts = [part.strip() for part in raw.replace("\n", separator).split(separator)]
    return dedupe_items(part for part in parts if part)


def read_gold_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required = {
        "sample_id",
        "question",
        "gold_objects",
        "gold_processes",
        "gold_states",
        "gold_relations",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Gold CSV is missing required columns: {sorted(missing)}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Graph JSON must contain an object: {path}")
    return data


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_present(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return []


def extract_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("label", "name", "id", "value"):
            raw = value.get(key)
            if raw:
                return str(raw)
    return value_to_text(value)


def extract_nodes(data: dict[str, Any], keys: Iterable[str]) -> list[str]:
    return dedupe_items(extract_label(item) for item in ensure_list(first_present(data, keys)))


def extract_relations(data: dict[str, Any]) -> list[str]:
    relations: list[str] = []
    for raw_link in ensure_list(first_present(data, LINK_KEYS)):
        if isinstance(raw_link, dict):
            source = raw_link.get("source") or raw_link.get("from") or raw_link.get("src")
            relationship = (
                raw_link.get("relationship")
                or raw_link.get("type")
                or raw_link.get("label")
                or raw_link.get("relation")
            )
            target = raw_link.get("target") or raw_link.get("to") or raw_link.get("dst")
            relation = " -> ".join(
                part for part in (value_to_text(source), value_to_text(relationship), value_to_text(target)) if part
            )
        else:
            relation = value_to_text(raw_link)
        relations.append(relation)
    return dedupe_items(relations)


def extract_graph_parts(path: Path) -> dict[str, list[str]]:
    data = read_json(path)
    objects = extract_nodes(data, OBJECT_KEYS)
    processes = extract_nodes(data, PROCESS_KEYS)
    states = extract_nodes(data, STATE_KEYS)

    for raw_node in ensure_list(data.get("nodes")):
        if not isinstance(raw_node, dict):
            continue
        label = extract_label(raw_node)
        node_type = normalize_text(raw_node.get("type") or raw_node.get("kind"))
        if "object" in node_type:
            objects.append(label)
        elif "process" in node_type:
            processes.append(label)
        elif "state" in node_type:
            states.append(label)

    return {
        "objects": dedupe_items(objects),
        "processes": dedupe_items(processes),
        "states": dedupe_items(states),
        "relations": extract_relations(data),
    }


def count_true_positives(
    predicted: list[str], gold: list[str], threshold: float
) -> int:
    matched_gold: set[int] = set()
    true_positives = 0

    for predicted_item in predicted:
        exact_match_index = None
        for index, gold_item in enumerate(gold):
            if index in matched_gold:
                continue
            if normalize_text(predicted_item) == normalize_text(gold_item):
                exact_match_index = index
                break
        if exact_match_index is not None:
            matched_gold.add(exact_match_index)
            true_positives += 1
            continue

        partial_match_index = None
        for index, gold_item in enumerate(gold):
            if index in matched_gold:
                continue
            if item_matches(predicted_item, gold_item, threshold):
                partial_match_index = index
                break
        if partial_match_index is not None:
            matched_gold.add(partial_match_index)
            true_positives += 1

    return true_positives


def precision_recall_f1(true_positives: int, predicted_count: int, gold_count: int) -> tuple[float, float, float]:
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def format_float(value: float) -> str:
    return f"{value:.4f}"


def evaluate_row(
    row: dict[str, str], graphs_dir: Path, threshold: float
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    sample_id = row.get("sample_id", "").strip()
    graph_path = graphs_dir / f"{sample_id}.json"
    has_graph = graph_path.exists()
    predicted = (
        extract_graph_parts(graph_path)
        if has_graph
        else {category: [] for category in CATEGORIES}
    )
    gold = {
        "objects": parse_gold_items(row.get("gold_objects", "")),
        "processes": parse_gold_items(row.get("gold_processes", "")),
        "states": parse_gold_items(row.get("gold_states", "")),
        "relations": parse_gold_items(row.get("gold_relations", "")),
    }

    csv_row: dict[str, str] = {
        "sample_id": sample_id,
        "graph_file": display_path(graph_path) if has_graph else "",
        "has_graph": str(has_graph).lower(),
    }
    counts: dict[str, dict[str, int]] = {}

    for category in CATEGORIES:
        true_positives = count_true_positives(predicted[category], gold[category], threshold)
        predicted_count = len(predicted[category])
        gold_count = len(gold[category])
        precision, recall, f1 = precision_recall_f1(
            true_positives, predicted_count, gold_count
        )

        counts[category] = {
            "tp": true_positives,
            "predicted": predicted_count,
            "gold": gold_count,
        }
        csv_row[f"{category}_tp"] = str(true_positives)
        csv_row[f"{category}_predicted"] = str(predicted_count)
        csv_row[f"{category}_gold"] = str(gold_count)
        csv_row[f"{category}_precision"] = format_float(precision)
        csv_row[f"{category}_recall"] = format_float(recall)
        csv_row[f"{category}_f1"] = format_float(f1)

    overall_tp = sum(counts[category]["tp"] for category in CATEGORIES)
    overall_predicted = sum(counts[category]["predicted"] for category in CATEGORIES)
    overall_gold = sum(counts[category]["gold"] for category in CATEGORIES)
    precision, recall, f1 = precision_recall_f1(
        overall_tp, overall_predicted, overall_gold
    )
    counts["overall"] = {
        "tp": overall_tp,
        "predicted": overall_predicted,
        "gold": overall_gold,
    }
    csv_row["overall_tp"] = str(overall_tp)
    csv_row["overall_predicted"] = str(overall_predicted)
    csv_row["overall_gold"] = str(overall_gold)
    csv_row["overall_precision"] = format_float(precision)
    csv_row["overall_recall"] = format_float(recall)
    csv_row["overall_f1"] = format_float(f1)

    return csv_row, counts


def aggregate_counts(
    per_row_counts: list[dict[str, dict[str, int]]]
) -> dict[str, dict[str, int]]:
    aggregate = {
        category: {"tp": 0, "predicted": 0, "gold": 0}
        for category in (*CATEGORIES, "overall")
    }
    for row_counts in per_row_counts:
        for category, counts in row_counts.items():
            aggregate[category]["tp"] += counts["tp"]
            aggregate[category]["predicted"] += counts["predicted"]
            aggregate[category]["gold"] += counts["gold"]
    return aggregate


def rows_with_gold(rows: list[dict[str, str]]) -> int:
    gold_fields = ("gold_objects", "gold_processes", "gold_states", "gold_relations")
    return sum(
        1 for row in rows if any(parse_gold_items(row.get(field, "")) for field in gold_fields)
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "graph_file",
        "has_graph",
        *(
            f"{category}_{metric}"
            for category in (*CATEGORIES, "overall")
            for metric in ("tp", "predicted", "gold", "precision", "recall", "f1")
        ),
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    *,
    gold_path: Path,
    graphs_dir: Path,
    output_csv: Path,
    gold_rows: list[dict[str, str]],
    csv_rows: list[dict[str, str]],
    aggregate: dict[str, dict[str, int]],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_graphs = sum(1 for row in csv_rows if row["has_graph"] != "true")
    annotated_rows = rows_with_gold(gold_rows)

    lines = [
        "# OPM Extraction Accuracy Summary",
        "",
        "> Lightweight extraction audit against manually supplied gold OPM annotations. This is not clinical validation.",
        "",
        "## Inputs and outputs",
        "",
        f"- Gold CSV: `{display_path(gold_path)}`",
        f"- Graphs directory: `{display_path(graphs_dir)}`",
        f"- Per-sample metrics CSV: `{display_path(output_csv)}`",
        f"- Partial match threshold: `{threshold:.2f}` token overlap",
        "",
        "## Coverage",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| Gold template rows | {len(gold_rows)} |",
        f"| Rows with any gold annotation | {annotated_rows} |",
        f"| Rows evaluated | {len(csv_rows)} |",
        f"| Rows missing graph JSON | {missing_graphs} |",
        "",
    ]

    if annotated_rows < len(gold_rows):
        lines.extend(
            [
                "> Note: Some gold fields are blank. Fill the template before interpreting these metrics as extraction accuracy.",
                "",
            ]
        )

    lines.extend(
        [
            "## Aggregate metrics",
            "",
            "| Category | TP | Predicted | Gold | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category in (*CATEGORIES, "overall"):
        counts = aggregate[category]
        precision, recall, f1 = precision_recall_f1(
            counts["tp"], counts["predicted"], counts["gold"]
        )
        lines.append(
            f"| `{category}` | {counts['tp']} | {counts['predicted']} | {counts['gold']} | "
            f"{precision:.4f} | {recall:.4f} | {f1:.4f} |"
        )
    lines.append("")

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    gold_path = resolve_path(args.gold)
    graphs_dir = resolve_path(args.graphs_dir)
    output_csv = resolve_path(args.output_csv)
    summary_path = resolve_path(args.summary)

    if not graphs_dir.exists():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")

    gold_rows = read_gold_rows(gold_path)
    csv_rows: list[dict[str, str]] = []
    per_row_counts: list[dict[str, dict[str, int]]] = []

    for row in gold_rows:
        csv_row, counts = evaluate_row(row, graphs_dir, args.partial_threshold)
        csv_rows.append(csv_row)
        per_row_counts.append(counts)

    aggregate = aggregate_counts(per_row_counts)
    write_csv(output_csv, csv_rows)
    write_summary(
        summary_path,
        gold_path=gold_path,
        graphs_dir=graphs_dir,
        output_csv=output_csv,
        gold_rows=gold_rows,
        csv_rows=csv_rows,
        aggregate=aggregate,
        threshold=args.partial_threshold,
    )

    print(f"Wrote per-sample metrics to {display_path(output_csv)}")
    print(f"Wrote summary to {display_path(summary_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
