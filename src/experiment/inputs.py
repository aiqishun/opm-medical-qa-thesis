"""Answer-free inputs for experiment runs.

This module may only read model inputs: question files under the dataset's
``inputs/`` directory, the fixed retrieval sets (passage identifiers only) and
the local evidence corpus. Reference answers are never loaded here; scoring is
done by a separate script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data/versions/thesis-v0.2"
INPUTS_DIR = DATASET_DIR / "inputs"
RETRIEVAL_DIR = ROOT / "experiments/retrieval/medqa-textbooks-en-v1__bm25_k10"
CORPUS_PASSAGES = ROOT / "data/corpus/medqa-textbooks-en-v1/passages.jsonl"

SPLIT_FILES = {"pilot": "pilot_questions.jsonl", "test": "test_questions.jsonl"}
ALLOWED_QUESTION_FIELDS = {"id", "question", "options"}
OPTION_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    options: dict[str, str]


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    rank: int
    text: str


def question_file(split: str, inputs_dir: Path | None = None) -> Path:
    inputs_dir = INPUTS_DIR if inputs_dir is None else inputs_dir
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    path = (inputs_dir / SPLIT_FILES[split]).resolve()
    if path.parent != inputs_dir.resolve():
        raise ValueError(f"question file must be directly under {inputs_dir}")
    return path


def load_questions(path: Path) -> list[Question]:
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            extra = set(record) - ALLOWED_QUESTION_FIELDS
            if extra:  # guards against an answer field leaking into the model input
                raise ValueError(f"{path}:{line_no}: unexpected fields {sorted(extra)}")
            options = record["options"]
            if sorted(options) != list(OPTION_LABELS):
                raise ValueError(f"{path}:{line_no}: options must be exactly A-D")
            questions.append(Question(id=record["id"], question=record["question"], options=dict(options)))
    return questions


def load_retrieval(path: Path) -> dict[str, list[tuple[str, int]]]:
    """question id -> [(passage_id, rank), ...] in rank order."""
    result: dict[str, list[tuple[str, int]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                ranked = sorted(record["passages"], key=lambda p: p["rank"])
                result[record["id"]] = [(p["passage_id"], p["rank"]) for p in ranked]
    return result


def load_passage_texts(path: Path, wanted: set[str]) -> dict[str, str]:
    """Stream the corpus and keep only the requested passages."""
    texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in wanted:
                texts[record["id"]] = record["text"]
                if len(texts) == len(wanted):
                    break
    missing = wanted - set(texts)
    if missing:
        raise KeyError(f"{len(missing)} passage ids not found in corpus, e.g. {sorted(missing)[:3]}")
    return texts


def build_evidence(retrieval: dict[str, list[tuple[str, int]]], questions: list[Question],
                   corpus_path: Path) -> dict[str, list[EvidencePassage]]:
    missing = [q.id for q in questions if q.id not in retrieval]
    if missing:
        raise KeyError(f"no retrieval set for {len(missing)} questions, e.g. {missing[:3]}")
    wanted = {pid for q in questions for pid, _ in retrieval[q.id]}
    texts = load_passage_texts(corpus_path, wanted)
    return {q.id: [EvidencePassage(pid, rank, texts[pid]) for pid, rank in retrieval[q.id]] for q in questions}
