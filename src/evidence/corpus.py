"""Deterministic, paragraph-aware chunking of textbook text into passages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Passage:
    """One retrievable unit of the evidence corpus."""

    id: str        # "<book>:<6-digit index>", stable for a given corpus version
    book: str      # textbook file stem, e.g. "InternalMed_Harrison"
    index: int     # position of the passage within its book
    text: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def _windows(words: list[str], max_words: int, stride: int) -> Iterator[list[str]]:
    """Split an over-long paragraph into overlapping word windows."""
    start = 0
    while True:
        yield words[start:start + max_words]
        if start + max_words >= len(words):
            return
        start += stride


def chunk_text(text: str, *, max_words: int = 200, overlap: int = 30) -> list[str]:
    """Pack whole paragraphs into passages of at most ``max_words`` words.

    Paragraphs are separated by line breaks. Consecutive paragraphs are packed
    together until the next one would exceed ``max_words``. A paragraph that is
    itself longer than ``max_words`` is split into windows that overlap by
    ``overlap`` words. Whitespace inside a passage is normalized to single spaces.
    """
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("require max_words >= 1 and 0 <= overlap < max_words")
    stride = max_words - overlap
    passages: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if len(words) > max_words:
            if current:
                passages.append(" ".join(current))
                current = []
            passages.extend(" ".join(window) for window in _windows(words, max_words, stride))
            continue
        if current and len(current) + len(words) > max_words:
            passages.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        passages.append(" ".join(current))
    return passages


def chunk_book(path: Path, *, max_words: int = 200, overlap: int = 30) -> list[Passage]:
    book = path.stem
    text = path.read_text(encoding="utf-8", errors="strict")
    return [Passage(id=f"{book}:{i:06d}", book=book, index=i, text=chunk)
            for i, chunk in enumerate(chunk_text(text, max_words=max_words, overlap=overlap))]


def load_passages(path: Path) -> list[Passage]:
    passages = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                passages.append(Passage(**json.loads(line)))
    return passages
