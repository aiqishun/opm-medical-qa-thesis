#!/usr/bin/env python3
"""Build the versioned evidence corpus from the MedQA English textbooks.

All textbooks in the input directory are used; no book or chapter is selected
by topic. The passages file contains copyrighted textbook text and is kept out
of git; the manifest (checksums, counts, parameters) is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence.corpus import chunk_book  # noqa: E402

CORPUS_VERSION = "medqa-textbooks-en-v1"
DEFAULT_TEXTBOOKS = ROOT / "medqa_opm_task/data/raw/external/MedQA/data_clean/textbooks/en"
DEFAULT_OUTPUT = ROOT / "data/corpus" / CORPUS_VERSION


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(textbooks: Path, output: Path, *, max_words: int, overlap: int) -> dict:
    books = sorted(p for p in textbooks.glob("*.txt") if not p.name.startswith("._"))
    if not books:
        raise SystemExit(f"No textbook .txt files found in {textbooks}")
    output.mkdir(parents=True, exist_ok=True)
    passages_path = output / "passages.jsonl"
    per_book = []
    total = 0
    with passages_path.open("w", encoding="utf-8") as handle:
        for book in books:
            passages = chunk_book(book, max_words=max_words, overlap=overlap)
            for passage in passages:
                handle.write(passage.to_json() + "\n")
            words = sum(len(p.text.split()) for p in passages)
            per_book.append({"book": book.stem, "file_sha256": sha256_file(book),
                             "passages": len(passages), "words": words})
            total += len(passages)
            print(f"  {book.stem:<28} {len(passages):>7} passages", flush=True)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "source": "MedQA English textbooks (data_clean/textbooks/en); all books, no topic selection",
        "chunking": {"method": "pack whole paragraphs (line breaks) up to max_words; split longer paragraphs into overlapping windows",
                     "max_words": max_words, "overlap_words": overlap},
        "passage_id_format": "<book>:<6-digit index>",
        "books": per_book,
        "total_passages": total,
        "passages_file": "passages.jsonl",
        "passages_sha256": sha256_file(passages_path),
        "build_script_sha256": sha256_file(Path(__file__)),
        "notes": ["passages.jsonl contains textbook text and is not committed; rebuild it locally from the same source files.",
                  "Textbook file checksums above identify the exact source files."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textbooks", type=Path, default=DEFAULT_TEXTBOOKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()
    print(f"Building {CORPUS_VERSION} from {args.textbooks}")
    manifest = build(args.textbooks, args.output, max_words=args.max_words, overlap=args.overlap)
    print(f"Total: {manifest['total_passages']} passages from {len(manifest['books'])} books")
    print(f"passages_sha256: {manifest['passages_sha256']}")


if __name__ == "__main__":
    main()
