#!/usr/bin/env python3
"""Retrieve a fixed BM25 passage set E_q for every question, once, for all conditions.

Queries use the question stem and the four option texts only. The output stores
passage identifiers and scores, not textbook text, so it can be committed to a
public repository; agents resolve the text from the local corpus.

The script also checks that no question stem appears verbatim in the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence.bm25 import STOPWORDS, BM25Index, tokenize  # noqa: E402
from evidence.corpus import load_passages  # noqa: E402

DATASET = ROOT / "data/versions/thesis-v0.2"
CORPUS = ROOT / "data/corpus/medqa-textbooks-en-v1"
SPLITS = {"pilot": "inputs/pilot_questions.jsonl", "test": "inputs/test_questions.jsonl"}
ALLOWED_INPUT_FIELDS = {"id", "question", "options"}
STEM_PROBE_CHARS = 120


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", text).lower()).split())


def read_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        extra = set(row) - ALLOWED_INPUT_FIELDS
        if extra:
            raise SystemExit(f"{path.name}: input record carries forbidden fields {sorted(extra)}")
    return rows


def build_query(row: dict) -> str:
    options = " ".join(row["options"][key] for key in sorted(row["options"]))
    return f"{row['question']} {options}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    passages_path = args.corpus / "passages.jsonl"
    if sha256_file(passages_path) != corpus_manifest["passages_sha256"]:
        raise SystemExit("passages.jsonl does not match the corpus manifest; rebuild the corpus")
    passages = load_passages(passages_path)
    print(f"Loaded {len(passages)} passages; indexing ...", flush=True)
    index = BM25Index([tokenize(p.text) for p in passages], k1=args.k1, b=args.b)

    run_name = f"{corpus_manifest['corpus_version']}__bm25_k{args.k}"
    output = args.output or ROOT / "experiments/retrieval" / run_name
    output.mkdir(parents=True, exist_ok=True)

    corpus_norm = None
    leakage = {}
    question_files = {}
    for split, relative in SPLITS.items():
        questions = read_questions(args.dataset / relative)
        question_files[split] = {"path": relative, "sha256": sha256_file(args.dataset / relative), "questions": len(questions)}
        records = []
        for row in questions:
            hits = index.top_k(tokenize(build_query(row)), args.k)
            records.append({"id": row["id"], "passages": [
                {"rank": rank, "passage_id": passages[doc].id, "score": round(score, 6)}
                for rank, (doc, score) in enumerate(hits, 1)]})
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
        if corpus_norm is None:
            corpus_norm = normalize(" ".join(p.text for p in passages))
        found = [row["id"] for row in questions if normalize(row["question"])[:STEM_PROBE_CHARS] in corpus_norm]
        leakage[split] = {"stems_checked": len(questions), "stems_found_in_corpus": len(found), "ids": found}
        print(f"  {split}: {len(records)} questions retrieved; stems found in corpus: {len(found)}", flush=True)

    manifest = {
        "run": run_name,
        "dataset_version": args.dataset.name,
        "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
        "question_files": question_files,
        "corpus_version": corpus_manifest["corpus_version"],
        "corpus_passages_sha256": corpus_manifest["passages_sha256"],
        "retrieval": {"method": "Okapi BM25 (src/evidence/bm25.py)", "k": args.k, "k1": args.k1, "b": args.b,
                      "query": "question stem + option texts A-D (no answer, no labels)",
                      "tokenizer": "lower-case [a-z0-9] tokens, fixed stop-word list, no stemming",
                      "stopwords": sorted(STOPWORDS), "tie_break": "lower corpus position"},
        "leakage_check": {"method": f"first {STEM_PROBE_CHARS} normalized characters of each stem searched in the normalized corpus",
                          **leakage},
        "script_sha256": sha256_file(Path(__file__)),
        "bm25_sha256": sha256_file(SRC_DIR / "evidence/bm25.py"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
