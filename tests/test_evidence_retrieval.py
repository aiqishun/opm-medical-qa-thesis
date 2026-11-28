"""Checks for evidence-corpus chunking, BM25 ranking and fixed retrieval."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evidence.bm25 import BM25Index, tokenize
from evidence.corpus import chunk_book, chunk_text

ROOT = Path(__file__).resolve().parents[1]

CARDIO = ("Aortic stenosis causes a crescendo-decrescendo systolic murmur at the right upper sternal border.\n"
          "Calcification of the aortic valve narrows the outflow tract and leads to left ventricular hypertrophy.\n")
OTHER = ("Tinea corporis is a superficial dermatophyte infection of the skin.\n"
         "A potassium hydroxide preparation shows branching hyphae.\n")
PLANTED_STEM = "A 70-year-old man has exertional syncope and a harsh systolic murmur radiating to the carotids."


class ChunkingTests(unittest.TestCase):
    def test_paragraphs_are_packed_up_to_limit(self) -> None:
        text = "\n".join(["one two three"] * 5)
        self.assertEqual(chunk_text(text, max_words=6, overlap=0), ["one two three one two three"] * 2 + ["one two three"])

    def test_long_paragraph_is_split_with_overlap(self) -> None:
        words = " ".join(str(i) for i in range(10))
        chunks = chunk_text(words, max_words=4, overlap=1)
        self.assertEqual(chunks[0].split(), ["0", "1", "2", "3"])
        self.assertEqual(chunks[1].split()[0], "3")
        self.assertTrue(all(len(c.split()) <= 4 for c in chunks))
        self.assertEqual(chunks[-1].split()[-1], "9")

    def test_no_words_lost_without_overlap(self) -> None:
        text = CARDIO + "\n\n" + OTHER
        self.assertEqual(" ".join(chunk_text(text, max_words=12, overlap=0)).split(), text.split())

    def test_invalid_parameters_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("a b", max_words=3, overlap=3)

    def test_passage_ids_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "Pathology_Test.txt"
            book.write_text(CARDIO + OTHER, encoding="utf-8")
            first = chunk_book(book, max_words=20, overlap=5)
            second = chunk_book(book, max_words=20, overlap=5)
            self.assertEqual(first, second)
            self.assertEqual(first[0].id, "Pathology_Test:000000")


class BM25Tests(unittest.TestCase):
    def test_relevant_document_ranks_first(self) -> None:
        docs = [tokenize(OTHER), tokenize(CARDIO), tokenize("Heart failure with reduced ejection fraction.")]
        index = BM25Index(docs)
        top = index.top_k(tokenize("calcified aortic valve systolic murmur"), 3)
        self.assertEqual(top[0][0], 1)

    def test_ties_break_by_position(self) -> None:
        index = BM25Index([["murmur"], ["murmur"], ["rash"]])
        self.assertEqual([doc for doc, _ in index.top_k(["murmur"], 2)], [0, 1])

    def test_stopwords_removed(self) -> None:
        self.assertEqual(tokenize("Which of the following is the most likely diagnosis?"), ["diagnosis"])


class RetrievalScriptTests(unittest.TestCase):
    def _fixture(self, root: Path, *, extra_field: bool = False) -> tuple[Path, Path, Path]:
        books = root / "books"
        books.mkdir()
        (books / "Cardio_Book.txt").write_text(CARDIO + PLANTED_STEM + "\n", encoding="utf-8")
        (books / "Derm_Book.txt").write_text(OTHER, encoding="utf-8")
        (books / "._Cardio_Book.txt").write_bytes(b"\x00\x05mac-metadata")
        corpus = root / "corpus"
        subprocess.run([sys.executable, str(ROOT / "scripts/build_evidence_corpus.py"), "--textbooks", str(books),
                        "--output", str(corpus), "--max-words", "25", "--overlap", "5"],
                       check=True, capture_output=True)
        dataset = root / "thesis-v0.2"
        (dataset / "inputs").mkdir(parents=True)
        (dataset / "manifest.json").write_text("{}", encoding="utf-8")
        pilot = {"id": "mq_pilot", "question": "Which valve lesion causes a systolic murmur at the right upper sternal border?",
                 "options": {"A": "Aortic stenosis", "B": "Tinea", "C": "Asthma", "D": "Gout"}}
        test = {"id": "mq_test", "question": PLANTED_STEM,
                "options": {"A": "Aortic stenosis", "B": "Mitral stenosis", "C": "Tinea", "D": "Gout"}}
        if extra_field:
            test["answer_idx"] = "A"
        (dataset / "inputs/pilot_questions.jsonl").write_text(json.dumps(pilot) + "\n", encoding="utf-8")
        (dataset / "inputs/test_questions.jsonl").write_text(json.dumps(test) + "\n", encoding="utf-8")
        return corpus, dataset, root / "retrieval"

    def _retrieve(self, corpus: Path, dataset: Path, out: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(ROOT / "scripts/retrieve_evidence.py"), "--dataset", str(dataset),
                               "--corpus", str(corpus), "--k", "2", "--output", str(out)],
                              capture_output=True, text=True)

    def test_retrieval_outputs_ids_scores_and_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, dataset, out = self._fixture(Path(tmp))
            manifest = json.loads((corpus / "manifest.json").read_text())
            self.assertEqual([b["book"] for b in manifest["books"]], ["Cardio_Book", "Derm_Book"])
            result = self._retrieve(corpus, dataset, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            pilot = json.loads((out / "pilot.jsonl").read_text())
            self.assertEqual(pilot["passages"][0]["passage_id"].split(":")[0], "Cardio_Book")
            self.assertEqual(set(pilot["passages"][0]), {"rank", "passage_id", "score"})
            run = json.loads((out / "manifest.json").read_text())
            self.assertEqual(run["leakage_check"]["test"]["ids"], ["mq_test"])
            self.assertEqual(run["leakage_check"]["pilot"]["stems_found_in_corpus"], 0)

    def test_retrieval_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, dataset, out = self._fixture(Path(tmp))
            self._retrieve(corpus, dataset, out)
            first = (out / "test.jsonl").read_bytes()
            self._retrieve(corpus, dataset, out)
            self.assertEqual((out / "test.jsonl").read_bytes(), first)

    def test_input_with_answer_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, dataset, out = self._fixture(Path(tmp), extra_field=True)
            result = self._retrieve(corpus, dataset, out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("forbidden fields", result.stdout + result.stderr)

    def test_modified_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, dataset, out = self._fixture(Path(tmp))
            with (corpus / "passages.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"id": "x:000000", "book": "x", "index": 0, "text": "tampered"}\n')
            result = self._retrieve(corpus, dataset, out)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
