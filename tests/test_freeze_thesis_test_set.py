"""Checks for freezing the thesis evaluation set from review decisions."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import freeze_thesis_test_set as freeze

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "data/versions/thesis-v0.1"
DECISIONS = ROOT / "data/review/thesis-v0.2"
FROZEN = ROOT / "data/versions/thesis-v0.2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FreezeRealDecisions(unittest.TestCase):
    """Run the freeze on the committed decisions into a temporary directory."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "thesis-v0.2"
        cls.manifest = freeze.freeze(PARENT, DECISIONS, cls.out, frozen_on="2026-11-28")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_inputs_carry_only_question_fields(self) -> None:
        for row in read_jsonl(self.out / "inputs/test_questions.jsonl"):
            self.assertEqual(set(row), {"id", "question", "options"})

    def test_answers_align_with_inputs(self) -> None:
        inputs = [row["id"] for row in read_jsonl(self.out / "inputs/test_questions.jsonl")]
        answers = [row["id"] for row in read_jsonl(self.out / "evaluator_only/test_answers.jsonl")]
        self.assertEqual(inputs, answers)
        self.assertEqual(inputs, self.manifest["evaluation_ids"])

    def test_only_included_decisions_are_frozen(self) -> None:
        with (DECISIONS / "prospective_test_decisions.csv").open(encoding="utf-8", newline="") as handle:
            included = {row["id"] for row in csv.DictReader(handle) if row["include_for_evaluation"] == "yes"}
        frozen = {row["id"] for row in read_jsonl(self.out / "inputs/test_questions.jsonl")}
        self.assertEqual(frozen, included)

    def test_duplicate_pair_never_evaluated(self) -> None:
        never = set(json.loads((self.out / "quarantine/never_evaluate_ids.json").read_text()))
        frozen = {row["id"] for row in read_jsonl(self.out / "inputs/test_questions.jsonl")}
        self.assertTrue(never)
        self.assertFalse(never & frozen)

    def test_development_files_unchanged(self) -> None:
        for relative in freeze.CARRIED:
            self.assertEqual((self.out / relative).read_bytes(), (PARENT / relative).read_bytes())

    def test_checksums_cover_every_file(self) -> None:
        listed = {}
        for line in (self.out / "MANIFEST.sha256").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            listed[relative] = digest
            self.assertEqual(freeze.sha256(self.out / relative), digest)
        files = {p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(files - {"MANIFEST.sha256"}, set(listed))

    def test_freeze_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            again = Path(other) / "thesis-v0.2"
            freeze.freeze(PARENT, DECISIONS, again, frozen_on="2026-11-28")
            self.assertEqual((again / "MANIFEST.sha256").read_bytes(), (self.out / "MANIFEST.sha256").read_bytes())

    def test_committed_version_matches_rebuild(self) -> None:
        if not FROZEN.exists():
            self.skipTest("thesis-v0.2 not committed yet")
        self.assertEqual((FROZEN / "MANIFEST.sha256").read_bytes(), (self.out / "MANIFEST.sha256").read_bytes())


class FreezeRejectsBadDecisions(unittest.TestCase):
    def _run_with(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decisions = Path(tmp) / "decisions"
            shutil.copytree(DECISIONS, decisions)
            path = decisions / "prospective_test_decisions.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            mutate(rows)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            freeze.freeze(PARENT, decisions, Path(tmp) / "out", frozen_on="2026-11-28")

    def test_unclear_label_blocks_freeze(self) -> None:
        def mutate(rows):
            rows[0]["include_for_evaluation"] = "unclear"
        with self.assertRaises(freeze.FreezeError):
            self._run_with(mutate)

    def test_out_of_scope_item_cannot_be_included(self) -> None:
        def mutate(rows):
            row = next(r for r in rows if r["is_cardiology"] == "no")
            row["include_for_evaluation"], row["exclusion_reason"] = "yes", ""
        with self.assertRaises(freeze.FreezeError):
            self._run_with(mutate)

    def test_duplicate_item_cannot_be_included(self) -> None:
        def mutate(rows):
            row = next(r for r in rows if r["exclusion_reason"] == "near_duplicate_across_source_splits")
            row["include_for_evaluation"], row["exclusion_reason"] = "yes", ""
        with self.assertRaises(freeze.FreezeError):
            self._run_with(mutate)

    def test_missing_decision_blocks_freeze(self) -> None:
        with self.assertRaises(freeze.FreezeError):
            self._run_with(lambda rows: rows.pop())

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "thesis-v0.2"
            out.mkdir()
            with self.assertRaises(freeze.FreezeError):
                freeze.freeze(PARENT, DECISIONS, out, frozen_on="2026-11-28")


if __name__ == "__main__":
    unittest.main()
