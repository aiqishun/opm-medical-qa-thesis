"""Checks for dataset identity, provenance, answer separation and review gates."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_thesis_dataset as dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(root: Path) -> tuple[Path, list[dict]]:
    questions = [
        "Development case one asks about heart failure.",
        "Development case two asks about aortic stenosis.",
        "Development case three asks about skin.",
        "Previously reviewed test case asks about arrhythmia.",
        "New evaluation vignette asks about coronary artery disease.",
        "Which mineral is measured in this unrelated question?",
        "Fresh train vignette asks about pericarditis.",
        "Conflicting classification case asks about heart failure.",
        "The figure shows an aortic stenosis measurement.",
    ]
    test_indexes = {3, 4, 5, 8}
    rows = [{"question": question, "options": {"A": "choice one", "B": "choice two"},
             "answer_idx": "A", "answer": "choice one",
             "split": "test" if i in test_indexes else "train"}
            for i, question in enumerate(questions)]
    # These fields must not make an unrelated stem a screening candidate.
    rows[5]["options"]["B"] = "heart failure"
    rows[5]["explanation"] = "myocardial infarction"
    rows[5]["matched_terms"] = ["aortic stenosis"]
    raw_path = root / "local_raw.jsonl"
    write_jsonl(raw_path, rows)
    sources = {"dev": [0, 1, 3, 7], "test": [0, 2, 7]}
    for split, indexes in sources.items():
        write_jsonl(root / dataset.SOURCES[split], [rows[i] for i in indexes])
        public = []
        annotations = []
        gold = []
        for position, raw_index in enumerate(indexes):
            row = rows[raw_index]
            legacy_id = f"q{position:04d}"
            keep = raw_index not in {2} and not (raw_index == 7 and split == "test")
            public.append({"id": legacy_id, "split": split, "source_record_index": position,
                           "question": row["question"], "options": row["options"]})
            annotations.append({"sample_id": legacy_id, "keep_for_cardiology_dataset": "yes" if keep else "no",
                                "reason": "Fixture annotation, not a clinical judgment.",
                                "expected_topic": "fixture", "cardiology_relevance": "yes" if keep else "no"})
            gold.append({"id": legacy_id, "is_cardiology": keep})
        write_jsonl(root / dataset.TASK / "data/input" / f"{split}_questions.jsonl", public)
        gold_relative = ("data/reference/dev_labels.jsonl" if split == "dev" else "answer/cardio_gold_test.jsonl")
        write_jsonl(root / dataset.TASK / gold_relative, gold)
        path = root / dataset.AUDITS[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(annotations[0]))
            writer.writeheader()
            writer.writerows(annotations)
    for name in ["manual_sanity_check_20.csv", "opm_extraction_gold_20.csv"]:
        (root / "annotations" / name).write_text("sample_id,question\n")
    (root / "scripts").mkdir()
    (root / "scripts/prepare_medqa.py").write_text('def record_text(record):\n    return record.get("answer", "")\n')
    license_path = root / dataset.TASK / "reference/MedQA_LICENSE"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_text("Fixture license notice\n")
    return raw_path, rows


class IdentityAndInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {"question": "A case with 5 mg of X?", "options": {"A": "one", "B": "two"},
                    "answer_idx": "A", "answer": "one", "split": "train"}

    def test_identity_ignores_option_dictionary_order_and_whitespace(self) -> None:
        other = copy.deepcopy(self.row)
        other["question"] = " A case\nwith 5 mg of X? "
        other["options"] = {"B": "two", "A": "one"}
        self.assertEqual(dataset.case_id(self.row), dataset.case_id(other))

    def test_identity_preserves_numbers_case_negation_and_option_labels(self) -> None:
        for question in ["A case with 50 mg of X?", "A case with 5 mg of x?", "A case without 5 mg of X?"]:
            other = {**self.row, "question": question}
            self.assertNotEqual(dataset.case_id(self.row), dataset.case_id(other))
        other = {**self.row, "options": {"B": "one", "A": "two"}}
        self.assertNotEqual(dataset.case_id(self.row), dataset.case_id(other))
        self.assertEqual(dataset.stem_id(self.row["question"]), dataset.stem_id(other["question"]))

    def test_screening_never_reads_answers_options_or_filter_metadata(self) -> None:
        row = {**self.row, "answer": "heart failure", "explanation": "aortic stenosis",
               "options": {"A": "pericarditis", "B": "arrhythmia"}, "matched_terms": ["angina"]}
        self.assertEqual(dataset.question_terms(row), [])
        row["question"] = "A question on heart failure."
        self.assertIn("heart failure", dataset.question_terms(row))

    def test_agent_input_is_strict_allowlist(self) -> None:
        row = {**self.row, "reason": "gold explanation", "is_cardiology": True,
               "llm_is_cardiology_relevant": True, "filter_confidence": "high_confidence"}
        self.assertEqual(set(dataset.agent_input(row)), {"id", "question", "options"})

    def test_invalid_mcq_reference_is_rejected(self) -> None:
        for changes in [{"answer_idx": "Z"}, {"answer": "wrong text"}, {"split": "unknown"}]:
            with self.assertRaises(dataset.DatasetError):
                dataset.validate_mcq({**self.row, **changes}, "test")

    def test_pilot_is_independent_of_input_order(self) -> None:
        rows = [{**self.row, "question": f"Development question {i}"} for i in range(8)]
        self.assertEqual(dataset.choose_pilot(rows, 3, 42), dataset.choose_pilot(list(reversed(rows)), 3, 42))
        with self.assertRaises(dataset.DatasetError):
            dataset.choose_pilot(rows, 9, 42)


class BuildTests(unittest.TestCase):
    def test_split_provenance_conflict_quarantine_and_clean_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, rows = fixture(root)
            source_bytes = raw.read_bytes()
            output = root / "built"
            result = dataset.build(root, raw, output, pilot_size=2)
            self.assertFalse(result["test_frozen"])
            self.assertEqual(result["counts"]["legacy_cross_split_duplicate_pairs"], 2)
            self.assertEqual(result["counts"]["relevance_conflicts_quarantined"], 1)
            self.assertEqual(result["counts"]["prospective_test_candidates"], 2)
            self.assertEqual(result["counts"]["legacy_packaged_original_splits"]["dev"], {"test": 1, "train": 3})
            # Identical q0001 IDs in two lists refer to different questions.
            annotations = dataset.read_jsonl(output / "catalog/reviewed_annotations.jsonl")
            ones = [row for row in annotations if row["legacy_id"] == "q0001"]
            self.assertEqual(len({row["id"] for row in ones}), 2)
            pilot_gold = dataset.read_jsonl(output / "evaluator_only/pilot_gold.jsonl")
            self.assertEqual({row["original_split"] for row in pilot_gold}, {"train"})
            self.assertEqual({row["is_cardiology"] for row in pilot_gold}, {True})
            self.assertNotIn(dataset.case_id(rows[7]), {row["id"] for row in pilot_gold})
            conflict = dataset.read_jsonl(output / "quarantine/relevance_conflicts.jsonl")[0]
            self.assertEqual({a["is_cardiology"] for a in conflict["annotations"]}, {True, False})
            self.assertEqual(conflict["status"], "unresolved_no_label_selected")
            self.assertEqual(source_bytes, raw.read_bytes())

    def test_build_is_deterministic_and_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _ = fixture(root)
            left, right = root / "left", root / "right"
            dataset.build(root, raw, left, pilot_size=2)
            dataset.build(root, raw, right, pilot_size=2)
            first = {p.relative_to(left).as_posix(): p.read_bytes() for p in left.rglob("*") if p.is_file()}
            second = {p.relative_to(right).as_posix(): p.read_bytes() for p in right.rglob("*") if p.is_file()}
            self.assertEqual(first, second)
            with self.assertRaises(dataset.DatasetError):
                dataset.build(root, raw, left, pilot_size=2)
            self.assertEqual(first["MANIFEST.sha256"], (left / "MANIFEST.sha256").read_bytes())

    def test_wrong_legacy_row_mapping_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _ = fixture(root)
            path = root / dataset.TASK / "data/input/dev_questions.jsonl"
            public = dataset.read_jsonl(path)
            public[0]["question"] = "A different question."
            write_jsonl(path, public)
            with self.assertRaisesRegex(dataset.DatasetError, "mapping mismatch"):
                dataset.build(root, raw, root / "output", pilot_size=2)
            self.assertFalse((root / "output").exists())

    def test_test_stem_option_variant_of_known_case_is_not_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, rows = fixture(root)
            rows[4]["question"] = rows[3]["question"]
            rows[4]["options"] = {"A": "different option", "B": "choice two"}
            rows[4]["answer"] = "different option"
            write_jsonl(raw, rows)
            result = dataset.build(root, raw, root / "output", pilot_size=2)
            self.assertEqual(result["counts"]["prospective_test_candidates"], 1)

    def test_cross_original_split_stem_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, rows = fixture(root)
            rows[4]["question"] = rows[6]["question"]
            rows[4]["options"] = {"A": "new option", "B": "choice two"}
            rows[4]["answer"] = "new option"
            write_jsonl(raw, rows)
            result = dataset.build(root, raw, root / "output", pilot_size=2)
            self.assertEqual(result["counts"]["raw_cross_split_stem_groups"], 1)
            self.assertEqual(result["counts"]["prospective_test_candidates"], 1)

    def test_all_model_input_and_reference_ids_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _ = fixture(root)
            output = root / "output"
            dataset.build(root, raw, output, pilot_size=2)
            for input_name, gold_name in [("pilot_questions", "pilot_gold"),
                                         ("reviewed_train_questions", "reviewed_train_gold"),
                                         ("prospective_test_questions", "prospective_test_answers")]:
                inputs = dataset.read_jsonl(output / "inputs" / (input_name + ".jsonl"))
                gold = dataset.read_jsonl(output / "evaluator_only" / (gold_name + ".jsonl"))
                self.assertEqual({r["id"] for r in inputs}, {r["id"] for r in gold})
                self.assertTrue(all(set(row) == {"id", "question", "options"} for row in inputs))

    def test_lexical_near_duplicate_is_reviewed_not_merged(self) -> None:
        question = " ".join(f"word{i}" for i in range(100))
        rows = {"left": {"question": question}, "right": {"question": question + " not"}}
        result = dataset.near_pairs(rows, {"left"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "manual_review_required_not_merged")
        self.assertEqual(len(rows), 2)

    def test_near_pair_review_contains_both_inputs_and_original_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, rows = fixture(root)
            rows[6]["question"] = "heart failure " + " ".join(f"word{i}" for i in range(100))
            rows[4]["question"] = rows[6]["question"] + " not"
            write_jsonl(raw, rows)
            output = root / "output"
            result = dataset.build(root, raw, output, pilot_size=2)
            self.assertEqual(result["counts"]["near_duplicate_review_pairs"], 1)
            with (output / "review/near_duplicate_pairs.csv").open(newline="") as handle:
                pair = next(csv.DictReader(handle))
            self.assertEqual({pair["left_original_split"], pair["right_original_split"]}, {"train", "test"})
            self.assertEqual(pair["left_question"], rows[4]["question"])
            self.assertEqual(pair["right_question"], rows[6]["question"])
            self.assertFalse(pair["same_underlying_item"])
            self.assertNotIn("answer_idx", pair)


if __name__ == "__main__":
    unittest.main()
