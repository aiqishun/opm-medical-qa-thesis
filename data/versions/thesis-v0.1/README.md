# Thesis dataset preparation — thesis-v0.1

Status: **development-ready; evaluation review required; no final test set frozen**.

## What was prepared

| Item | Count |
| --- | ---: |
| Local raw records | 11451 |
| Inherited processed occurrences | 1481 |
| Unique inherited candidates | 935 |
| Unique manually reviewed cases | 188 |
| Cross-package dev/test duplicate pairs | 12 |
| Conflicting cardiology labels, quarantined | 2 |
| Consistent positive / negative labels | 134 / 52 |
| Reviewed original-train cases with consistent labels | 167 |
| Eligible positive original-train cases | 118 |
| Fixed-seed development pilot | 20 |
| Fresh question-only original-test candidates, pending review | 51 |
| Fresh original-train candidates, counted but not yet reviewed | 456 |
| Test candidate visual-reference flags | 9 |
| Lexical near-duplicate pairs requiring review | 1 |

## Inputs and evaluation references

- `inputs/pilot_questions.jsonl`: selected from consistent positive, source-reported train cases; suitable for engineering and prompt development, not a held-out result.
- `inputs/reviewed_train_questions.jsonl`: reviewed original-train records, including positive and negative relevance examples; use the intended task-specific subset.
- `inputs/prospective_test_questions.jsonl`: candidate pool only. Do not use for prompt tuning or call it a finalized cardiology benchmark.
- `evaluator_only/`: MCQ answer references and inherited relevance labels. The agent runner must load only the relevant `inputs/` file. Directory naming is not an access-control boundary; keep evaluator material out of agent tools and retrieval.

Each model-input record contains only `id`, `question`, and `options`. It does not carry answer keys, relevance labels, filter scores, matched terms, prior model decisions, or annotation reasons. Evaluation reference IDs match the corresponding input file exactly.

## Provenance and known limitations

The local raw source has source-reported splits {"test": 1273, "train": 10178}. No separate upstream development file was present. Raw release provenance is not independently certified; its exact file checksum is pinned in `manifest.json`.

The inherited packaged development questions originated from {"test": 7, "train": 93}, and its packaged test questions from {"test": 12, "train": 88}. Those legacy package names must not be mistaken for original MedQA partitions. All previous candidate question stems and previous graph annotations are excluded from the prospective evaluation pool, including option variants with an already exposed stem.

The current inherited keyword selector examines answer/explanation fields. The new screening function examines the question alone and records its vocabulary in the manifest. Keyword membership is not a human judgment of the main tested concept. The initial source-release split and duplicate checks do not prove semantic independence or absence of model pretraining exposure.

Canonical IDs hash the question plus sorted option keys after NFC/whitespace normalization. Case, punctuation, units, signs, numbers and option labels are preserved. Original row positions and `(benchmark_split, legacy_id)` annotations remain in `catalog/`. Different options can produce different case IDs; stem grouping identifies those variants. Lexical near matches are flagged for review and never automatically merged.

## Review before freezing an evaluation version

1. Review all rows in `review/prospective_test_screening.csv` against the prespecified criterion: the primary tested concept is cardiovascular, rather than incidental history, vital signs, or a distractor. Record yes/no/unclear, inclusion decision, reviewer/date, and reason. Reviewers should not use model success to choose questions.
2. Resolve visual-reference, option-variant, and near-duplicate flags; record whether essential source material is available. Compare both stems/options in `review/near_duplicate_pairs.csv` without exposing answer keys. Do not equate a visual-reference flag with a proven defective question.
3. Keep the two conflicting cases isolated until documented adjudication in `review/relevance_conflicts.csv`. No existing label has been changed.
4. Freeze a subsequent version with final IDs, selection criteria, reference answers, source/corpus hashes, and separate evaluator access. Do not overwrite this version or the inherited data.
5. Finish prompt/model selection on development data, then evaluate the frozen test once under the fixed protocol. Document any later protocol change.

The review spreadsheet intentionally omits MCQ answer keys. OPM mechanism annotations and independent expert judgments have not been created by this build. Existing relevance labels are not gold OPM graphs.

## Reproduce

From the repository root, with the original local raw JSONL available:

```bash
python scripts/build_thesis_dataset.py --raw-source /path/to/medqa_full.jsonl --output /tmp/thesis-data-rebuild --pilot-size 20 --seed 42
```

Use the raw checksum in `manifest.json`; the full raw file is not newly bundled here. The builder checks MCQ keys/text, legacy row mappings and labels, refuses an existing output directory, and writes only a new version. For a fixed input/code version it produces deterministic artifacts. `MANIFEST.sha256` covers every generated file except itself.

MedQA source notices are preserved at `../../../medqa_opm_task/reference/MedQA_LICENSE` and `MedQA_README.md`. This is research data preparation, not clinical validation.
