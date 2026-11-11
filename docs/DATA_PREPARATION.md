# Data preparation: first thesis version

The generated data card and checksums are in [thesis-v0.1](../data/versions/thesis-v0.1/README.md). This version is ready for development work. It is not a frozen final test set.

## Findings

The local raw file contains 11,451 distinct question/option records, with source-reported train/test counts of 10,178/1,273. No separate original development file was present. Its SHA-256 is `1b404642b2698fc8cd6618bad6c94281441909e51f13c44c27802e111b895d0f`.

The two inherited processed sources contain 546 and 935 records. All 546 records in the first source occur in the second: the union is 935, not 1,481 independent questions. Every processed record matches one raw question/options record, and its source-reported split and answer key agree with that record.

The legacy package's `dev` input contains 93 source-train and 7 source-test questions. Its `test` input contains 88 source-train and 12 source-test questions. These package names must not be used as if they were original dataset splits.

The 200 inherited annotation rows cover 188 distinct cases: 134 have consistently positive relevance judgments, 52 consistently negative judgments, and 2 have conflicting judgments. These are inherited labels, not new expert adjudications. The sources do not establish reviewer qualification or inter-rater reliability.

There are 119 consistently positive source-train cases. One has an exact-stem option variant in the raw file; the pilot excludes unresolved variants, leaving 118 eligible cases. A deterministic 20-question development pilot was selected from those 118.

The current legacy keyword-filter helper reads question, answer, answer index, explanation, and options. This creates a risk of answer-conditioned sample selection. This audit does not prove which historical code revision created every old artifact. The new prospective screen reads only the question, excludes all previous candidate/graph-annotation stems, and preserves the source-reported split. Its vocabulary is recorded in the generated manifest.

That screen found 456 new source-train candidates and 51 new source-test candidates. The training count is an inventory, not a newly labeled training set. The 51 source-test records are an unreviewed queue, not 51 confirmed cardiology questions. Nine contain textual references to visual material; one lexical near-duplicate pair needs review. Neither flag alone establishes an invalid or duplicated question.

The flagged lexical pair spans source-reported test and train partitions. Its status remains unresolved; the exact-stem checks alone therefore cannot establish independence of the final partitions.

## What to use now

| File under `data/versions/thesis-v0.1/` | Intended use |
| --- | --- |
| `inputs/pilot_questions.jsonl` | Develop and debug the agent system on 20 known training-source questions |
| `evaluator_only/pilot_gold.jsonl` | Score pilot answers outside agent inputs; do not report this as held-out performance |
| `inputs/reviewed_train_questions.jsonl` | Development/routing pool with positive and negative relevance examples |
| `review/relevance_conflicts.csv` | Two original label disagreements, with the existing annotations and blank adjudication fields |
| `review/prospective_test_screening.csv` | Review scope, visual material, near matches and eventual inclusion for all 51 candidates |
| `audit/near_duplicate_review.jsonl` | Identify the cases involved in the lexical review pair |
| `review/near_duplicate_pairs.csv` | Compare both candidate stems/options and source splits without MCQ answer keys |
| `catalog/legacy_records.jsonl` | Trace canonical IDs to source rows and original split metadata |
| `catalog/reviewed_annotations.jsonl` | Preserve each original `(benchmark_split, legacy_id)` annotation separately |
| `manifest.json`, `MANIFEST.sha256` | Reproduce the selection policy and verify file identities |

Model input rows contain only `id`, `question`, and `options`. Filtering metadata, answers, labels and prior-model judgments remain outside the model input. An evaluator-only directory is a convention, not a security boundary: restrict the runner's file/tool access and retrieval corpus accordingly.

This version uses canonical `mq_...` IDs. The inherited `run_inference.py` and task grader still operate on their original `qNNNN` package and are not a version-aware runner/evaluator for this dataset. That adapter belongs to the next implementation step.

## Review instructions

First agree and record the inclusion policy: a question's **main tested concept** must fall within the thesis's cardiovascular scope. A cardiac history, vital-sign value, risk factor, or cardiovascular distractor is not sufficient on its own. Explicitly decide how to handle pharmacology, calculations, vascular disease and secondary neurological injury before reviewing model performance. Preserve disagreements and boundary cases rather than adapting the policy to obtain better scores.

For every row of `review/prospective_test_screening.csv`, complete:

- `is_cardiology`: `yes`, `no`, or `unclear` under the written scope definition.
- `include_for_evaluation`: `yes`, `no`, or `unclear`; account for missing essential material and unresolved duplicate relationships as well as scope.
- `reviewer`, `review_date`, and `reason_and_evidence`: actual provenance of the decision.

The review queue omits MCQ answer keys. Selection must not depend on whether an experimental model answers correctly. A `visual_reference_flag` prompts inspection of the original source: determine whether an essential figure is available, unnecessary, or missing. Do not invent a figure or infer absent measurements.

For near matches, inspect differences in the stem, negation, numerical values, choices and source identity. If they represent the same underlying item, keep the duplicate cluster in one partition. The lexical thresholds only generate candidates; they do not certify that all semantic duplicates have been found.

In `review/relevance_conflicts.csv`, fill `adjudicated_label`, the reviewer/date and the supporting reason when an adjudication has actually occurred. The original labels remain preserved. No label has been selected automatically.

## Next data milestone

After the review is complete, build a new version containing finalized evaluation IDs, exclusion reasons, paired reference answers, immutable selection/corpus hashes and evaluator isolation. Freeze prompts and model choices using development data before running that evaluation. Report its actual sample size and uncertainty; the 20-question pilot is an engineering sample.

Expert OPM reference graphs, evidence-span labels, and independent semantic review are separate annotation tasks. Consistent relevance labels and correct MCQ keys do not supply gold OPM mechanisms.

## Reproduction and validation

```bash
python scripts/build_thesis_dataset.py --raw-source /path/to/medqa_full.jsonl --output /tmp/thesis-data-rebuild --pilot-size 20 --seed 42
python -m pytest tests/test_build_thesis_dataset.py -q
```

Use the pinned raw-file checksum. The raw file remains in the original local project/backup and is not silently downloaded or replaced. The builder refuses to overwrite a version and preserves original source files. With the same input and code bytes, the generated data and manifests are deterministic.

Validation on 11 November 2026: all 301 tests passed, including 14 data-pipeline checks. Two independent builds on the full local source produced 18 byte-identical files; all 17 listed file hashes matched. Model-input allowlists, unique IDs, paired input/reference coverage, and development/prospective-test separation were checked. The two conflict adjudications and all 51 screening decisions remain blank and pending.

The upstream release is not independently re-certified by a local hash match, and public benchmark questions may have appeared in model pretraining. Record these limits in the English thesis.
