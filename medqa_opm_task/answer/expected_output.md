# Expected output and acceptance notes

The participant submission must contain the eight files listed in `prompt.md`.
The automatic grader treats the 100 IDs in `cardio_gold_test.jsonl` as the
fixed test denominator. Missing, duplicate, malformed, or unknown-ID
predictions cannot improve the test score.

## Hidden references

| File | Records | Purpose |
| --- | ---: | --- |
| `cardio_gold_test.jsonl` | 100 | Binary cardiology labels and audit reasons |
| `topic_gold_test.jsonl` | 100 | Expected topic and recorded error type |
| `opm_gold.jsonl` | 20 | Task-level OPL reference statements |

The binary test reference contains 56 positive and 44 negative labels. This
class count is evaluator metadata and must not be copied into agent-visible
files.

## Quantitative acceptance

The 20-point hidden-test classification component uses the exact F1 bands in
`rubrics.md`; it does not use a subjective tolerance. Precision, Recall and F1
use the fixed gold set. When no positive prediction exists, Precision and F1
are zero.

OPM graphs are not required to match a single node-for-node graph. The grader
checks file structure and the required task-level OPL sentence pattern;
medical synonymy, relationship semantics and evidence quality are reviewed
under `judge_prompt.md` and `rubrics.md`.

No reference chart is required for this task. Expert decisions are preserved
in the hidden JSONL references and the source annotation CSV files under the
hidden reference path. The 20 OPM records map one-to-one by `id` to
`reference/opm_extraction_gold_20.csv`.
