# Thesis dataset — thesis-v0.2

Status: **evaluation set frozen on 2026-11-28**. Parent version: `thesis-v0.1`.

| Item | Count |
| --- | ---: |
| Candidates reviewed | 51 |
| Included for evaluation | 28 |
| Excluded: essential missing exhibit | 2 |
| Excluded: near duplicate across source splits | 1 |
| Excluded: out of scope | 19 |
| Excluded: source quality | 1 |
| Relevance conflicts adjudicated | 2 |
| Near-duplicate pairs resolved | 1 |

## Files

- `inputs/test_questions.jsonl`: frozen evaluation questions. Each record has only `id`, `question`, `options`.
- `evaluator_only/test_answers.jsonl`: reference answers for scoring only. Never pass to an agent or index for retrieval.
- `inputs/pilot_questions.jsonl`, `evaluator_only/pilot_gold.jsonl`: unchanged 20-question development pilot. Not held-out.
- `inputs/reviewed_train_questions.jsonl`, `evaluator_only/reviewed_train_gold.jsonl`: unchanged development pool.
- `review/`: scope rules, labeling protocol and every review decision with its reason.
- `quarantine/never_evaluate_ids.json`: records that must not enter any evaluation set (near-duplicate pair).

## Rules of use

1. Prompts, retrieval settings and agent logic are tuned on the pilot only.
2. Freeze the system configuration (git tag) before the first run on `inputs/test_questions.jsonl`.
3. Report every test item, including abstentions and invalid outputs.
4. Do not change this version. Any later change to the evaluation set creates a new version.
