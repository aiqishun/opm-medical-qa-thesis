# Tools

## Submission validator

`validate_submission.py` validates the required filenames, JSON/JSONL syntax,
JSON Schemas, per-split ID coverage, duplicate/unknown IDs, semantic annotation
character offsets, and task-level OPL string format.

```bash
python3 tools/validate_submission.py --submission submission
```

The validator reads only agent-visible input and contract files. It does not
read hidden answers, calculate test metrics, access the network, change input
data, or write into the submission.

## Legacy pipeline

The existing matching, reasoning, graph, and evaluation implementation is
under `legacy_project/`. Its CLI scripts may write outputs only to paths
selected by the user. The knowledge file used by the task is
`data/processed/cardiology_knowledge.json`.

## Interface boundary

- Input and output schemas are under `data/contracts/`.
- JSONL means one complete JSON object per non-empty line.
- Use `(split, id)` when joining development and test records.
- Do not read evaluator-hidden paths listed in `task.json`.
- Do not write into `data/`, `answer/`, `grader/`, `reference/`, or `traces/`.
- Do not include API keys, tokens, environment dumps, or external service logs
  in the submission.
