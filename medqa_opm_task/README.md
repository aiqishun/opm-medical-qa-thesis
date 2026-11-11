# MedQA cardiology routing and OPM audit task

This package evaluates cardiology-topic routing, contextual medical-phrase
handling, and OPM/OPL explanation auditing over two manually reviewed sets.
It is a research benchmark package and is not a clinical system.

## Package structure

| Path | Purpose | Visibility |
| --- | --- | --- |
| `task.json` | Task metadata and workspace policy | Platform configuration |
| `prompt.md` | Complete participant instruction | Agent-visible |
| `data/input/` | Sanitized development and test questions | Agent-visible |
| `data/reference/` | Public development labels | Agent-visible |
| `data/contracts/` | Output JSON Schemas | Agent-visible |
| `legacy_project/` | Existing pipeline source | Agent-visible |
| `tools/` | Public, answer-free validation CLI | Agent-visible |
| `reference/` | Dataset license, citation, labeling guide | Mixed; see `task.json` |
| `answer/` | Hidden test references and expected-output notes | Evaluator-hidden |
| `grader/` | Deterministic evaluator | Evaluator-hidden |
| `traces/` | Measured reference run outputs and metrics | Evaluator-hidden |
| `rubrics.md` | 100-point scoring rubric and score caps | Evaluator-hidden |
| `judge_prompt.md` | Qualitative review instructions | Evaluator-hidden |

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r env/requirements.txt
```

The reference environment is also defined in `env/Dockerfile`.

## Participant workflow

1. Read `prompt.md`, `data/README.md`, the contracts, and the public reference
   files.
2. Implement the method and a deterministic `submission/run_all.sh`.
3. Generate all eight required files under `submission/`.
4. Run the public validator:

```bash
python3 tools/validate_submission.py --submission submission
```

The public validator checks structure and traceability only. It does not read
test gold labels or report test metrics.

## Evaluator workflow

```bash
python3 grader/grade.py --submission submission
```

The grader emits JSON containing deterministic file checks, confusion-matrix
counts, Precision, Recall, F1, and the rubric's 20-point F1 component. The
remaining rubric dimensions require the review procedure in `judge_prompt.md`.

## Reproducibility and boundaries

- JSONL files contain one JSON object per non-empty line.
- Development/test joins use `(split, id)`, not `id` alone.
- Test input files do not contain answer text, answer indices, test labels, or
  previous LLM decisions.
- API credentials are neither required by the evaluator nor stored in this
  package.
- The required OPL sentence is a task-level syntax check; passing it does not
  certify full ISO OPM compliance.
- Measured trace summaries distinguish computed metrics from unavailable
  telemetry. Missing token counts or latency values are marked unavailable,
  not estimated.

## License and citation

The bundled MedQA source notice and MIT license are in `reference/`. No
separate license for the original task-specific repository material was found
when this delivery package was assembled; `task.json` records that fact rather
than assigning one.
