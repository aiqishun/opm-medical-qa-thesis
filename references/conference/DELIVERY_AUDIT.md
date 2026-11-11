# Delivery audit

This audit is not included in `medqa_opm_task.zip`.

## Checklist from the delivery specification

| Required item | Delivered path | Objective check |
| --- | --- | --- |
| Task metadata | `task.json` | JSON parses; required metadata categories are present |
| Agent prompt | `prompt.md` | Inputs, prohibited paths, required processing, outputs and reporting boundaries are explicit |
| Expected output | `answer/` | 100 cardiology labels, 100 topic labels, 20 OPL references, and acceptance notes |
| Rubric | `rubrics.md` | Six dimensions total 100 points; thresholds and score caps are stated |
| Automatic grader | `grader/grade.py` | Fixed-denominator Precision/Recall/F1, schema/ID error isolation, and rubric F1 points |
| Environment | `env/` | Python 3.11 requirements, Conda definition, Dockerfile, environment template, validation and grading launch scripts |
| Tools | `tools/` | Public submission validator and interface/permission documentation |
| Data | `data/` | 100 dev + 100 test public records, public dev labels, contracts, data card and knowledge file |
| References | `reference/` | MedQA citation/license, labeling guideline, dev/test source annotation CSV files, and the 20-record OPM annotation source |
| Traces | `traces/` | Two 100-record runs, computed metrics and unavailable-telemetry declarations |
| Reproduction guide | `README.md` | Setup, participant flow, evaluation flow and safety/data boundaries |

## Mechanical verification

- Dev input: 100 rows, 100 unique IDs.
- Test input: 100 rows, 100 unique IDs.
- Dev input IDs exactly equal public dev-label IDs.
- Test input IDs exactly equal hidden test-label IDs.
- Public input forbidden fields found: none.
- Missing paths declared by `task.json`: none.
- Legacy pipeline tests: 287 passed.
- Public validator and complete grader exercised with a 200-record temporary fixture: passed.
- `MANIFEST.sha256`: all listed files passed verification.
- ZIP integrity test: passed.
- Repeated builds from unchanged sources produce the same ZIP SHA-256.
- Cache, `.DS_Store`, nested `.git` and `.pyc` entries in ZIP: none.

## Measured traces

| Run | Valid rows | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 49 | 6 | 7 | 38 | 0.890909 | 0.875000 | 0.882883 | 0.870000 |
| 2 | 100 | 50 | 8 | 6 | 36 | 0.862069 | 0.892857 | 0.877193 | 0.860000 |

Cross-run agreement is 97/100. These values are calculated from the saved
JSONL files and hidden gold labels; no causal explanation is asserted.

## Owner-supplied metadata still not established

- Task difficulty: recorded as `unrated`; no source difficulty rating exists.
- Task-specific material license: recorded as not specified; only the bundled
  MedQA MIT license is established.

These two values were not inferred because doing so would require an owner or
organizer decision.

## Artifact

- File: `delivery/medqa_opm_task.zip`
- SHA-256: `5317649f2b25ca700d64a6b6edb1528f5b6746473134602cd4704f186d5e5997`
- Size at verification: approximately 339 KiB.
