# Measured model traces

`filter_results_run1.jsonl` and `filter_results_run2.jsonl` are two completed
classification passes over the same 100-record hidden test set. Both were
created with the repository inference runner using model identifier
`gpt-5.6-sol` and temperature `0.2`.

`summary.json` is computed directly from the two JSONL files and the hidden
binary gold file. It contains row counts, schema-valid counts, confusion
matrices, Accuracy, Precision, Recall and F1. It is evaluator-hidden because it
contains hidden-test measurements.

The runner did not persist full HTTP request/response bodies, token usage,
latency, seed, provider version or endpoint revision. These values are recorded
as unavailable in `summary.json`; they are not estimated.

These traces support repeatability checks and error analysis. They are not
training data and must not be exposed to the participant agent.
