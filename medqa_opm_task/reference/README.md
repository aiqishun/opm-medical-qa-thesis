# References

This directory records sources already present in the repository. It does not
add new medical claims.

- `MedQA_README.md`: bundled upstream dataset description and citation.
- `MedQA_LICENSE`: bundled upstream MIT license text.
- `manual_audit_labeling_guideline_v2.md`: controlled vocabulary and migration
  rules used by the existing manual audits.
- `dev_annotations.csv`: source expert annotations for public development
  labels.
- `test_annotations.csv`: source expert annotations for hidden test labels;
  evaluator-hidden according to `task.json`.
- `opm_extraction_gold_20.csv`: source manual annotations for the 20 hidden
  OPM/OPL reference records; evaluator-hidden according to `task.json`.

The project describes its graph format as an OPM-style research artifact and
not as a standard-compliant OPM serialization. Accordingly, this benchmark
evaluates the explicit OPL string contract in `data/contracts/` and does not
claim complete ISO OPM conformance.
