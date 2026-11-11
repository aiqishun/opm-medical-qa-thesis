# Reproducible environment

The reference environment is Python 3.11. The automatic grader and public
validator require `jsonschema`; participant methods may declare additional
dependencies in their own `submission/run_all.sh` workflow.

Local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r env/requirements.txt
```

Equivalent Conda setup, using the same Python version and requirements file:

```bash
conda env create -f env/environment.yml
conda activate medqa-opm-eval
```

Public validation:

```bash
bash env/run_validate.sh submission
```

Evaluator-only grading:

```bash
bash env/run_grade.sh submission
```

Docker build:

```bash
docker build -f env/Dockerfile -t medqa-opm-eval .
docker run --rm -it -v "$PWD/submission:/task/submission" medqa-opm-eval
```

`.env.example` contains no credential fields. No mock external service is
required by the validator or grader, and no reset script is supplied because
the reference evaluation is read-only.
