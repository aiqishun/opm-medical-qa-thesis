#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_PATH="${1:-$TASK_ROOT/submission}"

python3 "$TASK_ROOT/tools/validate_submission.py" --submission "$SUBMISSION_PATH"
