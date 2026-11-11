#!/usr/bin/env python3
"""Project-root entry point for the MedQA inference runner."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "medqa_opm_task" / "run_inference.py"
    runpy.run_path(str(script), run_name="__main__")
