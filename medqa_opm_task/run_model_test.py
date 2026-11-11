#!/usr/bin/env python3
"""Run two independent inference passes with the corrected question mapping."""

from run_inference import run_inference_batch


if __name__ == "__main__":
    for run_id in range(1, 3):
        run_inference_batch(run_id)
