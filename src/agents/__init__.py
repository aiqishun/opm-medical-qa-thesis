"""Experimental conditions. Each condition is one module; register it here.

Planned: C2 (single agent + evidence-linked OPM graph), C3 (three agents with
verification/repair). C4 is the stored round-0 output of C3, not a separate run.
"""

from __future__ import annotations

import importlib

from agents.base import Condition, ModelConfig

CONDITIONS = {
    "C1": "agents.c1:C1Agent",
}


def get_condition(name: str) -> type[Condition]:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; implemented: {sorted(CONDITIONS)}")
    module_name, class_name = CONDITIONS[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)


__all__ = ["CONDITIONS", "Condition", "ModelConfig", "get_condition"]
