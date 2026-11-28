"""Grounding rules shared by explanation sentences and graph items (C1, C2, C3).

Every grounded item carries ``grounding`` (question / passages / none),
``question_quote`` and ``passage_ids``:

* question: restates the question stem; ``question_quote`` is a verbatim span of the stem;
* passages: supported by at least one cited passage; no quote;
* none:     neither (model knowledge); no quote and no citations.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

GROUNDINGS = ("question", "passages", "none")

_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "‐": "-", "‑": "-", "–": "-", "—": "-", "−": "-",
                            " ": " "})


def normalize_text(text: str) -> str:
    """NFKC, unified quotes and dashes, case-fold, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).casefold()
    return re.sub(r"\s+", " ", text).strip()


def quote_in_stem(quote: str, stem: str) -> bool:
    """True if the normalized quote (outer punctuation stripped) occurs in the normalized stem."""
    core = normalize_text(quote).strip(" .,;:!?\"'()[]")
    return bool(core) and core in normalize_text(stem)


def grounding_violations(item: dict[str, Any]) -> list[str]:
    """Rule violations of one grounded item (sentence, element or relation)."""
    grounding, quote, ids = item.get("grounding"), item.get("question_quote"), item.get("passage_ids") or []
    problems = []
    if grounding == "question" and not (quote and quote.strip()):
        problems.append("question_without_quote")
    if grounding != "question" and quote is not None:
        problems.append("quote_on_non_question_item")
    if grounding == "passages" and not ids:
        problems.append("passages_without_citation")
    if grounding == "none" and ids:
        problems.append("none_with_citation")
    return problems
