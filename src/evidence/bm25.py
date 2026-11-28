"""A small, dependency-free Okapi BM25 index.

Implemented here rather than imported so that tokenization, parameters and tie
breaking are fully specified and reproducible for the thesis.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

# Short, fixed English stop-word list (recorded in the retrieval manifest).
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves following most likely which patient
""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stop words removed. No stemming."""
    return [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over pre-tokenized documents with an inverted index."""

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("cannot index an empty corpus")
        self.k1, self.b = k1, b
        self.n_docs = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        self.avg_len = sum(self.doc_len) / self.n_docs
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                postings[term].append((doc_id, freq))
        self.postings = dict(postings)
        # BM25 idf with the +1 inside the log so that it is never negative.
        self.idf = {term: math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
                    for term, plist in self.postings.items()}

    def scores(self, query: Iterable[str]) -> dict[int, float]:
        """BM25 score for every document that contains at least one query term."""
        result: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(query).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_id, freq in plist:
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len)
                result[doc_id] += qtf * idf * freq * (self.k1 + 1) / norm
        return result

    def top_k(self, query: Iterable[str], k: int) -> list[tuple[int, float]]:
        """Top ``k`` (doc_id, score) pairs; ties broken by lower doc_id."""
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
