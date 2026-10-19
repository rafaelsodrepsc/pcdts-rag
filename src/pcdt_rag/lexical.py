"""BM25 over the chunk texts.

Dense retrieval matches meaning, but questions about scales and drug names depend on exact
tokens ("PASI", "CPK", "dapagliflozina") that an embedding can blur. BM25 ranks by those
tokens. Accents are removed so that "maxima" and "máxima" match; there is no stemming, and
numbers are split at the punctuation ("3.000" becomes "3" and "000").
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

K1 = 1.2
B = 0.75
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return TOKEN.findall(text)


class BM25:
    def __init__(self, texts: list[str], k1: float = K1, b: float = B) -> None:
        docs = [Counter(tokenize(t)) for t in texts]
        lengths = np.array([sum(d.values()) for d in docs], dtype=float)
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                postings[term].append((i, tf))
        self.size = len(docs)
        # Each term keeps the documents that contain it and its finished BM25 weight in each.
        self.weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, entries in postings.items():
            ids = np.array([i for i, _ in entries])
            tf = np.array([tf for _, tf in entries], dtype=float)
            idf = math.log(1 + (self.size - len(ids) + 0.5) / (len(ids) + 0.5))
            self.weights[term] = (ids, idf * tf * (k1 + 1) / (tf + norm[ids]))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.size)
        for term in set(tokenize(query)):
            if term in self.weights:
                ids, weights = self.weights[term]
                out[ids] += weights
        return out
