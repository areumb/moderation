"""Embedding function for the policy vector store.

Default: sentence-transformers all-MiniLM-L6-v2 (local; downloads once on
first init, then works fully offline).

Fallback: a deterministic hashed bag-of-words embedding used when
sentence-transformers (or its cached model) is unavailable — e.g. in a
network-restricted CI job. It is much weaker semantically but keeps the whole
pipeline runnable and testable offline. The active backend is logged and
recorded in the collection metadata so it is never silently ambiguous.

Both classes implement chromadb's EmbeddingFunction interface.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

_DIM = 384  # match MiniLM dimensionality so the two backends are swappable


class HashedBowEmbedder(EmbeddingFunction[Documents]):
    """Deterministic hashed bag-of-words embedding (offline fallback)."""

    backend_name = "hashed-bow-fallback"

    def __init__(self) -> None:  # explicit: required by newer chromadb EF interface
        pass

    @staticmethod
    def name() -> str:
        return "hashed_bow_fallback"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashedBowEmbedder":
        return HashedBowEmbedder()

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        tokens = re.findall(r"[a-z0-9\[\]']+", text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % _DIM
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(t) for t in input]


class SentenceTransformerEmbedder(EmbeddingFunction[Documents]):
    backend_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @staticmethod
    def name() -> str:
        return "policy_minilm"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "SentenceTransformerEmbedder":
        return SentenceTransformerEmbedder()

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), normalize_embeddings=True).tolist()


def get_embedder() -> EmbeddingFunction:
    """Return the best available embedding function."""
    try:
        emb = SentenceTransformerEmbedder()
        logger.info("Using sentence-transformers embeddings.")
        return emb
    except Exception as exc:  # ImportError, download failure, ...
        logger.warning(
            "sentence-transformers unavailable (%s); falling back to deterministic "
            "hashed bag-of-words embeddings. Retrieval quality will be reduced.",
            exc,
        )
        return HashedBowEmbedder()
