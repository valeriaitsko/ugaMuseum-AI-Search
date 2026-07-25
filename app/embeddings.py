"""Semantic index over the *enriched* specimen text.

What enrichment does and doesn't buy, measured on this collection with
all-MiniLM-L6-v2:

  "T rex"          raw text already ranks T. rex first (0.60). The tokenizer
                   splits "Tyrannosaurus rex" and the shared "rex" subword
                   carries it. Enrichment lifts the score, not the rank.

  "large predator" raw text ranks a *deer* first and never surfaces the wolf.
                   Enriched, Canis lupus takes rank 1. This is the case keyword
                   search can never reach, and the reason embeddings are here.

Enrichment text must stay short. MiniLM mean-pools its token vectors, so
provenance words (locality, collector, institution) drag the vector away from
what the specimen is. A 34-word prose paragraph about T. rex scores 0.395 on
"big meat-eating dinosaur"; a 10-word alias phrase carrying the same facts
scores 0.583. See ENRICH_SYSTEM in ai.py.
"""

import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_PATH = Path(__file__).parent / "data" / "vectors.npz"

_model = None


def available() -> bool:
    """False when sentence-transformers isn't installed, so search can degrade
    to lexical matching over the enriched text instead of crashing."""
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _fingerprint(texts: list[str]) -> str:
    """Cache key. Changing any enriched text, or reordering, forces a re-encode."""
    digest = hashlib.sha256()
    digest.update(MODEL_NAME.encode())
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _encode(texts: list[str]) -> np.ndarray:
    """Encode to unit-length vectors, so cosine similarity is a plain dot product."""
    vectors = _load_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def build_index(texts: list[str]) -> np.ndarray:
    """Return an (n_specimens, dim) matrix, reusing the cache when texts are unchanged."""
    fingerprint = _fingerprint(texts)

    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH)
        if str(cached["fingerprint"]) == fingerprint:
            return cached["vectors"]

    vectors = _encode(texts)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez(CACHE_PATH, vectors=vectors, fingerprint=np.array(fingerprint))
    return vectors


def similarities(query: str, index: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against every row of `index`, in [-1, 1]."""
    query_vector = _encode([query])[0]
    return index @ query_vector
