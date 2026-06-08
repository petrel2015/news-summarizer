"""Embedding layer.

Primary: BAAI/bge-small-en-v1.5 (384-dim, ~50MB, runs locally via sentence-transformers).
Fallback: deterministic hash-based pseudo-embedding (always available, no model download).

The fallback is good enough for tests + smoke runs. Real semantic clustering only works
with the real model. Set NEWS_SUMMARIZER_FORCE_HASH=1 to force fallback even if the
real model is available (useful for fast unit tests).
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np

EMBEDDING_DIM = 384

# Real model identifier (BGE-small is widely available, small, English-tuned).
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Lazy-loaded model handle (None until first use)
_model = None
_model_name: str | None = None
_force_hash: bool = bool(os.environ.get("NEWS_SUMMARIZER_FORCE_HASH"))


def get_model() -> str:
    """Return the model identifier currently in use (real model name or 'hash-fallback')."""
    if _force_hash:
        return "hash-fallback"
    if _model is None:
        _load_model()
    return _model_name or "hash-fallback"


def _load_model() -> None:
    """Try to load sentence-transformers model. Fall back to hash on any failure."""
    global _model, _model_name
    if _force_hash:
        _model_name = "hash-fallback"
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(DEFAULT_MODEL)
        _model_name = DEFAULT_MODEL
    except Exception:
        # Model not installed / network down / OOM / etc. — silent fallback.
        _model = None
        _model_name = "hash-fallback"


def _hash_embed(text: str) -> np.ndarray:
    """Deterministic 384-dim pseudo-embedding from text hash + bag-of-words.

    This is NOT semantically meaningful. It only provides a stable vector for
    identical inputs (good for dedup tests), with random direction for distinct
    inputs. Real semantic clustering requires the real model.
    """
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    # Bucket 1: SHA-256 hash, mapped to a deterministic seed
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    v += rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.1
    # Bucket 2: bag-of-words bonus (a few n-grams hash to indices)
    tokens = text.lower().split()
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % EMBEDDING_DIM
        v[idx] += 0.5
    # Normalize
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def embed_text(text: str) -> np.ndarray:
    """Embed a single text → 384-dim float32 vector (L2-normalized)."""
    if _force_hash or _model is None:
        # Lazy load attempt (skipped if force_hash)
        if _model is None and not _force_hash:
            _load_model()
    if _model is None or _force_hash:
        return _hash_embed(text)
    arr = _model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return arr[0].astype(np.float32)


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed a batch of texts. Returns list of 384-dim float32 vectors."""
    if not texts:
        return []
    if _force_hash or _model is None:
        if _model is None and not _force_hash:
            _load_model()
    if _model is None or _force_hash:
        return [_hash_embed(t) for t in texts]
    arr = _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    return [a.astype(np.float32) for a in arr]
