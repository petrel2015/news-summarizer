"""Clustering layer.

Strategy: cosine similarity to cluster centroids. Above CLUSTER_MATCH_THRESHOLD → join.
Below → create new cluster. Centroid is the mean of all member embeddings.

We use a simple in-memory centroid index, computed on demand from cluster_articles +
embeddings. At news-aggregator's scale (≤ 10k stories), this is fast and avoids
needing a separate vector index.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .storage import (
    add_cluster,
    add_article_to_cluster,
    get_cluster,
)
from .embed import EMBEDDING_DIM

# Cosine sim threshold: above this, an article joins an existing cluster.
CLUSTER_MATCH_THRESHOLD = 0.85


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D float vectors (assumed L2-normalized)."""
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _all_cluster_centroids() -> list[dict[str, Any]]:
    """Compute centroid for every cluster by averaging member article embeddings.

    Returns: list of {slug, centroid (ndarray), article_count}.
    """
    from .storage import connect
    with connect() as con:
        rows = con.execute(
            """SELECT c.slug, ca.bronze_article_id, c.id as cluster_id
               FROM clusters c
               JOIN cluster_articles ca ON ca.cluster_id = c.id"""
        ).fetchall()
    # Group by cluster
    by_cluster: dict[int, list[int]] = {}
    slug_by_id: dict[int, str] = {}
    for r in rows:
        by_cluster.setdefault(r["cluster_id"], []).append(r["bronze_article_id"])
        slug_by_id[r["cluster_id"]] = r["slug"]
    if not by_cluster:
        return []
    # Fetch all needed embeddings from vec_embeddings
    article_ids = {a for arts in by_cluster.values() for a in arts}
    emb_map = _fetch_embeddings(list(article_ids))
    out = []
    for cid, arts in by_cluster.items():
        vecs = [emb_map[a] for a in arts if a in emb_map]
        if not vecs:
            continue
        centroid = np.mean(np.stack(vecs), axis=0).astype(np.float32)
        n = float(np.linalg.norm(centroid))
        if n > 0:
            centroid /= n
        out.append({"slug": slug_by_id[cid], "centroid": centroid, "article_count": len(arts)})
    return out


def _fetch_embeddings(bronze_article_ids: list[int]) -> dict[int, np.ndarray]:
    """Fetch embeddings for a list of bronze article IDs.

    Reads from vec_embeddings virtual table if available, else returns empty
    (clustering will be purely by title until embeddings are populated).
    """
    from .storage import connect, _HAS_SQLITE_VEC
    out: dict[int, np.ndarray] = {}
    if not bronze_article_ids or not _HAS_SQLITE_VEC:
        return out
    with connect() as con:
        try:
            for aid in bronze_article_ids:
                row = con.execute(
                    "SELECT embedding FROM vec_embeddings WHERE bronze_article_id = ?",
                    (aid,),
                ).fetchone()
                if row and row[0] is not None:
                    # sqlite-vec returns BLOB; convert to ndarray
                    blob = row[0]
                    if isinstance(blob, bytes):
                        arr = np.frombuffer(blob, dtype=np.float32)
                    else:
                        arr = np.asarray(blob, dtype=np.float32)
                    out[aid] = arr
        except Exception:
            return {}
    return out


def store_embedding(bronze_article_id: int, embedding: np.ndarray, model: str = "") -> None:
    """Persist a single embedding to vec_embeddings + embeddings metadata table."""
    from .storage import connect, _HAS_SQLITE_VEC
    if not _HAS_SQLITE_VEC:
        return
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"embedding must be shape ({EMBEDDING_DIM},), got {embedding.shape}")
    # Normalize before storing
    v = embedding.astype(np.float32).copy()
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    now = time.time()
    with connect() as con:
        try:
            con.execute(
                "INSERT OR REPLACE INTO vec_embeddings(bronze_article_id, embedding) VALUES (?, ?)",
                (bronze_article_id, v.tobytes()),
            )
            con.execute(
                """INSERT OR REPLACE INTO embeddings(bronze_article_id, model, dim, generated_at)
                   VALUES (?, ?, ?, ?)""",
                (bronze_article_id, model or "unknown", EMBEDDING_DIM, now),
            )
            con.commit()
        except Exception:
            pass


def find_similar_cluster(embedding: np.ndarray) -> dict | None:
    """Find the best-matching cluster above threshold, or None.

    Returns: {slug, similarity, article_count} or None.
    """
    centroids = _all_cluster_centroids()
    if not centroids:
        return None
    best = None
    best_sim = -1.0
    v = embedding.astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    for c in centroids:
        sim = cosine_sim(v, c["centroid"])
        if sim > best_sim:
            best_sim = sim
            best = c
    if best is None or best_sim < CLUSTER_MATCH_THRESHOLD:
        return None
    return {"slug": best["slug"], "similarity": best_sim, "article_count": best["article_count"]}


def cluster_articles(
    *,
    bronze_article_id: int,
    embedding: np.ndarray,
    title: str,
    summary_short: str = "",
) -> dict:
    """Decide what to do with a new article: join existing or create new.

    Returns: {action: 'created'|'joined'|'updated', cluster_slug, similarity?}
    """
    # Store embedding first (so future centroid calcs include it)
    store_embedding(bronze_article_id, embedding)
    # Find best match
    match = find_similar_cluster(embedding)
    if match is None:
        # Create new
        slug = _title_to_slug(title)
        existing = get_cluster(slug)
        if existing:
            # Slug collision (different topic, same slug) → append hash
            import uuid
            slug = f"{slug}-{uuid.uuid4().hex[:6]}"
        cid = add_cluster(slug=slug, title=title, summary=summary_short)
        add_article_to_cluster(
            cluster_slug=slug, bronze_article_id=bronze_article_id, relevance=1.0, position_in_timeline=0
        )
        return {"action": "created", "cluster_slug": slug, "cluster_id": cid}
    else:
        add_article_to_cluster(
            cluster_slug=match["slug"],
            bronze_article_id=bronze_article_id,
            relevance=match["similarity"],
            position_in_timeline=0,
        )
        return {
            "action": "joined",
            "cluster_slug": match["slug"],
            "similarity": match["similarity"],
        }


def _title_to_slug(title: str) -> str:
    import re
    s = title.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80]
