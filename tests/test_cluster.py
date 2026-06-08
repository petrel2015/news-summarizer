"""Tests for clustering layer."""
import pytest
import numpy as np
from news_summarizer import storage
from news_summarizer.cluster import (
    cluster_articles,
    find_similar_cluster,
    cosine_sim,
    CLUSTER_MATCH_THRESHOLD,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    storage.init_db()
    return db


def test_cosine_sim_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_sim(a, b) - 1.0) < 1e-5


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_sim(a, b)) < 1e-5


def test_cosine_sim_normalized_inputs():
    """Pre-normalized vectors: cos sim is just dot product."""
    a = np.array([0.6, 0.8])
    b = np.array([0.6, 0.8])
    assert abs(cosine_sim(a, b) - 1.0) < 1e-5


def test_cluster_articles_first_creates_new(fresh_db):
    """First article → no existing cluster → creates new."""
    vec = np.ones(384, dtype=np.float32)
    result = cluster_articles(
        bronze_article_id=1,
        embedding=vec,
        title="Iran nuclear facility attack",
        summary_short="Strike reported",
    )
    assert result["action"] == "created"
    assert result["cluster_slug"] == "iran-nuclear-facility-attack"
    # Cluster exists in DB
    c = storage.get_cluster(result["cluster_slug"])
    assert c is not None
    assert c["article_count"] == 1


def test_cluster_articles_similar_joins_existing(fresh_db):
    """Second article with near-identical vector → joins existing cluster."""
    base = np.random.default_rng(42).standard_normal(384).astype(np.float32)
    base /= np.linalg.norm(base)
    # First article: creates cluster
    r1 = cluster_articles(
        bronze_article_id=1,
        embedding=base,
        title="Iran nuclear facility attack",
        summary_short="...",
    )
    assert r1["action"] == "created"
    # Second article: same direction + tiny noise
    near_same = base + np.random.default_rng(99).standard_normal(384).astype(np.float32) * 0.01
    near_same /= np.linalg.norm(near_same)
    r2 = cluster_articles(
        bronze_article_id=2,
        embedding=near_same,
        title="Strike on Iranian nuclear site",
        summary_short="...",
    )
    assert r2["action"] == "joined"
    assert r2["cluster_slug"] == r1["cluster_slug"]
    # Cluster now has 2 articles
    c = storage.get_cluster(r1["cluster_slug"])
    assert c["article_count"] == 2


def test_cluster_articles_different_creates_new(fresh_db):
    """Second article with very different vector → creates new cluster."""
    v1 = np.random.default_rng(1).standard_normal(384).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.default_rng(2).standard_normal(384).astype(np.float32)
    v2 /= np.linalg.norm(v2)
    r1 = cluster_articles(bronze_article_id=1, embedding=v1, title="A", summary_short="...")
    r2 = cluster_articles(bronze_article_id=2, embedding=v2, title="B", summary_short="...")
    assert r1["action"] == "created"
    assert r2["action"] == "created"
    assert r1["cluster_slug"] != r2["cluster_slug"]


def test_cluster_threshold_is_sensible():
    """Match threshold in (0.5, 0.99) — tight enough to avoid false merges."""
    assert 0.5 < CLUSTER_MATCH_THRESHOLD < 0.99


def test_find_similar_cluster_returns_best_match(fresh_db):
    """find_similar_cluster picks the cluster with highest cosine sim above threshold."""
    from news_summarizer.cluster import store_embedding
    v1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
    v_query = np.array([0.99, 0.01, 0.0] + [0.0] * 381, dtype=np.float32)
    storage.add_cluster(slug="topic-a", title="A", summary="")
    storage.add_cluster(slug="topic-b", title="B", summary="")
    storage.add_article_to_cluster(cluster_slug="topic-a", bronze_article_id=1, relevance=1.0, position_in_timeline=0)
    storage.add_article_to_cluster(cluster_slug="topic-b", bronze_article_id=2, relevance=1.0, position_in_timeline=0)
    # Store embeddings so centroids can be computed
    store_embedding(1, v1)
    store_embedding(2, v2)
    result = find_similar_cluster(v_query)
    assert result is not None
    assert result["slug"] == "topic-a"
    assert result["similarity"] > CLUSTER_MATCH_THRESHOLD
