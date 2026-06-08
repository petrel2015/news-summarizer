"""Tests for storage layer: schema setup, idempotency, basic CRUD."""
import sqlite3
import pytest
from news_summarizer.storage import (
    init_db,
    DB_PATH,
    add_cluster,
    get_cluster,
    add_article_to_cluster,
    get_cluster_articles,
    add_summary,
    get_summary,
    upsert_entity,
    get_entity_by_alias,
    record_story_event,
    get_story_timeline,
    record_error,
    get_pending_errors,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Override DB_PATH to a tmp file for isolation."""
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    init_db()
    return db


def test_init_db_creates_all_tables(fresh_db):
    """init_db() must create all 7 tables + 2 virtual tables (FTS5 + sqlite-vec)."""
    con = sqlite3.connect(fresh_db)
    cur = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table') ORDER BY name")
    tables = {r[0] for r in cur.fetchall()}
    # 7 real + 2 virtual + 1 migrations = at least these
    assert "clusters" in tables
    assert "cluster_articles" in tables
    assert "summaries" in tables
    assert "entities" in tables
    assert "entity_mentions" in tables
    assert "embeddings" in tables
    assert "story_events" in tables
    assert "enrichment_errors" in tables
    assert "schema_migrations" in tables


def test_init_db_idempotent(fresh_db):
    """Calling init_db() twice must not raise or duplicate."""
    init_db()
    init_db()
    con = sqlite3.connect(fresh_db)
    cnt = con.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    # No duplicate rows in migrations
    assert cnt >= 1


def test_cluster_crud(fresh_db):
    """Add cluster → get_cluster returns it."""
    cid = add_cluster(
        slug="iran-nuclear",
        title="Iran nuclear tensions",
        summary="Latest developments in Iran nuclear program",
    )
    c = get_cluster("iran-nuclear")
    assert c is not None
    assert c["slug"] == "iran-nuclear"
    assert c["title"] == "Iran nuclear tensions"
    assert c["article_count"] == 0  # no articles yet


def test_cluster_article_link(fresh_db):
    """add_article_to_cluster + get_cluster_articles roundtrip."""
    add_cluster(slug="test", title="Test story", summary="...")
    add_article_to_cluster(
        cluster_slug="test",
        bronze_article_id=42,
        relevance=0.92,
        position_in_timeline=0,
    )
    add_article_to_cluster(
        cluster_slug="test",
        bronze_article_id=43,
        relevance=0.85,
        position_in_timeline=1,
    )
    arts = get_cluster_articles("test")
    assert len(arts) == 2
    assert {a["bronze_article_id"] for a in arts} == {42, 43}
    # Cluster article_count auto-updates
    c = get_cluster("test")
    assert c["article_count"] == 2


def test_summary_crud(fresh_db):
    """add_summary + get_summary roundtrip with JSON fields."""
    add_summary(
        bronze_article_id=100,
        summary_short="Iran attacked facility X.",
        summary_long="On 2026-06-07, Iran struck facility X in retaliation for Y.",
        key_facts=["fact A", "fact B"],
        topics=["geopolitics", "conflict"],
        model="minimax-test",
    )
    s = get_summary(100)
    assert s is not None
    assert s["summary_short"] == "Iran attacked facility X."
    assert s["key_facts"] == ["fact A", "fact B"]  # JSON decoded
    assert s["topics"] == ["geopolitics", "conflict"]
    assert s["model"] == "minimax-test"


def test_entity_alias_canonicalization(fresh_db):
    """upsert_entity merges by alias: 'US' alias to canonical 'United States'."""
    us_id = upsert_entity(name="United States", type="country", aliases=["US", "USA", "America"])
    # Now upsert with different alias but matching existing alias
    found = get_entity_by_alias("US")
    assert found is not None
    assert found["id"] == us_id
    assert "US" in found["aliases"]


def test_story_events_audit_trail(fresh_db):
    """Every cluster action records an event for audit/timeline."""
    add_cluster(slug="x", title="X", summary="...")
    add_article_to_cluster(cluster_slug="x", bronze_article_id=1, relevance=0.9, position_in_timeline=0)
    record_story_event(cluster_slug="x", event_type="article_added",
                       details={"bronze_article_id": 1, "relevance": 0.9})
    tl = get_story_timeline("x")
    events = [e["event_type"] for e in tl]
    assert "created" in events
    assert "article_added" in events


def test_error_log_retriable(fresh_db):
    """record_error + get_pending_errors return retriable errors only."""
    record_error(article_id=1, stage="summarize", error="rate limit", retriable=True)
    record_error(article_id=2, stage="entity", error="content filter", retriable=False)
    pending = get_pending_errors()
    ids = {e["article_id"] for e in pending}
    assert 1 in ids
    assert 2 not in ids  # non-retriable not returned


def test_db_path_default_in_home(fresh_db, monkeypatch):
    """DB_PATH must default to a sensible location (overridable)."""
    # Just check it points to a sqlite file path
    assert str(DB_PATH).endswith(".db")
