"""Tests for ask/query layer: cross-source story retrieval + timeline + comparison.

This is the killer feature: given a topic, return a synthesized answer that
draws on historical clusters + summaries + entities.
"""
import time
import pytest
from news_summarizer import storage
from news_summarizer.query import (
    ask_topic,
    compare_sources,
    get_story_timeline,
    get_cluster_summary,
    list_clusters,
    search_summaries_fts,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    storage.init_db()
    return db


def _seed_story(db, slug, title, summary, articles_data, summary_data):
    """Helper: create a cluster with articles + summaries."""
    storage.add_cluster(slug=slug, title=title, summary=summary)
    for bronze_article_id, relevance in articles_data:
        storage.add_article_to_cluster(
            cluster_slug=slug, bronze_article_id=bronze_article_id,
            relevance=relevance, position_in_timeline=0
        )
    for bronze_article_id, short, long, facts, topics in summary_data:
        storage.add_summary(
            bronze_article_id=bronze_article_id,
            summary_short=short,
            summary_long=long,
            key_facts=facts,
            topics=topics,
            model="test",
        )


def test_list_clusters_returns_all(fresh_db):
    """list_clusters() returns slug, title, article_count, summary_updated_at."""
    storage.add_cluster(slug="a", title="A", summary="")
    storage.add_cluster(slug="b", title="B", summary="")
    clusters = list_clusters()
    slugs = {c["slug"] for c in clusters}
    assert "a" in slugs
    assert "b" in slugs


def test_list_clusters_sorted_by_recency(fresh_db):
    """Newer clusters first (by summary_updated_at or created_at)."""
    storage.add_cluster(slug="old", title="Old", summary="")
    time.sleep(0.01)
    storage.add_cluster(slug="new", title="New", summary="")
    clusters = list_clusters()
    # New should come first
    assert clusters[0]["slug"] == "new"


def test_get_cluster_summary_returns_aggregated_view(fresh_db):
    """Returns {title, summary, articles: [{bronze_article_id, summary_short, source}], entities: [...]}"""
    _seed_story(
        fresh_db, "iran", "Iran nuclear tensions",
        summary="Latest developments",
        articles_data=[(1, 0.9), (2, 0.85)],
        summary_data=[
            (1, "Iran struck facility.", "Long body 1", ["fact1"], ["geopolitics"]),
            (2, "Russia responds.", "Long body 2", ["fact2"], ["geopolitics"]),
        ],
    )
    out = get_cluster_summary("iran")
    assert out["title"] == "Iran nuclear tensions"
    assert out["article_count"] == 2
    assert len(out["articles"]) == 2
    shorts = {a["summary_short"] for a in out["articles"]}
    assert "Iran struck facility." in shorts
    assert "Russia responds." in shorts


def test_get_cluster_summary_missing_returns_none(fresh_db):
    assert get_cluster_summary("nonexistent") is None


def test_get_story_timeline_chronological(fresh_db):
    """Returns story events in chronological order."""
    _seed_story(
        fresh_db, "x", "X", "...",
        articles_data=[(1, 0.9)],
        summary_data=[(1, "short", "long", [], [])],
    )
    # Add a second article to trigger 'article_added' event
    storage.add_article_to_cluster(cluster_slug="x", bronze_article_id=2, relevance=0.8, position_in_timeline=1)
    timeline = get_story_timeline("x")
    event_types = [e["event_type"] for e in timeline]
    assert event_types[0] == "created"
    assert "article_added" in event_types


def test_search_summaries_fts_finds_keywords(fresh_db):
    """FTS5 search returns matching summaries."""
    _seed_story(
        fresh_db, "nuke", "Nuclear talks",
        "...",
        articles_data=[(1, 1.0)],
        summary_data=[
            (1, "Iran nuclear facility attack", "Strikes on Iranian nuclear enrichment sites reported", ["fact"], ["conflict"]),
        ],
    )
    results = search_summaries_fts("nuclear", limit=5)
    assert len(results) >= 1
    assert "nuclear" in results[0]["summary_short"].lower() or "nuclear" in results[0]["summary_long"].lower()


def test_search_summaries_fts_empty_query(fresh_db):
    """Empty query returns empty list (don't match everything)."""
    _seed_story(
        fresh_db, "x", "X", "...",
        articles_data=[(1, 1.0)],
        summary_data=[(1, "Some summary text", "long body", [], [])],
    )
    results = search_summaries_fts("", limit=5)
    assert results == []


def test_ask_topic_returns_rich_response(fresh_db):
    """ask_topic('Iran nuclear') returns {matches, suggested_clusters, top_articles, entities}."""
    _seed_story(
        fresh_db, "iran-nuclear", "Iran nuclear tensions",
        "Latest developments",
        articles_data=[(1, 0.9), (2, 0.85)],
        summary_data=[
            (1, "Iran nuclear facility attack.", "Long body 1 about nuclear strikes", ["fact1"], ["geopolitics"]),
            (2, "Russia responds to UN on Iran nuclear talks.", "Long body 2 about nuclear diplomacy", ["fact2"], ["geopolitics"]),
        ],
    )
    out = ask_topic("Iran nuclear")
    assert "matches" in out
    assert "top_articles" in out
    assert "suggested_clusters" in out
    # Should find the iran-nuclear cluster via FTS
    slugs = {c["slug"] for c in out["suggested_clusters"]}
    assert "iran-nuclear" in slugs


def test_ask_topic_no_results(fresh_db):
    """ask_topic('xyzzy nothing matches') returns empty suggestion list."""
    out = ask_topic("xyzzy nothing matches")
    assert out["matches"] == [] or out["suggested_clusters"] == []


def test_compare_sources_groups_by_source(fresh_db):
    """compare_sources('Iran') groups summaries by their source attribution.

    Since silver doesn't store source attribution directly, we infer via the
    bronze DB (out of scope here) — for now, we just verify the function
    returns a non-empty structure when there are matching summaries.
    """
    _seed_story(
        fresh_db, "x", "X", "...",
        articles_data=[(1, 1.0)],
        summary_data=[
            (1, "Iran attacked.", "Long A", ["f1"], ["conflict"]),
            (2, "Iran talks.", "Long B", ["f2"], ["diplomacy"]),
        ],
    )
    out = compare_sources("Iran")
    assert "summaries" in out
    assert len(out["summaries"]) >= 2
