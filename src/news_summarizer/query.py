"""Query layer: ask, compare, timeline, FTS search.

This is the read-path: take a question, return synthesized answer that draws
on clusters + summaries + entities. The killer feature of news-summarizer.
"""
from __future__ import annotations

import re
import time
from typing import Any

from . import storage


# ---------- cluster listing ----------

def list_clusters(limit: int = 100) -> list[dict]:
    """Return clusters sorted by recency (newest first)."""
    with storage.connect() as con:
        rows = con.execute(
            """SELECT * FROM clusters
               ORDER BY COALESCE(summary_updated_at, created_at) DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- cluster detail ----------

def get_cluster_summary(slug: str) -> dict | None:
    """Return aggregated view of a cluster: title + summary + articles + (later) entities.

    Articles: each with bronze_article_id, summary_short, summary_long, key_facts, topics.
    """
    cluster = storage.get_cluster(slug)
    if not cluster:
        return None
    arts = storage.get_cluster_articles(slug)
    article_views = []
    for a in arts:
        s = storage.get_summary(a["bronze_article_id"])
        article_views.append({
            "bronze_article_id": a["bronze_article_id"],
            "relevance": a["relevance"],
            "added_at": a["added_at"],
            "summary_short": s["summary_short"] if s else "",
            "summary_long": s["summary_long"] if s else "",
            "key_facts": s["key_facts"] if s else [],
            "topics": s["topics"] if s else [],
        })
    return {
        "slug": cluster["slug"],
        "title": cluster["title"],
        "summary": cluster["summary"],
        "summary_updated_at": cluster["summary_updated_at"],
        "article_count": cluster["article_count"],
        "created_at": cluster["created_at"],
        "articles": article_views,
    }


# ---------- timeline ----------

def get_story_timeline(slug: str) -> list[dict]:
    """Return chronological list of story events (already in storage helper)."""
    return storage.get_story_timeline(slug)


# ---------- FTS search ----------

_FTS_SAFE = re.compile(r"[\"\'\(\)\*\^\-\+]")


def _sanitize_fts_query(q: str) -> str:
    """Strip FTS5 special chars to avoid syntax errors on user input."""
    q = q.strip()
    if not q:
        return ""
    return _FTS_SAFE.sub(" ", q)


def search_summaries_fts(query: str, limit: int = 20) -> list[dict]:
    """FTS5 search over summaries (short + long). Returns matching bronze_article_ids + excerpts."""
    safe = _sanitize_fts_query(query)
    if not safe:
        return []
    # Try multi-word AND: "word1 word2" → 'word1' AND 'word2' for better recall
    terms = [t for t in safe.split() if t]
    if not terms:
        return []
    fts_query = " AND ".join(f'"{t}"' for t in terms[:6])  # cap to 6 terms
    with storage.connect() as con:
        try:
            rows = con.execute(
                """SELECT s.bronze_article_id, s.summary_short, s.summary_long,
                          s.key_facts, s.topics, s.model,
                          snippet(fts_summaries, 1, '[', ']', '...', 12) AS excerpt
                   FROM fts_summaries f
                   JOIN summaries s ON s.bronze_article_id = f.rowid
                   WHERE fts_summaries MATCH ?
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        except Exception:
            # FTS syntax error on edge cases → fallback to LIKE
            like = f"%{safe[:80]}%"
            rows = con.execute(
                """SELECT bronze_article_id, summary_short, summary_long,
                          key_facts, topics, model,
                          summary_short AS excerpt
                   FROM summaries
                   WHERE summary_short LIKE ? OR summary_long LIKE ?
                   LIMIT ?""",
                (like, like, limit),
            ).fetchall()
    import json
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["key_facts"] = json.loads(d["key_facts"])
            d["topics"] = json.loads(d["topics"])
        except Exception:
            d["key_facts"] = []
            d["topics"] = []
        out.append(d)
    return out


# ---------- ask ----------

def ask_topic(question: str, limit: int = 5) -> dict:
    """Main ask entry point. Given a question/topic, return a rich response.

    Returns: {
        question: str,
        matches: [fts hits...],
        suggested_clusters: [cluster summaries...],
        top_articles: [article summaries...],
        entities: [entity names mentioned in matches],
    }

    Strategy:
    1. FTS search over summaries → matches
    2. For each match, find its cluster(s) → suggested_clusters
    3. Top articles: take up to `limit` matches, sorted by relevance
    4. Entities: collect all entity mentions in the matching articles (best-effort)
    """
    question = question.strip()
    matches = search_summaries_fts(question, limit=limit * 4)
    # Find which clusters these articles belong to
    cluster_slugs: set[str] = set()
    for m in matches:
        # Query cluster_articles by bronze_article_id
        with storage.connect() as con:
            r = con.execute(
                """SELECT c.slug FROM clusters c
                   JOIN cluster_articles ca ON ca.cluster_id = c.id
                   WHERE ca.bronze_article_id = ?""",
                (m["bronze_article_id"],),
            ).fetchall()
        for row in r:
            cluster_slugs.add(row["slug"])
    suggested = []
    for slug in cluster_slugs:
        c = get_cluster_summary(slug)
        if c:
            suggested.append(c)
    # Top articles: first `limit` matches
    top_articles = matches[:limit]
    return {
        "question": question,
        "matches": matches,
        "suggested_clusters": suggested,
        "top_articles": top_articles,
        "entities": [],  # TODO: collect from entity_mentions (bronze lookup needed)
    }


# ---------- compare ----------

def compare_sources(topic: str, limit: int = 50) -> dict:
    """Find summaries mentioning topic, return all of them for cross-source analysis.

    Without explicit source attribution in silver (that lives in bronze), we return
    the matching summaries; the consumer can group by bronze article_id → source
    via news-aggregator.
    """
    matches = search_summaries_fts(topic, limit=limit)
    return {
        "topic": topic,
        "summaries": matches,
    }
