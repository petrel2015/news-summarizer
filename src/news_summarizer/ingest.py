"""Ingest layer: read news-aggregator (bronze) articles and feed into silver.

For v0.1, we read directly from the bronze SQLite DB.
For v0.3+, we can also use news-aggregator's MCP/HTTP API.

Bronze article schema (relevant fields):
  - id (INTEGER)
  - source_id (TEXT)
  - title (TEXT)
  - url (TEXT)
  - summary (TEXT)  -- the bronze short summary
  - body_excerpt (TEXT)  -- the bronze body snippet (may be NULL)
  - published_at (REAL)
  - fetched_at (REAL)
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import storage
from .embed import embed_text
from .cluster import cluster_articles
from .summarize import summarize_article
from .entities import extract_entities


DEFAULT_BRONZE_HOME = Path(os.environ.get("NEWS_AGGREGATOR_HOME", Path.home() / ".local" / "share" / "news-aggregator"))
BRONZE_DB_PATH: Path = DEFAULT_BRONZE_HOME / "news.db"


def _get_bronze_db_path() -> Path | None:
    """Locate the bronze DB. Returns None if not found."""
    if BRONZE_DB_PATH.exists():
        return BRONZE_DB_PATH
    # Try alternative locations
    for cand in [
        Path("/home/user/code/news-aggregator/data/news.db"),
        Path("/home/user/.local/share/news-aggregator/news.db"),
    ]:
        if cand.exists():
            return cand
    return None


def read_bronze_articles(
    *,
    since_hours: int = 24,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read recent bronze articles from news-aggregator DB.

    Returns: list of {id, source_id, title, summary, body_excerpt, published_at, url}
    """
    db_path = _get_bronze_db_path()
    if db_path is None:
        return []
    cutoff = time.time() - (since_hours * 3600)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT id, source_id, title, summary, body_excerpt, published_at, url
               FROM news
               WHERE (published_at IS NULL OR published_at >= ?)
                  OR (fetched_at IS NULL OR fetched_at >= ?)
               ORDER BY COALESCE(published_at, fetched_at) DESC
               LIMIT ?""",
            (cutoff, cutoff, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        # Schema may differ in older news-aggregator versions
        if "no such column" in str(e):
            rows = con.execute(
                """SELECT id, source_id, title, summary, published_at, url
                   FROM news
                   ORDER BY COALESCE(published_at, fetched_at) DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            for r in rows:
                r = dict(r)
                r.setdefault("body_excerpt", None)
        else:
            raise
    con.close()
    out = []
    for r in rows:
        d = dict(r)
        d.setdefault("body_excerpt", None)
        out.append(d)
    return out


def _ingest_one(
    article: dict[str, Any],
    *,
    do_summarize: bool = True,
    do_entities: bool = True,
) -> dict[str, Any]:
    """Ingest a single bronze article: embed + cluster + summarize + entity extract."""
    bronze_id = article["id"]
    title = article.get("title") or ""
    summary = article.get("summary") or ""
    body = article.get("body_excerpt") or summary or ""
    # 1. Embed
    try:
        text_to_embed = f"{title}\n\n{body}"[:4000]
        vec = embed_text(text_to_embed)
    except Exception as e:
        storage.record_error(article_id=bronze_id, stage="embed", error=str(e)[:500], retriable=True)
        return {"bronze_article_id": bronze_id, "status": "embed_error", "error": str(e)}
    # 2. Cluster
    try:
        cluster_result = cluster_articles(
            bronze_article_id=bronze_id,
            embedding=vec,
            title=title[:200] or f"Article {bronze_id}",
            summary_short=summary[:280],
        )
    except Exception as e:
        storage.record_error(article_id=bronze_id, stage="cluster", error=str(e)[:500], retriable=True)
        return {"bronze_article_id": bronze_id, "status": "cluster_error", "error": str(e)}
    out = {
        "bronze_article_id": bronze_id,
        "status": "ok",
        "cluster_slug": cluster_result["cluster_slug"],
        "cluster_action": cluster_result["action"],
    }
    # 3. Summarize (LLM-first, extractive fallback)
    if do_summarize:
        try:
            sr = summarize_article(bronze_article_id=bronze_id, title=title, body=body)
            out["summary_source"] = sr.source
        except Exception as e:
            out["summary_source"] = "error"
            out["summary_error"] = str(e)[:200]
    # 4. Entities
    if do_entities:
        try:
            er = extract_entities(bronze_article_id=bronze_id, title=title, body=body)
            out["entity_count"] = len(er.entities)
            out["entity_source"] = er.source
        except Exception as e:
            out["entity_count"] = 0
            out["entity_source"] = "error"
            out["entity_error"] = str(e)[:200]
    return out


def ingest_batch(
    *,
    limit: int = 50,
    since_hours: int = 24,
    do_summarize: bool = True,
    do_entities: bool = True,
) -> dict[str, Any]:
    """Ingest a batch of bronze articles into silver.

    Returns: {total, ok, errors, by_status, results}
    """
    articles = read_bronze_articles(since_hours=since_hours, limit=limit)
    if not articles:
        return {
            "total": 0,
            "ok": 0,
            "errors": 0,
            "by_status": {},
            "results": [],
            "note": "no bronze articles found (DB missing or no recent articles)",
        }
    results = []
    by_status: dict[str, int] = {}
    for art in articles:
        r = _ingest_one(art, do_summarize=do_summarize, do_entities=do_entities)
        results.append(r)
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
    return {
        "total": len(articles),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "errors": sum(1 for r in results if r.get("status") != "ok"),
        "by_status": by_status,
        "results": results,
    }
