"""Storage layer: SQLite + FTS5 + sqlite-vec.

Single-file DB at ~/.local/share/news-summarizer/silver.db by default.
Override via DB_PATH module attribute (tests do this).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# Lazy / optional sqlite-vec import. Tests that don't touch vec can run without it.
try:
    import sqlite_vec  # noqa: F401  (loaded via conn.enable_load_extension below)
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

# Default DB location. Tests override this with monkeypatch.
DEFAULT_HOME = Path(os.environ.get("NEWS_SUMMARIZER_HOME", Path.home() / ".local" / "share" / "news-summarizer"))
DB_PATH: Path = DEFAULT_HOME / "silver.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection, register sqlite-vec if available, set row factory."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    if _HAS_SQLITE_VEC:
        try:
            con.enable_load_extension(True)
            sqlite_vec.load(con)
            con.enable_load_extension(False)
        except Exception:
            # Some sandboxes block extension loading; vec features silently degrade.
            pass
    return con


@contextmanager
def connect(db_path: Path | None = None):
    """Context-managed connection with auto-close."""
    con = _connect(db_path)
    try:
        yield con
    finally:
        con.close()


# ---------- schema ----------

SCHEMA = [
    # 0. migrations bookkeeping
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at REAL NOT NULL
    )
    """,
    # 1. clusters (stories)
    """
    CREATE TABLE IF NOT EXISTS clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slug TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        summary_updated_at REAL,
        article_count INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    )
    """,
    # 2. cluster <-> bronze articles (many-to-many)
    """
    CREATE TABLE IF NOT EXISTS cluster_articles (
        cluster_id INTEGER NOT NULL,
        bronze_article_id INTEGER NOT NULL,
        relevance REAL NOT NULL DEFAULT 0.0,
        position_in_timeline INTEGER NOT NULL DEFAULT 0,
        added_at REAL NOT NULL,
        PRIMARY KEY (cluster_id, bronze_article_id),
        FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cluster_articles_article ON cluster_articles(bronze_article_id)",
    # 3. per-article summaries
    """
    CREATE TABLE IF NOT EXISTS summaries (
        bronze_article_id INTEGER PRIMARY KEY,
        summary_short TEXT NOT NULL,
        summary_long TEXT NOT NULL,
        key_facts TEXT NOT NULL DEFAULT '[]',  -- JSON array
        topics TEXT NOT NULL DEFAULT '[]',     -- JSON array
        model TEXT NOT NULL DEFAULT '',
        generated_at REAL NOT NULL
    )
    """,
    # 4. entities (people / places / orgs)
    """
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,  -- person|place|org|date|other
        canonical_id INTEGER,  -- self-FK for alias merge
        aliases TEXT NOT NULL DEFAULT '[]',  -- JSON array
        created_at REAL NOT NULL,
        FOREIGN KEY (canonical_id) REFERENCES entities(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_id)",
    # 5. entity mentions (per article)
    """
    CREATE TABLE IF NOT EXISTS entity_mentions (
        entity_id INTEGER NOT NULL,
        bronze_article_id INTEGER NOT NULL,
        count INTEGER NOT NULL DEFAULT 1,
        context TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (entity_id, bronze_article_id),
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mentions_article ON entity_mentions(bronze_article_id)",
    # 6. embeddings metadata (the vec itself lives in vec_embeddings virtual table)
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        bronze_article_id INTEGER PRIMARY KEY,
        model TEXT NOT NULL,
        dim INTEGER NOT NULL,
        generated_at REAL NOT NULL
    )
    """,
    # 7. story events audit trail
    """
    CREATE TABLE IF NOT EXISTS story_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,  -- created|article_added|merged|split|summary_refreshed
        details TEXT NOT NULL DEFAULT '{}',  -- JSON
        created_at REAL NOT NULL,
        FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_story_events_cluster ON story_events(cluster_id, created_at)",
    # 8. enrichment errors
    """
    CREATE TABLE IF NOT EXISTS enrichment_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL,
        stage TEXT NOT NULL,  -- embed|summarize|entity|cluster
        error TEXT NOT NULL,
        retriable INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_errors_pending ON enrichment_errors(retriable, article_id)",
]

# FTS5 virtual table (contentless mirror of summaries)
FTS_SCHEMA = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_summaries USING fts5(
        summary_short,
        summary_long,
        content='summaries',
        content_rowid='bronze_article_id',
        tokenize='porter unicode61'
    )
    """,
    # Triggers to keep FTS in sync with summaries table
    """
    CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
        INSERT INTO fts_summaries(rowid, summary_short, summary_long)
        VALUES (new.bronze_article_id, new.summary_short, new.summary_long);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
        INSERT INTO fts_summaries(fts_summaries, rowid, summary_short, summary_long)
        VALUES ('delete', old.bronze_article_id, old.summary_short, old.summary_long);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON summaries BEGIN
        INSERT INTO fts_summaries(fts_summaries, rowid, summary_short, summary_long)
        VALUES ('delete', old.bronze_article_id, old.summary_short, old.summary_long);
        INSERT INTO fts_summaries(rowid, summary_short, summary_long)
        VALUES (new.bronze_article_id, new.summary_short, new.summary_long);
    END
    """,
]

# sqlite-vec virtual table (only created if extension loaded)
VEC_SCHEMA = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
        bronze_article_id INTEGER PRIMARY KEY,
        embedding FLOAT[384]
    )
    """,
]


def init_db(db_path: Path | None = None) -> None:
    """Idempotently create all tables, FTS, and (if available) vec."""
    with connect(db_path) as con:
        for stmt in SCHEMA:
            con.execute(stmt)
        for stmt in FTS_SCHEMA:
            con.execute(stmt)
        if _HAS_SQLITE_VEC:
            for stmt in VEC_SCHEMA:
                try:
                    con.execute(stmt)
                except Exception:
                    pass  # vec0 might fail in restricted envs
        # Record migration
        existing = {r[0] for r in con.execute("SELECT version FROM schema_migrations").fetchall()}
        for v in (1,):
            if v not in existing:
                con.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (v, time.time()))
        con.commit()


# ---------- helpers ----------

def _slugify(s: str) -> str:
    """Stable slug for a title (lowercase, alnum + dash)."""
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80] or uuid.uuid4().hex[:8]


# ---------- cluster CRUD ----------

def add_cluster(*, slug: str | None = None, title: str, summary: str = "") -> int:
    """Insert a new cluster. Slug auto-generated from title if not given. Returns id."""
    if not slug:
        slug = _slugify(title)
    now = time.time()
    with connect() as con:
        cur = con.execute(
            "INSERT INTO clusters(slug, title, summary, created_at) VALUES (?, ?, ?, ?)",
            (slug, title, summary, now),
        )
        cid = cur.lastrowid
        con.execute(
            "INSERT INTO story_events(cluster_id, event_type, details, created_at) VALUES (?, ?, ?, ?)",
            (cid, "created", json.dumps({"title": title}), now),
        )
        con.commit()
    return cid


def get_cluster(slug: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM clusters WHERE slug = ?", (slug,)).fetchone()
    return dict(row) if row else None


def add_article_to_cluster(
    *, cluster_slug: str, bronze_article_id: int, relevance: float = 0.0, position_in_timeline: int = 0
) -> None:
    """Link a bronze article to a cluster. Updates article_count + logs event."""
    now = time.time()
    with connect() as con:
        cid = con.execute("SELECT id FROM clusters WHERE slug = ?", (cluster_slug,)).fetchone()
        if not cid:
            raise ValueError(f"cluster not found: {cluster_slug}")
        cid = cid[0]
        con.execute(
            """INSERT OR REPLACE INTO cluster_articles
               (cluster_id, bronze_article_id, relevance, position_in_timeline, added_at)
               VALUES (?, ?, ?, ?, ?)""",
            (cid, bronze_article_id, relevance, position_in_timeline, now),
        )
        con.execute(
            """UPDATE clusters SET article_count = (
                 SELECT COUNT(*) FROM cluster_articles WHERE cluster_id = ?
               ) WHERE id = ?""",
            (cid, cid),
        )
        con.execute(
            "INSERT INTO story_events(cluster_id, event_type, details, created_at) VALUES (?, ?, ?, ?)",
            (cid, "article_added", json.dumps({"bronze_article_id": bronze_article_id, "relevance": relevance}), now),
        )
        con.commit()


def get_cluster_articles(cluster_slug: str) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT ca.* FROM cluster_articles ca
               JOIN clusters c ON c.id = ca.cluster_id
               WHERE c.slug = ?
               ORDER BY ca.position_in_timeline, ca.added_at""",
            (cluster_slug,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- summary CRUD ----------

def add_summary(
    *,
    bronze_article_id: int,
    summary_short: str,
    summary_long: str,
    key_facts: list[str] | None = None,
    topics: list[str] | None = None,
    model: str = "",
) -> None:
    now = time.time()
    with connect() as con:
        con.execute(
            """INSERT OR REPLACE INTO summaries
               (bronze_article_id, summary_short, summary_long, key_facts, topics, model, generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                bronze_article_id,
                summary_short,
                summary_long,
                json.dumps(key_facts or []),
                json.dumps(topics or []),
                model,
                now,
            ),
        )
        con.commit()


def get_summary(bronze_article_id: int) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM summaries WHERE bronze_article_id = ?", (bronze_article_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["key_facts"] = json.loads(d["key_facts"])
    d["topics"] = json.loads(d["topics"])
    return d


# ---------- entity CRUD ----------

def upsert_entity(*, name: str, type: str, aliases: list[str] | None = None) -> int:
    """Insert entity, or merge into existing one if name or any alias matches.

    Aliases are stored as JSON. Aliases default to [name] if not given.
    Matching is case-insensitive on both name and aliases.
    """
    aliases = list(aliases or [name])
    if name not in aliases:
        aliases.append(name)
    now = time.time()
    target_keys = {a.lower().strip() for a in aliases} | {name.lower().strip()}
    with connect() as con:
        # Check if any existing entity matches by name or alias
        rows = con.execute("SELECT id, name, aliases FROM entities").fetchall()
        for r in rows:
            existing_keys = {r["name"].lower().strip()} | {
                a.lower().strip() for a in json.loads(r["aliases"])
            }
            if target_keys & existing_keys:
                # Merge: add new aliases to existing entity
                merged = list(set(json.loads(r["aliases"])) | set(aliases))
                con.execute(
                    "UPDATE entities SET aliases = ? WHERE id = ?",
                    (json.dumps(merged), r["id"]),
                )
                con.commit()
                return r["id"]
        # No match: insert new
        cur = con.execute(
            "INSERT INTO entities(name, type, aliases, created_at) VALUES (?, ?, ?, ?)",
            (name, type, json.dumps(aliases), now),
        )
        con.commit()
        return cur.lastrowid


def get_entity_by_alias(alias: str) -> dict | None:
    """Find entity by any of its aliases (case-insensitive contains)."""
    with connect() as con:
        rows = con.execute("SELECT * FROM entities").fetchall()
    target = alias.lower().strip()
    for r in rows:
        aliases = json.loads(r["aliases"])
        if any(target == a.lower() for a in aliases):
            return dict(r)
    return None


# ---------- story events ----------

def record_story_event(*, cluster_slug: str, event_type: str, details: dict | None = None) -> None:
    now = time.time()
    with connect() as con:
        cid = con.execute("SELECT id FROM clusters WHERE slug = ?", (cluster_slug,)).fetchone()
        if not cid:
            raise ValueError(f"cluster not found: {cluster_slug}")
        con.execute(
            "INSERT INTO story_events(cluster_id, event_type, details, created_at) VALUES (?, ?, ?, ?)",
            (cid[0], event_type, json.dumps(details or {}), now),
        )
        con.commit()


def get_story_timeline(cluster_slug: str) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT se.* FROM story_events se
               JOIN clusters c ON c.id = se.cluster_id
               WHERE c.slug = ?
               ORDER BY se.created_at""",
            (cluster_slug,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d["details"])
        out.append(d)
    return out


# ---------- error log ----------

def record_error(*, article_id: int, stage: str, error: str, retriable: bool = True) -> None:
    now = time.time()
    with connect() as con:
        con.execute(
            "INSERT INTO enrichment_errors(article_id, stage, error, retriable, created_at) VALUES (?, ?, ?, ?, ?)",
            (article_id, stage, error, int(retriable), now),
        )
        con.commit()


def get_pending_errors() -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT * FROM enrichment_errors
               WHERE retriable = 1
               ORDER BY created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]
