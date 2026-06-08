# Changelog

All notable changes to news-summarizer.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-08

### Added
- **`summarize` layer** — LLM-first per-article summarization with extractive fallback
  - 3-layer defense: LLM → extractive heuristic → error logged to `enrichment_errors`
  - Short summary (≤280 chars) + long summary (≤1500 chars) + key facts + topics
  - `summarize_article()` + `summarize_batch()` (failures isolated per article)
- **`entities` layer** — Named-entity recognition with alias canonicalization
  - 5 entity types: person / place / org / date / other (strict — invalid types dropped)
  - `upsert_entity()` merges by name OR any alias (case-insensitive)
  - `extract_entities()` + `extract_entities_batch()`
- **`query` layer** — The killer feature: synthesize answers across history
  - `ask_topic(question)` → FTS search → suggested clusters → top articles
  - `compare_sources(topic)` → all matching summaries for cross-source analysis
  - `get_story_timeline(slug)` → chronological event audit trail
  - `search_summaries_fts(query)` — FTS5 with `porter unicode61` tokenizer
  - `list_clusters()` + `get_cluster_summary(slug)`
- **CLI** — 6 subcommands via Click
  - `ingest [--limit] [--since-hours] [--no-summarize] [--no-entities]`
  - `list-clusters [--limit]`
  - `show-cluster <slug> [--timeline] [--json]`
  - `ask <question...> [--limit] [--json/--no-json]`
  - `compare <topic...> [--limit] [--json/--no-json]`
  - `stats`
- **End-to-end smoke verified** — 20 real bronze articles ingested successfully
  (5 distinct stories: Iran-Israel, oil prices, Eriksen collapse, Romanian bank,
  Armenian elections)

### Storage
- New tables: `summaries`, `entities`, `entity_mentions`, `embeddings`,
  `story_events`, `enrichment_errors`
- FTS5 virtual table `fts_summaries` with `porter unicode61` tokenizer + 3
  triggers keeping it in sync with `summaries`
- `upsert_entity()` now also matches by **name** (not just aliases) — fixes a
  bug where "United States" and "America" didn't merge

### Reliability
- LLM failure never breaks the pipeline — extractive fallback always produces
  something
- Errors persisted to `enrichment_errors` with `retriable=1` for later retry
- FTS5 queries with special characters fall back to LIKE search

## [0.1.0] - 2026-06-08

### Added
- **`storage` layer** — 7 real tables + 2 virtual tables, idempotent migrations
  - `clusters`, `cluster_articles`, `summaries` (placeholder), `entities`,
    `entity_mentions`, `embeddings`, `story_events`, `enrichment_errors`
  - `schema_migrations` for versioned, idempotent setup
  - All CRUD helpers: `add_cluster`, `get_cluster`, `add_article_to_cluster`,
    `get_cluster_articles`, `add_summary`, `get_summary`, `upsert_entity`,
    `get_entity_by_alias`, `record_story_event`, `get_story_timeline`,
    `record_error`, `get_pending_errors`
  - sqlite-vec extension loaded with **silent fallback** if unavailable
- **`embed` layer** — 384-dim article embeddings
  - **Primary:** `BAAI/bge-small-en-v1.5` via sentence-transformers (offline,
    ~50MB, no API)
  - **Fallback:** deterministic hash-based pseudo-embedding (always available)
  - `NEWS_SUMMARIZER_FORCE_HASH=1` env var forces fallback for fast tests
- **`cluster` layer** — Cosine-similarity clustering
  - Threshold 0.85 — tight enough to avoid false merges
  - Centroid = mean of member embeddings (computed on demand)
  - `find_similar_cluster()` + `cluster_articles()` + `store_embedding()`
- **Tests** — 23 green at v0.1 (storage 9 + embed 6 + cluster 8)

### Design
- **Medallion architecture** — this is the Silver layer over the Bronze
  `news-aggregator`; future Gold layer (trends/predictions) is out of scope
- **Story-level deduplication** — bronze is per-article raw, silver is per-story
  evolving knowledge. Cross-source articles about the same event merge into one
  cluster; a new article updates the cluster's summary
- **No background workers** — passive service. CLI/cron runs the pipeline.
  Other agents query via CLI/MCP/HTTP (MCP/HTTP coming in v0.3)

[0.2.0]: https://github.com/petrel2015/news-summarizer/releases/tag/v0.2.0
[0.1.0]: https://github.com/petrel2015/news-summarizer/releases/tag/v0.1.0
