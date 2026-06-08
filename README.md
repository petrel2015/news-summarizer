# news-summarizer

**Silver layer over [news-aggregator](https://github.com/petrel2015/news-aggregator)** — cluster, summarize, understand, and update news stories across sources and time.

Part of a 3-tier medallion architecture:

| Layer | Project | Role |
|---|---|---|
| **Bronze** | `news-aggregator` | Raw RSS/CDP fetches, denormalized, never modified |
| **Silver** | `news-summarizer` (this) | Cluster articles into stories, LLM-summarize, extract entities, evolve over time |
| **Gold** | _(future)_ | Trends, predictions, decisions |

## What it does

Given the raw news stream from `news-aggregator`, news-summarizer:

1. **Embeds** every article (local BGE-small model, no API cost; deterministic hash fallback for tests)
2. **Clusters** articles into evolving stories by cosine similarity (threshold 0.85)
3. **Summarizes** each article via LLM (minimax-cn default), with extractive fallback on any failure
4. **Extracts entities** (people, places, orgs, dates) with alias canonicalization
5. **Answers questions** across the historical record — *"tell me about Iran nuclear tensions"* returns the full story timeline, all source summaries, and entity graph

The killer feature: it **grows with the news**. Each new article either joins an existing story (refreshing its summary) or starts a new one. Every action is recorded in a `story_events` audit trail so you can see exactly how a story evolved.

## Quick start

```bash
git clone https://github.com/petrel2015/news-summarizer
cd news-summarizer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Initialize the silver DB
news-summarizer stats

# Ingest recent bronze articles (last 72h, max 50, with LLM)
news-summarizer ingest --limit 50 --since-hours 72

# Ask a question
news-summarizer ask "伊朗核问题"

# Compare how a topic is covered
news-summarizer compare "oil prices"

# Show a single story
news-summarizer show-cluster iran-nuclear

# List all known stories
news-summarizer list-clusters
```

## CLI

| Command | Purpose |
|---|---|
| `ingest` | Read bronze articles, embed, cluster, summarize, extract entities |
| `list-clusters` | List all known stories (newest first) |
| `show-cluster <slug>` | Show one story: title, articles, summaries, entities, timeline |
| `ask <question>` | Synthesize an answer across all stored stories (FTS + cluster ranking) |
| `compare <topic>` | Return all matching summaries for cross-source analysis |
| `stats` | DB row counts for all tables |

## Architecture

```
bronze (news-aggregator)
  │
  │  ingest_batch() reads news.db
  ▼
┌─────────────────────────────────────────┐
│  silver.db (SQLite + FTS5 + sqlite-vec) │
│                                         │
│  7 tables:                              │
│    clusters          (stories)          │
│    cluster_articles  (m:n mapping)      │
│    summaries         (per-article LLM)  │
│    entities          (people/places)    │
│    entity_mentions   (per-article)      │
│    embeddings        (vec metadata)     │
│    story_events      (audit trail)      │
│    enrichment_errors (LLM failures)     │
│                                         │
│  Virtual tables:                        │
│    fts_summaries     (FTS5 over LLM)    │
│    vec_embeddings    (sqlite-vec)       │
└─────────────────────────────────────────┘
  │
  │  ask / compare / show
  ▼
agent / cron / dashboard
```

## Reliability

**3-layer defense against LLM outages:**

1. **LLM call** (`summarize_article` / `extract_entities`) — primary path
2. **Extractive fallback** — deterministic sentence-extraction heuristic, no model needed
3. **Error log** — every failure recorded in `enrichment_errors` with `retriable=1` for later retry

The pipeline **never blocks** on LLM failures. A news-aggregator ingest batch always produces clusters + summaries, even when the LLM is down.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NEWS_SUMMARIZER_HOME` | `~/.local/share/news-summarizer` | Where silver.db lives |
| `NEWS_AGGREGATOR_HOME` | `~/.local/share/news-aggregator` | Where to find bronze news.db |
| `SUMMARIZER_LLM_URL` | `http://127.0.0.1:7897/v1/chat/completions` | LLM endpoint (via verge-mihomo) |
| `SUMMARIZER_LLM_MODEL` | `minimax-cn` | Model name |
| `SUMMARIZER_LLM_API_KEY` | _empty_ | Bearer token if required |
| `NEWS_SUMMARIZER_FORCE_HASH` | _unset_ | Force hash-based embeddings (for tests / no-model dev) |

## Testing

```bash
source .venv/bin/activate
pytest            # 58 tests, ~0.7s
```

| Module | Tests | Coverage |
|---|---|---|
| `test_storage.py` | 9 | 7 tables + FTS5 + sqlite-vec + idempotent migrations |
| `test_embed.py` | 6 | BGE-small + hash fallback determinism |
| `test_cluster.py` | 8 | Cosine sim, threshold, join/create branching |
| `test_summarize.py` | 8 | LLM happy path, extractive fallback, error recording |
| `test_entities.py` | 7 | NER happy path, type filtering, alias canonicalization |
| `test_query.py` | 10 | list/show/timeline/FTS/ask/compare |
| `test_cli.py` | 10 | All 6 subcommands (Click runner) |

## Roadmap

| Version | Status | Content |
|---|---|---|
| v0.1 | ✅ shipped | storage + embed + cluster (in-memory, no LLM) |
| v0.2 | ✅ shipped | summarize + entities + ask/compare + CLI |
| v0.3 | 📋 planned | MCP server (`think_about_topic`, `compare_sources`) + HTTP API |
| v0.4 | 📋 planned | Story merge/split/summary-refresh + cron integration |
| v1.0 | 📋 planned | SKILL.md, CI, real bronze ingest pipeline, 70% fulltext rate |

## License

MIT
