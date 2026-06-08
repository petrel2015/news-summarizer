"""CLI for news-summarizer.

Subcommands:
  ingest        Read bronze articles from news-aggregator, embed, cluster, summarize
  list-clusters List all known story clusters
  show-cluster  Show one cluster: articles, summaries, entities, timeline
  ask           Ask a question/topic; returns synthesized answer across history
  compare       Compare how a topic is covered (returns all matching summaries)
  stats         Show DB stats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import storage
from .query import (
    ask_topic,
    compare_sources,
    get_cluster_summary,
    get_story_timeline,
    list_clusters,
    search_summaries_fts,
)


def _json_default(o):
    try:
        return o.__dict__
    except Exception:
        return str(o)


def _emit_json(data, **kwargs):
    click.echo(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default, **kwargs))


# ---------- ingest ----------

@click.command("ingest")
@click.option("--limit", default=50, help="Max bronze articles to process in this batch.")
@click.option("--since-hours", default=24, help="Only ingest articles newer than this many hours.")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization (faster, cluster-only).")
@click.option("--no-entities", is_flag=True, help="Skip entity extraction.")
def cmd_ingest(limit: int, since_hours: int, no_summarize: bool, no_entities: bool):
    """Read bronze articles from news-aggregator and ingest into silver."""
    from .ingest import ingest_batch
    storage.init_db()
    result = ingest_batch(
        limit=limit,
        since_hours=since_hours,
        do_summarize=not no_summarize,
        do_entities=not no_entities,
    )
    _emit_json(result)


# ---------- list-clusters ----------

@click.command("list-clusters")
@click.option("--limit", default=50)
def cmd_list_clusters(limit: int):
    """List all known story clusters."""
    storage.init_db()
    clusters = list_clusters(limit=limit)
    # Output as a compact list
    for c in clusters:
        click.echo(f"{c['slug']}\t{c['article_count']}\t{c['title']}")
    click.echo(f"\n({len(clusters)} clusters)", err=True)


# ---------- show-cluster ----------

@click.command("show-cluster")
@click.argument("slug")
@click.option("--timeline/--no-timeline", default=True)
@click.option("--json/--no-json", "as_json", default=False)
def cmd_show_cluster(slug: str, timeline: bool, as_json: bool):
    """Show one cluster's details."""
    storage.init_db()
    c = get_cluster_summary(slug)
    if c is None:
        click.echo(f"Cluster not found: {slug}", err=True)
        sys.exit(2)
    out = dict(c)
    if timeline:
        out["timeline"] = get_story_timeline(slug)
    if as_json:
        _emit_json(out)
        return
    # Human-readable
    click.echo(f"# {c['title']}  ({c['slug']})")
    click.echo(f"Articles: {c['article_count']}")
    if c.get("summary"):
        click.echo(f"Summary: {c['summary']}")
    click.echo()
    for a in c["articles"]:
        click.echo(f"  [{a['bronze_article_id']}] {a['summary_short']}")
        if a.get("key_facts"):
            for f in a["key_facts"][:3]:
                click.echo(f"      - {f}")
    if timeline:
        click.echo("\n## Timeline")
        for ev in out["timeline"]:
            click.echo(f"  [{ev['event_type']}] {ev.get('details', {})}")


# ---------- ask ----------

@click.command("ask")
@click.argument("question", nargs=-1)
@click.option("--limit", default=5)
@click.option("--json/--no-json", "as_json", default=True)
def cmd_ask(question: tuple[str, ...], limit: int, as_json: bool):
    """Ask a question. Returns synthesized answer across stored stories.

    Example: news-summarizer ask "伊朗核问题"
    """
    q = " ".join(question).strip()
    if not q:
        click.echo("Usage: news-summarizer ask <question>", err=True)
        sys.exit(2)
    storage.init_db()
    out = ask_topic(q, limit=limit)
    if as_json:
        _emit_json(out)
    else:
        click.echo(f"# {q}\n")
        if not out["suggested_clusters"]:
            click.echo("No matches in silver DB. Try `news-summarizer ingest` first.")
            return
        for c in out["suggested_clusters"][:limit]:
            click.echo(f"\n## {c['title']}  ({c['slug']}, {c['article_count']} articles)")
            for a in c["articles"][:3]:
                click.echo(f"  - {a['summary_short']}")


# ---------- compare ----------

@click.command("compare")
@click.argument("topic", nargs=-1)
@click.option("--limit", default=50)
@click.option("--json/--no-json", "as_json", default=True)
def cmd_compare(topic: tuple[str, ...], limit: int, as_json: bool):
    """Compare how a topic is covered (all matching summaries)."""
    t = " ".join(topic).strip()
    if not t:
        click.echo("Usage: news-summarizer compare <topic>", err=True)
        sys.exit(2)
    storage.init_db()
    out = compare_sources(t, limit=limit)
    if as_json:
        _emit_json(out)
    else:
        click.echo(f"# {t}\n  ({len(out['summaries'])} summaries)")
        for s in out["summaries"][:10]:
            click.echo(f"  - {s['summary_short']}")


# ---------- stats ----------

@click.command("stats")
def cmd_stats():
    """Show DB statistics."""
    storage.init_db()
    with storage.connect() as con:
        clusters = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        articles = con.execute("SELECT COUNT(*) FROM cluster_articles").fetchone()[0]
        summaries = con.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        entities = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        errors = con.execute("SELECT COUNT(*) FROM enrichment_errors WHERE retriable=1").fetchone()[0]
    click.echo(f"Clusters:           {clusters}")
    click.echo(f"Cluster-articles:   {articles}")
    click.echo(f"Summaries:          {summaries}")
    click.echo(f"Entities:           {entities}")
    click.echo(f"Pending errors:     {errors}")


# ---------- main ----------

@click.group()
def cli():
    """news-summarizer: Silver layer over news-aggregator."""
    pass


cli.add_command(cmd_ingest)
cli.add_command(cmd_list_clusters)
cli.add_command(cmd_show_cluster)
cli.add_command(cmd_ask)
cli.add_command(cmd_compare)
cli.add_command(cmd_stats)


def main():
    cli()


if __name__ == "__main__":
    main()
