"""Tests for CLI entry points (Click testing)."""
import json
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from news_summarizer import storage
from news_summarizer.cli import cli


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    storage.init_db()
    return db


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "news-summarizer" in result.output


def test_cli_stats_empty(fresh_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "Clusters:" in result.output


def test_cli_list_clusters_empty(fresh_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["list-clusters"])
    assert result.exit_code == 0


def test_cli_list_clusters_with_data(fresh_db):
    storage.add_cluster(slug="a", title="A", summary="")
    storage.add_cluster(slug="b", title="B", summary="")
    runner = CliRunner()
    result = runner.invoke(cli, ["list-clusters"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "b" in result.output


def test_cli_show_cluster_missing(fresh_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["show-cluster", "nonexistent"])
    assert result.exit_code == 2  # not found
    assert "not found" in result.output


def test_cli_show_cluster_json(fresh_db):
    storage.add_cluster(slug="x", title="Topic X", summary="A summary")
    storage.add_article_to_cluster(cluster_slug="x", bronze_article_id=1, relevance=0.9, position_in_timeline=0)
    storage.add_summary(bronze_article_id=1, summary_short="s1", summary_long="l1", key_facts=[], topics=[], model="t")
    runner = CliRunner()
    result = runner.invoke(cli, ["show-cluster", "x", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["slug"] == "x"
    assert data["title"] == "Topic X"
    assert data["article_count"] == 1
    assert "timeline" in data


def test_cli_ask_no_question():
    runner = CliRunner()
    result = runner.invoke(cli, ["ask"])
    assert result.exit_code == 2


def test_cli_ask_with_question(fresh_db):
    storage.add_cluster(slug="iran", title="Iran nuclear tensions", summary="...")
    storage.add_article_to_cluster(cluster_slug="iran", bronze_article_id=1, relevance=0.9, position_in_timeline=0)
    storage.add_summary(bronze_article_id=1, summary_short="Iran nuclear facility strike", summary_long="Long body about nuclear", key_facts=[], topics=[], model="t")
    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "Iran", "nuclear"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["question"] == "Iran nuclear"
    slugs = {c["slug"] for c in data["suggested_clusters"]}
    assert "iran" in slugs


def test_cli_ingest_no_bronze_db(fresh_db, monkeypatch):
    """When bronze DB is missing, ingest reports 0 articles gracefully."""
    from news_summarizer import ingest
    monkeypatch.setattr(ingest, "_get_bronze_db_path", lambda: None)
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--no-summarize", "--no-entities"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 0


def test_cli_compare_with_topic(fresh_db):
    storage.add_summary(bronze_article_id=1, summary_short="Iran attack", summary_long="body", key_facts=[], topics=[], model="t")
    runner = CliRunner()
    result = runner.invoke(cli, ["compare", "Iran"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["topic"] == "Iran"
