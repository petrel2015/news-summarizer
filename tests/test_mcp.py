"""Tests for MCP server (JSON-RPC over stdio).

We use a minimal JSON-RPC 2.0 implementation rather than the official mcp SDK,
because (a) the SDK pulls many heavy deps, and (b) JSON-RPC over stdio is the
entire MCP wire protocol anyway. This makes the server testable + dependency-free.

Tools exposed:
  - think_about_topic(question, limit=5) -> synthesized answer
  - compare_sources(topic, limit=50)    -> matching summaries
  - get_story_timeline(slug)            -> story events
  - list_clusters(limit=100)            -> cluster list
  - get_cluster(slug)                   -> cluster detail
  - stats()                             -> DB stats
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from news_summarizer import storage
from news_summarizer.server_mcp import (
    handle_request,
    TOOL_DEFINITIONS,
    MCP_SERVER_INFO,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    storage.init_db()
    return db


# ---------- protocol-level tests (no subprocess) ----------


def test_mcp_server_info(fresh_db):
    """handle_request('initialize') returns server info + capabilities."""
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "news-summarizer"
    assert "capabilities" in resp["result"]


def test_mcp_tools_list(fresh_db):
    """handle_request('tools/list') returns tool definitions."""
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = handle_request(req)
    assert "result" in resp
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "think_about_topic" in names
    assert "compare_sources" in names
    assert "get_story_timeline" in names
    assert "list_clusters" in names
    assert "get_cluster" in names
    assert "stats" in names


def test_mcp_call_think_about_topic(fresh_db):
    """Call think_about_topic tool with a question."""
    storage.add_cluster(slug="iran", title="Iran nuclear", summary="...")
    storage.add_article_to_cluster(cluster_slug="iran", bronze_article_id=1, relevance=0.9, position_in_timeline=0)
    storage.add_summary(bronze_article_id=1, summary_short="Iran nuclear facility strike", summary_long="body about nuclear", key_facts=[], topics=[], model="t")
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "think_about_topic", "arguments": {"question": "Iran nuclear", "limit": 5}},
    }
    resp = handle_request(req)
    assert "result" in resp
    content = resp["result"]["content"]
    assert len(content) >= 1
    # content is a list of {type, text} — text is JSON string
    text = next(c["text"] for c in content if c["type"] == "text")
    data = json.loads(text)
    assert data["question"] == "Iran nuclear"
    slugs = {c["slug"] for c in data["suggested_clusters"]}
    assert "iran" in slugs


def test_mcp_call_compare_sources(fresh_db):
    """Call compare_sources tool."""
    storage.add_summary(bronze_article_id=1, summary_short="Iran attack", summary_long="body", key_facts=[], topics=[], model="t")
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "compare_sources", "arguments": {"topic": "Iran", "limit": 10}},
    }
    resp = handle_request(req)
    assert "result" in resp
    text = next(c["text"] for c in resp["result"]["content"] if c["type"] == "text")
    data = json.loads(text)
    assert data["topic"] == "Iran"


def test_mcp_call_get_story_timeline(fresh_db):
    """Call get_story_timeline tool."""
    storage.add_cluster(slug="x", title="X", summary="")
    storage.add_article_to_cluster(cluster_slug="x", bronze_article_id=1, relevance=0.9, position_in_timeline=0)
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "get_story_timeline", "arguments": {"slug": "x"}},
    }
    resp = handle_request(req)
    text = next(c["text"] for c in resp["result"]["content"] if c["type"] == "text")
    data = json.loads(text)
    assert isinstance(data, list)
    assert any(e["event_type"] == "created" for e in data)


def test_mcp_call_list_clusters(fresh_db):
    """Call list_clusters tool."""
    storage.add_cluster(slug="a", title="A", summary="")
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "list_clusters", "arguments": {"limit": 10}},
    }
    resp = handle_request(req)
    text = next(c["text"] for c in resp["result"]["content"] if c["type"] == "text")
    data = json.loads(text)
    slugs = {c["slug"] for c in data}
    assert "a" in slugs


def test_mcp_call_get_cluster(fresh_db):
    """Call get_cluster tool."""
    storage.add_cluster(slug="b", title="B Topic", summary="A summary")
    storage.add_article_to_cluster(cluster_slug="b", bronze_article_id=1, relevance=0.9, position_in_timeline=0)
    storage.add_summary(bronze_article_id=1, summary_short="s1", summary_long="l1", key_facts=[], topics=[], model="t")
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "get_cluster", "arguments": {"slug": "b"}},
    }
    resp = handle_request(req)
    text = next(c["text"] for c in resp["result"]["content"] if c["type"] == "text")
    data = json.loads(text)
    assert data["slug"] == "b"
    assert data["article_count"] == 1


def test_mcp_call_stats(fresh_db):
    """Call stats tool."""
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "stats", "arguments": {}},
    }
    resp = handle_request(req)
    text = next(c["text"] for c in resp["result"]["content"] if c["type"] == "text")
    data = json.loads(text)
    assert "clusters" in data
    assert "summaries" in data


def test_mcp_unknown_method_returns_error(fresh_db):
    """Unknown method → JSON-RPC error -32601 (Method not found)."""
    req = {"jsonrpc": "2.0", "id": 99, "method": "foo/bar", "params": {}}
    resp = handle_request(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_mcp_unknown_tool_returns_error(fresh_db):
    """Unknown tool name → error in tool result."""
    req = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": "no_such_tool", "arguments": {}},
    }
    resp = handle_request(req)
    assert "error" in resp
    assert "no_such_tool" in str(resp["error"])


def test_mcp_tool_definitions_have_required_fields(fresh_db):
    """Every tool definition has name, description, inputSchema."""
    for t in TOOL_DEFINITIONS:
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_mcp_server_info_required_fields():
    """MCP_SERVER_INFO has protocolVersion, capabilities, serverInfo."""
    assert "protocolVersion" in MCP_SERVER_INFO
    assert "capabilities" in MCP_SERVER_INFO
    assert "serverInfo" in MCP_SERVER_INFO
    assert MCP_SERVER_INFO["serverInfo"]["name"] == "news-summarizer"
