"""MCP server (JSON-RPC 2.0 over stdio) for news-summarizer.

Implements the Model Context Protocol wire format directly (no SDK dep) so the
server is testable and lightweight. Tools are exposed for AI agents to call.

Wire protocol:
  - Read JSON-RPC 2.0 requests from stdin (one per line)
  - Write JSON-RPC 2.0 responses to stdout (one per line)
  - Initialize handshake: client sends 'initialize', server returns capabilities
  - Tool discovery: client sends 'tools/list', server returns tool definitions
  - Tool invocation: client sends 'tools/call' with name + arguments

Run via: python -m news_summarizer.server_mcp
Or:     news-summarizer-mcp (console script)
"""
from __future__ import annotations

import json
import sys
from typing import Any

from . import storage
from .query import (
    ask_topic,
    compare_sources,
    get_cluster_summary,
    get_story_timeline,
    list_clusters,
)


# ---------- server metadata ----------

MCP_SERVER_INFO = {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {
        "name": "news-summarizer",
        "version": "0.3.0",
    },
}


# ---------- tool definitions ----------

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "think_about_topic",
        "description": (
            "Ask a question or describe a topic. Returns a synthesized answer that "
            "draws on all stored stories in the silver DB, including FTS-matched "
            "articles, suggested clusters, and top relevant articles. Use this for "
            "'tell me about X' / 'what's happening with Y' / cross-source synthesis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or topic to think about",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of top articles to return (default 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "compare_sources",
        "description": (
            "Compare how a topic is covered. Returns all matching summaries for "
            "cross-source analysis. Use this when you need to see multiple "
            "perspectives on the same event."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to compare",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max summaries to return (default 50)",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_story_timeline",
        "description": (
            "Get the chronological event timeline of a single story (cluster). "
            "Returns story_events: created, article_added, merged, split, "
            "summary_refreshed. Use this to understand how a story evolved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "The cluster slug (e.g. 'iran-nuclear')",
                },
            },
            "required": ["slug"],
        },
    },
    {
        "name": "list_clusters",
        "description": (
            "List all known story clusters, sorted by recency (newest first). "
            "Returns slug, title, article_count, summary_updated_at."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max clusters to return (default 100)",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 500,
                },
            },
        },
    },
    {
        "name": "get_cluster",
        "description": (
            "Get full detail of a single story (cluster) by slug. Returns "
            "title, summary, all articles with their LLM summaries, key facts, "
            "and topics. Returns 404-equivalent (null) if cluster doesn't exist."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "The cluster slug",
                },
            },
            "required": ["slug"],
        },
    },
    {
        "name": "stats",
        "description": "Get DB row counts for all silver tables. Useful for health checks.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------- tool implementations ----------

def _stats() -> dict:
    with storage.connect() as con:
        clusters = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        articles = con.execute("SELECT COUNT(*) FROM cluster_articles").fetchone()[0]
        summaries = con.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        entities = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        errors = con.execute("SELECT COUNT(*) FROM enrichment_errors WHERE retriable=1").fetchone()[0]
    return {
        "clusters": clusters,
        "cluster_articles": articles,
        "summaries": summaries,
        "entities": entities,
        "pending_errors": errors,
    }


def _call_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call. Returns the result data (not the JSON-RPC envelope)."""
    if name == "think_about_topic":
        question = arguments["question"]
        limit = arguments.get("limit", 5)
        return ask_topic(question, limit=limit)
    elif name == "compare_sources":
        topic = arguments["topic"]
        limit = arguments.get("limit", 50)
        return compare_sources(topic, limit=limit)
    elif name == "get_story_timeline":
        slug = arguments["slug"]
        return get_story_timeline(slug)
    elif name == "list_clusters":
        limit = arguments.get("limit", 100)
        return list_clusters(limit=limit)
    elif name == "get_cluster":
        slug = arguments["slug"]
        return get_cluster_summary(slug)
    elif name == "stats":
        return _stats()
    else:
        raise ValueError(f"unknown tool: {name}")


# ---------- JSON-RPC request handler ----------

def _make_result(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _make_error(req_id, code, message, data=None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def handle_request(req: dict) -> dict:
    """Handle a single JSON-RPC 2.0 request and return the response dict.

    Supported methods:
      - initialize → returns server info + capabilities
      - tools/list → returns tool definitions
      - tools/call → invokes a tool, returns result wrapped in MCP content format
    """
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "initialize":
        return _make_result(req_id, MCP_SERVER_INFO)
    elif method == "tools/list":
        return _make_result(req_id, {"tools": TOOL_DEFINITIONS})
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not name:
            return _make_error(req_id, -32602, "missing tool name")
        try:
            data = _call_tool(name, arguments)
            # MCP content format: list of {type, text} where text is JSON string
            content = [{"type": "text", "text": json.dumps(data, ensure_ascii=False, default=str)}]
            return _make_result(req_id, {"content": content})
        except ValueError as e:
            return _make_error(req_id, -32602, str(e))
        except Exception as e:
            return _make_error(req_id, -32603, f"tool execution failed: {type(e).__name__}: {e}")
    else:
        return _make_error(req_id, -32601, f"method not found: {method}")


# ---------- stdio loop ----------

def _stdio_loop() -> None:
    """Read JSON-RPC requests from stdin, write responses to stdout. One per line."""
    storage.init_db()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
        except json.JSONDecodeError as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {e}"}}
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": f"internal error: {e}"}}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main():
    """Run MCP server over stdio. Use: news-summarizer-mcp"""
    _stdio_loop()


if __name__ == "__main__":
    main()
