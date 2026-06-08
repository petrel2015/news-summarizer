"""Summarization layer.

Strategy: LLM-first (minimax-cn) → extractive fallback (sentence-extraction heuristic) →
silent record into enrichment_errors table. Never block the pipeline.

The LLM is called via httpx (we already have httpx in deps). The exact endpoint + model
are configurable via env vars:
  SUMMARIZER_LLM_URL  (default: http://127.0.0.1:7897/v1/chat/completions)
  SUMMARIZER_LLM_MODEL (default: minimax-cn)
  SUMMARIZER_LLM_API_KEY (default: empty)

For tests, _call_llm is monkey-patched.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx
import tenacity

from . import storage


# Max chars for a 'short' summary (tweet-length).
SUMMARY_SHORT_MAX_CHARS = 280
# Max chars for 'long' summary.
SUMMARY_LONG_MAX_CHARS = 1500
# Default LLM config
DEFAULT_LLM_URL = os.environ.get("SUMMARIZER_LLM_URL", "http://127.0.0.1:7897/v1/chat/completions")
DEFAULT_LLM_MODEL = os.environ.get("SUMMARIZER_LLM_MODEL", "minimax-cn")
DEFAULT_LLM_API_KEY = os.environ.get("SUMMARIZER_LLM_API_KEY", "")
# Timeout per LLM call
LLM_TIMEOUT_SECONDS = 30


@dataclass
class SummarizeResult:
    """Result of summarizing a single article."""
    bronze_article_id: int
    summary_short: str
    summary_long: str
    key_facts: list[str]
    topics: list[str]
    source: str  # 'llm' | 'extractive'
    model: str = ""


# ---------- LLM call ----------

SUMMARIZE_PROMPT = """You are a news summarizer. Given an article title and body, produce a JSON object with:
- "summary_short": 1-2 sentences, max {short_max} chars, captures the lead
- "summary_long": 3-5 sentences, max {long_max} chars, captures context + key details
- "key_facts": array of 2-5 short factual claims (each <100 chars)
- "topics": array of 1-4 topical tags (e.g. "geopolitics", "markets", "conflict")

Return ONLY the JSON object, no other text.

Title: {title}
Body: {body}
"""


def _call_llm(title: str, body: str) -> dict:
    """Call the LLM and return parsed JSON dict. Raises on any failure.

    Failure modes: network error, non-2xx, bad JSON, content filter, etc.
    Callers are expected to catch all exceptions and fall back.
    """
    prompt = SUMMARIZE_PROMPT.format(
        short_max=SUMMARY_SHORT_MAX_CHARS,
        long_max=SUMMARY_LONG_MAX_CHARS,
        title=title[:300],
        body=body[:4000],  # hard cap to keep prompt small
    )
    headers = {"Content-Type": "application/json"}
    if DEFAULT_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_LLM_API_KEY}"
    payload = {
        "model": DEFAULT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    with httpx.Client(timeout=LLM_TIMEOUT_SECONDS) as client:
        resp = client.post(DEFAULT_LLM_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    # Extract content
    content = data["choices"][0]["message"]["content"].strip()
    # Try to find JSON block
    if content.startswith("```"):
        # Strip code fence
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    # Validate shape
    for k in ("summary_short", "summary_long", "key_facts", "topics"):
        if k not in parsed:
            raise ValueError(f"LLM response missing field: {k}")
    # Enforce char caps (truncate if over)
    parsed["summary_short"] = parsed["summary_short"][:SUMMARY_SHORT_MAX_CHARS]
    parsed["summary_long"] = parsed["summary_long"][:SUMMARY_LONG_MAX_CHARS]
    if not isinstance(parsed["key_facts"], list):
        parsed["key_facts"] = []
    if not isinstance(parsed["topics"], list):
        parsed["topics"] = []
    return parsed


# ---------- Extractive fallback ----------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extractive_summary(title: str, body: str) -> tuple[str, str, list[str], list[str]]:
    """Produce a deterministic extractive summary when LLM fails.

    Heuristic:
    - short: first sentence (or title if body empty)
    - long: first 3 sentences
    - key_facts: first 3 short sentences (< 200 chars each)
    - topics: empty (we don't have a tagger in fallback mode)
    """
    if not body and not title:
        return ("", "", [], [])
    text = (body or "").strip()
    if not text:
        # body empty → use title for everything
        return (title[:SUMMARY_SHORT_MAX_CHARS], title[:SUMMARY_LONG_MAX_CHARS], [], [])
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return (text[:SUMMARY_SHORT_MAX_CHARS], text[:SUMMARY_LONG_MAX_CHARS], [], [])
    short = sentences[0][:SUMMARY_SHORT_MAX_CHARS]
    long = " ".join(sentences[:3])[:SUMMARY_LONG_MAX_CHARS]
    key_facts = [s[:200] for s in sentences[:3] if 20 < len(s) < 200]
    if not key_facts and sentences:
        key_facts = [sentences[0][:200]]
    return (short, long, key_facts, [])


# ---------- Public API ----------

def summarize_article(
    *,
    bronze_article_id: int,
    title: str,
    body: str,
) -> SummarizeResult:
    """Summarize one article. Tries LLM, falls back to extractive on any error.

    Always returns a result and always stores it in DB.
    """
    try:
        llm_out = _call_llm(title, body)
        result = SummarizeResult(
            bronze_article_id=bronze_article_id,
            summary_short=llm_out["summary_short"],
            summary_long=llm_out["summary_long"],
            key_facts=llm_out["key_facts"],
            topics=llm_out["topics"],
            source="llm",
            model=DEFAULT_LLM_MODEL,
        )
    except Exception as e:
        # Record error (retriable by default — LLM outages are transient)
        try:
            storage.record_error(
                article_id=bronze_article_id,
                stage="summarize",
                error=f"{type(e).__name__}: {e}"[:500],
                retriable=True,
            )
        except Exception:
            pass  # don't let error-logging break the pipeline
        # Fall back
        short, long, facts, topics = _extractive_summary(title, body)
        result = SummarizeResult(
            bronze_article_id=bronze_article_id,
            summary_short=short,
            summary_long=long,
            key_facts=facts,
            topics=topics,
            source="extractive",
            model="extractive-v1",
        )
    # Always store
    storage.add_summary(
        bronze_article_id=bronze_article_id,
        summary_short=result.summary_short,
        summary_long=result.summary_long,
        key_facts=result.key_facts,
        topics=result.topics,
        model=result.model,
    )
    return result


def summarize_batch(items: list[tuple[int, str, str]]) -> list[SummarizeResult]:
    """Summarize multiple articles. Failures in one don't block others.

    items: list of (bronze_article_id, title, body)
    """
    results = []
    for bronze_article_id, title, body in items:
        try:
            r = summarize_article(bronze_article_id=bronze_article_id, title=title, body=body)
            results.append(r)
        except Exception:
            # summarize_article already records errors + falls back; this is double-belt
            results.append(SummarizeResult(
                bronze_article_id=bronze_article_id,
                summary_short=title[:SUMMARY_SHORT_MAX_CHARS],
                summary_long=title[:SUMMARY_LONG_MAX_CHARS],
                key_facts=[],
                topics=[],
                source="extractive",
                model="extractive-v1",
            ))
    return results
