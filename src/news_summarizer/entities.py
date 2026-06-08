"""Entity extraction layer.

Same LLM-first + fallback pattern as summarize.py. Returns people/places/orgs/dates
extracted from an article, with alias canonicalization (e.g. "US" / "USA" / "America"
all merge into one entity).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

from . import storage


# Allowed entity types
KNOWN_TYPES = {"person", "place", "org", "date", "other"}

# Default LLM config (same env vars as summarize, but separate function name for mocking)
DEFAULT_LLM_URL = os.environ.get("ENTITY_LLM_URL", os.environ.get("SUMMARIZER_LLM_URL", "http://127.0.0.1:7897/v1/chat/completions"))
DEFAULT_LLM_MODEL = os.environ.get("ENTITY_LLM_MODEL", os.environ.get("SUMMARIZER_LLM_MODEL", "minimax-cn"))
DEFAULT_LLM_API_KEY = os.environ.get("ENTITY_LLM_API_KEY", os.environ.get("SUMMARIZER_LLM_API_KEY", ""))
LLM_TIMEOUT_SECONDS = 30


@dataclass
class EntityResult:
    bronze_article_id: int
    entities: list[dict]  # [{name, type, aliases, entity_id}]
    source: str  # 'llm' | 'fallback'


NER_PROMPT = """Extract named entities from this news article. Return ONLY a JSON array of objects with:
- "name": the canonical entity name (string)
- "type": one of {types}
- "aliases": array of alternate names/strings that refer to the same entity (can be empty)

Limit to 10 most important entities.

Title: {title}
Body: {body}
"""


def _call_llm_ner(title: str, body: str) -> list[dict]:
    """Call LLM for NER. Raises on any failure."""
    prompt = NER_PROMPT.format(
        types=", ".join(sorted(KNOWN_TYPES)),
        title=title[:300],
        body=body[:4000],
    )
    headers = {"Content-Type": "application/json"}
    if DEFAULT_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_LLM_API_KEY}"
    payload = {
        "model": DEFAULT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 600,
    }
    with httpx.Client(timeout=LLM_TIMEOUT_SECONDS) as client:
        resp = client.post(DEFAULT_LLM_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("LLM NER response is not a list")
    return parsed


def _filter_and_normalize(entities: list[dict]) -> list[dict]:
    """Drop entities with invalid types (not in KNOWN_TYPES), coerce shape, dedupe by name within the same call."""
    seen = set()
    out = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").strip()
        if not name or len(name) > 200:
            continue
        etype = (e.get("type") or "").strip().lower()
        # Strict: drop if type is missing or not in KNOWN_TYPES
        if etype not in KNOWN_TYPES:
            continue
        aliases = e.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        aliases = [str(a).strip() for a in aliases if a and str(a).strip() and str(a).strip() != name]
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "type": etype, "aliases": aliases})
    return out


def extract_entities(
    *,
    bronze_article_id: int,
    title: str,
    body: str,
) -> EntityResult:
    """Extract entities from one article. LLM-first, fallback to empty on error.

    Always returns a result; entity extraction failures are recorded in enrichment_errors.
    """
    try:
        raw = _call_llm_ner(title, body)
        normalized = _filter_and_normalize(raw)
        # Upsert each entity (handles alias canonicalization)
        for e in normalized:
            eid = storage.upsert_entity(name=e["name"], type=e["type"], aliases=e["aliases"])
            e["entity_id"] = eid
        result = EntityResult(
            bronze_article_id=bronze_article_id,
            entities=normalized,
            source="llm",
        )
    except Exception as e:
        try:
            storage.record_error(
                article_id=bronze_article_id,
                stage="entity",
                error=f"{type(e).__name__}: {e}"[:500],
                retriable=True,
            )
        except Exception:
            pass
        result = EntityResult(
            bronze_article_id=bronze_article_id,
            entities=[],
            source="fallback",
        )
    return result


def extract_entities_batch(items: list[tuple[int, str, str]]) -> list[EntityResult]:
    """Batch process. Isolates failures per article."""
    results = []
    for bronze_article_id, title, body in items:
        try:
            r = extract_entities(bronze_article_id=bronze_article_id, title=title, body=body)
            results.append(r)
        except Exception:
            results.append(EntityResult(bronze_article_id=bronze_article_id, entities=[], source="fallback"))
    return results
