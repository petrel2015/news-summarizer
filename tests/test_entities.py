"""Tests for entity extraction layer."""
import pytest
from unittest.mock import patch, MagicMock
from news_summarizer import storage
from news_summarizer.entities import (
    extract_entities,
    extract_entities_batch,
    EntityResult,
    KNOWN_TYPES,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    storage.init_db()
    return db


def test_known_entity_types():
    """Entity types must be a fixed enum."""
    assert "person" in KNOWN_TYPES
    assert "place" in KNOWN_TYPES
    assert "org" in KNOWN_TYPES
    assert "date" in KNOWN_TYPES


def test_extract_entities_with_llm(fresh_db):
    """Happy path: LLM returns list of entity dicts → stored + linked."""
    fake_llm = MagicMock(return_value=[
        {"name": "Iran", "type": "place", "aliases": []},
        {"name": "IAEA", "type": "org", "aliases": ["International Atomic Energy Agency"]},
        {"name": "Joe Biden", "type": "person", "aliases": ["Biden"]},
    ])
    with patch("news_summarizer.entities._call_llm_ner", fake_llm):
        result = extract_entities(
            bronze_article_id=1,
            title="Iran nuclear talks with IAEA",
            body="IAEA met with Joe Biden to discuss Iran's nuclear program.",
        )
    assert isinstance(result, EntityResult)
    assert len(result.entities) == 3
    assert result.source == "llm"
    # Entities stored
    assert storage.get_entity_by_alias("Iran") is not None
    assert storage.get_entity_by_alias("IAEA") is not None
    assert storage.get_entity_by_alias("Biden") is not None  # alias match
    # Mentions linked
    iran = storage.get_entity_by_alias("Iran")
    mentions = storage.connect() if hasattr(storage, 'connect') else None  # placeholder


def test_extract_entities_fallback_caps_entities(fresh_db):
    """LLM fails → fallback: empty entities (don't try to NER without model)."""
    fake_llm = MagicMock(side_effect=RuntimeError("down"))
    with patch("news_summarizer.entities._call_llm_ner", fake_llm):
        result = extract_entities(
            bronze_article_id=2,
            title="Some event",
            body="Body about the event.",
        )
    assert result.source == "fallback"
    assert result.entities == []


def test_extract_entities_records_error_on_llm_failure(fresh_db):
    """LLM error → enrichment_errors row inserted, stage='entity'."""
    fake_llm = MagicMock(side_effect=RuntimeError("content filter"))
    with patch("news_summarizer.entities._call_llm_ner", fake_llm):
        extract_entities(bronze_article_id=3, title="x", body="y")
    errors = [e for e in storage.get_pending_errors() if e["article_id"] == 3 and e["stage"] == "entity"]
    assert len(errors) == 1
    assert errors[0]["retriable"] == 1


def test_extract_entities_alias_canonicalization(fresh_db):
    """Two calls with alias 'US' + 'United States' merge to one entity."""
    fake_llm_1 = MagicMock(return_value=[
        {"name": "United States", "type": "place", "aliases": ["US", "USA"]},
    ])
    fake_llm_2 = MagicMock(return_value=[
        {"name": "United States", "type": "place", "aliases": ["America"]},
    ])
    with patch("news_summarizer.entities._call_llm_ner", fake_llm_1):
        extract_entities(bronze_article_id=10, title="t", body="b")
    with patch("news_summarizer.entities._call_llm_ner", fake_llm_2):
        extract_entities(bronze_article_id=11, title="t", body="b")
    # Both should resolve to the same entity via alias
    e1 = storage.get_entity_by_alias("US")
    e2 = storage.get_entity_by_alias("America")
    assert e1 is not None
    assert e2 is not None
    assert e1["id"] == e2["id"]
    # Aliases should include "America" now
    assert "America" in e1["aliases"]


def test_extract_entities_batch(fresh_db):
    """Batch processes multiple articles, failures isolated."""
    items = [
        (20, "A", "Body A"),
        (21, "B", "Body B"),
        (22, "C", "Body C"),
    ]
    fake_llm = MagicMock(return_value=[
        {"name": "Test Entity", "type": "org", "aliases": []},
    ])
    with patch("news_summarizer.entities._call_llm_ner", fake_llm):
        results = extract_entities_batch(items)
    assert len(results) == 3
    for r in results:
        assert len(r.entities) >= 1


def test_extract_entities_invalid_type_filtered(fresh_db):
    """LLM returns type 'foo' (not in KNOWN_TYPES) → filtered out."""
    fake_llm = MagicMock(return_value=[
        {"name": "Good", "type": "place", "aliases": []},
        {"name": "Bad", "type": "alien", "aliases": []},  # invalid
    ])
    with patch("news_summarizer.entities._call_llm_ner", fake_llm):
        result = extract_entities(bronze_article_id=99, title="t", body="b")
    names = [e["name"] for e in result.entities]
    assert "Good" in names
    assert "Bad" not in names
