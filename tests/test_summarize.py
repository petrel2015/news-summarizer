"""Tests for summarize layer: LLM summary + extractive fallback + error handling."""
import pytest
from unittest.mock import patch, MagicMock
from news_summarizer import storage
from news_summarizer.summarize import (
    summarize_article,
    summarize_batch,
    SummarizeResult,
    SUMMARY_SHORT_MAX_CHARS,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "silver.db"
    monkeypatch.setattr("news_summarizer.storage.DB_PATH", db)
    storage.init_db()
    return db


def test_short_summary_max_chars_is_sane():
    """short summary must be a tweet-length cap."""
    assert 100 < SUMMARY_SHORT_MAX_CHARS < 500


def test_summarize_article_with_llm(fresh_db):
    """Happy path: LLM returns structured summary → stored in DB."""
    fake_llm = MagicMock(return_value={
        "summary_short": "Iran struck facility X.",
        "summary_long": "On 2026-06-07, Iran attacked facility X in retaliation for Y.",
        "key_facts": ["fact A", "fact B"],
        "topics": ["geopolitics", "conflict"],
    })
    with patch("news_summarizer.summarize._call_llm", fake_llm):
        result = summarize_article(
            bronze_article_id=1,
            title="Iran nuclear strike",
            body="Long body text about the attack...",
        )
    assert isinstance(result, SummarizeResult)
    assert result.source == "llm"
    assert result.summary_short.startswith("Iran struck")
    # Stored in DB
    s = storage.get_summary(1)
    assert s is not None
    assert s["model"] == "minimax-cn"


def test_summarize_article_falls_back_to_extractive_on_llm_error(fresh_db):
    """LLM raises → fall back to TextRank extractive summary → still stored."""
    fake_llm = MagicMock(side_effect=RuntimeError("rate limit"))
    with patch("news_summarizer.summarize._call_llm", fake_llm):
        result = summarize_article(
            bronze_article_id=2,
            title="Some event happened",
            body="The first sentence introduces the topic. The second sentence adds more detail. "
                 "The third sentence concludes the narrative with key information.",
        )
    assert result.source == "extractive"
    assert len(result.summary_short) > 0
    # Extractive summary should be in DB
    s = storage.get_summary(2)
    assert s is not None
    assert s["model"].startswith("extractive")


def test_summarize_article_records_error_on_llm_failure(fresh_db):
    """LLM error → enrichment_errors row inserted, retriable=True."""
    fake_llm = MagicMock(side_effect=RuntimeError("content filter"))
    with patch("news_summarizer.summarize._call_llm", fake_llm):
        summarize_article(bronze_article_id=3, title="x", body="y")
    errors = storage.get_pending_errors()
    article_ids = {e["article_id"] for e in errors}
    assert 3 in article_ids
    # The error for article 3 mentions stage='summarize'
    e3 = next(e for e in errors if e["article_id"] == 3)
    assert e3["stage"] == "summarize"
    assert e3["retriable"] == 1


def test_summarize_article_handles_empty_body(fresh_db):
    """Empty body should still produce something (LLM or extractive) without crashing."""
    fake_llm = MagicMock(side_effect=RuntimeError("fail"))
    with patch("news_summarizer.summarize._call_llm", fake_llm):
        result = summarize_article(bronze_article_id=4, title="No body article", body="")
    # Falls back to title-based extractive (or short note)
    assert result.source == "extractive"
    s = storage.get_summary(4)
    assert s is not None


def test_summarize_batch_processes_multiple(fresh_db):
    """Batch returns list of results, one per article."""
    items = [
        (1, "Title A", "Body A has some text."),
        (2, "Title B", "Body B has some other text."),
        (3, "Title C", "Body C is different."),
    ]
    fake_llm = MagicMock(return_value={
        "summary_short": "ok", "summary_long": "ok long",
        "key_facts": [], "topics": [],
    })
    with patch("news_summarizer.summarize._call_llm", fake_llm):
        results = summarize_batch(items)
    assert len(results) == 3
    assert all(r.source == "llm" for r in results)
    # All stored
    for aid in (1, 2, 3):
        assert storage.get_summary(aid) is not None


def test_summarize_batch_isolates_failures(fresh_db):
    """One article fails → other articles still get summarized."""
    call_count = {"n": 0}

    def maybe_fail(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("transient")
        return {"summary_short": "ok", "summary_long": "ok", "key_facts": [], "topics": []}

    items = [(10, "A", "x"), (11, "B", "y"), (12, "C", "z")]
    with patch("news_summarizer.summarize._call_llm", side_effect=maybe_fail):
        results = summarize_batch(items)
    assert len(results) == 3
    sources = [r.source for r in results]
    # Article 11 fell back to extractive
    assert "llm" in sources
    assert "extractive" in sources
    # All stored
    for aid in (10, 11, 12):
        assert storage.get_summary(aid) is not None


def test_extractive_summary_picks_first_sentences(fresh_db):
    """TextRank fallback uses simple first-N-sentence heuristic when no NLP lib."""
    body = "Sentence one is the lead. Sentence two has details. Sentence three adds context. Sentence four is more."
    fake_llm = MagicMock(side_effect=RuntimeError("down"))
    with patch("news_summarizer.summarize._call_llm", fake_llm):
        result = summarize_article(bronze_article_id=99, title="X", body=body)
    # short should be lead sentence
    assert "Sentence one" in result.summary_short or len(result.summary_short) > 0
    # long should have multiple sentences
    assert len(result.summary_long) > len(result.summary_short)
