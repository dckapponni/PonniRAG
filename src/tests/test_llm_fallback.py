"""
Tests for LLM failure graceful fallback behavior.
Covers: health cache poisoning on transient failure, cache-before-extractive
fallback, fallback_reason in response, history-mode cache fallback,
and consistent thresholds across sync/async paths.
"""
import pytest
import time
from unittest.mock import patch, MagicMock, AsyncMock

import hybrid_search as hs
from llm import (
    _gemini_health_cache,
    _gemini_health_lock,
    _mark_gemini_unhealthy,
    generate_llm_answer,
    generate_llm_answer_async,
)
from cache import _response_cache


def _make_merged_doc(content="தமிழ் மொழி பற்றிய விவரம் " * 30, score=0.95):
    """Helper to create a mock merged document."""
    return {
        "content": content,
        "volume": "vol1",
        "heading": "தமிழ்",
        "doc_id": "doc1",
        "doc_issue": "1",
        "author_name": "",
        "word_count": 100,
        "chunk_count": 2,
        "score": score,
        "tags": [],
    }


def _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge):
    """Common setup for ask_question tests with vector search path."""
    mock_health.return_value = {"healthy": True, "points_count": 1000}
    mock_client.return_value = MagicMock()

    mock_searcher = MagicMock()
    point = MagicMock()
    point.score = 0.95
    point.payload = {
        "type": "article",
        "content": "test",
        "chunk_id": 0,
        "metadata": {
            "doc_id": "doc1",
            "doc_issue": "1",
            "volume": "vol1",
            "title": "தமிழ்",
            "tags": [],
        },
    }
    mock_searcher.search.return_value = [point]
    mock_search_class.return_value = mock_searcher
    mock_merge.return_value = [_make_merged_doc()]


# ============================================================================
# HEALTH CACHE POISONING TESTS
# ============================================================================


class TestHealthCachePoisoning:
    """Test that transient LLM failures poison the Gemini health cache."""

    def setup_method(self):
        """Reset health cache before each test."""
        with _gemini_health_lock:
            _gemini_health_cache["result"] = None
            _gemini_health_cache["timestamp"] = 0

    def test_mark_gemini_unhealthy_updates_cache(self):
        """_mark_gemini_unhealthy should set healthy=False in the cache."""
        _mark_gemini_unhealthy("rate limit exceeded")

        with _gemini_health_lock:
            result = _gemini_health_cache["result"]
        assert result is not None
        assert result["healthy"] is False
        assert result["error"] == "transient_failure"
        assert "rate limit" in result["message"]

    def test_mark_gemini_unhealthy_sets_timestamp(self):
        """_mark_gemini_unhealthy should set a recent timestamp for TTL."""
        before = time.time()
        _mark_gemini_unhealthy("503 error")
        after = time.time()

        with _gemini_health_lock:
            ts = _gemini_health_cache["timestamp"]
        assert before <= ts <= after

    @patch("llm._get_gemini_client")
    def test_generate_llm_answer_poisons_cache_on_retryable_error(self, mock_client):
        """generate_llm_answer should poison health cache on retryable errors."""
        mock_client.return_value.models.generate_content.side_effect = (
            ConnectionError("refused")
        )

        with patch("llm.with_gemini_retry", side_effect=ConnectionError("refused")):
            result = generate_llm_answer("test", "ctx", "csv")

        assert result == ""
        with _gemini_health_lock:
            cached = _gemini_health_cache["result"]
        assert cached is not None
        assert cached["healthy"] is False

    @patch("llm._get_gemini_client")
    def test_generate_llm_answer_no_poison_on_non_retryable(self, mock_client):
        """generate_llm_answer should NOT poison cache on non-retryable errors."""
        # Set cache to healthy first
        with _gemini_health_lock:
            _gemini_health_cache["result"] = {"healthy": True}
            _gemini_health_cache["timestamp"] = time.time()

        with patch("llm.with_gemini_retry", side_effect=ValueError("bad input")):
            result = generate_llm_answer("test", "ctx", "csv")

        assert result == ""
        with _gemini_health_lock:
            cached = _gemini_health_cache["result"]
        # Should still be healthy — ValueError is not retryable
        assert cached["healthy"] is True

    @pytest.mark.asyncio
    async def test_async_generate_poisons_cache_on_retryable(self):
        """generate_llm_answer_async should poison cache on retryable errors."""
        with patch("llm._get_gemini_client") as mock_client:
            with patch(
                "llm.with_gemini_retry_async",
                side_effect=TimeoutError("timeout"),
            ):
                result = await generate_llm_answer_async("test", "ctx", "csv")

        assert result == ""
        with _gemini_health_lock:
            cached = _gemini_health_cache["result"]
        assert cached is not None
        assert cached["healthy"] is False


# ============================================================================
# CACHE-BEFORE-EXTRACTIVE FALLBACK TESTS
# ============================================================================


class TestCacheBeforeExtractiveFallback:
    """Test that cache is tried before extractive when LLM fails."""

    def setup_method(self):
        """Clear caches before each test."""
        _response_cache.clear()
        with _gemini_health_lock:
            _gemini_health_cache["result"] = None
            _gemini_health_cache["timestamp"] = 0

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_fallback_uses_cache_when_available(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """When LLM fails and cache has a previous answer, use cache."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        # Pre-populate cache with a good answer
        cached_answer = "இது ஒரு நீண்ட தமிழ் பதில் " * 20
        _response_cache.put("test question", {"answer": cached_answer, "sources": []})

        # Pass history so the top-level cache check is skipped (it only
        # fires when history is empty).  The LLM call then fails with "",
        # and _llm_fallback_answer finds the cached response.
        history = [{"role": "user", "content": "previous"}]
        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer", return_value=""):
                result = hs.ask_question("test question", use_llm=True, history=history)

        assert result["answer"] == cached_answer
        assert result.get("fallback_reason") == "cached_response"

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_fallback_uses_extractive_when_no_cache(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """When LLM fails and cache is empty, use extractive."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer", return_value=""):
                with patch("hybrid_search.extract_key_facts", return_value=[]):
                    result = hs.ask_question("new question nobody asked", use_llm=True)

        assert result.get("fallback_reason") == "extractive"

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_no_fallback_reason_on_success(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """When LLM succeeds, no fallback_reason in response."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        good_answer = "இது ஒரு நல்ல தமிழ் பதில் " * 20
        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer", return_value=good_answer):
                result = hs.ask_question("test question", use_llm=True)

        assert result["answer"] == good_answer
        assert "fallback_reason" not in result

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_history_mode_cache_fallback(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """In history mode, cache was skipped at top but should be used as fallback."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        # Pre-populate cache
        cached_answer = "இது முன்பு பதிலளிக்கப்பட்ட பதில் " * 20
        _response_cache.put("test question", {"answer": cached_answer, "sources": []})

        history = [
            {"role": "user", "content": "முதல் கேள்வி"},
            {"role": "assistant", "content": "முதல் பதில்"},
        ]

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer", return_value=""):
                result = hs.ask_question(
                    "test question", use_llm=True, history=history
                )

        # Should use cached response even in history mode
        assert result["answer"] == cached_answer
        assert result.get("fallback_reason") == "cached_response"


# ============================================================================
# ASYNC FALLBACK TESTS
# ============================================================================


class TestAsyncFallback:
    """Test async path uses consistent thresholds and fallback."""

    def setup_method(self):
        _response_cache.clear()
        with _gemini_health_lock:
            _gemini_health_cache["result"] = None
            _gemini_health_cache["timestamp"] = 0

    @pytest.mark.asyncio
    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    async def test_async_uses_cache_fallback(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """Async path should try cache before extractive on LLM failure."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        cached_answer = "இது ஒரு நீண்ட தமிழ் பதில் " * 20
        _response_cache.put("test question", {"answer": cached_answer, "sources": []})

        # Pass history to skip top-level cache lookup; LLM fails → _llm_fallback_answer uses cache
        history = [{"role": "user", "content": "previous"}]
        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer_async", new_callable=AsyncMock, return_value=""):
                result = await hs.ask_question_async("test question", use_llm=True, history=history)

        assert result["answer"] == cached_answer
        assert result.get("fallback_reason") == "cached_response"

    @pytest.mark.asyncio
    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    async def test_async_fallback_reason_extractive(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """Async path should set fallback_reason when using extractive."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer_async", new_callable=AsyncMock, return_value=""):
                with patch("hybrid_search.extract_key_facts", return_value=[]):
                    result = await hs.ask_question_async(
                        "never asked question", use_llm=True
                    )

        assert result.get("fallback_reason") == "extractive"

    @pytest.mark.asyncio
    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    async def test_async_threshold_matches_sync(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """Async should use same 150-char threshold as sync (was 100 before fix)."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        # Answer that's 120 chars — would pass old async threshold (100) but not new (150)
        short_answer = "x" * 120

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer_async", new_callable=AsyncMock, return_value=short_answer):
                with patch("hybrid_search.extract_key_facts", return_value=[]):
                    result = await hs.ask_question_async(
                        "threshold test", use_llm=True
                    )

        # Should have fallen back because 120 < 150
        assert result.get("fallback_reason") is not None


# ============================================================================
# STREAMING FALLBACK TESTS
# ============================================================================


class TestStreamingFallback:
    """Test streaming path fallback behavior."""

    def setup_method(self):
        _response_cache.clear()
        with _gemini_health_lock:
            _gemini_health_cache["result"] = None
            _gemini_health_cache["timestamp"] = 0

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_stream_fallback_emits_fallback_event(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """Streaming should emit a fallback event when LLM produces too few tokens."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        def empty_stream(*args, **kwargs):
            return iter([])  # No tokens

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer_stream", side_effect=empty_stream):
                with patch("hybrid_search.extract_key_facts", return_value=[]):
                    events = list(hs.ask_question_stream("stream test"))

        # Should have token, fallback, and sources events
        event_types = [e["type"] for e in events]
        assert "fallback" in event_types
        fallback_event = next(e for e in events if e["type"] == "fallback")
        assert fallback_event["reason"] in ("cached_response", "extractive")

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_stream_fallback_uses_cache(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """Streaming should use cached response when available and LLM fails."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        cached_answer = "இது ஒரு நீண்ட தமிழ் பதில் " * 20
        _response_cache.put("stream test", {"answer": cached_answer, "sources": []})

        def empty_stream(*args, **kwargs):
            return iter([])

        # Pass history to skip top-level cache lookup; empty stream → _llm_fallback_answer uses cache
        history = [{"role": "user", "content": "previous"}]
        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch("hybrid_search.generate_llm_answer_stream", side_effect=empty_stream):
                events = list(hs.ask_question_stream("stream test", history=history))

        # Find the token event with the cached answer
        token_events = [e for e in events if e["type"] == "token"]
        assert any(cached_answer in e["content"] for e in token_events)

        # Should have fallback reason
        fallback_events = [e for e in events if e["type"] == "fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0]["reason"] == "cached_response"


# ============================================================================
# GEMINI UNHEALTHY SHORT-CIRCUIT TESTS
# ============================================================================


class TestGeminiUnhealthyShortCircuit:
    """Test that unhealthy Gemini causes immediate fallback without LLM call."""

    def setup_method(self):
        _response_cache.clear()
        with _gemini_health_lock:
            _gemini_health_cache["result"] = None
            _gemini_health_cache["timestamp"] = 0

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_unhealthy_gemini_skips_llm_call(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """When Gemini is unhealthy, LLM should not be called."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        with patch(
            "hybrid_search.check_gemini_health",
            return_value={"healthy": False, "error": "transient_failure"},
        ):
            with patch("hybrid_search.generate_llm_answer") as mock_llm:
                with patch("hybrid_search.extract_key_facts", return_value=[]):
                    result = hs.ask_question("test", use_llm=True)

        # LLM should NOT have been called
        mock_llm.assert_not_called()
        # Should still get a valid answer (extractive fallback)
        assert result.get("fallback_reason") is not None

    @patch("hybrid_search.check_qdrant_health")
    @patch("hybrid_search.get_qdrant_client")
    @patch("hybrid_search.HybridQdrantSearch")
    @patch("hybrid_search.merge_consecutive_chunks")
    def test_poisoned_cache_causes_short_circuit(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """After health cache poisoning, next request should skip LLM."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        # Poison the health cache (simulating a previous 429 failure)
        _mark_gemini_unhealthy("429 rate limit exceeded")

        with patch("hybrid_search.generate_llm_answer") as mock_llm:
            with patch("hybrid_search.extract_key_facts", return_value=[]):
                result = hs.ask_question("test after rate limit", use_llm=True)

        # LLM should NOT have been called because health cache says unhealthy
        mock_llm.assert_not_called()
        assert result.get("fallback_reason") is not None
