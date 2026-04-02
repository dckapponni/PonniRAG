"""Test LLM failure graceful fallback behavior.

Cover health cache poisoning on transient failure,
cache-before-extractive fallback, fallback_reason in response,
history-mode cache fallback, and consistent thresholds across
sync/async paths.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import hybrid_search as hs
import pytest
from cache import _response_cache
from llm import (
    _gemini_health_cache,
    _gemini_health_lock,
    _mark_gemini_unhealthy,
    generate_llm_answer,
    generate_llm_answer_async,
)


def _make_merged_doc(content="தமிழ் மொழி பற்றிய விவரம் " * 30, score=0.95):
    """Create a mock merged document."""
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
    """Set up ask_question tests with vector search path."""
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
        mock_client.return_value.models.generate_content.side_effect = ConnectionError(
            "refused"
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
        with patch("llm._get_gemini_client"):
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

        # Pass history so top-level cache check is skipped (it only
        # fires when history is empty). The LLM call then fails
        # with "", and _llm_fallback_answer finds cached response.
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
                result = hs.ask_question("test question", use_llm=True, history=history)

        # Should use cached response even in history mode
        assert result["answer"] == cached_answer
        assert result.get("fallback_reason") == "cached_response"


# ============================================================================
# ASYNC FALLBACK TESTS
# ============================================================================


class TestAsyncFallback:
    """Test async path uses consistent thresholds and fallback."""

    def setup_method(self):
        """Clear caches before each test."""
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

        # Pass history to skip top-level cache lookup;
        # LLM fails, _llm_fallback_answer uses cache
        history = [{"role": "user", "content": "previous"}]
        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch(
                "hybrid_search.generate_llm_answer_async",
                new_callable=AsyncMock,
                return_value="",
            ):
                result = await hs.ask_question_async(
                    "test question", use_llm=True, history=history
                )

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
            with patch(
                "hybrid_search.generate_llm_answer_async",
                new_callable=AsyncMock,
                return_value="",
            ):
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

        # 120 chars: passes old async threshold (100) but not new (150)
        short_answer = "x" * 120

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch(
                "hybrid_search.generate_llm_answer_async",
                new_callable=AsyncMock,
                return_value=short_answer,
            ):
                with patch("hybrid_search.extract_key_facts", return_value=[]):
                    result = await hs.ask_question_async("threshold test", use_llm=True)

        # Should have fallen back because 120 < 150
        assert result.get("fallback_reason") is not None


# ============================================================================
# STREAMING FALLBACK TESTS
# ============================================================================


class TestStreamingFallback:
    """Test streaming path fallback behavior."""

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
    def test_stream_fallback_emits_fallback_event(
        self, mock_merge, mock_search_class, mock_client, mock_health
    ):
        """Streaming should emit a fallback event when LLM produces too few tokens."""
        _mock_search_setup(mock_health, mock_client, mock_search_class, mock_merge)

        def empty_stream(*args, **kwargs):
            """Return an empty iterator."""
            return iter([])  # No tokens

        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch(
                "hybrid_search.generate_llm_answer_stream", side_effect=empty_stream
            ):
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
            """Return an empty iterator."""
            return iter([])

        # Pass history to skip top-level cache lookup;
        # empty stream, _llm_fallback_answer uses cache
        history = [{"role": "user", "content": "previous"}]
        with patch("hybrid_search.check_gemini_health", return_value={"healthy": True}):
            with patch(
                "hybrid_search.generate_llm_answer_stream", side_effect=empty_stream
            ):
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
        """Clear caches before each test."""
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


@patch("db.llm.with_gemini_retry")
def test_llm_fallback(mock_retry):
    """Test LLM fallback on retry failure."""
    mock_retry.side_effect = Exception("fail")

    from db.llm import generate_llm_answer

    result = generate_llm_answer("test", "context", "")

    assert result == ""


def test_llm_empty_sources():
    """Test LLM with empty sources."""
    from db.llm import generate_llm_answer

    result = generate_llm_answer("test", "", "")

    assert isinstance(result, str)


@patch("db.llm.with_gemini_retry")
def test_llm_success(mock_retry):
    """Test successful LLM answer generation."""
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.candidates = []

    mock_retry.return_value = mock_response

    from db.llm import generate_llm_answer

    result = generate_llm_answer("test", "context", "")

    assert isinstance(result, str)


@patch("db.llm.with_gemini_retry")
def test_llm_stream(mock_retry):
    """Test LLM streaming generation."""
    mock_chunk = MagicMock()
    mock_chunk.text = "hello"

    mock_retry.return_value = [mock_chunk]

    from db.llm import generate_llm_answer_stream

    result = list(generate_llm_answer_stream("q", "c", ""))

    assert isinstance(result, list)


# ============================================================================
# TRUNCATE AT SENTENCE BOUNDARY TESTS (lines 245-261)
# ============================================================================


class TestTruncateAtSentenceBoundary:
    """Tests for _truncate_at_sentence_boundary helper."""

    def test_empty_string_returns_empty(self):
        """Return empty string unchanged."""
        from llm import _truncate_at_sentence_boundary

        assert _truncate_at_sentence_boundary("") == ""

    def test_none_returns_none(self):
        """Return None unchanged."""
        from llm import _truncate_at_sentence_boundary

        assert _truncate_at_sentence_boundary(None) is None

    def test_complete_sentence_unchanged(self):
        """Return text ending with period unchanged."""
        from llm import _truncate_at_sentence_boundary

        text = "This is complete."
        assert _truncate_at_sentence_boundary(text) == text

    def test_question_mark_ending(self):
        """Return text ending with question mark unchanged."""
        from llm import _truncate_at_sentence_boundary

        text = "Is this complete?"
        assert _truncate_at_sentence_boundary(text) == text

    def test_exclamation_ending(self):
        """Return text ending with exclamation unchanged."""
        from llm import _truncate_at_sentence_boundary

        assert _truncate_at_sentence_boundary("Done!") == "Done!"

    def test_tamil_danda_ending(self):
        """Return text ending with Tamil danda unchanged."""
        from llm import _truncate_at_sentence_boundary

        assert _truncate_at_sentence_boundary("text\u0964") == "text\u0964"

    def test_truncates_at_last_period(self):
        """Truncate at last sentence boundary when text is incomplete."""
        from llm import _truncate_at_sentence_boundary

        text = "First sentence. Second sentence. Incomplete"
        result = _truncate_at_sentence_boundary(text)
        assert result.endswith(".")
        assert "Incomplete" not in result

    def test_returns_stripped_when_no_boundary_found(self):
        """Return stripped text when no sentence boundary past midpoint."""
        from llm import _truncate_at_sentence_boundary

        text = "no boundaries here at all"
        result = _truncate_at_sentence_boundary(text)
        assert result == text.rstrip()

    def test_trailing_whitespace_stripped(self):
        """Strip trailing whitespace before checking boundary."""
        from llm import _truncate_at_sentence_boundary

        text = "Complete sentence.   "
        assert _truncate_at_sentence_boundary(text) == "Complete sentence."


# ============================================================================
# GET GEMINI CLIENT TESTS (lines 270-271)
# ============================================================================


class TestGetGeminiClient:
    """Tests for _get_gemini_client singleton."""

    def test_raises_when_no_api_key(self):
        """Raise ValueError when GEMINI_API_KEY is not set."""
        from llm import _get_gemini_client

        with patch("llm.GEMINI_API_KEY", None):
            with patch("llm._gemini_client", None):
                with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                    _get_gemini_client()

    @patch("llm.genai")
    def test_creates_client_with_api_key(self, mock_genai):
        """Create a Gemini client when API key is set."""
        from llm import _get_gemini_client

        mock_genai.Client.return_value = MagicMock()
        with patch("llm.GEMINI_API_KEY", "test-key"):
            with patch("llm._gemini_client", None):
                client = _get_gemini_client()
        mock_genai.Client.assert_called_once_with(api_key="test-key")
        assert client is not None


# ============================================================================
# GENERATION CONFIG TESTS (line 371)
# ============================================================================


class TestGeminiGenerationConfig:
    """Tests for _gemini_generation_config helper."""

    def test_disable_thinking_sets_budget_zero(self):
        """Set thinking_budget=0 when disable_thinking is True."""
        from llm import _gemini_generation_config

        config = _gemini_generation_config(
            system_instruction="test", disable_thinking=True
        )
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 0

    def test_default_thinking_is_none(self):
        """Leave thinking_config None by default."""
        from llm import _gemini_generation_config

        config = _gemini_generation_config(system_instruction="test")
        assert config.thinking_config is None

    def test_custom_max_output_tokens(self):
        """Pass custom max_output_tokens to config."""
        from llm import _gemini_generation_config

        config = _gemini_generation_config(
            system_instruction="test", max_output_tokens=8192
        )
        assert config.max_output_tokens == 8192

    def test_default_max_output_tokens(self):
        """Default max_output_tokens is 4096."""
        from llm import _gemini_generation_config

        config = _gemini_generation_config(system_instruction="test")
        assert config.max_output_tokens == 4096


# ============================================================================
# SYSTEM PROMPT SELECTION TESTS (lines 412, 419)
# ============================================================================


class TestSystemPromptSelection:
    """Tests for language-based prompt selection."""

    def test_get_system_prompt_english(self):
        """Return English prompt when language is en."""
        from llm import ENGLISH_ANSWER_SYSTEM_PROMPT, _get_system_prompt

        assert _get_system_prompt("en") == ENGLISH_ANSWER_SYSTEM_PROMPT

    def test_get_system_prompt_tamil_default(self):
        """Return Tamil prompt by default."""
        from llm import TAMIL_ANSWER_SYSTEM_PROMPT, _get_system_prompt

        assert _get_system_prompt("ta") == TAMIL_ANSWER_SYSTEM_PROMPT

    def test_get_csv_system_prompt_english(self):
        """Return English CSV prompt when language is en."""
        from llm import _CSV_SYSTEM_PROMPT_EN, _get_csv_system_prompt

        result = _get_csv_system_prompt("en")
        assert result == _CSV_SYSTEM_PROMPT_EN

    def test_get_csv_system_prompt_tamil_default(self):
        """Return Tamil CSV prompt by default."""
        from llm import _CSV_SYSTEM_PROMPT, _get_csv_system_prompt

        assert _get_csv_system_prompt("ta") == _CSV_SYSTEM_PROMPT


# ============================================================================
# BUILD USER CONTENT TESTS (lines 450-487)
# ============================================================================


class TestBuildUserContent:
    """Tests for _build_user_content prompt builder."""

    def test_omits_empty_context(self):
        """Omit document context section when context is empty."""
        from llm import _build_user_content

        result = _build_user_content("question", "", "csv data")
        assert "CSV" in result
        assert "Document Context" not in result
        assert "ஆவண சூழல்" not in result

    def test_omits_empty_csv(self):
        """Omit CSV section when csv_context is empty."""
        from llm import _build_user_content

        result = _build_user_content("question", "doc ctx", "")
        assert "CSV" not in result

    def test_english_wh_question_closing(self):
        """Use English wh-question closing instruction."""
        from llm import _build_user_content

        result = _build_user_content("who wrote this?", "ctx", "csv", language="en")
        assert "direct answer clearly" in result

    def test_tamil_wh_question_closing(self):
        """Use Tamil wh-question closing instruction."""
        from llm import _build_user_content

        result = _build_user_content("யார் எழுதினார்?", "ctx", "csv", language="ta")
        assert "நேரடியான பதிலை" in result

    def test_english_non_wh_closing(self):
        """Use English detailed answer closing for non-wh questions."""
        from llm import _build_user_content

        result = _build_user_content("describe ponni", "ctx", "csv", language="en")
        assert "Detailed answer" in result

    def test_multi_doc_english_closing(self):
        """Add multi-document instruction in English."""
        from llm import _build_user_content

        result = _build_user_content(
            "who wrote?",
            "ctx",
            "csv",
            context_doc_count=3,
            language="en",
        )
        assert "3 documents" in result
        assert "Integrate" in result

    def test_multi_doc_tamil_closing(self):
        """Add multi-document instruction in Tamil."""
        from llm import _build_user_content

        result = _build_user_content(
            "யார்?",
            "ctx",
            "csv",
            context_doc_count=2,
            language="ta",
        )
        assert "2 ஆவணங்கள்" in result

    def test_english_labels(self):
        """Use English labels when language is en."""
        from llm import _build_user_content

        result = _build_user_content("test", "doc", "csv", language="en")
        assert "Question:" in result
        assert "Document Context:" in result


# ============================================================================
# BUILD CSV USER CONTENT TESTS (lines 533-561)
# ============================================================================


class TestBuildCsvUserContent:
    """Tests for _build_csv_user_content prompt builder."""

    def test_yes_no_english(self):
        """Use English yes/no closing for yes/no question."""
        from llm import _build_csv_user_content

        # Use Tamil yes/no pattern with English output
        result = _build_csv_user_content("எழுதியுள்ளாரா ponni?", "data", language="en")
        assert "Yes" in result and "No" in result

    def test_yes_no_tamil(self):
        """Use Tamil yes/no closing for Tamil yes/no question."""
        from llm import _build_csv_user_content

        result = _build_csv_user_content("எழுதியுள்ளாரா?", "data", language="ta")
        assert "ஆம்" in result

    def test_wh_english(self):
        """Use English wh closing for wh question."""
        from llm import _build_csv_user_content

        result = _build_csv_user_content("who is the author?", "data", language="en")
        assert "direct answer" in result

    def test_wh_tamil(self):
        """Use Tamil wh closing for Tamil wh question."""
        from llm import _build_csv_user_content

        result = _build_csv_user_content("யார் எழுதினார்?", "data", language="ta")
        assert "நேரடியான பதிலை" in result

    def test_default_english(self):
        """Use English summary closing for generic question."""
        from llm import _build_csv_user_content

        result = _build_csv_user_content("tell me about ponni", "data", language="en")
        assert "Summarize" in result

    def test_default_tamil(self):
        """Use Tamil summary closing for generic Tamil question."""
        from llm import _build_csv_user_content

        result = _build_csv_user_content("பொன்னி பற்றி கூறுக", "data", language="ta")
        assert "சுருக்கமாக" in result

    def test_contains_data_block(self):
        """Include the CSV data in the output."""
        from llm import _build_csv_user_content

        result = _build_csv_user_content("test", "my csv data here", language="ta")
        assert "my csv data here" in result


# ============================================================================
# BUILD MULTI-TURN CONTENTS TESTS (lines 586-591)
# ============================================================================


class TestBuildMultiTurnContents:
    """Tests for _build_multi_turn_contents helper."""

    def test_maps_assistant_to_model_role(self):
        """Map assistant role to Gemini model role."""
        from llm import _build_multi_turn_contents

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _build_multi_turn_contents(history, "new q")
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"
        assert result[2]["role"] == "user"
        assert result[2]["parts"][0]["text"] == "new q"

    def test_empty_history(self):
        """Handle empty history with just current content."""
        from llm import _build_multi_turn_contents

        result = _build_multi_turn_contents([], "current")
        assert len(result) == 1
        assert result[0]["parts"][0]["text"] == "current"


# ============================================================================
# GENERATE LLM ANSWER WITH HISTORY / OVERRIDES (lines 613-651)
# ============================================================================


class TestGenerateLlmAnswerOverrides:
    """Tests for generate_llm_answer with overrides and history."""

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_uses_custom_system_prompt(self, mock_client, mock_retry):
        """Use provided system_prompt instead of default."""
        mock_resp = MagicMock()
        mock_resp.text = "answer"
        mock_resp.candidates = []
        mock_retry.return_value = mock_resp

        result = generate_llm_answer("q", "ctx", "", system_prompt="custom prompt")
        assert result != ""
        # Verify config was called with custom prompt
        call_kwargs = mock_retry.call_args
        config = call_kwargs.kwargs["config"]
        assert "custom prompt" in config.system_instruction

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_uses_custom_user_content(self, mock_client, mock_retry):
        """Use provided user_content instead of building it."""
        mock_resp = MagicMock()
        mock_resp.text = "answer"
        mock_resp.candidates = []
        mock_retry.return_value = mock_resp

        result = generate_llm_answer("q", "ctx", "", user_content="my custom content")
        assert result != ""
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["contents"] == "my custom content"

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_uses_history_for_multi_turn(self, mock_client, mock_retry):
        """Build multi-turn contents when history is provided."""
        mock_resp = MagicMock()
        mock_resp.text = "answer"
        mock_resp.candidates = []
        mock_retry.return_value = mock_resp

        history = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        result = generate_llm_answer("new q", "ctx", "csv", history=history)
        assert result != ""
        call_kwargs = mock_retry.call_args
        contents = call_kwargs.kwargs["contents"]
        assert isinstance(contents, list)
        assert len(contents) == 3

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_english_language_selects_english_prompt(self, mock_client, mock_retry):
        """Select English system prompt when language=en."""
        mock_resp = MagicMock()
        mock_resp.text = "answer"
        mock_resp.candidates = []
        mock_retry.return_value = mock_resp

        generate_llm_answer("q", "ctx", "", language="en")
        call_kwargs = mock_retry.call_args
        config = call_kwargs.kwargs["config"]
        assert "English" in config.system_instruction

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_max_tokens_truncation(self, mock_client, mock_retry):
        """Truncate at sentence boundary when MAX_TOKENS reached."""
        mock_resp = MagicMock()
        mock_resp.text = "First sentence. Second sentence. Incompl"
        candidate = MagicMock()
        candidate.finish_reason = "MAX_TOKENS"
        mock_resp.candidates = [candidate]
        mock_retry.return_value = mock_resp

        result = generate_llm_answer("q", "ctx", "")
        assert result.endswith(".")
        assert "Incompl" not in result


# ============================================================================
# ASYNC GENERATE WITH HISTORY / MAX_TOKENS (lines 685-722)
# ============================================================================


class TestGenerateLlmAnswerAsyncOverrides:
    """Tests for generate_llm_answer_async with overrides."""

    @pytest.mark.asyncio
    async def test_async_uses_history(self):
        """Build multi-turn contents in async path."""
        mock_resp = MagicMock()
        mock_resp.text = "async answer"
        mock_resp.candidates = []

        with patch("llm._get_gemini_client"):
            with patch(
                "llm.with_gemini_retry_async",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ) as mock_retry:
                history = [
                    {"role": "user", "content": "prev"},
                    {"role": "assistant", "content": "resp"},
                ]
                result = await generate_llm_answer_async(
                    "q", "ctx", "csv", history=history
                )
        assert result != ""
        call_kwargs = mock_retry.call_args
        contents = call_kwargs.kwargs["contents"]
        assert isinstance(contents, list)

    @pytest.mark.asyncio
    async def test_async_english_prompt(self):
        """Select English prompt in async path."""
        mock_resp = MagicMock()
        mock_resp.text = "english answer"
        mock_resp.candidates = []

        with patch("llm._get_gemini_client"):
            with patch(
                "llm.with_gemini_retry_async",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ) as mock_retry:
                await generate_llm_answer_async("q", "ctx", "", language="en")
        call_kwargs = mock_retry.call_args
        config = call_kwargs.kwargs["config"]
        assert "English" in config.system_instruction

    @pytest.mark.asyncio
    async def test_async_max_tokens_truncation(self):
        """Truncate at sentence boundary in async on MAX_TOKENS."""
        mock_resp = MagicMock()
        mock_resp.text = "Done. Partial"
        candidate = MagicMock()
        candidate.finish_reason = "MAX_TOKENS"
        mock_resp.candidates = [candidate]

        with patch("llm._get_gemini_client"):
            with patch(
                "llm.with_gemini_retry_async",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ):
                result = await generate_llm_answer_async("q", "ctx", "")
        assert result.endswith(".")
        assert "Partial" not in result


# ============================================================================
# STREAMING FULL PATH TESTS (lines 758-791)
# ============================================================================


class TestStreamingFullPath:
    """Tests for generate_llm_answer_stream full coverage."""

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_stream_with_history(self, mock_client, mock_retry):
        """Build multi-turn contents in streaming path."""
        from llm import generate_llm_answer_stream

        chunk1 = MagicMock()
        chunk1.text = "hello "
        chunk2 = MagicMock()
        chunk2.text = "world"
        mock_retry.return_value = [chunk1, chunk2]

        history = [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "resp"},
        ]
        tokens = list(generate_llm_answer_stream("q", "ctx", "csv", history=history))
        assert tokens == ["hello ", "world"]
        call_kwargs = mock_retry.call_args
        contents = call_kwargs.kwargs["contents"]
        assert isinstance(contents, list)

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_stream_with_english_prompt(self, mock_client, mock_retry):
        """Use English system prompt in streaming path."""
        from llm import generate_llm_answer_stream

        mock_retry.return_value = []
        list(generate_llm_answer_stream("q", "ctx", "", language="en"))
        call_kwargs = mock_retry.call_args
        config = call_kwargs.kwargs["config"]
        assert "English" in config.system_instruction

    @patch("llm.with_gemini_retry")
    @patch("llm._get_gemini_client")
    def test_stream_with_custom_overrides(self, mock_client, mock_retry):
        """Use custom user_content and system_prompt in stream."""
        from llm import generate_llm_answer_stream

        chunk = MagicMock()
        chunk.text = "token"
        mock_retry.return_value = [chunk]

        tokens = list(
            generate_llm_answer_stream(
                "q",
                "ctx",
                "",
                user_content="custom",
                system_prompt="sys",
            )
        )
        assert tokens == ["token"]
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs["contents"] == "custom"

    @patch("llm._get_gemini_client")
    def test_stream_skips_none_tokens(self, mock_client):
        """Skip chunks with None text in streaming."""
        from llm import generate_llm_answer_stream

        chunk_none = MagicMock()
        chunk_none.text = None
        chunk_ok = MagicMock()
        chunk_ok.text = "ok"

        with patch(
            "llm.with_gemini_retry",
            return_value=[chunk_none, chunk_ok],
        ):
            tokens = list(generate_llm_answer_stream("q", "ctx", ""))
        assert tokens == ["ok"]

    @patch("llm._get_gemini_client")
    def test_stream_error_poisons_cache_on_retryable(self, mock_client):
        """Poison health cache when streaming hits retryable error."""
        from llm import generate_llm_answer_stream

        with _gemini_health_lock:
            _gemini_health_cache["result"] = None
            _gemini_health_cache["timestamp"] = 0

        with patch(
            "llm.with_gemini_retry",
            side_effect=ConnectionError("refused"),
        ):
            tokens = list(generate_llm_answer_stream("q", "ctx", ""))
        assert tokens == []
        with _gemini_health_lock:
            cached = _gemini_health_cache["result"]
        assert cached is not None
        assert cached["healthy"] is False


# ============================================================================
# VALIDATE GEMINI API TESTS (lines 824-832)
# ============================================================================


class TestValidateGeminiApi:
    """Tests for validate_gemini_api startup check."""

    @patch("llm.GEMINI_API_KEY", None)
    def test_warns_when_no_api_key(self, caplog):
        """Warn and return when API key is not set."""
        import logging

        from llm import validate_gemini_api

        with caplog.at_level(logging.WARNING):
            validate_gemini_api()
        assert "GEMINI_API_KEY not set" in caplog.text

    @patch("llm.check_gemini_health")
    @patch("llm.GEMINI_API_KEY", "test-key")
    def test_logs_success_when_healthy(self, mock_health, caplog):
        """Log success when Gemini API is healthy."""
        import logging

        from llm import validate_gemini_api

        mock_health.return_value = {
            "healthy": True,
            "message": "ok",
        }
        with caplog.at_level(logging.INFO):
            validate_gemini_api()
        assert "validated successfully" in caplog.text

    @patch("llm.check_gemini_health")
    @patch("llm.GEMINI_API_KEY", "test-key")
    def test_warns_when_unhealthy(self, mock_health, caplog):
        """Warn when Gemini API validation fails."""
        import logging

        from llm import validate_gemini_api

        mock_health.return_value = {
            "healthy": False,
            "message": "API down",
        }
        with caplog.at_level(logging.WARNING):
            validate_gemini_api()
        assert "validation failed" in caplog.text


# ============================================================================
# EXTRACTIVE ANSWER TESTS
# ============================================================================


class TestExtractiveAnswer:
    """Tests for generate_extractive_answer."""

    def test_empty_facts_returns_empty(self):
        """Return empty string when no facts provided."""
        from llm import generate_extractive_answer

        assert generate_extractive_answer([], "question") == ""

    def test_short_sentences_filtered_out(self):
        """Filter out sentences shorter than 30 chars."""
        from llm import generate_extractive_answer

        facts = [{"sentence": "short"}]
        assert generate_extractive_answer(facts, "q") == ""

    def test_joins_valid_sentences(self):
        """Join valid sentences with period separator."""
        from llm import generate_extractive_answer

        facts = [
            {"sentence": "A" * 40},
            {"sentence": "B" * 40},
        ]
        result = generate_extractive_answer(facts, "q")
        assert ". " in result
        assert result.endswith(".")

    def test_cleans_markup_from_sentences(self):
        """Remove markup patterns from sentences."""
        from llm import generate_extractive_answer

        facts = [
            {
                "sentence": (
                    "__bold__ பொன்னி களஞ்சியம் actual content "
                    "that is long enough to pass filter"
                )
            }
        ]
        result = generate_extractive_answer(facts, "q")
        assert "__" not in result
        assert "பொன்னி களஞ்சியம்" not in result


# ============================================================================
# CONTENT DISPLAY QUERY DETECTION
# ============================================================================


class TestIsContentDisplayQuery:
    """Test is_content_display_query detects show/read intent."""

    def test_tamil_show_pattern(self):
        """Detect Tamil 'காட்டு' pattern."""
        from llm import is_content_display_query

        assert is_content_display_query("வளையல் வாங்கலீயோ கதையை காட்டு")

    def test_tamil_read_pattern(self):
        """Detect Tamil 'படிக்க' pattern."""
        from llm import is_content_display_query

        assert is_content_display_query("இந்தக் கவிதையை படிக்க வேண்டும்")

    def test_tamil_full_content(self):
        """Detect Tamil 'முழு கதை' pattern."""
        from llm import is_content_display_query

        assert is_content_display_query("முழு கதை என்ன")

    def test_english_show(self):
        """Detect English 'show content' pattern."""
        from llm import is_content_display_query

        assert is_content_display_query("show the article about Dravidian movement")

    def test_english_full_text(self):
        """Detect English 'full text' pattern."""
        from llm import is_content_display_query

        assert is_content_display_query("full text of the poem")

    def test_non_display_query(self):
        """Regular question should not match."""
        from llm import is_content_display_query

        assert not is_content_display_query("பொன்னி இதழ் பற்றி கூறுக")

    def test_summary_query_not_display(self):
        """Summary question should not match display patterns."""
        from llm import is_content_display_query

        assert not is_content_display_query("கருணாநிதி கருத்து என்ன")


class TestBuildUserContentDisplay:
    """Test _build_user_content uses display closing for content queries."""

    def test_display_closing_tamil(self):
        """Content display query gets display-specific closing."""
        from llm import _build_user_content

        result = _build_user_content("கதையை காட்டு", "doc content", "", language="ta")
        assert "அசல் உரை" in result or "உள்ளடக்கத்தை" in result

    def test_display_closing_english(self):
        """English content display query gets display-specific closing."""
        from llm import _build_user_content

        result = _build_user_content(
            "show the article", "doc content", "", language="en"
        )
        assert "READ" in result or "Display" in result

    def test_normal_query_no_display_closing(self):
        """Normal query should not get display closing."""
        from llm import _build_user_content

        result = _build_user_content(
            "பொன்னி இதழ் பற்றி கூறுக", "doc content", "", language="ta"
        )
        assert "அசல் உரை" not in result
