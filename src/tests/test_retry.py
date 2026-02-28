"""
Test suite for retry utilities (src/db/retry.py).
Covers exception predicates, backoff computation, sync/async retry loops,
service-specific wrappers, and non-retryable error propagation.
"""
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from retry import (
    is_retryable_qdrant,
    is_retryable_gemini,
    _compute_delay,
    retry_sync,
    retry_async,
    with_qdrant_retry,
    with_gemini_retry,
    with_gemini_retry_async,
)


# ============================================================================
# QDRANT PREDICATE TESTS
# ============================================================================


class TestQdrantPredicate:
    """Test is_retryable_qdrant for various exception types."""

    def test_connection_error(self):
        assert is_retryable_qdrant(ConnectionError("refused")) is True

    def test_timeout_error(self):
        assert is_retryable_qdrant(TimeoutError("timed out")) is True

    def test_os_error(self):
        assert is_retryable_qdrant(OSError("network unreachable")) is True

    def test_httpx_connect_error(self):
        try:
            import httpx
            exc = httpx.ConnectError("connection refused")
            assert is_retryable_qdrant(exc) is True
        except ImportError:
            pytest.skip("httpx not installed")

    def test_response_handling_exception(self):
        try:
            from qdrant_client.http.exceptions import ResponseHandlingException
            exc = ResponseHandlingException("bad response")
            assert is_retryable_qdrant(exc) is True
        except ImportError:
            pytest.skip("qdrant_client not installed")

    def test_unexpected_response_retryable_status(self):
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse
            for status in (429, 502, 503, 504):
                exc = UnexpectedResponse.__new__(UnexpectedResponse)
                exc.status_code = status
                assert is_retryable_qdrant(exc) is True, f"status {status} should be retryable"
        except ImportError:
            pytest.skip("qdrant_client not installed")

    def test_unexpected_response_non_retryable_status(self):
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse
            exc = UnexpectedResponse.__new__(UnexpectedResponse)
            exc.status_code = 400
            assert is_retryable_qdrant(exc) is False
        except ImportError:
            pytest.skip("qdrant_client not installed")

    def test_resource_exhausted_response(self):
        # Simulate by creating a class with matching name
        class ResourceExhaustedResponse(Exception):
            pass
        exc = ResourceExhaustedResponse("quota")
        assert is_retryable_qdrant(exc) is True

    def test_value_error_not_retryable(self):
        assert is_retryable_qdrant(ValueError("bad value")) is False

    def test_type_error_not_retryable(self):
        assert is_retryable_qdrant(TypeError("wrong type")) is False

    def test_key_error_not_retryable(self):
        assert is_retryable_qdrant(KeyError("missing key")) is False


# ============================================================================
# GEMINI PREDICATE TESTS
# ============================================================================


class TestGeminiPredicate:
    """Test is_retryable_gemini for various exception types."""

    def test_connection_error(self):
        assert is_retryable_gemini(ConnectionError("refused")) is True

    def test_timeout_error(self):
        assert is_retryable_gemini(TimeoutError("timed out")) is True

    def test_httpx_connect_error(self):
        try:
            import httpx
            exc = httpx.ConnectError("connection refused")
            assert is_retryable_gemini(exc) is True
        except ImportError:
            pytest.skip("httpx not installed")

    def test_server_error(self):
        try:
            from google.genai.errors import ServerError
            exc = ServerError.__new__(ServerError)
            assert is_retryable_gemini(exc) is True
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_client_error_429(self):
        try:
            from google.genai.errors import ClientError
            exc = ClientError.__new__(ClientError)
            exc.code = 429
            assert is_retryable_gemini(exc) is True
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_client_error_400_not_retryable(self):
        try:
            from google.genai.errors import ClientError
            exc = ClientError.__new__(ClientError)
            exc.code = 400
            # ClientError with non-429 code — not retryable via ClientError check,
            # but may still match via message. Use a clean message.
            exc.args = ("bad request",)
            assert is_retryable_gemini(exc) is False
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_rate_limit_in_message(self):
        exc = Exception("rate limit exceeded for model")
        assert is_retryable_gemini(exc) is True

    def test_resource_exhausted_in_message(self):
        exc = Exception("RESOURCE EXHAUSTED: quota reached")
        assert is_retryable_gemini(exc) is True

    def test_value_error_not_retryable(self):
        assert is_retryable_gemini(ValueError("bad value")) is False

    def test_type_error_not_retryable(self):
        assert is_retryable_gemini(TypeError("wrong type")) is False

    def test_key_error_not_retryable(self):
        assert is_retryable_gemini(KeyError("missing key")) is False


# ============================================================================
# BACKOFF COMPUTATION TESTS
# ============================================================================


class TestBackoff:
    """Test _compute_delay for exponential backoff and jitter."""

    def test_first_attempt_delay(self):
        delay = _compute_delay(attempt=0, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(0.5)

    def test_second_attempt_delay(self):
        delay = _compute_delay(attempt=1, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(1.0)

    def test_third_attempt_delay(self):
        delay = _compute_delay(attempt=2, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(2.0)

    def test_delay_capped_at_maximum(self):
        delay = _compute_delay(attempt=10, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(4.0)

    def test_jitter_varies_delay(self):
        delays = set()
        for _ in range(50):
            d = _compute_delay(attempt=1, base=1.0, maximum=8.0, jitter=0.25)
            delays.add(round(d, 4))
        # With 25% jitter on base=2.0, range should be [1.5, 2.5]
        assert len(delays) > 1, "Jitter should produce varying delays"
        for d in delays:
            assert 1.5 <= d <= 2.5, f"Delay {d} outside expected jitter range"

    def test_qdrant_config_values(self):
        """Verify Qdrant default backoff: 0.5s base, 4s max."""
        d0 = _compute_delay(0, base=0.5, maximum=4.0, jitter=0.0)
        d1 = _compute_delay(1, base=0.5, maximum=4.0, jitter=0.0)
        assert d0 == pytest.approx(0.5)
        assert d1 == pytest.approx(1.0)

    def test_gemini_config_values(self):
        """Verify Gemini default backoff: 1s base, 8s max."""
        d0 = _compute_delay(0, base=1.0, maximum=8.0, jitter=0.0)
        d1 = _compute_delay(1, base=1.0, maximum=8.0, jitter=0.0)
        d2 = _compute_delay(2, base=1.0, maximum=8.0, jitter=0.0)
        assert d0 == pytest.approx(1.0)
        assert d1 == pytest.approx(2.0)
        assert d2 == pytest.approx(4.0)


# ============================================================================
# QDRANT RETRY WRAPPER TESTS
# ============================================================================


class TestQdrantRetry:
    """Test with_qdrant_retry wrapper."""

    @patch("retry.time.sleep")
    def test_succeeds_first_attempt(self, mock_sleep):
        fn = MagicMock(return_value="ok")
        result = with_qdrant_retry(fn, "arg1", key="val")
        assert result == "ok"
        fn.assert_called_once_with("arg1", key="val")
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_succeeds_after_transient_failure(self, mock_sleep):
        fn = MagicMock(side_effect=[ConnectionError("refused"), "ok"])
        result = with_qdrant_retry(fn, "arg1")
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("retry.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        fn = MagicMock(side_effect=ConnectionError("refused"))
        with pytest.raises(ConnectionError):
            with_qdrant_retry(fn)
        assert fn.call_count == 3  # max_attempts=3


# ============================================================================
# GEMINI RETRY WRAPPER TESTS
# ============================================================================


class TestGeminiRetry:
    """Test with_gemini_retry wrapper."""

    @patch("retry.time.sleep")
    def test_succeeds_after_one_failure(self, mock_sleep):
        fn = MagicMock(side_effect=[TimeoutError("timeout"), "answer"])
        result = with_gemini_retry(fn, model="gemini")
        assert result == "answer"
        assert fn.call_count == 2

    @patch("retry.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        fn = MagicMock(side_effect=TimeoutError("timeout"))
        with pytest.raises(TimeoutError):
            with_gemini_retry(fn)
        assert fn.call_count == 3


# ============================================================================
# NON-RETRYABLE ERROR TESTS
# ============================================================================


class TestNonRetryableErrors:
    """Verify programming errors propagate immediately without retry."""

    @patch("retry.time.sleep")
    def test_value_error_qdrant(self, mock_sleep):
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError):
            with_qdrant_retry(fn)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_type_error_gemini(self, mock_sleep):
        fn = MagicMock(side_effect=TypeError("wrong"))
        with pytest.raises(TypeError):
            with_gemini_retry(fn)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_key_error_qdrant(self, mock_sleep):
        fn = MagicMock(side_effect=KeyError("missing"))
        with pytest.raises(KeyError):
            with_qdrant_retry(fn)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()


# ============================================================================
# ASYNC RETRY TESTS
# ============================================================================


class TestAsyncRetry:
    """Test async retry variants."""

    @pytest.mark.asyncio
    async def test_async_succeeds_after_failure(self):
        call_count = 0
        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")
            return "result"

        with patch("retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_gemini_retry_async(flaky, "arg")
        assert result == "result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_exhausted_raises(self):
        async def always_fail(*args, **kwargs):
            raise TimeoutError("timeout")

        with patch("retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError):
                await with_gemini_retry_async(always_fail)

    @pytest.mark.asyncio
    async def test_async_non_retryable_propagates(self):
        call_count = 0
        async def bad_code(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("programming error")

        with patch("retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError):
                await with_gemini_retry_async(bad_code)
        assert call_count == 1
