"""Test the retry utilities module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from retry import (
    _compute_delay,
    is_retryable_gemini,
    is_retryable_qdrant,
    with_gemini_retry,
    with_gemini_retry_async,
    with_qdrant_retry,
)

# ============================================================================
# QDRANT PREDICATE TESTS
# ============================================================================


class TestQdrantPredicate:
    """Test is_retryable_qdrant for various exception types."""

    def test_connection_error(self):
        """Retry on ConnectionError."""
        assert is_retryable_qdrant(ConnectionError("refused")) is True

    def test_timeout_error(self):
        """Retry on TimeoutError."""
        assert is_retryable_qdrant(TimeoutError("timed out")) is True

    def test_os_error(self):
        """Retry on OSError."""
        assert is_retryable_qdrant(OSError("network unreachable")) is True

    def test_httpx_connect_error(self):
        """Retry on httpx ConnectError."""
        try:
            import httpx

            exc = httpx.ConnectError("connection refused")
            assert is_retryable_qdrant(exc) is True
        except ImportError:
            pytest.skip("httpx not installed")

    def test_response_handling_exception(self):
        """Retry on ResponseHandlingException."""
        try:
            from qdrant_client.http.exceptions import ResponseHandlingException

            exc = ResponseHandlingException("bad response")
            assert is_retryable_qdrant(exc) is True
        except ImportError:
            pytest.skip("qdrant_client not installed")

    def test_unexpected_response_retryable_status(self):
        """Retry on retryable HTTP status codes."""
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse

            for status in (429, 502, 503, 504):
                exc = UnexpectedResponse.__new__(UnexpectedResponse)
                exc.status_code = status
                assert (
                    is_retryable_qdrant(exc) is True
                ), f"status {status} should be retryable"
        except ImportError:
            pytest.skip("qdrant_client not installed")

    def test_unexpected_response_non_retryable_status(self):
        """Skip retry for non-retryable HTTP status codes."""
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse

            exc = UnexpectedResponse.__new__(UnexpectedResponse)
            exc.status_code = 400
            assert is_retryable_qdrant(exc) is False
        except ImportError:
            pytest.skip("qdrant_client not installed")

    def test_resource_exhausted_response(self):
        """Retry on ResourceExhausted-like exception."""

        class ResourceExhaustedResponse(Exception):
            """Simulated resource exhausted error."""

        exc = ResourceExhaustedResponse("quota")
        assert is_retryable_qdrant(exc) is True

    def test_value_error_not_retryable(self):
        """Skip retry on ValueError."""
        assert is_retryable_qdrant(ValueError("bad value")) is False

    def test_type_error_not_retryable(self):
        """Skip retry on TypeError."""
        assert is_retryable_qdrant(TypeError("wrong type")) is False

    def test_key_error_not_retryable(self):
        """Skip retry on KeyError."""
        assert is_retryable_qdrant(KeyError("missing key")) is False


# ============================================================================
# GEMINI PREDICATE TESTS
# ============================================================================


class TestGeminiPredicate:
    """Test is_retryable_gemini for various exception types."""

    def test_connection_error(self):
        """Retry on ConnectionError."""
        assert is_retryable_gemini(ConnectionError("refused")) is True

    def test_timeout_error(self):
        """Retry on TimeoutError."""
        assert is_retryable_gemini(TimeoutError("timed out")) is True

    def test_httpx_connect_error(self):
        """Retry on httpx ConnectError."""
        try:
            import httpx

            exc = httpx.ConnectError("connection refused")
            assert is_retryable_gemini(exc) is True
        except ImportError:
            pytest.skip("httpx not installed")

    def test_server_error(self):
        """Retry on Gemini ServerError."""
        try:
            from google.genai.errors import ServerError

            exc = ServerError.__new__(ServerError)
            assert is_retryable_gemini(exc) is True
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_client_error_429(self):
        """Retry on ClientError with 429 status."""
        try:
            from google.genai.errors import ClientError

            exc = ClientError.__new__(ClientError)
            exc.code = 429
            assert is_retryable_gemini(exc) is True
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_client_error_400_not_retryable(self):
        """Skip retry on ClientError with 400 status."""
        try:
            from google.genai.errors import ClientError

            exc = ClientError.__new__(ClientError)
            exc.code = 400
            exc.args = ("bad request",)
            assert is_retryable_gemini(exc) is False
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_rate_limit_in_message(self):
        """Retry on rate limit message."""
        exc = Exception("rate limit exceeded for model")
        assert is_retryable_gemini(exc) is True

    def test_resource_exhausted_in_message(self):
        """Retry on resource exhausted message."""
        exc = Exception("RESOURCE EXHAUSTED: quota reached")
        assert is_retryable_gemini(exc) is True

    def test_value_error_not_retryable(self):
        """Skip retry on ValueError."""
        assert is_retryable_gemini(ValueError("bad value")) is False

    def test_type_error_not_retryable(self):
        """Skip retry on TypeError."""
        assert is_retryable_gemini(TypeError("wrong type")) is False

    def test_key_error_not_retryable(self):
        """Skip retry on KeyError."""
        assert is_retryable_gemini(KeyError("missing key")) is False


# ============================================================================
# BACKOFF COMPUTATION TESTS
# ============================================================================


class TestBackoff:
    """Test _compute_delay for exponential backoff and jitter."""

    def test_first_attempt_delay(self):
        """Compute delay for first attempt."""
        delay = _compute_delay(attempt=0, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(0.5)

    def test_second_attempt_delay(self):
        """Compute delay for second attempt."""
        delay = _compute_delay(attempt=1, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(1.0)

    def test_third_attempt_delay(self):
        """Compute delay for third attempt."""
        delay = _compute_delay(attempt=2, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(2.0)

    def test_delay_capped_at_maximum(self):
        """Cap delay at maximum value."""
        delay = _compute_delay(attempt=10, base=0.5, maximum=4.0, jitter=0.0)
        assert delay == pytest.approx(4.0)

    def test_jitter_varies_delay(self):
        """Produce varying delays with jitter enabled."""
        delays = set()
        for _ in range(50):
            d = _compute_delay(
                attempt=1,
                base=1.0,
                maximum=8.0,
                jitter=0.25,
            )
            delays.add(round(d, 4))
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
        """Succeed on first attempt without sleeping."""
        fn = MagicMock(return_value="ok")
        result = with_qdrant_retry(fn, "arg1", key="val")
        assert result == "ok"
        fn.assert_called_once_with("arg1", key="val")
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_succeeds_after_transient_failure(self, mock_sleep):
        """Succeed after one transient failure."""
        fn = MagicMock(side_effect=[ConnectionError("refused"), "ok"])
        result = with_qdrant_retry(fn, "arg1")
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("retry.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        """Raise after exhausting all retry attempts."""
        fn = MagicMock(side_effect=ConnectionError("refused"))
        with pytest.raises(ConnectionError):
            with_qdrant_retry(fn)
        assert fn.call_count == 3


# ============================================================================
# GEMINI RETRY WRAPPER TESTS
# ============================================================================


class TestGeminiRetry:
    """Test with_gemini_retry wrapper."""

    @patch("retry.time.sleep")
    def test_succeeds_after_one_failure(self, mock_sleep):
        """Succeed after one transient failure."""
        fn = MagicMock(
            side_effect=[
                TimeoutError("timeout"),
                "answer",
            ]
        )
        result = with_gemini_retry(fn, model="gemini")
        assert result == "answer"
        assert fn.call_count == 2

    @patch("retry.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        """Raise after exhausting all retry attempts."""
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
        """Propagate ValueError immediately from Qdrant retry."""
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError):
            with_qdrant_retry(fn)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_type_error_gemini(self, mock_sleep):
        """Propagate TypeError immediately from Gemini retry."""
        fn = MagicMock(side_effect=TypeError("wrong"))
        with pytest.raises(TypeError):
            with_gemini_retry(fn)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_key_error_qdrant(self, mock_sleep):
        """Propagate KeyError immediately from Qdrant retry."""
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
        """Succeed after one async transient failure."""
        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")
            return "result"

        with patch(
            "retry.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await with_gemini_retry_async(flaky, "arg")
        assert result == "result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_exhausted_raises(self):
        """Raise after exhausting async retry attempts."""

        async def always_fail(*args, **kwargs):
            raise TimeoutError("timeout")

        with patch(
            "retry.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(TimeoutError):
                await with_gemini_retry_async(always_fail)

    @pytest.mark.asyncio
    async def test_async_non_retryable_propagates(self):
        """Propagate non-retryable errors in async retry."""
        call_count = 0

        async def bad_code(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("programming error")

        with patch(
            "retry.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(ValueError):
                await with_gemini_retry_async(bad_code)
        assert call_count == 1
