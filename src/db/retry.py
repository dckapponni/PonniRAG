"""
Retry utilities with exponential backoff for external service calls.
Provides service-specific wrappers for Qdrant and Gemini API calls.
No external dependencies — uses only stdlib + safe imports of client libraries.
"""
import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception predicates
# ---------------------------------------------------------------------------

def is_retryable_qdrant(exc: Exception) -> bool:
    """Return True if the Qdrant exception is transient and worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    # httpx transport errors (connection refused, timeout, etc.)
    try:
        import httpx
        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:
        pass

    # qdrant-client specific transient errors
    try:
        from qdrant_client.http.exceptions import (
            ResponseHandlingException,
            UnexpectedResponse,
        )
        if isinstance(exc, ResponseHandlingException):
            return True
        if isinstance(exc, UnexpectedResponse):
            status = getattr(exc, "status_code", None)
            if status in (429, 502, 503, 504):
                return True
    except ImportError:
        pass

    # Resource exhausted (quota)
    exc_type_name = type(exc).__name__
    if exc_type_name == "ResourceExhaustedResponse":
        return True

    return False


def is_retryable_gemini(exc: Exception) -> bool:
    """Return True if the Gemini exception is transient and worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    # httpx transport errors
    try:
        import httpx
        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:
        pass

    # google-genai specific errors
    try:
        from google.genai.errors import ServerError, ClientError, APIError
        if isinstance(exc, ServerError):
            return True
        if isinstance(exc, ClientError):
            code = getattr(exc, "code", None)
            if code == 429:
                return True
        if isinstance(exc, APIError):
            code = getattr(exc, "code", None)
            if code in (429, 500, 502, 503, 504):
                return True
    except ImportError:
        pass

    # Fallback: check error message for rate-limit keywords
    msg = str(exc).lower()
    if "rate limit" in msg or "resource exhausted" in msg:
        return True

    return False


# ---------------------------------------------------------------------------
# Backoff computation
# ---------------------------------------------------------------------------

def _compute_delay(
    attempt: int,
    base: float,
    maximum: float,
    jitter: float,
) -> float:
    """Compute delay with exponential backoff and random jitter.

    delay = min(base * 2^attempt, maximum) * (1 ± jitter)
    """
    delay = min(base * (2 ** attempt), maximum)
    if jitter > 0:
        delay *= 1 + random.uniform(-jitter, jitter)
    return max(0, delay)


# ---------------------------------------------------------------------------
# Core retry loops
# ---------------------------------------------------------------------------

def retry_sync(
    fn,
    args=(),
    kwargs=None,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    jitter: float = 0.25,
    is_retryable=None,
):
    """Synchronous retry loop with exponential backoff.

    Calls fn(*args, **kwargs) up to max_attempts times. Non-retryable
    exceptions propagate immediately.
    """
    kwargs = kwargs or {}
    if is_retryable is None:
        is_retryable = lambda e: False

    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = _compute_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                "Retry %d/%d for %s after %.2fs: %s",
                attempt + 1, max_attempts,
                getattr(fn, "__name__", repr(fn)), delay, exc,
            )
            time.sleep(delay)

    raise last_exc  # pragma: no cover — unreachable, satisfies type checker


async def retry_async(
    fn,
    args=(),
    kwargs=None,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    jitter: float = 0.25,
    is_retryable=None,
):
    """Async retry loop with exponential backoff.

    Calls await fn(*args, **kwargs) up to max_attempts times.
    """
    kwargs = kwargs or {}
    if is_retryable is None:
        is_retryable = lambda e: False

    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = _compute_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                "Async retry %d/%d for %s after %.2fs: %s",
                attempt + 1, max_attempts,
                getattr(fn, "__name__", repr(fn)), delay, exc,
            )
            await asyncio.sleep(delay)

    raise last_exc  # pragma: no cover


# ---------------------------------------------------------------------------
# Service-specific convenience wrappers
# ---------------------------------------------------------------------------

def with_qdrant_retry(fn, *args, **kwargs):
    """Retry a Qdrant call: 3 attempts, 0.5s→4s backoff, 25% jitter."""
    return retry_sync(
        fn,
        args=args,
        kwargs=kwargs,
        max_attempts=3,
        base_delay=0.5,
        max_delay=4.0,
        jitter=0.25,
        is_retryable=is_retryable_qdrant,
    )


def with_gemini_retry(fn, *args, **kwargs):
    """Retry a Gemini call: 3 attempts, 1s→8s backoff, 25% jitter."""
    return retry_sync(
        fn,
        args=args,
        kwargs=kwargs,
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
        jitter=0.25,
        is_retryable=is_retryable_gemini,
    )


async def with_gemini_retry_async(fn, *args, **kwargs):
    """Async retry a Gemini call: 3 attempts, 1s→8s backoff, 25% jitter."""
    return await retry_async(
        fn,
        args=args,
        kwargs=kwargs,
        max_attempts=3,
        base_delay=1.0,
        max_delay=8.0,
        jitter=0.25,
        is_retryable=is_retryable_gemini,
    )
