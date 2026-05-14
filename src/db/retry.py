"""Retry utilities with exponential backoff for external service calls.

Provides service-specific wrappers for Qdrant and Gemini API calls.
No external dependencies -- uses only stdlib plus safe client imports.
"""

import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


def is_retryable_qdrant(exc: Exception) -> bool:
    """Return True if the Qdrant exception is transient and worth retrying.

    Checks the exception against the following categories in order:

    - **Standard library transients**: ``ConnectionError``, ``TimeoutError``,
      ``OSError``.
    - **httpx transport errors**: any ``httpx.TransportError`` subclass
      (connection refused, timeout, etc.). Skipped gracefully if ``httpx``
      is not installed.
    - **Qdrant client errors**: ``ResponseHandlingException`` (always
      retryable) and ``UnexpectedResponse`` with HTTP status codes
      ``429``, ``502``, ``503``, or ``504``. Skipped gracefully if
      ``qdrant_client`` is not installed.
    - **Resource exhaustion**: exceptions whose class name is
      ``"ResourceExhaustedResponse"``.

    Args:
        exc (Exception): The exception raised by a Qdrant API call.

    Returns:
        bool: ``True`` if the error is likely transient and the call should
        be retried; ``False`` if it is a permanent or unrecognized error.
    """
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
    """Return True if the Gemini exception is transient and worth retrying.

    Checks the exception against the following categories in order:

    - **Standard library transients**: ``ConnectionError``, ``TimeoutError``,
      ``OSError``.
    - **httpx transport errors**: any ``httpx.TransportError`` subclass.
      Skipped gracefully if ``httpx`` is not installed.
    - **google-genai errors**: ``ServerError`` (always retryable),
      ``ClientError`` with HTTP status ``429`` (rate limit), and
      ``APIError`` with status codes ``429``, ``500``, ``502``, ``503``,
      or ``504``. Skipped gracefully if ``google.genai`` is not installed.
    - **Message-based fallback**: any exception whose string representation
      contains ``"rate limit"`` or ``"resource exhausted"`` (case-insensitive).

    Args:
        exc (Exception): The exception raised by a Gemini API call.

    Returns:
        bool: ``True`` if the error is likely transient and the call should
        be retried; ``False`` if it is a permanent or unrecognized error.
    """
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
        from google.genai.errors import APIError, ClientError, ServerError

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


def _default_not_retryable(e: Exception) -> bool:
    """Return False for all exceptions, treating every error as non-retryable.

    Used as the default ``is_retryable`` predicate in :func:`retry_sync`
    and :func:`retry_async` when the caller does not supply one, ensuring
    that unguarded calls fail fast on the first exception rather than
    silently retrying.

    Args:
        e (Exception): The caught exception (unused).

    Returns:
        bool: Always ``False``.
    """
    return False


def _compute_delay(
    attempt: int,
    base: float,
    maximum: float,
    jitter: float,
) -> float:
    """Compute a retry delay using exponential backoff with random jitter.

    Calculates the delay as:

    .. code-block:: text

        delay = min(base * 2^attempt, maximum) * (1 ± jitter)

    The jitter factor is sampled uniformly from
    ``[-jitter, +jitter]``, so the actual multiplier is drawn from
    ``(1 - jitter, 1 + jitter)``. The result is clamped to a minimum
    of ``0`` to guard against negative values when ``jitter > 1``.

    Args:
        attempt (int): Zero-based attempt index. ``0`` yields approximately
            ``base`` seconds; each subsequent attempt doubles the delay.
        base (float): Base delay in seconds for the first retry.
        maximum (float): Hard ceiling on the pre-jitter delay in seconds.
        jitter (float): Fractional jitter range. ``0`` disables jitter;
            ``0.25`` introduces ±25 % randomness.

    Returns:
        float: Delay in seconds, always non-negative.
    """
    delay = min(base * (2**attempt), maximum)
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
    """Run a synchronous function with exponential-backoff retry.

    Calls ``fn(*args, **kwargs)`` up to ``max_attempts`` times. On each
    failure the exception is passed to ``is_retryable``; non-retryable
    exceptions and final-attempt failures propagate immediately without
    sleeping. Retryable failures are logged at WARNING level and followed
    by a :func:`time.sleep` of the computed backoff delay.

    Args:
        fn (callable): Synchronous function to call.
        args (tuple): Positional arguments forwarded to ``fn``.
        kwargs (dict | None): Keyword arguments forwarded to ``fn``.
            Defaults to an empty dict.
        max_attempts (int): Total number of attempts including the first
            call. Defaults to ``3``.
        base_delay (float): Base retry delay in seconds. Defaults to
            ``0.5``.
        max_delay (float): Maximum retry delay in seconds. Defaults to
            ``4.0``.
        jitter (float): Fractional jitter applied to each delay.
            Defaults to ``0.25``.
        is_retryable (callable | None): Single-argument predicate that
            receives the caught exception and returns ``True`` if it is
            worth retrying. When ``None``, all exceptions are treated as
            non-retryable and propagate on the first failure.

    Returns:
        Any: The return value of ``fn`` on success.

    Raises:
        Exception: Re-raises the last exception when all attempts are
        exhausted or when ``is_retryable`` returns ``False``.
    """
    kwargs = kwargs or {}
    check_retryable = is_retryable or _default_not_retryable

    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not check_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = _compute_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                "Retry %d/%d for %s after %.2fs: %s",
                attempt + 1,
                max_attempts,
                getattr(fn, "__name__", repr(fn)),
                delay,
                exc,
            )
            time.sleep(delay)

    raise last_exc  # pragma: no cover


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
    """Run an async function with exponential-backoff retry.

    Async counterpart of :func:`retry_sync`. Calls ``await fn(*args,
    **kwargs)`` up to ``max_attempts`` times and uses
    ``asyncio.sleep`` between attempts so the event loop is not blocked
    during the backoff window.

    Args:
        fn (coroutine function): Async function to call.
        args (tuple): Positional arguments forwarded to ``fn``.
        kwargs (dict | None): Keyword arguments forwarded to ``fn``.
            Defaults to an empty dict.
        max_attempts (int): Total number of attempts including the first
            call. Defaults to ``3``.
        base_delay (float): Base retry delay in seconds. Defaults to
            ``0.5``.
        max_delay (float): Maximum retry delay in seconds. Defaults to
            ``4.0``.
        jitter (float): Fractional jitter applied to each delay.
            Defaults to ``0.25``.
        is_retryable (callable | None): Single-argument predicate that
            receives the caught exception and returns ``True`` if it is
            worth retrying. When ``None``, all exceptions are treated as
            non-retryable and propagate on the first failure.

    Returns:
        Any: The return value of ``await fn(...)`` on success.

    Raises:
        Exception: Re-raises the last exception when all attempts are
        exhausted or when ``is_retryable`` returns ``False``.
    """
    kwargs = kwargs or {}
    check_retryable = is_retryable or _default_not_retryable

    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not check_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = _compute_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                "Async retry %d/%d for %s after %.2fs: %s",
                attempt + 1,
                max_attempts,
                getattr(fn, "__name__", repr(fn)),
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    raise last_exc  # pragma: no cover


def with_qdrant_retry(fn, *args, **kwargs):
    """Retry a Qdrant API call with service-tuned exponential backoff.

    Convenience wrapper around :func:`retry_sync` pre-configured for
    Qdrant: 3 attempts, base delay of 0.5 s, maximum delay of 4 s, and
    25 % jitter. Uses :func:`is_retryable_qdrant` to classify exceptions.

    Args:
        fn (callable): Qdrant client method to call.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Any: The return value of ``fn`` on success.

    Raises:
        Exception: Re-raises the last exception after all retry attempts
        are exhausted or on a non-retryable error.
    """
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
    """Retry a synchronous Gemini API call with service-tuned exponential backoff.

    Convenience wrapper around :func:`retry_sync` pre-configured for
    Gemini: 3 attempts, base delay of 1 s, maximum delay of 8 s, and
    25 % jitter. The longer base and maximum delays account for Gemini's
    rate-limit recovery windows, which are typically longer than Qdrant's.
    Uses :func:`is_retryable_gemini` to classify exceptions.

    Args:
        fn (callable): Synchronous Gemini API method to call.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Any: The return value of ``fn`` on success.

    Raises:
        Exception: Re-raises the last exception after all retry attempts
        are exhausted or on a non-retryable error.
    """
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
    """Retry an async Gemini API call with service-tuned exponential backoff.

    Async convenience wrapper around :func:`retry_async` pre-configured for
    Gemini: 3 attempts, base delay of 1 s, maximum delay of 8 s, and 25 %
    jitter. Uses :func:`is_retryable_gemini` to classify exceptions.

    Args:
        fn (coroutine function): Async Gemini API method to call.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Any: The return value of ``await fn(...)`` on success.

    Raises:
        Exception: Re-raises the last exception after all retry attempts
        are exhausted or on a non-retryable error.
    """
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
