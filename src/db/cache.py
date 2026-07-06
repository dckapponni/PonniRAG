"""Thread-safe TTL + LRU response cache for ask_question results.

Combines a TTL expiry with LRU eviction to keep frequently used answers
in memory and automatically discard stale or least-recently-used entries.
Cache is also invalidated when the Qdrant index changes (points_count shift).

Usage::

    from cache import _response_cache

    result = _response_cache.get(question)
    if result is None:
        result = ask_question(question)
        _response_cache.put(question, result)
"""

import hashlib
import logging
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """Thread-safe TTL + LRU cache for ask_question results.

    Evicts entries when they exceed ``ttl_seconds`` or when the cache
    reaches ``max_size`` (least-recently-used entry dropped first).
    Automatically invalidates all entries when the Qdrant points count
    changes. Safe for concurrent use via an internal ``threading.Lock``.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize the cache with a maximum entry count and TTL.

        Args:
            max_size: Maximum number of entries before LRU eviction kicks in.
            ttl_seconds: Seconds before a cached entry is considered stale.
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._version: Optional[int] = None  # Tracks Qdrant points_count

    @staticmethod
    def _make_key(question: str) -> str:
        """Return a stable SHA-256 cache key for a question string.

        Normalises Unicode (NFC), collapses whitespace, and lowercases
        before hashing so minor formatting differences hit the same key.

        Args:
            question: Raw question string from the caller.

        Returns:
            64-character hex digest string.
        """
        normalized = unicodedata.normalize("NFC", question)
        normalized = re.sub(r"\s+", " ", normalized.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        """Return a cached result, or None if absent or expired.

        Moves a hit entry to the MRU end of the OrderedDict and increments
        hit/miss counters. Expired entries are deleted on access.

        Args:
            question: Question string to look up.

        Returns:
            Cached result dict, or None on miss or TTL expiry.
        """
        key = self._make_key(question)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            if time.time() - self._timestamps.get(key, 0) > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                logger.info(f"[CACHE] TTL expired for {key[:12]}...")
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            total = self._hits + self._misses
            rate = f"{self._hits / total:.0%}"
            logger.info(
                "[CACHE] HIT (hits=%d, misses=%d," " rate=%s)",
                self._hits,
                self._misses,
                rate,
            )
            return self._cache[key]

    def put(self, question: str, result: Dict) -> None:
        """Store a result, evicting the least-recently-used entry if full.

        If the key already exists its value and timestamp are updated in place
        without changing cache size. New entries trigger LRU eviction when
        the cache is at capacity.

        Args:
            question: Question string used to derive the cache key.
            result: Result dict to store.
        """
        key = self._make_key(question)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = result
                self._timestamps[key] = time.time()
                return
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(evicted_key, None)
            self._cache[key] = result
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        """Evict all entries and reset the index version baseline."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._version = None
            logger.info("[CACHE] Cache cleared")

    def check_version(self, points_count: Any) -> None:
        """Clear cache when Qdrant index changes.

        Called after each health check. If points_count differs from
        the stored baseline, all cached entries are invalidated because
        the underlying documents/embeddings may have changed.

        Args:
            points_count: Current points_count from Qdrant health
                check. None means Qdrant is unreachable.
        """
        if points_count is None:
            return
        try:
            points_count = int(points_count)
        except (TypeError, ValueError):
            return
        with self._lock:
            if self._version is None:
                self._version = points_count
                logger.info(f"[CACHE] Index version initialized: {points_count}")
                return
            if points_count != self._version:
                logger.info(
                    "[CACHE] Index version changed:" " %s → %s. Clearing cache.",
                    self._version,
                    points_count,
                )
                self._cache.clear()
                self._timestamps.clear()
                self._version = points_count

    def stats(self) -> Dict:
        """Return a snapshot of cache metrics.

        Returns:
            Dict with keys: size, max_size, ttl_seconds, hits, misses,
            hit_rate (formatted as a percentage string), and version.
        """
        with self._lock:
            total = max(1, self._hits + self._misses)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.0%}",
                "version": self._version,
            }


_response_cache = ResponseCache(max_size=100, ttl_seconds=3600)
