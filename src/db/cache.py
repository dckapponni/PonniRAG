"""Thread-safe TTL + LRU response cache for ask_question results."""

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
    """Thread-safe TTL + LRU cache for ask_question results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize cache with given max size and TTL."""
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
        """Compute a deterministic cache key from a question string."""
        normalized = unicodedata.normalize("NFC", question)
        normalized = re.sub(r"\s+", " ", normalized.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        """Retrieve a cached result by question, or None if missing/expired."""
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
        """Store a result in the cache, evicting oldest if full."""
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
        """Clear all cached entries and reset version."""
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
        """Return cache statistics including hit rate and version."""
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
