"""
Thread-safe TTL + LRU response cache for ask_question results.
"""
import re
import hashlib
import time
import threading
import logging
import unicodedata
from collections import OrderedDict
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ResponseCache:
    """Thread-safe TTL + LRU cache for ask_question results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(question: str) -> str:
        normalized = unicodedata.normalize("NFC", question)
        normalized = re.sub(r'\s+', ' ', normalized.strip().lower())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
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
            logger.info(f"[CACHE] HIT (hits={self._hits}, misses={self._misses}, rate={self._hits / total:.0%})")
            return self._cache[key]

    def put(self, question: str, result: Dict) -> None:
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
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info("[CACHE] Cache cleared")

    def stats(self) -> Dict:
        with self._lock:
            total = max(1, self._hits + self._misses)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.0%}",
            }


_response_cache = ResponseCache(max_size=100, ttl_seconds=3600)
