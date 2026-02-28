"""
Unit tests for ResponseCache (TTL + LRU + index version awareness).
"""
import time
import threading
import pytest

from cache import ResponseCache


# ============================================================================
# Basic cache operations
# ============================================================================

class TestResponseCacheBasic:
    """Test core TTL + LRU cache behaviour."""

    def test_put_and_get(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("hello", {"answer": "world"})
        assert cache.get("hello") == {"answer": "world"}

    def test_get_miss(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = ResponseCache(max_size=10, ttl_seconds=1)
        cache.put("q", {"answer": "a"})
        time.sleep(1.1)
        assert cache.get("q") is None

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=2, ttl_seconds=60)
        cache.put("a", {"answer": "1"})
        cache.put("b", {"answer": "2"})
        cache.put("c", {"answer": "3"})  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == {"answer": "2"}
        assert cache.get("c") == {"answer": "3"}

    def test_lru_access_refreshes_order(self):
        cache = ResponseCache(max_size=2, ttl_seconds=60)
        cache.put("a", {"answer": "1"})
        cache.put("b", {"answer": "2"})
        cache.get("a")  # refresh "a" — "b" is now oldest
        cache.put("c", {"answer": "3"})  # evicts "b"
        assert cache.get("a") == {"answer": "1"}
        assert cache.get("b") is None

    def test_clear(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("x", {"answer": "y"})
        cache.clear()
        assert cache.get("x") is None
        assert cache.stats()["size"] == 0

    def test_stats(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("q", {"answer": "a"})
        cache.get("q")       # hit
        cache.get("miss")    # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 60

    def test_key_normalization(self):
        """Whitespace and case differences map to the same key."""
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("  Hello  World  ", {"answer": "hw"})
        assert cache.get("hello world") == {"answer": "hw"}


# ============================================================================
# Index version tracking
# ============================================================================

class TestResponseCacheVersioning:
    """Test check_version() index-aware cache invalidation."""

    def test_check_version_initializes_on_first_call(self):
        cache = ResponseCache()
        cache.check_version(1000)
        assert cache.stats()["version"] == 1000

    def test_check_version_no_clear_when_unchanged(self):
        cache = ResponseCache()
        cache.check_version(500)
        cache.put("q", {"answer": "a"})
        cache.check_version(500)
        assert cache.get("q") == {"answer": "a"}

    def test_check_version_clears_on_change(self):
        cache = ResponseCache()
        cache.check_version(500)
        cache.put("q", {"answer": "a"})
        cache.check_version(600)
        assert cache.get("q") is None
        assert cache.stats()["version"] == 600

    def test_check_version_ignores_none(self):
        cache = ResponseCache()
        cache.check_version(500)
        cache.put("q", {"answer": "a"})
        cache.check_version(None)  # Qdrant unreachable
        assert cache.get("q") == {"answer": "a"}
        assert cache.stats()["version"] == 500

    def test_check_version_none_before_initialization(self):
        cache = ResponseCache()
        cache.check_version(None)  # None first call — version stays None
        assert cache.stats()["version"] is None

    def test_check_version_after_manual_clear(self):
        cache = ResponseCache()
        cache.check_version(500)
        assert cache.stats()["version"] == 500
        cache.clear()
        assert cache.stats()["version"] is None
        cache.check_version(600)
        assert cache.stats()["version"] == 600

    def test_check_version_decrease(self):
        """Fewer points (partial reindex) should still clear."""
        cache = ResponseCache()
        cache.check_version(1000)
        cache.put("q", {"answer": "a"})
        cache.check_version(800)
        assert cache.get("q") is None
        assert cache.stats()["version"] == 800

    def test_check_version_zero(self):
        """Empty collection (0 points) should clear."""
        cache = ResponseCache()
        cache.check_version(1000)
        cache.put("q", {"answer": "a"})
        cache.check_version(0)
        assert cache.get("q") is None
        assert cache.stats()["version"] == 0

    def test_version_survives_cache_operations(self):
        """put/get/eviction don't affect version."""
        cache = ResponseCache(max_size=2, ttl_seconds=60)
        cache.check_version(42)
        cache.put("a", {"answer": "1"})
        cache.put("b", {"answer": "2"})
        cache.put("c", {"answer": "3"})  # evicts "a"
        cache.get("b")
        cache.get("miss")
        assert cache.stats()["version"] == 42

    def test_version_in_stats(self):
        cache = ResponseCache()
        assert cache.stats()["version"] is None
        cache.check_version(123)
        assert cache.stats()["version"] == 123


# ============================================================================
# Thread safety
# ============================================================================

class TestResponseCacheThreadSafety:
    """Verify no crashes under concurrent access."""

    def test_concurrent_version_changes(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        cache.check_version(100)
        errors = []

        def toggle_version(v):
            try:
                for _ in range(50):
                    cache.check_version(v)
                    cache.put(f"q-{v}", {"answer": str(v)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_version, args=(i,)) for i in range(200, 210)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_reads_writes_during_version_change(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        cache.check_version(100)
        for i in range(50):
            cache.put(f"q{i}", {"answer": str(i)})
        errors = []

        def reader():
            try:
                for i in range(100):
                    cache.get(f"q{i % 50}")
            except Exception as e:
                errors.append(e)

        def version_changer():
            try:
                for v in range(200, 300):
                    cache.check_version(v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=version_changer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
