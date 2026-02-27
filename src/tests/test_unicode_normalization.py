"""
Tests for Unicode NFC normalization across the PonniRAG pipeline.

Verifies that Tamil text in different Unicode representations (NFC vs NFD)
produces identical results for hashing, embeddings, fuzzy matching, and caching.
"""
import unicodedata
import pytest
from unittest.mock import patch, MagicMock

from tamil_text import normalize_unicode, fuzzy_match_score, _edit_distance_one


# Tamil test strings in NFC (precomposed) form
TAMIL_NFC = "தமிழ்"  # Standard composed form
# Same string decomposed: base consonants + separate vowel signs
TAMIL_NFD = unicodedata.normalize("NFD", TAMIL_NFC)

# A longer Tamil phrase for realistic testing
PHRASE_NFC = "பொன்னி பத்திரிகை ஆசிரியர்"
PHRASE_NFD = unicodedata.normalize("NFD", PHRASE_NFC)


class TestNormalizeUnicode:
    """Tests for the normalize_unicode() utility in tamil_text.py."""

    def test_nfc_input_unchanged(self):
        """NFC text passes through unchanged."""
        assert normalize_unicode(TAMIL_NFC) == TAMIL_NFC

    def test_nfd_converted_to_nfc(self):
        """NFD text is converted to NFC."""
        result = normalize_unicode(TAMIL_NFD)
        assert result == TAMIL_NFC
        assert unicodedata.is_normalized("NFC", result)

    def test_nfd_and_nfc_produce_same_output(self):
        """Both representations normalize to the same bytes."""
        assert normalize_unicode(TAMIL_NFC) == normalize_unicode(TAMIL_NFD)

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert normalize_unicode("") == ""

    def test_none_returns_none(self):
        """None input returns None (falsy passthrough)."""
        assert normalize_unicode(None) is None

    def test_ascii_unchanged(self):
        """Pure ASCII text is unaffected by NFC normalization."""
        text = "hello world 123"
        assert normalize_unicode(text) == text

    def test_mixed_tamil_english(self):
        """Mixed Tamil/English text normalizes correctly."""
        mixed_nfd = unicodedata.normalize("NFD", "Volume 5: தமிழ் கட்டுரை")
        result = normalize_unicode(mixed_nfd)
        assert unicodedata.is_normalized("NFC", result)
        assert "Volume 5" in result
        assert "தமிழ்" in result

    def test_phrase_normalization(self):
        """Longer Tamil phrase normalizes consistently."""
        assert normalize_unicode(PHRASE_NFD) == PHRASE_NFC


class TestFuzzyMatchUnicode:
    """Tests that fuzzy matching produces identical scores for NFC/NFD input."""

    def test_same_score_nfc_vs_nfd(self):
        """Fuzzy match of NFC vs NFD forms of the same string scores 1.0."""
        score = fuzzy_match_score(TAMIL_NFC, TAMIL_NFD)
        assert score == 1.0

    def test_cross_form_matching(self):
        """Matching NFC query against NFD reference gives same score as NFC vs NFC."""
        reference = "பொன்னி"
        reference_nfd = unicodedata.normalize("NFD", reference)

        score_same = fuzzy_match_score(reference, reference)
        score_cross = fuzzy_match_score(reference, reference_nfd)
        assert score_same == score_cross

    def test_phrase_fuzzy_match(self):
        """Longer phrases match identically regardless of Unicode form."""
        score = fuzzy_match_score(PHRASE_NFC, PHRASE_NFD)
        assert score == 1.0


class TestEditDistanceUnicode:
    """Tests that _edit_distance_one handles NFC/NFD consistently."""

    def test_identical_nfc_nfd(self):
        """NFC and NFD forms of the same word are recognized as within edit distance 1."""
        # After normalization both forms are identical (distance 0 ≤ 1), so True
        assert _edit_distance_one(TAMIL_NFC, TAMIL_NFD)

    def test_one_char_diff_across_forms(self):
        """One-char difference detected even when forms differ."""
        a_nfc = "தமிழ்"
        b_nfc = "தமிள்"  # ழ → ள  (one character difference)
        b_nfd = unicodedata.normalize("NFD", b_nfc)
        # Should give the same result regardless of form
        assert _edit_distance_one(a_nfc, b_nfc) == _edit_distance_one(a_nfc, b_nfd)


class TestCacheKeyUnicode:
    """Tests that cache keys are identical for NFC/NFD representations."""

    def test_cache_key_consistency(self):
        """Same question in NFC and NFD produces the same cache key."""
        from cache import ResponseCache

        key_nfc = ResponseCache._make_key(PHRASE_NFC)
        key_nfd = ResponseCache._make_key(PHRASE_NFD)
        assert key_nfc == key_nfd

    def test_cache_hit_across_forms(self):
        """A cached NFC response is retrieved by an NFD query."""
        from cache import ResponseCache

        cache = ResponseCache(max_size=10, ttl_seconds=300)
        result = {"answer": "test answer", "sources": []}

        cache.put(PHRASE_NFC, result)
        retrieved = cache.get(PHRASE_NFD)
        assert retrieved is not None
        assert retrieved["answer"] == "test answer"


class TestEmbeddingUnicode:
    """Tests that embedding functions normalize Unicode before encoding."""

    def test_dense_embed_normalizes(self):
        """dense_embed_query normalizes input before encoding."""
        from embeddings import dense_embed_query

        # Both forms should call encode with the same NFC string
        with patch("embeddings.get_embed_model") as mock_model:
            import numpy as np
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.zeros(384)
            mock_model.return_value = mock_instance

            dense_embed_query(TAMIL_NFD)

            call_args = mock_instance.encode.call_args
            encoded_text = call_args[0][0]
            # The text passed to encode should be NFC-normalized
            assert unicodedata.is_normalized("NFC", encoded_text)

    def test_sparse_embed_normalizes(self):
        """sparse_embed normalizes input before tokenizing."""
        from embeddings import sparse_embed

        result_nfc = sparse_embed(TAMIL_NFC)
        result_nfd = sparse_embed(TAMIL_NFD)

        # Same indices and values for NFC vs NFD input
        assert sorted(result_nfc.indices) == sorted(result_nfd.indices)
        assert sorted(result_nfc.values) == sorted(result_nfd.values)


class TestTruncateQueryUnicode:
    """Tests that truncate_query applies NFC normalization."""

    def test_nfd_input_normalized(self):
        """truncate_query normalizes NFD input to NFC."""
        from hybrid_search import truncate_query

        result = truncate_query(PHRASE_NFD)
        assert unicodedata.is_normalized("NFC", result)
        assert result == PHRASE_NFC

    def test_long_nfd_query_normalized_before_truncation(self):
        """Long NFD query is first normalized, then truncated."""
        from hybrid_search import truncate_query

        # Build a long NFD string
        long_nfd = unicodedata.normalize("NFD", "தமிழ் " * 200)
        result = truncate_query(long_nfd, max_length=50)
        assert len(result) <= 50
        assert unicodedata.is_normalized("NFC", result)


class TestCsvQueryUnicode:
    """Tests that CSV query functions normalize author names."""

    def test_normalize_author_name_nfd(self):
        """normalize_author_name converts NFD Tamil names to NFC."""
        from csv_queries import normalize_author_name

        name_nfd = unicodedata.normalize("NFD", "கல்கி")
        result = normalize_author_name(name_nfd)
        assert unicodedata.is_normalized("NFC", result)
