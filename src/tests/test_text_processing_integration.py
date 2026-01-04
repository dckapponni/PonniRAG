"""Integration tests for text_processing - calls real functions"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.text_processing import (
    normalize_text, normalize_title, is_valid_heading,
    fuzzy_match_author, extract_author_from_line,
    find_author_in_range, get_intro_keywords
)

class TestTextProcessingIntegration:
    def test_normalize_text_real(self):
        assert normalize_text("Test Text") == "testtext"
        assert normalize_text("T.A.Name") == "taname"
        assert normalize_text("  Spaces  ") == "spaces"
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_normalize_title_real(self):
        result = normalize_title("Title, with! punctuation?")
        assert result
        assert "," not in result
    
    def test_is_valid_heading_real(self):
        assert is_valid_heading("Valid") is True
        assert is_valid_heading("") is False
        assert is_valid_heading("   ") is False
        assert is_valid_heading("123") is False
        assert is_valid_heading("— Author") is False
        assert is_valid_heading(None) is False
    
    def test_fuzzy_match_real(self):
        authors_norm = ["authorone"]
        authors_orig = ["Author One"]
        author, score = fuzzy_match_author("Author One", authors_norm, authors_orig)
        assert author == "Author One"
        assert score >= 0.8
    
    def test_extract_author_real(self):
        authors_norm = ["authorone"]
        authors_orig = ["Author One"]
        assert extract_author_from_line("Author One", authors_norm, authors_orig) == "Author One"
        assert extract_author_from_line("— Author One", authors_norm, authors_orig) == "Author One"
        assert extract_author_from_line("", authors_norm, authors_orig) is None
    
    def test_find_author_in_range_real(self):
        lines = ["Line 1", "Author One", "Line 3"]
        authors_norm = ["authorone"]
        authors_orig = ["Author One"]
        author, idx = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
        assert author == "Author One"
        assert idx == 1
    
    def test_get_intro_keywords_real(self):
        keywords = get_intro_keywords()
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "எங்கள் எண்ணம்" in keywords

if __name__ == "__main__":
    pytest.main([__file__, "-v"])