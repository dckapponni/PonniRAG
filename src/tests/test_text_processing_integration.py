import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.text_processing import (
    normalize_text, is_valid_heading,
    fuzzy_match_author, extract_author_from_line,
    get_intro_keywords
)


class TestTextProcessingIntegration:
    """Integration test suite for text_processing module.
    
    These tests call actual functions without mocking to verify real-world
    behavior and ensure all components work correctly together. Each test
    uses realistic inputs and validates expected outputs.
    """
    
    def test_normalize_text_real(self):
        """Test normalize_text with real inputs and expected outputs.
        
        Verifies that text normalization correctly:
        - Converts text to lowercase
        - Removes dots from text (e.g., initials)
        - Normalizes whitespace to single spaces
        - Handles empty strings
        - Handles None values
        
        This validates the core text cleaning functionality used throughout
        the application for consistent text comparison.
        """
        assert normalize_text("Test Text") == "testtext"
        assert normalize_text("T.A.Name") == "taname"
        assert normalize_text("  Spaces  ") == "spaces"
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_is_valid_heading_real(self):
        """Test is_valid_heading with various real-world heading formats.
        
        Verifies heading validation logic correctly:
        - Accepts simple valid headings (returns True)
        - Rejects empty strings (returns False)
        - Rejects whitespace-only strings (returns False)
        - Rejects numeric-only strings (returns False)
        - Rejects headings starting with em dash — (returns False)
        - Rejects None values (returns False)
        
        This ensures that only legitimate headings are processed, filtering
        out common false positives like page numbers and attribution lines.
        """
        assert is_valid_heading("Valid") is True
        assert is_valid_heading("") is False
        assert is_valid_heading("   ") is False
        assert is_valid_heading("123") is False
        assert is_valid_heading("— Author") is False
        assert is_valid_heading(None) is False
    
    def test_fuzzy_match_real(self):
        """Test fuzzy_match_author with real author name matching.
        
        Verifies that fuzzy matching:
        - Correctly identifies exact matches
        - Returns the original (non-normalized) author name
        - Produces high similarity scores (>= 0.8) for matches
        
        The fuzzy matching is essential for finding author attributions
        that may have slight variations in formatting or spelling across
        different parts of a document.
        """
        authors_norm = ["authorone"]
        authors_orig = ["Author One"]
        author, score = fuzzy_match_author("Author One", authors_norm, authors_orig)
        assert author == "Author One"
        assert score >= 0.8
    
    def test_extract_author_real(self):
        """Test extract_author_from_line with real author line formats.
        
        Verifies that author extraction:
        - Identifies plain author names (returns "Author One")
        - Handles em dash prefixes: "— Author One" (returns "Author One")
        - Returns None for empty lines
        
        This validates the core logic for extracting author names from
        various common formatting patterns found in Tamil literary texts,
        including attribution lines and bylines.
        """
        authors_norm = ["authorone"]
        authors_orig = ["Author One"]
        assert extract_author_from_line("Author One", authors_norm, authors_orig) == "Author One"
        assert extract_author_from_line("— Author One", authors_norm, authors_orig) == "Author One"
        assert extract_author_from_line("", authors_norm, authors_orig) is None
    
    def test_get_intro_keywords_real(self):
        """Test get_intro_keywords returns valid Tamil introduction keywords.
        
        Verifies that the function:
        - Returns a list data type
        - Returns at least one keyword (non-empty list)
        - Contains expected Tamil keywords like "எங்கள் எண்ணம்"
        
        These keywords are used to identify introductory sections in Tamil
        literary magazines and anthologies, helping to distinguish editorial
        content from creative works.
        """
        keywords = get_intro_keywords()
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "எங்கள் எண்ணம்" in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])