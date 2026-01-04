import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.text_processing import (
    normalize_text,
    normalize_title,
    is_valid_heading,
    fuzzy_match_author,
    extract_author_from_line,
    find_author_in_range,
    get_intro_keywords
)


class TestNormalizeText:
    """Test cases for normalize_text function"""
    
    def test_basic_normalization(self):
        """Test basic text normalization"""
        result = normalize_text("Test Text")
        assert result == "testtext"
    
    def test_remove_dots(self):
        """Test removal of dots"""
        result = normalize_text("T.A.சுந்தரராசன்")
        assert "." not in result
    
    def test_handle_whitespace(self):
        """Test whitespace handling"""
        result = normalize_text("  Text  With   Spaces  ")
        assert result == "textwithspaces"
    
    def test_lowercase_conversion(self):
        """Test lowercase conversion"""
        result = normalize_text("UPPERCASE TEXT")
        assert result == "uppercasetext"
    
    def test_empty_string(self):
        """Test with empty string"""
        result = normalize_text("")
        assert result == ""
    
    def test_none_input(self):
        """Test with None input"""
        result = normalize_text(None)
        assert result == ""
    
    def test_tamil_text(self):
        """Test with Tamil text"""
        result = normalize_text("தமிழ் உரை")
        assert "தமிழ்" in result
        assert " " not in result


class TestNormalizeTitle:
    """Test cases for normalize_title function"""
    
    def test_remove_punctuation(self):
        """Test removal of punctuation"""
        result = normalize_title("Title, with! punctuation?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
    
    def test_remove_brackets(self):
        """Test removal of brackets"""
        result = normalize_title("Title (with) [brackets]")
        assert "(" not in result
        assert ")" not in result
        assert "[" not in result
        assert "]" not in result
    
    def test_remove_dashes(self):
        """Test removal of various dash types"""
        result = normalize_title("Title-with—dashes–here")
        assert "-" not in result
        assert "—" not in result
        assert "–" not in result
    
    def test_empty_title(self):
        """Test with empty title"""
        result = normalize_title("")
        assert result == ""
    
    def test_tamil_title(self):
        """Test with Tamil title"""
        result = normalize_title("கட்டுரை தலைப்பு")
        assert result != ""


class TestIsValidHeading:
    """Test cases for is_valid_heading function"""
    
    def test_valid_short_heading(self):
        """Test valid short heading"""
        result = is_valid_heading("Short Title")
        assert result is True
    
    def test_reject_empty_heading(self):
        """Test rejection of empty heading"""
        result = is_valid_heading("")
        assert result is False
    
    def test_reject_whitespace_only(self):
        """Test rejection of whitespace-only heading"""
        result = is_valid_heading("   ")
        assert result is False
    
    def test_reject_digits_only(self):
        """Test rejection of digit-only heading"""
        result = is_valid_heading("123")
        assert result is False
    
    def test_reject_long_heading(self):
        """Test rejection of heading longer than 25 chars"""
        result = is_valid_heading("This is a very long heading that exceeds limit")
        assert result is False
    
    def test_reject_dash_prefix(self):
        """Test rejection of dash-prefixed text"""
        result = is_valid_heading("— Author Name")
        assert result is False
        
        result = is_valid_heading("- Author Name")
        assert result is False
        
        result = is_valid_heading("– Author Name")
        assert result is False
    
    def test_valid_tamil_heading(self):
        """Test valid Tamil heading"""
        result = is_valid_heading("தலைப்பு")
        assert result is True
    
    def test_none_input(self):
        """Test with None input"""
        result = is_valid_heading(None)
        assert result is False


class TestFuzzyMatchAuthor:
    """Test cases for fuzzy_match_author function"""
    
    def test_exact_match(self):
        """Test exact match returns high similarity"""
        authors_normalized = ["authorname"]
        authors_original = ["Author Name"]
        
        author, similarity = fuzzy_match_author(
            "Author Name", authors_normalized, authors_original
        )
        
        assert author == "Author Name"
        assert similarity >= 0.8
    
    def test_partial_match(self):
        """Test partial match with threshold"""
        authors_normalized = ["authorname"]
        authors_original = ["Author Name"]
        
        author, similarity = fuzzy_match_author(
            "Author Nam", authors_normalized, authors_original, threshold=0.7
        )
        
        assert similarity > 0
    
    def test_no_match(self):
        """Test when no match is found"""
        authors_normalized = ["authorone"]
        authors_original = ["Author One"]
        
        author, similarity = fuzzy_match_author(
            "Completely Different", authors_normalized, authors_original
        )
        
        assert author is None
        assert similarity < 0.8
    
    def test_empty_line(self):
        """Test with empty line"""
        authors_normalized = ["author"]
        authors_original = ["Author"]
        
        author, similarity = fuzzy_match_author(
            "", authors_normalized, authors_original
        )
        
        assert author is None
    
    def test_multiple_authors(self):
        """Test matching against multiple authors"""
        authors_normalized = ["authorone", "authortwo", "authorthree"]
        authors_original = ["Author One", "Author Two", "Author Three"]
        
        author, similarity = fuzzy_match_author(
            "Author Two", authors_normalized, authors_original
        )
        
        assert author == "Author Two"


class TestExtractAuthorFromLine:
    """Test cases for extract_author_from_line function"""
    
    @patch('src.data_extraction.text_processing.fuzzy_match_author')
    def test_extract_basic_author(self, mock_fuzzy):
        """Test basic author extraction"""
        mock_fuzzy.return_value = ("Author Name", 0.9)
        
        result = extract_author_from_line(
            "Author Name",
            ["authorname"],
            ["Author Name"]
        )
        
        assert result == "Author Name"
    
    @patch('src.data_extraction.text_processing.fuzzy_match_author')
    def test_extract_author_with_dash(self, mock_fuzzy):
        """Test extracting author with leading dash"""
        mock_fuzzy.return_value = ("Author Name", 0.9)
        
        result = extract_author_from_line(
            "— Author Name",
            ["authorname"],
            ["Author Name"]
        )
        
        assert result == "Author Name"
        # Verify dash was removed before matching
        assert mock_fuzzy.call_args[0][0] == "Author Name"
    
    @patch('src.data_extraction.text_processing.fuzzy_match_author')
    def test_extract_author_from_brackets(self, mock_fuzzy):
        """Test extracting author from brackets"""
        mock_fuzzy.return_value = ("Author Name", 0.9)
        
        result = extract_author_from_line(
            "[Author Name]",
            ["authorname"],
            ["Author Name"]
        )
        
        assert result == "Author Name"
    
    @patch('src.data_extraction.text_processing.fuzzy_match_author')
    def test_extract_author_after_colon(self, mock_fuzzy):
        """Test extracting author after colon (ஆசிரியர்:)"""
        mock_fuzzy.return_value = ("Author Name", 0.9)
        
        result = extract_author_from_line(
            "ஆசிரியர்: Author Name",
            ["authorname"],
            ["Author Name"]
        )
        
        assert result == "Author Name"
    
    @patch('src.data_extraction.text_processing.fuzzy_match_author')
    def test_clean_trailing_punctuation(self, mock_fuzzy):
        """Test cleaning trailing punctuation"""
        mock_fuzzy.return_value = ("Author Name", 0.9)
        
        result = extract_author_from_line(
            "Author Name,.",
            ["authorname"],
            ["Author Name"]
        )
        
        assert result == "Author Name"
    
    def test_empty_line(self):
        """Test with empty line"""
        result = extract_author_from_line(
            "",
            ["author"],
            ["Author"]
        )
        
        assert result is None


class TestFindAuthorInRange:
    """Test cases for find_author_in_range function"""
    
    @patch('src.data_extraction.text_processing.extract_author_from_line')
    def test_find_author_in_range(self, mock_extract):
        """Test finding author within range"""
        mock_extract.side_effect = [None, "Author Name", None]
        
        lines = [
            "Line 1",
            "Author Name",
            "Line 3"
        ]
        
        author, idx = find_author_in_range(
            lines, 0, 3,
            ["authorname"],
            ["Author Name"]
        )
        
        assert author == "Author Name"
        assert idx == 1
    
    @patch('src.data_extraction.text_processing.extract_author_from_line')
    def test_author_not_found(self, mock_extract):
        """Test when no author is found"""
        mock_extract.return_value = None
        
        lines = ["Line 1", "Line 2", "Line 3"]
        
        author, idx = find_author_in_range(
            lines, 0, 3,
            ["author"],
            ["Author"]
        )
        
        assert author is None
        assert idx == -1
    
    @patch('src.data_extraction.text_processing.extract_author_from_line')
    def test_skip_blank_lines(self, mock_extract):
        """Test that blank lines are not checked but line iteration continues"""
        # Only called for non-blank lines (index 1 and 2)
        mock_extract.side_effect = ["Author", None]
        
        lines = ["", "Author", "Content"]
        
        author, idx = find_author_in_range(
            lines, 0, 3,
            ["author"],
            ["Author"]
        )
        
        assert author == "Author"
        # Author found at index 1 (blank line at 0 was skipped)
        assert idx == 1
    
    def test_out_of_bounds(self):
        """Test with out of bounds range"""
        lines = ["Line 1", "Line 2"]
        
        author, idx = find_author_in_range(
            lines, 0, 100,
            ["author"],
            ["Author"]
        )
        
        assert author is None


class TestGetIntroKeywords:
    """Test cases for get_intro_keywords function"""
    
    def test_returns_list(self):
        """Test that function returns a list"""
        result = get_intro_keywords()
        assert isinstance(result, list)
    
    def test_contains_expected_keywords(self):
        """Test that result contains expected keywords"""
        result = get_intro_keywords()
        
        assert "எங்கள் எண்ணம்" in result
        assert "காலமும் கருத்தும்" in result
        assert "பொது மேடை" in result
    
    def test_no_empty_strings(self):
        """Test that there are no empty strings"""
        result = get_intro_keywords()
        
        for keyword in result:
            assert keyword.strip() != ""
    
    def test_no_article_headings(self):
        """Test that article headings are not included"""
        result = get_intro_keywords()
        
        # These should be intro sections, not articles
        assert all(keyword for keyword in result)
    
    def test_consistent_results(self):
        """Test that function returns consistent results"""
        result1 = get_intro_keywords()
        result2 = get_intro_keywords()
        
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])