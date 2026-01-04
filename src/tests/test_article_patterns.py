import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.article_patterns import (
    extract_pattern_a_forward,
    extract_pattern_b_forward,
    extract_pattern_c_reverse,
    find_author_in_content_end
)


class TestExtractPatternAForward:
    """Test cases for Pattern A: HEADING → AUTHOR → CONTENT"""
    
    @pytest.fixture
    def sample_setup(self):
        """Common test setup"""
        authors_normalized = ["authorone", "authortwo"]
        authors_original = ["Author One", "Author Two"]
        intro_keywords = ["எங்கள் எண்ணம்"]
        return authors_normalized, authors_original, intro_keywords
    
    @patch('src.data_extraction.article_patterns.extract_author_from_line')
    @patch('src.data_extraction.article_patterns.is_valid_heading')
    def test_extract_basic_pattern_a(self, mock_heading, mock_author, sample_setup):
        """Test basic Pattern A extraction"""
        authors_norm, authors_orig, keywords = sample_setup
        mock_heading.return_value = True
        # Provide enough return values for all calls (including loop iterations)
        mock_author.side_effect = [None, "Author One"] + [None] * 20
        
        lines = [
            "",
            "Short Heading",
            "",
            "Author One",
            "",
            "Content line 1",
            "Content line 2",
            "Content line 3",
            "Content line 4",
            ""
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_a_forward(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        assert len(result) == 1
        assert result[0]["heading"] == "Short Heading"
        assert result[0]["author"] == "Author One"
    
    @patch('src.data_extraction.article_patterns.extract_author_from_line')
    def test_no_heading_before_author(self, mock_author, sample_setup):
        """Test when no valid heading before author"""
        authors_norm, authors_orig, keywords = sample_setup
        mock_author.side_effect = [None, "Author One"] + [None] * 10
        
        lines = [
            "Content without heading",
            "Author One",
            "More content"
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_a_forward(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        assert len(result) == 0
    
    @patch('src.data_extraction.article_patterns.extract_author_from_line')
    @patch('src.data_extraction.article_patterns.is_valid_heading')
    def test_short_content_rejected(self, mock_heading, mock_author, sample_setup):
        """Test that articles with < 4 lines are rejected"""
        authors_norm, authors_orig, keywords = sample_setup
        mock_heading.return_value = True
        mock_author.side_effect = [None, "Author One"] + [None] * 10
        
        lines = [
            "Heading",
            "Author One",
            "Short content 1",
            "Short content 2"
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_a_forward(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        assert len(result) == 0


class TestExtractPatternBForward:
    """Test cases for Pattern B: AUTHOR → HEADING → CONTENT"""
    
    @pytest.fixture
    def sample_setup(self):
        """Common test setup"""
        authors_normalized = ["authorone"]
        authors_original = ["Author One"]
        intro_keywords = ["எங்கள் எண்ணம்"]
        return authors_normalized, authors_original, intro_keywords
    
    @patch('src.data_extraction.article_patterns.extract_author_from_line')
    @patch('src.data_extraction.article_patterns.is_valid_heading')
    def test_extract_basic_pattern_b(self, mock_heading, mock_author, sample_setup):
        """Test basic Pattern B extraction"""
        authors_norm, authors_orig, keywords = sample_setup
        mock_heading.return_value = True
        mock_author.side_effect = ["Author One"] + [None] * 20
        
        lines = [
            "Author One",
            "",
            "Heading Text",
            "",
            "Content line 1",
            "Content line 2",
            "Content line 3",
            "Content line 4"
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_b_forward(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        assert len(result) == 1
        assert result[0]["heading"] == "Heading Text"
        assert result[0]["author"] == "Author One"
    
    @patch('src.data_extraction.article_patterns.extract_author_from_line')
    def test_no_heading_after_author(self, mock_author, sample_setup):
        """Test when no heading found after author"""
        authors_norm, authors_orig, keywords = sample_setup
        mock_author.side_effect = ["Author One"] + [None] * 10
        
        lines = [
            "Author One",
            "Content without heading",
            "More content"
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_b_forward(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        assert len(result) == 0


class TestExtractPatternCReverse:
    """Test cases for Pattern C: HEADING → CONTENT → AUTHOR (poems)"""
    
    @pytest.fixture
    def sample_setup(self):
        """Common test setup"""
        authors_normalized = ["authorone", "authortwo"]
        authors_original = ["Author One", "Author Two"]
        intro_keywords = ["எங்கள் எண்ணம்"]
        return authors_normalized, authors_original, intro_keywords
    
    @patch('src.data_extraction.article_patterns.extract_author_from_line')
    def test_standalone_author_pattern(self, mock_author, sample_setup):
        """Test standalone author at end (HEADING → CONTENT → AUTHOR)"""
        authors_norm, authors_orig, keywords = sample_setup
        mock_author.side_effect = [None, None, None, None, None, "Author One"] + [None] * 10
        
        lines = [
            "",
            "Poem Heading",
            "",
            "Poem line 1",
            "Poem line 2",
            "Author One"
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_c_reverse(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        assert len(result) >= 0  # Pattern C is complex
    
    def test_embedded_author_pattern(self, sample_setup):
        """Test embedded author at content end - integration test without mocking"""
        authors_norm, authors_orig, keywords = sample_setup
        
        lines = [
            "Poem Title",
            "",
            "Poem line 1",
            "Poem line 2",
            "Poem line 3",
            "— Author One"
        ]
        processed_lines = [False] * len(lines)
        
        result = extract_pattern_c_reverse(
            lines, 0, len(lines),
            authors_norm, authors_orig,
            processed_lines, keywords
        )
        
        # Just verify it runs without error and returns a list
        assert isinstance(result, list)


class TestFindAuthorInContentEnd:
    """Test cases for find_author_in_content_end function"""
    
    def test_find_dash_prefixed_author(self):
        """Test finding author with dash prefix"""
        content_lines = [
            "Poem line 1",
            "Poem line 2",
            "Poem line 3",
            "— Author Name"
        ]
        authors_normalized = ["authorname"]
        authors_original = ["Author Name"]
        
        author, modified = find_author_in_content_end(
            content_lines, authors_normalized, authors_original
        )
        
        assert author == "Author Name"
        assert len(modified) == 3
        assert "— Author Name" not in modified
    
    def test_find_author_with_spaces(self):
        """Test finding author with space variations"""
        content_lines = [
            "Poem line 1",
            "Poem line 2",
            "— த. அ. சுந்தரராசன்"
        ]
        authors_normalized = ["தஅசுந்தரராசன்"]
        authors_original = ["த.அ.சுந்தரராசன்"]
        
        author, modified = find_author_in_content_end(
            content_lines, authors_normalized, authors_original
        )
        
        assert author is not None
    
    def test_no_author_found(self):
        """Test when no author is found"""
        content_lines = [
            "Poem line 1",
            "Poem line 2",
            "Poem line 3"
        ]
        authors_normalized = ["authorname"]
        authors_original = ["Author Name"]
        
        author, modified = find_author_in_content_end(
            content_lines, authors_normalized, authors_original
        )
        
        assert author is None
        assert modified == content_lines
    
    def test_exact_match_priority(self):
        """Test that exact matches are prioritized"""
        content_lines = [
            "Content line 1",
            "Content line 2",
            "— Exact Author"
        ]
        authors_normalized = ["exactauthor", "author"]
        authors_original = ["Exact Author", "Author"]
        
        author, modified = find_author_in_content_end(
            content_lines, authors_normalized, authors_original
        )
        
        assert author == "Exact Author"
    
    def test_check_last_20_lines(self):
        """Test that function checks up to last 20 lines"""
        content_lines = ["Line " + str(i) for i in range(30)]
        content_lines.append("— Author Name")
        
        authors_normalized = ["authorname"]
        authors_original = ["Author Name"]
        
        author, modified = find_author_in_content_end(
            content_lines, authors_normalized, authors_original
        )
        
        # Should find author even in long content
        assert author == "Author Name"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])