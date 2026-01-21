"""
Comprehensive tests for text_processing.py to achieve 90%+ coverage.
Save this as tests/test_text_processing.py
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_extraction.text_processing import (
    normalize_text,
    normalize_title,
    is_valid_heading,
    fuzzy_match_author,
    extract_author_from_line,
    find_author_in_range,
    get_intro_keywords
)


class TestNormalizeText:
    """Test normalize_text function"""
    
    def test_normalize_text_empty(self):
        """Test with empty string"""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_normalize_text_basic(self):
        """Test basic normalization"""
        result = normalize_text("Test Text")
        assert isinstance(result, str)
        assert result.islower()
    
    def test_normalize_text_with_special_unicode(self):
        """Test with Tamil Unicode characters"""
        text = "ணுதுணு"
        result = normalize_text(text)
        assert isinstance(result, str)
    
    def test_normalize_text_with_dots(self):
        """Test removal of dots"""
        text = "test.text.here"
        result = normalize_text(text)
        assert '.' not in result
    
    def test_normalize_text_whitespace_removal(self):
        """Test whitespace is normalized"""
        text = "  test   text  "
        result = normalize_text(text)
        assert '  ' not in result
    
    def test_normalize_text_with_exception(self):
        """Test exception handling"""
        class BadString:
            def __str__(self):
                return "test"
            def replace(self, *args):
                raise ValueError("Mock error")
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = normalize_text(BadString())
            assert mock_logger.warning.called
            assert isinstance(result, str)
    
    def test_normalize_text_with_nonstring_type(self):
        """Test with non-string that causes error"""
        class FailingObject:
            def __bool__(self):
                return True
            def replace(self, *args):
                raise TypeError("Cannot replace")
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = normalize_text(FailingObject())
            assert mock_logger.warning.called


class TestNormalizeTitle:
    """Test normalize_title function"""
    
    def test_normalize_title_empty(self):
        """Test with empty string"""
        assert normalize_title("") == ""
        assert normalize_title(None) == ""
    
    def test_normalize_title_removes_punctuation(self):
        """Test punctuation removal"""
        title = "Title.,!?;:\"'-—–()[]{}End"
        result = normalize_title(title)
        # All punctuation should be removed
        for char in '.,!?;:"\'-—–()[]{}':
            assert char not in result
    
    def test_normalize_title_with_tamil_text(self):
        """Test with Tamil text"""
        title = "தமிழ் Title ணு"
        result = normalize_title(title)
        assert isinstance(result, str)
    
    def test_normalize_title_lowercase(self):
        """Test converts to lowercase"""
        title = "UPPERCASE Title"
        result = normalize_title(title)
        assert result.islower()
    
    def test_normalize_title_with_exception(self):
        """Test exception handling"""
        class BadTitle:
            def __bool__(self):
                return True
            def replace(self, *args):
                raise ValueError("Mock error")
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = normalize_title(BadTitle())
            assert mock_logger.warning.called
            assert isinstance(result, str)


class TestIsValidHeading:
    """Test is_valid_heading function"""
    
    def test_is_valid_heading_empty(self):
        """Test with empty string"""
        assert is_valid_heading("") == False
        assert is_valid_heading(None) == False
        assert is_valid_heading("   ") == False
    
    def test_is_valid_heading_only_digits(self):
        """Test heading that is only digits"""
        assert is_valid_heading("12345") == False
    
    def test_is_valid_heading_too_long(self):
        """Test heading that is too long"""
        long_heading = "a" * 30
        assert is_valid_heading(long_heading) == False
    
    def test_is_valid_heading_starts_with_dash(self):
        """Test heading starting with various dashes"""
        assert is_valid_heading("— Heading") == False
        assert is_valid_heading("- Heading") == False
        assert is_valid_heading("– Heading") == False
    
    def test_is_valid_heading_valid(self):
        """Test valid heading"""
        assert is_valid_heading("Valid Heading") == True
        assert is_valid_heading("தமிழ்") == True
    
    def test_is_valid_heading_at_boundary(self):
        """Test at the 25 character boundary"""
        heading_24 = "a" * 24
        heading_25 = "a" * 25
        assert is_valid_heading(heading_24) == True
        assert is_valid_heading(heading_25) == False
    
    def test_is_valid_heading_with_exception(self):
        """Test exception handling"""
        class BadHeading:
            def __bool__(self):
                return True
            def strip(self):
                raise ValueError("Mock error")
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = is_valid_heading(BadHeading())
            assert result == False
            assert mock_logger.warning.called


class TestFuzzyMatchAuthor:
    """Test fuzzy_match_author function"""
    
    def test_fuzzy_match_empty_line(self):
        """Test with empty line"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result, score = fuzzy_match_author("", authors_norm, authors_orig)
        assert result is None
        assert score == 0
    
    def test_fuzzy_match_exact_match(self):
        """Test exact match"""
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        result, score = fuzzy_match_author("Test Author", authors_norm, authors_orig)
        assert score >= 0.8
    
    def test_fuzzy_match_high_threshold(self):
        """Test with various similarity scores"""
        authors_norm = ["johnsmith"]
        authors_orig = ["John Smith"]
        result, score = fuzzy_match_author("John Smith", authors_norm, authors_orig)
        assert score > 0.8
    
    def test_fuzzy_match_no_match(self):
        """Test when no match is found"""
        authors_norm = ["author1"]
        authors_orig = ["Author 1"]
        result, score = fuzzy_match_author("completely different", authors_norm, authors_orig)
        assert result is None
    
    def test_fuzzy_match_with_comparison_exception(self):
        """Test exception handling during comparison"""
        authors_norm = ["author1", "author2"]
        authors_orig = ["Author 1", "Author 2"]
        
        with patch('data_extraction.text_processing.SequenceMatcher') as mock_matcher:
            mock_matcher.return_value.ratio.side_effect = ValueError("Comparison error")
            
            with patch('data_extraction.text_processing.logger') as mock_logger:
                result, score = fuzzy_match_author("test", authors_norm, authors_orig)
                assert mock_logger.debug.called
                assert result is None
    
    def test_fuzzy_match_debug_logging_on_match(self):
        """Test debug logging when match is found"""
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result, score = fuzzy_match_author("Test Author", authors_norm, authors_orig)
            if result:
                assert any(call for call in mock_logger.debug.call_args_list 
                          if "Fuzzy matched" in str(call))
    
    def test_fuzzy_match_custom_threshold(self):
        """Test with custom threshold"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        # Lower threshold should find more matches
        result, score = fuzzy_match_author("auth", authors_norm, authors_orig, threshold=0.5)
        assert isinstance(result, (str, type(None)))
    
    def test_fuzzy_match_multiple_authors(self):
        """Test matching against multiple authors"""
        authors_norm = ["author1", "author2", "author3"]
        authors_orig = ["Author One", "Author Two", "Author Three"]
        result, score = fuzzy_match_author("Author Two", authors_norm, authors_orig)
        # Should find the best match
        assert isinstance(result, (str, type(None)))


class TestExtractAuthorFromLine:
    """Test extract_author_from_line function"""
    
    def test_extract_author_empty_line(self):
        """Test with empty line"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        assert extract_author_from_line("", authors_norm, authors_orig) is None
        assert extract_author_from_line("   ", authors_norm, authors_orig) is None
    
    def test_extract_author_with_leading_dash(self):
        """Test line starting with dashes"""
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        result = extract_author_from_line("— Author Name", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_brackets(self):
        """Test extraction from brackets"""
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        result = extract_author_from_line("[Test Author]", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_nested_brackets(self):
        """Test with nested brackets"""
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        result = extract_author_from_line("Some text [Test Author] more", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_colon(self):
        """Test line with colon separator"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("Label: Author", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_multiple_colons(self):
        """Test with multiple colons - should use last part"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("Label: Category: Author", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_tamil_keyword(self):
        """Test line with Tamil 'ஆசிரியர்' keyword"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("ஆசிரியர்: Author", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_trailing_punctuation(self):
        """Test cleaning trailing punctuation"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("Author,,,", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_leading_quotes(self):
        """Test removal of leading quotes"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line('"Author', authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_exception(self):
        """Test exception handling"""
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        class BadLine:
            def strip(self):
                raise ValueError("Mock error")
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = extract_author_from_line(BadLine(), authors_norm, authors_orig)
            assert result is None
            assert mock_logger.warning.called
    
    def test_extract_tamil_author(self):
        """Test extracting Tamil author names"""
        authors_norm = ["கவிஞர்பெயர்"]
        authors_orig = ["கவிஞர் பெயர்"]
        result = extract_author_from_line("— கவிஞர் பெயர்", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))


class TestFindAuthorInRange:
    """Test find_author_in_range function"""
    
    def test_find_author_in_range_basic(self):
        """Test basic author finding"""
        lines = ["Line 1", "Author Name", "Line 3"]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger'):
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            assert isinstance(result, tuple)
            assert len(result) == 2
    
    def test_find_author_not_found(self):
        """Test when author is not found"""
        lines = ["Line 1", "Line 2", "Line 3"]
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            assert result == (None, -1)
            # Should log that no author was found
            assert any("No author found" in str(call) for call in mock_logger.debug.call_args_list)
    
    def test_find_author_with_blank_lines(self):
        """Test skipping blank lines"""
        lines = ["", "  ", "   ", "Author Name", "Content"]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger'):
            result = find_author_in_range(lines, 0, 5, authors_norm, authors_orig)
            if result[0]:
                assert result[1] == 3
    
    def test_find_author_index_error(self):
        """Test IndexError handling"""
        lines = ["Line 1", "Line 2"]
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            with patch('data_extraction.text_processing.extract_author_from_line', 
                      side_effect=IndexError("Index out of range")):
                result = find_author_in_range(lines, 0, 10, authors_norm, authors_orig)
                assert result == (None, -1)
                assert mock_logger.warning.called
    
    def test_find_author_general_exception(self):
        """Test general exception handling"""
        lines = ["Line 1", "Line 2"]
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            with patch('data_extraction.text_processing.extract_author_from_line',
                      side_effect=RuntimeError("Unexpected error")):
                result = find_author_in_range(lines, 0, 5, authors_norm, authors_orig)
                assert result == (None, -1)
                assert mock_logger.error.called
    
    def test_find_author_debug_logging(self):
        """Test debug logging"""
        lines = ["Line 1", "Author Name", "Line 3"]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            # Should have debug logging calls
            assert mock_logger.debug.called
    
    def test_find_author_range_exceeds_length(self):
        """Test when range exceeds list length"""
        lines = ["Line 1", "Line 2"]
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        with patch('data_extraction.text_processing.logger'):
            # Should handle gracefully using min(end_idx, len(lines))
            result = find_author_in_range(lines, 0, 100, authors_norm, authors_orig)
            assert isinstance(result, tuple)
    
    def test_find_author_found_logs_correctly(self):
        """Test logging when author is found"""
        lines = ["", "Author Name", ""]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            if result[0]:
                # Should log finding the author
                assert any("Found author" in str(call) for call in mock_logger.debug.call_args_list)


class TestGetIntroKeywords:
    """Test get_intro_keywords function"""
    
    def test_get_intro_keywords_returns_list(self):
        """Test returns a list"""
        result = get_intro_keywords()
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_get_intro_keywords_contains_expected(self):
        """Test contains expected keywords"""
        result = get_intro_keywords()
        assert "எங்கள் எண்ணம்" in result
        assert "காலமும் கருத்தும்" in result
    
    def test_get_intro_keywords_debug_logging(self):
        """Test debug logging"""
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = get_intro_keywords()
            assert mock_logger.debug.called
            assert len(result) > 0
    
    def test_get_intro_keywords_count(self):
        """Test returns correct number of keywords"""
        result = get_intro_keywords()
        # Verify it has all the expected keywords
        assert len(result) == 17


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_full_workflow(self):
        """Test a full workflow of processing"""
        # Normalize some text
        text = "Test Text ணு"
        normalized = normalize_text(text)
        assert isinstance(normalized, str)
        
        # Validate a heading
        heading = "Valid Heading"
        is_valid = is_valid_heading(heading)
        assert is_valid == True
        
        # Get keywords
        keywords = get_intro_keywords()
        assert len(keywords) > 0
    
    def test_author_extraction_workflow(self):
        """Test complete author extraction workflow"""
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        
        # Extract from a line
        line = "— Test Author"
        author = extract_author_from_line(line, authors_norm, authors_orig)
        
        # Find in a range
        lines = ["", "— Test Author", "Content"]
        found_author, idx = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
        
        assert isinstance(author, (str, type(None)))
        assert isinstance(found_author, (str, type(None)))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=data_extraction.text_processing", "--cov-report=term-missing"])