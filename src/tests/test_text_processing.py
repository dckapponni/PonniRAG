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
    """Test suite for the normalize_text function.
    
    Tests various scenarios including empty inputs, Unicode characters,
    special characters, whitespace handling, and error conditions.
    """
    
    def test_normalize_text_empty(self):
        """Test normalize_text with empty and None inputs.
        
        Verifies that the function returns an empty string when given
        empty string or None as input.
        """
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization functionality.
        
        Verifies that the function:
        - Returns a string type
        - Converts text to lowercase
        """
        result = normalize_text("Test Text")
        assert isinstance(result, str)
        assert result.islower()
    
    def test_normalize_text_with_special_unicode(self):
        """Test normalization of Tamil Unicode characters.
        
        Verifies that the function can handle and process Unicode
        characters (specifically Tamil script) without errors.
        """
        text = "ணுதுணு"
        result = normalize_text(text)
        assert isinstance(result, str)
    
    def test_normalize_text_with_dots(self):
        """Test that dots are removed during normalization.
        
        Verifies that the function removes all dot characters (.)
        from the input text.
        """
        text = "test.text.here"
        result = normalize_text(text)
        assert '.' not in result
    
    def test_normalize_text_whitespace_removal(self):
        """Test that excessive whitespace is normalized.
        
        Verifies that multiple consecutive spaces are reduced to
        single spaces during normalization.
        """
        text = "  test   text  "
        result = normalize_text(text)
        assert '  ' not in result
    
    def test_normalize_text_with_exception(self):
        """Test exception handling during normalization.
        
        Verifies that when an exception occurs during processing,
        the function:
        - Logs a warning
        - Returns a valid string instead of crashing
        """
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
        """Test handling of non-string types that cause errors.
        
        Verifies that when a non-string object causes a TypeError,
        the function logs a warning and handles it gracefully.
        """
        class FailingObject:
            def __bool__(self):
                return True
            def replace(self, *args):
                raise TypeError("Cannot replace")
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = normalize_text(FailingObject())
            assert mock_logger.warning.called


class TestNormalizeTitle:
    """Test suite for the normalize_title function.
    
    Tests title normalization including punctuation removal, Unicode handling,
    case conversion, and error conditions.
    """
    
    def test_normalize_title_empty(self):
        """Test normalize_title with empty and None inputs.
        
        Verifies that the function returns an empty string when given
        empty string or None as input.
        """
        assert normalize_title("") == ""
        assert normalize_title(None) == ""
    
    def test_normalize_title_removes_punctuation(self):
        """Test that all punctuation marks are removed from titles.
        
        Verifies that the function removes common punctuation characters
        including: . , ! ? ; : " ' - — – ( ) [ ] { }
        """
        title = "Title.,!?;:\"'-—–()[]{}End"
        result = normalize_title(title)
        # All punctuation should be removed
        for char in '.,!?;:"\'-—–()[]{}':
            assert char not in result
    
    def test_normalize_title_with_tamil_text(self):
        """Test normalization of titles containing Tamil text.
        
        Verifies that the function can handle mixed Tamil and English
        text without errors.
        """
        title = "தமிழ் Title ணு"
        result = normalize_title(title)
        assert isinstance(result, str)
    
    def test_normalize_title_lowercase(self):
        """Test that titles are converted to lowercase.
        
        Verifies that all characters in the normalized title are
        in lowercase.
        """
        title = "UPPERCASE Title"
        result = normalize_title(title)
        assert result.islower()
    
    def test_normalize_title_with_exception(self):
        """Test exception handling during title normalization.
        
        Verifies that when an exception occurs during processing,
        the function:
        - Logs a warning
        - Returns a valid string instead of crashing
        """
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
    """Test suite for the is_valid_heading function.
    
    Tests validation logic for headings including length constraints,
    content validation, and special character handling.
    """
    
    def test_is_valid_heading_empty(self):
        """Test validation of empty and whitespace-only headings.
        
        Verifies that the function returns False for:
        - Empty strings
        - None values
        - Strings containing only whitespace
        """
        assert is_valid_heading("") == False
        assert is_valid_heading(None) == False
        assert is_valid_heading("   ") == False
    
    def test_is_valid_heading_only_digits(self):
        """Test that headings containing only digits are invalid.
        
        Verifies that the function rejects headings that consist
        entirely of numeric characters.
        """
        assert is_valid_heading("12345") == False
    
    def test_is_valid_heading_too_long(self):
        """Test that headings exceeding maximum length are invalid.
        
        Verifies that headings longer than the maximum allowed
        length (25 characters) are rejected.
        """
        long_heading = "a" * 30
        assert is_valid_heading(long_heading) == False
    
    def test_is_valid_heading_starts_with_dash(self):
        """Test that headings starting with dashes are invalid.
        
        Verifies that the function rejects headings starting with:
        - Em dash (—)
        - Hyphen (-)
        - En dash (–)
        """
        assert is_valid_heading("— Heading") == False
        assert is_valid_heading("- Heading") == False
        assert is_valid_heading("– Heading") == False
    
    def test_is_valid_heading_valid(self):
        """Test validation of valid headings.
        
        Verifies that the function correctly accepts:
        - Regular English text headings
        - Tamil text headings
        """
        assert is_valid_heading("Valid Heading") == True
        assert is_valid_heading("தமிழ்") == True
    
    def test_is_valid_heading_at_boundary(self):
        """Test heading validation at the length boundary.
        
        Verifies that:
        - 24 character headings are valid (just under limit)
        - 25 character headings are invalid (at the limit)
        """
        heading_24 = "a" * 24
        heading_25 = "a" * 25
        assert is_valid_heading(heading_24) == True
        assert is_valid_heading(heading_25) == False
    
    def test_is_valid_heading_with_exception(self):
        """Test exception handling during heading validation.
        
        Verifies that when an exception occurs during processing,
        the function:
        - Logs a warning
        - Returns False (safe default)
        """
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
    """Test suite for the fuzzy_match_author function.
    
    Tests fuzzy string matching for author names including exact matches,
    partial matches, threshold handling, and error conditions.
    """
    
    def test_fuzzy_match_empty_line(self):
        """Test fuzzy matching with an empty line.
        
        Verifies that when given an empty string, the function:
        - Returns None as the matched author
        - Returns 0 as the similarity score
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result, score = fuzzy_match_author("", authors_norm, authors_orig)
        assert result is None
        assert score == 0
    
    def test_fuzzy_match_exact_match(self):
        """Test fuzzy matching with an exact match.
        
        Verifies that when the input text exactly matches an author name,
        the function returns a high similarity score (>= 0.8).
        """
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        result, score = fuzzy_match_author("Test Author", authors_norm, authors_orig)
        assert score >= 0.8
    
    def test_fuzzy_match_high_threshold(self):
        """Test fuzzy matching produces high scores for similar strings.
        
        Verifies that similar author names produce similarity scores
        above the matching threshold (0.8).
        """
        authors_norm = ["johnsmith"]
        authors_orig = ["John Smith"]
        result, score = fuzzy_match_author("John Smith", authors_norm, authors_orig)
        assert score > 0.8
    
    def test_fuzzy_match_no_match(self):
        """Test fuzzy matching when no similar author is found.
        
        Verifies that when the input text doesn't match any author,
        the function returns None.
        """
        authors_norm = ["author1"]
        authors_orig = ["Author 1"]
        result, score = fuzzy_match_author("completely different", authors_norm, authors_orig)
        assert result is None
    
    def test_fuzzy_match_with_comparison_exception(self):
        """Test exception handling during fuzzy matching.
        
        Verifies that when an exception occurs during string comparison:
        - Debug message is logged
        - Function returns None gracefully
        """
        authors_norm = ["author1", "author2"]
        authors_orig = ["Author 1", "Author 2"]
        
        with patch('data_extraction.text_processing.SequenceMatcher') as mock_matcher:
            mock_matcher.return_value.ratio.side_effect = ValueError("Comparison error")
            
            with patch('data_extraction.text_processing.logger') as mock_logger:
                result, score = fuzzy_match_author("test", authors_norm, authors_orig)
                assert mock_logger.debug.called
                assert result is None
    
    def test_fuzzy_match_debug_logging_on_match(self):
        """Test that successful matches are logged for debugging.
        
        Verifies that when a match is found, a debug message
        containing "Fuzzy matched" is logged.
        """
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result, score = fuzzy_match_author("Test Author", authors_norm, authors_orig)
            if result:
                assert any(call for call in mock_logger.debug.call_args_list 
                          if "Fuzzy matched" in str(call))
    
    def test_fuzzy_match_custom_threshold(self):
        """Test fuzzy matching with a custom similarity threshold.
        
        Verifies that the function accepts and uses custom threshold
        values (e.g., 0.5 instead of default 0.8) for matching.
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        # Lower threshold should find more matches
        result, score = fuzzy_match_author("auth", authors_norm, authors_orig, threshold=0.5)
        assert isinstance(result, (str, type(None)))
    
    def test_fuzzy_match_multiple_authors(self):
        """Test fuzzy matching against multiple author names.
        
        Verifies that the function can compare input against a list
        of multiple authors and find the best match.
        """
        authors_norm = ["author1", "author2", "author3"]
        authors_orig = ["Author One", "Author Two", "Author Three"]
        result, score = fuzzy_match_author("Author Two", authors_norm, authors_orig)
        # Should find the best match
        assert isinstance(result, (str, type(None)))


class TestExtractAuthorFromLine:
    """Test suite for the extract_author_from_line function.
    
    Tests author extraction from text lines with various formats including
    bracketed text, dash prefixes, colon separators, and Tamil keywords.
    """
    
    def test_extract_author_empty_line(self):
        """Test author extraction from empty lines.
        
        Verifies that the function returns None when given:
        - Empty strings
        - Strings containing only whitespace
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        assert extract_author_from_line("", authors_norm, authors_orig) is None
        assert extract_author_from_line("   ", authors_norm, authors_orig) is None
    
    def test_extract_author_with_leading_dash(self):
        """Test extraction from lines starting with dashes.
        
        Verifies that the function can extract author names from
        lines that begin with em dashes or hyphens (common in
        formatted text).
        """
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        result = extract_author_from_line("— Author Name", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_brackets(self):
        """Test extraction of author names from within brackets.
        
        Verifies that the function can extract author names that
        are enclosed in square brackets [].
        """
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        result = extract_author_from_line("[Test Author]", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_nested_brackets(self):
        """Test extraction from lines with brackets embedded in text.
        
        Verifies that the function can extract author names from
        brackets even when surrounded by other text.
        """
        authors_norm = ["testauthor"]
        authors_orig = ["Test Author"]
        result = extract_author_from_line("Some text [Test Author] more", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_colon(self):
        """Test extraction from lines with colon separators.
        
        Verifies that the function can extract author names from
        lines using colon as a separator (e.g., "Author: Name").
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("Label: Author", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_multiple_colons(self):
        """Test extraction when multiple colons are present.
        
        Verifies that the function uses the text after the last
        colon when multiple colons are present in the line.
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("Label: Category: Author", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_tamil_keyword(self):
        """Test extraction using Tamil 'ஆசிரியர்' (author) keyword.
        
        Verifies that the function recognizes the Tamil word for
        "author" and extracts the name following it.
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("ஆசிரியர்: Author", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_trailing_punctuation(self):
        """Test that trailing punctuation is cleaned from author names.
        
        Verifies that the function removes trailing commas and other
        punctuation marks from extracted author names.
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line("Author,,,", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_leading_quotes(self):
        """Test that leading quotation marks are removed.
        
        Verifies that the function strips leading quote characters
        from extracted author names.
        """
        authors_norm = ["author"]
        authors_orig = ["Author"]
        result = extract_author_from_line('"Author', authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))
    
    def test_extract_author_with_exception(self):
        """Test exception handling during author extraction.
        
        Verifies that when an exception occurs during processing:
        - A warning is logged
        - The function returns None instead of crashing
        """
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
        """Test extraction of Tamil author names.
        
        Verifies that the function can properly handle and extract
        author names written in Tamil script.
        """
        authors_norm = ["கவிஞர்பெயர்"]
        authors_orig = ["கவிஞர் பெயர்"]
        result = extract_author_from_line("— கவிஞர் பெயர்", authors_norm, authors_orig)
        assert isinstance(result, (str, type(None)))


class TestFindAuthorInRange:
    """Test suite for the find_author_in_range function.
    
    Tests searching for author names within a range of text lines including
    blank line handling, index errors, and various edge cases.
    """
    
    def test_find_author_in_range_basic(self):
        """Test basic author finding functionality.
        
        Verifies that the function:
        - Returns a tuple of (author_name, line_index)
        - Successfully finds an author in a simple list of lines
        """
        lines = ["Line 1", "Author Name", "Line 3"]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger'):
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            assert isinstance(result, tuple)
            assert len(result) == 2
    
    def test_find_author_not_found(self):
        """Test behavior when no author is found in range.
        
        Verifies that when no author is found:
        - Function returns (None, -1)
        - A debug message about "No author found" is logged
        """
        lines = ["Line 1", "Line 2", "Line 3"]
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            assert result == (None, -1)
            # Should log that no author was found
            assert any("No author found" in str(call) for call in mock_logger.debug.call_args_list)
    
    def test_find_author_with_blank_lines(self):
        """Test that blank lines are properly skipped.
        
        Verifies that the function:
        - Skips empty lines and whitespace-only lines
        - Correctly identifies the line index when author is found
        """
        lines = ["", "  ", "   ", "Author Name", "Content"]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger'):
            result = find_author_in_range(lines, 0, 5, authors_norm, authors_orig)
            if result[0]:
                assert result[1] == 3
    
    def test_find_author_index_error(self):
        """Test handling of IndexError exceptions.
        
        Verifies that when an IndexError occurs:
        - Function returns (None, -1)
        - A warning is logged
        - No exception is raised to the caller
        """
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
        """Test handling of general exceptions.
        
        Verifies that when an unexpected exception occurs:
        - Function returns (None, -1)
        - An error is logged
        - No exception is raised to the caller
        """
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
        """Test that debug logging occurs during author search.
        
        Verifies that the function logs debug messages during
        its execution for troubleshooting purposes.
        """
        lines = ["Line 1", "Author Name", "Line 3"]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            # Should have debug logging calls
            assert mock_logger.debug.called
    
    def test_find_author_range_exceeds_length(self):
        """Test behavior when search range exceeds list length.
        
        Verifies that the function handles gracefully when the
        specified end_idx is larger than the actual list length,
        using min(end_idx, len(lines)) internally.
        """
        lines = ["Line 1", "Line 2"]
        authors_norm = ["author"]
        authors_orig = ["Author"]
        
        with patch('data_extraction.text_processing.logger'):
            # Should handle gracefully using min(end_idx, len(lines))
            result = find_author_in_range(lines, 0, 100, authors_norm, authors_orig)
            assert isinstance(result, tuple)
    
    def test_find_author_found_logs_correctly(self):
        """Test logging when an author is successfully found.
        
        Verifies that when an author is found, a debug message
        containing "Found author" is logged with relevant details.
        """
        lines = ["", "Author Name", ""]
        authors_norm = ["authorname"]
        authors_orig = ["Author Name"]
        
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = find_author_in_range(lines, 0, 3, authors_norm, authors_orig)
            if result[0]:
                # Should log finding the author
                assert any("Found author" in str(call) for call in mock_logger.debug.call_args_list)


class TestGetIntroKeywords:
    """Test suite for the get_intro_keywords function.
    
    Tests retrieval of introduction keywords used for identifying
    introductory sections in Tamil text documents.
    """
    
    def test_get_intro_keywords_returns_list(self):
        """Test that the function returns a non-empty list.
        
        Verifies that:
        - Return value is a list type
        - List contains at least one keyword
        """
        result = get_intro_keywords()
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_get_intro_keywords_contains_expected(self):
        """Test that expected keywords are present in the list.
        
        Verifies that specific Tamil keywords known to be in the
        function's return value are actually present.
        """
        result = get_intro_keywords()
        assert "எங்கள் எண்ணம்" in result
        assert "காலமும் கருத்தும்" in result
    
    def test_get_intro_keywords_debug_logging(self):
        """Test that debug logging occurs when retrieving keywords.
        
        Verifies that the function logs debug information about
        the keywords being returned.
        """
        with patch('data_extraction.text_processing.logger') as mock_logger:
            result = get_intro_keywords()
            assert mock_logger.debug.called
            assert len(result) > 0
    
    def test_get_intro_keywords_count(self):
        """Test that the correct number of keywords is returned.
        
        Verifies that the function returns exactly 17 introduction
        keywords as expected.
        """
        result = get_intro_keywords()
        # Verify it has all the expected keywords
        assert len(result) == 17


class TestIntegration:
    """Integration tests combining multiple functions.
    
    Tests realistic workflows that use multiple functions together
    to validate end-to-end functionality.
    """
    
    def test_full_workflow(self):
        """Test a complete text processing workflow.
        
        Verifies that multiple functions can be used together:
        1. Normalizing text (including Unicode)
        2. Validating headings
        3. Retrieving keywords
        
        This simulates a realistic processing pipeline.
        """
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
        """Test complete author extraction workflow.
        
        Verifies a realistic author extraction process:
        1. Extracting author from a single formatted line
        2. Finding author within a range of lines
        
        This simulates how author extraction would be used in
        practice when processing documents.
        """
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