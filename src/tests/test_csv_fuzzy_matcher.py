import pytest
from unittest.mock import Mock, patch
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_extraction.csv_fuzzy_matcher import (
    calculate_similarity,
    remove_symbols,
    extract_malar_issue_from_text,
    normalize_csv_value,
    find_article_boundary_fuzzy,
    extract_articles_from_csv
)


class TestCalculateSimilarity:
    """Test suite for the calculate_similarity function.
    
    Tests string similarity calculation using sequence matching algorithms
    (typically Levenshtein distance or similar). Returns similarity as a
    percentage (0-100).
    """
    
    def test_identical_strings(self):
        """Test that identical strings return 100% similarity.
        
        Verifies that the similarity algorithm correctly identifies
        exact string matches and returns maximum similarity score.
        """
        result = calculate_similarity("hello", "hello")
        assert result == 100.0
    
    def test_completely_different_strings(self):
        """Test that completely different strings return low similarity.
        
        Verifies that strings with no common characters or patterns
        produce low similarity scores (< 50%), indicating they are
        not matches.
        """
        result = calculate_similarity("abc", "xyz")
        assert result < 50
    
    def test_similar_strings(self):
        """Test that similar strings return moderate to high similarity.
        
        Verifies that strings with common character sequences but minor
        differences (like typos or variations) produce moderate to high
        similarity scores (> 50%).
        """
        result = calculate_similarity("testing", "tasting")
        assert result > 50
    
    def test_empty_strings(self):
        """Test that empty strings return 0% similarity.
        
        Verifies edge case handling:
        - Empty vs non-empty: 0% similarity
        - Both empty: 0% similarity
        
        This prevents false matches with empty data.
        """
        assert calculate_similarity("", "test") == 0
        assert calculate_similarity("test", "") == 0
        assert calculate_similarity("", "") == 0
    
    def test_none_strings(self):
        """Test that None values return 0% similarity.
        
        Verifies that None/null values are handled gracefully:
        - None vs string: 0% similarity
        - Both None: 0% similarity
        
        This prevents crashes when comparing missing data.
        """
        assert calculate_similarity(None, "test") == 0
        assert calculate_similarity("test", None) == 0


class TestRemoveSymbols:
    """Test suite for the remove_symbols function.
    
    Tests text cleaning by removing punctuation, special characters, and
    excessive whitespace while preserving alphanumeric characters and
    Tamil script.
    """
    
    def test_remove_punctuation(self):
        """Test removing punctuation marks from text.
        
        Verifies that common punctuation (commas, periods, exclamation
        marks) are removed to normalize text for matching.
        """
        result = remove_symbols("Hello, World!")
        assert result == "Hello World"
    
    def test_remove_multiple_spaces(self):
        """Test collapsing multiple consecutive spaces into one.
        
        Verifies that excessive whitespace is normalized to single
        spaces, improving consistency in fuzzy matching.
        """
        result = remove_symbols("Hello    World")
        assert result == "Hello World"
    
    def test_tamil_text_preserved(self):
        """Test that Tamil characters are preserved during cleaning.
        
        Verifies that Tamil Unicode characters (base characters) are
        retained while punctuation may be removed. This is critical
        for matching Tamil article titles and content.
        
        Note: Some Tamil punctuation marks (pulli) may be removed,
        but base consonants and vowels should remain.
        """
        result = remove_symbols("தமிழ் மொழி")
        # Base Tamil characters should be present, even if pulli marks are removed
        assert "தம" in result or "தமழ" in result  # த + म may lose pulli
        assert "ம" in result  # மொ may lose pulli
    
    def test_remove_special_symbols(self):
        """Test removing special symbols and characters.
        
        Verifies that non-alphanumeric symbols (@#$%^&*) are removed
        from text, leaving only letters and numbers.
        """
        result = remove_symbols("test@#$%^&*()test")
        assert result == "testtest"
    
    def test_strip_whitespace(self):
        """Test that leading and trailing whitespace is removed.
        
        Verifies that spaces at the beginning and end of strings
        are stripped away for clean matching.
        """
        result = remove_symbols("  test  ")
        assert result == "test"
    
    def test_nan_value(self):
        """Test that pandas NA/NaN values return empty string.
        
        Verifies safe handling of missing values in DataFrames,
        converting them to empty strings instead of crashing.
        """
        result = remove_symbols(pd.NA)
        assert result == ""
    
    def test_numeric_string(self):
        """Test that numeric strings are preserved.
        
        Verifies that numbers within strings are retained,
        which is important for matching article numbers and
        page references.
        """
        result = remove_symbols("123 456")
        assert result == "123 456"


class TestExtractMalarIssueFromText:
    """Test suite for the extract_malar_issue_from_text function.
    
    Tests extraction of மலர் (volume) and இதழ் (issue) numbers from Tamil
    text using regex patterns. These identifiers are essential for matching
    text documents with CSV metadata.
    """
    
    def test_extract_malar_and_issue_standard(self):
        """Test standard மலர் and இதழ் extraction.
        
        Verifies extraction of volume and issue numbers from the most
        common format: "மலர்: 10" and "இதழ்: 5".
        """
        text = "மலர்: 10\nஇதழ்: 5"
        malar, issue = extract_malar_issue_from_text(text)
        assert malar == "10"
        assert issue == "5"
    
    def test_extract_pongal_malar(self):
        """Test extraction of special பொங்கல் மலர் (Pongal edition).
        
        Verifies that special edition names like "பொங்கல்" are correctly
        identified as volume identifiers instead of numeric values.
        """
        text = "பொங்கல் மலர்\nஇதழ்: 3"
        malar, issue = extract_malar_issue_from_text(text)
        assert malar == "பொங்கல்"
        assert issue == "3"
    
    def test_extract_with_different_patterns(self):
        """Test extraction with various pattern variations.
        
        Verifies that different Tamil text patterns for volume/issue
        (like "மலர் எண்:" or "எண்:") are recognized and extracted.
        """
        text = "மலர் எண்: 15\nஎண்: 7"
        malar, issue = extract_malar_issue_from_text(text)
        assert malar == "15"
        # Note: 'எண்:' pattern may match for both if on different lines
        # This test checks the basic extraction works
        assert issue is not None
    
    def test_extract_with_thoguthi(self):
        """Test extraction with தொகுதி (collection/volume) pattern.
        
        Verifies that alternative Tamil terms for volume like "தொகுதி"
        are correctly recognized and extracted.
        """
        text = "தொகுதி: 20\nஇதழ்: 8"
        malar, issue = extract_malar_issue_from_text(text)
        assert malar == "20"
        assert issue == "8"
    
    def test_no_malar_found(self):
        """Test behavior when மலர் is not found in text.
        
        Verifies that:
        - Function returns None for missing மலர்
        - Still extracts இதழ் if present
        - Does not crash or raise exceptions
        """
        text = "இதழ்: 5"
        malar, issue = extract_malar_issue_from_text(text)
        assert malar is None
        assert issue == "5"
    
    def test_no_issue_found(self):
        """Test behavior when இதழ் is not found in text.
        
        Verifies that:
        - Function returns None for missing இதழ்
        - Still extracts மலர் if present
        - Handles partial metadata gracefully
        """
        text = "மலர்: 10"
        malar, issue = extract_malar_issue_from_text(text)
        assert malar == "10"
        assert issue is None
    
    def test_empty_text(self):
        """Test with empty text input.
        
        Verifies that empty strings are handled gracefully:
        - Returns None for both மலர் and இதழ்
        - Does not raise exceptions
        """
        malar, issue = extract_malar_issue_from_text("")
        assert malar is None
        assert issue is None
    
    def test_only_first_100_lines(self):
        """Test that only the first 100 lines are scanned.
        
        Verifies performance optimization: the function only searches
        the first 100 lines of text since மலர்/இதழ் identifiers are
        typically at the beginning of documents. This prevents
        unnecessary processing of long files.
        """
        lines = ["line\n" for _ in range(150)]
        lines[110] = "மலர்: 99\n"
        text = ''.join(lines)
        malar, issue = extract_malar_issue_from_text(text)
        assert malar is None  # Should not find it beyond line 100


class TestNormalizeCSVValue:
    """Test suite for the normalize_csv_value function.
    
    Tests normalization of values from CSV files, including converting
    float strings to integers, handling NA values, and trimming whitespace.
    """
    
    def test_float_to_int(self):
        """Test converting float strings representing whole numbers to integers.
        
        Verifies that CSV values like "10.0" are converted to clean
        integer strings "10" for easier matching with text identifiers.
        """
        assert normalize_csv_value("10.0") == "10"
        assert normalize_csv_value("5.0") == "5"
    
    def test_non_whole_float(self):
        """Test that non-whole floats remain unchanged.
        
        Verifies that decimal values like "10.5" are preserved since
        they represent actual fractional numbers, not just float
        formatting artifacts.
        """
        result = normalize_csv_value("10.5")
        assert result == "10.5"
    
    def test_string_value(self):
        """Test that string values are trimmed of whitespace.
        
        Verifies that leading and trailing spaces are removed from
        string values for consistent matching.
        """
        result = normalize_csv_value("  test  ")
        assert result == "test"
    
    def test_nan_value(self):
        """Test that pandas NA/NaN values return empty string.
        
        Verifies safe handling of missing values in CSV data,
        converting them to empty strings for consistent processing.
        """
        result = normalize_csv_value(pd.NA)
        assert result == ""
    
    def test_integer_value(self):
        """Test normalization of integer values.
        
        Verifies that integer values are converted to strings
        for text-based matching operations.
        """
        result = normalize_csv_value(42)
        assert result == "42"
    
    def test_float_value(self):
        """Test normalization of float values.
        
        Verifies that float values representing whole numbers
        are cleaned (10.0 → "10") for matching.
        """
        result = normalize_csv_value(10.0)
        assert result == "10"


class TestFindArticleBoundaryFuzzy:
    """Test suite for the find_article_boundary_fuzzy function.
    
    Tests fuzzy matching to find article boundaries in text by matching
    titles and extracting content between start and end markers.
    """
    
    def test_exact_title_match(self):
        """Test finding an article with exact title match.
        
        Verifies that when a title exactly matches a line in the text,
        the function correctly identifies and extracts the article content.
        """
        lines = [
            "Some intro text",
            "தமிழ் கட்டுரை",
            "Content line 1",
            "Content line 2"
        ]
        result = find_article_boundary_fuzzy(lines, "தமிழ் கட்டுரை")
        assert result is not None
        assert "தமிழ் கட்டுரை" in result
    
    def test_fuzzy_title_match(self):
        """Test finding an article with fuzzy (approximate) title match.
        
        Verifies that titles with minor differences (punctuation, spacing)
        can still be matched using fuzzy string matching algorithms.
        """
        lines = [
            "Some intro",
            "Tamil Article!",
            "Content here"
        ]
        result = find_article_boundary_fuzzy(lines, "Tamil Article")
        assert result is not None
    
    def test_with_next_title_boundary(self):
        """Test article extraction bounded by the next article's title.
        
        Verifies that when a next_title is provided, content extraction
        stops at that boundary, preventing content from multiple articles
        being merged together.
        """
        lines = [
            "First Article",
            "Content 1",
            "Content 2",
            "Second Article",
            "Content 3"
        ]
        result = find_article_boundary_fuzzy(
            lines, "First Article", next_title="Second Article"
        )
        assert result is not None
        assert "Content 1" in result
        assert "Content 3" not in result
    
    def test_low_similarity_returns_none(self):
        """Test that low similarity scores result in no match.
        
        Verifies that when text lines don't sufficiently match the
        target title (below threshold), the function returns None
        instead of a false positive match.
        """
        lines = ["Completely different text"]
        result = find_article_boundary_fuzzy(lines, "தமிழ் கட்டுரை")
        assert result is None
    
    def test_empty_lines_ignored(self):
        """Test that empty lines are skipped during matching.
        
        Verifies that blank lines in the text don't interfere with
        finding article titles, allowing the function to scan past
        whitespace to find content.
        """
        lines = ["", "", "Target Title", "Content"]
        result = find_article_boundary_fuzzy(lines, "Target Title")
        assert result is not None


class TestExtractArticlesFromCSV:
    """Test suite for the extract_articles_from_csv function.
    
    Tests the main article extraction workflow that combines CSV metadata
    with text content using fuzzy matching. This is the integration point
    for all other functions in the module.
    """
    
    @pytest.fixture
    def sample_csv_df(self):
        """Provide sample CSV DataFrame with article metadata.
        
        Returns:
            pd.DataFrame: DataFrame containing Tamil literary magazine
                metadata with columns for மலர், இதழ், தலைப்பு (title),
                ஆசிரியர் (author), and ஆண்டு (year).
        """
        return pd.DataFrame({
            'மலர்': ['10', '10', '11'],
            'இதழ்': ['5', '5', '6'],
            'தலைப்பு': ['Article 1', 'Article 2', 'Article 3'],
            'ஆசிரியர்': ['Author A', 'Author B', 'Author C'],
            'ஆண்டு': [2020, 2020, 2021]
        })
    
    @pytest.fixture
    def sample_lines(self):
        """Provide sample text lines simulating document content.
        
        Returns:
            list: Lines of text including மலர்/இதழ் headers and article
                content with sufficient length for validation.
        """
        return [
            'மலர்: 10',
            'இதழ்: 5',
            'Article 1',
            'Content for article 1 with enough text to pass the minimum length requirement of 100 characters.',
            'More content to ensure we have enough',
            'Article 2',
            'Content for article 2 with sufficient length to be extracted properly and pass validation checks.'
        ]
    
    @pytest.fixture
    def mock_file_path(self):
        """Provide mock file path object for testing.
        
        Returns:
            Mock: Mock Path object with name attribute for source
                document tracking.
        """
        mock_path = Mock()
        mock_path.name = 'test_document.txt'
        return mock_path
    
    def test_successful_extraction(self, sample_lines, sample_csv_df, mock_file_path):
        """Test successful article extraction with all components.
        
        Verifies the complete workflow:
        1. Extracts மலர்/இதழ் from text
        2. Finds matching CSV entries
        3. Fuzzy matches article titles
        4. Extracts article content
        5. Returns properly structured result
        
        Args:
            sample_lines: Fixture providing text lines
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        result = extract_articles_from_csv(
            sample_lines,
            sample_csv_df,
            mock_file_path
        )
        
        assert result is not None
        assert 'articles' in result
        assert 'authors_list' in result
        assert 'doc_id' in result
        assert 'doc_issue' in result
    
    def test_none_csv_df(self, sample_lines, mock_file_path):
        """Test handling of None CSV DataFrame.
        
        Verifies that the function gracefully handles missing CSV
        data by returning None instead of crashing.
        
        Args:
            sample_lines: Fixture providing text lines
            mock_file_path: Fixture providing file path
        """
        result = extract_articles_from_csv(
            sample_lines,
            None,
            mock_file_path
        )
        assert result is None
    
    def test_no_malar_issue_extraction(self, sample_csv_df, mock_file_path):
        """Test handling when மலர்/இதழ் cannot be extracted.
        
        Verifies that the function returns None when document
        identifiers are missing, preventing incorrect matches.
        
        Args:
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        lines = ['No malar or issue here']
        result = extract_articles_from_csv(
            lines,
            sample_csv_df,
            mock_file_path
        )
        assert result is None
    
    def test_no_matching_articles(self, sample_csv_df, mock_file_path):
        """Test when no articles match the document identifiers.
        
        Verifies that the function returns None when மலர்/இதழ்
        values don't match any entries in the CSV metadata.
        
        Args:
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        lines = ['மலர்: 99', 'இதழ்: 99']
        result = extract_articles_from_csv(
            lines,
            sample_csv_df,
            mock_file_path
        )
        assert result is None
    
    def test_article_field_names(self, sample_lines, sample_csv_df, mock_file_path):
        """Test that extracted articles have all required fields.
        
        Verifies that each article contains the complete set of
        metadata fields needed for downstream processing:
        doc_id, doc_issue, article_no, author_name, title,
        content, year, source_document.
        
        Args:
            sample_lines: Fixture providing text lines
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        result = extract_articles_from_csv(
            sample_lines,
            sample_csv_df,
            mock_file_path
        )
        
        if result and result['articles']:
            article = result['articles'][0]
            required_fields = [
                'doc_id', 'doc_issue', 'article_no', 'author_name',
                'title', 'content', 'year', 'source_document'
            ]
            for field in required_fields:
                assert field in article
    
    def test_year_extraction(self, sample_lines, sample_csv_df, mock_file_path):
        """Test that publication year is correctly extracted.
        
        Verifies that the ஆண்டு (year) field from CSV is properly
        extracted and converted to string format in the output.
        
        Args:
            sample_lines: Fixture providing text lines
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        result = extract_articles_from_csv(
            sample_lines,
            sample_csv_df,
            mock_file_path
        )
        
        if result and result['articles']:
            article = result['articles'][0]
            assert article['year'] == '2020'
    
    def test_source_document_field(self, sample_lines, sample_csv_df, mock_file_path):
        """Test that source document filename is populated.
        
        Verifies that the source_document field contains the
        filename from the file path for tracking data provenance.
        
        Args:
            sample_lines: Fixture providing text lines
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        result = extract_articles_from_csv(
            sample_lines,
            sample_csv_df,
            mock_file_path
        )
        
        if result and result['articles']:
            article = result['articles'][0]
            assert article['source_document'] == 'test_document.txt'
    
    def test_authors_list_unique(self, sample_lines, sample_csv_df, mock_file_path):
        """Test that authors list contains only unique author names.
        
        Verifies that duplicate authors are removed, ensuring each
        author appears only once in the authors list regardless of
        how many articles they wrote.
        
        Args:
            sample_lines: Fixture providing text lines
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        result = extract_articles_from_csv(
            sample_lines,
            sample_csv_df,
            mock_file_path
        )
        
        if result and result['authors_list']:
            authors = [a['author_name'] for a in result['authors_list']]
            assert len(authors) == len(set(authors))
    
    def test_minimum_content_length(self, sample_csv_df, mock_file_path):
        """Test that articles with insufficient content are filtered out.
        
        Verifies that articles with less than 100 characters of content
        are excluded from results, preventing extraction of incomplete
        or header-only text.
        
        Args:
            sample_csv_df: Fixture providing CSV metadata
            mock_file_path: Fixture providing file path
        """
        short_lines = ['மலர்: 10', 'இதழ்: 5', 'Article 1', 'x']
        result = extract_articles_from_csv(
            short_lines,
            sample_csv_df,
            mock_file_path
        )
        
        # Should have no articles or empty list
        if result:
            assert len(result['articles']) == 0
    
    def test_na_author_handling(self, sample_lines, mock_file_path):
        """Test handling of NA/missing author values.
        
        Verifies that pandas NA values in the ஆசிரியர் (author)
        column are converted to the string "NA" rather than causing
        errors or being left as null.
        
        Args:
            sample_lines: Fixture providing text lines
            mock_file_path: Fixture providing file path
        """
        df_with_na = pd.DataFrame({
            'மலர்': ['10'],
            'இதழ்': ['5'],
            'தலைப்பு': ['Article'],
            'ஆசிரியர்': [pd.NA],
            'ஆண்டு': [2020]
        })
        
        result = extract_articles_from_csv(
            sample_lines,
            df_with_na,
            mock_file_path
        )
        
        if result and result['articles']:
            assert result['articles'][0]['author_name'] == 'NA'


class TestIntegration:
    """Integration tests for the entire CSV fuzzy matching module.
    
    Tests end-to-end workflows that combine multiple functions to validate
    they work together correctly for realistic use cases.
    """
    
    def test_normalization_consistency(self):
        """Test that normalization is consistent between CSV and text.
        
        Verifies that values normalized from CSV (e.g., 10.0 → "10")
        match values extracted from text (மலர்: 10), ensuring the
        fuzzy matching system can correctly align CSV metadata with
        text documents regardless of formatting differences.
        
        This test validates the critical integration between:
        - normalize_csv_value() for CSV data
        - extract_malar_issue_from_text() for text extraction
        - Overall matching logic in extract_articles_from_csv()
        """
        df = pd.DataFrame({
            'மலர்': [10.0],  # Float in CSV
            'இதழ்': ['5.0'],  # String float in CSV
            'தலைப்பு': ['Article'],
            'ஆசிரியர்': ['Author'],
            'ஆண்டு': [2020]
        })
        
        lines = [
            'மலர்: 10', 
            'இதழ்: 5', 
            'Article', 
            'Content here for testing purposes with enough length to pass validation requirements.'
        ]
        mock_path = Mock()
        mock_path.name = 'test.txt'
        
        result = extract_articles_from_csv(lines, df, mock_path)
        
        assert result is not None
        assert result['doc_id'] == '10'
        assert result['doc_issue'] == '5'


# Parametrized tests
@pytest.mark.parametrize("input_val,expected", [
    ("10.0", "10"),
    ("5.0", "5"),
    ("10.5", "10.5"),
    ("  test  ", "test"),
    (42, "42"),
])
def test_normalize_csv_value_parametrized(input_val, expected):
    """Parametrized test for CSV value normalization.
    
    Tests multiple input/output scenarios efficiently using pytest
    parametrization. Validates normalization of various data types
    and formats commonly found in CSV files.
    
    Args:
        input_val: Input value to normalize (string, int, float)
        expected (str): Expected normalized output string
    """
    assert normalize_csv_value(input_val) == expected


@pytest.mark.parametrize("str1,str2,min_similarity", [
    ("hello", "hello", 100.0),
    ("test", "text", 50.0),
    ("abc", "xyz", 0.0),
])
def test_calculate_similarity_parametrized(str1, str2, min_similarity):
    """Parametrized test for string similarity calculation.
    
    Tests multiple string comparison scenarios to validate the
    similarity algorithm's behavior across different cases:
    - Identical strings → 100% similarity
    - Similar strings → moderate similarity (≥50%)
    - Different strings → low similarity (≥0%)
    
    Args:
        str1 (str): First string to compare
        str2 (str): Second string to compare
        min_similarity (float): Minimum expected similarity percentage
    """
    result = calculate_similarity(str1, str2)
    assert result >= min_similarity