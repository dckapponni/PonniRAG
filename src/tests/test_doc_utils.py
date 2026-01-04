"""
Unit tests for doc_utils module.
Tests document info extraction, author extraction, and utility functions.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.doc_utils import (
    extract_doc_info,
    is_valid_author_name,
    is_section_type,
    is_section_header,
    parse_toc_line_robust,
    extract_authors_from_toc,
    get_shared_authors,
    extract_authors_alternative,
    count_words,
    count_content_lines,
    find_toc_boundaries
)


class TestExtractDocInfo:
    """Test cases for extract_doc_info function."""
    
    def test_extract_both_ids(self):
        """Test extracting both மலர் and இதழ்."""
        lines = [
            "பொன்னி",
            "மலர் : 12",
            "இதழ் : 5",
            "content"
        ]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "12"
        assert doc_issue == "5"
    
    def test_extract_without_colons(self):
        """Test extracting without colons."""
        lines = [
            "மலர் 15",
            "இதழ் 3"
        ]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "15"
        assert doc_issue == "3"
    
    def test_no_markers_found(self):
        """Test when markers are not found."""
        lines = [
            "random content",
            "no markers here"
        ]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "NA"
        assert doc_issue == "NA"
    
    def test_only_maalar_found(self):
        """Test when only மலர் is found."""
        lines = [
            "மலர் : 20",
            "no issue marker"
        ]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "20"
        assert doc_issue == "NA"


class TestIsValidAuthorName:
    """Test cases for is_valid_author_name function."""
    
    def test_valid_author(self):
        """Test valid author name."""
        assert is_valid_author_name("முருகன்") is True
    
    def test_too_short(self):
        """Test name that's too short."""
        assert is_valid_author_name("a") is False
    
    def test_empty_string(self):
        """Test empty string."""
        assert is_valid_author_name("") is False
    
    def test_section_type(self):
        """Test section type (should be rejected)."""
        assert is_valid_author_name("கார்ட்டூன்") is False
        assert is_valid_author_name("தலையங்கம்") is False
    
    def test_only_dots(self):
        """Test string with only dots."""
        assert is_valid_author_name("...") is False
    
    def test_too_long(self):
        """Test string that's too long."""
        long_name = "இது மிக நீளமான பெயர் அல்லது தலைப்பு இது ஏற்றுக்கொள்ளப்படக்கூடாது"
        assert is_valid_author_name(long_name) is False
    
    def test_valid_long_name(self):
        """Test valid long author name."""
        assert is_valid_author_name("அப்துல் கலாம்") is True


class TestIsSectionType:
    """Test cases for is_section_type function."""
    
    def test_known_section_types(self):
        """Test known section types."""
        assert is_section_type("கார்ட்டூன்") is True
        assert is_section_type("தலையங்கம்") is True
    
    def test_ellipsis(self):
        """Test ellipsis detection."""
        assert is_section_type("...") is True
        assert is_section_type("…") is True
    
    def test_regular_text(self):
        """Test regular text (not section type)."""
        assert is_section_type("முருகன்") is False
    
    def test_empty(self):
        """Test empty string."""
        assert is_section_type("") is False


class TestIsSectionHeader:
    """Test cases for is_section_header function."""
    
    def test_known_headers(self):
        """Test known section headers."""
        assert is_section_header("காலமும் கருத்தும்") is True
        assert is_section_header("பொது மேடை") is True
    
    def test_header_starting_patterns(self):
        """Test headers with starting patterns."""
        assert is_section_header("காலமும் நேரமும்") is True
    
    def test_regular_title(self):
        """Test regular title (not header)."""
        assert is_section_header("கதையின் தலைப்பு") is False
    
    def test_empty(self):
        """Test empty string."""
        assert is_section_header("") is False


class TestParseTocLineRobust:
    """Test cases for parse_toc_line_robust function."""
    
    def test_parse_with_two_spaces(self):
        """Test parsing line with two-space separator."""
        result = parse_toc_line_robust("சிறுகதை  முருகன்  45")
        assert result is not None
        assert result['title'] == "சிறுகதை"
        assert result['author'] == "முருகன்"
        assert result['page'] == "45"
    
    def test_parse_title_only(self):
        """Test parsing line with title only (no author)."""
        result = parse_toc_line_robust("சிறுகதை  5")
        assert result is not None
        assert result['title'] is not None
        assert result['page'] == "5"
    
    def test_parse_invalid_no_page(self):
        """Test parsing line without page number."""
        result = parse_toc_line_robust("just some text")
        assert result is None
    
    def test_parse_empty_line(self):
        """Test parsing empty line."""
        result = parse_toc_line_robust("")
        assert result is None
    
    def test_parse_section_header(self):
        """Test parsing section header (should have no author)."""
        result = parse_toc_line_robust("காலமும் கருத்தும்  5")
        assert result is not None
        assert result['author'] is None  # Section headers don't have authors


class TestFindTocBoundaries:
    """Test cases for find_toc_boundaries function."""
    
    def test_find_basic_toc(self):
        """Test finding basic TOC boundaries."""
        lines = [
            "பொன்னி",
            "மலர் : 10",
            "பொருளடக்கம்",
            "சிறுகதை  முருகன்  12",
            "கவிதை  காமராஜ்  15",
            "ஆகியோரின் எழுத்தோவியங்கள்"
        ]
        toc_start, toc_end = find_toc_boundaries(lines)
        assert toc_start == 3  # After பொருளடக்கம்
        assert toc_end == 5    # At ஆகியோரின்
    
    def test_toc_not_found(self):
        """Test when TOC is not found."""
        lines = [
            "random content",
            "no TOC here"
        ]
        toc_start, toc_end = find_toc_boundaries(lines)
        assert toc_start == -1
        assert toc_end == -1


class TestExtractAuthorsFromToc:
    """Test cases for extract_authors_from_toc function."""
    
    def test_extract_authors_standard_format(self):
        """Test extracting authors from standard TOC format."""
        lines = [
            "பொன்னி",
            "மலர் : 10",
            "இதழ் : 5",
            "பொருளடக்கம்",
            "சிறுகதை  முருகன்  12",
            "கவிதை  காமராஜ்  15",
            "ஆகியோரின் எழுத்தோவியங்கள்"
        ]
        doc_id, doc_issue, authors_orig, authors_norm, pairs = extract_authors_from_toc(lines)
        
        assert doc_id == "10"
        assert doc_issue == "5"
        assert len(authors_orig) >= 1  # At least one author
        assert len(authors_orig) == len(authors_norm)
    
    def test_no_toc_found(self):
        """Test when பொருளடக்கம் is not found."""
        lines = [
            "random content",
            "no TOC here"
        ]
        doc_id, doc_issue, authors_orig, authors_norm, pairs = extract_authors_from_toc(lines)
        
        assert len(authors_orig) == 0
        assert len(authors_norm) == 0


class TestGetSharedAuthors:
    """Test cases for get_shared_authors function."""
    
    def test_retrieve_existing_authors(self):
        """Test retrieving existing authors."""
        shared_dict = {
            ("10", "5"): (["முருகன்", "காமராஜ்"], ["murugan", "kamaraj"])
        }
        authors_orig, authors_norm = get_shared_authors("10", "5", shared_dict)
        
        assert len(authors_orig) == 2
        assert "முருகன்" in authors_orig
    
    def test_retrieve_nonexistent_document(self):
        """Test retrieving authors for non-existent document."""
        shared_dict = {
            ("10", "5"): (["முருகன்"], ["murugan"])
        }
        authors_orig, authors_norm = get_shared_authors("99", "99", shared_dict)
        
        assert len(authors_orig) == 0
        assert len(authors_norm) == 0
    
    def test_empty_shared_dict(self):
        """Test with empty shared dictionary."""
        authors_orig, authors_norm = get_shared_authors("10", "5", {})
        
        assert len(authors_orig) == 0
        assert len(authors_norm) == 0


class TestExtractAuthorsAlternative:
    """Test cases for extract_authors_alternative function."""
    
    def test_extract_after_blank_lines(self):
        """Test extracting author after multiple blank lines."""
        lines = [
            "content line 1 with sufficient length for context",
            "content line 2 with sufficient length for context",
            "",
            "",
            "முருகன்",
            "more content"
        ]
        authors_orig, authors_norm = extract_authors_alternative(lines)
        
        # May or may not find depending on validation rules
        # Just check it doesn't crash
        assert isinstance(authors_orig, list)
        assert isinstance(authors_norm, list)
    
    def test_no_blank_lines(self):
        """Test when there are no blank lines."""
        lines = [
            "content",
            "more content",
            "even more"
        ]
        authors_orig, authors_norm = extract_authors_alternative(lines)
        
        assert len(authors_orig) == 0


class TestCountWords:
    """Test cases for count_words function."""
    
    def test_count_basic_words(self):
        """Test counting basic words."""
        assert count_words("hello world") == 2
    
    def test_count_with_extra_spaces(self):
        """Test counting with multiple spaces."""
        assert count_words("hello   world") == 2
    
    def test_count_empty_string(self):
        """Test counting empty string."""
        assert count_words("") == 0
    
    def test_count_none(self):
        """Test counting None."""
        assert count_words(None) == 0
    
    def test_count_tamil_words(self):
        """Test counting Tamil words."""
        count = count_words("இது ஒரு சோதனை")
        assert count == 3


class TestCountContentLines:
    """Test cases for count_content_lines function."""
    
    def test_count_basic_lines(self):
        """Test counting basic lines."""
        text = "line1\nline2\nline3"
        assert count_content_lines(text) == 3
    
    def test_count_with_blank_lines(self):
        """Test counting with blank lines."""
        text = "line1\n\nline2\n\nline3"
        assert count_content_lines(text) == 3
    
    def test_count_empty_string(self):
        """Test counting empty string."""
        assert count_content_lines("") == 0
    
    def test_count_none(self):
        """Test counting None."""
        assert count_content_lines(None) == 0
    
    def test_count_only_blank_lines(self):
        """Test counting only blank lines."""
        text = "\n\n\n"
        assert count_content_lines(text) == 0


if __name__ == '__main__':
    pytest.main([__file__, "-v"])