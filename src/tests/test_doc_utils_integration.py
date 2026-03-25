"""Integration tests for document utility functions."""

import sys
from pathlib import Path

import pytest

from src.data_extraction.doc_utils import count_content_lines  # noqa: E402
from src.data_extraction.doc_utils import (
    extract_authors_from_toc,
    extract_doc_info,
    find_toc_boundaries,
    get_shared_authors,
    is_section_header,
    is_section_type,
    is_valid_author_name,
    parse_toc_line_robust,
)

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDocUtilsIntegration:
    """Integration tests for document utility functions with real implementations."""

    def test_extract_doc_info_real(self):
        """
        Test extraction of document ID and issue number from lines.

        Verifies that the function correctly parses document metadata
        and returns 'NA' when information is not found.
        """
        lines = ["மலர் : 12", "இதழ் : 5"]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "12"
        assert doc_issue == "5"

        lines = ["random"]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "NA"
        assert doc_issue == "NA"

    def test_is_valid_author_name_real(self):
        """
        Test validation of author names.

        Verifies that the function correctly identifies valid author names
        and rejects empty, single-character, or placeholder names.
        """
        assert is_valid_author_name("முருகன்") is True
        assert is_valid_author_name("a") is False
        assert is_valid_author_name("") is False
        assert is_valid_author_name("...") is False

    def test_is_section_type_real(self):
        """
        Test identification of section type markers.

        Verifies detection of special section types like cartoons
        and placeholder markers.
        """
        assert is_section_type("கார்ட்டூன்") is True
        assert is_section_type("...") is True
        assert is_section_type("regular") is False

    def test_is_section_header_real(self):
        """
        Test identification of section headers.

        Verifies recognition of predefined section header patterns.
        """
        assert is_section_header("காலமும் கருத்தும்") is True
        assert is_section_header("regular title") is False

    def test_parse_toc_line_real(self):
        """
        Test parsing of table of contents lines.

        Verifies extraction of title, author, and page information
        from TOC entries.
        """
        result = parse_toc_line_robust("சிறுகதை  முருகன்  45")
        assert result is not None
        assert result["page"] == "45"

    def test_count_content_lines_real(self):
        """
        Test counting of content lines.

        Verifies accurate line counting in multi-line and empty strings.
        """
        assert count_content_lines("line1\nline2") == 2
        assert count_content_lines("") == 0

    def test_find_toc_boundaries_real(self):
        """
        Test detection of table of contents boundaries.

        Verifies identification of start and end markers for the
        table of contents section in document lines.
        """
        lines = ["பொன்னி", "பொருளடக்கம்", "content", "ஆகியோரின்"]
        start, end = find_toc_boundaries(lines)
        assert start == 2
        assert end == 3

    def test_extract_authors_from_toc_real(self):
        """
        Test extraction of authors from table of contents.

        Verifies extraction of document metadata and author lists
        from TOC section of the document.
        """
        lines = ["மலர் : 10", "இதழ் : 5", "பொருளடக்கம்", "Author", "ஆகியோரின்"]
        doc_id, doc_issue, authors_o, authors_n, pairs = extract_authors_from_toc(lines)
        assert doc_id == "10"
        assert doc_issue == "5"

    def test_get_shared_authors_real(self):
        """
        Test retrieval of shared author information.

        Verifies lookup of author data for specific document ID and issue
        combinations, returning empty lists when not found.
        """
        shared = {("10", "5"): (["Author"], ["author"])}
        authors_o, authors_n = get_shared_authors("10", "5", shared)
        assert len(authors_o) == 1

        authors_o, authors_n = get_shared_authors("99", "99", shared)
        assert len(authors_o) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
