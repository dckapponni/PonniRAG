"""Integration tests for doc_utils - calls real functions"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.doc_utils import (
    extract_doc_info, is_valid_author_name, is_section_type,
    is_section_header, parse_toc_line_robust, count_words,
    count_content_lines, find_toc_boundaries, extract_authors_from_toc,
    get_shared_authors
)

class TestDocUtilsIntegration:
    def test_extract_doc_info_real(self):
        lines = ["மலர் : 12", "இதழ் : 5"]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "12"
        assert doc_issue == "5"
        
        lines = ["random"]
        doc_id, doc_issue = extract_doc_info(lines)
        assert doc_id == "NA"
        assert doc_issue == "NA"
    
    def test_is_valid_author_name_real(self):
        assert is_valid_author_name("முருகன்") is True
        assert is_valid_author_name("a") is False
        assert is_valid_author_name("") is False
        assert is_valid_author_name("...") is False
    
    def test_is_section_type_real(self):
        assert is_section_type("கார்ட்டூன்") is True
        assert is_section_type("...") is True
        assert is_section_type("regular") is False
    
    def test_is_section_header_real(self):
        assert is_section_header("காலமும் கருத்தும்") is True
        assert is_section_header("regular title") is False
    
    def test_parse_toc_line_real(self):
        result = parse_toc_line_robust("சிறுகதை  முருகன்  45")
        assert result is not None
        assert result['page'] == "45"
    
    def test_count_words_real(self):
        assert count_words("hello world") == 2
        assert count_words("") == 0
        assert count_words(None) == 0
    
    def test_count_content_lines_real(self):
        assert count_content_lines("line1\nline2") == 2
        assert count_content_lines("") == 0
    
    def test_find_toc_boundaries_real(self):
        lines = ["பொன்னி", "பொருளடக்கம்", "content", "ஆகியோரின்"]
        start, end = find_toc_boundaries(lines)
        assert start == 2
        assert end == 3
    
    def test_extract_authors_from_toc_real(self):
        lines = ["மலர் : 10", "இதழ் : 5", "பொருளடக்கம்", "Author", "ஆகியோரின்"]
        doc_id, doc_issue, authors_o, authors_n, pairs = extract_authors_from_toc(lines)
        assert doc_id == "10"
        assert doc_issue == "5"
    
    def test_get_shared_authors_real(self):
        shared = {("10", "5"): (["Author"], ["author"])}
        authors_o, authors_n = get_shared_authors("10", "5", shared)
        assert len(authors_o) == 1
        
        authors_o, authors_n = get_shared_authors("99", "99", shared)
        assert len(authors_o) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])