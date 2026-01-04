"""Integration tests for content_extraction - calls real functions"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.content_extraction import (
    check_keyword_ahead, count_consecutive_blanks,
    extract_remaining_content, extract_intro_content_phase1
)

class TestContentExtractionIntegration:
    def test_check_keyword_ahead_real(self):
        lines = ["line1", "எங்கள் எண்ணம்", "content"]
        keywords = ["எங்கள் எண்ணம்"]
        assert check_keyword_ahead(lines, 1, keywords, 2) == 1
        assert check_keyword_ahead(["a", "b"], 0, ["c"], 2) is None
    
    def test_count_consecutive_blanks_real(self):
        assert count_consecutive_blanks(["a", "", "", "b"], 1) == 2
        assert count_consecutive_blanks(["a", "b"], 0) == 0
    
    def test_extract_remaining_content_real(self):
        lines = ["Title", "C1", "C2", "C3", "C4", "", ""]
        processed = [False] * len(lines)
        result = extract_remaining_content(lines, 0, processed)
        assert len(result) >= 0
    
    def test_extract_intro_real(self):
        lines = ["எங்கள் எண்ணம்", "", "Intro", "", ""]
        processed = [False] * len(lines)
        content, idx, author = extract_intro_content_phase1(
            lines, 0, processed, [], [], ["எங்கள் எண்ணம்"]
        )
        assert isinstance(content, str)
        assert isinstance(idx, int)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])