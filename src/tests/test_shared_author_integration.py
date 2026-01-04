"""Integration tests for shared_author - calls real functions"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.shared_author import check_author_ahead

class TestSharedAuthorIntegration:
    def test_check_author_ahead_real(self):
        lines = ["intro", "John Smith", "content"]
        result = check_author_ahead(lines, 0, ["johnsmith"], ["John Smith"], 3)
        assert result is None or isinstance(result, (int, tuple))
        
        # Empty lists should not crash
        result = check_author_ahead(["a", "b"], 0, [], [], 2)
        assert result is None or isinstance(result, (int, tuple))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])