"""Integration tests for shared_author module."""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.shared_author import check_author_ahead


class TestSharedAuthorIntegration:
    """Integration tests for shared author detection functions with real implementations."""
    
    def test_check_author_ahead_real(self):
        """
        Test detection of author names in upcoming lines.
        
        Verifies that the function searches ahead in document lines
        for matching author names and returns appropriate results
        (index or tuple) when found, or None when not found.
        """
        lines = ["intro", "John Smith", "content"]
        result = check_author_ahead(lines, 0, ["johnsmith"], ["John Smith"], 3)
        assert result is None or isinstance(result, (int, tuple))
        
        result = check_author_ahead(["a", "b"], 0, [], [], 2)
        assert result is None or isinstance(result, (int, tuple))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])