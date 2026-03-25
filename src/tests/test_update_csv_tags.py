"""Tests for update_csv_tags module.

Covers get_tags_from_qdrant and main() with mocked Qdrant
and CSV dependencies.
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
from update_csv_tags import get_tags_from_qdrant, main

# ================================================================
# HELPERS
# ================================================================


def _make_point(tags_tamil=None, payload_extra=None):
    """Create a mock Qdrant point with metadata."""
    metadata = {"tags_tamil": tags_tamil or []}
    if payload_extra:
        metadata.update(payload_extra)
    point = MagicMock()
    point.payload = {"metadata": metadata}
    return point


def _make_client(scroll_return=None):
    """Create a mock QdrantClient with configurable scroll."""
    client = MagicMock()
    if scroll_return is None:
        scroll_return = ([], None)
    client.scroll.return_value = scroll_return
    return client


# ================================================================
# get_tags_from_qdrant TESTS
# ================================================================


class TestGetTagsFromQdrant:
    """Tests for get_tags_from_qdrant function."""

    def test_returns_tags_when_article_no_matches(self):
        """Return tags when article_no int match succeeds."""
        tags = ["கவிதை", "சிறுகதை"]
        client = _make_client(([_make_point(tags)], None))
        result = get_tags_from_qdrant(client, "1", "2", "3")
        assert result == tags

    def test_returns_empty_list_when_no_points(self):
        """Return empty list when no points found."""
        client = _make_client(([], None))
        result = get_tags_from_qdrant(client, "1", "2", "3")
        assert result == []

    def test_fallback_without_article_no(self):
        """Fall back to doc_id+doc_issue when article_no is empty."""
        tags = ["கட்டுரை"]
        client = _make_client(([_make_point(tags)], None))
        result = get_tags_from_qdrant(client, "1", "2", "")
        assert result == tags
        # Should have been called once (no article_no branch)
        assert client.scroll.call_count == 1

    def test_fallback_when_int_match_fails(self):
        """Fall back to doc_id+doc_issue when int parse raises."""
        tags = ["நாடகம்"]
        client = MagicMock()
        # First call (with article_no) raises, second succeeds
        client.scroll.side_effect = [
            Exception("int match failed"),
            ([_make_point(tags)], None),
        ]
        result = get_tags_from_qdrant(client, "1", "2", "abc")
        assert result == tags

    def test_fallback_when_int_match_returns_empty(self):
        """Fall back when int match returns no points."""
        tags = ["ஆய்வு"]
        client = MagicMock()
        # First scroll returns empty, second returns tags
        client.scroll.side_effect = [
            ([], None),
            ([_make_point(tags)], None),
        ]
        result = get_tags_from_qdrant(client, "1", "2", "5")
        assert result == tags

    def test_returns_empty_when_both_paths_empty(self):
        """Return empty when both article_no and fallback find nothing."""
        client = MagicMock()
        client.scroll.side_effect = [
            ([], None),
            ([], None),
        ]
        result = get_tags_from_qdrant(client, "1", "2", "5")
        assert result == []

    def test_handles_none_payload(self):
        """Handle point with None payload gracefully."""
        point = MagicMock()
        point.payload = None
        client = _make_client(([point], None))
        result = get_tags_from_qdrant(client, "1", "2", "")
        assert result == []

    def test_handles_missing_tags_tamil_key(self):
        """Handle metadata without tags_tamil key."""
        point = MagicMock()
        point.payload = {"metadata": {"other": "data"}}
        client = _make_client(([point], None))
        result = get_tags_from_qdrant(client, "1", "2", "")
        assert result == []


# ================================================================
# main() TESTS
# ================================================================


class TestMain:
    """Tests for the main function."""

    @patch("update_csv_tags.CSV_PATH")
    def test_exits_when_csv_not_found(self, mock_path, caplog):
        """Return early when CSV file does not exist."""
        mock_path.exists.return_value = False
        with caplog.at_level(logging.ERROR):
            main()
        assert "CSV not found" in caplog.text

    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_exits_when_required_columns_missing(
        self, mock_path, mock_load, mock_qclient, caplog
    ):
        """Return early when required columns are missing."""
        mock_path.exists.return_value = True
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        mock_load.return_value = df
        with caplog.at_level(logging.ERROR):
            main()
        assert "Required columns missing" in caplog.text

    @patch("update_csv_tags.get_tags_from_qdrant")
    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_updates_csv_with_tags(
        self, mock_path, mock_load, mock_qclient, mock_get_tags
    ):
        """Write tags column and save CSV."""
        mock_path.exists.return_value = True
        df = pd.DataFrame(
            {
                "வ.எ.": ["1", "2"],
                "ஆண்டு": ["1947", "1948"],
                "மலர்": ["1", "2"],
                "இதழ்": ["1", "3"],
                "தலைப்பு": ["title1", "title2"],
                "ஆசிரியர்": ["author1", "author2"],
            }
        )
        mock_load.return_value = df
        mock_get_tags.side_effect = [
            ["கவிதை"],
            ["சிறுகதை", "நாடகம்"],
        ]
        with patch.object(df, "to_csv"):
            main()
        assert "வகை" in df.columns
        assert df["வகை"].tolist() == [
            "கவிதை",
            "சிறுகதை, நாடகம்",
        ]

    @patch("update_csv_tags.get_tags_from_qdrant")
    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_handles_empty_doc_id(
        self, mock_path, mock_load, mock_qclient, mock_get_tags
    ):
        """Skip rows with empty doc_id or doc_issue."""
        mock_path.exists.return_value = True
        df = pd.DataFrame(
            {
                "வ.எ.": ["1"],
                "ஆண்டு": ["1947"],
                "மலர்": [""],
                "இதழ்": ["1"],
                "தலைப்பு": ["title"],
                "ஆசிரியர்": ["author"],
            }
        )
        mock_load.return_value = df
        with patch.object(df, "to_csv"):
            main()
        assert df["வகை"].tolist() == [""]
        mock_get_tags.assert_not_called()

    @patch("update_csv_tags.get_tags_from_qdrant")
    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_normalises_trailing_dot_zero(
        self, mock_path, mock_load, mock_qclient, mock_get_tags
    ):
        """Strip trailing .0 from numeric values read as float."""
        mock_path.exists.return_value = True
        df = pd.DataFrame(
            {
                "வ.எ.": ["1.0"],
                "ஆண்டு": ["1947"],
                "மலர்": ["2.0"],
                "இதழ்": ["3.0"],
                "தலைப்பு": ["title"],
                "ஆசிரியர்": ["author"],
            }
        )
        mock_load.return_value = df
        mock_get_tags.return_value = ["கவிதை"]
        with patch.object(df, "to_csv"):
            main()
        # Verify the normalised values were passed
        call_args = mock_get_tags.call_args
        assert call_args[0][1] == "2"  # doc_id
        assert call_args[0][2] == "3"  # doc_issue
        assert call_args[0][3] == "1"  # article_no

    @patch("update_csv_tags.get_tags_from_qdrant")
    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_handles_no_serial_column(
        self, mock_path, mock_load, mock_qclient, mock_get_tags, caplog
    ):
        """Warn and proceed when serial number column is missing."""
        mock_path.exists.return_value = True
        df = pd.DataFrame(
            {
                "ஆண்டு": ["1947"],
                "மலர்": ["1"],
                "இதழ்": ["1"],
                "தலைப்பு": ["title"],
                "ஆசிரியர்": ["author"],
            }
        )
        mock_load.return_value = df
        mock_get_tags.return_value = []
        with caplog.at_level(logging.WARNING):
            with patch.object(df, "to_csv"):
                main()
        assert "வ.எ." in caplog.text
        # article_no should be empty string
        call_args = mock_get_tags.call_args
        assert call_args[0][3] == ""

    @patch("update_csv_tags.get_tags_from_qdrant")
    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_not_found_counter(
        self, mock_path, mock_load, mock_qclient, mock_get_tags, caplog
    ):
        """Track not-found count for rows with no tags."""
        mock_path.exists.return_value = True
        df = pd.DataFrame(
            {
                "வ.எ.": ["1", "2", "3"],
                "ஆண்டு": ["1947", "1947", "1947"],
                "மலர்": ["1", "1", "1"],
                "இதழ்": ["1", "2", "3"],
                "தலைப்பு": ["t1", "t2", "t3"],
                "ஆசிரியர்": ["a1", "a2", "a3"],
            }
        )
        mock_load.return_value = df
        mock_get_tags.side_effect = [["tag"], [], []]
        with caplog.at_level(logging.INFO):
            with patch.object(df, "to_csv"):
                main()
        assert "2 not found" in caplog.text

    @patch("update_csv_tags.get_tags_from_qdrant")
    @patch("update_csv_tags.QdrantClient")
    @patch("update_csv_tags.load_csv")
    @patch("update_csv_tags.CSV_PATH")
    def test_progress_logging_at_100_rows(
        self, mock_path, mock_load, mock_qclient, mock_get_tags, caplog
    ):
        """Log progress every 100 rows."""
        mock_path.exists.return_value = True
        rows = 101
        df = pd.DataFrame(
            {
                "வ.எ.": [str(i) for i in range(rows)],
                "ஆண்டு": ["1947"] * rows,
                "மலர்": ["1"] * rows,
                "இதழ்": ["1"] * rows,
                "தலைப்பு": ["t"] * rows,
                "ஆசிரியர்": ["a"] * rows,
            }
        )
        mock_load.return_value = df
        mock_get_tags.return_value = []
        with caplog.at_level(logging.INFO):
            with patch.object(df, "to_csv"):
                main()
        assert "Processed 100/" in caplog.text
