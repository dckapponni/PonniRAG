"""Test _format_author_display and its integration in csv_queries."""

from pathlib import Path

import pandas as pd
from csv_queries import (
    EnhancedAuthorQuerySystem,
    _format_author_display,
    _parse_csv_authors,
)

# ============================================================================
# TestParseCSVAuthors
# ============================================================================


class TestParseCSVAuthors:
    """Test _parse_csv_authors bracket-wrapped multi-author parsing."""

    def test_empty_string_returns_empty_list(self):
        """Return empty list for empty string input."""
        assert _parse_csv_authors("") == []

    def test_none_returns_empty_list(self):
        """Return empty list for None input."""
        assert _parse_csv_authors(None) == []

    def test_na_string_returns_empty_list(self):
        """Return empty list for NA sentinel."""
        assert _parse_csv_authors("NA") == []

    def test_nan_string_returns_empty_list(self):
        """Filter out pandas NaN converted to str('nan')."""
        assert _parse_csv_authors("nan") == []

    def test_none_string_returns_empty_list(self):
        """Return empty list for 'None' string."""
        assert _parse_csv_authors("None") == []

    def test_empty_bracket_returns_empty_list(self):
        """Return empty list for empty brackets."""
        assert _parse_csv_authors("[]") == []

    def test_single_author_no_brackets(self):
        """Parse legacy plain author name without brackets."""
        result = _parse_csv_authors("நக்கீரன்")
        assert result == ["நக்கீரன்"]

    def test_single_author_with_brackets(self):
        """Parse single author wrapped in brackets."""
        result = _parse_csv_authors("[நக்கீரன்]")
        assert result == ["நக்கீரன்"]

    def test_multiple_authors_with_brackets(self):
        """Parse multi-author bracket format."""
        raw = "[பாண்டியன், நா. வேத்தரசன், வணங்காமுடி]"
        result = _parse_csv_authors(raw)
        assert result == [
            "பாண்டியன்",
            "நா. வேத்தரசன்",
            "வணங்காமுடி",
        ]

    def test_two_authors_with_brackets(self):
        """Parse two authors in bracket format."""
        result = _parse_csv_authors("[கருணாநிதி, பெரியார்]")
        assert len(result) == 2
        assert "கருணாநிதி" in result
        assert "பெரியார்" in result

    def test_whitespace_trimmed_from_each_author(self):
        """Strip leading/trailing whitespace inside brackets."""
        result = _parse_csv_authors("[  அண்ணா  ,  பாரதிதாசன்  ]")
        assert result == ["அண்ணா", "பாரதிதாசன்"]

    def test_na_sentinel_filtered_within_multi_author(self):
        """Filter NA entries inside bracket list."""
        result = _parse_csv_authors("[நக்கீரன், NA, nan]")
        assert result == ["நக்கீரன்"]

    def test_author_with_period_in_name(self):
        """Preserve Tamil author abbreviations like 'நா.'."""
        result = _parse_csv_authors("[நா. வேத்தரசன்]")
        assert result == ["நா. வேத்தரசன்"]

    def test_single_author_with_whitespace_only_entry_filtered(self):
        """Filter blank entries after strip."""
        result = _parse_csv_authors("[நக்கீரன், ,  ]")
        assert result == ["நக்கீரன்"]

    def test_returns_list_type(self):
        """Return a list type."""
        result = _parse_csv_authors("கருணாநிதி")
        assert isinstance(result, list)


# ============================================================================
# TestFormatAuthorDisplay
# ============================================================================


class TestFormatAuthorDisplay:
    """Test _format_author_display human-readable formatter."""

    def test_empty_string_returns_na(self):
        """Return NA for empty string."""
        assert _format_author_display("") == "NA"

    def test_none_returns_na(self):
        """Return NA for None."""
        assert _format_author_display(None) == "NA"

    def test_na_string_returns_na(self):
        """Return NA for NA sentinel."""
        assert _format_author_display("NA") == "NA"

    def test_nan_string_returns_na(self):
        """Return NA for nan string."""
        assert _format_author_display("nan") == "NA"

    def test_empty_brackets_returns_na(self):
        """Return NA for empty brackets."""
        assert _format_author_display("[]") == "NA"

    def test_single_author_no_brackets(self):
        """Return plain name as-is for legacy format."""
        assert _format_author_display("நக்கீரன்") == "நக்கீரன்"

    def test_single_author_with_brackets_strips_brackets(self):
        """Remove bracket wrapper and return plain name."""
        assert _format_author_display("[நக்கீரன்]") == "நக்கீரன்"

    def test_multiple_authors_joined_with_comma_space(self):
        """Join multiple authors with comma-space, no brackets."""
        result = _format_author_display("[பாண்டியன், நா. வேத்தரசன், வணங்காமுடி]")
        assert result == ("பாண்டியன், நா. வேத்தரசன், வணங்காமுடி")

    def test_two_authors_joined(self):
        """Join two authors with comma-space."""
        result = _format_author_display("[கருணாநிதி, பெரியார்]")
        assert result == "கருணாநிதி, பெரியார்"

    def test_output_contains_no_square_brackets(self):
        """Verify no brackets in formatted output."""
        result = _format_author_display("[அண்ணா, பாரதிதாசன்]")
        assert "[" not in result
        assert "]" not in result

    def test_output_is_string_type(self):
        """Return string type for all inputs."""
        assert isinstance(_format_author_display("நக்கீரன்"), str)
        assert isinstance(_format_author_display(""), str)

    def test_na_entries_in_bracket_list_excluded(self):
        """Exclude NA sentinels from output."""
        result = _format_author_display("[நக்கீரன், NA]")
        assert "NA" not in result
        assert result == "நக்கீரன்"

    def test_author_with_period_preserved(self):
        """Preserve periods in Tamil author abbreviations."""
        result = _format_author_display("[நா. வேத்தரசன்]")
        assert result == "நா. வேத்தரசன்"

    def test_whitespace_only_na_list_returns_na(self):
        """Collapse bracket with only whitespace entries to NA."""
        result = _format_author_display("[  ,  ,  ]")
        assert result == "NA"


# ============================================================================
# TestGetTopicsByAuthorArticleFormat
# ============================================================================


def _make_system_with_df(df: pd.DataFrame) -> EnhancedAuthorQuerySystem:
    """Create an EnhancedAuthorQuerySystem with in-memory DataFrame."""
    system = object.__new__(EnhancedAuthorQuerySystem)
    system.csv_path = Path("/fake/test.csv")
    system.df = df
    return system


class TestGetTopicsByAuthorArticleFormat:
    """Test get_topics_by_author formats author without brackets."""

    def _make_df(self, rows):
        """Build a DataFrame from row dicts."""
        return pd.DataFrame(rows)

    def test_single_author_article_no_brackets_in_output(self):
        """Verify single author has no brackets in output."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "கருணாநிதி",
                    "தலைப்பு": "திராவிட கட்டுரை",
                    "ஆண்டு": 1950,
                    "இதழ்": "3",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_topics_by_author("கருணாநிதி")
        assert result["success"] is True
        article = result["articles"][0]
        assert "author" in article
        assert "[" not in article["author"]
        assert "]" not in article["author"]
        assert article["author"] == "கருணாநிதி"

    def test_multi_author_article_formatted_as_comma_joined(self):
        """Verify multi-author uses comma-joined format."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "[கருணாநிதி, பெரியார்]",
                    "தலைப்பு": "கூட்டு கட்டுரை",
                    "ஆண்டு": 1951,
                    "இதழ்": "5",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_topics_by_author("கருணாநிதி")
        assert result["success"] is True
        article = result["articles"][0]
        assert "[" not in article["author"]
        assert "]" not in article["author"]
        assert "கருணாநிதி" in article["author"]
        assert "பெரியார்" in article["author"]
        assert ", " in article["author"]

    def test_na_author_displayed_as_na_string(self):
        """Verify NA author field produces 'NA' string."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "NA",
                    "தலைப்பு": "அநாமிக கட்டுரை",
                    "ஆண்டு": 1948,
                    "இதழ்": "1",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_topics_by_author("NA")
        if result["success"]:
            article = result["articles"][0]
            assert article["author"] == "NA"

    def test_bracket_empty_author_displayed_as_na(self):
        """Verify '[]' author produces 'NA' in article dict."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "கருணாநிதி",
                    "தலைப்பு": "Article A",
                    "ஆண்டு": 1950,
                    "இதழ்": "2",
                },
                {
                    "ஆசிரியர்": "[]",
                    "தலைப்பு": "Article B",
                    "ஆண்டு": 1950,
                    "இதழ்": "2",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_topics_by_author("கருணாநிதி")
        assert result["success"] is True
        assert len(result["articles"]) == 1
        assert result["articles"][0]["author"] == "கருணாநிதி"


# ============================================================================
# TestGetAuthorByTopicArticleFormat
# ============================================================================


class TestGetAuthorByTopicArticleFormat:
    """Test get_author_by_topic formats author without brackets."""

    def _make_df(self, rows):
        """Build a DataFrame from row dicts."""
        return pd.DataFrame(rows)

    def test_single_author_no_brackets(self):
        """Verify single author has no brackets."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "பெரியார்",
                    "தலைப்பு": "சமூக சீர்திருத்தம்",
                    "ஆண்டு": 1949,
                    "இதழ்": "4",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("சமூக சீர்திருத்தம்")
        assert result["success"] is True
        article = result["articles"][0]
        assert "[" not in article["author"]
        assert article["author"] == "பெரியார்"

    def test_multi_author_bracket_format_produces_comma_joined(self):
        """Verify bracket format produces comma-joined output."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "[பெரியார், அண்ணா]",
                    "தலைப்பு": "திராவிட தத்துவம்",
                    "ஆண்டு": 1950,
                    "இதழ்": "6",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("திராவிட தத்துவம்")
        assert result["success"] is True
        article = result["articles"][0]
        assert "[" not in article["author"]
        assert "]" not in article["author"]
        assert "பெரியார்" in article["author"]
        assert "அண்ணா" in article["author"]
        assert ", " in article["author"]

    def test_three_authors_joined_correctly(self):
        """Verify three authors joined with comma-space."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": ("[பாண்டியன், நா. வேத்தரசன்," " வணங்காமுடி]"),
                    "தலைப்பு": "கூட்டு ஆய்வு கட்டுரை",
                    "ஆண்டு": 1952,
                    "இதழ்": "8",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("கூட்டு ஆய்வு கட்டுரை")
        assert result["success"] is True
        article = result["articles"][0]
        assert article["author"] == ("பாண்டியன், நா. வேத்தரசன், வணங்காமுடி")

    def test_na_author_field_produces_na_string(self):
        """Verify NA author field produces 'NA' string."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "NA",
                    "தலைப்பு": "அநாமிக தலைப்பு",
                    "ஆண்டு": 1947,
                    "இதழ்": "1",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("அநாமிக தலைப்பு")
        if result["success"]:
            article = result["articles"][0]
            assert article["author"] == "NA"

    def test_nan_string_author_field_produces_na_string(self):
        """Verify pandas NaN string produces 'NA' in output."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "nan",
                    "தலைப்பு": "தலைப்பு உடன்",
                    "ஆண்டு": 1948,
                    "இதழ்": "2",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("தலைப்பு உடன்")
        if result["success"]:
            article = result["articles"][0]
            assert article["author"] == "NA"

    def test_article_dict_has_author_key(self):
        """Verify 'author' key is present in every article."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "கண்ணதாசன்",
                    "தலைப்பு": "கவிதை தலைப்பு",
                    "ஆண்டு": 1953,
                    "இதழ்": "10",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("கவிதை தலைப்பு")
        assert result["success"] is True
        for article in result["articles"]:
            assert "author" in article

    def test_no_match_returns_empty_articles_list(self):
        """Return empty articles list when topic is not found."""
        df = self._make_df(
            [
                {
                    "ஆசிரியர்": "பெரியார்",
                    "தலைப்பு": "சமூக சீர்திருத்தம்",
                    "ஆண்டு": 1949,
                    "இதழ்": "4",
                },
            ]
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("nonexistent xyz abc def ghi")
        assert result["success"] is False
        assert result["articles"] == []
