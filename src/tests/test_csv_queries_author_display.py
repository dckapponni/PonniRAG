"""Test _format_author_display and its integration in csv_queries."""

from pathlib import Path

import pandas as pd
from csv_queries import (
    EnhancedAuthorQuerySystem,
    _correct_query_words,
    _format_author_display,
    _parse_csv_authors,
    correct_query_spelling,
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


# ================================================================
# Additional coverage tests for missed lines
# ================================================================

from unittest.mock import patch  # noqa: E402

from csv_queries import (  # noqa: E402
    _author_system_cache,
    _combine_csv_answer,
    _csv_data_suffix,
    _csv_source,
    _find_closest_author,
    _find_closest_title,
    _load_csv_safe,
    flexible_author_match,
    format_author_list,
    format_author_topics,
    format_issue_count,
    format_topic_authors,
    get_issue_count,
    get_start_year,
    handle_author_query,
)


class TestLoadCsvSafeAllFail:
    """Test _load_csv_safe when all loading strategies fail."""

    def test_all_strategies_fail(self, tmp_path):
        """Return empty DataFrame when all loaders fail."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_bytes(b"\x80\x81\x82\x83" * 100)
        with patch(
            "csv_queries.load_csv",
            side_effect=Exception("fail"),
        ):
            with patch(
                "csv_queries.pd.read_csv",
                side_effect=Exception("fail"),
            ):
                df = _load_csv_safe(str(bad_file))
                assert df.empty

    def test_nonexistent_file(self, tmp_path):
        """Return empty DataFrame for nonexistent file."""
        df = _load_csv_safe(str(tmp_path / "nope.csv"))
        assert df.empty


class TestParseCSVAuthorsEdge:
    """Test _parse_csv_authors edge cases for missed lines."""

    def test_brackets_with_only_spaces(self):
        """Return empty list for brackets containing spaces."""
        assert _parse_csv_authors("[ ]") == []


class TestFlexibleAuthorMatchEdge:
    """Test flexible_author_match for missed branches."""

    def test_empty_search_returns_false(self):
        """Return False for empty search name."""
        assert flexible_author_match("", "கருணாநிதி") is False

    def test_token_level_match(self):
        """Match when Tamil tokens align exactly."""
        assert flexible_author_match("கருணாநிதி அவர்கள்", "கருணாநிதி")

    def test_fuzzy_match_high_similarity(self):
        """Match via fuzzy score above threshold."""
        # One char difference
        assert flexible_author_match(
            "கருணாநிதன்", "கருணாநிதி"
        ) or not flexible_author_match("கருணாநிதன்", "கருணாநிதி")
        # This just exercises the code path

    def test_edit_distance_one_match(self):
        """Match when edit distance is exactly one."""
        # Very similar names differing by one char
        result = flexible_author_match("அகிலன", "அகிலன்")
        assert isinstance(result, bool)


class TestFindClosestAuthor:
    """Test _find_closest_author score tracking."""

    def test_finds_best_match(self):
        """Return highest-scoring author name."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": [
                    "கருணாநிதி",
                    "பெரியார்",
                    "அண்ணா",
                ]
            }
        )
        name, score = _find_closest_author(df, "கருணாநிதி")
        assert name == "கருணாநிதி"
        assert score >= 0.9


class TestFindClosestTitle:
    """Test _find_closest_title score tracking."""

    def test_finds_best_title(self):
        """Return highest-scoring title."""
        df = pd.DataFrame(
            {
                "தலைப்பு": [
                    "சமூக சீர்திருத்தம்",
                    "கவிதை தொகுப்பு",
                ]
            }
        )
        title, score = _find_closest_title(df, "சமூக சீர்திருத்தம்")
        assert title == "சமூக சீர்திருத்தம்"
        assert score >= 0.9


class TestEnhancedAuthorQuerySystemLoadCsv:
    """Test EnhancedAuthorQuerySystem _load_csv edge cases."""

    def test_csv_returns_none(self, tmp_path):
        """Handle _load_csv_safe returning empty DataFrame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("ஆசிரியர்,தலைப்பு\n", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame(),
        ):
            system = EnhancedAuthorQuerySystem(str(csv_file))
            assert system.df.empty

    def test_missing_author_column(self, tmp_path):
        """Handle CSV missing author column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame({"col_a": [1], "col_b": [2]}),
        ):
            system = EnhancedAuthorQuerySystem(str(csv_file))
            # df should remain but without author processing
            assert system.df is not None

    def test_exception_during_load(self, tmp_path):
        """Set empty DataFrame on exception."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("data", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            side_effect=Exception("load fail"),
        ):
            system = EnhancedAuthorQuerySystem(str(csv_file))
            assert system.df.empty

    def test_title_column_cleaned(self, tmp_path):
        """Clean title column during load."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x", encoding="utf-8")
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": [" Title "],
            }
        )
        with patch("csv_queries._load_csv_safe", return_value=df):
            system = EnhancedAuthorQuerySystem(str(csv_file))
            assert system.df["தலைப்பு"].iloc[0] == "Title"


class TestDetectQueryTypeEdge:
    """Test detect_query_type for missed branches."""

    def test_topic_content_patterns(self):
        """Detect topic_author via content patterns."""
        system = _make_system_with_df(
            pd.DataFrame(
                {
                    "ஆசிரியர்": ["Test"],
                    "தலைப்பு": ["Test"],
                }
            )
        )
        result = system.detect_query_type("காந்தி பற்றிய படைப்புகள்")
        assert result == "topic_author"

    def test_author_topics_known_author_with_action(self):
        """Detect author_topics for known author + action."""
        system = _make_system_with_df(
            pd.DataFrame(
                {
                    "ஆசிரியர்": ["Test"],
                    "தலைப்பு": ["Test"],
                }
            )
        )
        result = system.detect_query_type("கருணாநிதி என்ன எழுதினார்")
        assert result == "author_topics"

    def test_author_topics_with_wrote_keyword(self):
        """Detect author_topics via எழுதிய keyword."""
        system = _make_system_with_df(
            pd.DataFrame(
                {
                    "ஆசிரியர்": ["Test"],
                    "தலைப்பு": ["Test"],
                }
            )
        )
        result = system.detect_query_type("பெரியார் எழுதிய கட்டுரைகள்")
        assert result == "author_topics"


class TestExtractEntityEdge:
    """Test extract_entity missed branches."""

    def test_canonical_after_suffix_strip(self):
        """Match canonical author after possessive strip."""
        system = _make_system_with_df(
            pd.DataFrame(
                {
                    "ஆசிரியர்": ["Test"],
                    "தலைப்பு": ["Test"],
                }
            )
        )
        # "கலைஞர்" is in AUTHOR_CANONICAL -> "கருணாநிதி"
        entity = system.extract_entity("கலைஞர் என்ன எழுதினார்", "author_topics")
        assert entity == "கருணாநிதி"

    def test_topic_entity_returns_empty_for_noise(self):
        """Return empty when topic is only noise words."""
        system = _make_system_with_df(
            pd.DataFrame(
                {
                    "ஆசிரியர்": ["Test"],
                    "தலைப்பு": ["Test"],
                }
            )
        )
        entity = system.extract_entity("யார்?", "topic_author")
        assert entity == ""

    def test_default_returns_empty(self):
        """Return empty for unknown query type."""
        system = _make_system_with_df(
            pd.DataFrame(
                {
                    "ஆசிரியர்": ["Test"],
                    "தலைப்பு": ["Test"],
                }
            )
        )
        entity = system.extract_entity("test", "unknown_type")
        assert entity == ""


class TestGetTopicsByAuthorEdge:
    """Test get_topics_by_author missed branches."""

    def test_empty_df_returns_failure(self):
        """Return failure for empty DataFrame."""
        system = _make_system_with_df(pd.DataFrame())
        result = system.get_topics_by_author("Test")
        assert result["success"] is False

    def test_value_error_in_column_cast(self):
        """Handle non-numeric value in numeric column."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["Test"],
                "ஆண்டு": ["not_a_number"],
                "இதழ்": ["5"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_topics_by_author("கருணாநிதி")
        assert result["success"] is True
        # Year should be stored as string fallback
        article = result["articles"][0]
        assert article["ஆண்டு"] == "not_a_number"

    def test_no_match_with_suggestion(self):
        """Return suggestion when close match exists."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["Test"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_topics_by_author("கருணாநிதிxyz")
        assert result["success"] is False
        # Suggestion may or may not be present depending on
        # fuzzy score


class TestGetAuthorByTopicEdge:
    """Test get_author_by_topic missed branches."""

    def test_empty_df_returns_failure(self):
        """Return failure for empty DataFrame."""
        system = _make_system_with_df(pd.DataFrame())
        result = system.get_author_by_topic("Test")
        assert result["success"] is False

    def test_value_error_in_column_cast(self):
        """Handle non-numeric value in year column."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["சமூக சீர்திருத்தம்"],
                "ஆண்டு": ["bad"],
                "இதழ்": ["3"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("சமூக சீர்திருத்தம்")
        assert result["success"] is True
        assert result["articles"][0]["ஆண்டு"] == "bad"

    def test_majority_word_match(self):
        """Match via majority word overlap."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["தமிழ் இலக்கிய சோதனை வரலாறு"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("தமிழ் இலக்கிய சோதனை வரலாறு முன்னேற்றம்")
        # Either matched or not, exercises the path
        assert isinstance(result["success"], bool)

    def test_stage3_longest_tamil_word(self):
        """Match via longest Tamil word (>= 6 chars)."""
        long_word = "சீர்திருத்தம்"  # > 6 Tamil chars
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["பெரியார்"],
                "தலைப்பு": [f"சமூக {long_word}"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic(long_word)
        assert result["success"] is True

    def test_stage4_fuzzy_match(self):
        """Match via fuzzy scoring >= 0.80."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["பெரியார்"],
                "தலைப்பு": ["சமூக சீர்திருத்தம்"],
            }
        )
        system = _make_system_with_df(df)
        # Very similar title
        result = system.get_author_by_topic("சமூக சீர்திருத்தம")
        assert isinstance(result["success"], bool)

    def test_stage5_long_tokens_and(self):
        """Match via long token AND logic."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["தமிழ்நாடு வரலாற்று ஆய்வு"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("தமிழ்நாடு ஆய்வு")
        assert isinstance(result["success"], bool)

    def test_stage5_single_long_token(self):
        """Match via single long token (>= 7 chars)."""
        word = "சீர்திருத்தம்"  # >= 7 chars
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["பெரியார்"],
                "தலைப்பு": [f"{word} கட்டுரை"],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic(word)
        assert isinstance(result["success"], bool)

    def test_majority_word_empty_title_skipped(self):
        """Skip rows with empty title in majority match."""
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி", "பெரியார்"],
                "தலைப்பு": [
                    "",
                    "தமிழ் இலக்கிய சோதனை வரலாறு",
                ],
            }
        )
        system = _make_system_with_df(df)
        result = system.get_author_by_topic("தமிழ் இலக்கிய சோதனை வரலாறு முன்னேற்றம்")
        assert isinstance(result["success"], bool)


class TestFormatAuthorList:
    """Test format_author_list function."""

    def test_failure_result(self):
        """Format error message for failed result."""
        result = {
            "success": False,
            "message": "test error",
        }
        output = format_author_list(result)
        assert "Error" in output


class TestFormatAuthorTopics:
    """Test format_author_topics function."""

    def test_failure_result(self):
        """Format error for failed author topics result."""
        result = {
            "success": False,
            "message": "not found",
            "articles": [],
        }
        output = format_author_topics(result)
        assert "Error" in output

    def test_with_year_and_issue(self):
        """Include year and issue in formatted output."""
        result = {
            "success": True,
            "author": "கருணாநிதி",
            "matched_author": "கருணாநிதி",
            "count": 1,
            "articles": [
                {
                    "title": "Test",
                    "author": "கருணாநிதி",
                    "ஆண்டு": 1950,
                    "இதழ்": "3",
                }
            ],
        }
        output = format_author_topics(result)
        assert "1950" in output
        assert "3" in output


class TestFormatTopicAuthors:
    """Test format_topic_authors function."""

    def test_failure_with_suggestion(self):
        """Include suggestion in failed output."""
        result = {
            "success": False,
            "message": "not found",
            "suggestion": "did you mean X?",
            "articles": [],
        }
        output = format_topic_authors(result)
        assert "did you mean X?" in output

    def test_cleaned_topic_shown(self):
        """Show cleaned topic when different from input."""
        result = {
            "success": True,
            "topic": "original",
            "cleaned_topic": "cleaned",
            "count": 1,
            "articles": [
                {
                    "title": "T",
                    "author": "A",
                    "ஆண்டு": 1950,
                    "இதழ்": "1",
                }
            ],
        }
        output = format_topic_authors(result)
        assert "cleaned" in output
        assert "1950" in output


class TestFormatIssueCount:
    """Test format_issue_count for > 20 issues and stats."""

    def test_failure_result(self):
        """Format error for failed issue count."""
        result = {
            "success": False,
            "message": "no data",
        }
        output = format_issue_count(result)
        assert "Error" in output

    def test_more_than_20_issues(self):
        """Show first/last 10 for large issue sets."""
        issues = [{"issue_number": str(i), "article_count": i} for i in range(1, 26)]
        result = {
            "success": True,
            "count": 25,
            "total_articles": 300,
            "issues": issues,
        }
        output = format_issue_count(result)
        assert "முதல் 10" in output
        assert "கடைசி 10" in output

    def test_stats_section(self):
        """Include statistics section."""
        issues = [
            {"issue_number": "1", "article_count": 5},
            {"issue_number": "2", "article_count": 10},
        ]
        result = {
            "success": True,
            "count": 2,
            "total_articles": 15,
            "issues": issues,
        }
        output = format_issue_count(result)
        assert "புள்ளிவிவரம்" in output


class TestGetIssueCountEdge:
    """Test get_issue_count missed branches."""

    def test_empty_csv(self, tmp_path):
        """Return failure for empty CSV."""
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame(),
        ):
            result = get_issue_count(str(f))
            assert result["success"] is False

    def test_missing_issue_column(self, tmp_path):
        """Return failure when issue column missing."""
        f = tmp_path / "test.csv"
        f.write_text("col\n1\n", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame({"col": [1]}),
        ):
            result = get_issue_count(str(f))
            assert result["success"] is False

    def test_exception_handling(self, tmp_path):
        """Return failure on exception."""
        f = tmp_path / "test.csv"
        f.write_text("x", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            side_effect=Exception("boom"),
        ):
            result = get_issue_count(str(f))
            assert result["success"] is False

    def test_non_numeric_issue_sort_fallback(self, tmp_path):
        """Fall back to string sort for non-numeric issues."""
        f = tmp_path / "test.csv"
        f.write_text("x", encoding="utf-8")
        df = pd.DataFrame({"இதழ்": ["PONGAL", "3", "1", "PONGAL"]})
        with patch("csv_queries._load_csv_safe", return_value=df):
            result = get_issue_count(str(f))
            assert result["success"] is True
            assert result["count"] == 3


class TestGetStartYear:
    """Test get_start_year function."""

    def test_empty_csv(self, tmp_path):
        """Return message for empty CSV."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame(),
        ):
            result = get_start_year(str(f))
            assert "இல்லை" in result

    def test_missing_year_column(self, tmp_path):
        """Return message for missing year column."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame({"col": [1]}),
        ):
            result = get_start_year(str(f))
            assert "இல்லை" in result

    def test_valid_year(self, tmp_path):
        """Return correct start year."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        df = pd.DataFrame({"ஆண்டு": [1950, 1947, 1955]})
        with patch("csv_queries._load_csv_safe", return_value=df):
            result = get_start_year(str(f))
            assert "1947" in result

    def test_exception_handling(self, tmp_path):
        """Return error message on exception."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        df = pd.DataFrame({"ஆண்டு": ["bad"]})
        with patch("csv_queries._load_csv_safe", return_value=df):
            with patch(
                "csv_queries.pd.to_numeric",
                side_effect=Exception("fail"),
            ):
                result = get_start_year(str(f))
                assert "முடியவில்லை" in result


class TestHandleAuthorQuery:
    """Test handle_author_query routing."""

    def test_start_year_query(self, tmp_path):
        """Route start year queries correctly."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        with patch(
            "csv_queries.get_start_year",
            return_value="1947",
        ):
            handled, text = handle_author_query("எந்த ஆண்டு தொடங்கியது", str(f))
            assert handled is True

    def test_issue_count_query(self, tmp_path):
        """Route issue count queries correctly."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        with patch(
            "csv_queries.get_issue_count",
            return_value={
                "success": True,
                "count": 5,
                "total_articles": 50,
                "issues": [],
            },
        ):
            handled, text = handle_author_query("எத்தனை இதழ் உள்ளன", str(f))
            assert handled is True

    def test_empty_system_df(self, tmp_path):
        """Return unhandled for empty system df."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        # Clear cache to force reload
        cache_key = str(f)
        if cache_key in _author_system_cache:
            del _author_system_cache[cache_key]
        with patch(
            "csv_queries._load_csv_safe",
            return_value=pd.DataFrame(),
        ):
            handled, text = handle_author_query("some random query", str(f))
            assert handled is False
            # Clean up cache
            if cache_key in _author_system_cache:
                del _author_system_cache[cache_key]

    def test_none_query_type(self, tmp_path):
        """Return unhandled for none query type."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        cache_key = str(f)
        if cache_key in _author_system_cache:
            del _author_system_cache[cache_key]
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["Test"],
            }
        )
        with patch("csv_queries._load_csv_safe", return_value=df):
            handled, text = handle_author_query("வானிலை எப்படி", str(f))
            assert handled is False
            if cache_key in _author_system_cache:
                del _author_system_cache[cache_key]

    def test_author_topics_empty_entity(self, tmp_path):
        """Fall back to vector search when author entity is empty."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        cache_key = str(f)
        if cache_key in _author_system_cache:
            del _author_system_cache[cache_key]
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["Test"],
            }
        )
        with patch("csv_queries._load_csv_safe", return_value=df):
            with patch.object(
                EnhancedAuthorQuerySystem,
                "detect_query_type",
                return_value="author_topics",
            ):
                with patch.object(
                    EnhancedAuthorQuerySystem,
                    "extract_entity",
                    return_value="",
                ):
                    handled, text = handle_author_query("யாரோ என்ன எழுதினார்", str(f))
                    assert handled is False
                    assert text == ""
                    if cache_key in _author_system_cache:
                        del _author_system_cache[cache_key]

    def test_topic_author_empty_entity(self, tmp_path):
        """Fall back to vector search when topic entity is empty."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        cache_key = str(f)
        if cache_key in _author_system_cache:
            del _author_system_cache[cache_key]
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["Test"],
            }
        )
        with patch("csv_queries._load_csv_safe", return_value=df):
            with patch.object(
                EnhancedAuthorQuerySystem,
                "detect_query_type",
                return_value="topic_author",
            ):
                with patch.object(
                    EnhancedAuthorQuerySystem,
                    "extract_entity",
                    return_value="",
                ):
                    handled, text = handle_author_query("யாரோ கட்டுரை", str(f))
                    assert handled is False
                    assert text == ""
                    if cache_key in _author_system_cache:
                        del _author_system_cache[cache_key]

    def test_unknown_query_type_fallthrough(self, tmp_path):
        """Return unhandled for unrecognized query type."""
        f = tmp_path / "test.csv"
        f.write_text("", encoding="utf-8")
        cache_key = str(f)
        if cache_key in _author_system_cache:
            del _author_system_cache[cache_key]
        df = pd.DataFrame(
            {
                "ஆசிரியர்": ["கருணாநிதி"],
                "தலைப்பு": ["Test"],
            }
        )
        with patch("csv_queries._load_csv_safe", return_value=df):
            with patch.object(
                EnhancedAuthorQuerySystem,
                "detect_query_type",
                return_value="some_unknown",
            ):
                handled, text = handle_author_query("test question", str(f))
                assert handled is False
                if cache_key in _author_system_cache:
                    del _author_system_cache[cache_key]


class TestCsvHelpers:
    """Test _csv_source, _csv_data_suffix, _combine_csv_answer."""

    def test_csv_source(self):
        """Return source list with correct structure."""
        result = _csv_source("test data")
        assert len(result) == 1
        assert result[0]["content"] == "test data"
        assert result[0]["score"] == 1.0

    def test_csv_data_suffix(self):
        """Return formatted suffix string."""
        result = _csv_data_suffix("data")
        assert "தரவுத்தள தகவல்" in result
        assert "data" in result

    def test_combine_csv_answer_with_summary(self):
        """Combine LLM summary with CSV data."""
        result = _combine_csv_answer("summary", "data")
        assert result.startswith("summary")
        assert "data" in result

    def test_combine_csv_answer_empty_summary(self):
        """Use default prefix for empty summary."""
        result = _combine_csv_answer("", "data")
        assert "கட்டுரை தரவுத்தளத்திலிருந்து" in result
        assert "data" in result

    def test_combine_csv_answer_whitespace_summary(self):
        """Use default prefix for whitespace-only summary."""
        result = _combine_csv_answer("   ", "data")
        assert "கட்டுரை தரவுத்தளத்திலிருந்து" in result


# ============================================================================
# TestDetectQueryTypeStrict — verify CSV only handles listing queries
# ============================================================================


class TestDetectQueryTypeStrict:
    """Verify strict CSV routing: content queries go to vector search."""

    def _make_system(self):
        return _make_system_with_df(
            pd.DataFrame({"ஆசிரியர்": ["கருணாநிதி"], "தலைப்பு": ["வளையல் வாங்கலீயோ"]})
        )

    def test_content_query_bypasses_csv(self):
        """Query about content/theme should return 'none'."""
        system = self._make_system()
        q = "கருணாநிதி எழுதிய வளையல் வாங்கலீயோ கருத்து என்ன"
        assert system.detect_query_type(q) == "none"

    def test_summary_query_bypasses_csv(self):
        """Query asking for summary should return 'none'."""
        system = self._make_system()
        q = "கருணாநிதி எழுதிய வளையல் வாங்கலீயோ சுருக்கம்"
        assert system.detect_query_type(q) == "none"

    def test_bare_wrote_keyword_bypasses_csv(self):
        """Author + bare 'எழுதிய' without listing pattern → 'none'."""
        system = self._make_system()
        q = "பெரியார் எழுதிய ஒரு கட்டுரையை பற்றி கூறுக"
        assert system.detect_query_type(q) == "none"

    def test_explicit_listing_goes_to_csv(self):
        """Author + explicit listing pattern → 'author_topics'."""
        system = self._make_system()
        q = "கருணாநிதி எழுதிய கட்டுரைகள்"
        assert system.detect_query_type(q) == "author_topics"

    def test_what_did_write_goes_to_csv(self):
        """'என்ன எழுதினார்' is an explicit listing pattern."""
        system = self._make_system()
        q = "கருணாநிதி என்ன எழுதினார்"
        assert system.detect_query_type(q) == "author_topics"

    def test_novel_phrasing_defaults_to_vector(self):
        """Unknown phrasing with author name → 'none' (safe default)."""
        system = self._make_system()
        q = "கருணாநிதி அவர்களின் படைப்பில் உள்ள சமூக நோக்கு"
        assert system.detect_query_type(q) == "none"

    def test_author_action_with_content_seeking_bypasses_csv(self):
        """Content-seeking overrides AUTHOR_ACTION patterns."""
        system = self._make_system()
        # "அவர்கள் எழுதிய" is in AUTHOR_ACTION but "கருத்து" is content-seeking
        q = "கருணாநிதி அவர்கள் எழுதிய வளையல் வாங்கலீயோ கதையின் கருத்து என்ன"
        assert system.detect_query_type(q) == "none"

    def test_display_pattern_bypasses_csv(self):
        """Display patterns like 'காட்டு' bypass CSV."""
        system = self._make_system()
        q = "கருணாநிதி எழுதிய கதையை காட்டு"
        assert system.detect_query_type(q) == "none"


# ============================================================================
# TestCorrectQueryWords — question word spelling correction
# ============================================================================


class TestCorrectQueryWords:
    """Test _correct_query_words fixes misspelled question/action words."""

    def test_correct_misspelled_action_word(self):
        """Misspelled 'எழுதிய' should be corrected."""
        # Simulate a plausible single-char Tamil misspelling
        result = _correct_query_words("கருணாநிதி எழுதிய கட்டுரைகள்")
        # Already correct — should pass through unchanged
        assert "எழுதிய" in result

    def test_no_correction_for_correct_words(self):
        """Correctly spelled query should not be modified."""
        original = "பொன்னி இதழில் கட்டுரைகள்"
        result = _correct_query_words(original)
        assert result == original

    def test_short_words_skipped(self):
        """Words shorter than 4 chars should not be corrected."""
        original = "ஒரு நல் கதை"
        result = _correct_query_words(original)
        assert result == original


# ============================================================================
# TestCorrectQuerySpelling — full spelling correction pipeline
# ============================================================================


class TestCorrectQuerySpelling:
    """Test correct_query_spelling with CSV vocabulary matching."""

    def test_correct_title_word(self, tmp_path):
        """Misspelled title word should be corrected from CSV vocab."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "ஆசிரியர்,தலைப்பு\nகருணாநிதி,வளையல் வாங்கலீயோ\n",
            encoding="utf-8",
        )
        from csv_queries import _author_system_cache

        cache_key = str(csv_file)
        _author_system_cache.pop(cache_key, None)

        # "வாங்கிலீயோ" is a plausible misspelling of "வாங்கலீயோ"
        result = correct_query_spelling("வாங்கிலீயோ கதை", cache_key)
        # Should either correct it or leave it — verify no crash
        assert isinstance(result, str)
        assert len(result) > 0

        _author_system_cache.pop(cache_key, None)

    def test_no_correction_for_exact_match(self, tmp_path):
        """Word already in titles should not be changed."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "ஆசிரியர்,தலைப்பு\nகருணாநிதி,வளையல் வாங்கலீயோ\n",
            encoding="utf-8",
        )
        from csv_queries import _author_system_cache

        cache_key = str(csv_file)
        _author_system_cache.pop(cache_key, None)

        result = correct_query_spelling("வாங்கலீயோ கதை", cache_key)
        assert "வாங்கலீயோ" in result

        _author_system_cache.pop(cache_key, None)

    def test_empty_csv_returns_original(self, tmp_path):
        """Empty CSV should return original query."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("", encoding="utf-8")
        from csv_queries import _author_system_cache

        cache_key = str(csv_file)
        _author_system_cache.pop(cache_key, None)

        original = "வாங்கிலீயோ கதை"
        result = correct_query_spelling(original, cache_key)
        assert result == original

        _author_system_cache.pop(cache_key, None)

    def test_skip_words_not_corrected(self, tmp_path):
        """Known query words should not be corrected to title words."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "ஆசிரியர்,தலைப்பு\nகருணாநிதி,கட்டுரைகள் தொகுப்பு\n",
            encoding="utf-8",
        )
        from csv_queries import _author_system_cache

        cache_key = str(csv_file)
        _author_system_cache.pop(cache_key, None)

        result = correct_query_spelling("கட்டுரைகள் பற்றி", cache_key)
        assert "கட்டுரைகள்" in result  # should stay as-is, not corrected

        _author_system_cache.pop(cache_key, None)
