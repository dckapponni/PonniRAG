"""Tests for article seperation module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_extraction.article_seperation import (  # noqa: E402
    extract_year_from_s3_key,
    is_file_already_processed,
    load_csv_from_local,
    parse_tamil_document_csv_first,
    process_s3_files,
    save_authors_to_s3,
    setup_logging,
    validate_processed_file,
)


@pytest.fixture
def sample_csv_df():
    """
    Provide sample CSV dataframe for testing.

    Returns:
        pd.DataFrame: DataFrame with sample article data
    """
    return pd.DataFrame(
        {
            "title": ["Article 1", "Article 2", "Article 3"],
            "author": ["Author 1", "Author 2", "Author 3"],
            "content": ["Content 1", "Content 2", "Content 3"],
        }
    )


@pytest.fixture
def sample_lines():
    """
    Provide sample text lines representing Tamil document structure.

    Returns:
        list: Lines of text mimicking a Tamil document with metadata and content
    """
    return [
        "மலர் 1",
        "இதழ் 2",
        "பொருளடக்கம்",
        "Author 1",
        "Author 2",
        "Author 3",
        "ஆகியோரின் எழுத்தோவியங்கள்",
        "",
        "Article Title",
        "Author 1",
        "Content line 1",
        "Content line 2",
        "Content line 3",
        "Content line 4",
        "Content line 5",
    ]


@pytest.fixture
def mock_logger():
    """
    Provide mock logger to prevent log file creation during tests.

    Yields:
        MagicMock: Mocked logger instance
    """
    with patch("data_extraction.article_seperation.logger") as mock_log:
        yield mock_log


class TestSetupLogging:
    """Test suite for logging setup functionality."""

    @patch("data_extraction.article_seperation.logging.FileHandler")
    @patch("data_extraction.article_seperation.logging.StreamHandler")
    @patch("data_extraction.article_seperation.Path")
    @patch("logging.getLogger")
    def test_setup_logging_creates_handlers(
        self, mock_get_logger, mock_path, mock_stream_handler, mock_file_handler
    ):
        """
        Test that setup_logging creates and configures file and console handlers.

        Verifies that both handlers are added and logger level is set.
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_logger.handlers = []

        mock_log_dir = MagicMock()
        mock_log_dir.__truediv__ = MagicMock(return_value="/tmp/logs/test.log")
        mock_path.return_value = mock_log_dir
        mock_log_dir.mkdir = MagicMock()

        mock_file_handler_instance = MagicMock()
        mock_stream_handler_instance = MagicMock()
        mock_file_handler.return_value = mock_file_handler_instance
        mock_stream_handler.return_value = mock_stream_handler_instance

        setup_logging("test.log")

        assert mock_logger.addHandler.call_count == 2
        mock_logger.setLevel.assert_called()

    @patch("data_extraction.article_seperation.logging.FileHandler")
    @patch("data_extraction.article_seperation.logging.StreamHandler")
    @patch("data_extraction.article_seperation.Path")
    @patch("logging.getLogger")
    def test_setup_logging_clears_existing_handlers(
        self, mock_get_logger, mock_path, mock_stream_handler, mock_file_handler
    ):
        """
        Test that existing handlers are cleared before adding new ones.

        Ensures clean logger state by replacing old handlers.
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        existing_handlers = [MagicMock(), MagicMock()]
        mock_logger.handlers = existing_handlers

        mock_log_dir = MagicMock()
        mock_log_dir.__truediv__ = MagicMock(return_value="/tmp/logs/test.log")
        mock_path.return_value = mock_log_dir
        mock_log_dir.mkdir = MagicMock()

        mock_file_handler_instance = MagicMock()
        mock_stream_handler_instance = MagicMock()
        mock_file_handler.return_value = mock_file_handler_instance
        mock_stream_handler.return_value = mock_stream_handler_instance

        setup_logging("test.log")

        assert mock_logger.addHandler.call_count == 2


class TestExtractYearFromS3Key:
    """Test suite for year extraction from S3 file paths."""

    def test_extract_year_2020s(self):
        """Test extraction of year from 2020s."""
        assert extract_year_from_s3_key("extracted_text/2023/document.txt") == "2023"

    def test_extract_year_1990s(self):
        """Test extraction of year from 1990s."""
        assert extract_year_from_s3_key("extracted_text/1995/document.txt") == "1995"

    def test_extract_year_2000s(self):
        """Test extraction of year from 2000s."""
        assert extract_year_from_s3_key("extracted_text/2005/document.txt") == "2005"

    def test_extract_year_2010s(self):
        """Test extraction of year from 2010s."""
        assert extract_year_from_s3_key("extracted_text/2015/document.txt") == "2015"

    def test_extract_year_no_year(self):
        """Test behavior when no year is present in path."""
        assert extract_year_from_s3_key("extracted_text/document.txt") == "Unknown"

    def test_extract_year_invalid_too_old(self):
        """Test handling of years before valid range."""
        result = extract_year_from_s3_key("extracted_text/1899/document.txt")
        assert isinstance(result, str)

    def test_extract_year_edge_1900(self):
        """Test edge case with year 1900."""
        result = extract_year_from_s3_key("extracted_text/1900/document.txt")
        assert result == "1900" or result == "Unknown"

    def test_extract_year_current_year(self):
        """Test extraction of current year."""
        import datetime

        current_year = datetime.datetime.now().year
        result = extract_year_from_s3_key(f"extracted_text/{current_year}/document.txt")
        assert result == str(current_year)

    def test_extract_year_multiple_years(self):
        """Test behavior when multiple year patterns exist in path."""
        result = extract_year_from_s3_key(
            "extracted_text/2020/backup_2021/document.txt"
        )
        assert result in ["2020", "2021"]

    def test_extract_year_exception_handling(self):
        """Test exception handling with empty string input."""
        result = extract_year_from_s3_key("")
        assert result == "Unknown"

    def test_extract_year_none_input(self):
        """Test handling of None input."""
        try:
            result = extract_year_from_s3_key(None)
            assert result == "Unknown"
        except (TypeError, AttributeError):
            pass


class TestIsFileAlreadyProcessed:
    """Test suite for checking file processing status."""

    @patch("data_extraction.article_seperation.file_exists")
    def test_file_exists(self, mock_file_exists):
        """Test detection of existing processed file."""
        mock_file_exists.return_value = True
        result = is_file_already_processed("test-bucket", "output/test.json")
        assert result is True

    @patch("data_extraction.article_seperation.file_exists")
    def test_file_not_exists(self, mock_file_exists):
        """Test detection of non-existent file."""
        mock_file_exists.return_value = False
        result = is_file_already_processed("test-bucket", "output/test.json")
        assert result is False

    @patch("data_extraction.article_seperation.file_exists")
    def test_exception_returns_false(self, mock_file_exists):
        """Test that exceptions are caught and return False."""
        mock_file_exists.side_effect = Exception("S3 error")
        result = is_file_already_processed("test-bucket", "output/test.json")
        assert result is False

    @patch("data_extraction.article_seperation.file_exists")
    def test_with_empty_bucket(self, mock_file_exists):
        """Test behavior with empty bucket name."""
        mock_file_exists.return_value = False
        result = is_file_already_processed("", "output/test.json")
        assert result is False

    @patch("data_extraction.article_seperation.file_exists")
    def test_with_empty_key(self, mock_file_exists):
        """Test behavior with empty key."""
        mock_file_exists.return_value = False
        result = is_file_already_processed("bucket", "")
        assert result is False


class TestValidateProcessedFile:
    """Test suite for validating processed JSON file structure."""

    @patch("s3_utils.read_json_from_s3")
    def test_valid_file_with_articles(self, mock_read_json):
        """Test validation of properly structured file with articles."""
        mock_read_json.return_value = {
            "articles": [{"title": "Test", "content": "Content"}]
        }
        assert validate_processed_file("bucket", "key") is True

    @patch("s3_utils.read_json_from_s3")
    def test_valid_file_empty_articles(self, mock_read_json):
        """Test validation of file with empty articles list."""
        mock_read_json.return_value = {"articles": []}
        assert validate_processed_file("bucket", "key") is True

    @patch("s3_utils.read_json_from_s3")
    def test_invalid_not_dict(self, mock_read_json):
        """Test rejection of non-dictionary data structure."""
        mock_read_json.return_value = ["not", "dict"]
        assert validate_processed_file("bucket", "key") is False

    @patch("s3_utils.read_json_from_s3")
    def test_invalid_no_articles_key(self, mock_read_json):
        """Test rejection when 'articles' key is missing."""
        mock_read_json.return_value = {"data": "test"}
        assert validate_processed_file("bucket", "key") is False

    @patch("s3_utils.read_json_from_s3")
    def test_invalid_articles_not_list(self, mock_read_json):
        """Test rejection when 'articles' value is not a list."""
        mock_read_json.return_value = {"articles": "not a list"}
        assert validate_processed_file("bucket", "key") is False

    @patch("s3_utils.read_json_from_s3")
    def test_json_decode_error(self, mock_read_json):
        """Test handling of JSON decode errors."""
        mock_read_json.side_effect = json.JSONDecodeError("error", "", 0)
        assert validate_processed_file("bucket", "key") is False

    @patch("s3_utils.read_json_from_s3")
    def test_generic_exception(self, mock_read_json):
        """Test handling of generic exceptions."""
        mock_read_json.side_effect = Exception("Generic error")
        assert validate_processed_file("bucket", "key") is False

    @patch("s3_utils.read_json_from_s3")
    def test_none_return(self, mock_read_json):
        """Test handling of None return value."""
        mock_read_json.return_value = None
        assert validate_processed_file("bucket", "key") is False

    @patch("s3_utils.read_json_from_s3")
    def test_valid_with_extra_fields(self, mock_read_json):
        """Test validation accepts files with extra metadata fields."""
        mock_read_json.return_value = {
            "articles": [{"title": "Test"}],
            "metadata": {"doc_id": "test"},
            "timestamp": "2024",
        }
        assert validate_processed_file("bucket", "key") is True


class TestLoadCSVFromLocal:
    """Test suite for CSV file loading functionality."""

    def test_load_csv_success(self, tmp_path):
        """Test successful loading of valid CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("title,author\nTest,Author\n", encoding="utf-8")
        df = load_csv_from_local(csv_file)
        assert len(df) == 1
        assert "title" in df.columns

    def test_load_csv_not_found(self):
        """Test FileNotFoundError when CSV doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_csv_from_local("/nonexistent/file.csv")

    def test_load_csv_empty(self, tmp_path):
        """Test handling of empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")
        with pytest.raises(pd.errors.EmptyDataError):
            load_csv_from_local(csv_file)

    def test_load_csv_bad_lines(self, tmp_path):
        """Test handling of malformed CSV lines."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("col1,col2\nval1,val2\nbad\nval3,val4\n", encoding="utf-8")
        df = load_csv_from_local(csv_file)
        assert len(df) >= 1

    def test_load_csv_unicode(self, tmp_path):
        """Test loading CSV with Unicode (Tamil) characters."""
        csv_file = tmp_path / "unicode.csv"
        csv_file.write_text("title\nதமிழ்\n", encoding="utf-8")
        df = load_csv_from_local(csv_file)
        assert len(df) == 1

    def test_load_csv_parser_error(self, tmp_path):
        """Test handling of CSV parser errors."""
        csv_file = tmp_path / "malformed.csv"
        csv_file.write_text('col1,col2\n"unclosed quote,val2\n', encoding="utf-8")
        try:
            df = load_csv_from_local(csv_file)
            assert isinstance(df, pd.DataFrame)
        except pd.errors.ParserError:
            pass

    def test_load_csv_with_path_object(self, tmp_path):
        """Test loading CSV using Path object."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1\nval1\n", encoding="utf-8")
        df = load_csv_from_local(Path(csv_file))
        assert len(df) == 1

    def test_load_csv_generic_exception(self, tmp_path):
        """Test handling of generic exceptions during CSV loading."""
        with patch("pandas.read_csv", side_effect=Exception("Generic error")):
            csv_file = tmp_path / "test.csv"
            csv_file.write_text("col1\nval1\n", encoding="utf-8")
            with pytest.raises(Exception):
                load_csv_from_local(csv_file)


class TestSaveAuthorsToS3:
    """Test suite for saving author information to S3."""

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_save_new_authors_dict_format(self, mock_read, mock_exists, mock_upload):
        """Test saving new authors in dictionary format."""
        mock_exists.return_value = False
        authors = [{"author_name": "Author 1"}, {"author_name": "Author 2"}]

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", authors)

        saved_data = mock_upload.call_args[0][2]
        assert len(saved_data) == 1
        assert saved_data[0]["authors"] == ["Author 1", "Author 2"]

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_save_authors_string_format(self, mock_read, mock_exists, mock_upload):
        """Test saving authors in string list format."""
        mock_exists.return_value = False
        authors = ["Author 1", "Author 2"]

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", authors)

        saved_data = mock_upload.call_args[0][2]
        assert saved_data[0]["authors"] == ["Author 1", "Author 2"]

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_save_authors_mixed_format(self, mock_read, mock_exists, mock_upload):
        """Test conversion of non-string authors to strings."""
        mock_exists.return_value = False
        authors = [123, 456]

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", authors)

        saved_data = mock_upload.call_args[0][2]
        assert saved_data[0]["authors"] == ["123", "456"]

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_update_existing_authors(self, mock_read, mock_exists, mock_upload):
        """Test updating authors for existing document entry."""
        mock_exists.return_value = True
        mock_read.return_value = [
            {"doc_id": "மலர்_1", "doc_issue": "இதழ்_1", "authors": ["Old"]}
        ]

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", ["New"])

        saved_data = mock_upload.call_args[0][2]
        assert saved_data[0]["authors"] == ["New"]

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_add_new_document(self, mock_read, mock_exists, mock_upload):
        """Test adding new document entry to existing authors file."""
        mock_exists.return_value = True
        mock_read.return_value = [
            {"doc_id": "மலர்_1", "doc_issue": "இதழ்_1", "authors": ["Author 1"]}
        ]

        save_authors_to_s3("bucket", "output/", "மலர்_2", "இதழ்_2", ["Author 2"])

        saved_data = mock_upload.call_args[0][2]
        assert len(saved_data) == 2

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    def test_empty_authors_list(self, mock_exists, mock_upload):
        """Test saving empty authors list."""
        mock_exists.return_value = False

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", [])

        saved_data = mock_upload.call_args[0][2]
        assert saved_data[0]["authors"] == []

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_read_json_exception(self, mock_read, mock_exists, mock_upload):
        """Test handling of exception when reading existing authors file."""
        mock_exists.return_value = True
        mock_read.side_effect = Exception("Read error")

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", ["Author"])

        assert mock_upload.called

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    def test_upload_exception(self, mock_exists, mock_upload):
        """Test handling of exception during upload."""
        mock_exists.return_value = False
        mock_upload.side_effect = Exception("Upload error")

        with pytest.raises(Exception):
            save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", ["Author"])


class TestParseTamilDocumentCSVFirst:
    """Test suite for Tamil document parsing with CSV-first strategy."""

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    def test_csv_success(self, mock_csv, sample_lines, sample_csv_df):
        """Test successful extraction using CSV method."""
        mock_csv.return_value = {
            "articles": [
                {
                    "title": "Test",
                    "author": "Author",
                    "content": "Content",
                    "year": "2023",
                }
            ]
        }

        result = parse_tamil_document_csv_first(
            sample_lines, {}, sample_csv_df, "extracted_text/2023/test.txt"
        )

        assert len(result["articles"]) == 1
        assert result["articles"][0]["source_document"] == "test.txt"
        assert result["articles"][0]["year"] == "2023"

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    def test_csv_success_without_year(self, mock_csv, sample_lines, sample_csv_df):
        """Test that year is extracted from S3 key when not present in CSV."""
        mock_csv.return_value = {
            "articles": [
                {
                    "title": "Test",
                    "author": "Author",
                    "content": "Content",
                    "year": "Unknown",
                }
            ]
        }

        result = parse_tamil_document_csv_first(
            sample_lines, {}, sample_csv_df, "extracted_text/2020/test.txt"
        )

        assert result["articles"][0]["year"] == "2020"

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_pattern_fallback_method1(
        self,
        mock_remaining,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test fallback to pattern extraction using table of contents markers."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = ([], [])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_intro.return_value = ("", 0, None)
        mock_remaining.return_value = []

        lines_with_markers = [
            "மலர் 1",
            "பொருளடக்கம்",
            "Author 1",
            "Author 2",
            "ஆகியோரின் எழுத்தோவியங்கள்",
            "Content",
        ]

        result = parse_tamil_document_csv_first(
            lines_with_markers, {}, sample_csv_df, "test.txt"
        )

        assert "articles" in result
        assert result["doc_id"] == "மலர்_1"

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_authors_alternative")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_pattern_fallback_method3_and_4(
        self,
        mock_remaining,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_alt,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test fallback using shared authors and alternative author extraction."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = (["Shared Author"], ["shared"])
        mock_alt.return_value = (["Alt Author"], ["alt"])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_intro.return_value = ("", 0, None)
        mock_remaining.return_value = []

        result = parse_tamil_document_csv_first(
            ["மலர் 1", "Content"], {}, sample_csv_df, "test.txt"
        )

        assert "articles" in result

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    def test_doc_info_na_na(self, mock_doc, mock_csv, sample_lines, sample_csv_df):
        """Test handling when document info extraction returns NA values."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("NA", "NA")

        result = parse_tamil_document_csv_first(
            sample_lines, {}, sample_csv_df, "test.txt"
        )

        assert result["doc_id"] == "NA"
        assert result["doc_issue"] == "NA"
        assert len(result["articles"]) == 0

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.count_content_lines")
    @patch("data_extraction.article_seperation.get_intro_keywords")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_intro_extraction_with_author(
        self,
        mock_remaining,
        mock_keywords,
        mock_count,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test extraction of introduction section with author attribution."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = ([], [])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_keywords.return_value = ["முன்னுரை"]
        mock_intro.return_value = (
            "Intro content line1\nline2\nline3\nline4",
            5,
            "Author",
        )
        mock_count.return_value = 5
        mock_remaining.return_value = []

        lines_with_intro = ["மலர் 1", "", "முன்னுரை", "Content"]

        result = parse_tamil_document_csv_first(
            lines_with_intro, {}, sample_csv_df, "test.txt"
        )

        assert any(a["title"] == "முன்னுரை" for a in result["articles"])

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.count_content_lines")
    @patch("data_extraction.article_seperation.get_intro_keywords")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_intro_extraction_insufficient_content(
        self,
        mock_remaining,
        mock_keywords,
        mock_count,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test that intro sections with insufficient content are skipped."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = ([], [])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_keywords.return_value = ["முன்னுரை"]
        mock_intro.return_value = ("Short", 3, None)
        mock_count.return_value = 2
        mock_remaining.return_value = []

        lines_with_intro = ["மலர் 1", "", "முன்னுரை", "Short"]

        result = parse_tamil_document_csv_first(
            lines_with_intro, {}, sample_csv_df, "test.txt"
        )

        assert not any(a["title"] == "முன்னுரை" for a in result["articles"])

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.get_intro_keywords")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_remaining_content_extraction(
        self,
        mock_remaining,
        mock_keywords,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test extraction of remaining unprocessed content."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = ([], [])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_keywords.return_value = []
        mock_intro.return_value = ("", 0, None)
        mock_remaining.return_value = [
            {"heading": "Remaining", "author": "Author", "content": "Content"}
        ]

        result = parse_tamil_document_csv_first(
            sample_lines, {}, sample_csv_df, "test.txt"
        )

        assert any(a["title"] == "Remaining" for a in result["articles"])

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.get_intro_keywords")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_remaining_content_exception(
        self,
        mock_remaining,
        mock_keywords,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test exception handling during remaining content extraction."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = ([], [])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_keywords.return_value = []
        mock_intro.return_value = ("", 0, None)
        mock_remaining.side_effect = Exception("Remaining error")

        result = parse_tamil_document_csv_first(
            sample_lines, {}, sample_csv_df, "test.txt"
        )

        assert "articles" in result

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    def test_critical_exception(self, mock_csv, sample_lines, sample_csv_df):
        """Test handling of critical exceptions during parsing."""
        mock_csv.side_effect = Exception("Critical error")

        with pytest.raises(Exception):
            parse_tamil_document_csv_first(sample_lines, {}, sample_csv_df, "test.txt")

    @patch("data_extraction.article_seperation.extract_articles_from_csv")
    @patch("data_extraction.article_seperation.extract_doc_info")
    @patch("data_extraction.article_seperation.get_shared_authors")
    @patch("data_extraction.article_seperation.extract_pattern_a_forward")
    @patch("data_extraction.article_seperation.extract_pattern_b_forward")
    @patch("data_extraction.article_seperation.extract_pattern_c_reverse")
    @patch("data_extraction.article_seperation.extract_intro_content_phase1")
    @patch("data_extraction.article_seperation.get_intro_keywords")
    @patch("data_extraction.article_seperation.extract_remaining_content")
    def test_intro_extraction_exception(
        self,
        mock_remaining,
        mock_keywords,
        mock_intro,
        mock_c,
        mock_b,
        mock_a,
        mock_shared,
        mock_doc,
        mock_csv,
        sample_lines,
        sample_csv_df,
    ):
        """Test exception handling during intro extraction."""
        mock_csv.return_value = {"articles": []}
        mock_doc.return_value = ("மலர்_1", "இதழ்_1")
        mock_shared.return_value = ([], [])
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_keywords.return_value = ["முன்னுரை"]
        mock_intro.side_effect = Exception("Intro error")
        mock_remaining.return_value = []

        lines_with_intro = ["மலர் 1", "முன்னுரை", "Content"]

        result = parse_tamil_document_csv_first(
            lines_with_intro, {}, sample_csv_df, "test.txt"
        )

        assert "articles" in result


class TestProcessS3Files:
    """Test suite for S3 file processing orchestration."""

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    def test_no_txt_files(self, mock_build, mock_list, mock_load):
        """Test behavior when no TXT files are found in S3."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = []
        mock_build.return_value = {}

        process_s3_files(force_reprocess=False)

        mock_list.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    def test_csv_not_found(self, mock_load):
        """Test handling when CSV file is not found."""
        mock_load.side_effect = FileNotFoundError("CSV not found")

        process_s3_files(force_reprocess=False)

        mock_load.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    def test_csv_generic_exception(self, mock_load):
        """Test handling of generic CSV loading exceptions."""
        mock_load.side_effect = Exception("Generic error")

        process_s3_files(force_reprocess=False)

        mock_load.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.is_file_already_processed")
    @patch("data_extraction.article_seperation.validate_processed_file")
    def test_skip_already_processed_valid(
        self, mock_validate, mock_is_processed, mock_build, mock_list, mock_load
    ):
        """Test that already processed and valid files are skipped."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["file1.txt"]
        mock_build.return_value = {}
        mock_is_processed.return_value = True
        mock_validate.return_value = True

        process_s3_files(force_reprocess=False)

        mock_validate.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.is_file_already_processed")
    @patch("data_extraction.article_seperation.validate_processed_file")
    @patch("data_extraction.article_seperation.read_text_from_s3")
    @patch("data_extraction.article_seperation.parse_tamil_document_csv_first")
    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.save_authors_to_s3")
    def test_reprocess_invalid_file(
        self,
        mock_save,
        mock_upload,
        mock_parse,
        mock_read,
        mock_validate,
        mock_is_processed,
        mock_build,
        mock_list,
        mock_load,
    ):
        """Test that invalid processed files are reprocessed."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["file1.txt"]
        mock_build.return_value = {}
        mock_is_processed.return_value = True
        mock_validate.return_value = False
        mock_read.return_value = "line1\nline2"
        mock_parse.return_value = {
            "articles": [],
            "authors_list": [],
            "doc_id": "test",
            "doc_issue": "test",
        }

        process_s3_files(force_reprocess=False)

        mock_parse.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.read_text_from_s3")
    @patch("data_extraction.article_seperation.parse_tamil_document_csv_first")
    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.save_authors_to_s3")
    def test_force_reprocess(
        self,
        mock_save,
        mock_upload,
        mock_parse,
        mock_read,
        mock_build,
        mock_list,
        mock_load,
    ):
        """Test force reprocess mode ignores existing files."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["file1.txt"]
        mock_build.return_value = {}
        mock_read.return_value = "line1\nline2"
        mock_parse.return_value = {
            "articles": [],
            "authors_list": [],
            "doc_id": "test",
            "doc_issue": "test",
        }

        process_s3_files(force_reprocess=True)

        mock_parse.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.is_file_already_processed")
    @patch("data_extraction.article_seperation.read_text_from_s3")
    @patch("data_extraction.article_seperation.parse_tamil_document_csv_first")
    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.save_authors_to_s3")
    def test_successful_processing_csv_method(
        self,
        mock_save,
        mock_upload,
        mock_parse,
        mock_read,
        mock_is_processed,
        mock_build,
        mock_list,
        mock_load,
    ):
        """Test successful file processing using CSV extraction method."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["extracted_text/2023/file1.txt"]
        mock_build.return_value = {}
        mock_is_processed.return_value = False
        mock_read.return_value = "line1\nline2"
        mock_parse.return_value = {
            "articles": [{"year": "2023", "title": "Test"}],
            "authors_list": [],
            "doc_id": "test",
            "doc_issue": "test",
        }

        process_s3_files(force_reprocess=False)

        mock_upload.assert_called_once()
        mock_save.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.is_file_already_processed")
    @patch("data_extraction.article_seperation.read_text_from_s3")
    @patch("data_extraction.article_seperation.parse_tamil_document_csv_first")
    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.save_authors_to_s3")
    def test_successful_processing_pattern_method(
        self,
        mock_save,
        mock_upload,
        mock_parse,
        mock_read,
        mock_is_processed,
        mock_build,
        mock_list,
        mock_load,
    ):
        """Test successful file processing using pattern extraction fallback."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["file1.txt"]
        mock_build.return_value = {}
        mock_is_processed.return_value = False
        mock_read.return_value = "line1\nline2"
        mock_parse.return_value = {
            "articles": [{"year": "Unknown", "title": "Test"}],
            "authors_list": [],
            "doc_id": "test",
            "doc_issue": "test",
        }

        process_s3_files(force_reprocess=False)

        mock_upload.assert_called_once()

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.is_file_already_processed")
    @patch("data_extraction.article_seperation.read_text_from_s3")
    @patch("data_extraction.article_seperation.parse_tamil_document_csv_first")
    def test_processing_exception(
        self, mock_parse, mock_read, mock_is_processed, mock_build, mock_list, mock_load
    ):
        """Test that exceptions during processing are handled gracefully."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["file1.txt", "file2.txt"]
        mock_build.return_value = {}
        mock_is_processed.return_value = False
        mock_read.return_value = "line1\nline2"
        mock_parse.side_effect = Exception("Processing error")

        process_s3_files(force_reprocess=False)

        assert mock_parse.call_count == 2

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    def test_critical_exception(self, mock_build, mock_list, mock_load):
        """Test handling of critical exceptions during S3 operations."""
        mock_load.return_value = pd.DataFrame()
        mock_list.side_effect = Exception("Critical S3 error")

        with pytest.raises(Exception):
            process_s3_files(force_reprocess=False)

    @patch("data_extraction.article_seperation.load_csv_from_local")
    @patch("data_extraction.article_seperation.list_files")
    @patch("data_extraction.article_seperation.build_shared_authors_dict_s3")
    @patch("data_extraction.article_seperation.is_file_already_processed")
    @patch("data_extraction.article_seperation.validate_processed_file")
    @patch("data_extraction.article_seperation.read_text_from_s3")
    @patch("data_extraction.article_seperation.parse_tamil_document_csv_first")
    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.save_authors_to_s3")
    def test_multiple_files_mixed_results(
        self,
        mock_save,
        mock_upload,
        mock_parse,
        mock_read,
        mock_validate,
        mock_is_processed,
        mock_build,
        mock_list,
        mock_load,
    ):
        """Test processing multiple files with mixed success and failure outcomes."""
        mock_load.return_value = pd.DataFrame()
        mock_list.return_value = ["file1.txt", "file2.txt", "file3.txt"]
        mock_build.return_value = {}

        mock_is_processed.return_value = False
        mock_read.return_value = "line1\nline2"

        def parse_side_effect(lines, shared, csv, key):
            if "file3" in key:
                raise Exception("Parse error")
            return {
                "articles": [],
                "authors_list": [],
                "doc_id": "test",
                "doc_issue": "test",
            }

        mock_parse.side_effect = parse_side_effect

        process_s3_files(force_reprocess=False)

        assert mock_parse.call_count == 3
        assert mock_upload.call_count == 2


class TestEdgeCasesAndIntegration:
    """Test suite for edge cases and integration scenarios."""

    def test_extract_year_with_none(self):
        """Test year extraction with None input."""
        try:
            result = extract_year_from_s3_key(None)
            assert result == "Unknown"
        except (TypeError, AttributeError):
            pass

    def test_extract_year_with_special_chars(self):
        """Test year extraction from path with special characters."""
        result = extract_year_from_s3_key("path/!@#$%/2021/file.txt")
        assert result == "2021"

    @patch("data_extraction.article_seperation.file_exists")
    def test_is_processed_with_none_inputs(self, mock_exists):
        """Test file processing check with None inputs."""
        mock_exists.return_value = False

        try:
            result = is_file_already_processed(None, None)
            assert result is False
        except (TypeError, AttributeError):
            pass

    @patch("s3_utils.read_json_from_s3")
    def test_validate_with_deeply_nested_structure(self, mock_read):
        """Test validation with deeply nested JSON structure."""
        mock_read.return_value = {
            "articles": [{"title": "Test", "nested": {"deep": {"structure": "value"}}}]
        }

        assert validate_processed_file("bucket", "key") is True

    def test_load_csv_with_large_file(self, tmp_path):
        """Test loading CSV file with many rows."""
        csv_file = tmp_path / "large.csv"
        content = "col1,col2\n" + "val1,val2\n" * 1000
        csv_file.write_text(content, encoding="utf-8")

        df = load_csv_from_local(csv_file)

        assert len(df) == 1000

    @patch("data_extraction.article_seperation.upload_json")
    @patch("data_extraction.article_seperation.file_exists")
    @patch("s3_utils.read_json_from_s3")
    def test_save_authors_with_unicode(self, mock_read, mock_exists, mock_upload):
        """Test saving authors with Tamil Unicode characters."""
        mock_exists.return_value = False

        authors = ["கருணாநிதி", "பெரியார்", "அண்ணா"]

        save_authors_to_s3("bucket", "output/", "மலர்_1", "இதழ்_1", authors)

        saved_data = mock_upload.call_args[0][2]
        assert saved_data[0]["authors"] == authors


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=data_extraction.article_seperation",
            "--cov-report=term-missing",
            "--cov-report=html",
        ]
    )
