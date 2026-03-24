"""Test text extraction functions."""

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_extraction.text_extraction import (  # noqa: E402
    extract_text_from_file,
    normalize_key,
    process_all_docx_files,
)


class TestNormalizeKey:
    """
    Test suite for the normalize_key function.

    The normalize_key function standardizes filenames by:
    - Converting to lowercase
    - Replacing spaces with underscores
    - Removing redundant underscores before dots
    - Stripping leading/trailing whitespace

    This ensures consistent naming conventions across the system.
    """

    def test_lowercase_conversion(self):
        """
        Test that uppercase letters are converted to lowercase.

        Validates:
            - Mixed case filename → all lowercase
            - Uppercase file extension → lowercase

        Example:
            "My File.DOCX" → "my_file.docx"
        """
        result = normalize_key("My File.DOCX")
        assert result == "my_file.docx"

    def test_space_replacement(self):
        """
        Test that spaces are replaced with underscores.

        Validates:
            - Single spaces between words → underscores
            - Preserves file extension

        Example:
            "File With Spaces.txt" → "file_with_spaces.txt"
        """
        result = normalize_key("File With Spaces.txt")
        assert result == "file_with_spaces.txt"

    def test_multiple_spaces(self):
        """
        Test handling of multiple consecutive spaces.

        Validates:
            - Multiple spaces → single underscore
            - Whitespace normalization

        Example:
            "File   Multiple   Spaces.txt" → "file_multiple_spaces.txt"
        """
        result = normalize_key("File   Multiple   Spaces.txt")
        assert result == "file_multiple_spaces.txt"

    def test_underscore_dot_cleanup(self):
        """
        Test removal of underscore immediately before file extension.

        Validates:
            - Pattern "_.ext" → ".ext"
            - Prevents malformed filenames

        Example:
            "file_.txt" → "file.txt"
        """
        result = normalize_key("file_.txt")
        assert result == "file.txt"

    def test_strip_whitespace(self):
        """
        Test stripping of leading and trailing whitespace.

        Validates:
            - Leading spaces removed
            - Trailing spaces removed
            - Internal spacing preserved (then normalized)

        Example:
            "  file.txt  " → "file.txt"
        """
        result = normalize_key("  file.txt  ")
        assert result == "file.txt"


class TestExtractTextFromFile:
    """
    Test suite for the extract_text_from_file function.

    This function extracts text from Word documents, supporting both:
    - Modern .docx format (using docx2txt)
    - Legacy .doc format (fallback mechanism)

    It implements a two-tier extraction strategy with graceful degradation.
    """

    @patch("data_extraction.text_extraction.docx2txt")
    def test_successful_docx_extraction(self, mock_docx2txt):
        """
        Test successful extraction from modern DOCX format.

        Validates:
            - docx2txt.process called with file stream
            - Extracted text returned correctly
            - No fallback mechanism triggered

        Args:
            mock_docx2txt: Mocked docx2txt module

        Expected Flow:
            1. docx2txt.process() called
            2. Returns extracted text
            3. No exceptions raised
        """
        mock_docx2txt.process.return_value = "Extracted text"
        file_bytes = BytesIO(b"docx content")

        result = extract_text_from_file(file_bytes, "test.docx")

        assert result == "Extracted text"
        mock_docx2txt.process.assert_called_once()

    @patch("data_extraction.text_extraction.docx2txt")
    @patch("data_extraction.text_extraction.extract_text_from_doc")
    def test_fallback_to_doc_format(self, mock_extract_doc, mock_docx2txt):
        """
        Test fallback to legacy DOC format when DOCX extraction fails.

        Validates:
            - DOCX extraction attempted first
            - Exception triggers fallback
            - DOC extraction succeeds
            - Alternative text returned

        Args:
            mock_extract_doc: Mocked DOC extractor
            mock_docx2txt: Mocked DOCX extractor (configured to fail)

        Expected Flow:
            1. docx2txt.process() raises exception
            2. extract_text_from_doc() called
            3. Returns text from DOC parser
        """
        mock_docx2txt.process.side_effect = Exception("Not a DOCX")
        mock_extract_doc.return_value = "Text from DOC"
        file_bytes = BytesIO(b"doc content")

        result = extract_text_from_file(file_bytes, "test.doc")

        assert result == "Text from DOC"
        mock_extract_doc.assert_called_once()

    @patch("data_extraction.text_extraction.docx2txt")
    @patch("data_extraction.text_extraction.extract_text_from_doc")
    def test_both_formats_fail(self, mock_extract_doc, mock_docx2txt):
        """
        Test behavior when both DOCX and DOC extraction fail.

        Validates:
            - Both extraction methods attempted
            - Appropriate exception raised
            - Error message contains context

        Args:
            mock_extract_doc: Mocked DOC extractor (configured to fail)
            mock_docx2txt: Mocked DOCX extractor (configured to fail)

        Expected Flow:
            1. docx2txt.process() raises exception
            2. extract_text_from_doc() raises exception
            3. Function raises exception with descriptive message
        """
        mock_docx2txt.process.side_effect = Exception("DOCX failed")
        mock_extract_doc.side_effect = Exception("DOC failed")
        file_bytes = BytesIO(b"content")

        with pytest.raises(Exception, match="Could not extract text"):
            extract_text_from_file(file_bytes, "corrupt.docx")


class TestProcessAllDocxFiles:
    """
    Comprehensive test suite for the process_all_docx_files function.

    This is the main orchestration function that:
    - Lists files from S3 bucket
    - Filters already-processed files
    - Extracts text from each document
    - Uploads results to S3
    - Handles errors gracefully
    - Logs comprehensive statistics

    Test Categories:
        1. S3 Operations (lines 56-58, 62-64)
        2. Main Processing Loop (lines 91-94)
        3. Exception Handling (lines 182-189)
        4. Edge Cases and Integration Tests

    Fixtures:
        mock_all: Provides comprehensive mocking of all external dependencies
    """

    @pytest.fixture
    def mock_all(self):
        """
        Fixture providing comprehensive mocking of external dependencies.

        Mocks all external services and libraries:
            - AWS S3 operations (list_files, read_bytes, upload_text)
            - Configuration constants (BUCKET_NAME, INPUT_PREFIX, etc.)
            - Document processing library (docx2txt)
            - Logging infrastructure

        Yields:
            dict: Dictionary of mock objects for use in tests
                - list_files: Mock for S3 file listing
                - read_bytes: Mock for S3 file reading
                - upload_text: Mock for S3 file upload
                - docx2txt: Mock for document processing
                - logger: Mock for logging operations

        Usage:
            def test_example(self, mock_all):
                mock_all['list_files'].return_value = ['file.docx']
                mock_all['docx2txt'].process.return_value = "text"
        """
        # Mock the config imports at module level
        with patch(
            "data_extraction.text_extraction.BUCKET_NAME", "ponni-rag-bucket"
        ), patch("data_extraction.text_extraction.INPUT_PREFIX", "input/"), patch(
            "data_extraction.text_extraction.EXTRACTED_OUTPUT", "extracted/"
        ), patch(
            "data_extraction.text_extraction.list_files"
        ) as mock_list, patch(
            "data_extraction.text_extraction.read_bytes"
        ) as mock_read, patch(
            "data_extraction.text_extraction.upload_text"
        ) as mock_upload, patch(
            "data_extraction.text_extraction.docx2txt"
        ) as mock_docx2txt, patch(
            "data_extraction.text_extraction.logger"
        ) as mock_logger:
            yield {
                "list_files": mock_list,
                "read_bytes": mock_read,
                "upload_text": mock_upload,
                "docx2txt": mock_docx2txt,
                "logger": mock_logger,
            }

    def test_lines_56_58_list_docx_and_txt_files(self, mock_all):
        """
        Test lines 56-58: Verify correct S3 file listing calls.

        Target Code:
            Line 56: docx_files = list_files(
                BUCKET_NAME, INPUT_PREFIX, suffix='.docx')
            Line 57-58: existing_txt_files = list_files(
                BUCKET_NAME, EXTRACTED_OUTPUT, suffix='.txt')

        Validates:
            - list_files called twice (once for DOCX, once for TXT)
            - Correct bucket name used
            - Correct prefixes used (input/ and extracted/)
            - Correct file suffixes specified

        Args:
            mock_all: Fixture providing mocked dependencies
        """
        # Setup
        mock_all["list_files"].side_effect = [
            ["input/file1.docx", "input/file2.docx"],  # Line 56: docx_files
            ["extracted/file1.txt"],  # Line 57-58: existing_txt_files
        ]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Verify list_files called twice with correct parameters
        assert mock_all["list_files"].call_count == 2

        # First call: list DOCX files (line 56)
        first_call = mock_all["list_files"].call_args_list[0]
        assert first_call[0][0] == "ponni-rag-bucket"
        assert first_call[0][1] == "input/"
        assert first_call[1]["suffix"] == ".docx"

        # Second call: list TXT files (line 57-58)
        second_call = mock_all["list_files"].call_args_list[1]
        assert second_call[0][0] == "ponni-rag-bucket"
        assert second_call[0][1] == "extracted/"
        assert second_call[1]["suffix"] == ".txt"

    def test_lines_62_64_log_file_counts(self, mock_all):
        """
        Test lines 62-64: Verify file count logging.

        Target Code:
            Line 62: logger.info(f"Found {len(docx_files)} ...")
            Line 63: logger.info(f"Found ... extracted .txt")
            Line 64: logger.info("=" * 80)

        Validates:
            - DOCX file count logged correctly
            - TXT file count logged correctly
            - Separator line logged (80 equals signs)
            - Log messages contain expected content

        Args:
            mock_all: Fixture providing mocked dependencies

        Test Strategy:
            1. Configure list_files to return known quantities
            2. Execute function
            3. Inspect logger.info calls for expected messages
        """
        # Setup
        mock_all["list_files"].side_effect = [
            ["input/a.docx", "input/b.docx", "input/c.docx"],  # 3 DOCX files
            ["extracted/a.txt"],  # 1 TXT file
        ]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Verify logger.info calls for lines 62-64
        info_calls = [
            str(call[0][0]) for call in mock_all["logger"].info.call_args_list
        ]

        # Line 62: logger.info(f"Found {len(docx_files)} .docx files")
        assert any(
            "3" in msg and "docx" in msg.lower() for msg in info_calls
        ), f"Expected '3 .docx files' in: {info_calls}"

        # Line 63: logger.info(f"Found {len(existing_txt_files)} extracted .txt files")
        assert any(
            "1" in msg and ("txt" in msg.lower() or "extracted" in msg.lower())
            for msg in info_calls
        ), f"Expected '1 .txt files' in: {info_calls}"

        # Line 64: logger.info("=" * 80)
        assert any("=" * 80 in msg for msg in info_calls), "Expected separator line"

    def test_lines_62_64_with_no_files(self, mock_all):
        """
        Test lines 62-64: Verify logging when no files are found.

        Edge Case:
            Empty S3 bucket or no matching files

        Validates:
            - Function handles empty file lists gracefully
            - Logs "0 files" messages correctly
            - No processing attempted

        Args:
            mock_all: Fixture providing mocked dependencies
        """
        # Setup
        mock_all["list_files"].side_effect = [[], []]  # No DOCX files  # No TXT files

        # Execute
        process_all_docx_files()

        # Assert
        info_calls = [
            str(call[0][0]) for call in mock_all["logger"].info.call_args_list
        ]

        # Should log 0 files
        assert any("0" in msg and "docx" in msg.lower() for msg in info_calls)

    def test_lines_91_94_file_processing_success(self, mock_all):
        """
        Test lines 91-94: Verify complete file processing workflow.

        Target Code:
            Line 91: file_bytes = read_bytes(BUCKET_NAME, key)
            Line 92-93: text = extract_text_from_file(file_bytes, docx_file)
            Line 94: upload_text(BUCKET_NAME, output_key, text)

        Validates:
            - File read from S3 with correct parameters
            - Text extraction called with file bytes
            - Extracted text uploaded to correct S3 location
            - Output path matches expected pattern

        Args:
            mock_all: Fixture providing mocked dependencies

        Test Strategy:
            1. Configure single file for processing
            2. Track all function calls
            3. Verify call sequence and parameters
        """
        # Setup
        mock_all["list_files"].side_effect = [["input/test.docx"], []]
        file_bytes = BytesIO(b"docx content")
        mock_all["read_bytes"].return_value = file_bytes
        mock_all["docx2txt"].process.return_value = "Extracted text content"

        # Execute
        process_all_docx_files()

        # Assert - Verify lines 91-94 executed
        # Line 91: file_bytes = read_bytes(BUCKET_NAME, key)
        mock_all["read_bytes"].assert_called_once_with(
            "ponni-rag-bucket", "input/test.docx"
        )

        # Line 92-93: text = extract_text_from_file(file_bytes, docx_file)
        mock_all["docx2txt"].process.assert_called_once()

        # Line 94: upload_text(BUCKET_NAME, output_key, text)
        mock_all["upload_text"].assert_called_once()
        upload_args = mock_all["upload_text"].call_args
        assert upload_args[0][0] == "ponni-rag-bucket"
        assert "extracted/" in upload_args[0][1]
        assert upload_args[0][2] == "Extracted text content"

    def test_lines_91_94_multiple_files(self, mock_all):
        """
        Test lines 91-94: Verify batch processing of multiple files.

        Validates:
            - Loop executes for each file
            - Each file processed independently
            - All operations called correct number of times

        Args:
            mock_all: Fixture providing mocked dependencies

        Test Coverage:
            Ensures lines 91-94 execute multiple times in loop
        """
        # Setup
        mock_all["list_files"].side_effect = [
            ["input/file1.docx", "input/file2.docx", "input/file3.docx"],
            [],
        ]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Lines 91-94 executed 3 times
        assert mock_all["read_bytes"].call_count == 3
        assert mock_all["docx2txt"].process.call_count == 3
        assert mock_all["upload_text"].call_count == 3

    def test_lines_182_189_exception_during_processing(self, mock_all):
        """
        Test lines 182-189: Verify exception handling during file processing.

        Target Code:
            Exception handling block that catches and logs errors

        Validates:
            - Exceptions caught and logged
            - Processing continues for other files
            - Error message includes file name
            - Failed file tracked in statistics

        Args:
            mock_all: Fixture providing mocked dependencies

        Error Scenario:
            S3 read operation fails for specific file
        """
        # Setup
        mock_all["list_files"].side_effect = [["input/corrupt.docx"], []]
        # Make read_bytes raise exception to trigger except block
        mock_all["read_bytes"].side_effect = Exception("S3 read error")

        # Execute
        process_all_docx_files()

        # Assert - Lines 182-189: Exception logged
        error_calls = mock_all["logger"].error.call_args_list
        assert len(error_calls) > 0, "Expected error to be logged"

        # Verify error message contains file name and error
        error_msg = str(error_calls[0])
        assert "corrupt.docx" in error_msg
        assert "Error processing" in error_msg or "error" in error_msg.lower()

    def test_lines_182_189_extraction_fails(self, mock_all):
        """
        Test lines 182-189: Verify handling of text extraction failures.

        Validates:
            - Text extraction errors caught
            - Error logged with context
            - Failed count incremented
            - Processing continues (doesn't crash)

        Args:
            mock_all: Fixture providing mocked dependencies

        Error Scenario:
            Corrupted document that cannot be parsed
        """
        # Setup
        mock_all["list_files"].side_effect = [["input/bad.docx"], []]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        # Make docx2txt raise exception
        mock_all["docx2txt"].process.side_effect = Exception("Extraction failed")

        # Execute
        process_all_docx_files()

        # Assert - Exception logged, file added to failed count
        assert mock_all["logger"].error.called

        # Check final summary
        info_calls = [
            str(call[0][0]) for call in mock_all["logger"].info.call_args_list
        ]
        summary = next((msg for msg in info_calls if "Failed:" in msg), None)
        assert summary is not None
        assert "Failed: 1" in summary

    def test_lines_182_189_upload_fails(self, mock_all):
        """
        Test lines 182-189: Verify handling of upload failures.

        Validates:
            - Upload errors caught
            - Error logged with file details
            - Failed count incremented

        Args:
            mock_all: Fixture providing mocked dependencies

        Error Scenario:
            S3 upload operation fails (network, permissions, etc.)
        """
        # Setup
        mock_all["list_files"].side_effect = [["input/file.docx"], []]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Valid text"
        # Make upload fail
        mock_all["upload_text"].side_effect = Exception("Upload error")

        # Execute
        process_all_docx_files()

        # Assert - Error logged
        assert mock_all["logger"].error.called
        error_msg = str(mock_all["logger"].error.call_args_list[0])
        assert "file.docx" in error_msg

    def test_lines_182_189_corrupted_files_list(self, mock_all):
        """
        Test lines 182-189: Verify tracking of all corrupted files.

        Validates:
            - Multiple failures tracked separately
            - Corrupted files list logged
            - All failed filenames present in warnings
            - Successful files not affected

        Args:
            mock_all: Fixture providing mocked dependencies

        Test Scenario:
            Mix of corrupted and valid files
        """
        # Setup
        mock_all["list_files"].side_effect = [
            ["input/corrupt1.docx", "input/good.docx", "input/corrupt2.docx"],
            [],
        ]
        mock_all["read_bytes"].side_effect = [
            Exception("Fail 1"),
            BytesIO(b"content"),  # good file
            Exception("Fail 2"),
        ]
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Should log corrupted files
        warning_calls = [
            str(call[0][0]) for call in mock_all["logger"].warning.call_args_list
        ]
        assert any("Corrupted/failed files" in msg for msg in warning_calls)
        assert any("corrupt1.docx" in msg for msg in warning_calls)
        assert any("corrupt2.docx" in msg for msg in warning_calls)

    def test_skip_already_processed_files(self, mock_all):
        """
        Test that already-processed files are skipped efficiently.

        Validates:
            - Files with existing .txt outputs are skipped
            - Only unprocessed files are downloaded
            - Debug log messages indicate skipped files
            - Performance optimization works

        Args:
            mock_all: Fixture providing mocked dependencies

        Business Logic:
            Avoid redundant processing and S3 downloads
        """
        # Setup
        mock_all["list_files"].side_effect = [
            ["input/file1.docx", "input/file2.docx"],
            ["extracted/file1.txt"],  # file1 already processed
        ]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Only file2 should be processed
        assert mock_all["read_bytes"].call_count == 1
        assert mock_all["upload_text"].call_count == 1

        # Verify debug log for skipped file
        debug_calls = mock_all["logger"].debug.call_args_list
        assert len(debug_calls) > 0
        assert any(
            "Skipped" in str(call) and "file1.docx" in str(call) for call in debug_calls
        )

    def test_final_summary_all_success(self, mock_all):
        """
        Test final summary statistics when all files process successfully.

        Validates:
            - Processing summary logged
            - Correct counts (Processed, Skipped, Failed)
            - Timing information included
            - Average time per file calculated

        Args:
            mock_all: Fixture providing mocked dependencies

        Expected Output:
            "Processing complete - Processed: 2, Skipped: 0, Failed: 0"
            "Total processing time: X.XX seconds"
            "Average time per file: X.XX seconds"
        """
        # Setup
        mock_all["list_files"].side_effect = [
            ["input/file1.docx", "input/file2.docx"],
            [],
        ]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Check final summary
        info_calls = [
            str(call[0][0]) for call in mock_all["logger"].info.call_args_list
        ]

        # Should have summary line
        summary = next(
            (msg for msg in info_calls if "Processing complete" in msg), None
        )
        assert summary is not None
        assert "Processed: 2" in summary
        assert "Skipped: 0" in summary
        assert "Failed: 0" in summary

        # Should have timing info
        assert any("Total processing time" in msg for msg in info_calls)
        assert any("Average time per file" in msg for msg in info_calls)

    def test_list_files_exception_handling(self, mock_all):
        """
        Test graceful handling when S3 listing operation fails.

        Validates:
            - Exception during list_files caught
            - Appropriate error logged
            - Function returns early (doesn't crash)
            - No file processing attempted

        Args:
            mock_all: Fixture providing mocked dependencies

        Error Scenario:
            S3 connection failure, permissions issue, etc.
        """
        # Setup
        mock_all["list_files"].side_effect = Exception("S3 connection error")

        # Execute
        process_all_docx_files()

        # Assert - Should log error and return early
        assert mock_all["logger"].error.called
        error_msg = str(mock_all["logger"].error.call_args_list[0])
        assert "Failed to list files" in error_msg

        # Should not attempt to process any files
        mock_all["read_bytes"].assert_not_called()

    def test_nested_directory_structure(self, mock_all):
        """
        Test handling of files in nested directory structures.

        Validates:
            - Directory structure preserved in output
            - Correct S3 paths generated
            - Nested paths normalized properly

        Args:
            mock_all: Fixture providing mocked dependencies

        Test Case:
            input/folder1/folder2/file.docx → extracted/folder1/folder2/file.txt
        """
        # Setup
        mock_all["list_files"].side_effect = [["input/folder1/folder2/file.docx"], []]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "Text"

        # Execute
        process_all_docx_files()

        # Assert - Output should preserve directory structure
        upload_call = mock_all["upload_text"].call_args
        output_key = upload_call[0][1]
        assert "extracted/folder1/folder2/file.txt" in output_key

    def test_tamil_text_content(self, mock_all):
        """
        Test processing of Tamil language content (Unicode support).

        Validates:
            - Tamil text extracted correctly
            - Unicode characters preserved
            - Text uploaded without corruption
            - UTF-8 encoding maintained

        Args:
            mock_all: Fixture providing mocked dependencies

        Business Context:
            Ponni RAG system processes Tamil government documents
        """
        # Setup
        mock_all["list_files"].side_effect = [["input/tamil.docx"], []]
        mock_all["read_bytes"].return_value = BytesIO(b"content")
        mock_all["docx2txt"].process.return_value = "தமிழ் உரை - Tamil content"

        # Execute
        process_all_docx_files()

        # Assert - Tamil text should be uploaded
        upload_call = mock_all["upload_text"].call_args
        assert "தமிழ்" in upload_call[0][2]


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=data_extraction.text_extraction",
            "--cov-report=term-missing",
        ]
    )
