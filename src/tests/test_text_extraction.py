import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import io
import sys
from pathlib import Path

# Add project root to path (same as your extraction.py does)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from text_extraction.py
from src.data_extraction.text_extraction import normalize_key, process_all_docx_files


class TestNormalizeKey:
    """Test cases for normalize_key function"""
    
    def test_lowercase_conversion(self):
        """Test that keys are converted to lowercase"""
        assert normalize_key("FILE.DOCX") == "file.docx"
        assert normalize_key("MyFile.TXT") == "myfile.txt"
    
    def test_space_replacement(self):
        """Test that spaces are replaced with underscores"""
        assert normalize_key("my file.docx") == "my_file.docx"
        assert normalize_key("file  with  spaces.txt") == "file_with_spaces.txt"
    
    def test_strip_whitespace(self):
        """Test that leading/trailing whitespace is removed"""
        assert normalize_key("  file.docx  ") == "file.docx"
        assert normalize_key("\tfile.txt\n") == "file.txt"
    
    def test_underscore_dot_replacement(self):
        """Test that '_.' is replaced with '.'"""
        assert normalize_key("file_.txt") == "file.txt"
        assert normalize_key("my_file_.docx") == "my_file.docx"
    
    def test_combined_normalization(self):
        """Test multiple normalizations together"""
        assert normalize_key("  My File Name_.DOCX  ") == "my_file_name.docx"
        assert normalize_key("REPORT 2024_.TXT") == "report_2024.txt"
    
    def test_already_normalized(self):
        """Test that already normalized keys remain unchanged"""
        assert normalize_key("file.docx") == "file.docx"
        assert normalize_key("my_file.txt") == "my_file.txt"


class TestProcessAllDocxFiles:
    """Test cases for process_all_docx_files function"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Setup common mocks for all tests"""
        # Patch using the full module path from project root
        with patch('src.data_extraction.text_extraction.list_files') as mock_list, \
             patch('src.data_extraction.text_extraction.read_bytes') as mock_read, \
             patch('src.data_extraction.text_extraction.upload_text') as mock_upload, \
             patch('src.data_extraction.text_extraction.docx2txt') as mock_docx2txt, \
             patch('src.data_extraction.text_extraction.logger') as mock_logger, \
             patch('src.data_extraction.text_extraction.INPUT_PREFIX', 'input/'), \
             patch('src.data_extraction.text_extraction.EXTRACTED_OUTPUT', 'extracted/'):
            
            yield {
                'list_files': mock_list,
                'read_bytes': mock_read,
                'upload_text': mock_upload,
                'docx2txt': mock_docx2txt,
                'logger': mock_logger
            }
    
    def test_successful_processing(self, mock_dependencies):
        """Test successful processing of DOCX files"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx', 'input/file2.docx'],  # DOCX files
            []  # No existing TXT files
        ]
        mock_dependencies['read_bytes'].return_value = b'mock_bytes'
        mock_dependencies['docx2txt'].process.return_value = 'Extracted text'
        
        # Execute
        process_all_docx_files()
        
        # Assert
        assert mock_dependencies['read_bytes'].call_count == 2
        assert mock_dependencies['upload_text'].call_count == 2
        assert mock_dependencies['docx2txt'].process.call_count == 2
    
    def test_skip_already_processed_files(self, mock_dependencies):
        """Test that already processed files are skipped"""
        # Setup - Existing file should match the normalized output path
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx'],  # DOCX files
            ['extracted/file1.txt']  # Existing TXT files - exact normalized match
        ]
        
        # Execute
        process_all_docx_files()
        
        # Assert - should not process the file
        mock_dependencies['read_bytes'].assert_not_called()
        mock_dependencies['upload_text'].assert_not_called()
    
    def test_skip_case_insensitive_match(self, mock_dependencies):
        """Test that case-insensitive matching works for skipping"""
        # Setup - Normalized paths should match
        mock_dependencies['list_files'].side_effect = [
            ['input/File1.DOCX'],  # DOCX files (mixed case)
            ['extracted/file1.txt']  # Existing TXT files (all lowercase after normalization)
        ]
        
        # Execute
        process_all_docx_files()
        
        # Assert - should skip due to case-insensitive match
        mock_dependencies['read_bytes'].assert_not_called()
        mock_dependencies['upload_text'].assert_not_called()
    
    def test_no_docx_files_found(self, mock_dependencies):
        """Test handling when no DOCX files are found"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            [],  # No DOCX files
            []   # No TXT files
        ]
        
        # Execute
        process_all_docx_files()
        
        # Assert
        mock_dependencies['read_bytes'].assert_not_called()
        mock_dependencies['upload_text'].assert_not_called()
    
    def test_file_read_error(self, mock_dependencies):
        """Test handling of file read errors"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx'],
            []
        ]
        mock_dependencies['read_bytes'].side_effect = Exception("S3 read error")
        
        # Execute
        process_all_docx_files()
        
        # Assert - should log error and continue
        mock_dependencies['logger'].error.assert_called()
        mock_dependencies['upload_text'].assert_not_called()
    
    def test_docx_extraction_error(self, mock_dependencies):
        """Test handling of DOCX extraction errors"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx'],
            []
        ]
        mock_dependencies['read_bytes'].return_value = b'mock_bytes'
        mock_dependencies['docx2txt'].process.side_effect = Exception("Extraction error")
        
        # Execute
        process_all_docx_files()
        
        # Assert - should log error and not upload
        mock_dependencies['logger'].error.assert_called()
        mock_dependencies['upload_text'].assert_not_called()
    
    def test_upload_error(self, mock_dependencies):
        """Test handling of upload errors"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx'],
            []
        ]
        mock_dependencies['read_bytes'].return_value = b'mock_bytes'
        mock_dependencies['docx2txt'].process.return_value = 'Text content'
        mock_dependencies['upload_text'].side_effect = Exception("Upload error")
        
        # Execute
        process_all_docx_files()
        
        # Assert - should log error
        mock_dependencies['logger'].error.assert_called()
    
    def test_list_files_error(self, mock_dependencies):
        """Test handling of list_files error"""
        # Setup
        mock_dependencies['list_files'].side_effect = Exception("S3 list error")
        
        # Execute
        process_all_docx_files()
        
        # Assert - should log error and return early
        mock_dependencies['logger'].error.assert_called()
        mock_dependencies['read_bytes'].assert_not_called()
    
    def test_preserve_folder_structure(self, mock_dependencies):
        """Test that folder structure is preserved in output"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/subfolder/file1.docx'],
            []
        ]
        mock_dependencies['read_bytes'].return_value = b'mock_bytes'
        mock_dependencies['docx2txt'].process.return_value = 'Text'
        
        # Execute
        process_all_docx_files()
        
        # Assert - check that upload was called with correct path structure
        upload_calls = mock_dependencies['upload_text'].call_args_list
        assert len(upload_calls) == 1
        output_key = upload_calls[0][0][1]
        assert 'subfolder' in output_key
        assert output_key.endswith('.txt')
    
    def test_multiple_files_processing(self, mock_dependencies):
        """Test processing multiple files with mixed success/skip/failure"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx', 'input/file2.docx', 'input/file3.docx'],
            ['extracted/file2.txt']  # file2 already processed
        ]
        mock_dependencies['read_bytes'].side_effect = [
            b'bytes1',
            Exception("Error"),  # file3 fails
        ]
        mock_dependencies['docx2txt'].process.return_value = 'Text'
        
        # Execute
        process_all_docx_files()
        
        # Assert
        assert mock_dependencies['upload_text'].call_count == 1  # Only file1 succeeds
        assert mock_dependencies['logger'].error.call_count >= 1  # file3 error logged
    
    def test_empty_text_extraction(self, mock_dependencies):
        """Test handling of empty text extraction"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/file1.docx'],
            []
        ]
        mock_dependencies['read_bytes'].return_value = b'mock_bytes'
        mock_dependencies['docx2txt'].process.return_value = ''  # Empty text
        
        # Execute
        process_all_docx_files()
        
        # Assert - should still upload empty text
        mock_dependencies['upload_text'].assert_called_once()
        assert mock_dependencies['upload_text'].call_args[0][2] == ''
    
    def test_filename_extension_replacement(self, mock_dependencies):
        """Test that .docx extension is replaced with .txt"""
        # Setup
        mock_dependencies['list_files'].side_effect = [
            ['input/Document.DOCX', 'input/file.Docx'],
            []
        ]
        mock_dependencies['read_bytes'].return_value = b'mock_bytes'
        mock_dependencies['docx2txt'].process.return_value = 'Text'
        
        # Execute
        process_all_docx_files()
        
        # Assert - all uploaded files should have .txt extension
        for call in mock_dependencies['upload_text'].call_args_list:
            output_key = call[0][1]
            assert output_key.endswith('.txt')
            assert '.docx' not in output_key.lower()


class TestIntegration:
    """Integration tests for the extraction workflow"""
    
    @patch('src.data_extraction.text_extraction.BUCKET_NAME', 'test-bucket')
    @patch('src.data_extraction.text_extraction.INPUT_PREFIX', 'input/')
    @patch('src.data_extraction.text_extraction.EXTRACTED_OUTPUT', 'output/')
    def test_end_to_end_workflow(self):
        """Test complete workflow from listing to uploading"""
        with patch('src.data_extraction.text_extraction.list_files') as mock_list, \
             patch('src.data_extraction.text_extraction.read_bytes') as mock_read, \
             patch('src.data_extraction.text_extraction.upload_text') as mock_upload, \
             patch('src.data_extraction.text_extraction.docx2txt') as mock_docx:
            
            # Setup realistic scenario
            mock_list.side_effect = [
                ['input/report.docx', 'input/notes.docx'],
                ['output/notes.txt']  # notes already processed
            ]
            mock_read.return_value = b'document_bytes'
            mock_docx.process.return_value = 'Extracted document text'
            
            # Execute
            process_all_docx_files()
            
            # Assert
            assert mock_list.call_count == 2
            assert mock_read.call_count == 1  # Only report.docx
            assert mock_upload.call_count == 1
            
            # Verify correct parameters
            mock_upload.assert_called_once_with(
                'test-bucket',
                'output/report.txt',
                'Extracted document text'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])