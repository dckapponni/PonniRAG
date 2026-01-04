import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Imports will be handled by conftest.py
from article_seperation import (
    get_s3_folder_path,
    save_authors_to_s3_folder,
    parse_tamil_document,
    process_s3_files
)


class TestGetS3FolderPath:
    """Test cases for get_s3_folder_path function"""
    
    def test_extract_folder_from_full_path(self):
        """Test extracting folder from full S3 path"""
        result = get_s3_folder_path(
            "Raw_Proof_Read_Content/folder1/doc.txt",
            "Raw_Proof_Read_Content/"
        )
        assert result == "folder1/"
    
    def test_extract_nested_folder(self):
        """Test extracting nested folder structure"""
        result = get_s3_folder_path(
            "Raw_Proof_Read_Content/year/month/doc.txt",
            "Raw_Proof_Read_Content/"
        )
        assert result == "year/month/"
    
    def test_no_folder_structure(self):
        """Test file directly in base prefix"""
        result = get_s3_folder_path(
            "Raw_Proof_Read_Content/doc.txt",
            "Raw_Proof_Read_Content/"
        )
        assert result == ""
    
    def test_no_base_prefix_match(self):
        """Test when base prefix doesn't match"""
        result = get_s3_folder_path(
            "different_path/folder/doc.txt",
            "Raw_Proof_Read_Content/"
        )
        assert result == "different_path/folder/"
    
    def test_empty_key(self):
        """Test with empty S3 key"""
        result = get_s3_folder_path("", "base/")
        assert result == ""


class TestSaveAuthorsToS3Folder:
    """Test cases for save_authors_to_s3_folder function"""
    
    @pytest.fixture
    def mock_s3_operations(self):
        """Setup mocks for S3 operations"""
        with patch('article_seperation.file_exists') as mock_exists, \
             patch('article_seperation.read_json_from_s3') as mock_read, \
             patch('article_seperation.upload_json') as mock_upload:
            yield {
                'file_exists': mock_exists,
                'read_json': mock_read,
                'upload_json': mock_upload
            }
    
    def test_save_new_authors_file(self, mock_s3_operations):
        """Test creating new authors.json file"""
        mock_s3_operations['file_exists'].return_value = False
        
        authors_list = [
            {"doc_id": "1", "doc_issue": "1", "author_name": "Author 1"},
            {"doc_id": "1", "doc_issue": "1", "author_name": "Author 2"}
        ]
        
        save_authors_to_s3_folder(
            "test-bucket", "output/", "folder1/",
            "1", "1", authors_list
        )
        
        # Verify upload was called
        mock_s3_operations['upload_json'].assert_called_once()
        call_args = mock_s3_operations['upload_json'].call_args[0]
        
        assert call_args[0] == "test-bucket"
        assert call_args[1] == "output/folder1/authors.json"
        assert len(call_args[2]) == 1
        assert call_args[2][0]["authors"] == ["Author 1", "Author 2"]
    
    def test_update_existing_authors_file(self, mock_s3_operations):
        """Test updating existing authors.json"""
        mock_s3_operations['file_exists'].return_value = True
        mock_s3_operations['read_json'].return_value = [
            {"doc_id": "1", "doc_issue": "1", "authors": ["Old Author"]}
        ]
        
        authors_list = ["New Author"]
        
        save_authors_to_s3_folder(
            "test-bucket", "output/", "folder1/",
            "1", "1", authors_list
        )
        
        # Verify existing entry was updated
        call_args = mock_s3_operations['upload_json'].call_args[0]
        assert call_args[2][0]["authors"] == ["New Author"]
    
    def test_add_to_existing_authors_file(self, mock_s3_operations):
        """Test adding new document to existing authors.json"""
        mock_s3_operations['file_exists'].return_value = True
        mock_s3_operations['read_json'].return_value = [
            {"doc_id": "1", "doc_issue": "1", "authors": ["Author 1"]}
        ]
        
        authors_list = ["Author 2"]
        
        save_authors_to_s3_folder(
            "test-bucket", "output/", "folder1/",
            "2", "1", authors_list
        )
        
        # Verify new entry was added
        call_args = mock_s3_operations['upload_json'].call_args[0]
        assert len(call_args[2]) == 2
    
    def test_handle_dict_format_authors(self, mock_s3_operations):
        """Test handling authors in dict format"""
        mock_s3_operations['file_exists'].return_value = False
        
        authors_list = [
            {"author_name": "Author 1"},
            {"author_name": "Author 2"}
        ]
        
        save_authors_to_s3_folder(
            "test-bucket", "output/", "",
            "1", "1", authors_list
        )
        
        call_args = mock_s3_operations['upload_json'].call_args[0]
        assert call_args[2][0]["authors"] == ["Author 1", "Author 2"]


class TestParseTamilDocument:
    """Test cases for parse_tamil_document function"""
    
    @pytest.fixture
    def sample_document_lines(self):
        """Create sample document lines"""
        return [
            "மலர் 1",
            "இதழ் 1",
            "",
            "பொருளடக்கம்",
            "Author 1",
            "Author 2",
            "ஆகியோரின் எழுத்தோவியங்கள்",
            "",
            "Test Heading",
            "Author 1",
            "Test content line 1",
            "Test content line 2",
            "Test content line 3",
            "Test content line 4",
            ""
        ]
    
    @patch('article_seperation.extract_pattern_a_forward')
    @patch('article_seperation.extract_pattern_b_forward')
    @patch('article_seperation.extract_pattern_c_reverse')
    @patch('article_seperation.extract_intro_content_phase1')
    @patch('article_seperation.extract_remaining_content')
    def test_parse_basic_document(self, mock_remaining, mock_intro, mock_c, 
                                  mock_b, mock_a, sample_document_lines):
        """Test basic document parsing"""
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_intro.return_value = ("test content", 10, None)
        mock_remaining.return_value = []
        
        shared_authors = {}
        result = parse_tamil_document(sample_document_lines, shared_authors)
        
        assert result["doc_id"] == "1"
        assert result["doc_issue"] == "1"
        assert "intro" in result
        assert "articles" in result
        assert "authors_list" in result
    
    @patch('article_seperation.extract_pattern_a_forward')
    @patch('article_seperation.extract_pattern_b_forward')
    @patch('article_seperation.extract_pattern_c_reverse')
    @patch('article_seperation.extract_remaining_content')
    def test_extract_authors_method1(self, mock_remaining, mock_c, mock_b, 
                                     mock_a, sample_document_lines):
        """Test author extraction method 1"""
        mock_a.return_value = []
        mock_b.return_value = []
        mock_c.return_value = []
        mock_remaining.return_value = []
        
        shared_authors = {}
        result = parse_tamil_document(sample_document_lines, shared_authors)
        
        assert len(result["authors_list"]) == 2
        assert result["authors_list"][0]["author_name"] == "Author 1"
        assert result["authors_list"][1]["author_name"] == "Author 2"
    
    @patch('article_seperation.extract_pattern_a_forward')
    @patch('article_seperation.extract_pattern_b_forward')
    @patch('article_seperation.extract_pattern_c_reverse')
    @patch('article_seperation.extract_remaining_content')
    def test_pattern_extraction_order(self, mock_remaining, mock_c, mock_b, mock_a):
        """Test that patterns are extracted in correct order"""
        lines = ["மலர் 1", "இதழ் 1", "பொருளடக்கம்", "ஆகியோரின்"]
        mock_a.return_value = [{"heading": "H1", "author": "A1", "content": "C1"}]
        mock_b.return_value = [{"heading": "H2", "author": "A2", "content": "C2"}]
        mock_c.return_value = [{"heading": "H3", "author": "A3", "content": "C3"}]
        mock_remaining.return_value = []
        
        result = parse_tamil_document(lines, {})
        
        # Verify all patterns were called
        assert mock_a.called
        assert mock_b.called
        assert mock_c.called
        
        # Verify articles are numbered sequentially
        assert result["articles"][0]["article_no"] == 1
        assert result["articles"][1]["article_no"] == 2
        assert result["articles"][2]["article_no"] == 3


class TestProcessS3Files:
    """Test cases for process_s3_files function"""
    
    @pytest.fixture
    def mock_s3_and_processing(self):
        """Setup comprehensive mocks"""
        with patch('article_seperation.list_files') as mock_list, \
             patch('article_seperation.read_text_from_s3') as mock_read, \
             patch('article_seperation.upload_json') as mock_upload, \
             patch('article_seperation.parse_tamil_document') as mock_parse, \
             patch('article_seperation.save_authors_to_s3_folder') as mock_save_authors, \
             patch('article_seperation.build_shared_authors_dict_s3') as mock_build:
            yield {
                'list_files': mock_list,
                'read_text': mock_read,
                'upload_json': mock_upload,
                'parse_document': mock_parse,
                'save_authors': mock_save_authors,
                'build_shared': mock_build
            }
    
    def test_process_single_file(self, mock_s3_and_processing):
        """Test processing a single TXT file"""
        mock_s3_and_processing['list_files'].return_value = ['input/test.txt']
        mock_s3_and_processing['read_text'].return_value = "மலர் 1\nஇதழ் 1"
        mock_s3_and_processing['build_shared'].return_value = {}
        mock_s3_and_processing['parse_document'].return_value = {
            "doc_id": "1",
            "doc_issue": "1",
            "intro": [],
            "articles": [{"article_no": 1}],
            "authors_list": []
        }
        
        process_s3_files()
        
        assert mock_s3_and_processing['upload_json'].call_count == 1
        assert mock_s3_and_processing['save_authors'].call_count == 1
    
    def test_process_no_files(self, mock_s3_and_processing):
        """Test when no TXT files are found"""
        mock_s3_and_processing['list_files'].return_value = []
        
        process_s3_files()
        
        mock_s3_and_processing['parse_document'].assert_not_called()
        mock_s3_and_processing['upload_json'].assert_not_called()
    
    def test_process_multiple_files(self, mock_s3_and_processing):
        """Test processing multiple files"""
        mock_s3_and_processing['list_files'].return_value = [
            'input/file1.txt',
            'input/file2.txt'
        ]
        mock_s3_and_processing['read_text'].return_value = "மலர் 1\nஇதழ் 1"
        mock_s3_and_processing['build_shared'].return_value = {}
        mock_s3_and_processing['parse_document'].return_value = {
            "doc_id": "1",
            "doc_issue": "1",
            "intro": [],
            "articles": [],
            "authors_list": []
        }
        
        process_s3_files()
        
        assert mock_s3_and_processing['upload_json'].call_count == 2
    
    def test_handle_processing_error(self, mock_s3_and_processing):
        """Test handling of processing errors"""
        mock_s3_and_processing['list_files'].return_value = ['input/test.txt']
        mock_s3_and_processing['read_text'].side_effect = Exception("Read error")
        mock_s3_and_processing['build_shared'].return_value = {}
        
        # Should not raise exception
        process_s3_files()
        
        # Should not upload anything
        mock_s3_and_processing['upload_json'].assert_not_called()
    
    def test_preserve_folder_structure(self, mock_s3_and_processing):
        """Test that folder structure is preserved in output"""
        mock_s3_and_processing['list_files'].return_value = [
            'input/year/month/file.txt'
        ]
        mock_s3_and_processing['read_text'].return_value = "मलर् 1\nஇதழ் 1"
        mock_s3_and_processing['build_shared'].return_value = {}
        mock_s3_and_processing['parse_document'].return_value = {
            "doc_id": "1",
            "doc_issue": "1",
            "intro": [],
            "articles": [],
            "authors_list": []
        }
        
        process_s3_files()
        
        # Check that upload was called with correct path structure
        upload_call = mock_s3_and_processing['upload_json'].call_args_list[0]
        output_key = upload_call[0][1]
        assert 'year/month/' in output_key
        assert output_key.endswith('.json')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])