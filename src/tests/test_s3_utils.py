import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from io import BytesIO
from botocore.exceptions import ClientError

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_extraction.s3_utils import (
    list_files,
    read_bytes,
    upload_text,
    upload_json,
    read_text_from_s3,
    read_json_from_s3,
    file_exists,
    delete_file
)


class TestListFiles:
    """Test cases for list_files function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_list_all_files(self, mock_s3_client):
        """Test listing all files without suffix filter"""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {'Key': 'file1.txt'},
                    {'Key': 'file2.docx'},
                    {'Key': 'file3.json'}
                ]
            }
        ]
        
        result = list_files('test-bucket', 'prefix/', suffix=None)
        
        assert len(result) == 3
        assert 'file1.txt' in result
        assert 'file2.docx' in result
        assert 'file3.json' in result
    
    def test_list_with_suffix_filter(self, mock_s3_client):
        """Test listing files with suffix filter"""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {'Key': 'file1.txt'},
                    {'Key': 'file2.docx'},
                    {'Key': 'file3.txt'}
                ]
            }
        ]
        
        result = list_files('test-bucket', 'prefix/', suffix='.txt')
        
        assert len(result) == 2
        assert 'file1.txt' in result
        assert 'file3.txt' in result
        assert 'file2.docx' not in result
    
    def test_list_empty_bucket(self, mock_s3_client):
        """Test listing from empty bucket/prefix"""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]
        
        result = list_files('test-bucket', 'prefix/')
        
        assert result == []
    
    def test_list_multiple_pages(self, mock_s3_client):
        """Test listing with pagination"""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {'Contents': [{'Key': 'file1.txt'}]},
            {'Contents': [{'Key': 'file2.txt'}]},
            {'Contents': [{'Key': 'file3.txt'}]}
        ]
        
        result = list_files('test-bucket', 'prefix/')
        
        assert len(result) == 3
    
    def test_handle_no_such_bucket_error(self, mock_s3_client):
        """Test handling NoSuchBucket error"""
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_paginator.paginate.side_effect = ClientError(error_response, 'list_objects_v2')
        
        with pytest.raises(ClientError):
            list_files('nonexistent-bucket', 'prefix/')


class TestReadBytes:
    """Test cases for read_bytes function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_read_binary_file(self, mock_s3_client):
        """Test reading binary file"""
        mock_response = {
            'Body': BytesIO(b'binary content')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_bytes('test-bucket', 'file.docx')
        
        assert isinstance(result, BytesIO)
        assert result.read() == b'binary content'
    
    def test_handle_no_such_key_error(self, mock_s3_client):
        """Test handling NoSuchKey error"""
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'get_object')
        
        with pytest.raises(ClientError):
            read_bytes('test-bucket', 'nonexistent.docx')


class TestUploadText:
    """Test cases for upload_text function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_upload_text_file(self, mock_s3_client):
        """Test uploading text file"""
        mock_s3_client.put_object.return_value = {}
        
        upload_text('test-bucket', 'output/file.txt', 'test content')
        
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args[1]
        
        assert call_args['Bucket'] == 'test-bucket'
        assert call_args['Key'] == 'output/file.txt'
        assert call_args['Body'] == b'test content'
        assert 'text/plain' in call_args['ContentType']
    
    def test_upload_tamil_text(self, mock_s3_client):
        """Test uploading Tamil text"""
        mock_s3_client.put_object.return_value = {}
        tamil_text = "தமிழ் உரை"
        
        upload_text('test-bucket', 'file.txt', tamil_text)
        
        call_args = mock_s3_client.put_object.call_args[1]
        assert call_args['Body'] == tamil_text.encode('utf-8')
    
    def test_handle_non_string_input(self, mock_s3_client):
        """Test handling non-string input"""
        mock_s3_client.put_object.return_value = {}
        
        upload_text('test-bucket', 'file.txt', 123)
        
        call_args = mock_s3_client.put_object.call_args[1]
        assert call_args['Body'] == b'123'
    
    def test_handle_client_error(self, mock_s3_client):
        """Test handling S3 client error"""
        error_response = {'Error': {'Code': 'InternalError'}}
        mock_s3_client.put_object.side_effect = ClientError(error_response, 'put_object')
        
        with pytest.raises(ClientError):
            upload_text('test-bucket', 'file.txt', 'content')


class TestUploadJson:
    """Test cases for upload_json function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_upload_json_dict(self, mock_s3_client):
        """Test uploading JSON from dict"""
        mock_s3_client.put_object.return_value = {}
        data = {'key': 'value', 'number': 123}
        
        upload_json('test-bucket', 'output/data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        
        assert call_args['Bucket'] == 'test-bucket'
        assert call_args['Key'] == 'output/data.json'
        assert 'application/json' in call_args['ContentType']
        
        # Verify JSON can be parsed
        uploaded_data = json.loads(call_args['Body'].decode('utf-8'))
        assert uploaded_data == data
    
    def test_upload_json_list(self, mock_s3_client):
        """Test uploading JSON from list"""
        mock_s3_client.put_object.return_value = {}
        data = [1, 2, 3, 4, 5]
        
        upload_json('test-bucket', 'data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        uploaded_data = json.loads(call_args['Body'].decode('utf-8'))
        
        assert uploaded_data == data
    
    def test_upload_tamil_json(self, mock_s3_client):
        """Test uploading JSON with Tamil characters"""
        mock_s3_client.put_object.return_value = {}
        data = {'title': 'தலைப்பு', 'content': 'உள்ளடக்கம்'}
        
        upload_json('test-bucket', 'data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        uploaded_data = json.loads(call_args['Body'].decode('utf-8'))
        
        assert uploaded_data == data
    
    def test_json_formatting(self, mock_s3_client):
        """Test that JSON is formatted with indentation"""
        mock_s3_client.put_object.return_value = {}
        data = {'key': 'value'}
        
        upload_json('test-bucket', 'data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        json_str = call_args['Body'].decode('utf-8')
        
        # Should have indentation (newlines and spaces)
        assert '\n' in json_str
        assert '  ' in json_str


class TestReadTextFromS3:
    """Test cases for read_text_from_s3 function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_read_text_file(self, mock_s3_client):
        """Test reading text file"""
        mock_response = {
            'Body': BytesIO(b'text content')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_text_from_s3('test-bucket', 'file.txt')
        
        assert result == 'text content'
    
    def test_read_tamil_text(self, mock_s3_client):
        """Test reading Tamil text"""
        tamil_text = "தமிழ் உரை"
        mock_response = {
            'Body': BytesIO(tamil_text.encode('utf-8'))
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_text_from_s3('test-bucket', 'file.txt')
        
        assert result == tamil_text
    
    def test_handle_no_such_key(self, mock_s3_client):
        """Test handling file not found"""
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'get_object')
        
        with pytest.raises(ClientError):
            read_text_from_s3('test-bucket', 'nonexistent.txt')


class TestReadJsonFromS3:
    """Test cases for read_json_from_s3 function"""
    
    @pytest.fixture
    def mock_read_text(self):
        """Mock read_text_from_s3 at module level"""
        with patch('src.data_extraction.s3_utils.read_text_from_s3') as mock:
            yield mock
    
    def test_read_json_dict(self, mock_read_text):
        """Test reading JSON dict"""
        data = {'key': 'value', 'number': 123}
        mock_read_text.return_value = json.dumps(data)
        
        result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    def test_read_json_list(self, mock_read_text):
        """Test reading JSON list"""
        data = [1, 2, 3, 4, 5]
        mock_read_text.return_value = json.dumps(data)
        
        result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    def test_read_tamil_json(self, mock_read_text):
        """Test reading JSON with Tamil characters"""
        data = {'title': 'தலைப்பு'}
        mock_read_text.return_value = json.dumps(data, ensure_ascii=False)
        
        result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    def test_handle_invalid_json(self, mock_read_text):
        """Test handling invalid JSON"""
        mock_read_text.return_value = "not valid json {"
        
        with pytest.raises(json.JSONDecodeError):
            read_json_from_s3('test-bucket', 'invalid.json')


class TestFileExists:
    """Test cases for file_exists function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_file_exists(self, mock_s3_client):
        """Test when file exists"""
        mock_s3_client.head_object.return_value = {'ContentLength': 100}
        
        result = file_exists('test-bucket', 'existing-file.txt')
        
        assert result is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='existing-file.txt'
        )
    
    def test_file_not_exists(self, mock_s3_client):
        """Test when file does not exist"""
        error_response = {'Error': {'Code': '404'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'head_object')
        
        result = file_exists('test-bucket', 'nonexistent.txt')
        
        assert result is False
    
    def test_handle_other_errors(self, mock_s3_client):
        """Test handling other errors (returns False)"""
        error_response = {'Error': {'Code': 'InternalError'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'head_object')
        
        result = file_exists('test-bucket', 'file.txt')
        
        # Should return False for any error
        assert result is False


class TestDeleteFile:
    """Test cases for delete_file function"""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client at module level"""
        with patch('src.data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_delete_file(self, mock_s3_client):
        """Test deleting a file"""
        mock_s3_client.delete_object.return_value = {}
        
        delete_file('test-bucket', 'file.txt')
        
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='file.txt'
        )
    
    def test_delete_nonexistent_file(self, mock_s3_client):
        """Test deleting nonexistent file (should succeed)"""
        mock_s3_client.delete_object.return_value = {}
        
        # S3 delete_object succeeds even if file doesn't exist
        delete_file('test-bucket', 'nonexistent.txt')
        
        mock_s3_client.delete_object.assert_called_once()
    
    def test_handle_client_error(self, mock_s3_client):
        """Test handling client error"""
        error_response = {'Error': {'Code': 'InternalError'}}
        mock_s3_client.delete_object.side_effect = ClientError(error_response, 'delete_object')
        
        with pytest.raises(ClientError):
            delete_file('test-bucket', 'file.txt')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])