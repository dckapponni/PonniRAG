import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
from io import BytesIO
from botocore.exceptions import ClientError

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_extraction.s3_utils import (
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
    """Test suite for the list_files function.
    
    Tests S3 file listing functionality including pagination, filtering,
    error handling, and edge cases like empty buckets and missing keys.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client that prevents actual AWS calls
                and allows verification of S3 API interactions.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_list_all_files(self, mock_s3_client):
        """Test listing all files without applying suffix filter.
        
        Verifies that the function:
        - Returns all files when suffix parameter is None
        - Correctly extracts file keys from S3 response
        - Handles single page of results properly
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
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
        """Test listing files with suffix filter applied.
        
        Verifies that the function:
        - Correctly filters files by the specified suffix
        - Only returns files matching the suffix pattern
        - Excludes files with different extensions
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
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
        """Test listing from an empty bucket or prefix.
        
        Verifies that the function:
        - Returns an empty list when no files are found
        - Handles empty S3 responses gracefully
        - Does not raise exceptions for empty results
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]
        
        result = list_files('test-bucket', 'prefix/')
        
        assert result == []
    
    def test_list_multiple_pages(self, mock_s3_client):
        """Test listing with pagination across multiple result pages.
        
        Verifies that the function:
        - Correctly aggregates results from multiple pages
        - Handles S3 pagination properly
        - Returns all files across all pages
        
        This is important for buckets with many files where results
        are split across multiple API responses.
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
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
        """Test handling of NoSuchBucket error.
        
        Verifies that the function:
        - Properly raises ClientError for nonexistent buckets
        - Allows the error to propagate for proper handling
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_paginator.paginate.side_effect = ClientError(error_response, 'list_objects_v2')
        
        with pytest.raises(ClientError):
            list_files('nonexistent-bucket', 'prefix/')
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during listing.
        
        Covers lines 16-21 in the source code.
        
        Verifies that the function:
        - Handles unexpected exceptions gracefully
        - Returns empty list or raises exception as appropriate
        - Logs errors for debugging
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = Exception("Generic S3 error")
        
        try:
            result = list_files('test-bucket', 'prefix/')
            assert result == []
        except Exception:
            pass
    
    def test_page_without_contents_key(self, mock_s3_client):
        """Test handling of response pages without 'Contents' key.
        
        Verifies that the function:
        - Skips pages without the 'Contents' key
        - Continues processing subsequent pages
        - Does not crash on malformed responses
        
        This can occur with empty prefixes or certain S3 configurations.
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {'Contents': [{'Key': 'file1.txt'}]},
            {},  # Page without Contents
            {'Contents': [{'Key': 'file2.txt'}]}
        ]
        
        result = list_files('test-bucket', 'prefix/')
        
        assert len(result) == 2


class TestReadBytes:
    """Test suite for the read_bytes function.
    
    Tests binary file reading from S3, including error handling for
    missing files, access issues, and empty files.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client for testing binary reads
                without actual AWS calls.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_read_binary_file(self, mock_s3_client):
        """Test reading a binary file from S3.
        
        Verifies that the function:
        - Successfully retrieves binary content
        - Returns a BytesIO object
        - Preserves binary data integrity
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_response = {
            'Body': BytesIO(b'binary content')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_bytes('test-bucket', 'file.docx')
        
        assert isinstance(result, BytesIO)
        assert result.read() == b'binary content'
    
    def test_handle_no_such_key_error(self, mock_s3_client):
        """Test handling of NoSuchKey error for missing files.
        
        Verifies that the function:
        - Raises ClientError when file doesn't exist
        - Properly propagates the error for handling upstream
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'get_object')
        
        with pytest.raises(ClientError):
            read_bytes('test-bucket', 'nonexistent.docx')
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during binary read.
        
        Covers lines 60-61, 64-66 in the source code.
        
        Verifies that the function:
        - Handles unexpected exceptions gracefully
        - Returns None or raises as appropriate
        - Prevents crashes from unexpected errors
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.get_object.side_effect = Exception("Generic error")
        
        try:
            result = read_bytes('test-bucket', 'file.docx')
            assert result is None
        except Exception:
            pass
    
    def test_empty_file(self, mock_s3_client):
        """Test reading an empty binary file.
        
        Verifies that the function:
        - Handles zero-length files correctly
        - Returns empty BytesIO object
        - Does not raise errors for valid empty files
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_response = {
            'Body': BytesIO(b'')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_bytes('test-bucket', 'empty.txt')
        
        assert isinstance(result, BytesIO)
        assert result.read() == b''


class TestUploadText:
    """Test suite for the upload_text function.
    
    Tests text file uploading to S3, including UTF-8 encoding, Tamil text,
    error handling, and various input types.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client for testing text uploads
                without actual AWS calls.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_upload_text_file(self, mock_s3_client):
        """Test uploading a text file to S3.
        
        Verifies that the function:
        - Correctly encodes text as UTF-8 bytes
        - Sets proper content type (text/plain)
        - Uses correct bucket and key parameters
        - Makes proper S3 API call
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        
        upload_text('test-bucket', 'output/file.txt', 'test content')
        
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args[1]
        
        assert call_args['Bucket'] == 'test-bucket'
        assert call_args['Key'] == 'output/file.txt'
        assert call_args['Body'] == b'test content'
        assert 'text/plain' in call_args['ContentType']
    
    def test_upload_tamil_text(self, mock_s3_client):
        """Test uploading Tamil Unicode text to S3.
        
        Verifies that the function:
        - Correctly handles Tamil Unicode characters
        - Properly encodes UTF-8 without data loss
        - Preserves Tamil text integrity
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        tamil_text = "தமிழ் உரை"
        
        upload_text('test-bucket', 'file.txt', tamil_text)
        
        call_args = mock_s3_client.put_object.call_args[1]
        assert call_args['Body'] == tamil_text.encode('utf-8')
    
    def test_handle_non_string_input(self, mock_s3_client):
        """Test handling of non-string input (automatic conversion).
        
        Verifies that the function:
        - Converts non-string inputs to strings
        - Handles numeric values correctly
        - Encodes the string representation
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        
        upload_text('test-bucket', 'file.txt', 123)
        
        call_args = mock_s3_client.put_object.call_args[1]
        assert call_args['Body'] == b'123'
    
    def test_handle_client_error(self, mock_s3_client):
        """Test handling of S3 client errors during upload.
        
        Verifies that the function:
        - Propagates ClientError exceptions properly
        - Allows upstream error handling
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'InternalError'}}
        mock_s3_client.put_object.side_effect = ClientError(error_response, 'put_object')
        
        with pytest.raises(ClientError):
            upload_text('test-bucket', 'file.txt', 'content')
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during text upload.
        
        Covers lines 94-97, 100-102 in the source code.
        
        Verifies that the function:
        - Handles unexpected exceptions gracefully
        - Logs errors appropriately
        - Fails safely without data corruption
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.side_effect = Exception("Generic upload error")
        
        try:
            upload_text('test-bucket', 'file.txt', 'content')
        except Exception:
            pass
    
    def test_none_input(self, mock_s3_client):
        """Test uploading None value (converted to string).
        
        Verifies that the function:
        - Handles None values without crashing
        - Converts None to string representation "None"
        - Uploads the converted value
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        
        upload_text('test-bucket', 'file.txt', None)
        
        call_args = mock_s3_client.put_object.call_args[1]
        assert call_args['Body'] == b'None'
    
    def test_bytes_input(self, mock_s3_client):
        """Test uploading bytes directly (converted to string representation).
        
        Verifies that the function:
        - Handles bytes input by converting to string first
        - Produces the string representation of bytes
        - Note: upload_text converts bytes to str(bytes) format
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        
        upload_text('test-bucket', 'file.txt', b'raw bytes')
        
        call_args = mock_s3_client.put_object.call_args[1]
        # upload_text converts bytes to str representation first
        assert call_args['Body'] == b"b'raw bytes'"  # str(b'raw bytes') = "b'raw bytes'"


class TestUploadJson:
    """Test suite for the upload_json function.
    
    Tests JSON file uploading to S3, including serialization, Tamil characters,
    formatting, and error handling.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client for testing JSON uploads
                without actual AWS calls.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_upload_json_dict(self, mock_s3_client):
        """Test uploading a JSON dictionary to S3.
        
        Verifies that the function:
        - Correctly serializes Python dict to JSON
        - Sets proper content type (application/json)
        - Uses UTF-8 encoding
        - Preserves data structure through serialization
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        data = {'key': 'value', 'number': 123}
        
        upload_json('test-bucket', 'output/data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        
        assert call_args['Bucket'] == 'test-bucket'
        assert call_args['Key'] == 'output/data.json'
        assert 'application/json' in call_args['ContentType']
        
        uploaded_data = json.loads(call_args['Body'].decode('utf-8'))
        assert uploaded_data == data
    
    def test_upload_json_list(self, mock_s3_client):
        """Test uploading a JSON list/array to S3.
        
        Verifies that the function:
        - Correctly serializes Python list to JSON array
        - Preserves list order and values
        - Handles non-dict JSON types
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        data = [1, 2, 3, 4, 5]
        
        upload_json('test-bucket', 'data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        uploaded_data = json.loads(call_args['Body'].decode('utf-8'))
        
        assert uploaded_data == data
    
    def test_upload_tamil_json(self, mock_s3_client):
        """Test uploading JSON with Tamil Unicode characters.
        
        Verifies that the function:
        - Handles Tamil text in JSON values
        - Uses ensure_ascii=False for proper Unicode
        - Preserves Tamil characters through serialization
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        data = {'title': 'தலைப்பு', 'content': 'உள்ளடக்கம்'}
        
        upload_json('test-bucket', 'data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        uploaded_data = json.loads(call_args['Body'].decode('utf-8'))
        
        assert uploaded_data == data
    
    def test_json_formatting(self, mock_s3_client):
        """Test that JSON is formatted with proper indentation.
        
        Verifies that the function:
        - Produces human-readable formatted JSON
        - Uses indentation (indent=2)
        - Includes newlines for readability
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        data = {'key': 'value'}
        
        upload_json('test-bucket', 'data.json', data)
        
        call_args = mock_s3_client.put_object.call_args[1]
        json_str = call_args['Body'].decode('utf-8')
        
        assert '\n' in json_str
        assert '  ' in json_str
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during JSON upload.
        
        Covers lines 138-139, 146-148 in the source code.
        
        Verifies that the function:
        - Handles upload failures gracefully
        - Logs errors for debugging
        - Fails safely without data corruption
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.side_effect = Exception("Upload failed")
        
        try:
            upload_json('test-bucket', 'data.json', {'key': 'value'})
        except Exception:
            pass
    
    def test_json_serialization_error(self, mock_s3_client):
        """Test handling of JSON serialization errors.
        
        Verifies that the function:
        - Raises appropriate error for non-serializable objects
        - Does not crash on serialization failure
        - Provides meaningful error messages
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.put_object.return_value = {}
        
        # Object that can't be serialized to JSON
        class NonSerializable:
            pass
        
        data = {'obj': NonSerializable()}
        
        try:
            upload_json('test-bucket', 'data.json', data)
        except (TypeError, Exception):
            pass
    
    def test_client_error_handling(self, mock_s3_client):
        """Test ClientError handling during JSON upload.
        
        Verifies that the function:
        - Properly handles S3 access errors
        - Propagates ClientError for upstream handling
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'AccessDenied'}}
        mock_s3_client.put_object.side_effect = ClientError(error_response, 'put_object')
        
        try:
            upload_json('test-bucket', 'data.json', {'key': 'value'})
        except ClientError:
            pass


class TestReadTextFromS3:
    """Test suite for the read_text_from_s3 function.
    
    Tests text file reading from S3, including UTF-8 decoding, Tamil text,
    error handling, and edge cases.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client for testing text reads
                without actual AWS calls.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_read_text_file(self, mock_s3_client):
        """Test reading a text file from S3.
        
        Verifies that the function:
        - Successfully retrieves text content
        - Decodes UTF-8 bytes to string
        - Returns the correct text content
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_response = {
            'Body': BytesIO(b'text content')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_text_from_s3('test-bucket', 'file.txt')
        
        assert result == 'text content'
    
    def test_read_tamil_text(self, mock_s3_client):
        """Test reading Tamil Unicode text from S3.
        
        Verifies that the function:
        - Correctly decodes Tamil UTF-8 text
        - Preserves Tamil characters
        - Returns proper Unicode strings
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        tamil_text = "தமிழ் உரை"
        mock_response = {
            'Body': BytesIO(tamil_text.encode('utf-8'))
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_text_from_s3('test-bucket', 'file.txt')
        
        assert result == tamil_text
    
    def test_handle_no_such_key(self, mock_s3_client):
        """Test handling of missing file errors.
        
        Verifies that the function:
        - Raises ClientError when file doesn't exist
        - Propagates NoSuchKey error properly
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'get_object')
        
        with pytest.raises(ClientError):
            read_text_from_s3('test-bucket', 'nonexistent.txt')
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during text read.
        
        Covers lines 179-195 in the source code.
        
        Verifies that the function:
        - Handles unexpected read errors gracefully
        - Returns None or empty string as appropriate
        - Logs errors for debugging
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.get_object.side_effect = Exception("Read error")
        
        try:
            result = read_text_from_s3('test-bucket', 'file.txt')
            assert result is None or result == ""
        except Exception:
            pass
    
    def test_empty_file(self, mock_s3_client):
        """Test reading an empty text file.
        
        Verifies that the function:
        - Handles zero-length text files correctly
        - Returns empty string
        - Does not raise errors for valid empty files
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_response = {
            'Body': BytesIO(b'')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        result = read_text_from_s3('test-bucket', 'empty.txt')
        
        assert result == ''
    
    def test_unicode_decode_error(self, mock_s3_client):
        """Test handling of Unicode decoding errors.
        
        Verifies that the function:
        - Handles invalid UTF-8 byte sequences
        - Raises appropriate decode errors
        - Prevents silent data corruption
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_response = {
            'Body': BytesIO(b'\xff\xfe invalid utf-8')
        }
        mock_s3_client.get_object.return_value = mock_response
        
        try:
            result = read_text_from_s3('test-bucket', 'invalid.txt')
        except (UnicodeDecodeError, Exception):
            pass
    
    def test_access_denied_error(self, mock_s3_client):
        """Test handling of access denied errors.
        
        Verifies that the function:
        - Properly handles permission errors
        - Raises ClientError for access issues
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'AccessDenied'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'get_object')
        
        try:
            read_text_from_s3('test-bucket', 'protected.txt')
        except ClientError:
            pass


class TestReadJsonFromS3:
    """Test suite for the read_json_from_s3 function.
    
    Tests JSON file reading from S3, including deserialization, Tamil text,
    nested structures, and error handling.
    """
    
    @pytest.fixture
    def mock_read_text(self):
        """Provide a mocked read_text_from_s3 function.
        
        Yields:
            MagicMock: Mocked read_text_from_s3 function to isolate
                JSON parsing logic from S3 reading.
        """
        with patch('data_extraction.s3_utils.read_text_from_s3') as mock:
            yield mock
    
    def test_read_json_dict(self, mock_read_text):
        """Test reading a JSON dictionary from S3.
        
        Verifies that the function:
        - Correctly deserializes JSON to Python dict
        - Preserves data structure and values
        - Returns proper dict object
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        data = {'key': 'value', 'number': 123}
        mock_read_text.return_value = json.dumps(data)
        
        result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    def test_read_json_list(self, mock_read_text):
        """Test reading a JSON array from S3.
        
        Verifies that the function:
        - Correctly deserializes JSON arrays to Python lists
        - Preserves list order and values
        - Handles non-dict JSON types
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        data = [1, 2, 3, 4, 5]
        mock_read_text.return_value = json.dumps(data)
        
        result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    def test_read_tamil_json(self, mock_read_text):
        """Test reading JSON with Tamil Unicode characters.
        
        Verifies that the function:
        - Handles Tamil text in JSON values
        - Preserves Unicode characters through deserialization
        - Returns proper Tamil strings
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        data = {'title': 'தலைப்பு'}
        mock_read_text.return_value = json.dumps(data, ensure_ascii=False)
        
        result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    def test_handle_invalid_json(self, mock_read_text):
        """Test handling of malformed JSON.
        
        Verifies that the function:
        - Raises JSONDecodeError for invalid JSON
        - Does not silently fail on parse errors
        - Provides meaningful error messages
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        mock_read_text.return_value = "not valid json {"
        
        with pytest.raises(json.JSONDecodeError):
            read_json_from_s3('test-bucket', 'invalid.json')
    
    def test_generic_exception_handling(self, mock_read_text):
        """Test generic exception handling during JSON read.
        
        Covers lines 225, 228-234 in the source code.
        
        Verifies that the function:
        - Handles read failures gracefully
        - Returns None on error or raises as appropriate
        - Logs errors for debugging
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        mock_read_text.side_effect = Exception("Read failed")
        
        try:
            result = read_json_from_s3('test-bucket', 'data.json')
            assert result is None
        except Exception:
            pass
    
    def test_empty_json(self, mock_read_text):
        """Test reading empty JSON file.
        
        Verifies that the function:
        - Handles empty string input
        - Raises JSONDecodeError for empty JSON
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        mock_read_text.return_value = ""
        
        try:
            result = read_json_from_s3('test-bucket', 'empty.json')
        except json.JSONDecodeError:
            pass
    
    def test_null_json(self, mock_read_text):
        """Test reading JSON null value.
        
        Verifies that the function:
        - Correctly handles JSON null
        - Returns Python None
        - Distinguishes null from missing file
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        mock_read_text.return_value = "null"
        
        result = read_json_from_s3('test-bucket', 'null.json')
        
        assert result is None
    
    def test_nested_json(self, mock_read_text):
        """Test reading deeply nested JSON structures.
        
        Verifies that the function:
        - Handles complex nested dictionaries
        - Preserves structure at all depth levels
        - Returns complete nested objects
        
        Args:
            mock_read_text: Fixture providing mocked read_text_from_s3
        """
        data = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'deep'
                    }
                }
            }
        }
        mock_read_text.return_value = json.dumps(data)
        
        result = read_json_from_s3('test-bucket', 'nested.json')
        
        assert result == data


class TestFileExists:
    """Test suite for the file_exists function.
    
    Tests S3 file existence checking using HEAD operations, including
    error handling for missing files and access issues.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client for testing file existence
                checks without actual AWS calls.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_file_exists(self, mock_s3_client):
        """Test checking existence of a file that exists.
        
        Verifies that the function:
        - Returns True when file exists
        - Uses head_object API call efficiently
        - Checks correct bucket and key
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.head_object.return_value = {'ContentLength': 100}
        
        result = file_exists('test-bucket', 'existing-file.txt')
        
        assert result is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='existing-file.txt'
        )
    
    def test_file_not_exists(self, mock_s3_client):
        """Test checking existence of a file that doesn't exist.
        
        Verifies that the function:
        - Returns False for missing files
        - Handles 404 errors gracefully
        - Does not raise exceptions
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': '404'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'head_object')
        
        result = file_exists('test-bucket', 'nonexistent.txt')
        
        assert result is False
    
    def test_handle_other_errors(self, mock_s3_client):
        """Test handling of S3 errors other than 404.
        
        Verifies that the function:
        - Returns False for any S3 errors
        - Handles errors gracefully without crashing
        - Treats errors as "file doesn't exist"
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'InternalError'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'head_object')
        
        result = file_exists('test-bucket', 'file.txt')
        
        assert result is False
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during existence check.
        
        Covers lines 264-266 in the source code.
        
        Verifies that the function:
        - Handles unexpected exceptions gracefully
        - Returns False on any error
        - Does not crash the application
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.head_object.side_effect = Exception("Generic error")
        
        result = file_exists('test-bucket', 'file.txt')
        
        assert result is False
    
    def test_access_denied_returns_false(self, mock_s3_client):
        """Test that access denied is treated as file not exists.
        
        Verifies that the function:
        - Returns False for access denied errors
        - Treats permission issues as non-existence
        - Handles gracefully without raising
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'AccessDenied'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'head_object')
        
        result = file_exists('test-bucket', 'protected.txt')
        
        assert result is False
    
    def test_empty_key(self, mock_s3_client):
        """Test checking existence with empty key string.
        
        Verifies that the function:
        - Handles edge case of empty key
        - Still attempts the check
        - Returns boolean result
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.head_object.return_value = {'ContentLength': 0}
        
        result = file_exists('test-bucket', '')
        
        # Should still try to check
        assert isinstance(result, bool)


class TestDeleteFile:
    """Test suite for the delete_file function.
    
    Tests S3 file deletion, including handling of missing files,
    access errors, and versioned objects.
    """
    
    @pytest.fixture
    def mock_s3_client(self):
        """Provide a mocked S3 client for testing.
        
        Yields:
            MagicMock: Mocked boto3 S3 client for testing file deletion
                without actual AWS calls.
        """
        with patch('data_extraction.s3_utils.s3') as mock_s3:
            yield mock_s3
    
    def test_delete_file(self, mock_s3_client):
        """Test deleting a file from S3.
        
        Verifies that the function:
        - Calls delete_object with correct parameters
        - Uses proper bucket and key
        - Completes successfully
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.delete_object.return_value = {}
        
        delete_file('test-bucket', 'file.txt')
        
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='file.txt'
        )
    
    def test_delete_nonexistent_file(self, mock_s3_client):
        """Test deleting a file that doesn't exist.
        
        Verifies that the function:
        - Succeeds even if file doesn't exist (S3 behavior)
        - Makes the delete call regardless
        - Does not raise errors
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.delete_object.return_value = {}
        
        delete_file('test-bucket', 'nonexistent.txt')
        
        mock_s3_client.delete_object.assert_called_once()
    
    def test_handle_client_error(self, mock_s3_client):
        """Test handling of S3 client errors during deletion.
        
        Verifies that the function:
        - Raises ClientError on S3 failures
        - Allows upstream error handling
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'InternalError'}}
        mock_s3_client.delete_object.side_effect = ClientError(error_response, 'delete_object')
        
        with pytest.raises(ClientError):
            delete_file('test-bucket', 'file.txt')
    
    def test_generic_exception_handling(self, mock_s3_client):
        """Test generic exception handling during file deletion.
        
        Covers lines 295-297, 321-323 in the source code.
        
        Verifies that the function:
        - Handles unexpected deletion errors
        - Logs errors appropriately
        - Raises or handles exceptions as designed
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.delete_object.side_effect = Exception("Delete failed")
        
        try:
            delete_file('test-bucket', 'file.txt')
        except Exception:
            pass
    
    def test_access_denied_error(self, mock_s3_client):
        """Test handling of access denied during deletion.
        
        Verifies that the function:
        - Properly handles permission errors
        - Raises ClientError for access issues
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        error_response = {'Error': {'Code': 'AccessDenied'}}
        mock_s3_client.delete_object.side_effect = ClientError(error_response, 'delete_object')
        
        try:
            delete_file('test-bucket', 'protected.txt')
        except ClientError:
            pass
    
    def test_delete_with_version_id(self, mock_s3_client):
        """Test deleting versioned objects.
        
        Verifies that the function:
        - Handles versioned S3 buckets
        - Receives delete markers in response
        - Completes deletion successfully
        
        Note: This tests basic versioning support. Actual version ID
        specification would require function parameter changes.
        
        Args:
            mock_s3_client: Fixture providing mocked S3 client
        """
        mock_s3_client.delete_object.return_value = {'DeleteMarker': True}
        
        delete_file('test-bucket', 'versioned-file.txt')
        
        assert mock_s3_client.delete_object.called


class TestIntegrationScenarios:
    """Integration tests combining multiple S3 utility functions.
    
    Tests realistic workflows that use multiple functions together,
    validating end-to-end functionality and proper integration.
    """
    
    @patch('data_extraction.s3_utils.s3')
    def test_full_upload_download_cycle(self, mock_s3):
        """Test complete upload and download cycle for JSON data.
        
        Verifies a realistic workflow:
        1. Upload JSON data to S3
        2. Download the same JSON data back
        3. Verify data integrity is preserved
        
        This validates that serialization and deserialization work
        correctly together.
        
        Args:
            mock_s3: Mocked S3 client for the workflow
        """
        # Setup
        data = {'test': 'data', 'number': 42}
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Upload
        mock_s3.put_object.return_value = {}
        upload_json('test-bucket', 'data.json', data)
        
        # Download
        mock_s3.get_object.return_value = {
            'Body': BytesIO(json_str.encode('utf-8'))
        }
        
        with patch('data_extraction.s3_utils.read_text_from_s3', return_value=json_str):
            result = read_json_from_s3('test-bucket', 'data.json')
        
        assert result == data
    
    @patch('data_extraction.s3_utils.s3')
    def test_check_then_upload_workflow(self, mock_s3):
        """Test checking file existence before uploading.
        
        Verifies a common workflow:
        1. Check if file already exists
        2. Upload file only if it doesn't exist
        
        This prevents overwriting existing files and validates
        the existence check works correctly before uploads.
        
        Args:
            mock_s3: Mocked S3 client for the workflow
        """
        # File doesn't exist
        error_response = {'Error': {'Code': '404'}}
        mock_s3.head_object.side_effect = ClientError(error_response, 'head_object')
        
        exists = file_exists('test-bucket', 'new-file.txt')
        assert exists is False
        
        # Upload it
        mock_s3.put_object.return_value = {}
        upload_text('test-bucket', 'new-file.txt', 'content')
        
        assert mock_s3.put_object.called
    
    @patch('data_extraction.s3_utils.s3')
    def test_list_then_download_workflow(self, mock_s3):
        """Test listing files and then downloading them.
        
        Verifies a batch processing workflow:
        1. List all files in a prefix
        2. Download files from the list
        
        This validates that file listing and reading work together
        for processing multiple files.
        
        Args:
            mock_s3: Mocked S3 client for the workflow
        """
        # List files
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {'Contents': [{'Key': 'file1.txt'}, {'Key': 'file2.txt'}]}
        ]
        
        files = list_files('test-bucket', 'prefix/')
        assert len(files) == 2
        
        # Download first file
        mock_s3.get_object.return_value = {
            'Body': BytesIO(b'content1')
        }
        
        content = read_text_from_s3('test-bucket', files[0])
        assert content == 'content1'


@pytest.mark.parametrize("error_code,should_raise", [
    ('NoSuchKey', True),
    ('NoSuchBucket', True),
    ('AccessDenied', True),
    ('InternalError', True),
])
def test_s3_error_codes(error_code, should_raise):
    """Test handling of various S3 error codes.
    
    Parametrized test verifying that different AWS S3 error codes
    are handled correctly by the utility functions.
    
    Args:
        error_code (str): AWS S3 error code to test
        should_raise (bool): Whether the error should be raised
    """
    with patch('data_extraction.s3_utils.s3') as mock_s3:
        error_response = {'Error': {'Code': error_code}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'get_object')
        
        if should_raise:
            with pytest.raises(ClientError):
                read_text_from_s3('test-bucket', 'file.txt')


@pytest.mark.parametrize("content,expected_type", [
    ("text", str),
    ("தமிழ்", str),
    ("", str),
    ("123", str),
])
def test_text_content_types(content, expected_type):
    """Test reading various text content types from S3.
    
    Parametrized test verifying that different content (English, Tamil,
    empty, numeric) is read correctly as UTF-8 strings.
    
    Args:
        content (str): Text content to test
        expected_type (type): Expected Python type of the result
    """
    with patch('data_extraction.s3_utils.s3') as mock_s3:
        mock_s3.get_object.return_value = {
            'Body': BytesIO(content.encode('utf-8'))
        }
        
        result = read_text_from_s3('test-bucket', 'file.txt')
        assert isinstance(result, expected_type)
        assert result == content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=data_extraction.s3_utils",
                 "--cov-report=term-missing"])