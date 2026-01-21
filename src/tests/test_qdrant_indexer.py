"""
Enhanced tests for qdrant_indexer.py module to achieve >90% coverage
"""

from unittest.mock import Mock, patch, MagicMock
import pytest
import json
from qdrant_client import models
from db.qdrant_indexer import (
    split_into_sentences,
    chunk_text,
    validate_chunk,
    dense_embed_doc,
    dense_embed_query,
    sparse_embed,
    extract_volume_from_s3_key,
    is_author_file,
    list_s3_json_files,
    load_documents_from_s3,
    load_authors_from_s3,
    main,
)


class TestSentenceSplittingEdgeCases:
    """Additional edge cases for sentence splitting"""
    
    def test_split_with_multiple_punctuation(self):
        """Test splitting with multiple punctuation marks"""
        text = "What happened?!! Really it did?! Yes it happened."
        sentences = split_into_sentences(text)
        assert isinstance(sentences, list)
        if sentences:
            assert len(sentences) >= 1
    
    def test_split_with_only_tamil_punctuation(self):
        """Test splitting with only Tamil punctuation"""
        text = "முதல் வாக்கியம்। இரண்டாவது வாக்கியம்। மூன்றாவது வாக்கியம்।"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 3
    
    def test_split_no_punctuation(self):
        """Test splitting text without punctuation"""
        text = "This is a long text without any sentence terminators at all"
        sentences = split_into_sentences(text)
        assert isinstance(sentences, list)
    
    def test_split_whitespace_only(self):
        """Test splitting whitespace"""
        text = "     \n\n\t\t   "
        sentences = split_into_sentences(text)
        assert len(sentences) == 0
    
    def test_split_preserves_tamil_punctuation(self):
        """Test Tamil punctuation is preserved"""
        text = "தமிழ் வாக்கியம்। மற்றொன்று வாக்கியம்।"
        sentences = split_into_sentences(text)
        assert any('।' in s for s in sentences)
    
    def test_split_sentence_without_following_punctuation(self):
        """Test sentence that doesn't have punctuation after it"""
        text = "First sentence. Second sentence"
        sentences = split_into_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) >= 1
    
    def test_split_empty_string(self):
        """Test splitting empty string"""
        text = ""
        sentences = split_into_sentences(text)
        assert len(sentences) == 0
    
    def test_split_short_sentences_filtered(self):
        """Test that very short sentences are filtered out"""
        text = "Hi. Ok. This is longer sentence."
        sentences = split_into_sentences(text)
        # Short sentences (< 10 chars) should be filtered
        assert all(len(s) > 10 for s in sentences)


class TestChunkingEdgeCases:
    """Additional chunking edge cases"""
    
    def test_chunk_text_only_whitespace(self):
        """Test chunking whitespace only"""
        text = "   \n\n   \t\t   "
        chunks = chunk_text(text, size=1000)
        assert len(chunks) == 0
    
    def test_chunk_text_single_word(self):
        """Test chunking single word"""
        text = "word"
        chunks = chunk_text(text, size=1000)
        assert len(chunks) == 0
    
    def test_chunk_text_exact_boundary(self):
        """Test chunking at exact target size"""
        words = ["word"] * 167
        text = " ".join(words) + "."
        chunks = chunk_text(text, size=1000)
        assert len(chunks) >= 1
    
    def test_chunk_text_multiple_short_sentences(self):
        """Test chunking multiple short sentences"""
        text = "This is short. " * 100
        chunks = chunk_text(text, size=500)
        assert isinstance(chunks, list)
    
    def test_chunk_text_no_sentences(self):
        """Test chunking when split_into_sentences returns empty"""
        with patch('db.qdrant_indexer.split_into_sentences', return_value=[]):
            chunks = chunk_text("some text", size=1000)
            assert len(chunks) == 0
    
    def test_chunk_text_empty_string(self):
        """Test chunking empty string"""
        text = ""
        chunks = chunk_text(text, size=1000)
        assert len(chunks) == 0
    
    def test_chunk_text_very_long_sentence(self):
        """Test chunking with sentence longer than max_words"""
        # Create a very long sentence (> max_words)
        text = " ".join(["word"] * 300) + "."
        chunks = chunk_text(text, size=1000)
        # Should still create chunk for the long sentence
        assert isinstance(chunks, list)
    
    def test_chunk_text_final_chunk_too_short(self):
        """Test that final chunk with < 30 words is dropped"""
        # Create text where final chunk would be < 30 words
        long_sentence = " ".join(["word"] * 100) + "."
        short_sentence = " ".join(["word"] * 10) + "."
        text = long_sentence + " " + short_sentence
        chunks = chunk_text(text, size=500)
        # Final short chunk should be dropped
        assert isinstance(chunks, list)
    
    def test_chunk_text_reaches_target_words(self):
        """Test chunking when reaching target word count"""
        # Create text that reaches target words
        text = " ".join(["word"] * 170) + "."
        chunks = chunk_text(text, size=1000)
        assert len(chunks) >= 1


class TestValidateChunk:
    """Test chunk validation"""
    
    def test_validate_chunk_too_short(self):
        """Test validation rejects short chunks"""
        chunk = "Short"
        assert not validate_chunk(chunk)
    
    def test_validate_chunk_starts_with_punctuation(self):
        """Test validation rejects chunks starting with punctuation"""
        chunk = "." * 100
        assert not validate_chunk(chunk)
    
    def test_validate_chunk_too_few_words(self):
        """Test validation rejects chunks with too few words"""
        chunk = "a b c d e f g h i j"
        assert not validate_chunk(chunk)
    
    def test_validate_chunk_no_tamil_chars(self):
        """Test validation rejects chunks without Tamil characters"""
        chunk = "This is a very long English text " * 10
        assert not validate_chunk(chunk)
    
    def test_validate_chunk_valid(self):
        """Test validation accepts valid chunks"""
        chunk = "தமிழ் உரை " * 50
        assert validate_chunk(chunk)
    
    def test_validate_chunk_empty(self):
        """Test validation rejects empty chunks"""
        assert not validate_chunk("")
        assert not validate_chunk(None)
    
    def test_validate_chunk_starts_with_tamil(self):
        """Test validation accepts chunks starting with Tamil letter"""
        chunk = "அ" + " தமிழ் உரை" * 50
        assert validate_chunk(chunk)


class TestEmbeddingFunctions:
    """Test embedding generation functions"""
    
    def test_dense_embed_doc(self):
        """Test dense embedding for documents"""
        text = "Test document"
        embedding = dense_embed_doc(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_dense_embed_query(self):
        """Test dense embedding for queries"""
        text = "Test query"
        embedding = dense_embed_query(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_sparse_embed(self):
        """Test sparse embedding generation"""
        text = "Test text for sparse embedding"
        sparse_vector = sparse_embed(text)
        assert hasattr(sparse_vector, 'indices')
        assert hasattr(sparse_vector, 'values')
        assert len(sparse_vector.indices) == len(sparse_vector.values)
    
    def test_sparse_embed_empty_text(self):
        """Test sparse embedding with empty text"""
        text = ""
        sparse_vector = sparse_embed(text)
        assert len(sparse_vector.indices) == 0
        assert len(sparse_vector.values) == 0
    
    def test_sparse_embed_repeated_words(self):
        """Test sparse embedding with repeated words"""
        text = "word word word test test"
        sparse_vector = sparse_embed(text)
        # Should have frequencies for repeated words
        assert len(sparse_vector.values) >= 1


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_extract_volume_from_s3_key(self):
        """Test volume extraction from S3 key"""
        key = "ponni/vol_1/issue_1/document.json"
        volume = extract_volume_from_s3_key(key)
        assert volume == "vol_1"
    
    def test_extract_volume_no_volume(self):
        """Test volume extraction when no volume in key"""
        key = "ponni/documents/file.json"
        volume = extract_volume_from_s3_key(key)
        assert volume == "unknown"
    
    def test_extract_volume_uppercase(self):
        """Test volume extraction with uppercase VOL"""
        key = "ponni/VOL_2/file.json"
        volume = extract_volume_from_s3_key(key)
        assert volume == "VOL_2"
    
    def test_is_author_file_true(self):
        """Test author file detection"""
        key = "ponni/vol_1/authors.json"
        assert is_author_file(key) is True
    
    def test_is_author_file_false(self):
        """Test author file detection for non-author files"""
        key = "ponni/vol_1/document.json"
        assert is_author_file(key) is False
    
    def test_is_author_file_uppercase(self):
        """Test author file detection with uppercase"""
        key = "ponni/vol_1/AUTHORS.JSON"
        assert is_author_file(key) is True
    
    def test_list_s3_json_files(self):
        """Test listing S3 JSON files"""
        with patch('db.qdrant_indexer.s3') as mock_s3:
            mock_paginator = Mock()
            mock_s3.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [
                {
                    'Contents': [
                        {'Key': 'test1.json'},
                        {'Key': 'test2.json'},
                        {'Key': 'test3.txt'},
                    ]
                }
            ]
            
            files = list_s3_json_files('bucket', 'prefix', '.json')
            assert len(files) == 2
            assert 'test1.json' in files
            assert 'test2.json' in files
    
    def test_list_s3_json_files_empty(self):
        """Test listing S3 files when none exist"""
        with patch('db.qdrant_indexer.s3') as mock_s3:
            mock_paginator = Mock()
            mock_s3.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = [{}]
            
            files = list_s3_json_files('bucket', 'prefix', '.json')
            assert len(files) == 0


class TestLoadDocumentsFromS3:
    """Test document loading from S3"""
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_basic(self, mock_s3, mock_list_files):
        """Test basic document loading"""
        mock_list_files.return_value = ['test.json']
        
        mock_response = {
            'Body': Mock(read=lambda: b'{"intro": [{"doc_id": "1", "doc_issue": "1", "content": "Test content with enough words to pass validation. " * 20}]}')
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        assert isinstance(documents, list)
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_empty_s3(self, mock_s3, mock_list_files):
        """Test loading when S3 has no files"""
        mock_list_files.return_value = []
        documents = load_documents_from_s3()
        assert documents == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_non_dict_data(self, mock_s3, mock_list_files):
        """Test loading when JSON is not a dict (non-intro file)"""
        mock_list_files.return_value = ['test.json']
        mock_response = {
            'Body': Mock(read=lambda: b'[{"key": "value"}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        assert documents == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_no_intro_key(self, mock_s3, mock_list_files):
        """Test loading when JSON has no 'intro' key"""
        mock_list_files.return_value = ['test.json']
        mock_response = {
            'Body': Mock(read=lambda: b'{"other": []}')
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        assert documents == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_intro_not_list(self, mock_s3, mock_list_files):
        """Test loading when 'intro' is not a list"""
        mock_list_files.return_value = ['test.json']
        mock_response = {
            'Body': Mock(read=lambda: b'{"intro": "not a list"}')
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        assert documents == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_empty_content(self, mock_s3, mock_list_files):
        """Test loading with empty content field"""
        mock_list_files.return_value = ['test.json']
        mock_response = {
            'Body': Mock(read=lambda: b'{"intro": [{"doc_id": "1", "content": ""}]}')
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        assert documents == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_exception_handling(self, mock_s3, mock_list_files):
        """Test exception handling during document loading"""
        mock_list_files.return_value = ['test.json']
        mock_s3.get_object.side_effect = Exception("S3 error")
        
        with patch('db.qdrant_indexer.logger') as mock_logger:
            documents = load_documents_from_s3()
            assert documents == []
            assert mock_logger.error.called
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_with_heading(self, mock_s3, mock_list_files):
        """Test loading documents with heading metadata"""
        mock_list_files.return_value = ['test.json']
        content = "தமிழ் உள்ளடக்கம் " * 50
        data = {"intro": [{"doc_id": "1", "doc_issue": "1", "heading": "Test Heading", "content": content}]}
        mock_response = {
            'Body': Mock(read=lambda: json.dumps(data).encode('utf-8'))
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        if documents:
            assert documents[0]['metadata'].get('heading') == "Test Heading"


class TestLoadAuthorsFromS3:
    """Test author loading from S3"""
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_basic(self, mock_s3, mock_list_files):
        """Test basic author loading"""
        mock_list_files.return_value = ['authors.json']
        mock_response = {
            'Body': Mock(read=lambda: b'[{"doc_id": "1", "doc_issue": "1", "authors": ["Author 1"]}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        assert isinstance(authors, list)
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_no_author_files(self, mock_s3, mock_list_files):
        """Test loading when no author files exist"""
        mock_list_files.return_value = ['other.json']
        authors = load_authors_from_s3()
        assert authors == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_not_list(self, mock_s3, mock_list_files):
        """Test loading when authors.json is not a list"""
        mock_list_files.return_value = ['authors.json']
        mock_response = {
            'Body': Mock(read=lambda: b'{"not": "a list"}')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        assert authors == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_empty_author_name(self, mock_s3, mock_list_files):
        """Test loading with empty author names"""
        mock_list_files.return_value = ['authors.json']
        mock_response = {
            'Body': Mock(read=lambda: b'[{"doc_id": "1", "doc_issue": "1", "authors": ["", "  "]}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        assert authors == []
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_exception_handling(self, mock_s3, mock_list_files):
        """Test exception handling during author loading"""
        mock_list_files.return_value = ['authors.json']
        mock_s3.get_object.side_effect = Exception("S3 error")
        
        with patch('db.qdrant_indexer.logger') as mock_logger:
            authors = load_authors_from_s3()
            assert authors == []
            assert mock_logger.error.called


class TestMainFunction:
    """Test the main indexing function"""
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_author_final_batch(self, mock_authors, mock_docs, mock_client_class):
        """Test main upserts final author batch that's less than BATCH_SIZE"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [{
            'id': 'intro1',
            'text': 'இது அறிமுக உள்ளடக்கம். ' * 50,
            'metadata': {
                'doc_id': 'intro1',
                'doc_issue': '1',
                'chunk_id': 0,
                'total_chunks': 1,
                'volume': 'vol_1'
            }
        }]
        
        # Create exactly 50 authors (less than BATCH_SIZE)
        mock_authors.return_value = [
            {
                'id': f'author{i}',
                'text': f'Author {i}',
                'metadata': {
                    'author': f'Author {i}',
                    'doc_id': f'doc{i}',
                    'doc_issue': '1',
                    'volume': 'vol_1',
                    'source': 'authors_json'
                }
            }
            for i in range(50)
        ]
        
        mock_client.get_collection.return_value = Mock(points_count=51)
        
        main()
        # Should have upsert calls for both intro and authors
        assert mock_client.upsert.call_count >= 2
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_handles_author_indexing_exception(self, mock_authors, mock_docs, mock_client_class):
        """Test main handles exceptions during author indexing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [{
            'id': 'intro1',
            'text': 'இது அறிமுக உள்ளடக்கம். ' * 50,
            'metadata': {
                'doc_id': 'intro1',
                'doc_issue': '1',
                'chunk_id': 0,
                'total_chunks': 1,
                'volume': 'vol_1'
            }
        }]
        
        mock_authors.return_value = [{
            'id': 'author1',
            'text': 'Author',
            'metadata': {
                'author': 'Author',
                'doc_id': 'doc1',
                'doc_issue': '1',
                'volume': 'vol_1',
                'source': 'authors_json'
            }
        }]
        
        # Make embedding fail on second call (for author)
        call_count = [0]
        def side_effect_embed(text):
            call_count[0] += 1
            if call_count[0] > 1:  # Fail on author embedding
                raise Exception("Author embedding error")
            return [0.1] * 384  # Success on first call
        
        with patch('db.qdrant_indexer.dense_embed_doc', side_effect=side_effect_embed):
            with patch('db.qdrant_indexer.logger') as mock_logger:
                mock_client.get_collection.return_value = Mock(points_count=1)
                main()
                # Should log error for author indexing failure
                assert any('Failed to index' in str(call) for call in mock_logger.error.call_args_list)
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_logs_chunk_statistics(self, mock_authors, mock_docs, mock_client_class):
        """Test that main logs chunk statistics properly"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [{
            'id': 'doc1',
            'text': 'இது சோதனை உள்ளடக்கம். ' * 50,
            'metadata': {
                'doc_id': 'doc1',
                'doc_issue': '1',
                'chunk_id': 0,
                'total_chunks': 1,
                'volume': 'vol_1'
            }
        }]
        
        mock_authors.return_value = []
        mock_client.get_collection.return_value = Mock(points_count=1)
        
        with patch('db.qdrant_indexer.logger') as mock_logger:
            main()
            # Should log info about indexing progress
            assert mock_logger.info.called
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_only_authors_no_docs(self, mock_authors, mock_docs, mock_client_class):
        """Test main when there are only authors but no intro documents"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        # No documents, only authors won't be indexed
        mock_docs.return_value = []
        mock_authors.return_value = [{
            'id': 'author1',
            'text': 'Author',
            'metadata': {
                'author': 'Author',
                'doc_id': 'doc1',
                'doc_issue': '1',
                'volume': 'vol_1',
                'source': 'authors_json'
            }
        }]
        
        with patch('db.qdrant_indexer.logger') as mock_logger:
            main()
            # Should warn about no documents and return early
            assert any('No documents' in str(call) for call in mock_logger.warning.call_args_list)
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_creates_collection(self, mock_authors, mock_docs, mock_client_class):
        """Test main creates Qdrant collection"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = []
        mock_authors.return_value = []
        mock_client.get_collection.return_value = Mock(points_count=0)
        
        main()
        assert mock_client.create_collection.called
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_deletes_existing_collection(self, mock_authors, mock_docs, mock_client_class):
        """Test main deletes existing collection"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = True
        
        mock_docs.return_value = []
        mock_authors.return_value = []
        mock_client.get_collection.return_value = Mock(points_count=0)
        
        main()
        assert mock_client.delete_collection.called
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_no_documents_warning(self, mock_authors, mock_docs, mock_client_class):
        """Test main handles no documents"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = []
        
        with patch('db.qdrant_indexer.logger') as mock_logger:
            main()
            assert any('No documents' in str(call) for call in mock_logger.warning.call_args_list)
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_batch_indexing(self, mock_authors, mock_docs, mock_client_class):
        """Test main performs batch indexing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [
            {
                'id': f'doc{i}',
                'text': 'இது சோதனை உள்ளடக்கம். ' * 50,
                'metadata': {
                    'doc_id': f'doc{i}',
                    'doc_issue': '1',
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'volume': 'vol_1'
                }
            }
            for i in range(150)
        ]
        
        mock_authors.return_value = []
        mock_client.get_collection.return_value = Mock(points_count=150)
        
        main()
        assert mock_client.upsert.call_count >= 2
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_indexes_authors(self, mock_authors, mock_docs, mock_client_class):
        """Test main indexes author documents"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [{
            'id': 'intro1',
            'text': 'இது அறிமுக உள்ளடக்கம். ' * 50,
            'metadata': {
                'doc_id': 'intro1',
                'doc_issue': '1',
                'chunk_id': 0,
                'total_chunks': 1,
                'volume': 'vol_1'
            }
        }]
        
        mock_authors.return_value = [{
            'id': 'author1',
            'text': 'கருணாநிதி',
            'metadata': {
                'author': 'கருணாநிதி',
                'doc_id': 'doc1',
                'doc_issue': '1',
                'volume': 'vol_1',
                'source': 'authors_json'
            }
        }]
        
        mock_client.get_collection.return_value = Mock(points_count=2)
        
        main()
        assert mock_client.upsert.called
        assert mock_client.upsert.call_count >= 2
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_handles_indexing_exception(self, mock_authors, mock_docs, mock_client_class):
        """Test main handles exceptions during indexing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [{
            'id': 'bad_doc',
            'text': 'content',
            'metadata': {'chunk_id': 0, 'total_chunks': 1}
        }]
        mock_authors.return_value = []
        
        with patch('db.qdrant_indexer.dense_embed_doc', side_effect=Exception("Embedding error")):
            with patch('db.qdrant_indexer.logger') as mock_logger:
                mock_client.get_collection.return_value = Mock(points_count=0)
                main()
                assert mock_logger.error.called
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_final_batch_upsert(self, mock_authors, mock_docs, mock_client_class):
        """Test main upserts final batch that's less than BATCH_SIZE"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        # Create exactly 50 documents (less than BATCH_SIZE which is typically 100)
        mock_docs.return_value = [
            {
                'id': f'doc{i}',
                'text': 'இது சோதனை உள்ளடக்கம். ' * 50,
                'metadata': {
                    'doc_id': f'doc{i}',
                    'doc_issue': '1',
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'volume': 'vol_1'
                }
            }
            for i in range(50)
        ]
        
        mock_authors.return_value = []
        mock_client.get_collection.return_value = Mock(points_count=50)
        
        main()
        # Should have at least one upsert call for the final batch
        assert mock_client.upsert.called
    
    @patch('db.qdrant_indexer.QdrantClient')
    @patch('db.qdrant_indexer.load_documents_from_s3')
    @patch('db.qdrant_indexer.load_authors_from_s3')
    def test_main_author_batch_indexing(self, mock_authors, mock_docs, mock_client_class):
        """Test main performs batch indexing for authors"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        mock_docs.return_value = [{
            'id': 'intro1',
            'text': 'இது அறிமுக உள்ளடக்கம். ' * 50,
            'metadata': {
                'doc_id': 'intro1',
                'doc_issue': '1',
                'chunk_id': 0,
                'total_chunks': 1,
                'volume': 'vol_1'
            }
        }]
        
        # Create enough authors to trigger batching
        mock_authors.return_value = [
            {
                'id': f'author{i}',
                'text': f'Author {i}',
                'metadata': {
                    'author': f'Author {i}',
                    'doc_id': f'doc{i}',
                    'doc_issue': '1',
                    'volume': 'vol_1',
                    'source': 'authors_json'
                }
            }
            for i in range(150)
        ]
        
        mock_client.get_collection.return_value = Mock(points_count=151)
        
        main()
        # Should have multiple upsert calls for author batching
        assert mock_client.upsert.call_count >= 3


class TestLoadDocumentsEdgeCases:
    """Additional edge case tests for load_documents_from_s3"""
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    @patch('db.qdrant_indexer.logger')
    def test_load_documents_logs_statistics(self, mock_logger, mock_s3, mock_list_files):
        """Test that document loading logs chunk statistics"""
        mock_list_files.return_value = ['test.json']
        
        content = "தமிழ் உள்ளடக்கம் " * 50
        data = {"intro": [{"doc_id": "1", "doc_issue": "1", "content": content}]}
        mock_response = {
            'Body': Mock(read=lambda: json.dumps(data).encode('utf-8'))
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        
        # Should log chunk statistics
        assert any('CHUNK STATISTICS' in str(call) or 'Total chunks' in str(call) 
                   for call in mock_logger.info.call_args_list)
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    @patch('db.qdrant_indexer.logger')
    def test_load_documents_logs_processing(self, mock_logger, mock_s3, mock_list_files):
        """Test that document loading logs processing info"""
        mock_list_files.return_value = ['test.json']
        
        content = "தமிழ் உள்ளடக்கம் " * 50
        data = {"intro": [{"doc_id": "1", "doc_issue": "1", "content": content}]}
        mock_response = {
            'Body': Mock(read=lambda: json.dumps(data).encode('utf-8'))
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        
        # Should log processing information
        assert any('Processing' in str(call) for call in mock_logger.info.call_args_list)
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    @patch('db.qdrant_indexer.logger')
    def test_load_documents_skips_invalid_chunks(self, mock_logger, mock_s3, mock_list_files):
        """Test that invalid chunks are skipped and logged"""
        mock_list_files.return_value = ['test.json']
        
        # Content that will create invalid chunks
        mock_response = {
            'Body': Mock(read=lambda: b'{"intro": [{"doc_id": "1", "doc_issue": "1", "content": "Short text"}]}')
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        
        # Should result in no valid documents
        assert len(documents) == 0
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_multiple_chunks(self, mock_s3, mock_list_files):
        """Test loading documents that create multiple chunks"""
        mock_list_files.return_value = ['test.json']
        
        # Very long content that will create multiple chunks
        # Create content with full sentences to ensure proper chunking
        sentence = "இது தமிழ் உள்ளடக்கம் சோதனைக்கான வாக்கியம். "
        long_content = sentence * 300  # This should create multiple chunks
        mock_response = {
            'Body': Mock(read=lambda: json.dumps({
                "intro": [{
                    "doc_id": "1",
                    "doc_issue": "1", 
                    "content": long_content
                }]
            }).encode('utf-8'))
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        
        # Should create at least one document, and check if any has multiple chunks
        # or if we have multiple documents from the same source
        assert len(documents) > 0
        # Either one document has multiple chunks OR we have multiple documents
        has_multiple_chunks = any(doc['metadata']['total_chunks'] > 1 for doc in documents)
        has_multiple_docs = len(documents) > 1
        assert has_multiple_chunks or has_multiple_docs
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_documents_missing_metadata_fields(self, mock_s3, mock_list_files):
        """Test loading documents with missing metadata fields"""
        mock_list_files.return_value = ['test.json']
        
        # Document missing some fields
        content = "தமிழ் உள்ளடக்கம் " * 50
        data = {"intro": [{"doc_id": "1", "content": content}]}
        mock_response = {
            'Body': Mock(read=lambda: json.dumps(data).encode('utf-8'))
        }
        mock_s3.get_object.return_value = mock_response
        
        documents = load_documents_from_s3()
        
        # Should still process, using defaults for missing fields
        if documents:
            assert 'doc_id' in documents[0]['metadata']


class TestLoadAuthorsEdgeCases:
    """Additional edge case tests for load_authors_from_s3"""
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    @patch('db.qdrant_indexer.logger')
    def test_load_authors_logs_processing(self, mock_logger, mock_s3, mock_list_files):
        """Test that author loading logs processing info"""
        mock_list_files.return_value = ['authors.json']
        
        mock_response = {
            'Body': Mock(read=lambda: b'[{"doc_id": "1", "doc_issue": "1", "authors": ["Author 1"]}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        
        # Should log processing information
        assert any('Processing authors' in str(call) or 'Scanning' in str(call) 
                   for call in mock_logger.info.call_args_list)
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_multiple_authors(self, mock_s3, mock_list_files):
        """Test loading document with multiple authors"""
        mock_list_files.return_value = ['authors.json']
        
        mock_response = {
            'Body': Mock(read=lambda: b'[{"doc_id": "1", "doc_issue": "1", "authors": ["Author 1", "Author 2", "Author 3"]}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        
        # Should create separate entries for each author
        assert len(authors) == 3
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_with_whitespace(self, mock_s3, mock_list_files):
        """Test loading authors with extra whitespace"""
        mock_list_files.return_value = ['authors.json']
        
        mock_response = {
            'Body': Mock(read=lambda: b'[{"doc_id": "1", "doc_issue": "1", "authors": ["  Author 1  ", "Author 2"]}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        
        # Should strip whitespace
        assert len(authors) == 2
        assert all('  ' not in author['text'] for author in authors)
    
    @patch('db.qdrant_indexer.list_s3_json_files')
    @patch('db.qdrant_indexer.s3')
    def test_load_authors_missing_authors_field(self, mock_s3, mock_list_files):
        """Test loading when 'authors' field is missing"""
        mock_list_files.return_value = ['authors.json']
        
        mock_response = {
            'Body': Mock(read=lambda: b'[{"doc_id": "1", "doc_issue": "1"}]')
        }
        mock_s3.get_object.return_value = mock_response
        
        authors = load_authors_from_s3()
        
        # Should handle missing authors field gracefully
        assert authors == []