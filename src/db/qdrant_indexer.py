"""
Qdrant Index Module for Tamil Document Processing (Modified for new JSON format).
Handles document chunking, embedding generation, and indexing to Qdrant vector database.
Supports both intro documents and author metadata with hybrid search (dense + sparse vectors).

Modified to work with JSON format where articles are in an "articles" array.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List
import re
from collections import Counter

import boto3
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from src.config.config import (
    S3_BUCKET,
    S3_PREFIX,
    S3_SUFFIX,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    BATCH_SIZE,
)

BASE_DIR = Path(__file__).resolve().parent

QDRANT_PATH = str(BASE_DIR / "qdrant_data")

COLLECTION_NAME = Path(__file__).stem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

dense_model = SentenceTransformer(EMBEDDING_MODEL)


def dense_embed_doc(text: str) -> List[float]:
    """
    Generate dense embedding for documents during indexing.
    
    Args:
        text (str): Document text to embed
        
    Returns:
        list: Normalized dense embedding vector
    """
    return dense_model.encode(f"passage: {text}", normalize_embeddings=True).tolist()


def dense_embed_query(text: str) -> List[float]:
    """
    Generate dense embedding for search queries.
    
    Args:
        text (str): Query text to embed
        
    Returns:
        list: Normalized dense embedding vector
    """
    return dense_model.encode(f"query: {text}", normalize_embeddings=True).tolist()


def sparse_embed(text: str) -> models.SparseVector:
    """
    Generate BM25-style sparse embedding for text.
    
    Args:
        text (str): Text to embed
        
    Returns:
        models.SparseVector: Sparse vector with token indices and frequencies
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)

    indices: List[int] = []
    values: List[float] = []

    for token, freq in counts.items():
        idx = abs(hash(token)) % (2**31)
        indices.append(idx)
        values.append(float(freq))

    return models.SparseVector(indices=indices, values=values)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences respecting Tamil and English punctuation.
    
    Handles Tamil (।) and English (.!?) sentence terminators.
    Only returns substantial sentences (minimum 10 characters).
    
    Args:
        text (str): Text to split into sentences
        
    Returns:
        list: List of sentences with punctuation
    """
    sentences = re.split(r'([।.!?]+)', text)
    
    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        
        if not sentence:
            i += 1
            continue
        
        if i + 1 < len(sentences) and re.match(r'^[।.!?]+$', sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        if len(sentence) > 10:
            result.append(sentence.strip())
    
    return result


def chunk_text(text: str, size: int) -> List[str]:
    """
    Chunk text by sentences to ensure complete sentences in each chunk.
    
    Uses word-based chunking with target and maximum word counts.
    Ensures chunks start and end on sentence boundaries.
    
    Args:
        text (str): Text to chunk
        size (int): Target character size (converted to words internally)
        
    Returns:
        list: List of text chunks with complete sentences
    """
    if not text or not text.strip():
        return []
    
    target_words = size // 6
    max_words = int(target_words * 1.5)
    
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', sentence))
        
        if words > max_words:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue
        
        if current_word_count + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += words
        
        if current_word_count >= target_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        words_in_chunk = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', chunk_text))
        if words_in_chunk >= 30:
            chunks.append(chunk_text)
    
    return chunks


def validate_chunk(chunk: str) -> bool:
    """
    Validate chunk quality before indexing.
    
    Checks:
    - Minimum length (50 characters)
    - Starts with letter (not punctuation)
    - Minimum 30 words
    - Contains Tamil characters (minimum 20)
    
    Args:
        chunk (str): Text chunk to validate
        
    Returns:
        bool: True if chunk meets quality standards
    """
    if not chunk or len(chunk) < 50:
        return False
    
    if not re.match(r'^[a-zA-Zஅ-ஹ]', chunk.strip()):
        return False
    
    words = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', chunk))
    if words < 30:
        return False
    
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', chunk))
    if tamil_chars < 20:
        return False
    
    return True


def list_s3_json_files(bucket: str, prefix: str, suffix: str) -> List[str]:
    """
    List all JSON files in S3 bucket with given prefix and suffix.
    
    Args:
        bucket (str): S3 bucket name
        prefix (str): S3 key prefix to filter
        suffix (str): File suffix to filter (e.g., '.json')
        
    Returns:
        list: List of S3 keys matching criteria
    """
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                keys.append(key)
                logger.debug(f"Found: {key}") 
    return keys


def extract_volume_from_s3_key(s3_key: str) -> str:
    """
    Extract volume identifier from S3 key path.
    
    Looks for path segments like 'vol_1', 'vol_2', etc.
    
    Args:
        s3_key (str): S3 object key
        
    Returns:
        str: Volume identifier or 'unknown' if not found
    """
    for part in s3_key.split("/"):
        if part.lower().startswith("vol_"):
            return part
    return "unknown"


def is_author_file(s3_key: str) -> bool:
    """
    Check if S3 key points to an authors.json file.
    
    Args:
        s3_key (str): S3 object key
        
    Returns:
        bool: True if file is authors.json
    """
    return s3_key.lower().endswith("authors.json")


def load_documents_from_s3() -> List[Dict]:
    """
    Load and process article documents from S3.
    
    Performs:
    - Downloads JSON files from S3
    - Extracts articles from "articles" array
    - Chunks text into manageable pieces
    - Validates chunk quality
    - Creates document dictionaries with metadata
    
    Returns:
        list: List of document dictionaries ready for indexing
    """
    documents = []
    chunk_stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'word_counts': []
    }

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Found {len(keys)} JSON files in S3")

    for key in keys:
        # Skip authors.json files
        if is_author_file(key):
            continue
            
        logger.info(f"Processing: {key}")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            
            # Check if this is the new format with "articles" array
            if not isinstance(data, dict) or "articles" not in data:
                logger.info(f"Skipping non-articles JSON: {key}")
                continue
            
            articles = data.get("articles", [])
            if not isinstance(articles, list):
                logger.warning(f"'articles' is not a list in {key}")
                continue

            volume = extract_volume_from_s3_key(key)

            for article in articles:
                # Get content from the article
                content = article.get("content", "").strip()
                if not content:
                    continue

                chunks = chunk_text(content, CHUNK_SIZE)
                
                logger.info(f"  doc_id={article.get('doc_id')}, article_no={article.get('article_no')}: {len(chunks)} chunks")

                for idx, chunk in enumerate(chunks):
                    chunk_stats['total'] += 1
                    
                    if not validate_chunk(chunk):
                        chunk_stats['invalid'] += 1
                        logger.debug(f"    Skipped invalid chunk {idx}")
                        continue
                    
                    chunk_stats['valid'] += 1
                    word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', chunk))
                    chunk_stats['word_counts'].append(word_count)

                    documents.append({
                        "id": str(uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{key}-{article.get('doc_id')}-{article.get('article_no')}-{idx}"
                        )),
                        "text": chunk,
                        "metadata": {
                            "doc_id": article.get("doc_id"),
                            "doc_issue": article.get("doc_issue"),
                            "article_no": article.get("article_no"),
                            "author_name": article.get("author_name", ""),
                            "title": article.get("title", ""),
                            "year": article.get("year", ""),
                            "source_document": article.get("source_document", ""),
                            "source": "s3",
                            "s3_key": key,
                            "chunk_id": idx,
                            "total_chunks": len(chunks),
                            "volume": volume,
                        },
                    })

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")

    if chunk_stats['word_counts']:
        logger.info(f"\n{'='*60}")
        logger.info("CHUNK STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total chunks: {chunk_stats['total']}")
        logger.info(f"Valid chunks: {chunk_stats['valid']}")
        logger.info(f"Invalid chunks: {chunk_stats['invalid']}")
        logger.info(f"Words -> min={min(chunk_stats['word_counts'])}, "
                   f"max={max(chunk_stats['word_counts'])}, "
                   f"avg={sum(chunk_stats['word_counts'])//len(chunk_stats['word_counts'])}")

    logger.info(f"Prepared {len(documents)} valid chunks")
    return documents


def load_authors_from_s3() -> List[Dict]:
    """
    Load author metadata from S3 authors.json files.
    
    Extracts author information and creates document dictionaries
    with associated metadata (doc_id, issue, volume).
    
    Returns:
        list: List of author document dictionaries ready for indexing
    """
    documents = []

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Scanning {len(keys)} JSON files for authors.json")

    for key in keys:
        if not is_author_file(key):
            continue
        
        logger.info(f"Processing authors file: {key}")
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            
            if not isinstance(data, list):
                continue

            volume = extract_volume_from_s3_key(key)

            for item in data:
                doc_id = item.get("doc_id")
                doc_issue = item.get("doc_issue")

                for author in item.get("authors", []):
                    author = author.strip()
                    if not author:
                        continue

                    documents.append({
                        "id": str(uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{key}-{doc_id}-{doc_issue}-{author}"
                        )),
                        "text": author,
                        "metadata": {
                            "author": author,
                            "doc_id": doc_id,
                            "doc_issue": doc_issue,
                            "volume": volume,
                            "source": "authors_json",
                        }
                    })

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")

    logger.info(f"Prepared {len(documents)} authors")
    return documents


def main():
    """
    Main indexing function.
    
    Performs:
    1. Connects to Qdrant database
    2. Resets collection if exists
    3. Creates new collection with hybrid search support
    4. Indexes article documents with chunking and validation
    5. Indexes author metadata
    6. Logs statistics and completion status
    """
    client = QdrantClient(host="localhost", port=6333)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    logger.info(f"Created collection: {COLLECTION_NAME}")

    logger.info(f"\n{'-'*60}")
    logger.info("INDEXING ARTICLE DOCUMENTS")
    logger.info(f"{'-'*60}")
    
    documents = load_documents_from_s3()
    
    if not documents:
        logger.warning("No documents found")
        return

    points = []
    indexed = 0

    for doc in documents:
        try:
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_embed_doc(doc["text"]),
                    "sparse": sparse_embed(doc["text"]),
                },
                payload={
                    "content": doc["text"],
                    "chunk_id": doc["metadata"]["chunk_id"],
                    "type": "article",
                    "metadata": doc["metadata"],
                },
            )
            points.append(point)

            if len(points) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                indexed += len(points)
                logger.info(f"  Indexed {indexed}/{len(documents)}")
                points.clear()

        except Exception as e:
            logger.error(f"Failed to index: {e}")

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        indexed += len(points)
        logger.info(f"  Indexed {indexed}/{len(documents)}")

    logger.info(f"\n{'-'*60}")
    logger.info("INDEXING AUTHORS")
    logger.info(f"{'-'*60}")
    
    author_documents = load_authors_from_s3()
    
    points = []
    indexed_authors = 0

    for doc in author_documents:
        try:
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_embed_doc(doc["text"]),
                    "sparse": sparse_embed(doc["text"]),
                },
                payload={
                    "content": doc["text"],
                    "chunk_id": 0,
                    "type": "author",
                    "metadata": doc["metadata"],
                },
            )
            points.append(point)

            if len(points) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                indexed_authors += len(points)
                logger.info(f"  Indexed {indexed_authors}/{len(author_documents)} authors")
                points.clear()

        except Exception as e:
            logger.error(f"Failed to index author: {e}")

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        indexed_authors += len(points)
        logger.info(f"  Indexed {indexed_authors}/{len(author_documents)} authors")

    info = client.get_collection(COLLECTION_NAME)
    
    logger.info(f"\n{'='*60}")
    logger.info("INDEXING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total points: {info.points_count}")
    logger.info(f"  Article chunks: {indexed}")
    logger.info(f"  Authors: {indexed_authors}")
    logger.info(f"Database ready at: {QDRANT_PATH}")


if __name__ == "__main__":
    main()