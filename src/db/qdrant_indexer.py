"""
FIXED qdrant_index.py
Replace your entire file with this
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

# Qdrant storage will be created in the same directory as this file
QDRANT_PATH = str(BASE_DIR / "qdrant_data")

# Collection name derived from python file name
COLLECTION_NAME = Path(__file__).stem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

dense_model = SentenceTransformer(EMBEDDING_MODEL)

def dense_embed_doc(text: str) -> List[float]:
    """Dense embedding for DOCUMENTS (indexing)"""
    return dense_model.encode(f"passage: {text}", normalize_embeddings=True).tolist()


def dense_embed_query(text: str) -> List[float]:
    """Dense embedding for SEARCH QUERIES"""
    return dense_model.encode(f"query: {text}", normalize_embeddings=True).tolist()


def sparse_embed(text: str) -> models.SparseVector:
    """BM25-style sparse embedding"""
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
    """
    # Split by Tamil (।) and English (.!?) punctuation
    sentences = re.split(r'([।.!?]+)', text)
    
    # Recombine sentences with their punctuation
    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        
        if not sentence:
            i += 1
            continue
        
        # Check if next item is punctuation
        if i + 1 < len(sentences) and re.match(r'^[।.!?]+$', sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        # Only add substantial sentences (min 10 chars)
        if len(sentence) > 10:
            result.append(sentence.strip())
    
    return result


def chunk_text(text: str, size: int) -> List[str]:
    """
    FIXED: Chunk text by SENTENCES, not characters.
    Ensures no incomplete sentences.
    
    Args:
        text: Text to chunk
        size: Target CHARACTER size (converted to ~words internally)
        
    Returns:
        List of chunks with complete sentences
    """
    if not text or not text.strip():
        return []
    
    # Convert character size to approximate word target
    # Average Tamil word ~= 6 chars, English ~= 5 chars
    # So CHUNK_SIZE=1000 chars ≈ 170 words
    target_words = size // 6
    max_words = int(target_words * 1.5)  # Allow 50% overflow
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        # Count words in sentence (Tamil + English)
        words = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', sentence))
        
        # If single sentence is huge, take it as-is
        if words > max_words:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue
        
        # If adding would exceed max, save current chunk
        if current_word_count + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += words
        
        # If reached target, consider new chunk
        if current_word_count >= target_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
    
    # Add remaining chunk if substantial (min 30 words)
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        words_in_chunk = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', chunk_text))
        if words_in_chunk >= 30:
            chunks.append(chunk_text)
    
    return chunks


def validate_chunk(chunk: str) -> bool:
    """
    Validate chunk quality.
    
    Returns True if:
    - Starts with proper character (not punctuation)
    - Has minimum 30 words
    - Contains Tamil characters
    """
    if not chunk or len(chunk) < 50:
        return False
    
    # Must start with letter
    if not re.match(r'^[a-zA-Zஅ-ஹ]', chunk.strip()):
        return False
    
    # Count words
    words = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', chunk))
    if words < 30:
        return False
    
    # Must have Tamil content
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', chunk))
    if tamil_chars < 20:
        return False
    
    return True


def list_s3_json_files(bucket: str, prefix: str, suffix: str) -> List[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                keys.append(key)
                logger.debug(f"[S3 FOUND] {key}") 
    return keys


def extract_volume_from_s3_key(s3_key: str) -> str:
    """Extract volume name like vol_1, vol_2 from S3 path"""
    for part in s3_key.split("/"):
        if part.lower().startswith("vol_"):
            return part
    return "unknown"


def is_author_file(s3_key: str) -> bool:
    return s3_key.lower().endswith("authors.json")


def load_documents_from_s3() -> List[Dict]:
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
        logger.info(f"[PROCESSING] {key}")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            
            if not isinstance(data, dict):
                logger.info(f"[SKIP] Not an intro JSON: {key}")
                continue
            
            intro_items = data.get("intro", [])
            if not isinstance(intro_items, list):
                continue

            volume = extract_volume_from_s3_key(key)

            for item in intro_items:
                text = item.get("content", "").strip()
                if not text:
                    continue

                # Use improved chunking
                chunks = chunk_text(text, CHUNK_SIZE)
                
                logger.info(f"  doc_id={item.get('doc_id')}: {len(chunks)} chunks")

                for idx, chunk in enumerate(chunks):
                    chunk_stats['total'] += 1
                    
                    # Validate chunk
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
                            f"{key}-{item.get('doc_id')}-{idx}"
                        )),
                        "text": chunk,
                        "metadata": {
                            "doc_id": item.get("doc_id"),
                            "doc_issue": item.get("doc_issue"),
                            "heading": item.get("heading", ""),
                            "source": "s3",
                            "s3_key": key,
                            "chunk_id": idx,
                            "total_chunks": len(chunks),
                            "volume": volume,
                        },
                    })

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")

    # Log statistics
    if chunk_stats['word_counts']:
        logger.info(f"\n{'='*60}")
        logger.info("CHUNK STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total chunks: {chunk_stats['total']}")
        logger.info(f"Valid chunks: {chunk_stats['valid']}")
        logger.info(f"Invalid chunks: {chunk_stats['invalid']}")
        logger.info(f"Words → min={min(chunk_stats['word_counts'])}, "
                   f"max={max(chunk_stats['word_counts'])}, "
                   f"avg={sum(chunk_stats['word_counts'])//len(chunk_stats['word_counts'])}")

    logger.info(f"✓ Prepared {len(documents)} valid chunks")
    return documents


def load_authors_from_s3() -> List[Dict]:
    documents = []

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Scanning {len(keys)} JSON files for authors.json")

    for key in keys:
        if not is_author_file(key):
            continue
        
        logger.info(f"[AUTHORS FILE] {key}")
        
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

    logger.info(f"✓ Prepared {len(documents)} authors")
    return documents


def main():
    client = QdrantClient(host="localhost", port=6333)


    # Reset collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"✓ Deleted existing collection")

    # Create collection with hybrid search
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

    logger.info(f"✓ Created collection: {COLLECTION_NAME}")

    # Index intro documents
    logger.info(f"\n{'─'*60}")
    logger.info("INDEXING INTRO DOCUMENTS")
    logger.info(f"{'─'*60}")
    
    documents = load_documents_from_s3()
    
    if not documents:
        logger.warning("⚠ No documents found!")
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
                    "type": "intro",
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

    # Index authors
    logger.info(f"\n{'─'*60}")
    logger.info("INDEXING AUTHORS")
    logger.info(f"{'─'*60}")
    
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

    # Summary
    info = client.get_collection(COLLECTION_NAME)
    
    logger.info(f"\n{'='*60}")
    logger.info("INDEXING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total points: {info.points_count}")
    logger.info(f"  Intro chunks: {indexed}")
    logger.info(f"  Authors: {indexed_authors}")
    logger.info(f"✓ Database ready at: {QDRANT_PATH}")


if __name__ == "__main__":
    main()