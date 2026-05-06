"""Qdrant vector indexer for PonniRAG Tamil documents.

Loads articles and authors from S3, generates dense and sparse
embeddings, and upserts them into a Qdrant collection with
snapshot-based change detection.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from collections import Counter
from typing import Dict, List

import boto3
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config.config import (
    BATCH_SIZE,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    S3_BUCKET,
    S3_PREFIX,
    S3_SUFFIX,
)
from src.db.article_tagger import ArticleTagger
from src.db.retry import with_qdrant_retry
from src.db.snapshot_manager import (
    build_index_metadata,
    compute_source_data_hash,
    needs_reindex,
    restore_snapshot_from_s3,
    save_index_metadata,
    save_snapshot_to_s3,
)

COLLECTION_NAME = "qdrant_indexer"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # Already configured by another module

logger.info(f"Using device: {DEVICE}")

s3 = boto3.client("s3")

dense_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def dense_embed_doc(text: str) -> List[float]:
    """Generate dense embedding for documents during indexing."""
    text = unicodedata.normalize("NFC", text)
    return dense_model.encode(f"passage: {text}", normalize_embeddings=True).tolist()


def dense_embed_query(text: str) -> List[float]:
    """Generate dense embedding for search queries."""
    text = unicodedata.normalize("NFC", text)
    return dense_model.encode(f"query: {text}", normalize_embeddings=True).tolist()


def _deterministic_token_hash(token: str) -> int:
    """Deterministic token hash using MD5, consistent across processes.

    Python's built-in hash() is randomized per process (PYTHONHASHSEED),
    which causes sparse vectors at query time to mismatch those created
    at indexing time.  MD5 is deterministic and fast for this use case.
    """
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**31)


def sparse_embed(text: str) -> models.SparseVector:
    """Generate BM25-style sparse embedding for text."""
    text = unicodedata.normalize("NFC", text)
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)

    indices: List[int] = []
    values: List[float] = []

    for token, freq in counts.items():
        idx = _deterministic_token_hash(token)
        indices.append(idx)
        values.append(float(freq))

    return models.SparseVector(indices=indices, values=values)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences respecting Tamil and English punctuation."""
    sentences = re.split(r"([।.!?]+)", text)

    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()

        if not sentence:
            i += 1
            continue

        if i + 1 < len(sentences) and re.match(r"^[।.!?]+$", sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1

        if len(sentence) > 10:
            result.append(sentence.strip())

    return result


def chunk_text(text: str, size: int) -> List[str]:
    """Chunk text by sentences to ensure complete sentences in each chunk."""
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
        words = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", sentence))

        if words > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue

        if current_word_count + words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += words

        if current_word_count >= target_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

    if current_chunk:
        last_chunk = " ".join(current_chunk)
        words_in_chunk = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", last_chunk))
        if words_in_chunk >= 10:
            chunks.append(last_chunk)

    return chunks


def validate_chunk(chunk: str) -> bool:
    """Validate chunk quality before indexing."""
    if not chunk or len(chunk.strip()) < 20:
        return False

    # if not re.match(r"^[a-zA-Zஅ-ஹ]", chunk.strip()):
    #     return False

    words = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", chunk))
    if words < 10:
        return False

    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", chunk))
    if tamil_chars < 20:
        return False

    return True


def list_s3_json_files(bucket: str, prefix: str, suffix: str) -> List[str]:
    """List all JSON files in S3 bucket with given prefix and suffix."""
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
    """Extract volume identifier from S3 key path."""
    for part in s3_key.split("/"):
        if part.lower().startswith("vol_"):
            return part
    return "unknown"


def is_author_file(s3_key: str) -> bool:
    """Check if S3 key points to an authors.json file."""
    return s3_key.lower().endswith("authors.json")


def is_remaining_file(s3_key: str) -> bool:
    """Check if S3 key points to a remaining context file."""
    return "remaining_vol" in s3_key.lower()


def load_documents_from_s3() -> List[Dict]:
    """Load and process article documents from S3 with automatic tagging."""
    documents = []
    chunk_stats = {"total": 0, "valid": 0, "invalid": 0, "word_counts": []}

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Found {len(keys)} JSON files in S3")

    # --- Pass 1: Collect all articles for tagger training ---
    all_articles = []  # list of (article_dict, key, volume)
    for key in keys:
        if is_author_file(key):
            continue
        if is_remaining_file(key):
            logger.info(f"Skipping remaining file: {key}")
            continue
        logger.info(f"Loading: {key}")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))

            if not isinstance(data, dict) or "articles" not in data:
                logger.info(f"Skipping non-articles JSON: {key}")
                continue

            articles = data.get("articles", [])
            if not isinstance(articles, list):
                logger.warning(f"'articles' is not a list in {key}")
                continue

            volume = extract_volume_from_s3_key(key)

            for article in articles:
                content = article.get("content", "").strip()
                if not content:
                    continue
                all_articles.append((article, key, volume))

        except Exception as e:
            logger.error(f"Failed to load {key}: {e}")

    logger.info(f"Loaded {len(all_articles)} articles from S3")

    # --- Train article tagger on full corpus ---
    tagger = ArticleTagger()
    tagger.train_tfidf([a for a, _, _ in all_articles])
    logger.info("Article tagger trained")

    # --- Pass 2: Tag, chunk, and prepare documents ---
    for article, key, volume in all_articles:
        content = article.get("content", "").strip()

        # Tag at article level
        article_tags = tagger.tag_article(article)
        article_tags_tamil = tagger.get_tamil_tags(article_tags)

        chunks = chunk_text(content, CHUNK_SIZE)

        if "PONGAL" in key.upper():
            print("\n" + "=" * 80)
            print("TITLE:", article.get("title"))
            print("TOTAL CHUNKS:", len(chunks))

            for idx, chunk in enumerate(chunks):
                valid = validate_chunk(chunk)

                print(f"\nCHUNK {idx}")
                print("VALID:", valid)

                words = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", chunk))
                print("WORDS:", words)

                print("START:")
                print(repr(chunk[:120]))

        logger.info(
            f"  doc_id={article.get('doc_id')}, "
            f"article_no={article.get('article_no')}: "
            f"{len(chunks)} chunks, tags={article_tags}"
        )

        for idx, chunk in enumerate(chunks):
            chunk_stats["total"] += 1

            if not validate_chunk(chunk):
                chunk_stats["invalid"] += 1
                logger.debug(f"    Skipped invalid chunk {idx}")
                continue

            chunk_stats["valid"] += 1
            word_count = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", chunk))
            chunk_stats["word_counts"].append(word_count)

            doc_id = article.get("doc_id")
            art_no = article.get("article_no")
            uuid_name = f"{key}-{doc_id}-{art_no}-{idx}"
            documents.append(
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            uuid_name,
                        )
                    ),
                    "text": chunk,
                    "metadata": {
                        "doc_id": article.get("doc_id"),
                        "doc_issue": article.get("doc_issue"),
                        "article_no": article.get("article_no"),
                        "author_name": (
                            ", ".join(article.get("author_name", []))
                            if isinstance(article.get("author_name"), list)
                            else article.get("author_name", "")
                        ),
                        "title": article.get("title", ""),
                        "year": article.get("year", ""),
                        "source_document": article.get("source_document", ""),
                        "source": "s3",
                        "s3_key": key,
                        "chunk_id": idx,
                        "total_chunks": len(chunks),
                        "volume": volume,
                        "tags": article_tags,
                        "tags_tamil": article_tags_tamil,
                    },
                }
            )

    if chunk_stats["word_counts"]:
        logger.info(f"\n{'='*60}")
        logger.info("CHUNK STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total chunks: {chunk_stats['total']}")
        logger.info(f"Valid chunks: {chunk_stats['valid']}")
        logger.info(f"Invalid chunks: {chunk_stats['invalid']}")
        logger.info(
            f"Words -> min={min(chunk_stats['word_counts'])}, "
            f"max={max(chunk_stats['word_counts'])}, "
            f"avg={sum(chunk_stats['word_counts'])//len(chunk_stats['word_counts'])}"
        )

    logger.info(f"Prepared {len(documents)} valid chunks")
    return documents


def load_authors_from_s3() -> List[Dict]:
    """Load author metadata from S3 authors.json files."""
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

                    documents.append(
                        {
                            "id": str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_DNS,
                                    f"{key}-{doc_id}-{doc_issue}-{author}",
                                )
                            ),
                            "text": author,
                            "metadata": {
                                "author": author,
                                "doc_id": doc_id,
                                "doc_issue": doc_issue,
                                "volume": volume,
                                "source": "authors_json",
                            },
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")

    logger.info(f"Prepared {len(documents)} authors")
    return documents


def _do_full_index(client):
    """Run full indexing: create collection, embed articles + authors, upsert."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
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
                with_qdrant_retry(
                    client.upsert,
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                indexed += len(points)
                logger.info(f"  Indexed {indexed}/{len(documents)}")
                points.clear()

        except Exception as e:
            logger.error(f"Failed to index: {e}")

    if points:
        with_qdrant_retry(
            client.upsert,
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
                with_qdrant_retry(
                    client.upsert,
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                indexed_authors += len(points)
                logger.info(
                    f"  Indexed {indexed_authors}/{len(author_documents)} authors"
                )
                points.clear()

        except Exception as e:
            logger.error(f"Failed to index author: {e}")

    if points:
        with_qdrant_retry(
            client.upsert,
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


def main(force_reindex: bool = False):
    """Run smart indexing with change detection and snapshot management."""
    logger.info(f"Connecting to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    s3_client = s3

    if force_reindex:
        logger.info("--force-reindex: skipping change detection, running full index")
        _do_full_index(client)
        _save_snapshot_and_metadata(client, s3_client)
        return

    # Check if re-indexing is needed
    reindex_needed, reason = needs_reindex(s3_client, S3_BUCKET, S3_PREFIX, S3_SUFFIX)

    if not reindex_needed:
        logger.info(f"No reindex needed: {reason}")

        # Check if collection already exists in Qdrant (e.g. Docker volume survived)
        if client.collection_exists(COLLECTION_NAME):
            info = client.get_collection(COLLECTION_NAME)
            logger.info(
                f"Collection '{COLLECTION_NAME}' exists with "
                f"{info.points_count} points — nothing to do"
            )
            return

        # Collection missing — try to restore from S3 snapshot
        logger.info("Collection not found in Qdrant — attempting S3 snapshot restore")
        restored = restore_snapshot_from_s3(
            client, s3_client, S3_BUCKET, COLLECTION_NAME
        )
        if restored:
            info = client.get_collection(COLLECTION_NAME)
            logger.info(
                f"Restored collection with {info.points_count} points from S3 snapshot"
            )
            return

        logger.info("Snapshot restore failed — falling through to full reindex")

    else:
        logger.info(f"Reindex needed: {reason}")

    # Full re-index
    _do_full_index(client)
    _save_snapshot_and_metadata(client, s3_client)


def _save_snapshot_and_metadata(client, s3_client):
    """Save snapshot to S3 and update index metadata after indexing."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        snapshot_key = save_snapshot_to_s3(
            client, s3_client, S3_BUCKET, COLLECTION_NAME
        )
        source_hash = compute_source_data_hash(
            s3_client, S3_BUCKET, S3_PREFIX, S3_SUFFIX
        )
        metadata = build_index_metadata(
            snapshot_s3_key=snapshot_key,
            points_count=info.points_count,
            source_data_hash=source_hash,
        )
        save_index_metadata(s3_client, S3_BUCKET, metadata)
        logger.info("Snapshot and metadata saved to S3")
    except Exception as e:
        logger.error(f"Failed to save snapshot/metadata to S3: {e}")
        logger.info("Index is available in Qdrant but not persisted to S3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PonniRAG Qdrant Indexer")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force full re-index ignoring S3 state",
    )
    args = parser.parse_args()
    main(force_reindex=args.force_reindex)
