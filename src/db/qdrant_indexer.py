
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
    """
    Dense embedding for DOCUMENTS (indexing)
    """
    return dense_model.encode(f"passage: {text}").tolist()


def dense_embed_query(text: str) -> List[float]:
    """
    Dense embedding for SEARCH QUERIES
    """
    return dense_model.encode(f"query: {text}").tolist()

def sparse_embed(text: str) -> models.SparseVector:
    """
    Converts text into a sparse vector compatible with Qdrant.
    This is a lightweight BM25-style tokenizer with hashing.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)

    indices: List[int] = []
    values: List[float] = []

    for token, freq in counts.items():
        # Stable hash → positive int
        idx = abs(hash(token)) % (2**31)
        indices.append(idx)
        values.append(float(freq))

    return models.SparseVector(indices=indices, values=values)


def chunk_text(text: str, size: int) -> List[str]:
    if not text or not text.strip():
        return []
    return [
        text[i: i + size].strip()
        for i in range(0, len(text), size)
        if text[i: i + size].strip()
    ]


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
    """
    Extract volume name like vol_1, vol_2 from S3 path
    """
    for part in s3_key.split("/"):
        if part.lower().startswith("vol_"):
            return part
    return "unknown"

def is_author_file(s3_key: str) -> bool:
    return s3_key.lower().endswith("authors.json")

def load_documents_from_s3() -> List[Dict]:
    documents = []
    chunk_lengths = []

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Found {len(keys)} JSON files in S3")

    for key in keys:
        logger.info(f"[INTRO CHECK] Processing file: {key}")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            
            if not isinstance(data, dict):
                logger.info(f"[INTRO SKIP] Not an intro JSON (likely authors): {key}")
                continue
            logger.info(f"[INTRO JSON LOADED] {key} | keys={list(data.keys())}")
            intro_items = data.get("intro", [])
            if not isinstance(intro_items, list):
                continue

            volume = extract_volume_from_s3_key(key)

            for item in intro_items:
                text = item.get("content", "").strip()
                if not text:
                    continue

                chunks = chunk_text(text, CHUNK_SIZE)

                # ✅ per-document chunk count
                logger.info(
                    f"S3 file: {key} | doc_id: {item.get('doc_id')} | "
                    f"Chunks created: {len(chunks)}"
                )

                for idx, chunk in enumerate(chunks):
                    chunk_lengths.append(len(chunk))  # collect stats

                    documents.append(
                        {
                            "id": str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_DNS,
                                    f"{key}-{item.get('doc_id')}-{idx}"
                                )
                            ),
                            "text": chunk,
                            "metadata": {
                                "doc_id": item.get("doc_id"),
                                "doc_issue": item.get("doc_issue"),
                                "heading": item.get("heading"),
                                "source": "s3",
                                "s3_key": key,
                                "chunk_id": idx,
                                "total_chunks": len(chunks),
                                "volume": volume,
                            },
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")

    if chunk_lengths:
        logger.info(
            f"Chunk stats → "
            f"count={len(chunk_lengths)}, "
            f"min={min(chunk_lengths)}, "
            f"max={max(chunk_lengths)}, "
            f"avg={sum(chunk_lengths)//len(chunk_lengths)}"
        )

    logger.info(f"Prepared {len(documents)} text chunks for indexing")
    return documents
def load_authors_from_s3() -> List[Dict]:
    documents = []

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Scanning {len(keys)} JSON files for authors.json")

    for key in keys:
        if not is_author_file(key):
            logger.debug(f"[SKIP] Not authors.json: {key}")
            continue
        logger.info(f"[AUTHORS FILE] Processing: {key}")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            logger.info(
                f"[AUTHORS JSON LOADED] {key} | entries={len(data)}"
            )
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
                        "id": str(
                            uuid.uuid5(
                                uuid.NAMESPACE_DNS,
                                f"{key}-{doc_id}-{doc_issue}-{author}"
                            )
                        ),
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
            logger.error(f"Failed to process authors file {key}: {e}")

    logger.info(f"Prepared {len(documents)} author entries for indexing")
    return documents


def main():
    client = QdrantClient(path=QDRANT_PATH)

    # Reset collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")

    # Create dense-only collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
        "sparse": models.SparseVectorParams()
    }
    )

    logger.info(f"Created collection: {COLLECTION_NAME}")

    documents = load_documents_from_s3()
    if not documents:
        logger.warning("No documents found to index")
        return
    author_documents = load_authors_from_s3()

    points = []

    for doc in documents:
        try:
            point = models.PointStruct(
            id=doc["id"],
            vector={
                "dense": dense_embed_doc(doc["text"]),
                "sparse": sparse_embed(doc["text"]),
            },
            payload={
                # 🔑 what RAG reads
                "content": doc["text"],
                "chunk_id": doc["metadata"]["chunk_id"],
                "type": "intro",  # because this comes from "intro" JSON

                # 🔑 nested metadata (VERY IMPORTANT)
                "metadata": {
                    "doc_id": doc["metadata"]["doc_id"],
                    "doc_issue": doc["metadata"]["doc_issue"],
                    "heading": doc["metadata"]["heading"],
                    "volume": doc["metadata"]["volume"],
                    "source": doc["metadata"]["source"],
                    "s3_key": doc["metadata"]["s3_key"],
                    "total_chunks": doc["metadata"]["total_chunks"],
                },
            },)
            points.append(point)

            if len(points) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                logger.info(f"Indexed batch of {len(points)}")
                points.clear()

        except Exception as e:
            logger.error(f"Failed to embed/index document: {e}")

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Indexed final batch of {len(points)}")
    # -------------------------------
    # Index AUTHORS
    # -------------------------------
    points = []  # reset buffer

    for doc in author_documents:
        try:
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_embed_doc(doc["text"]),
                    "sparse": sparse_embed(doc["text"]),
                },
                payload={
                    "content": doc["text"],   # author name
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
                logger.info(f"Indexed author batch of {len(points)}")
                points.clear()

        except Exception as e:
            logger.error(f"Failed to embed/index author: {e}")

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Indexed final author batch of {len(points)}")

    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Total vectors indexed: {info.points_count}")
    logger.info(
    f"[SUMMARY] Intro chunks indexed={len(documents)}, "
    f"Author entries indexed={len(author_documents)}"
)


if __name__ == "__main__":
    main()
