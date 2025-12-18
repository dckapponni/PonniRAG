import json
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_extraction.s3_utils import list_files, read_bytes
from config.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    BATCH_SIZE,
    S3_BUCKET,
    S3_PREFIX,
    S3_SUFFIX,
    DEFAULT_VOLUME,
)

def load_documents_from_s3():
    documents = []

    print(f"Reading from S3 bucket: {S3_BUCKET}")
    print(f"Using prefix: {S3_PREFIX}")

    json_keys = list_files(
        bucket=S3_BUCKET,
        prefix=S3_PREFIX,
        suffix=S3_SUFFIX
    )

    print(f"Found {len(json_keys)} JSON files in S3")

    if not json_keys:
        return []

    for key in json_keys:
        try:
            raw_bytes = read_bytes(S3_BUCKET, key)
            data = json.loads(raw_bytes.read().decode("utf-8"))

            for item in data.get("intro", []):
                documents.append({
                    "content": item.get("content", ""),
                    "type": "intro",
                    "metadata": {
                        "doc_id": item.get("doc_id", ""),
                        "doc_issue": item.get("doc_issue", ""),
                        "heading": item.get("heading", ""),
                        "volume": DEFAULT_VOLUME
                    }
                })

            for item in data.get("articles", []):
                documents.append({
                    "content": item.get("article_content", ""),
                    "type": "article",
                    "metadata": {
                        "doc_id": item.get("doc_id", ""),
                        "doc_issue": item.get("doc_issue", ""),
                        "article_no": item.get("article_no", ""),
                        "article_heading": item.get("article_heading", ""),
                        "article_author_name": item.get("article_author_name", ""),
                        "volume": DEFAULT_VOLUME
                    }
                })

        except Exception as e:
            print(f"Failed to process file: {key}, error: {e}")

    print(f"Total documents loaded for embedding: {len(documents)}")
    return documents

def chunk_text(text: str):
    if not text or not text.strip():
        return []

    text = text.strip()
    return [
        text[i:i + CHUNK_SIZE]
        for i in range(0, len(text), CHUNK_SIZE)
        if text[i:i + CHUNK_SIZE].strip()
    ]


# ---------------------------
# Main Indexing Logic
# ---------------------------
def main():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)

    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )

    documents = load_documents_from_s3()
    if not documents:
        return

    points = []

    for doc in tqdm(documents, desc="Embedding"):
        chunks = chunk_text(doc["content"])
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            embedding = embedding_model.encode(
                [f"passage: {chunk}"]
            )[0]

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk,
                        "type": doc["type"],
                        "chunk_id": idx,
                        "total_chunks": total_chunks,
                        "metadata": doc["metadata"]
                    }
                )
            )

        if len(points) >= BATCH_SIZE:
            client.upsert(COLLECTION_NAME, points)
            points.clear()

    if points:
        client.upsert(COLLECTION_NAME, points)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Indexed vectors: {info.points_count}")


if __name__ == "__main__":
    main()
