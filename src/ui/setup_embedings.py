
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


BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(BASE_DIR / "qdrant_storage")

COLLECTION_NAME = "tamil_nexus_documents"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 500

S3_BUCKET = "ponni-dev"
S3_PREFIX = "output_json//vol_1/"


def load_documents_from_s3():
    documents = []

    print(f"Reading from S3 bucket: {S3_BUCKET}")
    print(f"Using prefix: {S3_PREFIX}")

    json_keys = list_files(
        bucket=S3_BUCKET,
        prefix=S3_PREFIX,
        suffix=".json"
    )

    print(f"Found {len(json_keys)} JSON files in S3")

    if not json_keys:
        print("No JSON files found. Check bucket, prefix, or permissions.")
        return []

    for key in json_keys:
        try:
            raw_bytes = read_bytes(S3_BUCKET, key)
            data = json.loads(raw_bytes.read().decode("utf-8"))

            if "intro" in data:
                for item in data["intro"]:
                    documents.append({
                        "content": item.get("content", ""),
                        "type": "intro",
                        "metadata": {
                            "doc_id": item.get("doc_id", ""),
                            "doc_issue": item.get("doc_issue", ""),
                            "heading": item.get("heading", ""),
                            "volume": "vol_1"
                        }
                    })

            if "articles" in data:
                for item in data["articles"]:
                    documents.append({
                        "content": item.get("article_content", ""),
                        "type": "article",
                        "metadata": {
                            "doc_id": item.get("doc_id", ""),
                            "doc_issue": item.get("doc_issue", ""),
                            "article_no": item.get("article_no", ""),
                            "article_heading": item.get("article_heading", ""),
                            "article_author_name": item.get("article_author_name", ""),
                            "volume": "vol_1"
                        }
                    })

        except Exception as e:
            print(f"Failed to process file: {key}, error: {e}")
            continue

    print(f"Total documents loaded for embedding: {len(documents)}")
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE):
    if not text or not text.strip():
        return []

    text = text.strip()
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]


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
    batch_size = 100

    for doc in tqdm(documents, desc="Embedding"):
        chunks = chunk_text(doc["content"])
        if not chunks:
            continue

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

        if len(points) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            points.clear()

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Indexed vectors: {collection_info.points_count}")


if __name__ == "__main__":
    main()
