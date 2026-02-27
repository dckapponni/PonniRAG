
import os
from pathlib import Path

BUCKET_NAME = "ponni-dev"
REGION_NAME = "ap-south-1"


INPUT_PREFIX = "Raw_Proof_Read_Content/"
OUTPUT_PREFIX = "output_json/"
EXTRACTED_OUTPUT= "extracted_text/"


CONFIG_DIR = Path(__file__).resolve().parent
CSV_PATH = CONFIG_DIR.parent / "data" / "summary.csv"

COLLECTION_NAME = "qdrant_indexer"
VECTOR_DISTANCE = "cosine"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 500
BATCH_SIZE = 100


S3_BUCKET = "ponni-dev"
S3_PREFIX = "output_json/"
S3_SUFFIX = ".json"

# Qdrant server connection (replaces local path mode)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# S3 snapshot persistence
SNAPSHOT_S3_PREFIX = "qdrant_snapshots/"


