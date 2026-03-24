"""Configuration loader for PonniRAG magazine registry and environment settings."""

import json
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

_registry_cache = None


def load_registry():
    """Load magazine_registry.json (cached after first call)."""
    global _registry_cache
    if _registry_cache is None:
        registry_path = CONFIG_DIR / "magazine_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            _registry_cache = json.load(f)
    return _registry_cache


def get_magazine_config(magazine_id="ponni"):
    """Return config dict for a specific magazine."""
    registry = load_registry()
    if magazine_id not in registry:
        raise ValueError(f"Unknown magazine: {magazine_id}")
    return registry[magazine_id]


_default = get_magazine_config("ponni")
_s3 = _default["s3"]

BUCKET_NAME = _s3["bucket"]
REGION_NAME = _s3["region"]

INPUT_PREFIX = _s3["raw_content"]
OUTPUT_PREFIX = _s3["output_json"]
EXTRACTED_OUTPUT = _s3["extracted_text"]

CSV_PATH = CONFIG_DIR.parent / "data" / "summary.csv"

COLLECTION_NAME = _default["qdrant_collection"]
VECTOR_DISTANCE = "cosine"

EMBEDDING_MODEL = _default["embedding"]["model"]
EMBEDDING_DIM = _default["embedding"]["dim"]
CHUNK_SIZE = 500
BATCH_SIZE = 100

S3_BUCKET = _s3["bucket"]
S3_PREFIX = _s3["output_json"]
S3_SUFFIX = ".json"

# Qdrant server connection (replaces local path mode)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# S3 snapshot persistence
SNAPSHOT_S3_PREFIX = _s3["snapshots"]
MAX_QUERY_LENGTH = 500
