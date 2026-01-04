
from pathlib import Path
BUCKET_NAME = "ponni-dev"
REGION_NAME = "ap-south-1"

INPUT_PREFIX = "Raw_Proof_Read_Content/"
OUTPUT_PREFIX = "output_json/"
EXTRACTED_OUTPUT= "extracted_text/"



BASE_DIR = Path(__file__).resolve().parents[1]/"ui"
QDRANT_PATH = str(BASE_DIR / "qdrant_storage")

COLLECTION_NAME = "tamil_nexus_documents"
VECTOR_DISTANCE = "cosine"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 500
BATCH_SIZE = 100


S3_BUCKET = "ponni-dev"
S3_PREFIX = "output_json/"
S3_SUFFIX = ".json"



