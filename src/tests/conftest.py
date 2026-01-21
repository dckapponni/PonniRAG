"""
Pytest configuration file to handle module imports and S3 mocking
Place this in: src/tests/conftest.py
"""
import sys
from pathlib import Path
import pytest
import boto3
from moto import mock_aws
from unittest.mock import Mock, patch
import numpy as np


# ========== MOCK HEAVY DEPENDENCIES BEFORE IMPORTS ==========
# Mock SentenceTransformer to avoid loading models during test collection
class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    
    def encode(self, text, **kwargs):
        """Return mock embeddings as numpy array"""
        if isinstance(text, str):
            # Return numpy array so .tolist() works
            return np.random.rand(384)
        return np.array([np.random.rand(384) for _ in text])
    
    def to(self, device):
        return self

# Mock the entire sentence_transformers module
mock_st_module = Mock()
mock_st_module.SentenceTransformer = MockSentenceTransformer
sys.modules['sentence_transformers'] = mock_st_module

# Mock fastembed to avoid model loading
class MockTextEmbedding:
    def __init__(self, *args, **kwargs):
        pass
    
    def embed(self, texts, **kwargs):
        """Return mock sparse embeddings"""
        for _ in texts:
            yield list(zip([1, 2, 3], [0.1, 0.2, 0.3]))

mock_fastembed = Mock()
mock_fastembed.TextEmbedding = MockTextEmbedding
sys.modules['fastembed'] = mock_fastembed


# ========== PATH SETUP ==========
# Get the src directory (parent of tests directory)
tests_dir = Path(__file__).resolve().parent  # src/tests/
src_dir = tests_dir.parent  # src/
project_root = src_dir.parent  # project root
data_extraction_dir = src_dir / "data_extraction"  # src/data_extraction/
db_dir = src_dir / "db"  # src/db/
ui_dir = src_dir / "ui"  # src/ui/

# Add directories to Python path (avoid duplicates)
paths_to_add = [
    str(db_dir),  # For imports like: from hybrid_search import ... (highest priority)
    str(data_extraction_dir),  # For imports like: from text_processing import ...
    str(ui_dir),  # For imports like: from streamlit_app import ...
    str(src_dir),  # For imports like: from app import ...
    str(project_root),  # For imports like: from src.hybrid_search import ...
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"[conftest.py] Added to sys.path:")
for path in paths_to_add:
    print(f"  - {path}")


# ========== PYTEST MARKERS ==========
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as integration test requiring external services (Qdrant, S3, etc.)"
    )


# ========== AWS/S3 FIXTURES ==========
@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def s3_client(aws_credentials):
    """Create a mocked S3 client."""
    with mock_aws():
        conn = boto3.client("s3", region_name="us-east-1")
        yield conn


@pytest.fixture(scope="function")
def s3_bucket(s3_client):
    """Create a test S3 bucket."""
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    yield bucket_name


# ========== EMBEDDING FIXTURES ==========
@pytest.fixture
def mock_dense_embedding():
    """Mock dense embedding function"""
    def _mock_embed(text):
        return np.random.rand(384).tolist()
    return _mock_embed


@pytest.fixture
def mock_sparse_embedding():
    """Mock sparse embedding function"""
    def _mock_embed(text):
        from qdrant_client.models import SparseVector
        return SparseVector(
            indices=[1, 2, 3, 4, 5],
            values=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
    return _mock_embed
