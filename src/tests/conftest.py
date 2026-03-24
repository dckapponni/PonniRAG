"""Pytest configuration for module imports and S3 mocking."""

import sys
from pathlib import Path
from unittest.mock import Mock

import boto3
import numpy as np
import pytest
from moto import mock_aws


class MockSentenceTransformer:
    """Mock SentenceTransformer to avoid loading heavy models."""

    def __init__(self, *args, **kwargs):
        """Initialize mock transformer."""

    def encode(self, text, **kwargs):
        """
        Generate mock embeddings for input text.

        Args:
            text: String or list of strings to encode
            **kwargs: Additional arguments (ignored)

        Returns:
            numpy.ndarray: Random 384-dimensional embedding(s)
        """
        if isinstance(text, str):
            return np.random.rand(384)
        return np.array([np.random.rand(384) for _ in text])

    def to(self, device):
        """Mock device transfer method."""
        return self


mock_st_module = Mock()
mock_st_module.SentenceTransformer = MockSentenceTransformer
sys.modules["sentence_transformers"] = mock_st_module


class MockTextEmbedding:
    """Mock FastEmbed TextEmbedding to avoid model loading."""

    def __init__(self, *args, **kwargs):
        """Initialize mock text embedding."""

    def embed(self, texts, **kwargs):
        """
        Generate mock sparse embeddings for input texts.

        Args:
            texts: Iterable of strings to embed
            **kwargs: Additional arguments (ignored)

        Yields:
            list: Tuples of (index, value) representing sparse embeddings
        """
        for _ in texts:
            yield list(zip([1, 2, 3], [0.1, 0.2, 0.3]))


mock_fastembed = Mock()
mock_fastembed.TextEmbedding = MockTextEmbedding
sys.modules["fastembed"] = mock_fastembed


tests_dir = Path(__file__).resolve().parent
src_dir = tests_dir.parent
project_root = src_dir.parent
data_extraction_dir = src_dir / "data_extraction"
db_dir = src_dir / "db"
ui_dir = src_dir / "ui"

paths_to_add = [
    str(db_dir),
    str(data_extraction_dir),
    str(ui_dir),
    str(src_dir),
    str(project_root),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

print("[conftest.py] Added to sys.path:")
for path in paths_to_add:
    print(f"  - {path}")


def pytest_configure(config):
    """Register custom pytest markers for test categorization."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
        " requiring external services (Qdrant, S3, etc.)",
    )


@pytest.fixture(scope="function")
def aws_credentials():
    """
    Set up mocked AWS credentials for testing.

    Configures environment variables with dummy AWS credentials
    to prevent accidental use of real credentials during tests.
    """
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def s3_client(aws_credentials):
    """
    Create a mocked S3 client for testing.

    Args:
        aws_credentials: Fixture providing mocked AWS credentials

    Yields:
        boto3.client: Mocked S3 client instance
    """
    with mock_aws():
        conn = boto3.client("s3", region_name="us-east-1")
        yield conn


@pytest.fixture(scope="function")
def s3_bucket(s3_client):
    """
    Create a test S3 bucket.

    Args:
        s3_client: Mocked S3 client fixture

    Yields:
        str: Name of the created test bucket
    """
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    yield bucket_name


@pytest.fixture
def mock_dense_embedding():
    """
    Provide a mock function for generating dense embeddings.

    Returns:
        callable: Function that returns random 384-dimensional embeddings
    """

    def _mock_embed(text):
        return np.random.rand(384).tolist()

    return _mock_embed


@pytest.fixture
def mock_sparse_embedding():
    """
    Provide a mock function for generating sparse embeddings.

    Returns:
        callable: Function that returns SparseVector with mock indices and values
    """

    def _mock_embed(text):
        from qdrant_client.models import SparseVector

        return SparseVector(indices=[1, 2, 3, 4, 5], values=[0.1, 0.2, 0.3, 0.4, 0.5])

    return _mock_embed
