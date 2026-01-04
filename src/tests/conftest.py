"""
Pytest configuration file to handle module imports and S3 mocking
Place this in: src/tests/conftest.py
"""
import sys
from pathlib import Path
import pytest
import boto3
from moto import mock_aws

# Add data_extraction directory to Python path so relative imports work
data_extraction_dir = Path(__file__).resolve().parent.parent / "data_extraction"
sys.path.insert(0, str(data_extraction_dir))

# Also add src directory
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


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