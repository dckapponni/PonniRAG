"""
Tests for src/db/snapshot_manager.py

Uses moto for S3 mocking and unittest.mock for Qdrant client mocking.
"""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import boto3
import pytest
from moto import mock_aws

from db.snapshot_manager import (
    compute_embedding_fingerprint,
    compute_source_data_hash,
    load_index_metadata,
    save_index_metadata,
    needs_reindex,
    save_snapshot_to_s3,
    restore_snapshot_from_s3,
    build_index_metadata,
)


BUCKET = "ponni-dev"
SNAPSHOT_PREFIX = "qdrant_snapshots/"


class TestComputeEmbeddingFingerprint:
    def test_deterministic(self):
        """Same config always produces the same fingerprint."""
        fp1 = compute_embedding_fingerprint()
        fp2 = compute_embedding_fingerprint()
        assert fp1 == fp2

    def test_changes_when_config_changes(self):
        """Fingerprint changes when embedding model/dim/chunk_size changes."""
        fp_original = compute_embedding_fingerprint()

        with patch("db.snapshot_manager.EMBEDDING_MODEL", "different-model"):
            fp_model = compute_embedding_fingerprint()
        assert fp_original != fp_model

        with patch("db.snapshot_manager.EMBEDDING_DIM", 768):
            fp_dim = compute_embedding_fingerprint()
        assert fp_original != fp_dim

        with patch("db.snapshot_manager.CHUNK_SIZE", 1000):
            fp_chunk = compute_embedding_fingerprint()
        assert fp_original != fp_chunk

    def test_is_hex_sha256(self):
        """Fingerprint is a valid hex SHA256 string."""
        fp = compute_embedding_fingerprint()
        assert len(fp) == 64
        int(fp, 16)  # Should not raise


class TestComputeSourceDataHash:
    @mock_aws
    def test_hash_changes_when_files_change(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        s3.put_object(Bucket=BUCKET, Key="output_json/vol_1/doc.json", Body=b"data1")
        hash1 = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")

        s3.put_object(Bucket=BUCKET, Key="output_json/vol_1/doc2.json", Body=b"data2")
        hash2 = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")

        assert hash1 != hash2

    @mock_aws
    def test_hash_stable_when_unchanged(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        s3.put_object(Bucket=BUCKET, Key="output_json/vol_1/doc.json", Body=b"data1")
        hash1 = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")
        hash2 = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")

        assert hash1 == hash2

    @mock_aws
    def test_empty_bucket(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        h = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")
        assert isinstance(h, str)
        assert len(h) == 64

    @mock_aws
    def test_ignores_non_matching_suffix(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        s3.put_object(Bucket=BUCKET, Key="output_json/file.txt", Body=b"data")
        s3.put_object(Bucket=BUCKET, Key="output_json/doc.json", Body=b"data")

        hash_with_txt = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")

        # Only .json should be considered
        s3.delete_object(Bucket=BUCKET, Key="output_json/file.txt")
        hash_without_txt = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")

        assert hash_with_txt == hash_without_txt


class TestLoadSaveIndexMetadata:
    @mock_aws
    def test_load_returns_none_when_missing(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        result = load_index_metadata(s3, BUCKET)
        assert result is None

    @mock_aws
    def test_save_and_load_roundtrip(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        metadata = {
            "embedding_fingerprint": "abc123",
            "source_data_hash": "def456",
            "snapshot_s3_key": "qdrant_snapshots/test.snapshot",
            "points_count": 100,
        }

        save_index_metadata(s3, BUCKET, metadata)
        loaded = load_index_metadata(s3, BUCKET)

        assert loaded is not None
        assert loaded["embedding_fingerprint"] == "abc123"
        assert loaded["source_data_hash"] == "def456"
        assert loaded["points_count"] == 100


class TestNeedsReindex:
    @mock_aws
    def test_returns_true_when_no_metadata(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        result, reason = needs_reindex(s3, BUCKET, "output_json/", ".json")
        assert result is True
        assert "no existing index metadata" in reason

    @mock_aws
    def test_returns_true_when_fingerprint_differs(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        metadata = {
            "embedding_fingerprint": "old_fingerprint",
            "source_data_hash": "some_hash",
        }
        save_index_metadata(s3, BUCKET, metadata)

        result, reason = needs_reindex(s3, BUCKET, "output_json/", ".json")
        assert result is True
        assert "embedding config changed" in reason

    @mock_aws
    def test_returns_true_when_source_hash_differs(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        # Save metadata with current fingerprint but wrong source hash
        current_fp = compute_embedding_fingerprint()
        metadata = {
            "embedding_fingerprint": current_fp,
            "source_data_hash": "old_source_hash",
        }
        save_index_metadata(s3, BUCKET, metadata)

        # Add a source file so the hash is different from "old_source_hash"
        s3.put_object(Bucket=BUCKET, Key="output_json/doc.json", Body=b"data")

        result, reason = needs_reindex(s3, BUCKET, "output_json/", ".json")
        assert result is True
        assert "source data" in reason

    @mock_aws
    def test_returns_false_when_everything_matches(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        # Add a source file
        s3.put_object(Bucket=BUCKET, Key="output_json/doc.json", Body=b"data")

        current_fp = compute_embedding_fingerprint()
        current_hash = compute_source_data_hash(s3, BUCKET, "output_json/", ".json")

        metadata = {
            "embedding_fingerprint": current_fp,
            "source_data_hash": current_hash,
        }
        save_index_metadata(s3, BUCKET, metadata)

        result, reason = needs_reindex(s3, BUCKET, "output_json/", ".json")
        assert result is False
        assert "up to date" in reason


class TestSaveSnapshotToS3:
    @mock_aws
    def test_save_snapshot(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        mock_qdrant = Mock()
        mock_snapshot_info = Mock()
        mock_snapshot_info.name = "test-collection-2026-02-27.snapshot"
        mock_qdrant.create_snapshot.return_value = mock_snapshot_info
        mock_qdrant.get_snapshot.return_value = b"fake_snapshot_data"

        key = save_snapshot_to_s3(mock_qdrant, s3, BUCKET, "test-collection")

        assert key == f"{SNAPSHOT_PREFIX}test-collection-2026-02-27.snapshot"
        mock_qdrant.create_snapshot.assert_called_once_with(collection_name="test-collection")

        # Verify the snapshot was uploaded to S3
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        assert obj["Body"].read() == b"fake_snapshot_data"

    @mock_aws
    def test_save_snapshot_streaming(self):
        """Test that streaming snapshot data (iterator) is handled."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        mock_qdrant = Mock()
        mock_snapshot_info = Mock()
        mock_snapshot_info.name = "streaming.snapshot"
        mock_qdrant.create_snapshot.return_value = mock_snapshot_info
        # Return an iterator instead of bytes
        mock_qdrant.get_snapshot.return_value = iter([b"chunk1", b"chunk2"])

        key = save_snapshot_to_s3(mock_qdrant, s3, BUCKET, "test-collection")

        obj = s3.get_object(Bucket=BUCKET, Key=key)
        assert obj["Body"].read() == b"chunk1chunk2"


class TestRestoreSnapshotFromS3:
    @mock_aws
    def test_restore_no_metadata(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        mock_qdrant = Mock()
        result = restore_snapshot_from_s3(mock_qdrant, s3, BUCKET, "test-collection")
        assert result is False

    @mock_aws
    def test_restore_no_snapshot_key_in_metadata(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        save_index_metadata(s3, BUCKET, {"embedding_fingerprint": "abc"})

        mock_qdrant = Mock()
        result = restore_snapshot_from_s3(mock_qdrant, s3, BUCKET, "test-collection")
        assert result is False

    @mock_aws
    def test_restore_snapshot_not_in_s3(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        save_index_metadata(s3, BUCKET, {
            "snapshot_s3_key": "qdrant_snapshots/missing.snapshot"
        })

        mock_qdrant = Mock()
        result = restore_snapshot_from_s3(mock_qdrant, s3, BUCKET, "test-collection")
        assert result is False

    @mock_aws
    def test_restore_success(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        snapshot_key = "qdrant_snapshots/test.snapshot"
        s3.put_object(Bucket=BUCKET, Key=snapshot_key, Body=b"snapshot_data")
        save_index_metadata(s3, BUCKET, {"snapshot_s3_key": snapshot_key})

        mock_qdrant = Mock()
        result = restore_snapshot_from_s3(mock_qdrant, s3, BUCKET, "test-collection")

        assert result is True
        mock_qdrant.recover_snapshot.assert_called_once()

    @mock_aws
    def test_restore_qdrant_failure(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        snapshot_key = "qdrant_snapshots/test.snapshot"
        s3.put_object(Bucket=BUCKET, Key=snapshot_key, Body=b"snapshot_data")
        save_index_metadata(s3, BUCKET, {"snapshot_s3_key": snapshot_key})

        mock_qdrant = Mock()
        mock_qdrant.recover_snapshot.side_effect = Exception("Qdrant error")

        result = restore_snapshot_from_s3(mock_qdrant, s3, BUCKET, "test-collection")
        assert result is False


class TestBuildIndexMetadata:
    def test_builds_correct_metadata(self):
        metadata = build_index_metadata(
            snapshot_s3_key="qdrant_snapshots/test.snapshot",
            points_count=5432,
            source_data_hash="abc123",
        )

        assert metadata["snapshot_s3_key"] == "qdrant_snapshots/test.snapshot"
        assert metadata["points_count"] == 5432
        assert metadata["source_data_hash"] == "abc123"
        assert metadata["embedding_fingerprint"] == compute_embedding_fingerprint()
        assert "indexed_at" in metadata
        assert metadata["embedding_model"] == "intfloat/multilingual-e5-large"
        assert metadata["embedding_dim"] == 1024
        assert metadata["chunk_size"] == 500
        assert metadata["collection_name"] == "qdrant_indexer"
