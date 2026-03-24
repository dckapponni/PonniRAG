"""Snapshot manager for Qdrant index persistence via S3.

Handles embedding fingerprint computation, source data hash
computation, index metadata storage, and snapshot save/restore.
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import httpx

from src.config.config import (
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    SNAPSHOT_S3_PREFIX,
)

logger = logging.getLogger(__name__)


def compute_embedding_fingerprint() -> str:
    """Compute SHA256 hash of embedding configuration.

    If any of these change, the index is invalidated and a full
    re-index is required.
    """
    payload = f"{EMBEDDING_MODEL}|{EMBEDDING_DIM}|{CHUNK_SIZE}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_source_data_hash(s3_client, bucket: str, prefix: str, suffix: str) -> str:
    """Compute hash of S3 source file keys and LastModified timestamps.

    Detects when source data has been added, modified, or removed.
    """
    entries = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                last_modified = obj["LastModified"].isoformat()
                entries.append(f"{key}|{last_modified}")

    entries.sort()
    combined = "\n".join(entries)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _metadata_s3_key() -> str:
    """Return the S3 key for the index metadata JSON file."""
    return f"{SNAPSHOT_S3_PREFIX}index_metadata.json"


def load_index_metadata(s3_client, bucket: str) -> dict | None:
    """Load metadata JSON from S3. Returns None if no metadata exists."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=_metadata_s3_key())
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"Failed to load index metadata from S3: {e}")
        return None


def save_index_metadata(s3_client, bucket: str, metadata: dict):
    """Save metadata JSON to S3."""
    body = json.dumps(metadata, indent=2, default=str)
    s3_client.put_object(
        Bucket=bucket,
        Key=_metadata_s3_key(),
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Saved index metadata to s3://{bucket}/{_metadata_s3_key()}")


def needs_reindex(s3_client, bucket: str, prefix: str, suffix: str) -> tuple[bool, str]:
    """Compare current fingerprint + source hash against stored metadata.

    Returns (needs_reindex: bool, reason: str).
    """
    metadata = load_index_metadata(s3_client, bucket)

    if metadata is None:
        return True, "no existing index metadata found in S3"

    current_fingerprint = compute_embedding_fingerprint()
    stored_fingerprint = metadata.get("embedding_fingerprint")

    if current_fingerprint != stored_fingerprint:
        return True, (
            f"embedding config changed "
            f"(model/dim/chunk_size): "
            f"{stored_fingerprint[:12]}..."
            f" -> {current_fingerprint[:12]}..."
        )

    current_source_hash = compute_source_data_hash(s3_client, bucket, prefix, suffix)
    stored_source_hash = metadata.get("source_data_hash")

    if current_source_hash != stored_source_hash:
        return True, "source data in S3 has changed"

    return False, "index is up to date"


def save_snapshot_to_s3(
    qdrant_client, s3_client, bucket: str, collection_name: str
) -> str:
    """Create Qdrant snapshot, download tarball, upload to S3.

    Returns the S3 key of the uploaded snapshot.
    """
    logger.info(f"Creating snapshot for collection '{collection_name}'...")
    snapshot_info = qdrant_client.create_snapshot(collection_name=collection_name)
    snapshot_name = snapshot_info.name

    logger.info(f"Snapshot created: {snapshot_name}")

    with tempfile.NamedTemporaryFile(suffix=".snapshot", delete=False) as tmp:
        tmp_path = tmp.name

    # Download snapshot via Qdrant REST API
    # (get_snapshot was removed in qdrant-client 1.12+)
    download_url = (
        f"http://{QDRANT_HOST}:{QDRANT_PORT}"
        f"/collections/{collection_name}/snapshots/{snapshot_name}"
    )
    logger.info(f"Downloading snapshot from {download_url}...")
    with httpx.stream("GET", download_url, timeout=300) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)

    s3_key = f"{SNAPSHOT_S3_PREFIX}{snapshot_name}"
    logger.info(f"Uploading snapshot to s3://{bucket}/{s3_key}...")

    s3_client.upload_file(tmp_path, bucket, s3_key)
    logger.info(f"Snapshot uploaded to s3://{bucket}/{s3_key}")

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return s3_key


def restore_snapshot_from_s3(
    qdrant_client, s3_client, bucket: str, collection_name: str
) -> bool:
    """Download snapshot from S3, upload to Qdrant via recover_snapshot.

    Returns True if restored successfully, False if no snapshot available.
    """
    metadata = load_index_metadata(s3_client, bucket)
    if metadata is None:
        logger.info("No index metadata found — cannot restore snapshot")
        return False

    snapshot_s3_key = metadata.get("snapshot_s3_key")
    if not snapshot_s3_key:
        logger.info("No snapshot_s3_key in metadata — cannot restore snapshot")
        return False

    # Check if snapshot exists in S3
    try:
        s3_client.head_object(Bucket=bucket, Key=snapshot_s3_key)
    except Exception:
        logger.warning(f"Snapshot not found at s3://{bucket}/{snapshot_s3_key}")
        return False

    with tempfile.NamedTemporaryFile(suffix=".snapshot", delete=False) as tmp:
        tmp_path = tmp.name

    logger.info(f"Downloading snapshot from s3://{bucket}/{snapshot_s3_key}...")
    s3_client.download_file(bucket, snapshot_s3_key, tmp_path)

    logger.info(f"Restoring snapshot into collection '{collection_name}'...")
    try:
        qdrant_client.recover_snapshot(
            collection_name=collection_name,
            location=tmp_path,
        )
        logger.info("Snapshot restored successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to restore snapshot: {e}")
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_index_metadata(
    snapshot_s3_key: str,
    points_count: int,
    source_data_hash: str,
) -> dict:
    """Build metadata dict for saving after indexing."""
    return {
        "embedding_fingerprint": compute_embedding_fingerprint(),
        "source_data_hash": source_data_hash,
        "snapshot_s3_key": snapshot_s3_key,
        "points_count": points_count,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "collection_name": COLLECTION_NAME,
    }
