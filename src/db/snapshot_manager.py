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
    """Compute a SHA-256 fingerprint of the current embedding configuration.

    Concatenates ``EMBEDDING_MODEL``, ``EMBEDDING_DIM``, and ``CHUNK_SIZE``
    with pipe delimiters and hashes the result. Any change to these three
    configuration values produces a different fingerprint, which
    :func:`needs_reindex` uses to detect when a full re-index is required.

    Returns:
        str: Hex-encoded SHA-256 digest of the embedding configuration
        string ``"{EMBEDDING_MODEL}|{EMBEDDING_DIM}|{CHUNK_SIZE}"``.
    """
    payload = f"{EMBEDDING_MODEL}|{EMBEDDING_DIM}|{CHUNK_SIZE}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_source_data_hash(s3_client, bucket: str, prefix: str, suffix: str) -> str:
    """Compute a SHA-256 hash of S3 source file keys and their last-modified timestamps.

    Paginates through all objects under ``prefix`` in ``bucket``, collects
    every key ending with ``suffix`` together with its ISO-formatted
    ``LastModified`` timestamp, sorts the entries lexicographically to ensure
    a deterministic order, and hashes the joined result. Any addition,
    deletion, or modification of a source file produces a different hash.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket (str): S3 bucket name to scan.
        prefix (str): Key prefix used to restrict the listing.
        suffix (str): File suffix that qualifying keys must end with.

    Returns:
        str: Hex-encoded SHA-256 digest of the sorted
        ``"key|LastModified"`` entries joined by newlines.
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
    """Return the S3 object key for the index metadata JSON file.

    Constructs the key by joining ``SNAPSHOT_S3_PREFIX`` with the fixed
    filename ``"index_metadata.json"``.

    Returns:
        str: Full S3 key string for the index metadata file.
    """
    return f"{SNAPSHOT_S3_PREFIX}index_metadata.json"


def load_index_metadata(s3_client, bucket: str) -> dict | None:
    """Load the index metadata JSON document from S3.

    Fetches the object at the key returned by :func:`_metadata_s3_key`,
    decodes it as UTF-8, and parses it as JSON. Returns ``None`` when the
    object does not exist (``NoSuchKey``) or when any other error occurs,
    so callers can treat a ``None`` return as "no prior index found."

    Args:
        s3_client: Boto3 S3 client instance.
        bucket (str): S3 bucket name containing the metadata file.

    Returns:
        dict | None: Parsed metadata dictionary, or ``None`` if the file
        does not exist or cannot be read.
    """
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=_metadata_s3_key())
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"Failed to load index metadata from S3: {e}")
        return None


def save_index_metadata(s3_client, bucket: str, metadata: dict):
    """Serialize and upload the index metadata dictionary to S3.

    Serializes ``metadata`` to indented JSON (with ``default=str`` for
    non-serializable values such as ``datetime`` objects) and writes it
    to the key returned by :func:`_metadata_s3_key` with
    ``ContentType="application/json"``.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket (str): S3 bucket name to write the metadata file into.
        metadata (dict): Index metadata dictionary as produced by
            :func:`build_index_metadata`.

    Returns:
        None
    """
    body = json.dumps(metadata, indent=2, default=str)
    s3_client.put_object(
        Bucket=bucket,
        Key=_metadata_s3_key(),
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Saved index metadata to s3://{bucket}/{_metadata_s3_key()}")


def needs_reindex(s3_client, bucket: str, prefix: str, suffix: str) -> tuple[bool, str]:
    """Determine whether the Qdrant index needs to be rebuilt.

    Compares two values against the stored index metadata:

    1. **Embedding fingerprint** — computed by
       :func:`compute_embedding_fingerprint`. A mismatch means the
       embedding model, dimension, or chunk size has changed, requiring
       a full re-index.
    2. **Source data hash** — computed by :func:`compute_source_data_hash`.
       A mismatch means S3 source files have been added, modified, or
       removed since the last index run.

    If no metadata exists yet, re-indexing is required unconditionally.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket (str): S3 bucket name containing source data and metadata.
        prefix (str): Key prefix for source file listing.
        suffix (str): File suffix for source file filtering.

    Returns:
        Tuple[bool, str]: A two-element tuple where the first element is
        ``True`` if re-indexing is needed and ``False`` otherwise, and the
        second element is a human-readable reason string describing the
        outcome.
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
    """Create a Qdrant collection snapshot and upload it to S3.

    Executes the following steps:

    1. Calls ``qdrant_client.create_snapshot`` to trigger snapshot creation
       on the Qdrant server.
    2. Downloads the snapshot tarball via the Qdrant REST API
       (``GET /collections/{name}/snapshots/{snapshot_name}``) using an
       ``httpx`` streaming request with a 300-second timeout, writing it to
       a temporary file.
    3. Uploads the temporary file to S3 under
       ``{SNAPSHOT_S3_PREFIX}{snapshot_name}``.
    4. Deletes the temporary file unconditionally.

    Args:
        qdrant_client: Connected Qdrant client instance used to create the
            snapshot.
        s3_client: Boto3 S3 client instance used for the upload.
        bucket (str): S3 bucket name to upload the snapshot into.
        collection_name (str): Name of the Qdrant collection to snapshot.

    Returns:
        str: S3 object key of the uploaded snapshot file.

    Raises:
        httpx.HTTPStatusError: If the snapshot download request returns a
            non-2xx status code.
        Exception: Propagates any Qdrant or S3 errors without suppression.
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
    """Download a Qdrant snapshot from S3 and restore it into the collection.

    Executes the following steps:

    1. Loads index metadata via :func:`load_index_metadata` to obtain the
       ``snapshot_s3_key``. Returns ``False`` immediately if metadata is
       missing or contains no snapshot key.
    2. Verifies the snapshot object exists in S3 via ``head_object``.
       Returns ``False`` if not found.
    3. Downloads the snapshot tarball from S3 to a temporary file.
    4. Calls ``qdrant_client.recover_snapshot`` with the local file path to
       restore the collection.
    5. Deletes the temporary file in a ``finally`` block.

    Args:
        qdrant_client: Connected Qdrant client instance used for snapshot
            recovery.
        s3_client: Boto3 S3 client instance used for metadata loading and
            snapshot download.
        bucket (str): S3 bucket name containing the snapshot and metadata.
        collection_name (str): Name of the Qdrant collection to restore
            into.

    Returns:
        bool: ``True`` if the snapshot was restored successfully; ``False``
        if no usable snapshot was found in S3 or if ``recover_snapshot``
        raised an exception.
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
    """Build the index metadata dictionary to be saved after a successful index run.

    Combines the provided runtime values with configuration constants and
    the current UTC timestamp into a single dict suitable for passing to
    :func:`save_index_metadata`.

    Args:
        snapshot_s3_key (str): S3 object key of the snapshot file produced
            by :func:`save_snapshot_to_s3`.
        points_count (int): Total number of points in the Qdrant collection
            after indexing, as reported by ``get_collection``.
        source_data_hash (str): SHA-256 hash of the S3 source files at the
            time of indexing, as produced by :func:`compute_source_data_hash`.

    Returns:
        dict: Metadata dictionary containing the keys
        ``"embedding_fingerprint"``, ``"source_data_hash"``,
        ``"snapshot_s3_key"``, ``"points_count"``, ``"indexed_at"``
        (ISO-formatted UTC datetime), ``"embedding_model"``,
        ``"embedding_dim"``, ``"chunk_size"``, and ``"collection_name"``.
    """
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
