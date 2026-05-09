"""Re-upload Ponni PDFs from Google Drive to S3 with verification.

Reads `magazine_registry.json`, downloads each issue's PDF from Drive,
validates the PDF is structurally complete (`%%EOF` + parseable), uploads
to the registry-defined S3 key, and re-verifies via S3 by tail-fetching
the last 1 KiB to confirm `%%EOF` survived the upload.

Idempotent: maintains a manifest at `src/scripts/upload_manifest.json`
mapping S3 key -> {"sha256": ..., "size": ...}. Re-runs skip already-verified
keys whose S3 object size + ETag still match the manifest.

Usage:
    cd src && python -m scripts.upload_pdfs_to_s3
    cd src && python -m scripts.upload_pdfs_to_s3 --volume 1
    cd src && python -m scripts.upload_pdfs_to_s3 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import boto3
import gdown
import pikepdf
from botocore.exceptions import ClientError
from gdown.exceptions import FileURLRetrievalError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import get_magazine_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("upload_pdfs")

MANIFEST_PATH = Path(__file__).resolve().parent / "upload_manifest.json"
EOF_TAIL_BYTES = 2048
DOWNLOAD_RETRIES = 4
DOWNLOAD_BACKOFF_BASE_S = 5.0
INTER_ISSUE_SLEEP_S = 2.0


def _download_with_retry(file_id: str, output: Path) -> None:
    """Download from Drive with exponential backoff on retrieval failures."""
    last_err: Optional[Exception] = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            gdown.download(id=file_id, output=str(output), quiet=True, fuzzy=True)
            if output.exists() and output.stat().st_size > 0:
                return
            raise FileURLRetrievalError("gdown produced empty output")
        except FileURLRetrievalError as e:
            last_err = e
            wait = DOWNLOAD_BACKOFF_BASE_S * (2 ** (attempt - 1))
            wait += random.uniform(0, wait * 0.25)  # jitter
            logger.warning(
                "[retry %d/%d] gdown failed for %s: %s — sleeping %.1fs",
                attempt,
                DOWNLOAD_RETRIES,
                file_id,
                e,
                wait,
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _drive_file_id(pdf_url: str) -> Optional[str]:
    """Extract the Google Drive file id from a `/d/<id>/` URL."""
    if "/d/" not in pdf_url:
        return None
    return pdf_url.split("/d/")[1].split("/")[0]


def _s3_key_for_issue(s3_conf: dict, vol_id: int, issue_num, year: str) -> str:
    """Build the S3 key for an issue using registry conventions."""
    folder = s3_conf["magazine_folder_pattern"].format(vol_id=vol_id)
    filename = s3_conf["magazine_file_pattern"].format(
        vol_id=vol_id, issue_num=issue_num, year=year
    )
    return f"{s3_conf['magazines']}{folder}{filename}"


def _validate_pdf(path: Path) -> None:
    """Raise on structural problems in the downloaded PDF."""
    size = path.stat().st_size
    if size < 256:
        raise ValueError(f"file too small: {size} bytes")
    with path.open("rb") as fh:
        fh.seek(-EOF_TAIL_BYTES, 2) if size > EOF_TAIL_BYTES else fh.seek(0)
        tail = fh.read()
    if b"%%EOF" not in tail:
        raise ValueError("missing %%EOF marker in trailer")
    # Full structural parse — catches malformed xref/trailer.
    PdfReader(str(path), strict=False)


def _linearize_pdf(src: Path, dst: Path) -> None:
    """Rewrite the PDF in linearized ("Fast Web View") form.

    Linearized PDFs let pdf.js render the first page before the entire
    file downloads — major win for time-to-first-paint on large scans.
    """
    with pikepdf.open(str(src)) as pdf:
        pdf.save(str(dst), linearize=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_s3_object(s3, bucket: str, key: str, expected_size: int) -> None:
    """Confirm S3 object matches expected size and contains %%EOF in tail."""
    head = s3.head_object(Bucket=bucket, Key=key)
    if head["ContentLength"] != expected_size:
        raise ValueError(
            f"S3 size mismatch: expected {expected_size}, got {head['ContentLength']}"
        )
    start = max(0, expected_size - EOF_TAIL_BYTES)
    tail = s3.get_object(
        Bucket=bucket, Key=key, Range=f"bytes={start}-{expected_size - 1}"
    )["Body"].read()
    if b"%%EOF" not in tail:
        raise ValueError("S3 object tail missing %%EOF after upload")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _already_uploaded(s3, bucket: str, key: str, manifest: dict) -> bool:
    entry = manifest.get(key)
    if not entry:
        return False
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return head["ContentLength"] == entry.get("size")


def process_issue(
    s3,
    bucket: str,
    s3_conf: dict,
    vol_id: int,
    vol_year: str,
    issue: dict,
    manifest: dict,
    force: bool,
) -> str:
    """Download, validate, upload, verify a single issue. Returns status."""
    issue_num = issue["num"]
    year = issue.get("year", vol_year)
    pdf_url = issue.get("pdf_url")
    if not pdf_url:
        return "no_url"

    file_id = _drive_file_id(pdf_url)
    if not file_id:
        return "bad_url"

    key = _s3_key_for_issue(s3_conf, vol_id, issue_num, year)

    if not force and _already_uploaded(s3, bucket, key, manifest):
        logger.info("[skip] %s already verified", key)
        return "skipped"

    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "issue.pdf"
        logger.info("[download] vol=%s issue=%s -> %s", vol_id, issue_num, key)
        try:
            _download_with_retry(file_id, local)
        except FileURLRetrievalError as e:
            logger.error("[download_failed] %s: %s", key, e)
            return "download_failed"

        try:
            _validate_pdf(local)
        except (ValueError, PdfReadError) as e:
            logger.error("[invalid] %s: %s", key, e)
            return "invalid_source"

        linearized = Path(td) / "issue.linearized.pdf"
        try:
            _linearize_pdf(local, linearized)
            _validate_pdf(linearized)
        except (pikepdf.PdfError, ValueError, PdfReadError) as e:
            logger.error("[linearize_failed] %s: %s", key, e)
            return "linearize_failed"

        upload_path = linearized
        digest = _sha256(upload_path)
        size = upload_path.stat().st_size

        logger.info(
            "[upload] %s (%d bytes, linearized from %d)",
            key,
            size,
            local.stat().st_size,
        )
        s3.upload_file(
            str(upload_path),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/pdf"},
        )

        try:
            _verify_s3_object(s3, bucket, key, size)
        except (ValueError, ClientError) as e:
            logger.error("[verify_failed] %s: %s", key, e)
            return "verify_failed"

    manifest[key] = {"sha256": digest, "size": size}
    _save_manifest(manifest)
    logger.info("[ok] %s", key)
    return "ok"


def main() -> int:
    """Parse CLI args and run the upload+verify loop over the registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, help="Process only this volume id")
    parser.add_argument(
        "--issue",
        help="Process only this issue (requires --volume)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even if manifest marks key as verified",
    )
    args = parser.parse_args()

    cfg = get_magazine_config("ponni")
    s3_conf = cfg["s3"]
    bucket = s3_conf["bucket"]
    s3 = boto3.client("s3", region_name=s3_conf["region"])

    manifest = _load_manifest()

    counts = {
        "ok": 0,
        "skipped": 0,
        "invalid_source": 0,
        "verify_failed": 0,
        "download_failed": 0,
        "linearize_failed": 0,
        "no_url": 0,
        "bad_url": 0,
    }
    failures: list = []

    for vol in cfg["volumes"]:
        if args.volume is not None and vol["id"] != args.volume:
            continue
        for issue in vol["issues"]:
            if args.issue is not None and str(issue["num"]) != str(args.issue):
                continue
            try:
                status = process_issue(
                    s3=s3,
                    bucket=bucket,
                    s3_conf=s3_conf,
                    vol_id=vol["id"],
                    vol_year=vol["year"],
                    issue=issue,
                    manifest=manifest,
                    force=args.force,
                )
            except (
                Exception
            ) as e:  # noqa: BLE001 — keep batch alive on any per-issue failure
                logger.exception(
                    "[unexpected] vol=%s issue=%s: %s", vol["id"], issue["num"], e
                )
                status = "download_failed"
            counts[status] = counts.get(status, 0) + 1
            if status in (
                "download_failed",
                "invalid_source",
                "verify_failed",
                "linearize_failed",
            ):
                failures.append(
                    {
                        "volume": vol["id"],
                        "issue": str(issue["num"]),
                        "status": status,
                        "drive_url": issue.get("pdf_url"),
                    }
                )
            time.sleep(INTER_ISSUE_SLEEP_S)

    logger.info("Done: %s", counts)
    if failures:
        logger.warning("Failures (%d):", len(failures))
        for f in failures:
            logger.warning(
                "  vol=%s issue=%s status=%s url=%s",
                f["volume"],
                f["issue"],
                f["status"],
                f["drive_url"],
            )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
