"""Build and upload the distributable Ponni corpus bundle.

Packages the proofread source documents (and, optionally, the extracted plain
text and per-article JSON) from S3 into a single versioned zip, together with a
manifest of per-file SHA-256 checksums, the datasheet, and the licence. The zip
is uploaded to the key the API serves for gated downloads
(``DATASET_BUNDLE_KEY``, default ``Proof_Read_Bundles/ponni-corpus-v1.zip``).

Run it after new proofread documents land in ``Raw_Proof_Read_Content/``::

    cd src && python -m scripts.build_corpus_bundle --dry-run
    cd src && python -m scripts.build_corpus_bundle

``--dry-run`` lists what would be included and exits without downloading,
zipping, or uploading — use it first to confirm the S3 layout.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from config.config import (  # noqa: E402
    BUCKET_NAME,
    EXTRACTED_OUTPUT,
    INPUT_PREFIX,
    OUTPUT_PREFIX,
    REGION_NAME,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("build_corpus_bundle")

REPO_ROOT = ROOT.parent
CORPUS_VERSION = os.environ.get("DATASET_CORPUS_VERSION", "ponni-corpus-v1")
BUNDLE_KEY = os.environ.get(
    "DATASET_BUNDLE_KEY", f"Proof_Read_Bundles/{CORPUS_VERSION}.zip"
)

#: S3 prefix -> directory name inside the zip.
SECTIONS = {
    "proofread": INPUT_PREFIX,
    "extracted_text": EXTRACTED_OUTPUT,
    "articles_json": OUTPUT_PREFIX,
}

#: Repository files copied into the zip root.
DOC_FILES = ["DATASHEET.md", "DATA_LICENSE.md", "LICENSE", "NOTICE"]

#: Source folders are named ``VOL 1 Proof read/``; distribute them as
#: ``Vol1/`` so the bundle matches the ``Magazines/Vol{N}/`` convention used
#: everywhere else.
_VOLUME_DIR_RE = re.compile(r"^VOL\s*(\d+)[^/]*/", re.IGNORECASE)


def normalise_path(relative: str) -> str:
    """Rewrite a volume folder name to the ``Vol{N}/`` convention.

    Args:
        relative: Object key with the section prefix already stripped.

    Returns:
        The path with a leading ``VOL 1 Proof read/``-style directory renamed
        to ``Vol1/``. Paths that do not match are returned unchanged.
    """
    return _VOLUME_DIR_RE.sub(lambda m: f"Vol{int(m.group(1))}/", relative)


def list_objects(s3, prefix: str) -> List[dict]:
    """List every object under an S3 prefix, skipping folder placeholders.

    Args:
        s3: Configured boto3 S3 client.
        prefix: Prefix to enumerate.

    Returns:
        List of S3 object dicts (``Key``, ``Size``, …), directory markers and
        zero-byte objects excluded.
    """
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/") or obj["Size"] == 0:
                continue
            objects.append(obj)
    return objects


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(sections: Dict[str, str], out_path: Path, workdir: Path) -> dict:
    """Download the selected sections, zip them, and write a manifest.

    Args:
        sections: Mapping of zip directory name to S3 prefix.
        out_path: Destination path for the zip.
        workdir: Scratch directory for downloads.

    Returns:
        The manifest dict embedded in the bundle.
    """
    s3 = boto3.client("s3", region_name=REGION_NAME)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "bucket": BUCKET_NAME,
        "sections": {},
        "files": [],
    }

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for section, prefix in sections.items():
            objects = list_objects(s3, prefix)
            logger.info(f"{section}: {len(objects)} objects under {prefix}")
            manifest["sections"][section] = {
                "s3_prefix": prefix,
                "file_count": len(objects),
            }

            for obj in objects:
                relative = obj["Key"][len(prefix) :].lstrip("/")
                local = workdir / section / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET_NAME, obj["Key"], str(local))

                arcname = f"{CORPUS_VERSION}/{section}/{normalise_path(relative)}"
                archive.write(local, arcname)
                manifest["files"].append(
                    {
                        "path": arcname,
                        "bytes": obj["Size"],
                        "sha256": sha256_file(local),
                    }
                )
                local.unlink()

        for doc in DOC_FILES:
            source = REPO_ROOT / doc
            if source.exists():
                archive.write(source, f"{CORPUS_VERSION}/{doc}")
            else:
                logger.warning(f"{doc} not found at {source} — omitted from bundle")

        archive.writestr(
            f"{CORPUS_VERSION}/MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    return manifest


def main() -> int:
    """Parse arguments, build the bundle, and upload it unless suppressed.

    Returns:
        Process exit code: ``0`` on success, ``1`` on an S3 or credential error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be bundled, then exit",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build the zip locally but do not upload it",
    )
    parser.add_argument(
        "--sections",
        default="proofread,extracted_text,articles_json",
        help="Comma-separated subset of: " + ", ".join(SECTIONS),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Local output path (default: ./<version>.zip)",
    )
    args = parser.parse_args()

    unknown = [s for s in args.sections.split(",") if s not in SECTIONS]
    if unknown:
        parser.error(f"Unknown section(s): {', '.join(unknown)}")
    selected = {s: SECTIONS[s] for s in args.sections.split(",")}

    try:
        if args.dry_run:
            s3 = boto3.client("s3", region_name=REGION_NAME)
            total_bytes = 0
            for section, prefix in selected.items():
                objects = list_objects(s3, prefix)
                size = sum(o["Size"] for o in objects)
                total_bytes += size
                logger.info(
                    f"{section:<15} {len(objects):>5} files  "
                    f"{size / 1_048_576:>8.1f} MiB  ({prefix})"
                )
                for obj in objects[:5]:
                    logger.info(f"    e.g. {obj['Key']}")
            logger.info(f"Total: {total_bytes / 1_048_576:.1f} MiB (uncompressed)")
            logger.info(f"Would upload to s3://{BUCKET_NAME}/{BUNDLE_KEY}")
            return 0

        out_path = Path(args.out) if args.out else Path(f"{CORPUS_VERSION}.zip")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build(selected, out_path, Path(tmp))

        size_mib = out_path.stat().st_size / 1_048_576
        logger.info(
            f"Built {out_path} — {len(manifest['files'])} files, {size_mib:.1f} MiB"
        )

        if args.no_upload:
            logger.info("Upload skipped (--no-upload)")
            return 0

        boto3.client("s3", region_name=REGION_NAME).upload_file(
            str(out_path),
            BUCKET_NAME,
            BUNDLE_KEY,
            ExtraArgs={"ContentType": "application/zip"},
        )
        logger.info(f"Uploaded to s3://{BUCKET_NAME}/{BUNDLE_KEY}")
        return 0

    except (ClientError, BotoCoreError, NoCredentialsError) as exc:
        logger.error(f"S3 error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
