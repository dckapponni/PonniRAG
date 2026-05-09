"""Tests for src/scripts/upload_pdfs_to_s3.py."""

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from scripts.upload_pdfs_to_s3 import (
    _drive_file_id,
    _linearize_pdf,
    _s3_key_for_issue,
    _validate_pdf,
    _verify_s3_object,
    process_issue,
)


def _make_valid_pdf_bytes() -> bytes:
    """Build a minimal one-page PDF via pikepdf (qpdf-compatible)."""
    import io

    import pikepdf

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))  # US-letter
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


class TestDriveFileId:
    """Tests for the Google Drive file-id extraction helper."""

    def test_extracts_id_from_drive_url(self):
        """Pull the id segment out of a `/d/<id>/` Drive URL."""
        url = "https://drive.google.com/file/d/ABC123xyz/view?usp=drive_link"
        assert _drive_file_id(url) == "ABC123xyz"

    def test_returns_none_for_non_drive_url(self):
        """Return None for URLs that don't follow the Drive `/d/` pattern."""
        assert _drive_file_id("https://example.com/file.pdf") is None


class TestS3KeyForIssue:
    """Tests for S3 key construction from registry conventions."""

    def test_builds_expected_key(self):
        """Build the canonical S3 key for a numeric issue."""
        s3_conf = {
            "magazines": "Magazines/",
            "magazine_folder_pattern": "Vol{vol_id}/",
            "magazine_file_pattern": "VOL{vol_id} - {issue_num} - {year}.pdf",
        }
        # gitleaks:allow
        assert _s3_key_for_issue(s3_conf, 1, 6, "1947") == (
            "Magazines/Vol1/VOL1 - 6 - 1947.pdf"  # gitleaks:allow
        )

    def test_pongal_issue(self):
        """PONGAL is a non-numeric issue name and must round-trip."""
        s3_conf = {
            "magazines": "Magazines/",
            "magazine_folder_pattern": "Vol{vol_id}/",
            "magazine_file_pattern": "VOL{vol_id} - {issue_num} - {year}.pdf",
        }
        assert _s3_key_for_issue(s3_conf, 1, "PONGAL", "1948") == (
            "Magazines/Vol1/VOL1 - PONGAL - 1948.pdf"  # gitleaks:allow
        )


class TestLinearizePdf:
    """Tests for the qpdf-backed linearization step."""

    def test_linearized_output_is_valid_and_marked_linearized(self, tmp_path: Path):
        """Linearized output passes validation and pikepdf reports linearized."""
        import pikepdf

        src = tmp_path / "src.pdf"
        src.write_bytes(_make_valid_pdf_bytes())
        dst = tmp_path / "dst.pdf"
        _linearize_pdf(src, dst)
        assert dst.exists()
        _validate_pdf(dst)
        with pikepdf.open(str(dst)) as pdf:
            assert pdf.is_linearized


class TestValidatePdf:
    """Tests for the local PDF structural validation step."""

    def test_valid_pdf_passes(self, tmp_path: Path):
        """A structurally complete PDF passes without raising."""
        pdf = tmp_path / "ok.pdf"
        pdf.write_bytes(_make_valid_pdf_bytes())
        _validate_pdf(pdf)  # no raise

    def test_too_small_raises(self, tmp_path: Path):
        """Files below the minimum size threshold are rejected."""
        pdf = tmp_path / "tiny.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        with pytest.raises(ValueError, match="too small"):
            _validate_pdf(pdf)

    def test_missing_eof_raises(self, tmp_path: Path):
        """Files lacking the `%%EOF` trailer are rejected."""
        pdf = tmp_path / "trunc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 4096)
        with pytest.raises(ValueError, match="%%EOF"):
            _validate_pdf(pdf)


class TestVerifyS3Object:
    """Tests for the post-upload S3 verification step."""

    @mock_aws
    def test_verify_passes_for_good_object(self):
        """An intact object with matching size + `%%EOF` tail validates."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        body = _make_valid_pdf_bytes()
        s3.put_object(Bucket="test-bucket", Key="x.pdf", Body=body)
        _verify_s3_object(s3, "test-bucket", "x.pdf", len(body))

    @mock_aws
    def test_verify_fails_on_missing_eof(self):
        """An object missing the `%%EOF` marker in the tail is rejected."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        body = b"%PDF-1.4\n" + b"\x00" * 4096
        s3.put_object(Bucket="test-bucket", Key="x.pdf", Body=body)
        with pytest.raises(ValueError, match="%%EOF"):
            _verify_s3_object(s3, "test-bucket", "x.pdf", len(body))

    @mock_aws
    def test_verify_fails_on_size_mismatch(self):
        """An object whose ContentLength differs from the expected is rejected."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="x.pdf", Body=b"abcd")
        with pytest.raises(ValueError, match="size mismatch"):
            _verify_s3_object(s3, "test-bucket", "x.pdf", 999)


class TestProcessIssue:
    """End-to-end tests for the per-issue download/upload/verify pipeline."""

    @mock_aws
    def test_full_loop_uploads_and_verifies(self, tmp_path: Path, monkeypatch):
        """Happy path: download mock → validate → upload → S3 verify → manifest."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3_conf = {
            "magazines": "Magazines/",
            "magazine_folder_pattern": "Vol{vol_id}/",
            "magazine_file_pattern": "VOL{vol_id} - {issue_num} - {year}.pdf",
        }
        valid_bytes = _make_valid_pdf_bytes()

        def fake_gdown(*args, **kwargs):
            output = Path(kwargs["output"])
            output.write_bytes(valid_bytes)
            return str(output)

        monkeypatch.setattr("scripts.upload_pdfs_to_s3.gdown.download", fake_gdown)
        manifest_file = tmp_path / "manifest.json"
        monkeypatch.setattr("scripts.upload_pdfs_to_s3.MANIFEST_PATH", manifest_file)

        issue = {
            "num": 6,
            "pdf_url": "https://drive.google.com/file/d/ABC123/view",
        }
        status = process_issue(
            s3=s3,
            bucket="test-bucket",
            s3_conf=s3_conf,
            vol_id=1,
            vol_year="1947",
            issue=issue,
            manifest={},
            force=False,
        )
        assert status == "ok"
        # Verify object actually present and linearized (uploaded body is
        # the linearized output, with a different size than the source).
        import io as _io

        import pikepdf

        body = s3.get_object(
            Bucket="test-bucket",
            Key="Magazines/Vol1/VOL1 - 6 - 1947.pdf",  # gitleaks:allow
        )["Body"].read()
        with pikepdf.open(_io.BytesIO(body)) as pdf:
            assert pdf.is_linearized
        assert manifest_file.exists()

    @mock_aws
    def test_invalid_source_pdf_does_not_upload(self, tmp_path: Path, monkeypatch):
        """A broken Drive download must not upload anything to S3."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3_conf = {
            "magazines": "Magazines/",
            "magazine_folder_pattern": "Vol{vol_id}/",
            "magazine_file_pattern": "VOL{vol_id} - {issue_num} - {year}.pdf",
        }

        def fake_gdown(*args, **kwargs):
            Path(kwargs["output"]).write_bytes(b"%PDF-1.4\n" + b"\x00" * 4096)

        monkeypatch.setattr("scripts.upload_pdfs_to_s3.gdown.download", fake_gdown)
        monkeypatch.setattr(
            "scripts.upload_pdfs_to_s3.MANIFEST_PATH", tmp_path / "m.json"
        )

        status = process_issue(
            s3=s3,
            bucket="test-bucket",
            s3_conf=s3_conf,
            vol_id=1,
            vol_year="1947",
            issue={"num": 6, "pdf_url": "https://drive.google.com/file/d/X/view"},
            manifest={},
            force=False,
        )
        assert status == "invalid_source"
        # Confirm nothing was uploaded
        listing = s3.list_objects_v2(Bucket="test-bucket")
        assert listing.get("KeyCount", 0) == 0

    @mock_aws
    def test_drive_failure_after_retries_returns_download_failed(
        self, tmp_path: Path, monkeypatch
    ):
        """Persistent gdown failures bubble up as `download_failed` status."""
        from gdown.exceptions import FileURLRetrievalError

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3_conf = {
            "magazines": "Magazines/",
            "magazine_folder_pattern": "Vol{vol_id}/",
            "magazine_file_pattern": "VOL{vol_id} - {issue_num} - {year}.pdf",
        }

        def always_fail(*args, **kwargs):
            raise FileURLRetrievalError("permission denied")

        monkeypatch.setattr("scripts.upload_pdfs_to_s3.gdown.download", always_fail)
        # Skip the actual sleep between retries to keep test fast.
        monkeypatch.setattr("scripts.upload_pdfs_to_s3.time.sleep", lambda _s: None)
        monkeypatch.setattr(
            "scripts.upload_pdfs_to_s3.MANIFEST_PATH", tmp_path / "m.json"
        )

        status = process_issue(
            s3=s3,
            bucket="test-bucket",
            s3_conf=s3_conf,
            vol_id=1,
            vol_year="1947",
            issue={"num": 6, "pdf_url": "https://drive.google.com/file/d/X/view"},
            manifest={},
            force=False,
        )
        assert status == "download_failed"
        listing = s3.list_objects_v2(Bucket="test-bucket")
        assert listing.get("KeyCount", 0) == 0

    @mock_aws
    def test_skip_when_already_in_manifest(self, tmp_path: Path, monkeypatch):
        """A key already in the manifest with matching S3 size is skipped."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3_conf = {
            "magazines": "Magazines/",
            "magazine_folder_pattern": "Vol{vol_id}/",
            "magazine_file_pattern": "VOL{vol_id} - {issue_num} - {year}.pdf",
        }
        body = _make_valid_pdf_bytes()
        key = "Magazines/Vol1/VOL1 - 6 - 1947.pdf"  # gitleaks:allow
        s3.put_object(Bucket="test-bucket", Key=key, Body=body)

        # gdown should never be called when skipping
        def fail_gdown(*args, **kwargs):
            raise AssertionError("gdown.download should not run on skip")

        monkeypatch.setattr("scripts.upload_pdfs_to_s3.gdown.download", fail_gdown)
        monkeypatch.setattr(
            "scripts.upload_pdfs_to_s3.MANIFEST_PATH", tmp_path / "m.json"
        )

        manifest = {key: {"sha256": "x", "size": len(body)}}
        status = process_issue(
            s3=s3,
            bucket="test-bucket",
            s3_conf=s3_conf,
            vol_id=1,
            vol_year="1947",
            issue={"num": 6, "pdf_url": "https://drive.google.com/file/d/X/view"},
            manifest=manifest,
            force=False,
        )
        assert status == "skipped"
