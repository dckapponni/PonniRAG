"""Endpoint tests for the gated dataset access API."""

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "db"))


@pytest.fixture()
def client():
    """Provide a FastAPI TestClient with lifespan deps mocked."""
    from fastapi.testclient import TestClient

    from db.api import app

    _csv_mock = MagicMock()
    _csv_mock.exists.return_value = False
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("db.api.check_qdrant_health", return_value={"healthy": True})
        )
        stack.enter_context(
            patch("db.api.check_gemini_health", return_value={"healthy": True})
        )
        stack.enter_context(patch("db.api.CSV_PATH", _csv_mock))
        stack.enter_context(patch("db.api._author_system_cache", {}))
        stack.enter_context(patch("db.api._author_system_lock", MagicMock()))
        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Reset rate-limit state so tests do not interfere with one another."""
    import dataset_access

    dataset_access.reset_rate_limits()
    yield
    dataset_access.reset_rate_limits()


VALID_BODY = {
    "name": "A. Researcher",
    "email": "reader@example.org",
    "affiliation": "Example University",
    "intended_use": "Corpus study of mid-century Tamil periodical prose.",
    "license_accepted": True,
    "dataset_id": "corpus",
}


class TestDatasetInfo:
    """GET /api/dataset/info."""

    def test_lists_open_and_gated_tiers(self, client):
        """Lists open and gated tiers."""
        resp = client.get("/api/dataset/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["corpus_version"].startswith("ponni-corpus-")
        assert data["contact_email"].endswith("ponniarchive.com")
        assert len(data["open_files"]) >= 3
        assert len(data["gated_files"]) == 1

    def test_open_files_carry_direct_urls_and_gated_do_not(self, client):
        """Open files carry direct urls and gated do not."""
        data = client.get("/api/dataset/info").json()
        assert all(f["download_url"] for f in data["open_files"])
        assert all(f["gated"] is False for f in data["open_files"])
        assert all(f["download_url"] is None for f in data["gated_files"])
        assert all(f["gated"] is True for f in data["gated_files"])

    def test_open_tier_is_cc_by(self, client):
        """Open tier is cc by."""
        data = client.get("/api/dataset/info").json()
        assert all(f["license"] == "CC BY 4.0" for f in data["open_files"])


class TestOpenDownloads:
    """GET /api/dataset/open/{file_id}."""

    def test_metadata_csv_downloads_as_attachment(self, client):
        """Metadata csv downloads as attachment."""
        resp = client.get("/api/dataset/open/metadata")
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert "ponni-article-metadata.csv" in resp.headers["content-disposition"]

    def test_licence_document_is_downloadable_without_request(self, client):
        """Licence document is downloadable without request."""
        resp = client.get("/api/dataset/open/license")
        assert resp.status_code == 200
        assert b"Data Use Agreement" in resp.content

    def test_unknown_file_is_404(self, client):
        """Unknown file is 404."""
        assert client.get("/api/dataset/open/nope").status_code == 404


class TestRequestEndpoint:
    """POST /api/dataset/request."""

    def test_valid_request_is_accepted_and_link_emailed(self, client):
        """Valid request is accepted and link emailed."""
        with patch("db.api.store_request", return_value="key.json") as store, patch(
            "db.api.notify_and_deliver", return_value=(True, True)
        ) as notify:
            resp = client.post("/api/dataset/request", json=VALID_BODY)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["email_sent"] is True
        assert len(data["request_id"]) == 32
        store.assert_called_once()

        # The emailed URL must carry a signed token, not a bare S3 path.
        emailed_url = notify.call_args[0][1]
        assert "/api/dataset/download?token=" in emailed_url
        assert "ponni-dev" not in emailed_url

    def test_request_without_licence_acceptance_is_rejected(self, client):
        """Request without licence acceptance is rejected."""
        body = dict(VALID_BODY, license_accepted=False)
        with patch("db.api.store_request") as store:
            resp = client.post("/api/dataset/request", json=body)
        assert resp.status_code == 400
        assert "use agreement" in resp.json()["detail"]
        store.assert_not_called()

    def test_invalid_email_is_rejected(self, client):
        """Invalid email is rejected."""
        body = dict(VALID_BODY, email="not-an-email")
        resp = client.post("/api/dataset/request", json=body)
        assert resp.status_code == 400

    def test_thin_intended_use_is_rejected(self, client):
        """Thin intended use is rejected."""
        body = dict(VALID_BODY, intended_use="research")
        resp = client.post("/api/dataset/request", json=body)
        assert resp.status_code == 400

    def test_unknown_dataset_is_404(self, client):
        """Unknown dataset is 404."""
        body = dict(VALID_BODY, dataset_id="nope")
        resp = client.post("/api/dataset/request", json=body)
        assert resp.status_code == 404

    def test_repeated_requests_are_rate_limited(self, client):
        """Repeated requests are rate limited."""
        with patch("db.api.store_request", return_value="key.json"), patch(
            "db.api.notify_and_deliver", return_value=(True, True)
        ):
            for _ in range(3):
                assert (
                    client.post("/api/dataset/request", json=VALID_BODY).status_code
                    == 200
                )
            resp = client.post("/api/dataset/request", json=VALID_BODY)
        assert resp.status_code == 429

    def test_undelivered_email_still_succeeds_with_fallback_message(self, client):
        """Undelivered email still succeeds with fallback message."""
        with patch("db.api.store_request", return_value="key.json"), patch(
            "db.api.notify_and_deliver", return_value=(False, False)
        ):
            resp = client.post("/api/dataset/request", json=VALID_BODY)
        data = resp.json()
        assert resp.status_code == 200
        assert data["success"] is True
        assert data["email_sent"] is False
        assert data["contact_email"]


class TestGatedDownload:
    """GET /api/dataset/download."""

    def _token(self, **kwargs):
        from dataset_access import issue_token

        params = {
            "email": "reader@example.org",
            "dataset_id": "corpus",
            "request_id": "req123",
        }
        params.update(kwargs)
        return issue_token(**params)

    def test_valid_token_streams_bundle_as_attachment(self, client):
        """Valid token streams bundle as attachment."""
        body = MagicMock()
        body.iter_chunks.return_value = [b"zipdata"]
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": body, "ContentLength": 7}

        with patch("db.api._s3_client", s3):
            resp = client.get(f"/api/dataset/download?token={self._token()}")

        assert resp.status_code == 200
        assert resp.content == b"zipdata"
        assert "attachment" in resp.headers["content-disposition"]
        assert resp.headers["cache-control"] == "no-store"

    def test_missing_token_is_422(self, client):
        """Missing token is 422."""
        assert client.get("/api/dataset/download").status_code == 422

    def test_forged_token_is_403(self, client):
        """Forged token is 403."""
        s3 = MagicMock()
        with patch("db.api._s3_client", s3):
            resp = client.get("/api/dataset/download?token=abc.def")
        assert resp.status_code == 403
        s3.get_object.assert_not_called()

    def test_expired_token_is_410(self, client):
        """Expired token is 410."""
        token = self._token(ttl_seconds=-1)
        s3 = MagicMock()
        with patch("db.api._s3_client", s3):
            resp = client.get(f"/api/dataset/download?token={token}")
        assert resp.status_code == 410
        s3.get_object.assert_not_called()

    def test_missing_bundle_reports_404_with_contact(self, client):
        """Missing bundle reports 404 with contact."""
        from botocore.exceptions import ClientError

        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        with patch("db.api._s3_client", s3):
            resp = client.get(f"/api/dataset/download?token={self._token()}")
        assert resp.status_code == 404
        assert "ponniarchive.com" in resp.json()["detail"]
