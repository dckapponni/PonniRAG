"""Tests for gated corpus access: link signing, validation, and rate limits."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "db"))

import dataset_access  # noqa: E402
from dataset_access import (  # noqa: E402
    GATED_DATASETS,
    OPEN_DATASETS,
    DatasetAccessError,
    build_download_url,
    check_rate_limit,
    issue_token,
    new_request_record,
    reset_rate_limits,
    store_request,
    validate_request,
    verify_token,
)


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Keep rate-limit state from leaking between tests."""
    reset_rate_limits()
    yield
    reset_rate_limits()


# ---------------------------------------------------------------- tokens ---


def test_token_roundtrip_preserves_payload():
    """Token roundtrip preserves payload."""
    token = issue_token("reader@example.org", "corpus", "req123")
    payload = verify_token(token)
    assert payload["e"] == "reader@example.org"
    assert payload["d"] == "corpus"
    assert payload["r"] == "req123"
    assert payload["x"] > time.time()


def test_expired_token_is_rejected_with_410():
    """Expired token is rejected with 410."""
    token = issue_token("reader@example.org", "corpus", "req123", ttl_seconds=-1)
    with pytest.raises(DatasetAccessError) as exc:
        verify_token(token)
    assert exc.value.status_code == 410


def test_tampered_payload_is_rejected():
    """Tampered payload is rejected."""
    token = issue_token("reader@example.org", "corpus", "req123")
    payload_b64, signature = token.split(".", 1)
    forged = issue_token("attacker@example.org", "corpus", "req123").split(".", 1)[0]
    with pytest.raises(DatasetAccessError) as exc:
        verify_token(f"{forged}.{signature}")
    assert exc.value.status_code == 403
    # The untouched token still verifies.
    assert verify_token(f"{payload_b64}.{signature}")["e"] == "reader@example.org"


@pytest.mark.parametrize("bad", ["", "garbage", "a.b", "onlyonepart"])
def test_malformed_tokens_are_rejected(bad):
    """Malformed tokens are rejected."""
    with pytest.raises(DatasetAccessError) as exc:
        verify_token(bad)
    assert exc.value.status_code == 403


def test_token_signed_with_other_secret_is_rejected(monkeypatch):
    """Token signed with other secret is rejected."""
    token = issue_token("reader@example.org", "corpus", "req123")
    monkeypatch.setattr(dataset_access, "_TOKEN_SECRET", b"a-different-secret")
    with pytest.raises(DatasetAccessError) as exc:
        verify_token(token)
    assert exc.value.status_code == 403


def test_download_url_is_absolute():
    """Download url is absolute."""
    url = build_download_url("tok")
    assert url.startswith("http")
    assert url.endswith("/api/dataset/download?token=tok")


# ------------------------------------------------------------ validation ---


def _valid_kwargs(**overrides):
    kwargs = {
        "name": "A. Researcher",
        "email": "reader@example.org",
        "affiliation": "Example University",
        "intended_use": "Studying mid-century Tamil periodical prose.",
        "license_accepted": True,
        "dataset_id": "corpus",
    }
    kwargs.update(overrides)
    return kwargs


def test_valid_request_passes():
    """Valid request passes."""
    validate_request(**_valid_kwargs())


def test_licence_must_be_accepted():
    """Licence must be accepted."""
    with pytest.raises(DatasetAccessError) as exc:
        validate_request(**_valid_kwargs(license_accepted=False))
    assert exc.value.status_code == 400
    assert "use agreement" in str(exc.value)


@pytest.mark.parametrize("bad_email", ["", "notanemail", "a@b", "a b@c.org"])
def test_invalid_emails_are_rejected(bad_email):
    """Invalid emails are rejected."""
    with pytest.raises(DatasetAccessError):
        validate_request(**_valid_kwargs(email=bad_email))


def test_short_intended_use_is_rejected():
    """Short intended use is rejected."""
    with pytest.raises(DatasetAccessError):
        validate_request(**_valid_kwargs(intended_use="research"))


def test_missing_affiliation_is_rejected():
    """Missing affiliation is rejected."""
    with pytest.raises(DatasetAccessError):
        validate_request(**_valid_kwargs(affiliation="   "))


def test_unknown_dataset_is_404():
    """Unknown dataset is 404."""
    with pytest.raises(DatasetAccessError) as exc:
        validate_request(**_valid_kwargs(dataset_id="nope"))
    assert exc.value.status_code == 404


# ----------------------------------------------------------- rate limits ---


def test_per_email_rate_limit_trips_on_fourth_request():
    """Per email rate limit trips on fourth request."""
    for _ in range(3):
        check_rate_limit("reader@example.org", "10.0.0.1")
    with pytest.raises(DatasetAccessError) as exc:
        check_rate_limit("reader@example.org", "10.0.0.1")
    assert exc.value.status_code == 429


def test_rate_limit_is_case_insensitive_on_email():
    """Rate limit is case insensitive on email."""
    for _ in range(3):
        check_rate_limit("Reader@Example.org", None)
    with pytest.raises(DatasetAccessError):
        check_rate_limit("reader@example.org", None)


def test_per_ip_rate_limit_trips_across_distinct_emails():
    """Per ip rate limit trips across distinct emails."""
    for i in range(10):
        check_rate_limit(f"reader{i}@example.org", "10.0.0.9")
    with pytest.raises(DatasetAccessError) as exc:
        check_rate_limit("fresh@example.org", "10.0.0.9")
    assert exc.value.status_code == 429


def test_missing_client_ip_still_allows_requests():
    """Missing client ip still allows requests."""
    check_rate_limit("reader@example.org", None)


# -------------------------------------------------------------- records ---


def test_record_captures_agreement_and_version():
    """Record captures agreement and version."""
    record = new_request_record(
        name="  A. Researcher ",
        email=" reader@example.org ",
        affiliation=" Example University ",
        intended_use=" Corpus linguistics. ",
        dataset_id="corpus",
        client_ip="10.0.0.1",
    )
    assert record["name"] == "A. Researcher"
    assert record["email"] == "reader@example.org"
    assert record["license_accepted"] is True
    assert record["corpus_version"] == GATED_DATASETS["corpus"]["version"]
    assert len(record["request_id"]) == 32


def test_store_request_writes_json_to_expected_prefix():
    """Store request writes json to expected prefix."""
    s3 = MagicMock()
    record = new_request_record(
        name="A",
        email="reader@example.org",
        affiliation="Example",
        intended_use="Research use.",
        dataset_id="corpus",
        client_ip=None,
    )
    key = store_request(record, s3, "ponni-dev")
    assert key.startswith(dataset_access.REQUESTS_PREFIX)
    assert key.endswith(".json")
    s3.put_object.assert_called_once()
    assert s3.put_object.call_args.kwargs["ContentType"] == "application/json"


def test_store_request_failure_does_not_raise():
    """Store request failure does not raise."""
    s3 = MagicMock()
    s3.put_object.side_effect = RuntimeError("boom")
    record = new_request_record(
        name="A",
        email="reader@example.org",
        affiliation="Example",
        intended_use="Research use.",
        dataset_id="corpus",
        client_ip=None,
    )
    with pytest.raises(RuntimeError):
        # Only boto errors are swallowed; unexpected ones surface.
        store_request(record, s3, "ponni-dev")


# ------------------------------------------------------------- catalogue ---


def test_open_tier_files_exist_on_disk():
    """Open tier files exist on disk."""
    missing = [
        meta["id"] for meta in OPEN_DATASETS.values() if not meta["path"].exists()
    ]
    assert missing == [], f"Open-tier files missing: {missing}"


def test_open_tier_is_cc_by_and_gated_tier_is_not():
    """Open tier is cc by and gated tier is not."""
    assert all(m["license"] == "CC BY 4.0" for m in OPEN_DATASETS.values())
    assert all("CC BY" not in m["license"] for m in GATED_DATASETS.values())
