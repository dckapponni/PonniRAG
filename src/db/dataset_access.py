"""Gated access to the Ponni Archive corpus release.

The corpus is released in two tiers (see ``DATA_LICENSE.md``):

- **Open tier** — article metadata, the category taxonomy, and the evaluation
  resources. Served directly, no request step, CC BY 4.0.
- **Gated tier** — the full proofread OCR corpus. A requester submits a short
  form, the request is recorded, ``contact@ponniarchive.com`` is notified, and a
  time-limited signed download link is emailed back automatically.

This module holds everything except the HTTP surface itself: the dataset
catalogue, HMAC link signing, request persistence, rate limiting, and email
delivery. The endpoints live in :mod:`api`.

Environment variables:
    ``DATASET_TOKEN_SECRET``   Secret for signing download links. **Set this in
        production** — when unset an ephemeral per-process secret is generated,
        so links stop working when the API restarts.
    ``DATASET_LINK_TTL_HOURS``  Download-link lifetime (default ``168``, 7 days).
    ``DATASET_CONTACT_EMAIL``   Notification recipient (default
        ``contact@ponniarchive.com``).
    ``DATASET_PUBLIC_BASE_URL``  Base URL used to build emailed links (default
        ``https://ponniarchive.com``).
    ``DATASET_REQUESTS_PREFIX``  S3 prefix for stored requests (default
        ``dataset_requests/``).
    ``DATASET_BUNDLE_KEY``      S3 key of the corpus bundle (default
        ``Proof_Read_Bundles/ponni-corpus-v1.zip``).
    ``DATASET_EMAIL_BACKEND``   ``ses`` | ``smtp`` | ``log`` (default ``log``,
        which records the message instead of sending it).
    ``SMTP_HOST`` / ``SMTP_PORT`` / ``SMTP_USER`` / ``SMTP_PASSWORD`` /
        ``SMTP_STARTTLS`` / ``SMTP_FROM``  SMTP backend settings.
    ``SES_REGION`` / ``SES_FROM``  SES backend settings.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import smtplib
import threading
import time
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CORPUS_VERSION = "ponni-corpus-v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]

CONTACT_EMAIL = os.environ.get("DATASET_CONTACT_EMAIL", "contact@ponniarchive.com")
PUBLIC_BASE_URL = os.environ.get(
    "DATASET_PUBLIC_BASE_URL", "https://ponniarchive.com"
).rstrip("/")
REQUESTS_PREFIX = os.environ.get("DATASET_REQUESTS_PREFIX", "dataset_requests/")
BUNDLE_KEY = os.environ.get(
    "DATASET_BUNDLE_KEY", "Proof_Read_Bundles/ponni-corpus-v1.zip"
)
LINK_TTL_SECONDS = int(os.environ.get("DATASET_LINK_TTL_HOURS", "168")) * 3600
EMAIL_BACKEND = os.environ.get("DATASET_EMAIL_BACKEND", "log").lower()

_secret_env = os.environ.get("DATASET_TOKEN_SECRET")
if _secret_env:
    _TOKEN_SECRET = _secret_env.encode("utf-8")
else:
    _TOKEN_SECRET = secrets.token_bytes(32)
    logger.warning(
        "DATASET_TOKEN_SECRET is not set; using an ephemeral secret. "
        "Download links will be invalidated on every API restart."
    )

# Requests allowed per email address and per client IP within the window.
_RATE_LIMIT_WINDOW_SECONDS = 3600
_RATE_LIMIT_MAX_PER_EMAIL = 3
_RATE_LIMIT_MAX_PER_IP = 10

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


# --------------------------------------------------------------------------
# Dataset catalogue
# --------------------------------------------------------------------------

#: Gated datasets, keyed by dataset id. ``s3_key`` is resolved at request time
#: so that the bundle can be re-pointed by environment variable without a code
#: change.
GATED_DATASETS: Dict[str, Dict[str, Any]] = {
    "corpus": {
        "id": "corpus",
        "title_en": "Full proofread Ponni corpus",
        "title_ta": "பொன்னி முழு திருத்தப்பட்ட உரைத்தொகுப்பு",
        "description_en": (
            "OCR-extracted and manually proofread Tamil text of all digitised "
            "Ponni issues, plus the derived per-article JSON."
        ),
        "description_ta": (
            "அனைத்து பொன்னி இதழ்களின் திருத்தப்பட்ட தமிழ் உரை மற்றும் "
            "கட்டுரை வாரியான JSON."
        ),
        "license": "Ponni Archive Data Use Agreement",
        "license_url": "/dataset/license",
        "version": CORPUS_VERSION,
        "filename": f"{CORPUS_VERSION}.zip",
        "media_type": "application/zip",
    }
}

#: Open-tier files served straight from the repository, no request needed.
OPEN_DATASETS: Dict[str, Dict[str, Any]] = {
    "metadata": {
        "id": "metadata",
        "title_en": "Article metadata (CSV)",
        "title_ta": "கட்டுரை விவரத்தரவு (CSV)",
        "description_en": (
            "Year, volume, issue, title and author for all catalogued articles."
        ),
        "description_ta": (
            "அனைத்து கட்டுரைகளுக்கும் ஆண்டு, மலர், இதழ், தலைப்பு, ஆசிரியர்."
        ),
        "path": _REPO_ROOT / "src" / "data" / "summary.csv",
        "filename": "ponni-article-metadata.csv",
        "media_type": "text/csv; charset=utf-8",
        "license": "CC BY 4.0",
    },
    "evaluation": {
        "id": "evaluation",
        "title_en": "Evaluation set (50 QA pairs)",
        "title_ta": "மதிப்பீட்டுத் தொகுப்பு (50 கேள்வி-பதில்)",
        "description_en": (
            "Tamil question-answer pairs with expert reference answers, used "
            "for the answer-quality evaluation."
        ),
        "description_ta": ("நிபுணர் பதில்களுடன் கூடிய தமிழ் கேள்வி-பதில் இணைகள்."),
        "path": _REPO_ROOT / "src" / "evaluation" / "sample_dataset.json",
        "filename": "ponni-evaluation-set.json",
        "media_type": "application/json",
        "license": "CC BY 4.0",
    },
    "ground_truth": {
        "id": "ground_truth",
        "title_en": "Ground-truth judgments (CSV)",
        "title_ta": "தரவரிசைத் தீர்ப்புகள் (CSV)",
        "description_en": (
            "Graded relevance judgments used for the retrieval ablation."
        ),
        "description_ta": "தேடல் மதிப்பீட்டிற்குப் பயன்படுத்தப்பட்ட தீர்ப்புகள்.",
        "path": _REPO_ROOT / "src" / "evaluation" / "ground_truth_data.csv",
        "filename": "ponni-relevance-judgments.csv",
        "media_type": "text/csv; charset=utf-8",
        "license": "CC BY 4.0",
    },
    "datasheet": {
        "id": "datasheet",
        "title_en": "Datasheet",
        "title_ta": "தரவுத்தாள்",
        "description_en": (
            "Corpus composition, coverage, collection procedure, known error "
            "modes, and intended uses."
        ),
        "description_ta": "தொகுப்பின் அமைப்பு, சேகரிப்பு முறை, அறியப்பட்ட பிழைகள்.",
        "path": _REPO_ROOT / "DATASHEET.md",
        "filename": "DATASHEET.md",
        "media_type": "text/markdown; charset=utf-8",
        "license": "CC BY 4.0",
    },
    "license": {
        "id": "license",
        "title_en": "Data licence and use agreement",
        "title_ta": "தரவு உரிமம் மற்றும் பயன்பாட்டு ஒப்பந்தம்",
        "description_en": "Full licence terms for both tiers of the release.",
        "description_ta": "இரு நிலைகளுக்குமான முழு உரிம விதிமுறைகள்.",
        "path": _REPO_ROOT / "DATA_LICENSE.md",
        "filename": "DATA_LICENSE.md",
        "media_type": "text/markdown; charset=utf-8",
        "license": "CC BY 4.0",
    },
}


class DatasetAccessError(Exception):
    """Raised for invalid requests, bad tokens, and rate-limit rejections.

    Attributes:
        status_code: HTTP status the API layer should return.
    """

    def __init__(self, message: str, status_code: int = 400):
        """Store the message and the HTTP status it maps to.

        Args:
            message: Human-readable explanation, safe to show a requester.
            status_code: HTTP status for the API layer to return.
        """
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------
# Signed download links
# --------------------------------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    """Return unpadded base64url text for ``raw``."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    """Decode unpadded base64url ``text`` back to bytes."""
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def issue_token(
    email: str,
    dataset_id: str,
    request_id: str,
    ttl_seconds: Optional[int] = None,
) -> str:
    """Mint a signed, expiring download token.

    Args:
        email: Address the link was issued to.
        dataset_id: Key into :data:`GATED_DATASETS`.
        request_id: Identifier of the stored request record.
        ttl_seconds: Lifetime override; defaults to ``DATASET_LINK_TTL_HOURS``.

    Returns:
        Token string of the form ``<payload>.<signature>``, safe for use in a
        URL query parameter.
    """
    expires_at = int(time.time()) + (
        ttl_seconds if ttl_seconds is not None else LINK_TTL_SECONDS
    )
    payload = {"e": email, "d": dataset_id, "r": request_id, "x": expires_at}
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        _TOKEN_SECRET, payload_b64.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{payload_b64}.{_b64url_encode(signature)}"


def verify_token(token: str) -> Dict[str, Any]:
    """Validate a download token and return its payload.

    Args:
        token: Token previously produced by :func:`issue_token`.

    Returns:
        Decoded payload dict with keys ``e`` (email), ``d`` (dataset id),
        ``r`` (request id) and ``x`` (expiry, epoch seconds).

    Raises:
        DatasetAccessError: 403 if the token is malformed or the signature does
            not verify; 410 if the link has expired.
    """
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        expected = hmac.new(
            _TOKEN_SECRET, payload_b64.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
            raise DatasetAccessError("Invalid download link", status_code=403)
        payload = json.loads(_b64url_decode(payload_b64))
    except DatasetAccessError:
        raise
    except Exception:
        raise DatasetAccessError("Invalid download link", status_code=403)

    if not isinstance(payload, dict) or "x" not in payload or "d" not in payload:
        raise DatasetAccessError("Invalid download link", status_code=403)
    if int(payload["x"]) < int(time.time()):
        raise DatasetAccessError(
            "This download link has expired. Please request a new one.",
            status_code=410,
        )
    if payload["d"] not in GATED_DATASETS:
        raise DatasetAccessError("Unknown dataset", status_code=404)
    return payload


def build_download_url(token: str) -> str:
    """Return the absolute download URL for ``token``."""
    return f"{PUBLIC_BASE_URL}/api/dataset/download?token={token}"


# --------------------------------------------------------------------------
# Validation and rate limiting
# --------------------------------------------------------------------------


def validate_request(
    name: str,
    email: str,
    affiliation: str,
    intended_use: str,
    license_accepted: bool,
    dataset_id: str,
) -> None:
    """Check a submitted request, raising on the first problem found.

    Args:
        name: Requester's name.
        email: Requester's email address.
        affiliation: Institution or organisation.
        intended_use: Free-text description of intended use.
        license_accepted: Whether the use agreement was accepted.
        dataset_id: Requested dataset key.

    Raises:
        DatasetAccessError: 400 for any validation failure, 404 for an unknown
            dataset id.
    """
    if dataset_id not in GATED_DATASETS:
        raise DatasetAccessError("Unknown dataset", status_code=404)
    if not license_accepted:
        raise DatasetAccessError(
            "The data use agreement must be accepted before the corpus can be "
            "released."
        )
    if not name or not name.strip():
        raise DatasetAccessError("Name is required")
    if not _EMAIL_RE.match((email or "").strip()):
        raise DatasetAccessError("A valid email address is required")
    if not affiliation or not affiliation.strip():
        raise DatasetAccessError("Affiliation is required")
    if not intended_use or len(intended_use.strip()) < 20:
        raise DatasetAccessError(
            "Please describe your intended use in at least 20 characters"
        )


_rate_lock = threading.Lock()
_rate_hits: Dict[str, list] = {}


def check_rate_limit(email: str, client_ip: Optional[str]) -> None:
    """Enforce per-email and per-IP submission limits.

    Args:
        email: Normalised requester email.
        client_ip: Client address, or ``None`` when unavailable.

    Raises:
        DatasetAccessError: 429 when either limit is exceeded.
    """
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
    checks = [(f"email:{email.lower()}", _RATE_LIMIT_MAX_PER_EMAIL)]
    if client_ip:
        checks.append((f"ip:{client_ip}", _RATE_LIMIT_MAX_PER_IP))

    with _rate_lock:
        for key, limit in checks:
            hits = [t for t in _rate_hits.get(key, []) if t > cutoff]
            if len(hits) >= limit:
                raise DatasetAccessError(
                    "Too many requests from this address. Please try again "
                    "later, or email " + CONTACT_EMAIL,
                    status_code=429,
                )
            _rate_hits[key] = hits
        for key, _ in checks:
            _rate_hits[key].append(now)


def reset_rate_limits() -> None:
    """Clear all rate-limit state. Intended for tests."""
    with _rate_lock:
        _rate_hits.clear()


# --------------------------------------------------------------------------
# Request persistence
# --------------------------------------------------------------------------


def store_request(record: Dict[str, Any], s3_client, bucket: str) -> Optional[str]:
    """Persist a request record to S3 as JSON.

    Failures are logged and swallowed: losing the audit copy must not stop a
    legitimate requester from receiving their link.

    Args:
        record: Request record to serialise.
        s3_client: Configured boto3 S3 client.
        bucket: Destination bucket name.

    Returns:
        The S3 key written, or ``None`` if persistence failed.
    """
    key = f"{REQUESTS_PREFIX}{record['request_id']}.json"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return key
    except (ClientError, BotoCoreError, NoCredentialsError) as exc:
        logger.warning(f"Could not store dataset request {record['request_id']}: {exc}")
        return None


def new_request_record(
    name: str,
    email: str,
    affiliation: str,
    intended_use: str,
    dataset_id: str,
    client_ip: Optional[str],
) -> Dict[str, Any]:
    """Build the record persisted for an accepted request.

    Args:
        name: Requester's name.
        email: Requester's email address.
        affiliation: Institution or organisation.
        intended_use: Free-text description of intended use.
        dataset_id: Requested dataset key.
        client_ip: Client address, or ``None``.

    Returns:
        A JSON-serialisable record including a generated ``request_id``.
    """
    return {
        "request_id": uuid.uuid4().hex,
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": name.strip(),
        "email": email.strip(),
        "affiliation": affiliation.strip(),
        "intended_use": intended_use.strip(),
        "dataset_id": dataset_id,
        "corpus_version": GATED_DATASETS[dataset_id]["version"],
        "license_accepted": True,
        "license_version": "1.0",
        "client_ip": client_ip,
    }


# --------------------------------------------------------------------------
# Email delivery
# --------------------------------------------------------------------------


def _send(to_address: str, subject: str, body: str) -> bool:
    """Deliver one plain-text message via the configured backend.

    Args:
        to_address: Recipient address.
        subject: Message subject.
        body: Plain-text message body.

    Returns:
        ``True`` if the backend accepted the message, ``False`` otherwise. The
        ``log`` backend returns ``False`` because nothing was actually sent.
    """
    if EMAIL_BACKEND == "ses":
        sender = os.environ.get("SES_FROM", CONTACT_EMAIL)
        try:
            client = boto3.client(
                "ses", region_name=os.environ.get("SES_REGION", "ap-south-1")
            )
            client.send_email(
                Source=sender,
                Destination={"ToAddresses": [to_address]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
                },
            )
            return True
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.error(f"SES send to {to_address} failed: {exc}")
            return False

    if EMAIL_BACKEND == "smtp":
        host = os.environ.get("SMTP_HOST")
        if not host:
            logger.error("DATASET_EMAIL_BACKEND=smtp but SMTP_HOST is unset")
            return False
        sender = os.environ.get("SMTP_FROM", CONTACT_EMAIL)
        message = EmailMessage()
        message["From"] = sender
        message["To"] = to_address
        message["Subject"] = subject
        message.set_content(body)
        try:
            with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as smtp:
                if os.environ.get("SMTP_STARTTLS", "true").lower() == "true":
                    smtp.starttls()
                user = os.environ.get("SMTP_USER")
                if user:
                    smtp.login(user, os.environ.get("SMTP_PASSWORD", ""))
                smtp.send_message(message)
            return True
        except (smtplib.SMTPException, OSError) as exc:
            logger.error(f"SMTP send to {to_address} failed: {exc}")
            return False

    logger.info(
        "Dataset email not sent (backend=log). To: %s | Subject: %s\n%s",
        to_address,
        subject,
        body,
    )
    return False


def notify_and_deliver(record: Dict[str, Any], download_url: str) -> Tuple[bool, bool]:
    """Email the archive team, then email the download link to the requester.

    Args:
        record: Stored request record.
        download_url: Signed, expiring download URL.

    Returns:
        ``(notified_team, delivered_link)`` — whether each message was accepted
        by the backend.
    """
    dataset = GATED_DATASETS[record["dataset_id"]]
    ttl_days = max(1, LINK_TTL_SECONDS // 86400)

    team_body = (
        f"New corpus download request.\n\n"
        f"Request ID : {record['request_id']}\n"
        f"Name       : {record['name']}\n"
        f"Email      : {record['email']}\n"
        f"Affiliation: {record['affiliation']}\n"
        f"Dataset    : {dataset['id']} ({dataset['version']})\n"
        f"Requested  : {record['requested_at']}\n\n"
        f"Intended use:\n{record['intended_use']}\n\n"
        f"The use agreement was accepted and a download link has been sent to "
        f"the requester.\n"
    )
    notified = _send(
        CONTACT_EMAIL,
        f"[Ponni Archive] Corpus request — {record['affiliation']}",
        team_body,
    )

    user_body = (
        f"Dear {record['name']},\n\n"
        f"Thank you for your interest in the Ponni Archive corpus. Your "
        f"download link for {dataset['title_en']} ({dataset['version']}) is "
        f"below. It is valid for {ttl_days} days:\n\n"
        f"{download_url}\n\n"
        f"Your use of this corpus is governed by the Ponni Archive Data Use "
        f"Agreement, which you accepted when submitting this request. In "
        f"short: non-commercial research, scholarship and teaching are "
        f"permitted; redistribution of the corpus is not — please point others "
        f"to {PUBLIC_BASE_URL}/dataset instead. If you publish work using the "
        f"corpus, please cite the resource paper and state the corpus version "
        f"({dataset['version']}).\n\n"
        f"The datasheet accompanying the corpus documents its coverage, known "
        f"error modes, and intended uses; please read it before use.\n\n"
        f"Questions, corrections, or a request for commercial permission: "
        f"reply to {CONTACT_EMAIL}.\n\n"
        f"Ponni Archive\n{PUBLIC_BASE_URL}\n"
    )
    delivered = _send(
        record["email"],
        "[Ponni Archive] Your corpus download link",
        user_body,
    )
    return notified, delivered
