"""FastAPI REST API for the Ponni RAG System.

This module provides the HTTP API layer for the Ponni Tamil Literary Archive
Retrieval-Augmented Generation (RAG) system. It exposes endpoints for:

- Semantic search and question answering (hybrid dense + sparse retrieval)
- Streaming AI-generated answers via Server-Sent Events (SSE)
- Browsing the digital library (volumes, issues, PDFs, cover images)
- Author and topic discovery via CSV-backed metadata
- Article tag/category listing with per-tag article counts
- Full article content retrieval from Qdrant vector store
- Health monitoring for the database and LLM backends

Typical usage:
    Run directly with uvicorn::

        uvicorn api:app --host 0.0.0.0 --port 8000 --reload

    Or via the ``__main__`` guard at the bottom of this file.

Dependencies:
    - FastAPI + Uvicorn (HTTP framework)
    - Qdrant (vector store)
    - Google Gemini (LLM backend, via ``llm`` module)
    - AWS S3 via boto3 (image and PDF storage)
    - EnhancedAuthorQuerySystem (CSV-based author/topic queries)
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from cache import _response_cache  # noqa: E402
from csv_queries import EnhancedAuthorQuerySystem  # noqa: E402
from csv_queries import _author_system_cache, _author_system_lock, get_issue_count
from embeddings import CSV_PATH  # noqa: E402
from embeddings import COLLECTION_NAME, check_qdrant_health, get_qdrant_client
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from hybrid_search import ask_question_async, ask_question_stream  # noqa: E402
from llm import check_gemini_health  # noqa: E402
from pydantic import BaseModel, Field
from qdrant_client import models  # noqa: E402

from config.config import get_magazine_config  # noqa: E402
from db.article_tagger import TAXONOMY  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Load magazine registry (single source of truth for volumes/issues/PDFs)
_magazine = get_magazine_config("ponni")
_s3_conf = _magazine["s3"]

# Build PDF_LINKS dict from registry for backward compatibility
PDF_LINKS = {}
for vol in _magazine["volumes"]:
    for issue in vol["issues"]:
        PDF_LINKS[f"vol_{vol['id']}_issue_{issue['num']}"] = issue["pdf_url"]

# S3 client for image proxy
_s3_client = boto3.client(
    "s3",
    region_name=_s3_conf["region"],
)

# Image cache duration (browser Cache-Control header)
_IMAGE_CACHE_SECONDS = 3600  # 1 hour

_MAX_HISTORY_CONTENT = 15000  # Truncate long history entries instead of rejecting


def _volume_cover_s3_key(volume_id: int) -> Optional[str]:
    """Return the S3 object key for a volume's cover image.

    Looks up the volume entry in the magazine registry and derives the
    key using the configured ``covers_prefix`` and the volume's
    ``cover_image`` filename.

    Args:
        volume_id: Integer volume identifier (e.g. 1, 2, …).

    Returns:
        The full S3 key string, or ``None`` if the volume is not found
        or has no ``cover_image`` entry in the registry.
    """
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data or not vol_data.get("cover_image"):
        return None
    return f"{_s3_conf['covers_prefix']}Volumes/{vol_data['cover_image']}"


def _issue_cover_s3_key(volume_id: int, issue_name: str) -> Optional[str]:
    """Return the S3 object key for an issue's cover image.

    Uses the magazine registry to derive the key
    Args:
        volume_id: Integer volume identifier (e.g. 1, 2, …).
        issue_name: String issue identifier (e.g. "1", "2", …).

    Returns:
        The full S3 key string, or ``None`` if the volume or issue is not found
        or has no ``cover_image`` entry in the registry.
    """
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data:
        return None
    iss_data = next(
        (i for i in vol_data["issues"] if str(i["num"]) == str(issue_name)), None
    )
    if not iss_data:
        return None
    year = iss_data.get("year", vol_data["year"])
    folder = _s3_conf["cover_folder_pattern"].format(vol_id=volume_id)
    filename = _s3_conf["cover_file_pattern"].format(
        vol_id=volume_id,
        issue_num=issue_name,
        year=year,
    )
    return f"{_s3_conf['covers_prefix']}{folder}{filename}"


def _issue_pdf_s3_key(volume_id: int, issue_name: str) -> Optional[str]:
    """Return the S3 object key for an issue's PDF.

    Args:
        volume_id: Integer volume identifier (e.g. 1, 2, …).
        issue_name: String issue identifier (e.g. "1", "2", …).

    Returns:
        The full S3 key string, or ``None`` if the volume or issue is not found
        or has no ``pdf_url`` entry in the registry.
    """
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data:
        return None
    iss_data = next(
        (i for i in vol_data["issues"] if str(i["num"]) == str(issue_name)), None
    )
    if not iss_data:
        return None
    year = iss_data.get("year", vol_data["year"])
    folder = _s3_conf["magazine_folder_pattern"].format(vol_id=volume_id)
    filename = _s3_conf["magazine_file_pattern"].format(
        vol_id=volume_id,
        issue_num=issue_name,
        year=year,
    )
    return f"{_s3_conf['magazines']}{folder}{filename}"


def _s3_key_with_fallback(key: str) -> list:
    """Return a list of S3 keys to attempt, cycling through image extensions.

    When the stored image extension is unknown, this helper returns the
    original key followed by alternatives with ``.jpg``, ``.png``, and
    ``.jpeg`` extensions so callers can try each in order.

    Args:
        key: Primary S3 object key (may include any extension).

    Returns:
        Ordered list of candidate keys. The original key is always first.
        If ``key`` has no extension, only the original key is returned.
    """
    if "." not in key:
        return [key]
    base, ext = key.rsplit(".", 1)
    keys = [key]
    for alt in ["jpg", "png", "jpeg"]:
        if alt != ext.lower():
            keys.append(f"{base}.{alt}")
    return keys


def _fetch_s3_image(key: str) -> Optional[dict]:
    """Download an image from S3, retrying with alternate file extensions.

    Iterates through candidate keys produced by :func:`_s3_key_with_fallback`
    and returns the first successful result.  Logs a warning if all
    candidates fail.

    Args:
        key: Primary S3 object key for the image.

    Returns:
        A dict with keys ``"body"`` (``bytes``) and ``"content_type"``
        (``str``) on success, or ``None`` if the image cannot be retrieved.
    """
    for candidate in _s3_key_with_fallback(key):
        try:
            resp = _s3_client.get_object(Bucket=_s3_conf["bucket"], Key=candidate)
            body = resp["Body"].read()
            content_type = resp.get("ContentType", "image/jpeg")
            return {"body": body, "content_type": content_type}
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                continue
            logger.warning(f"Failed to fetch S3 image {candidate}: {e}")
            return None
        except (NoCredentialsError, BotoCoreError) as e:
            logger.warning(f"Failed to fetch S3 image {candidate}: {e}")
            return None
    logger.warning(f"S3 image not found with any extension: {key}")
    return None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic request / response models


class HistoryMessage(BaseModel):
    """A single turn in a multi-turn conversation.

    Attributes:
        role: Speaker role — must be ``"user"`` or ``"assistant"``.
        content: Text content of the turn (non-empty).
    """

    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1)


class QuestionRequest(BaseModel):
    """Request body for the ``/api/ask`` and ``/api/ask/stream`` endpoints.

    Attributes:
        question: Natural-language question in Tamil or English.
        use_llm: When ``True`` (default) an LLM synthesises a prose answer
            from retrieved passages.  When ``False`` only source passages
            are returned.
        tags: Optional list of taxonomy tag IDs (e.g. ``["FICTION"]``) used
            to restrict retrieval to articles in those categories.
        history: Optional list of prior conversation turns for context.
            Each turn is truncated to :data:`_MAX_HISTORY_CONTENT` characters
            before being passed to the LLM.
        language: Target response language — ``"ta"`` (Tamil, default) or
            ``"en"`` (English).
    """

    question: str = Field(..., min_length=1, description="The question to ask")
    use_llm: bool = Field(default=True, description="Use LLM for answer generation")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tag IDs")
    history: Optional[List[HistoryMessage]] = Field(
        default=None, description="Previous Q&A turns for context"
    )
    language: str = Field(
        default="ta",
        pattern=r"^(ta|en)$",
        description="Response language: 'ta' for Tamil, 'en' for English",
    )


class SourceDocument(BaseModel):
    """A single retrieved passage returned alongside an answer.

    Attributes:
        volume: Source volume identifier.
        heading: Article heading or title.
        doc_issue: Issue identifier within the volume.
        content: Passage text.
        word_count: Word count of the passage.
        chunks_merged: Number of vector chunks merged into this passage.
        score: Retrieval similarity score (0–1).
        tags: Taxonomy tag IDs assigned to the parent article.
    """

    volume: Optional[str] = None
    heading: Optional[str] = None
    doc_issue: Optional[str] = None
    content: str
    word_count: Optional[int] = None
    chunks_merged: Optional[int] = None
    score: Optional[float] = None
    tags: Optional[List[str]] = None


class QuestionResponse(BaseModel):
    """Response body for the ``/api/ask`` and ``/api/search`` endpoints.

    Attributes:
        answer: LLM-generated answer text (empty string when ``use_llm``
            is ``False`` or retrieval returns no results).
        sources: Retrieved source passages that informed the answer.
        query_type: Internal classifier label (e.g. ``"hybrid"``).
        error: Structured error info when the pipeline fails non-fatally.
        fallback_reason: Human-readable reason if the system fell back to
            a simpler retrieval strategy.
    """

    answer: str
    sources: List[Dict[str, Any]] = []
    query_type: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    fallback_reason: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint.

    Attributes:
        status: Overall system status — ``"healthy"`` or ``"degraded"``.
        database: Qdrant health details (keys vary by client version).
        llm: Gemini API health details including latency.
        api: Always ``"healthy"`` — indicates the FastAPI process itself
            is responsive.
    """

    status: str
    database: Dict[str, Any]
    llm: Dict[str, Any]
    api: str = "healthy"


class AuthorInfo(BaseModel):
    """Author name and article count summary.

    Attributes:
        name: Author's display name (may be in Tamil).
        count: Total number of articles attributed to this author.
    """

    name: str
    count: int


class AuthorsListResponse(BaseModel):
    """Response body for ``GET /api/authors``.

    Attributes:
        success: ``True`` when the query completed without errors.
        total_authors: Number of distinct authors in the archive.
        total_articles: Total articles across all authors.
        authors: Sorted list of :class:`AuthorInfo` objects.
    """

    success: bool
    total_authors: int
    total_articles: int
    authors: List[AuthorInfo]


class ArticleInfo(BaseModel):
    """Lightweight article summary used in author/topic queries.

    Attributes:
        title: Article title.
        author: Author name.
        year: Publication year (alias ``ஆண்டு``).
        issue: Issue identifier (alias ``இதழ்``).
    """

    title: str
    author: str
    year: Optional[int] = Field(None, alias="ஆண்டு")
    issue: Optional[str] = Field(None, alias="இதழ்")

    class Config:
        """Pydantic model configuration."""

        populate_by_name = True


class AuthorArticlesResponse(BaseModel):
    """Response body for ``GET /api/authors/{author_name}/articles``.

    Attributes:
        success: ``True`` when the query completed without errors.
        author: The author name as supplied in the request.
        matched_author: Normalised/fuzzy-matched name used for the lookup.
        count: Number of articles found.
        articles: List of article dicts (title, year, issue, …).
        message: Human-readable status message (present on partial failure).
    """

    success: bool
    author: str
    matched_author: Optional[str] = None
    count: int
    articles: List[Dict[str, Any]]
    message: Optional[str] = None


class TopicSearchResponse(BaseModel):
    """Response body for ``GET /api/topics/search``.

    Attributes:
        success: ``True`` when the query completed without errors.
        topic: Topic keyword as used for the search.
        count: Number of matching articles.
        articles: List of article dicts whose titles contain the keyword.
        message: Human-readable status message (present on partial failure).
    """

    success: bool
    topic: str
    count: int
    articles: List[Dict[str, Any]]
    message: Optional[str] = None


class IssueInfo(BaseModel):
    """Article count for a single issue.

    Attributes:
        issue_number: Issue identifier string.
        article_count: Number of articles in the issue.
    """

    issue_number: str
    article_count: int


class IssueStatsResponse(BaseModel):
    """Response body for ``GET /api/issues/stats``.

    Attributes:
        success: ``True`` when the query completed without errors.
        count: Number of distinct issues in the archive.
        total_articles: Total articles across all issues.
        issues: Per-issue article count summaries.
        message: Human-readable status message (present on partial failure).
    """

    success: bool
    count: int
    total_articles: int
    issues: List[IssueInfo]
    message: Optional[str] = None


class VolumeInfo(BaseModel):
    """Metadata for a single volume in the digital library.

    Attributes:
        id: Volume number (1–8).
        year: Publication year or year range (e.g. ``"2018-2020"``).
        issue_count: Number of issues in this volume.
        cover_image_url: Proxy URL for the volume cover image.
    """

    id: int
    year: str
    issue_count: int
    cover_image_url: Optional[str] = None


class VolumeIssue(BaseModel):
    """Metadata for a single issue within a volume.

    Attributes:
        issue_number: Issue identifier string.
        year: Publication year for this specific issue.
        has_pdf: ``True`` if a PDF is available for this issue.
        pdf_url: Direct Google Drive URL, or ``None``.
        cover_image_url: Proxy URL for the issue cover image.
    """

    issue_number: str
    year: str
    has_pdf: bool
    pdf_url: Optional[str] = None
    cover_image_url: Optional[str] = None


class PDFLinkResponse(BaseModel):
    """Response body for ``GET /api/library/volumes/{volume_id}/issues/{issue_id}/pdf``.

    Attributes:
        volume_id: Volume identifier.
        issue_id: Issue identifier.
        pdf_url: Direct Google Drive URL, or ``None`` if unavailable.
        embed_url: Google Drive embed/preview URL, or ``None``.
        proxy_url: Server-side S3 stream URL (avoids CORS), or ``None``
            if S3 key cannot be derived.
        found: ``False`` when no PDF record exists for the requested issue.
    """

    volume_id: int
    issue_id: str
    pdf_url: Optional[str] = None
    embed_url: Optional[str] = None
    proxy_url: Optional[str] = None
    found: bool


class TagInfo(BaseModel):
    """A taxonomy category with article count.

    Attributes:
        id: Machine-readable tag identifier (e.g. ``"FICTION"``).
        tamil: Tamil display label.
        english: English display label.
        count: Number of articles tagged with this category.
    """

    id: str
    tamil: str
    english: str
    count: int


class TagsResponse(BaseModel):
    """Response body for ``GET /api/tags``.

    Attributes:
        success: ``True`` when the query completed without errors.
        tags: Full taxonomy, sorted by article count descending.
    """

    success: bool
    tags: List[TagInfo]


class TagArticleInfo(BaseModel):
    """Lightweight article record returned by tag-based listing.

    Attributes:
        doc_id: Volume/document identifier.
        doc_issue: Issue identifier.
        title: Article title.
        author_name: Author name(s); may be a string or list of strings.
        year: Publication year string.
        tags: All taxonomy tag IDs on this article.
    """

    doc_id: Optional[str] = None
    doc_issue: Optional[str] = None
    title: Optional[str] = None
    author_name: Optional[Union[str, List[str]]] = None
    year: Optional[str] = None
    tags: Optional[List[str]] = None


class TagArticlesResponse(BaseModel):
    """Response body for ``GET /api/tags/{tag_id}/articles``.

    Attributes:
        success: ``True`` when the query completed without errors.
        tag_id: The requested tag identifier.
        tag_tamil: Tamil display label for the tag.
        count: Number of articles in this category.
        articles: Deduplicated list of articles bearing the tag.
    """

    success: bool
    tag_id: str
    tag_tamil: str
    count: int
    articles: List[TagArticleInfo]


class IssueArticleInfo(BaseModel):
    """Article record for issue-level listings.

    Attributes:
        doc_id: Volume/document identifier.
        doc_issue: Issue identifier.
        article_no: Sequential article number within the issue.
        title: Article title.
        author_name: Author name(s); may be a string or list of strings.
        year: Publication year string.
        tags: Taxonomy tag IDs on this article.
    """

    doc_id: Optional[str] = None
    doc_issue: Optional[str] = None
    article_no: Optional[str] = None
    title: Optional[str] = None
    author_name: Optional[Union[str, List[str]]] = None
    year: Optional[str] = None
    tags: Optional[List[str]] = None


class IssueArticlesResponse(BaseModel):
    """Response for ``GET /api/library/volumes/{volume_id}/issues/{issue_id}/articles``.

    Attributes:
        success: ``True`` when the query completed without errors.
        volume_id: Requested volume identifier.
        issue_id: Requested issue identifier.
        count: Number of articles returned.
        articles: Deduplicated article records for the issue.
    """

    success: bool
    volume_id: int
    issue_id: str
    count: int
    articles: List[IssueArticleInfo]


class ArticleContentResponse(BaseModel):
    """Response body for ``GET /api/articles/content``.

    Attributes:
        success: ``True`` when the query completed without errors.
        title: Article title.
        author_name: Author name(s); may be a string or list of strings.
        year: Publication year string.
        doc_issue: Issue identifier.
        tags: Taxonomy tag IDs on this article.
        content: Full article text produced by concatenating all vector
            chunks in chunk-ID order.
        word_count: Number of whitespace-delimited tokens in ``content``.
        chunk_count: Number of vector chunks merged to produce ``content``.
    """

    success: bool
    title: Optional[str] = None
    author_name: Optional[Union[str, List[str]]] = None
    year: Optional[str] = None
    doc_issue: Optional[str] = None
    tags: Optional[List[str]] = None
    content: Optional[str] = None
    word_count: Optional[int] = None
    chunk_count: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown tasks.

    Startup sequence:
        1. Verify Qdrant connectivity and log point count.
        2. Pre-cache the :class:`EnhancedAuthorQuerySystem` from CSV.
        3. Pre-warm ML models (dense E5, sparse BM25, cross-encoder) and
           content vocabulary to avoid cold-start timeouts on the first
           user request.
        4. Validate the Gemini API key and log round-trip latency.

    Shutdown:
        Logs a shutdown message.  Resource cleanup is handled by the
        underlying libraries (Qdrant client, boto3 session).

    Args:
        app: The FastAPI application instance (injected by FastAPI).

    Yields:
        Control to the running application.
    """
    logger.info("Starting Ponni RAG API...")

    # Check database health on startup
    health = check_qdrant_health()
    if health["healthy"]:
        logger.info(f"Qdrant connected: {health.get('points_count', 0)} points")
    else:
        logger.warning(
            f"Qdrant not available: {health.get('message', 'Unknown error')}"
        )

    # Pre-cache the author query system at startup
    csv_path = str(CSV_PATH)
    if CSV_PATH.exists():
        with _author_system_lock:
            if csv_path not in _author_system_cache:
                _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
        logger.info("Author query system cached at startup")

    # Pre-warm ML models so the first user query does not pay the
    # cold-start tax (dense E5 ~5s, BM25 ~3s, cross-encoder ~15s on CPU).
    # Combined this can exceed the reverse-proxy timeout (504).
    try:
        from embeddings import get_embed_model
        from reranker import get_reranker
        from sparse import get_bm25_model

        get_embed_model()
        logger.info("Dense embedding model warm")
        get_bm25_model()
        logger.info("Sparse BM25 model warm")
        get_reranker()
        logger.info("Cross-encoder reranker warm")

        from hybrid_search import _build_content_vocab

        _build_content_vocab()
        logger.info("Content vocabulary warm")
    except Exception as e:
        logger.warning(f"Model pre-warm failed (will lazy-load on first call): {e!r}")

    # Validate Gemini API key (also primes the health check cache)
    gemini_health = check_gemini_health()
    if gemini_health["healthy"]:
        model = gemini_health.get("model", "unknown")
        latency = gemini_health.get("latency_ms", "N/A")
        logger.info(f"Gemini API validated (model: {model}," f" latency: {latency}ms)")
    else:
        logger.warning(
            f"Gemini API not available: {gemini_health.get('message', 'Unknown error')}"
        )

    yield

    logger.info("Shutting down Ponni RAG API...")


# FastAPI application
app = FastAPI(
    title="Ponni RAG API",
    description=(
        "REST API for the **Ponni Tamil Literary Archive** RAG System.\n\n"
        "Provides semantic search, AI-powered question answering, and full "
        "digital-library access for the Ponni magazine archive.  All search "
        "endpoints use hybrid dense (E5) + sparse (BM25) retrieval with "
        "cross-encoder reranking, filtered to passages with ≥ 80 % similarity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health of all system components.

    Runs Qdrant and Gemini health probes concurrently and returns an
    aggregated status.  The overall status is ``"degraded"`` if either
    backend is unavailable.

    Returns:
        :class:`HealthResponse` with per-component health detail.
    """
    db_health = check_qdrant_health()
    llm_health = await asyncio.to_thread(check_gemini_health)

    overall = "healthy"
    if not db_health["healthy"] or not llm_health["healthy"]:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        database=db_health,
        llm=llm_health,
        api="healthy",
    )


@app.get("/api/debug/s3", tags=["Health"])
async def debug_s3(request: Request):
    """Diagnose S3 connectivity and credential configuration.

    Performs a ``HeadObject`` call against the first volume cover to verify
    that the S3 client has valid credentials and that the expected object
    exists.  Returns sample proxy URLs for manual verification.

    Args:
        request: Incoming HTTP request (used to derive the base URL).

    Returns:
        Dict with keys ``credentials`` (bool), ``object_exists`` (bool),
        ``object_size`` (int), and sample proxy URLs.  An ``error`` key
        is included if an unexpected exception occurs.
    """
    base = str(request.base_url).rstrip("/")
    result = {"credentials": False, "error": None}
    try:
        vol = _magazine["volumes"][0]
        key = _volume_cover_s3_key(vol["id"])
        try:
            head = _s3_client.head_object(Bucket=_s3_conf["bucket"], Key=key)
            result["credentials"] = True
            result["object_exists"] = True
            result["object_size"] = head["ContentLength"]
        except (ClientError, NoCredentialsError) as e:
            result["object_exists"] = False
            result["head_error"] = str(e)

        result["sample_volume_url"] = f"{base}/api/images/volumes/{vol['id']}/cover"
        iss = vol["issues"][0]
        result["sample_issue_url"] = (
            f"{base}/api/images/volumes/{vol['id']}/issues/{iss['num']}/cover"
        )
    except Exception as e:
        result["error"] = str(e)
    return result


@app.get("/", tags=["Health"])
async def root():
    """Return basic API metadata.

    Returns:
        Dict with ``name``, ``version``, ``description``, ``docs``, and
        ``health`` keys.
    """
    return {
        "name": "Ponni RAG API",
        "version": "1.0.0",
        "description": "REST API for Tamil Literary Archive",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api/images/volumes/{volume_id}/cover", tags=["Images"])
async def get_volume_cover(volume_id: int):
    """Stream the cover image for a specific issue from S3.

    Args:
        volume_id: Integer volume identifier (1–8).
        issue_name: Issue identifier string (e.g. ``"1"``, ``"PONGAL"``).

    Returns:
        Raw image bytes with the appropriate ``Content-Type``.

    Raises:
        HTTPException 404: Issue not found in registry, or cover image
            absent from S3.
    """
    key = _volume_cover_s3_key(volume_id)
    if not key:
        raise HTTPException(status_code=404, detail="Volume not found")
    img = await asyncio.to_thread(_fetch_s3_image, key)
    if not img:
        raise HTTPException(status_code=404, detail="Cover image not found")
    return Response(
        content=img["body"],
        media_type=img["content_type"],
        headers={"Cache-Control": f"public, max-age={_IMAGE_CACHE_SECONDS}"},
    )


@app.get("/api/images/volumes/{volume_id}/issues/{issue_name}/cover", tags=["Images"])
async def get_issue_cover(volume_id: int, issue_name: str):
    """Stream an about-page image from S3.

    Only filenames matching the pattern ``about{1-9}.{png|jpg}`` are
    permitted to prevent path-traversal attacks.

    Args:
        filename: Image filename (e.g. ``"about1.png"``).

    Returns:
        Raw image bytes with the appropriate ``Content-Type``.

    Raises:
        HTTPException 404: Filename is not in the allowed set, or the
            image is absent from S3.
    """
    key = _issue_cover_s3_key(volume_id, issue_name)
    if not key:
        raise HTTPException(status_code=404, detail="Issue not found")
    img = await asyncio.to_thread(_fetch_s3_image, key)
    if not img:
        raise HTTPException(status_code=404, detail="Cover image not found")
    return Response(
        content=img["body"],
        media_type=img["content_type"],
        headers={"Cache-Control": f"public, max-age={_IMAGE_CACHE_SECONDS}"},
    )


@app.api_route("/api/images/about/{filename}", methods=["GET", "HEAD"], tags=["Images"])
async def get_about_image(filename: str):
    """Stream an about-page image from S3.

    Only filenames matching the pattern ``about{1-9}.{png|jpg}`` are
    permitted to prevent path-traversal attacks.

    Args:
        filename: Image filename (e.g. ``"about1.png"``).

    Returns:
        Raw image bytes with the appropriate ``Content-Type``.

    Raises:
        HTTPException 404: Filename is not in the allowed set, or the
            image is absent from S3.
    """
    # Only allow specific filenames to prevent path traversal
    allowed = {f"about{i}.{ext}" for i in range(1, 10) for ext in ("png", "jpg")}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Image not found")
    key = f"about/{filename}"
    img = await asyncio.to_thread(_fetch_s3_image, key)
    if not img:
        raise HTTPException(status_code=404, detail="About image not found")
    return Response(
        content=img["body"],
        media_type=img["content_type"],
        headers={"Cache-Control": f"public, max-age={_IMAGE_CACHE_SECONDS}"},
    )


def _truncate_history(request: "QuestionRequest"):
    """Truncate long history entries instead of rejecting the request."""
    if not request.history:
        return None
    history = []
    for h in request.history:
        content = h.content
        if len(content) > _MAX_HISTORY_CONTENT:
            content = content[:_MAX_HISTORY_CONTENT]
            logger.info(
                f"Truncated {h.role} history entry from "
                f"{len(h.content)} to {_MAX_HISTORY_CONTENT} chars"
            )
        history.append({"role": h.role, "content": content})
    return history


@app.post("/api/ask", response_model=QuestionResponse, tags=["Search"])
async def ask_question_endpoint(request: QuestionRequest):
    """Answer a question using hybrid retrieval and an LLM.

    Performs hybrid dense + sparse vector search against the Qdrant
    collection, reranks candidates with a cross-encoder, and optionally
    calls the Gemini LLM to synthesise a prose answer from the top
    passages.  Only passages with ≥ 80 % similarity score are returned.

    Args:
        request: Question, language, optional tag filters, and optional
            conversation history.

    Returns:
        :class:`QuestionResponse` containing the answer text and source
        passages.

    Raises:
        HTTPException 500: Unexpected error during retrieval or generation.
    """
    try:
        logger.info(f"Question received: {request.question[:100]}...")

        history = _truncate_history(request)
        result = await ask_question_async(
            question=request.question,
            return_formatted=False,
            use_llm=request.use_llm,
            filter_tags=request.tags,
            history=history,
            language=request.language,
        )

        return QuestionResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            query_type=result.get("query_type"),
            error=result.get("error"),
            fallback_reason=result.get("fallback_reason"),
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.post("/api/ask/stream", tags=["Search"])
async def ask_question_stream_endpoint(request: QuestionRequest):
    """Stream an AI-generated answer token-by-token via Server-Sent Events.

    The generator yields SSE frames until the answer is complete, then
    emits a ``sources`` frame followed by ``done``.  On error an ``error``
    frame is emitted before the generator exits.

    Args:
        request: Same shape as :func:`ask_question_endpoint`.

    Returns:
        :class:`StreamingResponse` with ``text/event-stream`` media type
        and headers that disable proxy buffering.
    """
    history = _truncate_history(request)

    def event_generator():
        """Yield SSE events for streaming response."""
        try:
            for event in ask_question_stream(
                question=request.question,
                filter_tags=request.tags,
                history=history,
                language=request.language,
            ):
                if event["type"] == "token":
                    data = json.dumps({"content": event["content"]})
                    yield f"event: token\ndata: {data}\n\n"
                elif event["type"] == "fallback":
                    data = json.dumps({"reason": event["reason"]})
                    yield f"event: fallback\ndata: {data}\n\n"
                elif event["type"] == "sources":
                    data = json.dumps({"sources": event["sources"]})
                    yield f"event: sources\ndata: {data}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            err = json.dumps({"error": "An internal error occurred."})
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/cache/stats", tags=["Cache"])
async def cache_stats():
    """Return response cache statistics."""
    return _response_cache.stats()


@app.post("/api/cache/clear", tags=["Cache"])
async def cache_clear():
    """Clear the response cache."""
    _response_cache.clear()
    return {"message": "Cache cleared"}


@app.get("/api/search", response_model=QuestionResponse, tags=["Search"])
async def search_endpoint(
    q: str = Query(..., min_length=1, max_length=2000, description="Search query"),
    use_llm: bool = Query(default=False, description="Use LLM for answer"),
    tags: Optional[str] = Query(
        default=None, description="Comma-separated tag IDs to filter by"
    ),
):
    """Search the archive with a URL query parameter.

    GET-friendly alternative to ``POST /api/ask``.  Accepts the same
    hybrid retrieval pipeline with optional LLM synthesis.  Only
    passages with ≥ 80 % similarity are returned.

    Args:
        q: Search query string (Tamil or English).
        use_llm: When ``True``, an LLM generates a prose answer.
        tags: Optional comma-separated tag IDs to filter results.

    Returns:
        :class:`QuestionResponse` containing passages and optional answer.

    Raises:
        HTTPException 500: Unexpected error during retrieval or generation.
    """
    try:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        result = await ask_question_async(
            question=q,
            return_formatted=False,
            use_llm=use_llm,
            filter_tags=tag_list,
        )

        return QuestionResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            query_type=result.get("query_type"),
            error=result.get("error"),
            fallback_reason=result.get("fallback_reason"),
        )

    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/tags", response_model=TagsResponse, tags=["Tags"])
async def list_tags():
    """List all 15 article taxonomy categories with article counts.

    Scrolls the Qdrant collection to count how many articles belong to
    each category.  If Qdrant is unavailable, returns the full taxonomy
    with zero counts so that UI dropdowns still populate.

    Returns:
        :class:`TagsResponse` with tags sorted by count descending.
    """
    try:
        client = get_qdrant_client()

        # Aggregate tag counts by scrolling all article points
        tag_counts = {}
        offset = None
        while True:
            points, offset = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type", match=models.MatchValue(value="article")
                        ),
                        models.FieldCondition(
                            key="metadata.chunk_id", match=models.MatchValue(value=0)
                        ),
                    ]
                ),
                limit=500,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                metadata = (p.payload or {}).get("metadata", {})
                for tag in metadata.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if offset is None:
                break

        tags_list = []
        for cat_id, cat_info in TAXONOMY.items():
            tags_list.append(
                TagInfo(
                    id=cat_id,
                    tamil=cat_info["tamil"],
                    english=cat_info["english"],
                    count=tag_counts.get(cat_id, 0),
                )
            )

        # Sort by count descending
        tags_list.sort(key=lambda t: t.count, reverse=True)

        return TagsResponse(success=True, tags=tags_list)

    except Exception as e:
        logger.error(f"Error listing tags: {e}")
        # Return taxonomy with zero counts instead of 500 error —
        # lets the dropdown populate even if Qdrant is unavailable.
        tags_list = [
            TagInfo(
                id=cat_id, tamil=cat_info["tamil"], english=cat_info["english"], count=0
            )
            for cat_id, cat_info in TAXONOMY.items()
        ]
        return TagsResponse(success=True, tags=tags_list)


@app.get(
    "/api/tags/{tag_id}/articles", response_model=TagArticlesResponse, tags=["Tags"]
)
async def get_tag_articles(tag_id: str):
    """Retrieve all articles associated with a given taxonomy category.

    Scrolls the Qdrant collection for chunk-0 article points that carry
    the requested tag, deduplicates by ``(doc_id, doc_issue, article_no)``,
    and returns the cleaned list.

    Args:
        tag_id: Taxonomy tag identifier (e.g. ``"FICTION"``).

    Returns:
        :class:`TagArticlesResponse` with a deduplicated article list.

    Raises:
        HTTPException 404: ``tag_id`` is not in the known taxonomy.
        HTTPException 500: Unexpected error during Qdrant scroll.
    """
    if tag_id not in TAXONOMY:
        raise HTTPException(status_code=404, detail=f"Unknown tag ID: {tag_id}")

    try:
        client = get_qdrant_client()

        seen = set()
        articles = []
        offset = None
        while True:
            points, offset = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type", match=models.MatchValue(value="article")
                        ),
                        models.FieldCondition(
                            key="metadata.tags", match=models.MatchAny(any=[tag_id])
                        ),
                        models.FieldCondition(
                            key="metadata.chunk_id", match=models.MatchValue(value=0)
                        ),
                    ]
                ),
                limit=500,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                metadata = (p.payload or {}).get("metadata", {})
                dedup_key = (
                    metadata.get("doc_id"),
                    metadata.get("doc_issue"),
                    metadata.get("article_no"),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                articles.append(
                    TagArticleInfo(
                        doc_id=metadata.get("doc_id"),
                        doc_issue=metadata.get("doc_issue"),
                        title=metadata.get("title"),
                        author_name=(
                            metadata.get("author_name", "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace('"', "")
                            .replace("'", "")
                            .strip()
                            if isinstance(metadata.get("author_name"), str)
                            else [
                                a.replace("[", "")
                                .replace("]", "")
                                .replace('"', "")
                                .replace("'", "")
                                .strip()
                                for a in (metadata.get("author_name") or [])
                            ]
                        ),
                        year=metadata.get("year"),
                        tags=metadata.get("tags", []),
                    )
                )
            if offset is None:
                break

        return TagArticlesResponse(
            success=True,
            tag_id=tag_id,
            tag_tamil=TAXONOMY[tag_id]["tamil"],
            count=len(articles),
            articles=articles,
        )

    except Exception as e:
        logger.error(f"Error getting tag articles: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/authors", response_model=AuthorsListResponse, tags=["Authors"])
async def list_authors():
    """List every author in the archive with their article count.

    Delegates to :class:`EnhancedAuthorQuerySystem`, which is pre-cached
    at startup.  The result is sorted by article count descending.

    Returns:
        :class:`AuthorsListResponse` with total author/article counts.

    Raises:
        HTTPException 404: Author CSV is missing from the filesystem.
        HTTPException 500: Author system failed to load or query.
    """
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Author data not available")

        csv_path = str(CSV_PATH)
        with _author_system_lock:
            if csv_path not in _author_system_cache:
                _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
            system = _author_system_cache[csv_path]

        result = await asyncio.to_thread(system.list_all_authors)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=result.get("message", "Failed to load authors")
            )

        return AuthorsListResponse(
            success=True,
            total_authors=result["total_authors"],
            total_articles=result["total_articles"],
            authors=[
                AuthorInfo(name=a["name"], count=a["count"]) for a in result["authors"]
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing authors: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get(
    "/api/authors/{author_name}/articles",
    response_model=AuthorArticlesResponse,
    tags=["Authors"],
)
async def get_author_articles(author_name: str):
    """Retrieve all articles written by a specific author.

    Author matching is fuzzy — the system normalises the supplied name
    and returns the closest match, reported in ``matched_author``.

    Args:
        author_name: Author name to look up (Tamil or transliterated).

    Returns:
        :class:`AuthorArticlesResponse` with matched author and article list.

    Raises:
        HTTPException 404: Author CSV is missing from the filesystem.
        HTTPException 500: Unexpected error during author lookup.
    """
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Author data not available")

        csv_path = str(CSV_PATH)
        with _author_system_lock:
            if csv_path not in _author_system_cache:
                _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
            system = _author_system_cache[csv_path]

        result = await asyncio.to_thread(system.get_topics_by_author, author_name)

        return AuthorArticlesResponse(
            success=result["success"],
            author=result["author"],
            matched_author=result.get("matched_author"),
            count=result.get("count", 0),
            articles=result.get("articles", []),
            message=result.get("message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting author articles: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/topics/search", response_model=TopicSearchResponse, tags=["Topics"])
async def search_by_topic(
    topic: str = Query(..., min_length=1, description="Topic to search for")
):
    """Find articles whose titles contain a specific keyword.

    The search is case-insensitive substring matching performed by
    :class:`EnhancedAuthorQuerySystem`.

    Args:
        topic: Topic keyword (Tamil or English).

    Returns:
        :class:`TopicSearchResponse` with matching article list.

    Raises:
        HTTPException 404: Author/topic CSV is missing from the filesystem.
        HTTPException 500: Unexpected error during topic search.
    """
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Topic data not available")

        csv_path = str(CSV_PATH)
        with _author_system_lock:
            if csv_path not in _author_system_cache:
                _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
            system = _author_system_cache[csv_path]

        result = await asyncio.to_thread(system.get_author_by_topic, topic)

        return TopicSearchResponse(
            success=result["success"],
            topic=result.get("topic", topic),
            count=result.get("count", 0),
            articles=result.get("articles", []),
            message=result.get("message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching topic: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/issues/stats", response_model=IssueStatsResponse, tags=["Issues"])
async def get_issue_statistics():
    """Return article counts for every issue in the archive.

    Delegates to :func:`get_issue_count` which reads the CSV index.

    Returns:
        :class:`IssueStatsResponse` with total issue count, total article
        count, and per-issue breakdown.

    Raises:
        HTTPException 404: Issue CSV is missing from the filesystem.
        HTTPException 500: Unexpected error during computation.
    """
    try:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=404, detail="Issue data not available")

        result = await asyncio.to_thread(get_issue_count, str(CSV_PATH))

        return IssueStatsResponse(
            success=result["success"],
            count=result.get("count", 0),
            total_articles=result.get("total_articles", 0),
            issues=[IssueInfo(**i) for i in result.get("issues", [])],
            message=result.get("message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting issue stats: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/library/volumes", response_model=List[VolumeInfo], tags=["Library"])
async def list_volumes():
    """List all volumes in the Ponni digital library.

    Derives year ranges from per-issue year fields so multi-year volumes
    display an accurate range (e.g. ``"2018-2020"``).

    Returns:
        List of :class:`VolumeInfo` objects, one per volume.
    """
    result = []
    for vol in _magazine["volumes"]:
        # Derive year range from per-issue years
        issue_years = sorted(set(iss.get("year", vol["year"]) for iss in vol["issues"]))
        if len(issue_years) > 1:
            year_display = f"{issue_years[0]}-{issue_years[-1]}"
        else:
            year_display = issue_years[0] if issue_years else vol["year"]

        result.append(
            VolumeInfo(
                id=vol["id"],
                year=year_display,
                issue_count=len(vol["issues"]),
                cover_image_url=f"/api/images/volumes/{vol['id']}/cover",
            )
        )
    return result


@app.get(
    "/api/library/volumes/{volume_id}/issues",
    response_model=List[VolumeIssue],
    tags=["Library"],
)
async def get_volume_issues(volume_id: int):
    """List all issues for a given volume.

    Args:
        volume_id: Volume number (1–8).

    Returns:
        List of :class:`VolumeIssue` objects in registry order.

    Raises:
        HTTPException 404: Volume not found, or no issues registered.
    """
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data:
        raise HTTPException(status_code=404, detail="Volume not found")

    issues = []
    for issue in vol_data["issues"]:
        issue_year = issue.get("year", vol_data["year"])
        issue_name = str(issue["num"])
        issues.append(
            VolumeIssue(
                issue_number=issue_name,
                year=issue_year,
                has_pdf=bool(issue.get("pdf_url")),
                pdf_url=issue.get("pdf_url"),
                cover_image_url=(
                    f"/api/images/volumes/" f"{volume_id}/issues/{issue_name}/cover"
                ),
            )
        )

    if not issues:
        raise HTTPException(status_code=404, detail="No issues found for this volume")

    return issues


@app.get(
    "/api/library/volumes/{volume_id}/issues/{issue_id}/pdf",
    response_model=PDFLinkResponse,
    tags=["Library"],
)
async def get_pdf_link(volume_id: int, issue_id: str, request: Request):
    """Return PDF access URLs for a specific issue.

    Three URL types are returned when available:

    - ``pdf_url`` — direct Google Drive link.
    - ``embed_url`` — Google Drive embed/preview URL (for ``<iframe>`` use).
    - ``proxy_url`` — server-side S3 stream path that avoids Drive CORS
      restrictions and supports HTTP Range requests for pdf.js lazy-loading.

    Args:
        volume_id: Volume number (1–8).
        issue_id: Issue identifier (e.g. ``"1"``, ``"PONGAL"``).
        request: Incoming HTTP request (not currently used, reserved for
            future base-URL construction).

    Returns:
        :class:`PDFLinkResponse`; ``found`` is ``False`` when no PDF
        record exists.
    """
    key = f"vol_{volume_id}_issue_{issue_id}"
    pdf_url = PDF_LINKS.get(key)

    if not pdf_url:
        return PDFLinkResponse(volume_id=volume_id, issue_id=issue_id, found=False)

    embed_url = None
    if "/d/" in pdf_url:
        file_id = pdf_url.split("/d/")[1].split("/")[0]
        embed_url = f"https://drive.google.com/file/d/{file_id}/preview"

    proxy_url = None
    if _issue_pdf_s3_key(volume_id, issue_id):
        proxy_url = f"/api/library/volumes/{volume_id}/issues/{issue_id}/pdf/stream"

    return PDFLinkResponse(
        volume_id=volume_id,
        issue_id=issue_id,
        pdf_url=pdf_url,
        embed_url=embed_url,
        proxy_url=proxy_url,
        found=True,
    )


_PDF_CACHE_SECONDS = 2592000  # 30 days; PDFs are immutable


@app.api_route(
    "/api/library/volumes/{volume_id}/issues/{issue_id}/pdf/stream",
    methods=["GET", "HEAD"],
    tags=["Library"],
)
async def stream_issue_pdf(volume_id: int, issue_id: str, request: Request):
    """Stream an issue PDF from S3 with HTTP Range request support.

    Supports byte-range requests so that pdf.js (used in the frontend)
    can lazy-load only the pages it needs.  PDFs are served with a
    30-day ``Cache-Control: immutable`` header.

    ``HEAD`` requests return headers only; the S3 response body is
    closed immediately without reading.

    Args:
        volume_id: Volume number (1–8).
        issue_id: Issue identifier (e.g. ``"1"``, ``"PONGAL"``).
        request: Incoming HTTP request; the ``Range`` header is forwarded
            to S3 when present.

    Returns:
        Streaming PDF response with ``Accept-Ranges``, ``ETag``, and
        ``Content-Length`` headers.

    Raises:
        HTTPException 404: Issue not in registry, or PDF absent from S3.
        HTTPException 416: Requested byte range is invalid.
        HTTPException 502: Unexpected S3 or credential error.
    """
    s3_key = _issue_pdf_s3_key(volume_id, issue_id)
    if not s3_key:
        raise HTTPException(status_code=404, detail="Issue not found")

    range_header = request.headers.get("range")

    s3_kwargs = {"Bucket": _s3_conf["bucket"], "Key": s3_key}
    if range_header:
        s3_kwargs["Range"] = range_header

    try:
        resp = await asyncio.to_thread(_s3_client.get_object, **s3_kwargs)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            raise HTTPException(status_code=404, detail="PDF not found in S3")
        if code in ("InvalidRange", "416"):
            raise HTTPException(status_code=416, detail="Invalid range")
        logger.warning(f"S3 PDF fetch failed for {s3_key}: {e}")
        raise HTTPException(status_code=502, detail="Upstream S3 error")
    except (NoCredentialsError, BotoCoreError) as e:
        logger.warning(f"S3 PDF fetch failed for {s3_key}: {e}")
        raise HTTPException(status_code=502, detail="Upstream S3 error")

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": f"public, max-age={_PDF_CACHE_SECONDS}, immutable",
        "Content-Length": str(resp["ContentLength"]),
        "Content-Disposition": (
            f'inline; filename="vol{volume_id}-issue{issue_id}.pdf"'
        ),
    }
    if resp.get("ETag"):
        headers["ETag"] = resp["ETag"]
    if resp.get("LastModified"):
        headers["Last-Modified"] = resp["LastModified"].strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
    if resp.get("ContentRange"):
        headers["Content-Range"] = resp["ContentRange"]

    status_code = 206 if range_header and resp.get("ContentRange") else 200

    if request.method == "HEAD":
        resp["Body"].close()
        return Response(
            status_code=status_code, headers=headers, media_type="application/pdf"
        )

    body = resp["Body"]

    def iter_chunks():
        try:
            for chunk in body.iter_chunks(chunk_size=65536):
                yield chunk
        finally:
            body.close()

    return StreamingResponse(
        iter_chunks(),
        status_code=status_code,
        headers=headers,
        media_type="application/pdf",
    )


@app.get(
    "/api/library/volumes/{volume_id}/issues/{issue_id}/articles",
    response_model=IssueArticlesResponse,
    tags=["Library"],
)
async def get_issue_articles(volume_id: int, issue_id: str):
    """Return all articles for a specific volume and issue.

    Retrieval strategy (applied in order):

    1. Query Qdrant for chunk-0 article points with ``doc_id == volume_id``.
    2. **Fallback:** if no points are found, fetch *all* chunk-0 articles and
       filter client-side by ``doc_id``.
    3. Normalise ``doc_issue`` values (e.g. map Pongal variants to a
       canonical form) and filter to those matching ``issue_id``.
    4. Deduplicate by ``(doc_id, doc_issue, article_no)`` and return.

    Args:
        volume_id: Volume number (1–8).
        issue_id: Issue identifier string (e.g. ``"1"``, ``"PONGAL"``).

    Returns:
        :class:`IssueArticlesResponse` with deduplicated article records.

    Raises:
        HTTPException 500: Unexpected error during Qdrant scroll.
    """
    try:
        client = get_qdrant_client()
        vol_str = str(volume_id)

        # Step 1: fetch chunk_0 articles filtered by doc_id
        all_points = []
        scroll_offset = None
        while True:
            points, scroll_offset = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type", match=models.MatchValue(value="article")
                        ),
                        models.FieldCondition(
                            key="metadata.chunk_id", match=models.MatchValue(value=0)
                        ),
                        models.FieldCondition(
                            key="metadata.doc_id",
                            match=models.MatchValue(value=vol_str),
                        ),
                    ]
                ),
                limit=500,
                offset=scroll_offset,
                with_payload=True,
            )
            all_points.extend(points)
            if scroll_offset is None:
                break

        logger.info(
            "get_issue_articles: vol=%s, doc_id filter" " → %d points",
            volume_id,
            len(all_points),
        )

        # Fallback: if doc_id filter found nothing,
        # fetch all chunk_0 and filter client-side
        if not all_points:
            logger.info("No points with doc_id filter — fetching all chunk_0 articles")
            scroll_offset = None
            while True:
                points, scroll_offset = await asyncio.to_thread(
                    client.scroll,
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type", match=models.MatchValue(value="article")
                            ),
                            models.FieldCondition(
                                key="metadata.chunk_id",
                                match=models.MatchValue(value=0),
                            ),
                        ]
                    ),
                    limit=500,
                    offset=scroll_offset,
                    with_payload=True,
                )
                all_points.extend(points)
                if scroll_offset is None:
                    break
            all_points = [
                p
                for p in all_points
                if str((p.payload or {}).get("metadata", {}).get("doc_id", ""))
                == vol_str
            ]
            logger.info(f"After client-side doc_id filter: {len(all_points)}")

        # Use issue_id directly as the doc_issue value
        # (actual issue number, not position)
        target_issue = str(issue_id)
        logger.info(f"Filtering by doc_issue='{target_issue}'")

        # Filter and deduplicate
        seen = set()
        articles = []
        for p in all_points:
            metadata = (p.payload or {}).get("metadata", {})
            doc_issue_str = str(metadata.get("doc_issue", "")).strip()
            normalized_doc_issue = doc_issue_str.strip().lower()
            normalized_target = str(target_issue).strip().lower()

            if "pongal" in normalized_doc_issue or "பொங்கல்" in normalized_doc_issue:
                normalized_doc_issue = "pongal"

            if "pongal" in normalized_target or "பொங்கல்" in normalized_target:
                normalized_target = "pongal"

            print("DOC ISSUE:", normalized_doc_issue)
            print("TARGET:", normalized_target)

            if normalized_doc_issue != normalized_target:
                continue
            print("MATCHED:", metadata.get("title"))
            doc_id = metadata.get("doc_id")
            article_no = str(metadata.get("article_no", ""))
            unique_key = f"{doc_id}_{doc_issue_str}_{article_no}"
            if unique_key in seen:
                continue
            seen.add(unique_key)
            articles.append(
                IssueArticleInfo(
                    doc_id=doc_id,
                    doc_issue=doc_issue_str,
                    article_no=article_no,
                    title=metadata.get("title"),
                    author_name=(
                        metadata.get("author_name", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .replace("'", "")
                        .strip()
                        if isinstance(metadata.get("author_name"), str)
                        else [
                            a.replace("[", "")
                            .replace("]", "")
                            .replace('"', "")
                            .replace("'", "")
                            .strip()
                            for a in (metadata.get("author_name") or [])
                        ]
                    ),
                    year=metadata.get("year"),
                    tags=metadata.get("tags", []),
                )
            )

        return IssueArticlesResponse(
            success=True,
            volume_id=volume_id,
            issue_id=issue_id,
            count=len(articles),
            articles=articles,
        )

    except Exception as e:
        logger.error(f"Error fetching issue articles: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get(
    "/api/articles/content", response_model=ArticleContentResponse, tags=["Library"]
)
async def get_article_content(
    doc_id: str = Query(..., description="Document/volume ID"),
    doc_issue: str = Query(..., description="Issue number"),
    article_no: Optional[str] = Query(
        default=None, description="Article number within the issue"
    ),
):
    """Retrieve the full text of an article by concatenating its vector chunks.

    Scrolls Qdrant for all chunks matching the supplied identifiers, sorts
    them by ``chunk_id``, and joins their ``content`` fields with newlines.
    Metadata (title, author, year, tags) is taken from the first chunk.

    Args:
        doc_id: Volume/document identifier (e.g. ``"3"``).
        doc_issue: Issue identifier (e.g. ``"1"``, ``"PONGAL"``).
        article_no: Optional article number.  When omitted, all articles
            in the issue are concatenated (useful for issue-level export).

    Returns:
        :class:`ArticleContentResponse` with full content and metadata.
        Returns a successful empty response when no chunks are found.

    Raises:
        HTTPException 500: Unexpected error during Qdrant scroll.
    """
    try:
        client = get_qdrant_client()

        filter_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="article")),
            models.FieldCondition(
                key="metadata.doc_id", match=models.MatchValue(value=str(doc_id))
            ),
            models.FieldCondition(
                key="metadata.doc_issue", match=models.MatchValue(value=str(doc_issue))
            ),
        ]
        if article_no:
            match_val = int(article_no) if str(article_no).isdigit() else article_no
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.article_no",
                    match=models.MatchValue(value=match_val),
                )
            )

        chunks = []
        scroll_offset = None
        while True:
            points, scroll_offset = await asyncio.to_thread(
                client.scroll,
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=100,
                offset=scroll_offset,
                with_payload=True,
            )
            for p in points:
                payload = p.payload or {}
                metadata = payload.get("metadata", {})
                chunks.append(
                    {
                        "chunk_id": metadata.get("chunk_id", 0),
                        "content": payload.get("content", ""),
                        "metadata": metadata,
                    }
                )
            if scroll_offset is None:
                break

        logger.info(
            f"get_article_content: doc_id={doc_id}, doc_issue={doc_issue}, "
            f"article_no={article_no}, chunks={len(chunks)}"
        )

        if not chunks:
            return ArticleContentResponse(success=True)

        chunks.sort(key=lambda c: c["chunk_id"])
        first_meta = chunks[0]["metadata"]
        full_content = "\n".join(c["content"] for c in chunks)

        return ArticleContentResponse(
            success=True,
            title=first_meta.get("title"),
            author_name=(
                first_meta.get("author_name", "")
                .replace("[", "")
                .replace("]", "")
                .replace('"', "")
                .replace("'", "")
                .strip()
                if isinstance(first_meta.get("author_name"), str)
                else [
                    a.replace("[", "")
                    .replace("]", "")
                    .replace('"', "")
                    .replace("'", "")
                    .strip()
                    for a in (first_meta.get("author_name") or [])
                ]
            ),
            year=first_meta.get("year"),
            doc_issue=first_meta.get("doc_issue"),
            tags=first_meta.get("tags", []),
            content=full_content,
            word_count=len(full_content.split()),
            chunk_count=len(chunks),
        )

    except Exception as e:
        logger.error(f"Error fetching article content: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
