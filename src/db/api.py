"""FastAPI REST API for Ponni RAG System.

Provides endpoints for search, question answering, and library access.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from article_tagger import TAXONOMY  # noqa: E402
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


def _volume_cover_s3_key(volume_id: int) -> Optional[str]:
    """Derive S3 key for a volume cover image."""
    vol_data = next((v for v in _magazine["volumes"] if v["id"] == volume_id), None)
    if not vol_data or not vol_data.get("cover_image"):
        return None
    return f"{_s3_conf['covers_prefix']}Volumes/{vol_data['cover_image']}"


def _issue_cover_s3_key(volume_id: int, issue_name: str) -> Optional[str]:
    """Derive S3 key for an issue cover image using convention patterns."""
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


def _s3_key_with_fallback(key: str) -> list:
    """Return S3 keys to try: original plus alternate extensions."""
    if "." not in key:
        return [key]
    base, ext = key.rsplit(".", 1)
    keys = [key]
    for alt in ["jpg", "png", "jpeg"]:
        if alt != ext.lower():
            keys.append(f"{base}.{alt}")
    return keys


def _fetch_s3_image(key: str) -> Optional[dict]:
    """Fetch an image from S3, trying alternate extensions on failure."""
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


class HistoryMessage(BaseModel):
    """A single conversation turn (user or assistant)."""

    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=5000)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""

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
    """Response model for source documents."""

    volume: Optional[str] = None
    heading: Optional[str] = None
    doc_issue: Optional[str] = None
    content: str
    word_count: Optional[int] = None
    chunks_merged: Optional[int] = None
    score: Optional[float] = None
    tags: Optional[List[str]] = None


class QuestionResponse(BaseModel):
    """Response model for question answering."""

    answer: str
    sources: List[Dict[str, Any]] = []
    query_type: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    fallback_reason: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    database: Dict[str, Any]
    llm: Dict[str, Any]
    api: str = "healthy"


class AuthorInfo(BaseModel):
    """Response model for author information."""

    name: str
    count: int


class AuthorsListResponse(BaseModel):
    """Response model for list of all authors."""

    success: bool
    total_authors: int
    total_articles: int
    authors: List[AuthorInfo]


class ArticleInfo(BaseModel):
    """Response model for article information."""

    title: str
    author: str
    year: Optional[int] = Field(None, alias="ஆண்டு")
    issue: Optional[str] = Field(None, alias="இதழ்")

    class Config:
        """Pydantic model configuration."""

        populate_by_name = True


class AuthorArticlesResponse(BaseModel):
    """Response model for articles by author."""

    success: bool
    author: str
    matched_author: Optional[str] = None
    count: int
    articles: List[Dict[str, Any]]
    message: Optional[str] = None


class TopicSearchResponse(BaseModel):
    """Response model for topic search."""

    success: bool
    topic: str
    count: int
    articles: List[Dict[str, Any]]
    message: Optional[str] = None


class IssueInfo(BaseModel):
    """Response model for issue information."""

    issue_number: str
    article_count: int


class IssueStatsResponse(BaseModel):
    """Response model for issue statistics."""

    success: bool
    count: int
    total_articles: int
    issues: List[IssueInfo]
    message: Optional[str] = None


class VolumeInfo(BaseModel):
    """Response model for volume information."""

    id: int
    year: str
    issue_count: int
    cover_image_url: Optional[str] = None


class VolumeIssue(BaseModel):
    """Response model for volume issue."""

    issue_number: str
    year: str
    has_pdf: bool
    pdf_url: Optional[str] = None
    cover_image_url: Optional[str] = None


class PDFLinkResponse(BaseModel):
    """Response model for PDF link."""

    volume_id: int
    issue_id: str
    pdf_url: Optional[str] = None
    embed_url: Optional[str] = None
    found: bool


class TagInfo(BaseModel):
    """Response model for a tag/category."""

    id: str
    tamil: str
    english: str
    count: int


class TagsResponse(BaseModel):
    """Response model for listing all tags."""

    success: bool
    tags: List[TagInfo]


class TagArticleInfo(BaseModel):
    """Response model for an article associated with a tag."""

    doc_id: Optional[str] = None
    doc_issue: Optional[str] = None
    title: Optional[str] = None
    author_name: Optional[Union[str, List[str]]] = None
    year: Optional[str] = None
    tags: Optional[List[str]] = None


class TagArticlesResponse(BaseModel):
    """Response model for articles under a tag."""

    success: bool
    tag_id: str
    tag_tamil: str
    count: int
    articles: List[TagArticleInfo]


class IssueArticleInfo(BaseModel):
    """Response model for an article within a specific issue."""

    doc_id: Optional[str] = None
    doc_issue: Optional[str] = None
    article_no: Optional[str] = None
    title: Optional[str] = None
    author_name: Optional[Union[str, List[str]]] = None
    year: Optional[str] = None
    tags: Optional[List[str]] = None


class IssueArticlesResponse(BaseModel):
    """Response model for articles in a volume/issue."""

    success: bool
    volume_id: int
    issue_id: int
    count: int
    articles: List[IssueArticleInfo]


class ArticleContentResponse(BaseModel):
    """Response model for full article content."""

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
    """Initialize resources on startup and cleanup on shutdown."""
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


app = FastAPI(
    title="Ponni RAG API",
    description="REST API for the Ponni Tamil Literary Archive RAG System",
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
    """
    Check API, database, and LLM health status.

    Returns the health status of the API, Qdrant database, and Gemini LLM.
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
    """Debug S3 connectivity — check credentials and test image proxy."""
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
    """Return root endpoint with API information."""
    return {
        "name": "Ponni RAG API",
        "version": "1.0.0",
        "description": "REST API for Tamil Literary Archive",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api/images/volumes/{volume_id}/cover", tags=["Images"])
async def get_volume_cover(volume_id: int):
    """Proxy volume cover image from S3. Never expires."""
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
    """Proxy issue cover image from S3. Never expires."""
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
    """Proxy about page images from S3. Never expires."""
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


@app.post("/api/ask", response_model=QuestionResponse, tags=["Search"])
async def ask_question_endpoint(request: QuestionRequest):
    """
    Ask a question and get an AI-generated answer with sources.

    This endpoint performs hybrid search (dense + sparse vectors) and
    optionally uses an LLM to generate a comprehensive answer.
    Results are filtered by score threshold (>= 80% similarity).

    - **question**: The question to ask (in Tamil or English)
    - **use_llm**: Whether to use LLM for answer generation
    """
    try:
        logger.info(f"Question received: {request.question[:100]}...")

        history = [h.model_dump() for h in request.history] if request.history else None
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
    """
    Ask a question and get a streaming AI-generated answer via SSE.

    Streams tokens as Server-Sent Events:
    - event: token — individual answer tokens
    - event: sources — source documents (JSON array)
    - event: done — signals completion
    """
    history = [h.model_dump() for h in request.history] if request.history else None

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
    """
    Search the archive with a query string.

    GET alternative to POST /api/ask for simpler search queries.
    Results are filtered by score threshold (>= 80% similarity).
    Optionally filter by tags (comma-separated, e.g. tags=FICTION,POETRY).
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
    """
    List all 15 article categories with article counts.

    Scrolls the Qdrant collection to aggregate tag counts across all articles.
    Returns the full taxonomy with counts.
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
    """
    Get all articles for a specific tag/category.

    Scrolls Qdrant for articles matching the tag, deduplicates by doc_id + doc_issue.
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
                dedup_key = (metadata.get("doc_id"), metadata.get("doc_issue"))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                articles.append(
                    TagArticleInfo(
                        doc_id=metadata.get("doc_id"),
                        doc_issue=metadata.get("doc_issue"),
                        title=metadata.get("title"),
                        author_name=metadata.get("author_name"),
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
    """
    List all authors in the Ponni archive with article counts.

    Returns a sorted list of all unique authors and the number of
    articles each has written.
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
    """
    Get all articles written by a specific author.

    - **author_name**: Name of the author (supports Tamil names)

    Returns a list of articles with titles, years, and issue numbers.
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

    except Exception as e:
        logger.error(f"Error getting author articles: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/topics/search", response_model=TopicSearchResponse, tags=["Topics"])
async def search_by_topic(
    topic: str = Query(..., min_length=1, description="Topic to search for")
):
    """
    Find articles about a specific topic.

    - **topic**: Topic keyword to search in article titles

    Returns articles whose titles contain the topic keyword.
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

    except Exception as e:
        logger.error(f"Error searching topic: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/issues/stats", response_model=IssueStatsResponse, tags=["Issues"])
async def get_issue_statistics():
    """
    Get statistics about all issues in the archive.

    Returns the total number of issues, articles per issue,
    and summary statistics.
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

    except Exception as e:
        logger.error(f"Error getting issue stats: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


@app.get("/api/library/volumes", response_model=List[VolumeInfo], tags=["Library"])
async def list_volumes(request: Request):
    """
    List all volumes in the Ponni digital library.

    Returns volume information including year, issue count, and proxy
    cover image URL.
    """
    base = str(request.base_url).rstrip("/")
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
                cover_image_url=f"{base}/api/images/volumes/{vol['id']}/cover",
            )
        )
    return result


@app.get(
    "/api/library/volumes/{volume_id}/issues",
    response_model=List[VolumeIssue],
    tags=["Library"],
)
async def get_volume_issues(volume_id: int, request: Request):
    """
    Get all issues for a specific volume.

    - **volume_id**: Volume number (1-8)

    Returns list of issues with PDF availability and proxy cover image URL.
    """
    base = str(request.base_url).rstrip("/")
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
                    f"{base}/api/images/volumes/"
                    f"{volume_id}/issues/{issue_name}/cover"
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
async def get_pdf_link(volume_id: int, issue_id: str):
    """
    Get PDF link for a specific issue.

    - **volume_id**: Volume number (1-8)
    - **issue_id**: Issue number or name (e.g., "1", "PONGAL")

    Returns the Google Drive PDF URL and embeddable preview URL.
    """
    key = f"vol_{volume_id}_issue_{issue_id}"
    pdf_url = PDF_LINKS.get(key)

    if not pdf_url:
        return PDFLinkResponse(volume_id=volume_id, issue_id=issue_id, found=False)

    # Extract file ID for embed URL
    embed_url = None
    if "/d/" in pdf_url:
        file_id = pdf_url.split("/d/")[1].split("/")[0]
        embed_url = f"https://drive.google.com/file/d/{file_id}/preview"

    return PDFLinkResponse(
        volume_id=volume_id,
        issue_id=issue_id,
        pdf_url=pdf_url,
        embed_url=embed_url,
        found=True,
    )


@app.get(
    "/api/library/volumes/{volume_id}/issues/{issue_id}/articles",
    response_model=IssueArticlesResponse,
    tags=["Library"],
)
async def get_issue_articles(volume_id: int, issue_id: int):
    """
    Get all articles for a specific volume and issue.

    Replicates the Streamlit fetch_issue_articles logic:
    1. Query Qdrant for chunk_0 articles with matching doc_id (volume).
    2. Fallback: fetch all chunk_0 articles, filter client-side.
    3. Build position-based mapping from sorted doc_issue values.
    4. Filter by mapped doc_issue for the requested issue_id.
    5. Deduplicate by doc_id + doc_issue + article_no.
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
            if doc_issue_str != str(target_issue):
                continue
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
                    author_name=metadata.get("author_name"),
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
    """
    Get the full content of a specific article by concatenating all its chunks.

    Identifies the article by doc_id + doc_issue + optional article_no.
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
            author_name=first_meta.get("author_name"),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
