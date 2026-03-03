"""
FastAPI REST API for Ponni RAG System.
Provides endpoints for search, question answering, and library access.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import real hybrid_search module
from qdrant_client import models

from hybrid_search import ask_question_async, ask_question_stream
from embeddings import check_qdrant_health, get_qdrant_client, COLLECTION_NAME, CSV_PATH
from csv_queries import EnhancedAuthorQuerySystem, get_issue_count, _author_system_cache, _author_system_lock
from llm import validate_gemini_api, check_gemini_health
from cache import _response_cache

from article_tagger import TAXONOMY

from pdf_links import PDF_LINKS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryMessage(BaseModel):
    """A single conversation turn (user or assistant)."""
    role: str = Field(..., pattern=r'^(user|assistant)$')
    content: str = Field(..., min_length=1, max_length=5000)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    use_llm: bool = Field(default=True, description="Use LLM for answer generation")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tag IDs")
    history: Optional[List[HistoryMessage]] = Field(default=None, description="Previous Q&A turns for context")


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


class VolumeIssue(BaseModel):
    """Response model for volume issue."""
    issue_number: int
    has_pdf: bool
    pdf_url: Optional[str] = None


class PDFLinkResponse(BaseModel):
    """Response model for PDF link."""
    volume_id: int
    issue_id: int
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
    """Response model for an article under a tag."""
    doc_id: Optional[str] = None
    doc_issue: Optional[str] = None
    title: Optional[str] = None
    author_name: Optional[str] = None
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
    author_name: Optional[str] = None
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
    author_name: Optional[str] = None
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
        logger.warning(f"Qdrant not available: {health.get('message', 'Unknown error')}")

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
        logger.info(f"Gemini API validated (model: {gemini_health['model']}, latency: {gemini_health['latency_ms']}ms)")
    else:
        logger.warning(f"Gemini API not available: {gemini_health['message']}")

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


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Ponni RAG API",
        "version": "1.0.0",
        "description": "REST API for Tamil Literary Archive",
        "docs": "/docs",
        "health": "/health"
    }

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
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


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
        try:
            for event in ask_question_stream(
                question=request.question,
                filter_tags=request.tags,
                history=history,
            ):
                if event["type"] == "token":
                    yield f"event: token\ndata: {json.dumps({'content': event['content']})}\n\n"
                elif event["type"] == "fallback":
                    yield f"event: fallback\ndata: {json.dumps({'reason': event['reason']})}\n\n"
                elif event["type"] == "sources":
                    yield f"event: sources\ndata: {json.dumps({'sources': event['sources']})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'An internal error occurred. Please try again.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
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
    tags: Optional[str] = Query(default=None, description="Comma-separated tag IDs to filter by"),
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
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


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
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="article")),
                    models.FieldCondition(key="metadata.chunk_id", match=models.MatchValue(value=0)),
                ]),
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
            tags_list.append(TagInfo(
                id=cat_id,
                tamil=cat_info["tamil"],
                english=cat_info["english"],
                count=tag_counts.get(cat_id, 0),
            ))

        # Sort by count descending
        tags_list.sort(key=lambda t: t.count, reverse=True)

        return TagsResponse(success=True, tags=tags_list)

    except Exception as e:
        logger.error(f"Error listing tags: {e}")
        # Return taxonomy with zero counts instead of 500 error —
        # lets the dropdown populate even if Qdrant is unavailable.
        tags_list = [
            TagInfo(id=cat_id, tamil=cat_info["tamil"], english=cat_info["english"], count=0)
            for cat_id, cat_info in TAXONOMY.items()
        ]
        return TagsResponse(success=True, tags=tags_list)


@app.get("/api/tags/{tag_id}/articles", response_model=TagArticlesResponse, tags=["Tags"])
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
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="article")),
                    models.FieldCondition(key="metadata.tags", match=models.MatchAny(any=[tag_id])),
                    models.FieldCondition(key="metadata.chunk_id", match=models.MatchValue(value=0)),
                ]),
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
                articles.append(TagArticleInfo(
                    doc_id=metadata.get("doc_id"),
                    doc_issue=metadata.get("doc_issue"),
                    title=metadata.get("title"),
                    author_name=metadata.get("author_name"),
                    year=metadata.get("year"),
                    tags=metadata.get("tags", []),
                ))
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
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


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
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to load authors"))

        return AuthorsListResponse(
            success=True,
            total_authors=result["total_authors"],
            total_articles=result["total_articles"],
            authors=[AuthorInfo(name=a["name"], count=a["count"]) for a in result["authors"]]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing authors: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@app.get("/api/authors/{author_name}/articles", response_model=AuthorArticlesResponse, tags=["Authors"])
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
            message=result.get("message")
        )

    except Exception as e:
        logger.error(f"Error getting author articles: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


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
            message=result.get("message")
        )

    except Exception as e:
        logger.error(f"Error searching topic: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")



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
            message=result.get("message")
        )

    except Exception as e:
        logger.error(f"Error getting issue stats: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@app.get("/api/library/volumes", response_model=List[VolumeInfo], tags=["Library"])
async def list_volumes():
    """
    List all volumes in the Ponni digital library.

    Returns volume information including year and issue count.
    """
    # Volume metadata
    volumes_data = [
        {"id": 1, "year": "1947", "issues": 7},
        {"id": 2, "year": "1948", "issues": 19},
        {"id": 3, "year": "1949", "issues": 21},
        {"id": 4, "year": "1950", "issues": 9},
        {"id": 5, "year": "1951", "issues": 21},
        {"id": 6, "year": "1952", "issues": 18},
        {"id": 7, "year": "1953", "issues": 1},
        {"id": 8, "year": "1954", "issues": 10},
    ]

    # Count actual PDF links per volume
    result = []
    for vol in volumes_data:
        vol_id = vol["id"]
        issue_count = sum(1 for k in PDF_LINKS.keys() if k.startswith(f"vol_{vol_id}_"))
        result.append(VolumeInfo(
            id=vol_id,
            year=vol["year"],
            issue_count=issue_count
        ))

    return result


@app.get("/api/library/volumes/{volume_id}/issues", response_model=List[VolumeIssue], tags=["Library"])
async def get_volume_issues(volume_id: int):
    """
    Get all issues for a specific volume.

    - **volume_id**: Volume number (1-8)

    Returns list of issues with PDF availability status.
    """
    if volume_id < 1 or volume_id > 8:
        raise HTTPException(status_code=404, detail="Volume not found")

    # Find all issues for this volume from PDF_LINKS
    issues = []
    prefix = f"vol_{volume_id}_issue_"

    for key in sorted(PDF_LINKS.keys()):
        if key.startswith(prefix):
            issue_num = int(key.replace(prefix, ""))
            issues.append(VolumeIssue(
                issue_number=issue_num,
                has_pdf=True,
                pdf_url=PDF_LINKS[key]
            ))

    if not issues:
        raise HTTPException(status_code=404, detail="No issues found for this volume")

    return issues


@app.get("/api/library/volumes/{volume_id}/issues/{issue_id}/pdf", response_model=PDFLinkResponse, tags=["Library"])
async def get_pdf_link(volume_id: int, issue_id: int):
    """
    Get PDF link for a specific issue.

    - **volume_id**: Volume number (1-8)
    - **issue_id**: Issue number within the volume

    Returns the Google Drive PDF URL and embeddable preview URL.
    """
    key = f"vol_{volume_id}_issue_{issue_id}"
    pdf_url = PDF_LINKS.get(key)

    if not pdf_url:
        return PDFLinkResponse(
            volume_id=volume_id,
            issue_id=issue_id,
            found=False
        )

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
        found=True
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
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="article")),
                    models.FieldCondition(key="metadata.chunk_id", match=models.MatchValue(value=0)),
                    models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=vol_str)),
                ]),
                limit=500,
                offset=scroll_offset,
                with_payload=True,
            )
            all_points.extend(points)
            if scroll_offset is None:
                break

        logger.info(f"get_issue_articles: vol={volume_id}, doc_id filter → {len(all_points)} points")

        # Fallback: if doc_id filter found nothing, fetch all chunk_0 and filter client-side
        if not all_points:
            logger.info("No points with doc_id filter — fetching all chunk_0 articles")
            scroll_offset = None
            while True:
                points, scroll_offset = await asyncio.to_thread(
                    client.scroll,
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value="article")),
                        models.FieldCondition(key="metadata.chunk_id", match=models.MatchValue(value=0)),
                    ]),
                    limit=500,
                    offset=scroll_offset,
                    with_payload=True,
                )
                all_points.extend(points)
                if scroll_offset is None:
                    break
            all_points = [
                p for p in all_points
                if str((p.payload or {}).get("metadata", {}).get("doc_id", "")) == vol_str
            ]
            logger.info(f"After client-side doc_id filter: {len(all_points)}")

        # Collect and sort unique doc_issue values
        all_issue_vals = set()
        for p in all_points:
            meta = (p.payload or {}).get("metadata", {})
            raw_val = meta.get("doc_issue")
            val = str(raw_val).strip() if raw_val is not None else "NA"
            all_issue_vals.add(val)
        sorted_issues = sorted(
            [v for v in all_issue_vals if v not in ("NA", "None", "")],
            key=lambda x: int(x) if x.isdigit() else float("inf"),
        )

        # Position-based mapping: sidebar position (1-based) → actual doc_issue
        issue_position_map = {i + 1: v for i, v in enumerate(sorted_issues)}
        target_issue = issue_position_map.get(issue_id, str(issue_id))
        logger.info(f"Issue mapping: position {issue_id} → doc_issue '{target_issue}'")

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
            articles.append(IssueArticleInfo(
                doc_id=doc_id,
                doc_issue=doc_issue_str,
                article_no=article_no,
                title=metadata.get("title"),
                author_name=metadata.get("author_name"),
                year=metadata.get("year"),
                tags=metadata.get("tags", []),
            ))

        return IssueArticlesResponse(
            success=True,
            volume_id=volume_id,
            issue_id=issue_id,
            count=len(articles),
            articles=articles,
        )

    except Exception as e:
        logger.error(f"Error fetching issue articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@app.get("/api/articles/content", response_model=ArticleContentResponse, tags=["Library"])
async def get_article_content(
    doc_id: str = Query(..., description="Document/volume ID"),
    doc_issue: str = Query(..., description="Issue number"),
    article_no: Optional[str] = Query(default=None, description="Article number within the issue"),
):
    """
    Get the full content of a specific article by concatenating all its chunks.

    Identifies the article by doc_id + doc_issue + optional article_no.
    """
    try:
        client = get_qdrant_client()

        filter_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="article")),
            models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=str(doc_id))),
            models.FieldCondition(key="metadata.doc_issue", match=models.MatchValue(value=str(doc_issue))),
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
                chunks.append({
                    "chunk_id": metadata.get("chunk_id", 0),
                    "content": payload.get("content", ""),
                    "metadata": metadata,
                })
            if scroll_offset is None:
                break

        logger.info(f"get_article_content: doc_id={doc_id}, doc_issue={doc_issue}, "
                     f"article_no={article_no}, chunks={len(chunks)}")

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
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )