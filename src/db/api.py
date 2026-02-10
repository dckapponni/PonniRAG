"""
FastAPI REST API for Ponni RAG System.
Provides endpoints for search, question answering, and library access.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import real hybrid_search module
from hybrid_search import (
    ask_question,
    check_qdrant_health,
    EnhancedAuthorQuerySystem,
    get_issue_count,
    CSV_PATH,
)

from pdf_links import PDF_LINKS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, description="The question to ask")
    use_llm: bool = Field(default=True, description="Use LLM for answer generation")


class SourceDocument(BaseModel):
    """Response model for source documents."""
    volume: Optional[str] = None
    heading: Optional[str] = None
    doc_issue: Optional[str] = None
    content: str
    word_count: Optional[int] = None
    chunks_merged: Optional[int] = None
    score: Optional[float] = None


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    sources: List[Dict[str, Any]] = []
    query_type: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    database: Dict[str, Any]
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API and database health status.

    Returns the health status of the API and Qdrant database connection.
    """
    db_health = check_qdrant_health()

    return HealthResponse(
        status="healthy" if db_health["healthy"] else "degraded",
        database=db_health,
        api="healthy"
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

        result = ask_question(
            question=request.question,
            return_formatted=False,
            use_llm=request.use_llm
        )

        return QuestionResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            query_type=result.get("query_type"),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search", response_model=QuestionResponse, tags=["Search"])
async def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    use_llm: bool = Query(default=False, description="Use LLM for answer")
):
    """
    Search the archive with a query string.

    GET alternative to POST /api/ask for simpler search queries.
    Results are filtered by score threshold (>= 80% similarity).
    """
    try:
        result = ask_question(
            question=q,
            return_formatted=False,
            use_llm=use_llm
        )

        return QuestionResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            query_type=result.get("query_type"),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

        system = EnhancedAuthorQuerySystem(str(CSV_PATH))
        result = system.list_all_authors()

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
        raise HTTPException(status_code=500, detail=str(e))


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

        system = EnhancedAuthorQuerySystem(str(CSV_PATH))
        result = system.get_topics_by_author(author_name)

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
        raise HTTPException(status_code=500, detail=str(e))


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

        system = EnhancedAuthorQuerySystem(str(CSV_PATH))
        result = system.get_author_by_topic(topic)

        return TopicSearchResponse(
            success=result["success"],
            topic=result.get("topic", topic),
            count=result.get("count", 0),
            articles=result.get("articles", []),
            message=result.get("message")
        )

    except Exception as e:
        logger.error(f"Error searching topic: {e}")
        raise HTTPException(status_code=500, detail=str(e))



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

        result = get_issue_count(str(CSV_PATH))

        return IssueStatsResponse(
            success=result["success"],
            count=result.get("count", 0),
            total_articles=result.get("total_articles", 0),
            issues=[IssueInfo(**i) for i in result.get("issues", [])],
            message=result.get("message")
        )

    except Exception as e:
        logger.error(f"Error getting issue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )