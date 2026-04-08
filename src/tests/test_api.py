"""Comprehensive tests for the FastAPI REST API (api.py)."""

import contextlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Shared mock data used across many tests
# ---------------------------------------------------------------------------

MOCK_MAGAZINE_CONFIG = {
    "s3": {
        "bucket": "test-bucket",
        "region": "us-east-1",
        "covers_prefix": "Front_cover_of_volumes/",
        "cover_folder_pattern": "volume {vol_id} cover images/",
        "cover_file_pattern": "VOL{vol_id} - {issue_num} - {year}.jpg",
    },
    "volumes": [
        {
            "id": 1,
            "year": "1947",
            "cover_image": "Volume1.jpg",
            "issues": [
                {
                    "num": 1,
                    "year": "1947",
                    "pdf_url": ("https://drive.google.com/file/d/ABC123/view"),
                },
                {
                    "num": 2,
                    "year": "1947",
                    "pdf_url": ("https://drive.google.com/file/d/DEF456/view"),
                },
                {
                    "num": "PONGAL",
                    "year": "1948",
                    "pdf_url": ("https://drive.google.com/file/d/GHI789/view"),
                },
            ],
        },
        {
            "id": 2,
            "year": "1948",
            "cover_image": "Volume2.jpg",
            "issues": [
                {
                    "num": 1,
                    "year": "1948",
                    "pdf_url": ("https://drive.google.com/file/d/JKL012/view"),
                },
            ],
        },
    ],
}

MOCK_TAXONOMY = {
    "FICTION": {"tamil": "புனைவு", "english": "Fiction/Serial"},
    "POETRY": {"tamil": "கவிதை", "english": "Poetry"},
}

# ---------------------------------------------------------------------------
# Patch heavy imports before importing api module
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Provide a FastAPI TestClient with lifespan deps mocked."""
    from fastapi.testclient import TestClient

    from db.api import app

    # Patch services called during lifespan startup
    _csv_mock = MagicMock()
    _csv_mock.exists.return_value = False
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                "db.api.check_qdrant_health",
                return_value={"healthy": True},
            )
        )
        stack.enter_context(
            patch(
                "db.api.check_gemini_health",
                return_value={"healthy": True},
            )
        )
        stack.enter_context(patch("db.api.CSV_PATH", _csv_mock))
        stack.enter_context(patch("db.api._author_system_cache", {}))
        stack.enter_context(patch("db.api._author_system_lock", MagicMock()))
        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc


# ===================================================================
# Health & root endpoints
# ===================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @patch("db.api.check_gemini_health")
    @patch("db.api.check_qdrant_health")
    def test_health_all_healthy(self, mock_qdrant, mock_gemini, client):
        """Return healthy when both services are up."""
        mock_qdrant.return_value = {"healthy": True, "points_count": 100}
        mock_gemini.return_value = {
            "healthy": True,
            "model": "gemini-2.5-flash",
            "latency_ms": 42,
        }
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["api"] == "healthy"

    @patch("db.api.check_gemini_health")
    @patch("db.api.check_qdrant_health")
    def test_health_degraded_db(self, mock_qdrant, mock_gemini, client):
        """Return degraded when database is unhealthy."""
        mock_qdrant.return_value = {
            "healthy": False,
            "message": "Connection refused",
        }
        mock_gemini.return_value = {"healthy": True, "model": "m", "latency_ms": 1}
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"

    @patch("db.api.check_gemini_health")
    @patch("db.api.check_qdrant_health")
    def test_health_degraded_llm(self, mock_qdrant, mock_gemini, client):
        """Return degraded when LLM is unhealthy."""
        mock_qdrant.return_value = {"healthy": True}
        mock_gemini.return_value = {"healthy": False, "message": "No key"}
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


class TestRootEndpoint:
    """Tests for the / root endpoint."""

    def test_root_returns_api_info(self, client):
        """Return API name and version from root."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Ponni RAG API"
        assert data["version"] == "1.0.0"


# ===================================================================
# /api/ask endpoint
# ===================================================================


class TestAskEndpoint:
    """Tests for POST /api/ask."""

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_ask_success(self, mock_ask, client):
        """Return answer and sources for a valid question."""
        mock_ask.return_value = {
            "answer": "Test answer",
            "sources": [{"content": "src"}],
            "query_type": "vector",
        }
        resp = client.post(
            "/api/ask",
            json={"question": "What is Ponni?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert len(data["sources"]) == 1

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_ask_with_tags_and_history(self, mock_ask, client):
        """Pass tags and history through to ask_question_async."""
        mock_ask.return_value = {"answer": "A", "sources": []}
        resp = client.post(
            "/api/ask",
            json={
                "question": "Q?",
                "tags": ["FICTION"],
                "history": [
                    {"role": "user", "content": "prev Q"},
                    {"role": "assistant", "content": "prev A"},
                ],
                "language": "en",
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_ask.call_args[1]
        assert call_kwargs["filter_tags"] == ["FICTION"]
        assert call_kwargs["language"] == "en"
        assert len(call_kwargs["history"]) == 2

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_ask_with_use_llm_false(self, mock_ask, client):
        """Disable LLM generation when use_llm is False."""
        mock_ask.return_value = {"answer": "", "sources": []}
        resp = client.post(
            "/api/ask",
            json={"question": "test", "use_llm": False},
        )
        assert resp.status_code == 200
        assert mock_ask.call_args[1]["use_llm"] is False

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_ask_internal_error(self, mock_ask, client):
        """Return 500 when ask_question_async raises."""
        mock_ask.side_effect = RuntimeError("boom")
        resp = client.post(
            "/api/ask",
            json={"question": "fail"},
        )
        assert resp.status_code == 500

    def test_ask_empty_question(self, client):
        """Reject empty question with 422."""
        resp = client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_missing_question(self, client):
        """Reject missing question field with 422."""
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 422

    def test_ask_invalid_language(self, client):
        """Reject invalid language code with 422."""
        resp = client.post(
            "/api/ask",
            json={"question": "hi", "language": "fr"},
        )
        assert resp.status_code == 422

    def test_ask_invalid_history_role(self, client):
        """Reject invalid history role with 422."""
        resp = client.post(
            "/api/ask",
            json={
                "question": "hi",
                "history": [{"role": "system", "content": "x"}],
            },
        )
        assert resp.status_code == 422


# ===================================================================
# /api/ask/stream endpoint
# ===================================================================


class TestAskStreamEndpoint:
    """Tests for POST /api/ask/stream (SSE)."""

    @patch("db.api.ask_question_stream")
    def test_stream_tokens(self, mock_stream, client):
        """Stream token and done events via SSE."""
        mock_stream.return_value = iter(
            [
                {"type": "token", "content": "Hello"},
                {"type": "token", "content": " world"},
                {
                    "type": "sources",
                    "sources": [{"content": "s"}],
                },
            ]
        )
        resp = client.post(
            "/api/ask/stream",
            json={"question": "test stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "event: token" in body
        assert "Hello" in body
        assert "event: sources" in body
        assert "event: done" in body

    @patch("db.api.ask_question_stream")
    def test_stream_fallback_event(self, mock_stream, client):
        """Stream fallback event when query cannot be answered."""
        mock_stream.return_value = iter([{"type": "fallback", "reason": "no results"}])
        resp = client.post(
            "/api/ask/stream",
            json={"question": "obscure"},
        )
        assert resp.status_code == 200
        assert "event: fallback" in resp.text

    @patch("db.api.ask_question_stream")
    def test_stream_error(self, mock_stream, client):
        """Stream error event when generator raises."""
        mock_stream.side_effect = RuntimeError("fail")
        resp = client.post(
            "/api/ask/stream",
            json={"question": "err"},
        )
        assert resp.status_code == 200
        assert "event: error" in resp.text

    @patch("db.api.ask_question_stream")
    def test_stream_with_history(self, mock_stream, client):
        """Pass history to streaming endpoint."""
        mock_stream.return_value = iter([])
        resp = client.post(
            "/api/ask/stream",
            json={
                "question": "q",
                "history": [
                    {"role": "user", "content": "old q"},
                    {"role": "assistant", "content": "old a"},
                ],
            },
        )
        assert resp.status_code == 200
        assert mock_stream.call_args[1]["history"] is not None


# ===================================================================
# /api/search endpoint
# ===================================================================


class TestSearchEndpoint:
    """Tests for GET /api/search."""

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_search_basic(self, mock_ask, client):
        """Return search results for a valid query."""
        mock_ask.return_value = {
            "answer": "found",
            "sources": [],
            "query_type": "vector",
        }
        resp = client.get("/api/search", params={"q": "ponni"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "found"
        assert mock_ask.call_args[1]["use_llm"] is False

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_search_with_tags(self, mock_ask, client):
        """Parse comma-separated tags parameter."""
        mock_ask.return_value = {"answer": "", "sources": []}
        resp = client.get(
            "/api/search",
            params={"q": "test", "tags": "FICTION,POETRY"},
        )
        assert resp.status_code == 200
        tags = mock_ask.call_args[1]["filter_tags"]
        assert tags == ["FICTION", "POETRY"]

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_search_with_llm(self, mock_ask, client):
        """Enable LLM via use_llm query param."""
        mock_ask.return_value = {"answer": "", "sources": []}
        resp = client.get(
            "/api/search",
            params={"q": "test", "use_llm": "true"},
        )
        assert resp.status_code == 200
        assert mock_ask.call_args[1]["use_llm"] is True

    def test_search_missing_query(self, client):
        """Reject missing query parameter with 422."""
        resp = client.get("/api/search")
        assert resp.status_code == 422

    @patch("db.api.ask_question_async", new_callable=AsyncMock)
    def test_search_internal_error(self, mock_ask, client):
        """Return 500 on internal error."""
        mock_ask.side_effect = RuntimeError("boom")
        resp = client.get("/api/search", params={"q": "x"})
        assert resp.status_code == 500


# ===================================================================
# /api/authors endpoints
# ===================================================================


class TestAuthorsEndpoint:
    """Tests for GET /api/authors."""

    @patch("db.api.CSV_PATH")
    @patch("db.api._author_system_cache", {})
    @patch("db.api._author_system_lock", MagicMock())
    def test_list_authors_success(self, mock_csv_path, client):
        """Return sorted author list with counts."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_system = MagicMock()
        mock_system.list_all_authors.return_value = {
            "success": True,
            "total_authors": 2,
            "total_articles": 10,
            "authors": [
                {"name": "Author A", "count": 6},
                {"name": "Author B", "count": 4},
            ],
        }
        with patch("db.api.EnhancedAuthorQuerySystem", return_value=mock_system):
            resp = client.get("/api/authors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_authors"] == 2

    @patch("db.api.CSV_PATH")
    def test_list_authors_csv_missing(self, mock_csv_path, client):
        """Return 404 when CSV data file is missing."""
        mock_csv_path.exists.return_value = False
        resp = client.get("/api/authors")
        assert resp.status_code == 404

    @patch("db.api.CSV_PATH")
    @patch("db.api._author_system_cache", {})
    @patch("db.api._author_system_lock", MagicMock())
    def test_list_authors_system_failure(self, mock_csv_path, client):
        """Return 500 when author system reports failure."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_system = MagicMock()
        mock_system.list_all_authors.return_value = {
            "success": False,
            "message": "Parse error",
        }
        with patch(
            "db.api.EnhancedAuthorQuerySystem",
            return_value=mock_system,
        ):
            resp = client.get("/api/authors")
        assert resp.status_code == 500


class TestAuthorArticlesEndpoint:
    """Tests for GET /api/authors/{name}/articles."""

    @patch("db.api.CSV_PATH")
    @patch("db.api._author_system_cache", {})
    @patch("db.api._author_system_lock", MagicMock())
    def test_get_author_articles_success(self, mock_csv_path, client):
        """Return articles for a known author."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_system = MagicMock()
        mock_system.get_topics_by_author.return_value = {
            "success": True,
            "author": "Test Author",
            "matched_author": "Test Author",
            "count": 1,
            "articles": [{"title": "Article 1"}],
        }
        with patch(
            "db.api.EnhancedAuthorQuerySystem",
            return_value=mock_system,
        ):
            resp = client.get("/api/authors/Test%20Author/articles")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    @patch("db.api.CSV_PATH")
    def test_get_author_articles_csv_missing(self, mock_csv_path, client):
        """Return 404 when CSV data file is missing."""
        mock_csv_path.exists.return_value = False
        resp = client.get("/api/authors/x/articles")
        assert resp.status_code == 404

    @patch("db.api.CSV_PATH")
    @patch("db.api._author_system_cache", {})
    @patch("db.api._author_system_lock", MagicMock())
    def test_get_author_articles_error(self, mock_csv_path, client):
        """Return 500 on unexpected error."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_system = MagicMock()
        mock_system.get_topics_by_author.side_effect = RuntimeError("x")
        with patch(
            "db.api.EnhancedAuthorQuerySystem",
            return_value=mock_system,
        ):
            resp = client.get("/api/authors/Author/articles")
        assert resp.status_code == 500


# ===================================================================
# /api/topics/search endpoint
# ===================================================================


class TestTopicsSearchEndpoint:
    """Tests for GET /api/topics/search."""

    @patch("db.api.CSV_PATH")
    @patch("db.api._author_system_cache", {})
    @patch("db.api._author_system_lock", MagicMock())
    def test_search_topic_success(self, mock_csv_path, client):
        """Return articles matching a topic keyword."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_system = MagicMock()
        mock_system.get_author_by_topic.return_value = {
            "success": True,
            "topic": "கவிதை",
            "count": 3,
            "articles": [{"title": "A"}],
        }
        with patch(
            "db.api.EnhancedAuthorQuerySystem",
            return_value=mock_system,
        ):
            resp = client.get(
                "/api/topics/search",
                params={"topic": "கவிதை"},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @patch("db.api.CSV_PATH")
    def test_search_topic_csv_missing(self, mock_csv_path, client):
        """Return 404 when CSV is missing."""
        mock_csv_path.exists.return_value = False
        resp = client.get("/api/topics/search", params={"topic": "x"})
        assert resp.status_code == 404

    @patch("db.api.CSV_PATH")
    @patch("db.api._author_system_cache", {})
    @patch("db.api._author_system_lock", MagicMock())
    def test_search_topic_error(self, mock_csv_path, client):
        """Return 500 on unexpected error."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_system = MagicMock()
        mock_system.get_author_by_topic.side_effect = RuntimeError("x")
        with patch(
            "db.api.EnhancedAuthorQuerySystem",
            return_value=mock_system,
        ):
            resp = client.get("/api/topics/search", params={"topic": "t"})
        assert resp.status_code == 500


# ===================================================================
# /api/tags endpoints
# ===================================================================


class TestTagsEndpoint:
    """Tests for GET /api/tags."""

    @patch("db.api.TAXONOMY", MOCK_TAXONOMY)
    @patch("db.api.get_qdrant_client")
    def test_list_tags_success(self, mock_get_client, client):
        """Return tags with aggregated counts from Qdrant."""
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {"metadata": {"tags": ["FICTION", "POETRY"]}}
        mock_client.scroll.return_value = ([mock_point], None)
        mock_get_client.return_value = mock_client
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["tags"]) == 2

    @patch("db.api.TAXONOMY", MOCK_TAXONOMY)
    @patch("db.api.get_qdrant_client")
    def test_list_tags_qdrant_error(self, mock_get_client, client):
        """Return tags with zero counts when Qdrant fails."""
        mock_get_client.side_effect = RuntimeError("down")
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        for t in data["tags"]:
            assert t["count"] == 0


class TestTagArticlesEndpoint:
    """Tests for GET /api/tags/{tag_id}/articles."""

    @patch("db.api.TAXONOMY", MOCK_TAXONOMY)
    @patch("db.api.get_qdrant_client")
    def test_get_tag_articles_success(self, mock_get_client, client):
        """Return articles for a valid tag ID."""
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {
            "metadata": {
                "doc_id": "1",
                "doc_issue": "1",
                "title": "Test",
                "author_name": "Author",
                "year": "1947",
                "tags": ["FICTION"],
            }
        }
        mock_client.scroll.return_value = ([mock_point], None)
        mock_get_client.return_value = mock_client
        resp = client.get("/api/tags/FICTION/articles")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["tag_id"] == "FICTION"
        assert data["count"] == 1

    @patch("db.api.TAXONOMY", MOCK_TAXONOMY)
    def test_get_tag_articles_unknown_tag(self, client):
        """Return 404 for unknown tag ID."""
        resp = client.get("/api/tags/UNKNOWN/articles")
        assert resp.status_code == 404

    @patch("db.api.TAXONOMY", MOCK_TAXONOMY)
    @patch("db.api.get_qdrant_client")
    def test_get_tag_articles_qdrant_error(self, mock_get_client, client):
        """Return 500 when Qdrant fails."""
        mock_get_client.side_effect = RuntimeError("down")
        resp = client.get("/api/tags/FICTION/articles")
        assert resp.status_code == 500

    @patch("db.api.TAXONOMY", MOCK_TAXONOMY)
    @patch("db.api.get_qdrant_client")
    def test_get_tag_articles_dedup(self, mock_get_client, client):
        """Deduplicate articles with same doc_id and doc_issue."""
        mock_client = MagicMock()
        dup_payload = {
            "metadata": {
                "doc_id": "1",
                "doc_issue": "1",
                "title": "T",
                "tags": ["FICTION"],
            }
        }
        p1 = MagicMock()
        p1.payload = dup_payload
        p2 = MagicMock()
        p2.payload = dup_payload
        mock_client.scroll.return_value = ([p1, p2], None)
        mock_get_client.return_value = mock_client
        resp = client.get("/api/tags/FICTION/articles")
        assert resp.json()["count"] == 1


# ===================================================================
# /api/library endpoints
# ===================================================================


class TestLibraryVolumesEndpoint:
    """Tests for GET /api/library/volumes."""

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_list_volumes(self, client):
        """Return all volumes with year and issue count."""
        resp = client.get("/api/library/volumes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[0]["issue_count"] == 3
        assert "/api/images/volumes/1/cover" in data[0]["cover_image_url"]

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_list_volumes_year_range(self, client):
        """Derive year range from per-issue years."""
        resp = client.get("/api/library/volumes")
        vol1 = resp.json()[0]
        # Volume 1 has issues in 1947 and 1948 (PONGAL)
        assert "1947" in vol1["year"]
        assert "1948" in vol1["year"]


class TestVolumeIssuesEndpoint:
    """Tests for GET /api/library/volumes/{id}/issues."""

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_get_volume_issues_success(self, client):
        """Return issues for a valid volume."""
        resp = client.get("/api/library/volumes/1/issues")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["issue_number"] == "1"
        assert data[0]["has_pdf"] is True
        assert "cover_image_url" in data[0]

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_get_volume_issues_not_found(self, client):
        """Return 404 for non-existent volume."""
        resp = client.get("/api/library/volumes/99/issues")
        assert resp.status_code == 404

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_get_volume_issues_pongal(self, client):
        """Include PONGAL as a valid string issue number."""
        resp = client.get("/api/library/volumes/1/issues")
        data = resp.json()
        pongal = [i for i in data if i["issue_number"] == "PONGAL"]
        assert len(pongal) == 1
        assert pongal[0]["year"] == "1948"


class TestPDFLinkEndpoint:
    """Tests for GET /api/library/volumes/{id}/issues/{issue}/pdf."""

    @patch(
        "db.api.PDF_LINKS",
        {"vol_1_issue_1": ("https://drive.google.com/file/d/ABC123/view")},
    )
    def test_pdf_link_found(self, client):
        """Return PDF and embed URLs for existing issue."""
        resp = client.get("/api/library/volumes/1/issues/1/pdf")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert "ABC123" in data["embed_url"]
        assert data["pdf_url"] is not None

    @patch("db.api.PDF_LINKS", {})
    def test_pdf_link_not_found(self, client):
        """Return found=False for missing PDF."""
        resp = client.get("/api/library/volumes/1/issues/99/pdf")
        data = resp.json()
        assert data["found"] is False
        assert data["pdf_url"] is None

    @patch(
        "db.api.PDF_LINKS",
        {"vol_1_issue_1": "https://example.com/plain-link"},
    )
    def test_pdf_link_no_embed(self, client):
        """Return None embed_url when URL has no /d/ segment."""
        resp = client.get("/api/library/volumes/1/issues/1/pdf")
        data = resp.json()
        assert data["found"] is True
        assert data["embed_url"] is None


# ===================================================================
# /api/library/volumes/{id}/issues/{issue}/articles
# ===================================================================


class TestIssueArticlesEndpoint:
    """Tests for GET /api/library/volumes/{id}/issues/{issue}/articles."""

    @patch("db.api.get_qdrant_client")
    def test_get_issue_articles_success(self, mock_get_client, client):
        """Return articles for a specific volume and issue."""
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {
            "metadata": {
                "doc_id": "1",
                "doc_issue": "3",
                "article_no": "1",
                "title": "Test Article",
                "author_name": "Author",
                "year": "1947",
                "tags": ["FICTION"],
            }
        }
        mock_client.scroll.return_value = ([mock_point], None)
        mock_get_client.return_value = mock_client
        resp = client.get("/api/library/volumes/1/issues/3/articles")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["count"] == 1

    @patch("db.api.get_qdrant_client")
    def test_get_issue_articles_empty(self, mock_get_client, client):
        """Return empty list when no articles found."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client
        resp = client.get("/api/library/volumes/1/issues/1/articles")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    @patch("db.api.get_qdrant_client")
    def test_get_issue_articles_fallback(self, mock_get_client, client):
        """Use fallback scroll when doc_id filter returns nothing."""
        mock_client = MagicMock()
        call_count = [0]

        def scroll_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                # First call (doc_id filtered) returns empty
                return ([], None)
            # Fallback call returns all articles
            p = MagicMock()
            p.payload = {
                "metadata": {
                    "doc_id": "1",
                    "doc_issue": "5",
                    "article_no": "1",
                    "title": "Fallback",
                    "tags": [],
                }
            }
            return ([p], None)

        mock_client.scroll.side_effect = scroll_side_effect
        mock_get_client.return_value = mock_client
        resp = client.get("/api/library/volumes/1/issues/5/articles")
        assert resp.status_code == 200

    @patch("db.api.get_qdrant_client")
    def test_get_issue_articles_error(self, mock_get_client, client):
        """Return 500 on Qdrant error."""
        mock_get_client.side_effect = RuntimeError("down")
        resp = client.get("/api/library/volumes/1/issues/1/articles")
        assert resp.status_code == 500


# ===================================================================
# /api/articles/content
# ===================================================================


class TestArticleContentEndpoint:
    """Tests for GET /api/articles/content."""

    @patch("db.api.get_qdrant_client")
    def test_get_article_content_success(self, mock_get_client, client):
        """Return concatenated chunks for a full article."""
        mock_client = MagicMock()
        p1 = MagicMock()
        p1.payload = {
            "content": "Chunk 0 text",
            "metadata": {
                "chunk_id": 0,
                "title": "Article Title",
                "author_name": "Author",
                "year": "1947",
                "doc_issue": "1",
                "tags": ["FICTION"],
            },
        }
        p2 = MagicMock()
        p2.payload = {
            "content": "Chunk 1 text",
            "metadata": {
                "chunk_id": 1,
                "title": "Article Title",
                "author_name": "Author",
                "year": "1947",
                "doc_issue": "1",
                "tags": ["FICTION"],
            },
        }
        mock_client.scroll.return_value = ([p1, p2], None)
        mock_get_client.return_value = mock_client
        resp = client.get(
            "/api/articles/content",
            params={"doc_id": "1", "doc_issue": "1", "article_no": "1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["chunk_count"] == 2
        assert "Chunk 0" in data["content"]
        assert "Chunk 1" in data["content"]

    @patch("db.api.get_qdrant_client")
    def test_get_article_content_empty(self, mock_get_client, client):
        """Return success with no content when article not found."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client
        resp = client.get(
            "/api/articles/content",
            params={"doc_id": "1", "doc_issue": "1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["content"] is None

    @patch("db.api.get_qdrant_client")
    def test_get_article_content_no_article_no(self, mock_get_client, client):
        """Omit article_no filter when not provided."""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client
        resp = client.get(
            "/api/articles/content",
            params={"doc_id": "1", "doc_issue": "1"},
        )
        assert resp.status_code == 200

    def test_get_article_content_missing_params(self, client):
        """Reject request missing required params with 422."""
        resp = client.get("/api/articles/content")
        assert resp.status_code == 422

    @patch("db.api.get_qdrant_client")
    def test_get_article_content_error(self, mock_get_client, client):
        """Return 500 on Qdrant error."""
        mock_get_client.side_effect = RuntimeError("down")
        resp = client.get(
            "/api/articles/content",
            params={"doc_id": "1", "doc_issue": "1"},
        )
        assert resp.status_code == 500


# ===================================================================
# Image proxy endpoints
# ===================================================================


class TestVolumeCoverEndpoint:
    """Tests for GET /api/images/volumes/{id}/cover."""

    @patch("db.api._fetch_s3_image")
    @patch("db.api._volume_cover_s3_key")
    def test_volume_cover_success(self, mock_key, mock_fetch, client):
        """Return image bytes with correct content type."""
        mock_key.return_value = "covers/Volume1.jpg"
        mock_fetch.return_value = {
            "body": b"\xff\xd8image-data",
            "content_type": "image/jpeg",
        }
        resp = client.get("/api/images/volumes/1/cover")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert "max-age=3600" in resp.headers["cache-control"]

    @patch("db.api._volume_cover_s3_key")
    def test_volume_cover_not_found(self, mock_key, client):
        """Return 404 when volume has no cover key."""
        mock_key.return_value = None
        resp = client.get("/api/images/volumes/99/cover")
        assert resp.status_code == 404

    @patch("db.api._fetch_s3_image")
    @patch("db.api._volume_cover_s3_key")
    def test_volume_cover_s3_missing(self, mock_key, mock_fetch, client):
        """Return 404 when S3 image fetch returns None."""
        mock_key.return_value = "covers/Missing.jpg"
        mock_fetch.return_value = None
        resp = client.get("/api/images/volumes/1/cover")
        assert resp.status_code == 404


class TestIssueCoverEndpoint:
    """Tests for GET /api/images/volumes/{id}/issues/{issue}/cover."""

    @patch("db.api._fetch_s3_image")
    @patch("db.api._issue_cover_s3_key")
    def test_issue_cover_success(self, mock_key, mock_fetch, client):
        """Return issue cover image from S3."""
        mock_key.return_value = "covers/vol1/issue1.jpg"
        mock_fetch.return_value = {
            "body": b"png-data",
            "content_type": "image/png",
        }
        resp = client.get("/api/images/volumes/1/issues/1/cover")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    @patch("db.api._issue_cover_s3_key")
    def test_issue_cover_not_found(self, mock_key, client):
        """Return 404 when issue key is None."""
        mock_key.return_value = None
        resp = client.get("/api/images/volumes/1/issues/99/cover")
        assert resp.status_code == 404


class TestAboutImageEndpoint:
    """Tests for GET /api/images/about/{filename}."""

    @patch("db.api._fetch_s3_image")
    def test_about_image_success(self, mock_fetch, client):
        """Return about page image from S3."""
        mock_fetch.return_value = {
            "body": b"img",
            "content_type": "image/png",
        }
        resp = client.get("/api/images/about/about1.png")
        assert resp.status_code == 200

    def test_about_image_disallowed_filename(self, client):
        """Reject filenames not in the allow list."""
        resp = client.get("/api/images/about/../../etc/passwd")
        assert resp.status_code == 404

    def test_about_image_unknown_filename(self, client):
        """Reject unknown about image filenames."""
        resp = client.get("/api/images/about/notallowed.txt")
        assert resp.status_code == 404

    @patch("db.api._fetch_s3_image")
    def test_about_image_s3_missing(self, mock_fetch, client):
        """Return 404 when S3 fetch fails."""
        mock_fetch.return_value = None
        resp = client.get("/api/images/about/about1.jpg")
        assert resp.status_code == 404


# ===================================================================
# Cache endpoints
# ===================================================================


class TestCacheEndpoints:
    """Tests for /api/cache/stats and /api/cache/clear."""

    @patch("db.api._response_cache")
    def test_cache_stats(self, mock_cache, client):
        """Return cache statistics."""
        mock_cache.stats.return_value = {
            "size": 5,
            "hits": 10,
            "misses": 3,
        }
        resp = client.get("/api/cache/stats")
        assert resp.status_code == 200
        assert resp.json()["size"] == 5

    @patch("db.api._response_cache")
    def test_cache_clear(self, mock_cache, client):
        """Clear cache and return confirmation."""
        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Cache cleared"
        mock_cache.clear.assert_called_once()


# ===================================================================
# /api/issues/stats endpoint
# ===================================================================


class TestIssueStatsEndpoint:
    """Tests for GET /api/issues/stats."""

    @patch("db.api.get_issue_count")
    @patch("db.api.CSV_PATH")
    def test_issue_stats_success(self, mock_csv_path, mock_get_count, client):
        """Return issue statistics from CSV."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_get_count.return_value = {
            "success": True,
            "count": 2,
            "total_articles": 20,
            "issues": [
                {"issue_number": "1", "article_count": 10},
                {"issue_number": "2", "article_count": 10},
            ],
        }
        resp = client.get("/api/issues/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    @patch("db.api.CSV_PATH")
    def test_issue_stats_csv_missing(self, mock_csv_path, client):
        """Return 404 when CSV is missing."""
        mock_csv_path.exists.return_value = False
        resp = client.get("/api/issues/stats")
        assert resp.status_code == 404

    @patch("db.api.get_issue_count")
    @patch("db.api.CSV_PATH")
    def test_issue_stats_error(self, mock_csv_path, mock_get_count, client):
        """Return 500 on unexpected error."""
        mock_csv_path.exists.return_value = True
        mock_csv_path.__str__ = Mock(return_value="/tmp/test.csv")
        mock_get_count.side_effect = RuntimeError("boom")
        resp = client.get("/api/issues/stats")
        assert resp.status_code == 500


# ===================================================================
# Helper function unit tests
# ===================================================================


class TestS3KeyHelpers:
    """Tests for S3 key derivation helper functions."""

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    @patch("db.api._s3_conf", MOCK_MAGAZINE_CONFIG["s3"])
    def test_volume_cover_s3_key_found(self):
        """Return S3 key for existing volume."""
        from db.api import _volume_cover_s3_key

        key = _volume_cover_s3_key(1)
        assert key == "Front_cover_of_volumes/Volumes/Volume1.jpg"

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_volume_cover_s3_key_not_found(self):
        """Return None for non-existent volume."""
        from db.api import _volume_cover_s3_key

        key = _volume_cover_s3_key(999)
        assert key is None

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    @patch("db.api._s3_conf", MOCK_MAGAZINE_CONFIG["s3"])
    def test_issue_cover_s3_key_found(self):
        """Return S3 key for existing issue."""
        from db.api import _issue_cover_s3_key

        key = _issue_cover_s3_key(1, "1")
        assert "VOL1" in key
        assert "1947" in key

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_issue_cover_s3_key_vol_not_found(self):
        """Return None when volume does not exist."""
        from db.api import _issue_cover_s3_key

        key = _issue_cover_s3_key(999, "1")
        assert key is None

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    def test_issue_cover_s3_key_issue_not_found(self):
        """Return None when issue does not exist in volume."""
        from db.api import _issue_cover_s3_key

        key = _issue_cover_s3_key(1, "999")
        assert key is None

    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    @patch("db.api._s3_conf", MOCK_MAGAZINE_CONFIG["s3"])
    def test_issue_cover_s3_key_pongal(self):
        """Return correct S3 key for PONGAL issue."""
        from db.api import _issue_cover_s3_key

        key = _issue_cover_s3_key(1, "PONGAL")
        assert key is not None
        assert "PONGAL" in key
        assert "1948" in key


class TestS3KeyWithFallback:
    """Tests for _s3_key_with_fallback helper."""

    def test_key_with_jpg(self):
        """Generate fallback keys for jpg file."""
        from db.api import _s3_key_with_fallback

        keys = _s3_key_with_fallback("path/image.jpg")
        assert keys[0] == "path/image.jpg"
        assert "path/image.png" in keys
        assert "path/image.jpeg" in keys

    def test_key_without_extension(self):
        """Return single key when no extension present."""
        from db.api import _s3_key_with_fallback

        keys = _s3_key_with_fallback("path/noext")
        assert keys == ["path/noext"]

    def test_key_with_png(self):
        """Generate fallback keys for png file."""
        from db.api import _s3_key_with_fallback

        keys = _s3_key_with_fallback("img.png")
        assert keys[0] == "img.png"
        assert "img.jpg" in keys
        assert "img.jpeg" in keys
        assert len(keys) == 3


class TestFetchS3Image:
    """Tests for _fetch_s3_image helper."""

    @patch("db.api._s3_client")
    @patch("db.api._s3_conf", {"bucket": "test-bucket"})
    def test_fetch_success(self, mock_s3):
        """Return image body and content type on success."""
        from db.api import _fetch_s3_image

        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=Mock(return_value=b"img")),
            "ContentType": "image/jpeg",
        }
        result = _fetch_s3_image("key.jpg")
        assert result is not None
        assert result["body"] == b"img"
        assert result["content_type"] == "image/jpeg"

    @patch("db.api._s3_client")
    @patch("db.api._s3_conf", {"bucket": "test-bucket"})
    def test_fetch_not_found_tries_fallback(self, mock_s3):
        """Try alternate extensions when NoSuchKey error occurs."""
        from botocore.exceptions import ClientError

        from db.api import _fetch_s3_image

        error_resp = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
        mock_s3.get_object.side_effect = ClientError(error_resp, "GetObject")
        result = _fetch_s3_image("key.jpg")
        assert result is None
        # Should have tried jpg, png, jpeg
        assert mock_s3.get_object.call_count == 3

    @patch("db.api._s3_client")
    @patch("db.api._s3_conf", {"bucket": "test-bucket"})
    def test_fetch_other_client_error(self, mock_s3):
        """Return None on non-NoSuchKey ClientError."""
        from botocore.exceptions import ClientError

        from db.api import _fetch_s3_image

        error_resp = {"Error": {"Code": "AccessDenied", "Message": "Denied"}}
        mock_s3.get_object.side_effect = ClientError(error_resp, "GetObject")
        result = _fetch_s3_image("key.jpg")
        assert result is None

    @patch("db.api._s3_client")
    @patch("db.api._s3_conf", {"bucket": "test-bucket"})
    def test_fetch_no_credentials(self, mock_s3):
        """Return None when AWS credentials are missing."""
        from botocore.exceptions import NoCredentialsError

        from db.api import _fetch_s3_image

        mock_s3.get_object.side_effect = NoCredentialsError()
        result = _fetch_s3_image("key.jpg")
        assert result is None


# ===================================================================
# Pydantic model validation
# ===================================================================


class TestPydanticModels:
    """Tests for request/response Pydantic models."""

    def test_question_request_defaults(self):
        """Validate QuestionRequest default values."""
        from db.api import QuestionRequest

        req = QuestionRequest(question="test")
        assert req.use_llm is True
        assert req.tags is None
        assert req.history is None
        assert req.language == "ta"

    def test_history_message_valid(self):
        """Validate HistoryMessage with valid role."""
        from db.api import HistoryMessage

        msg = HistoryMessage(role="user", content="hello")
        assert msg.role == "user"

    def test_history_message_long_content_accepted(self):
        """Accept any content length (truncation happens at endpoint level)."""
        from db.api import HistoryMessage

        long_content = "அ" * 30000
        msg = HistoryMessage(role="assistant", content=long_content)
        assert len(msg.content) == 30000

    def test_truncate_history_long_content(self):
        """Truncate history entries exceeding max content length."""
        from db.api import _MAX_HISTORY_CONTENT, QuestionRequest, _truncate_history

        long_content = "அ" * (_MAX_HISTORY_CONTENT + 5000)
        req = QuestionRequest(
            question="test",
            history=[
                {"role": "user", "content": "short"},
                {"role": "assistant", "content": long_content},
            ],
        )
        result = _truncate_history(req)
        assert result[0]["content"] == "short"
        assert len(result[1]["content"]) == _MAX_HISTORY_CONTENT

    def test_truncate_history_none(self):
        """Return None when request has no history."""
        from db.api import QuestionRequest, _truncate_history

        req = QuestionRequest(question="test")
        assert _truncate_history(req) is None

    def test_history_message_invalid_role(self):
        """Reject invalid role in HistoryMessage."""
        from pydantic import ValidationError

        from db.api import HistoryMessage

        with pytest.raises(ValidationError):
            HistoryMessage(role="system", content="hello")

    def test_question_request_invalid_language(self):
        """Reject invalid language in QuestionRequest."""
        from pydantic import ValidationError

        from db.api import QuestionRequest

        with pytest.raises(ValidationError):
            QuestionRequest(question="test", language="de")


# ===================================================================
# Debug endpoint
# ===================================================================


class TestDebugS3Endpoint:
    """Tests for GET /api/debug/s3."""

    @patch("db.api._s3_client")
    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    @patch("db.api._s3_conf", MOCK_MAGAZINE_CONFIG["s3"])
    def test_debug_s3_success(self, mock_s3, client):
        """Return S3 debug info with sample URLs."""
        mock_s3.head_object.return_value = {"ContentLength": 12345}
        resp = client.get("/api/debug/s3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["credentials"] is True
        assert "sample_volume_url" in data

    @patch("db.api._s3_client")
    @patch("db.api._magazine", MOCK_MAGAZINE_CONFIG)
    @patch("db.api._s3_conf", MOCK_MAGAZINE_CONFIG["s3"])
    def test_debug_s3_no_credentials(self, mock_s3, client):
        """Return credentials=False when S3 auth fails."""
        from botocore.exceptions import NoCredentialsError

        mock_s3.head_object.side_effect = NoCredentialsError()
        resp = client.get("/api/debug/s3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object_exists"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
