# Ponni RAG

**Ponni RAG** is an intelligent Retrieval-Augmented Generation system designed for Tamil literary documents. It uses a hybrid search approach that combines semantic vector search and keyword-based retrieval to deliver accurate, context-aware results from large collections of Tamil PDF and DOCX files. The system extracts and indexes individual literary articles while preserving author and structural metadata. Every query — whether it matches structured CSV data (authors, topics, issue counts) or requires vector search — is routed through Google's Gemini 2.5 Flash API with context-specific prompts to generate natural Tamil responses grounded strictly in the Ponni dataset. Additionally, the system provides an option to view the original PDF content of each Ponni article volume directly.

## Architecture Diagram:
![solution flow](assets/image.png)
## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
  - [Project Index](#project-index)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Docker Deployment](#docker-deployment)
  - [Services Architecture](#services-architecture)
  - [Quick Start with Docker Compose](#quick-start-with-docker-compose)
  - [Docker Commands](#docker-commands)
  - [Service Endpoints](#service-endpoints)
- [React Frontend](#react-frontend)
  - [Frontend Setup](#frontend-setup)
  - [Frontend Features](#frontend-features)
- [Mock Data Mode](#mock-data-mode)
  - [Enabling Mock Mode](#enabling-mock-mode)
  - [Mock Data Features](#mock-data-features)

---

## Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Hybrid search system combining semantic vector search and keyword-based retrieval (`hybrid_search.py`)</li><li>Dual-prompt LLM pipeline: lightweight CSV summarization prompt for structured queries, full document prompt with Ponni context for vector search queries</li><li>Utilizes Qdrant vector database for efficient similarity search and document retrieval (`qdrant_indexer.py`)</li><li>AWS S3 integration for scalable document storage and retrieval (`s3_utils.py`)</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Modular design with separate modules for extraction (`text_extraction.py`), article separation (`article_seperation.py`), and search (`hybrid_search.py`)</li><li>Centralized configuration settings in `config/config.py` for consistency and easy modification</li><li>Comprehensive test coverage with unit and integration tests</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with `AWS S3` for efficient storage and retrieval of documents and processed data</li><li>Utilizes `Qdrant` vector database for semantic search and similarity matching</li><li>Streamlit-based interactive UI for querying and visualization (`streamlit_app.py`)</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Separate modules for text extraction (`text_extraction.py`), content processing (`content_extraction.py`), and text processing (`text_processing.py`)</li><li>Article separation and pattern matching encapsulated in `article_seperation.py` and `article_patterns.py`</li><li>Configuration settings isolated in `config/config.py`</li><li>Comprehensive test suites for quality assessment in `tests/`</li></ul> |

---

## Project Structure

```
├── frontend/                    # React.js Frontend
│   ├── src/
│   │   ├── components/          # Navigation, ChatMessage, ChatInput
│   │   ├── pages/               # Home, Library, Issues, PDFViewer, About
│   │   ├── services/            # API client, translations
│   │   └── styles/              # CSS styles
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── src/
│   ├── config/
│   │   └── config.py
│   ├── data_extraction/
│   │   ├── article_patterns.py
│   │   ├── article_seperation.py
│   │   ├── content_extraction.py
│   │   ├── doc_utils.py
│   │   ├── s3_utils.py
│   │   ├── shared_author.py
│   │   ├── text_extraction.py
│   │   └── text_processing.py
│   ├── db/
│   │   ├── api.py               # FastAPI REST API
│   │   ├── hybrid_search.py
│   │   ├── mock_search.py       # Mock data for testing
│   │   ├── pdf_links.py
│   │   ├── qdrant_indexer.py
│   │   ├── streamlit_app.py
│   │   └── summary.csv
│   ├── tests/
│   └── evaluation/
│       ├── evaluate.py          # Evaluation pipeline & CLI
│       ├── metrics.py           # Semantic, BLEU, ROUGE, BERTScore
│       ├── dataset.py           # Dataset loading (JSON/CSV)
│       └── ground_truth_template.csv
├── docker-compose.yml           # Multi-service orchestration
├── Dockerfile                   # Streamlit container
├── Dockerfile.api               # FastAPI container
└── requirements.txt
```
###  Project Index
<details open>
    <summary><b><code>/</code></b></summary>
    <details> 
        <summary><b>config</b></summary>
        <blockquote>
            <table>
            <tr>
                <td><b><a href='config/config.py'>config.py</a></b></td>
                <td>- Defines configuration settings and parameters for the Ponni RAG system<br>- Specifies S3 storage locations, Qdrant connection details, embedding model configurations, and API credentials<br>- Centralizes key variables to ensure consistency across the codebase and enables easy modification of project-wide settings.</td>
            </tr>
            </table>
        </blockquote>
    </details>
    <details> 
        <summary><b>data_extraction</b></summary>
        <blockquote>
            <table>
            <tr>
                <td><b><a href='data_extraction/article_patterns.py'>article_patterns.py</a></b></td>
                <td>- Pattern recognition module that defines regex patterns and heuristics for identifying article boundaries, titles, authors, and structural elements<br>- Enables accurate detection of article starts, section markers, and metadata within Tamil literary documents<br>- Provides configurable patterns for robust document segmentation.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/article_seperation.py'>article_seperation.py</a></b></td>
                <td>- Article separation engine that splits multi-article documents into individual articles using pattern matching and structural analysis<br>- Processes documents to identify article boundaries, preserves metadata, and maintains document integrity<br>- Enables fine-grained indexing and retrieval by creating standalone article records for the vector database.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/content_extraction.py'>content_extraction.py</a></b></td>
                <td>- Content processing pipeline that extracts structured information from raw documents<br>- Handles text cleaning, format normalization, boilerplate removal, and content parsing<br>- Preserves meaningful document structure while preparing content for vectorization and semantic search.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/doc_utils.py'>doc_utils.py</a></b></td>
                <td>- Document handling utilities for processing PDF and DOCX files<br>- Provides functions for document loading, format conversion, validation, and batch processing<br>- Implements robust error handling and format-specific processing logic<br>- Supports streaming operations for large documents.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/s3_utils.py'>s3_utils.py</a></b></td>
                <td>- AWS S3 integration layer for document storage, retrieval, and lifecycle management<br>- Handles file uploads, downloads, metadata operations, and versioning<br>- Implements connection pooling, retry logic, and structured logging for reliability<br>- Provides production-ready S3 operations with comprehensive error handling.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/shared_author.py'>shared_author.py</a></b></td>
                <td>- Author extraction and metadata enrichment module for literary documents<br>- Identifies and extracts author information from documents with multi-author support<br>- Handles author name normalization and associates authors with respective articles<br>- Enables author-based search, filtering, and metadata organization.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/text_extraction.py'>text_extraction.py</a></b></td>
                <td>- Primary text extraction pipeline that converts documents to machine-readable text<br>- Implements OCR fallback for scanned documents and images<br>- Handles text encoding, language detection (Tamil/English), and character normalization<br>- Preserves document structure while extracting clean, searchable text.</td>
            </tr>
            <tr>
                <td><b><a href='data_extraction/text_processing.py'>text_processing.py</a></b></td>
                <td>- Text preprocessing and normalization utilities for Tamil literary content<br>- Implements cleaning, tokenization, and text standardization functions<br>- Handles special characters, whitespace, and formatting inconsistencies<br>- Prepares text for embedding generation and semantic analysis with configurable preprocessing pipelines.</td>
            </tr>
            </table>
        </blockquote>
    </details>
    <details> 
        <summary><b>db</b></summary>
        <blockquote>
            <table>
            <tr>
                <td><b><a href='db/hybrid_search.py'>hybrid_search.py</a></b></td>
                <td>- Hybrid search engine combining semantic vector search with keyword-based retrieval<br>- All queries route through Gemini LLM with context-specific prompts: a lightweight CSV prompt for structured data (authors, topics, issues) and a full document prompt for vector search results<br>- CSV-retrieved data is passed as evidence to the LLM for natural summarization, with raw data preserved as source attribution<br>- Embedding model auto-detects GPU/CPU and applies device-specific optimizations (TF32 on CUDA, thread tuning on CPU)</td>
            </tr>
            <tr>
                <td><b><a href='db/pdf_links.py'>pdf_links.py</a></b></td>
                <td>- PDF metadata and source link management module<br>- Maintains references and links to original PDF documents in S3<br>- Tracks document provenance, version information, and processing status<br>- Enables traceability from search results back to source documents<br>- Supports batch link updates and validation.</td>
            </tr>
            <tr>
                <td><b><a href='db/qdrant_indexer.py'>qdrant_indexer.py</a></b></td>
                <td>- Qdrant vector database integration for embedding storage and similarity search<br>- Handles collection creation, vector indexing, and batch insertion operations<br>- Implements efficient large-scale indexing with configurable distance metrics<br>- Provides similarity search with metadata filtering and payload storage<br>- Includes connection management, retry logic, and error recovery.</td>
            </tr>
            <tr>
                <td><b><a href='db/streamlit_app.py'>streamlit_app.py</a></b></td>
                <td>- Interactive web interface for querying the Ponni RAG system<br>- Provides user-friendly search and document exploration capabilities<br>- Displays search results with relevance scores, source attribution, and metadata<br>- Supports advanced filtering, query refinement, and result visualization<br>- Includes performance metrics and search quality indicators.</td>
            </tr>
            <tr>
                <td><b><a href='db/summary.csv'>summary.csv</a></b></td>
                <td>- Document catalog storing metadata, processing statistics, and indexing status<br>- Contains indexed document information, extraction metrics, and quality scores<br>- Enables quick lookups, reporting, and audit trails for the document collection<br>- Supports data governance and collection management.</td>
            </tr>
            </table>
        </blockquote>
    </details>
    <details> 
        <summary><b>tests</b></summary>
        <blockquote>
            <table>
            <tr>
                <td><b><a href='tests/conftest.py'>conftest.py</a></b></td>
                <td>- Pytest configuration and shared test fixtures<br>- Defines common test setup, teardown procedures, and mock objects<br>- Provides reusable test utilities and helper functions<br>- Configures test environment, logging, and test discovery settings.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_article_patterns.py'>test_article_patterns.py</a></b></td>
                <td>- Unit tests for pattern recognition and article boundary detection<br>- Validates regex patterns and heuristic rules for identifying article structures<br>- Tests edge cases and pattern matching accuracy.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_article_seperation.py'>test_article_seperation.py</a></b></td>
                <td>- Unit tests for document segmentation and article separation functionality<br>- Validates article boundary detection and metadata preservation<br>- Tests multi-article document processing.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_content_extraction.py'>test_content_extraction.py</a></b></td>
                <td>- Unit tests for content extraction and text cleaning processes<br>- Validates boilerplate removal and structure preservation<br>- Tests format normalization and content parsing logic.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_content_extraction_integration.py'>test_content_extraction_integration.py</a></b></td>
                <td>- Integration tests for end-to-end content extraction pipeline<br>- Validates complete workflow from document input to structured output<br>- Tests integration with storage and downstream components.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_doc_utils.py'>test_doc_utils.py</a></b></td>
                <td>- Unit tests for document processing utilities and format handling<br>- Validates PDF and DOCX loading, conversion, and validation<br>- Tests error handling for corrupted or invalid files.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_doc_utils_integration.py'>test_doc_utils_integration.py</a></b></td>
                <td>- Integration tests for document processing workflows<br>- Validates end-to-end document handling from loading to extraction<br>- Tests batch processing and streaming operations.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_hybrid_search.py'>test_hybrid_search.py</a></b></td>
                <td>- Unit tests for hybrid search scoring, ranking, and result merging<br>- Validates semantic and keyword search integration<br>- Tests result aggregation and filtering logic.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_qdrant_indexer.py'>test_qdrant_indexer.py</a></b></td>
                <td>- Unit tests for vector database operations and indexing logic<br>- Validates collection creation, vector insertion, and similarity search<br>- Tests batch operations and error recovery.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_s3_utils.py'>test_s3_utils.py</a></b></td>
                <td>- Unit tests for S3 storage operations and error handling<br>- Validates upload, download, and metadata operations<br>- Tests connection pooling and retry logic.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_shared_author.py'>test_shared_author.py</a></b></td>
                <td>- Unit tests for author extraction and metadata enrichment<br>- Validates author identification and name normalization<br>- Tests multi-author scenarios and edge cases.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_shared_author_integration.py'>test_shared_author_integration.py</a></b></td>
                <td>- Integration tests for author extraction pipeline<br>- Validates end-to-end author processing workflow<br>- Tests integration with document metadata and search.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_streamlit_app.py'>test_streamlit_app.py</a></b></td>
                <td>- Unit tests for Streamlit interface components and interactions<br>- Validates UI rendering, user input handling, and result display<br>- Tests search functionality and visualization components.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_text_extraction.py'>test_text_extraction.py</a></b></td>
                <td>- Unit tests for text extraction pipeline and OCR fallback<br>- Validates text encoding, language detection, and character normalization<br>- Tests extraction accuracy for various document types.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_text_processing.py'>test_text_processing.py</a></b></td>
                <td>- Unit tests for text preprocessing and normalization functions<br>- Validates cleaning, tokenization, and standardization logic<br>- Tests handling of special characters and Tamil text.</td>
            </tr>
            <tr>
                <td><b><a href='tests/test_text_processing_integration.py'>test_text_processing_integration.py</a></b></td>
                <td>- Integration tests for complete text processing workflows<br>- Validates end-to-end preprocessing pipeline<br>- Tests integration with embedding generation and search.</td>
            </tr>
            </table>
        </blockquote>
    </details>
</details>

## Document Storage and Processing Pipeline
This module provides utility functions for managing document storage and processing using **AWS S3**.

#### S3 Initialization
- Initializes an authenticated S3 client using `boto3`
- Fails fast if credentials are missing or invalid

#### Document Discovery
- Lists files in an S3 bucket/prefix with optional file-type filtering
- Used to identify raw or processed documents

#### Document Ingestion
- Reads binary files (PDF, DOCX, etc.) from S3 as byte streams
- Feeds documents into downstream processing (OCR/NLP)

#### Processed Output Storage
- Uploads extracted text as UTF-8 encoded `.txt` files
- Uploads structured outputs (metadata, results) as formatted `.json`

#### Data Retrieval
- Reads stored text and JSON files from S3 for further use

#### File Management
- Checks file existence to avoid reprocessing
- Deletes files for cleanup or lifecycle management

## Getting Started

### Prerequisites

Before getting started with Ponni RAG, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python 3.9+
- **Required Services:**
  - AWS S3 (for document storage)
  - Qdrant Vector Database (local or cloud instance)
  - Gemini API key ([get one here](https://ai.google.dev/))
  - CUDA-compatible GPU (optional, for faster embedding generation)

### Installation

**Build from source:**

1. Clone the Ponni RAG repository:
```sh
git clone https://github.com/nunnarilabs/PonniRAG.git
```

2. Navigate to the project directory:
```sh
cd PonniRAG
```

3. Create a Virtual Environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Upgrade pip:
```sh
pip install --upgrade pip
```

5. Install the project dependencies:
```sh
pip install -r requirements.txt
```

6. Configure environment variables:
```sh
cp .env.example .env
# Edit .env with your credentials:
#   AWS_ACCESS_KEY_ID=<your-access-key>
#   AWS_SECRET_ACCESS_KEY=<your-secret-key>
#   GEMINI_API_KEY=<your-gemini-api-key>
```

### Usage

**Run document indexing:**
```sh
cd src
python -m db.qdrant_indexer
```

**Run the Streamlit interface:**
```sh
cd src/db
streamlit run streamlit_app.py
```

**Run the FastAPI backend:**
```sh
cd src/db
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Run the React frontend:**
```sh
cd frontend
npm install
npm start
```

---

## Docker Deployment

The application can be deployed using Docker Compose, which orchestrates three services: Qdrant (vector database), FastAPI (REST API + Gemini LLM), and React (frontend). LLM inference uses the Gemini API — no local GPU required.

### Services Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Docker Compose                        │
├─────────────────┬─────────────────┬────────────────────────┤
│     Qdrant      │     FastAPI     │         React          │
│   :6333/6334    │      :8000      │         :3000          │
│  Vector Store   │  REST API +     │      Frontend UI       │
│                 │  Gemini LLM     │                        │
└─────────────────┴─────────────────┴────────────────────────┘
                          │
                    Gemini 2.5 Flash API
                   (Google Cloud, external)
```

### Quick Start with Docker Compose

1. **Configure environment variables:**
```sh
cp .env.docker.example .env.docker
# Edit .env.docker with your AWS credentials and Gemini API key
```

2. **Build and start all services:**
```sh
docker-compose up --build
```

3. **Run in detached mode (background):**
```sh
docker-compose up -d --build
```

### Docker Commands

| Command | Description |
|---------|-------------|
| `docker-compose up --build` | Build and start all services |
| `docker-compose up -d` | Start services in background |
| `docker-compose logs -f` | View logs from all services |
| `docker-compose logs -f api` | View logs from FastAPI only |
| `docker-compose down` | Stop all services |
| `docker-compose down -v` | Stop and remove volumes |
| `docker-compose ps` | List running services |
| `docker-compose restart api` | Restart a specific service |

### Service Endpoints

Once running, the services are available at:

| Service | URL | Description |
|---------|-----|-------------|
| React Frontend | http://localhost:3000 | Modern React.js interface |
| Streamlit UI | http://localhost:8501 | Legacy Streamlit interface |
| FastAPI Docs | http://localhost:8000/docs | Swagger API documentation |
| FastAPI Health | http://localhost:8000/health | API health check endpoint |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector database dashboard |

### API Endpoints

The FastAPI service exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (API + Qdrant) |
| `POST` | `/api/ask` | Ask a question (Gemini LLM) |
| `POST` | `/api/ask/stream` | Streaming answer via SSE |
| `GET` | `/api/search?q=<query>` | Search the archive |
| `GET` | `/api/authors` | List all authors |
| `GET` | `/api/authors/{name}/articles` | Get articles by author |
| `GET` | `/api/topics/search?topic=<topic>` | Search by topic |
| `GET` | `/api/issues/stats` | Issue statistics |
| `GET` | `/api/library/volumes` | List all volumes |
| `GET` | `/api/library/volumes/{id}/issues` | Get issues for a volume |
| `GET` | `/api/library/volumes/{id}/issues/{issue}/pdf` | Get PDF link |
| `GET` | `/api/cache/stats` | Cache hit/miss stats |
| `POST` | `/api/cache/clear` | Clear response cache |

### Building Individual Images

**Build Streamlit image:**
```sh
docker build -t ponni-streamlit -f Dockerfile .
```

**Build FastAPI image:**
```sh
docker build -t ponni-api -f Dockerfile.api .
```

**Build React frontend image:**
```sh
docker build -t ponni-frontend -f frontend/Dockerfile ./frontend
```

### GPU Support

LLM inference uses the Gemini API (cloud) — no local GPU is required. A GPU is only beneficial for faster embedding generation with `intfloat/multilingual-e5-large`. The embedding model automatically detects GPU availability at startup: on CUDA-capable hardware it enables TF32 acceleration; on CPU-only systems it optimizes PyTorch thread allocation for best throughput. For NVIDIA GPU support on the API container, ensure you have the NVIDIA Container Toolkit installed and add to `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## React Frontend

A modern React.js frontend is available as an alternative to the Streamlit UI. It provides the same features with a more responsive, production-ready interface.

### Frontend Setup

**Local Development:**

```sh
cd frontend
npm install
npm start
```

The React app will be available at http://localhost:3000

**With Docker Compose:**

```sh
docker-compose up frontend
```

### Frontend Features

| Feature | Description |
|---------|-------------|
| AI Search | Chat-based interface for querying the archive |
| Digital Library | Browse volumes and issues with cover images |
| PDF Viewer | Embedded PDF viewer for magazine issues |
| Language Toggle | Switch between Tamil and English |
| Responsive Design | Works on desktop and mobile devices |

### Frontend Structure

```
frontend/
├── src/
│   ├── components/      # Navigation, ChatMessage, ChatInput
│   ├── pages/           # Home, Library, Issues, PDFViewer, About
│   ├── services/        # API client, translations
│   └── styles/          # CSS matching Streamlit UI
├── Dockerfile
└── nginx.conf
```

### Service URLs (with Docker)

| Service | URL | Description |
|---------|-----|-------------|
| React Frontend | http://localhost:3000 | Modern React UI |
| FastAPI | http://localhost:8000 | REST API backend (+ Gemini LLM) |
| Qdrant | http://localhost:6333 | Vector database |
| Streamlit | http://localhost:8501 | Legacy Streamlit UI |

---

## Mock Data Mode

Mock data mode allows you to test the application without requiring AWS S3 credentials or an indexed Qdrant database. It provides sample Tamil literary data from the Ponni magazine archive.

### Enabling Mock Mode

**For Streamlit:**
```sh
cd src/db
USE_MOCK_DATA=true streamlit run streamlit_app.py
```

**For FastAPI:**
```sh
cd src/db
USE_MOCK_DATA=true uvicorn api:app --host 0.0.0.0 --port 8000
```

**With Docker Compose:**

Add the environment variable to `docker-compose.yml`:
```yaml
services:
  api:
    environment:
      - USE_MOCK_DATA=true
  streamlit:
    environment:
      - USE_MOCK_DATA=true
```

### Mock Data Features

| Feature | Description |
|---------|-------------|
| Sample Sources | 8 Tamil literary articles with metadata |
| Author Queries | Pre-defined responses for author-related questions |
| Topic Search | Mock results for Dravidian movement, Bharathidasan, etc. |
| Health Check | Returns healthy status without database connection |
| Bilingual | Supports both Tamil and English queries |

### Sample Mock Queries

Try these queries in mock mode:

| Tamil | English |
|-------|---------|
| பொன்னி இதழ் ஆசிரியர்கள் யார்? | Who are the authors in Ponni magazine? |
| பாரதிதாசன் கட்டுரைகள் | Articles by Bharathidasan |
| திராவிட இயக்கம் | Dravidian movement |
| பொன்னி நிறுவனர் யார்? | Who founded Ponni magazine? |

### Mock Mode Health Response

```json
{
  "status": "healthy",
  "database": {
    "healthy": true,
    "mock_mode": true,
    "message": "Running in mock mode - no database connection required",
    "points_count": 8
  },
  "api": "healthy"
}
```

---


