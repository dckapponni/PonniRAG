
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Verify CSV
RUN echo "=== CSV Check ===" && \
    ls -lh /app/src/db/summary.csv || echo "CSV NOT FOUND"

# Environment
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Qdrant
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Ollama
ENV OLLAMA_HOST=http://ollama:11434

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/db/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
