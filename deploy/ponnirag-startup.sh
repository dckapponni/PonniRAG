#!/usr/bin/env bash
# PonniRAG startup script — called by systemd on boot.
# Brings up Docker Compose, waits for health, runs warm-up query.
set -euo pipefail

APP_DIR="/home/ubuntu/Ponni_Rag/Tagging_feature"
API_URL="http://localhost:8000"
HEALTH_URL="${API_URL}/health"
WARMUP_URL="${API_URL}/api/ask"

MAX_HEALTH_WAIT=180   # seconds to wait for healthy status
HEALTH_INTERVAL=5     # seconds between health checks

log() { echo "[ponnirag] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# --- 1. Start containers ---
log "Starting Docker Compose..."
cd "$APP_DIR"
docker compose up -d

# --- 2. Wait for health ---
log "Waiting for services to become healthy (max ${MAX_HEALTH_WAIT}s)..."
elapsed=0
while [ $elapsed -lt $MAX_HEALTH_WAIT ]; do
    status=$(curl -sf "${HEALTH_URL}" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unreachable")

    if [ "$status" = "healthy" ]; then
        log "All services healthy after ${elapsed}s."
        break
    fi

    log "  status=${status}, retrying in ${HEALTH_INTERVAL}s..."
    sleep $HEALTH_INTERVAL
    elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "$status" != "healthy" ]; then
    log "ERROR: Services not healthy after ${MAX_HEALTH_WAIT}s. Status: ${status}"
    log "Containers will remain running for manual inspection."
    exit 1
fi

# --- 3. Warm-up query (loads embedding model + reranker + Qdrant connection pool) ---
log "Running warm-up query to preload models..."
warmup_response=$(curl -sf -X POST "${WARMUP_URL}" \
    -H "Content-Type: application/json" \
    -d '{"question": "பொன்னி இதழ் பற்றி சொல்லுங்கள்", "use_llm": false}' \
    --max-time 120 2>&1) || true

if echo "$warmup_response" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('sources') else 1)" 2>/dev/null; then
    log "Warm-up complete — embedding model and search pipeline loaded."
else
    log "WARN: Warm-up query returned no sources (non-fatal). Response: ${warmup_response:0:200}"
fi

log "PonniRAG is ready."
