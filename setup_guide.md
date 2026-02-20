# PonniRAG Deployment Guide

Step-by-step instructions to deploy PonniRAG on an AWS EC2 instance using Docker Compose.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [AWS Setup](#2-aws-setup)
3. [EC2 Instance Setup](#3-ec2-instance-setup)
4. [Clone and Configure](#4-clone-and-configure)
5. [Deploy with Docker Compose](#5-deploy-with-docker-compose)
6. [Index Documents (First-Time Only)](#6-index-documents-first-time-only)
7. [Verification](#7-verification)
8. [Monitoring and Maintenance](#8-monitoring-and-maintenance)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Overview

```
                     ┌──────────────────────────────────────────┐
                     │              EC2 Instance                 │
  User Browser ──────┤                                          │
   port 3000         │  ┌──────────┐       ┌──────────────────┐ │
         ├───────────┼─►│  nginx   │──/api─►│  FastAPI (8000)  │ │
         │           │  │  (3000→80)│       │  uvicorn, 2 wkrs │ │
         │           │  └──────────┘       └────────┬─────────┘ │
         │           │                       │              │    │
         │           │                ┌──────┘              │    │
         │           │                ▼                     ▼    │
         │           │  ┌──────────────────┐    Gemini 2.5 Flash │
         │           │  │  Qdrant (6333)   │    API (external)   │
         │           │  │  vector database │                     │
         │           │  └──────────────────┘                     │
         │           └──────────────────────────────────────────┘
```

| Container | Port | Resource | Purpose |
|-----------|------|----------|---------|
| nginx (frontend) | 3000 → 80 | ~100 MB RAM | Serves React app, proxies `/api` to FastAPI |
| FastAPI (api) | 8000 | ~4 GB RAM | REST API, embedding model, hybrid search, Gemini LLM calls |
| Qdrant | 6333 | ~1 GB RAM | Vector database for document storage |

LLM inference uses the **Gemini 2.5 Flash API** (external, cloud-hosted) — no local GPU or model download is required. A GPU is only beneficial for faster embedding generation with `intfloat/multilingual-e5-large`; the system auto-detects GPU availability and optimizes accordingly.

**Query routing:** All queries go through Gemini with context-specific prompts:
- **CSV queries** (authors, topics, issues) — lightweight `_CSV_SYSTEM_PROMPT`, only database evidence in context
- **Vector search queries** — full `TAMIL_ANSWER_SYSTEM_PROMPT` with Ponni background, document chunks (1500 chars × 3 docs), and CSV semantic matches

---

## 2. AWS Setup

### 2.1 Create IAM Role

1. **IAM Console** → **Roles** → **Create Role**
2. Trusted entity: **AWS Service** → **EC2**
3. Attach a custom policy for S3 access:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::ponni-dev",
                "arn:aws:s3:::ponni-dev/*"
            ]
        }
    ]
}
```

4. Name: `PonniRAG-EC2-Role`

### 2.2 Create Security Group

**EC2 Console** → **Security Groups** → **Create**

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| Custom TCP | 3000 | 0.0.0.0/0 | Frontend (nginx) |

Only port 3000 needs to be public. All API calls are proxied through nginx, so port 8000 does not need to be exposed externally.

Name: `PonniRAG-SG`

### 2.3 Launch EC2 Instance

| Setting | Value |
|---------|-------|
| Name | `PonniRAG-Server` |
| AMI | Ubuntu Server 22.04 LTS (HVM), x86_64 |
| Instance type | **t3.xlarge** (4 vCPU, 16 GB RAM) — no GPU required |
| Key pair | Select or create one |
| Security group | `PonniRAG-SG` |
| Storage | 30 GB gp3 |
| IAM instance profile | `PonniRAG-EC2-Role` |

> **GPU optional:** If you want faster embedding generation, use a **g4dn.xlarge** (T4 GPU) instead. The embedding model auto-detects CUDA and uses it when available. LLM inference always uses the Gemini API regardless.

For cost optimization, see [inference_recommendation_guide.md](inference_recommendation_guide.md) — spot instances can reduce cost by 60-70%.

### 2.4 Attach EBS Volume for Qdrant Persistence (Optional)

To survive spot interruptions, attach a separate EBS gp3 volume (30 GB, ~$2.40/month) for Qdrant data.

```bash
# After attaching in the console:
sudo mkfs.ext4 /dev/xvdf
sudo mkdir -p /data/qdrant
sudo mount /dev/xvdf /data/qdrant

# Add to /etc/fstab for auto-mount on reboot
echo '/dev/xvdf /data/qdrant ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

---

## 3. EC2 Instance Setup

### 3.1 Connect via SSH

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### 3.2 Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install prerequisites
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common git

# Add Docker GPG key and repo
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

### 3.3 Install NVIDIA Drivers + Container Toolkit (GPU instances only)

Skip this section if using a CPU-only instance (e.g., t3.xlarge).

```bash
# Install NVIDIA driver
sudo apt install -y nvidia-driver-535
sudo reboot
```

After reconnecting:

```bash
# Verify GPU is detected
nvidia-smi

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU works in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 4. Clone and Configure

### 4.1 Clone the Repository

```bash
cd ~
git clone https://github.com/nunnarilabs/PonniRAG.git
cd PonniRAG
git checkout frontend
```

### 4.2 Create Environment File

```bash
cp .env.docker.example .env.docker
nano .env.docker
```

Fill in your credentials:

```bash
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
AWS_DEFAULT_REGION=ap-south-1

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Gemini API (required)
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.5-flash
```

> Get a Gemini API key at [https://ai.google.dev/](https://ai.google.dev/)

### 4.3 Verify Key Files Exist

```bash
# These files are required by docker-compose.yml
ls -la Dockerfile.api           # FastAPI container build
ls -la frontend/Dockerfile      # React + nginx container build
ls -la frontend/nginx.conf      # nginx config (proxies /api to FastAPI)
```

---

## 5. Deploy with Docker Compose

### 5.1 Build and Start

```bash
cd ~/PonniRAG
docker compose up --build -d
```

### 5.2 Startup Sequence

Docker Compose uses healthchecks to enforce the correct startup order:

```
1. qdrant starts      → healthcheck: GET /healthz every 10s
2. api starts         → only after qdrant is healthy
   └─ Downloads embedding model on first run (~2.3 GB, one-time)
   └─ Validates Gemini API key on startup
   └─ Auto-detects GPU/CPU for embedding model
3. frontend starts    → only after api is up
```

### 5.3 Watch the Logs

```bash
# Follow all service logs
docker compose logs -f

# Or watch a specific service
docker compose logs -f api       # Watch embedding model download + API startup
```

**First startup takes 3-5 minutes** because FastAPI downloads the `intfloat/multilingual-e5-large` embedding model (~2.3 GB).

Subsequent startups take ~30-60 seconds.

### 5.4 Verify All Containers Are Running

```bash
docker compose ps
```

Expected output:

```
NAME             SERVICE    STATUS                  PORTS
qdrant           qdrant     Up (healthy)            0.0.0.0:6333->6333/tcp
ponni-api        api        Up (healthy)            0.0.0.0:8000->8000/tcp
ponni-frontend   frontend   Up                      0.0.0.0:3000->80/tcp
```

All services should show `Up`. Qdrant and API should show `(healthy)`.

---

## 6. Index Documents (First-Time Only)

If this is a fresh deployment (no existing Qdrant data), you need to index the Tamil documents:

```bash
# Enter the API container
docker compose exec api bash

# Run the indexer (from /app/src/db inside the container)
cd /app/src
python -m db.qdrant_indexer

# This reads documents from S3, embeds them, and stores in Qdrant.
# Takes 30-60 minutes depending on corpus size.
# Exit when done
exit
```

If you attached a separate EBS volume with existing Qdrant data, or are using Docker named volumes from a previous deployment, the data persists and you can skip this step.

---

## 7. Verification

### 7.1 Health Checks

```bash
# API health (from the EC2 instance)
curl http://localhost:8000/health

# Qdrant health
curl http://localhost:6333/healthz

# Frontend health (proxied through nginx to API)
curl http://localhost:3000/health
```

### 7.2 Test the API

```bash
# Ask a question (vector search + Gemini LLM)
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "பொன்னி இதழ் பற்றி கூறுக", "use_llm": true}'

# Ask a CSV query (routed through Gemini with CSV-specific prompt)
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "எழுத்தாளர்கள் யார்?", "use_llm": true}'

# Check cache stats (should show 0 hits after first query)
curl http://localhost:8000/api/cache/stats

# Ask the same question again — should be a cache hit
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "பொன்னி இதழ் பற்றி கூறுக", "use_llm": true}'
```

### 7.3 Test Streaming

```bash
curl -N -X POST http://localhost:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "பொன்னி இதழ் பற்றி கூறுக", "use_llm": true}'
```

You should see SSE events streaming in (`event: token`, `event: sources`, `event: done`).

### 7.4 Access from Browser

Open in your browser: `http://<EC2-PUBLIC-IP>:3000`

All API calls from the frontend go through nginx on the same origin, so no CORS issues.

---

## 8. Monitoring and Maintenance

### 8.1 Resource Monitoring

```bash
# Live container resource usage
docker stats

# GPU utilization (if GPU instance)
nvidia-smi

# Disk usage
df -h
```

### 8.2 Cache Management

```bash
# View cache hit rate
curl http://localhost:8000/api/cache/stats
# Returns: {"size": 5, "max_size": 100, "ttl_seconds": 3600, "hits": 12, "misses": 8, "hit_rate": "60%"}

# Clear cache (after updating prompts or data)
curl -X POST http://localhost:8000/api/cache/clear
```

### 8.3 Updating the Application

```bash
cd ~/PonniRAG
git pull origin frontend
docker compose up --build -d
```

Only containers with changed images will be rebuilt.

### 8.4 Auto-Start on Reboot

```bash
# Docker is already configured to start on boot. Containers with
# restart: unless-stopped will auto-start when Docker starts.
sudo systemctl enable docker
```

### 8.5 Log Rotation

Docker logs can grow large. Configure log rotation:

```bash
# Create /etc/docker/daemon.json
sudo tee /etc/docker/daemon.json <<EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    }
}
EOF

sudo systemctl restart docker
```

---

## 9. Troubleshooting

### Gemini API errors

```bash
docker compose logs api | grep -i gemini
```

Common causes:
- `GEMINI_API_KEY` not set or invalid — check `.env.docker`
- API quota exceeded — check your [Google AI Studio](https://ai.google.dev/) usage dashboard
- Network connectivity — ensure the EC2 instance has outbound HTTPS access

### API starts but queries fail

```bash
docker compose logs api
```

Common causes:
- Embedding model still downloading (first run) — wait for `"Loaded model intfloat/multilingual-e5-large"` in logs
- Qdrant collection not indexed — run the indexer (Section 6)
- Gemini API key invalid — look for `"Gemini API validated successfully"` in startup logs

### Frontend shows blank page or API errors

```bash
docker compose logs frontend
```

Common causes:
- API container not healthy yet — `docker compose ps` should show `(healthy)` for api
- nginx can't reach the API — verify `docker compose exec frontend curl http://api:8000/health`

### Out of memory

```bash
free -m
docker stats
```

If RAM is exhausted, add swap:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Full reset

```bash
# Stop everything, remove volumes (deletes Qdrant data!)
docker compose down -v
docker system prune -a

# Rebuild from scratch
docker compose up --build -d
```

---

## Quick Reference

```bash
docker compose up --build -d     # Build and start all services
docker compose ps                # Check container status
docker compose logs -f           # Follow all logs
docker compose logs -f api       # Follow specific service
docker compose restart api       # Restart one service
docker compose down              # Stop all services
docker compose exec api bash     # Shell into a container
```

| URL | Purpose |
|-----|---------|
| `http://<EC2-IP>:3000` | Frontend (all users access this) |
| `http://<EC2-IP>:3000/health` | Health check (proxied to API) |
| `http://localhost:8000/docs` | FastAPI Swagger docs (from EC2 only) |
| `http://localhost:6333/dashboard` | Qdrant admin UI (from EC2 only) |
