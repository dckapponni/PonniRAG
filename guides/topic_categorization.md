# Tamil Article Categorization - Deployment & Optimization Guide

## 📋 Table of Contents
1. [System Architecture](#system-architecture)
2. [Model Selection Strategy](#model-selection-strategy)
3. [Performance Optimization](#performance-optimization)
4. [Scaling Strategies](#scaling-strategies)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🏗️ System Architecture

### Recommended Architecture for Production

```
┌─────────────────┐
│  Article Input  │
│   (CSV/JSON)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Data Preprocessing Layer        │
│  - Text cleaning                    │
│  - Tamil Unicode normalization      │
│  - Length validation                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Categorization Pipeline          │
│                                     │
│  Stage 1: Fast Filtering (LDA/NMF) │
│  Stage 2: Semantic Analysis         │
│           (BERTopic with BGE-M3)    │
│  Stage 3: LLM Refinement            │
│           (Tamil-Llama, selective)  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Results Storage & API          │
│  - PostgreSQL (structured data)     │
│  - Redis (caching)                  │
│  - REST API for access              │
└─────────────────────────────────────┘
```

---

## 🎯 Model Selection Strategy

### Decision Matrix

| Use Case | Volume | Latency | Model Choice | GPU Required |
|----------|--------|---------|--------------|--------------|
| **Initial Categorization** | High | Low | LDA/NMF | No |
| **Semantic Tagging** | Medium | Medium | BERTopic (BGE-M3) | Yes (4GB+) |
| **High-Quality Tags** | Low | High | Tamil-Llama | Yes (16GB+) |
| **Real-time Tagging** | Medium | Low | BERTopic cached | Yes (8GB+) |
| **Batch Processing** | Very High | Flexible | Hybrid Pipeline | Optional |

### Model Recommendations by Infrastructure

#### 1. **CPU-Only Setup** (Budget: $0-500/month)
```python
# Best approach: Traditional NLP
taggers = {
    "primary": NMFTagger(n_topics=20),
    "backup": LDATagger(n_topics=20)
}

# Process in large batches
batch_size = 1000
```

**Performance:**
- Throughput: 1000-5000 articles/hour
- Latency: <100ms per article
- Quality: Good for broad categorization

#### 2. **GPU Setup - Entry Level** (Budget: $500-1500/month)
```python
# 8GB VRAM (e.g., RTX 3070, A10G)
tagger = BERTopicTagger(
    embedding_model="BAAI/bge-m3"  # ~2.3GB
)

# Process in medium batches
batch_size = 100
```

**Performance:**
- Throughput: 500-1000 articles/hour
- Latency: ~500ms per article
- Quality: Excellent semantic understanding

#### 3. **GPU Setup - Production** (Budget: $1500-3000/month)
```python
# 16GB+ VRAM (e.g., RTX 4090, A100)
bertopic = BERTopicTagger(embedding_model="BAAI/bge-m3")
llm = LLMTagger(
    model_name="abhinand/tamil-llama-7b-instruct-v0.2"
)
hybrid = HybridTagger(bertopic, llm)

batch_size = 50
```

**Performance:**
- Throughput: 200-500 articles/hour
- Latency: ~2s per article
- Quality: Best possible quality

---

## ⚡ Performance Optimization

### 1. **Embedding Caching**

```python
import pickle
from pathlib import Path

class CachedBERTopicTagger(BERTopicTagger):
    def __init__(self, cache_dir: str = ".cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def tag_batch(self, articles):
        # Generate cache key
        cache_key = hash(tuple(a.id for a in articles))
        cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"

        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Generate and cache
        results = super().tag_batch(articles)
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)

        return results
```

### 2. **Batch Processing Optimization**

```python
def optimized_batch_processing(
    articles: List[Article],
    tagger: ArticleTagger,
    batch_size: int = 100
):
    """Process articles in optimal batches with progress tracking"""
    from tqdm import tqdm

    results = []

    for i in tqdm(range(0, len(articles), batch_size)):
        batch = articles[i:i+batch_size]
        batch_results = tagger.tag_batch(batch)
        results.extend(batch_results)

        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
```

### 3. **Quantization for Large Models**

```python
from transformers import BitsAndBytesConfig

# 8-bit quantization (reduces memory by ~50%)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

llm_tagger = LLMTagger(
    model_name="abhinand/tamil-llama-7b-instruct-v0.2",
    quantization_config=quantization_config  # Add this parameter
)
```

### 4. **Parallel Processing**

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

def parallel_tagging(
    articles: List[Article],
    tagger: ArticleTagger,
    n_workers: int = 4
):
    """Process articles in parallel"""

    # Split into chunks
    chunk_size = len(articles) // n_workers
    chunks = [
        articles[i:i+chunk_size]
        for i in range(0, len(articles), chunk_size)
    ]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(tagger.tag_batch, chunks))

    # Flatten results
    return [r for chunk in results for r in chunk]
```

---

## 📈 Scaling Strategies

### Horizontal Scaling with Message Queue

```python
# Using Celery + Redis for distributed processing

from celery import Celery

app = Celery('tamil_tagger', broker='redis://localhost:6379')

@app.task
def tag_article_task(article_data):
    """Celery task for article tagging"""
    article = Article(**article_data)
    tagger = BERTopicTagger(embedding_model="BAAI/bge-m3")
    result = tagger.tag_article(article)
    return result.to_dict()

# Submit tasks
for article in articles:
    tag_article_task.delay(article.__dict__)
```

### Load Balancing Strategy

```python
class LoadBalancedTagger:
    """Distribute load across multiple tagging methods"""

    def __init__(self):
        # Fast tagger for high-volume
        self.fast_tagger = NMFTagger(n_topics=15)

        # Quality tagger for important articles
        self.quality_tagger = BERTopicTagger(
            embedding_model="BAAI/bge-m3"
        )

    def tag_article(self, article: Article, priority: str = "normal"):
        if priority == "high":
            return self.quality_tagger.tag_article(article)
        else:
            return self.fast_tagger.tag_article(article)
```

---

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models during build
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-m3')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0"]
```

### FastAPI Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Tamil Article Tagger API")

# Initialize tagger (singleton)
tagger = BERTopicTagger(embedding_model="BAAI/bge-m3")

class ArticleRequest(BaseModel):
    id: str
    title: str
    content: str

class TaggingResponse(BaseModel):
    article_id: str
    categories: List[str]
    tags: List[str]
    confidence_scores: dict

@app.post("/tag", response_model=TaggingResponse)
async def tag_article(request: ArticleRequest):
    """Tag a single article"""
    try:
        article = Article(
            id=request.id,
            title=request.title,
            content=request.content
        )
        result = tagger.tag_article(article)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tag/batch", response_model=List[TaggingResponse])
async def tag_batch(requests: List[ArticleRequest]):
    """Tag multiple articles"""
    articles = [
        Article(id=r.id, title=r.title, content=r.content)
        for r in requests
    ]
    results = tagger.tag_batch(articles)
    return [r.to_dict() for r in results]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tamil-tagger
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tamil-tagger
  template:
    metadata:
      labels:
        app: tamil-tagger
    spec:
      containers:
      - name: tagger
        image: tamil-tagger:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        env:
        - name: MODEL_NAME
          value: "BAAI/bge-m3"
---
apiVersion: v1
kind: Service
metadata:
  name: tamil-tagger-service
spec:
  selector:
    app: tamil-tagger
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📊 Monitoring & Maintenance

### Performance Monitoring

```python
import time
from functools import wraps
from prometheus_client import Counter, Histogram

# Metrics
tagging_requests = Counter(
    'tagging_requests_total',
    'Total tagging requests',
    ['method', 'status']
)
tagging_duration = Histogram(
    'tagging_duration_seconds',
    'Time spent tagging',
    ['method']
)

def monitor_performance(func):
    """Decorator to monitor tagging performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        method = args[0].__class__.__name__

        try:
            result = func(*args, **kwargs)
            tagging_requests.labels(method=method, status='success').inc()
            duration = time.time() - start
            tagging_duration.labels(method=method).observe(duration)
            return result
        except Exception as e:
            tagging_requests.labels(method=method, status='error').inc()
            raise

    return wrapper
```

### Quality Monitoring

```python
class QualityMonitor:
    """Monitor tagging quality over time"""

    def __init__(self):
        self.metrics = {
            'avg_tags_per_article': [],
            'avg_categories_per_article': [],
            'processing_time': []
        }

    def record_batch(self, results: List[TaggingResult], duration: float):
        avg_tags = sum(len(r.tags) for r in results) / len(results)
        avg_cats = sum(len(r.categories) for r in results) / len(results)

        self.metrics['avg_tags_per_article'].append(avg_tags)
        self.metrics['avg_categories_per_article'].append(avg_cats)
        self.metrics['processing_time'].append(duration)

    def get_report(self):
        return {
            'avg_tags': np.mean(self.metrics['avg_tags_per_article']),
            'avg_categories': np.mean(self.metrics['avg_categories_per_article']),
            'avg_processing_time': np.mean(self.metrics['processing_time'])
        }
```

### Alert Configuration

```python
import logging
from typing import Callable

class AlertSystem:
    """Simple alerting for production issues"""

    def __init__(self,
                 error_threshold: int = 10,
                 latency_threshold: float = 5.0):
        self.error_count = 0
        self.error_threshold = error_threshold
        self.latency_threshold = latency_threshold

    def check_and_alert(self,
                       error_occurred: bool = False,
                       latency: float = 0.0):
        if error_occurred:
            self.error_count += 1
            if self.error_count >= self.error_threshold:
                self.send_alert("High error rate detected!")
                self.error_count = 0

        if latency > self.latency_threshold:
            self.send_alert(f"High latency: {latency:.2f}s")

    def send_alert(self, message: str):
        # Implement your alerting mechanism
        logging.error(f"ALERT: {message}")
        # Could integrate with Slack, PagerDuty, etc.
```

---

## 🎓 Best Practices Summary

### Development Phase
1. ✅ Start with small dataset (100-1000 articles)
2. ✅ Compare all methods (LDA, NMF, BERTopic)
3. ✅ Use validation set for quality assessment
4. ✅ Tune hyperparameters based on results

### Testing Phase
1. ✅ Test with production-like data volume
2. ✅ Measure latency and throughput
3. ✅ Validate tag quality with domain experts
4. ✅ Test edge cases (very short/long articles)

### Production Phase
1. ✅ Start with conservative batch sizes
2. ✅ Implement comprehensive monitoring
3. ✅ Set up automated alerts
4. ✅ Plan for model updates and retraining

### Maintenance Phase
1. ✅ Monitor tag quality over time
2. ✅ Collect user feedback
3. ✅ Retrain models quarterly
4. ✅ Update to newer model versions when available

---

## 📞 Support & Resources

### Recommended Model Updates Schedule
- **Embeddings**: Check quarterly for new releases
- **LLMs**: Monitor HuggingFace Tamil model space monthly
- **BERTopic**: Update when major versions release

### Performance Targets
- **Latency**: <1s per article (BERTopic), <5s (LLM)
- **Throughput**: 500+ articles/hour (with GPU)
- **Accuracy**: 85%+ tag relevance (manual validation)
- **Uptime**: 99.5%+ for production systems

### Cost Estimation (Monthly)
- **CPU-only**: $200-500 (cloud compute)
- **GPU (8GB)**: $500-1500 (AWS P3, Azure NC)
- **GPU (16GB+)**: $1500-3000 (AWS P3, GCP A100)

---

## 📝 Conclusion

This guide provides a comprehensive approach to deploying Tamil article categorization in production. Start simple with traditional methods, then scale up based on quality requirements and budget constraints.

**Key Takeaways:**
1. Match method to infrastructure and requirements
2. Optimize with caching and batching
3. Monitor continuously for quality and performance
4. Scale horizontally for high-volume scenarios
5. Keep models updated for best results
