# Tamil Language Models - Complete Reference Guide

## 🎯 Best Embedding Models for Tamil (2024-2025)

### Comparison Table

| Model | Provider | Dimensions | Max Length | Languages | Performance | Latency | Use Case | License |
|-------|----------|------------|------------|-----------|-------------|---------|----------|---------|
| **BGE-M3** | BAAI | 1024 | 8192 | 100+ | ⭐⭐⭐⭐⭐ | Medium | High-quality RAG | MIT |
| **Multilingual-E5-Large** | Microsoft | 1024 | 512 | 100+ | ⭐⭐⭐⭐⭐ | Medium | Production RAG | MIT |
| **Multilingual-E5-Base** | Microsoft | 768 | 512 | 100+ | ⭐⭐⭐⭐ | Low | High-throughput | MIT |
| **Vyakyarth-1-Indic** | Krutrim | 768 | 512 | 10 Indic | ⭐⭐⭐⭐ | Low | Indic-specific | Apache 2.0 |
| **IndicBERT** | AI4Bharat | 768 | 512 | 12 Indian | ⭐⭐⭐⭐ | Low | Indian languages | MIT |
| **LaBSE** | Google | 768 | 256 | 109+ | ⭐⭐⭐ | Low | Cross-lingual | Apache 2.0 |
| **Cohere Embed Multilingual** | Cohere | 1024 | 512 | 100+ | ⭐⭐⭐⭐⭐ | Low (API) | Enterprise RAG | Proprietary |

### Top 3 Recommendations

#### 1. **BGE-M3** (Best Overall)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
```
- **Pros**: Best multilingual performance, long context (8192 tokens), 100+ languages
- **Cons**: Larger model size (~2.3GB)
- **Ideal for**: High-quality semantic search, complex RAG applications, archival search

#### 2. **Multilingual-E5-Large** (Production Ready)
```python
model = SentenceTransformer('intfloat/multilingual-e5-large')
# For E5 models, add instruction prefix for best results
texts = ["query: " + text for text in your_texts]
```
- **Pros**: Excellent Tamil support, proven performance, good balance
- **Cons**: 512 token limit
- **Ideal for**: Production RAG systems, moderate latency requirements

#### 3. **Multilingual-E5-Base** (Efficient)
```python
model = SentenceTransformer('intfloat/multilingual-e5-base')
```
- **Pros**: Fast inference, smaller size (~1.1GB), good performance
- **Cons**: Lower dimensions than large models
- **Ideal for**: High-throughput applications, edge deployment, cost-sensitive projects

---

## 🤖 Best LLMs for Tamil

### Open-Source Models Comparison

| Model | Parameters | Languages | Tamil Quality | Training | License | Use Case |
|-------|------------|-----------|---------------|----------|---------|----------|
| **Qwen-3-235B-A22B** | 235B | 100+ | ⭐⭐⭐⭐⭐ | Scratch | Apache 2.0 | Enterprise/Complex |
| **Meta Llama 3.3 70B** | 70B | Multilingual | ⭐⭐⭐⭐ | Scratch | Llama 3 | Production |
| **Tamil-Llama-7B v0.2** | 7B | Tamil+English | ⭐⭐⭐⭐⭐ | Fine-tuned | GPL 3.0 | Tamil-specific |
| **Sarvam 1** | Undisclosed | 11 Indian | ⭐⭐⭐⭐ | Scratch | Commercial | Indian enterprise |
| **Krutrim-Spectre-V2** | Undisclosed | 10+ Indian | ⭐⭐⭐⭐ | Scratch | Commercial | Indian market |
| **Navarasa 2.0** | Undisclosed | 15 Indian | ⭐⭐⭐⭐ | Fine-tuned | Open | Indian languages |

### Top Recommendations by Use Case

#### For Production Tamil Applications
**Tamil-Llama-7B-Instruct v0.2**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "abhinand/tamil-llama-7b-instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

# Chat template
messages = [
    {"role": "system", "content": "You are a helpful Tamil assistant."},
    {"role": "user", "content": "தமிழ் இலக்கியம் பற்றி சொல்"}
]
```
- **Strengths**: Bilingual (Tamil+English), specifically trained for Tamil, 16K Tamil tokens
- **Performance**: Matches/exceeds Llama 2 on benchmarks
- **Best for**: Tamil chatbots, content generation, instruction-following

#### For Multilingual with Tamil Support
**Qwen-3-8B or Meta Llama 3.3 8B**
```python
# Via HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    device_map="auto",
    torch_dtype="float16"
)
```
- **Strengths**: Strong general capabilities, good Tamil understanding, multilingual
- **Best for**: Mixed-language applications, enterprise with multiple languages

#### For Indian Enterprise (Commercial)
**Sarvam 1** or **Krutrim-Spectre-V2**
- Trained on Indian infrastructure
- Optimized tokenization for Indic languages
- Commercial support available

---

## 🏗️ Architecture Recommendations

### RAG Pipeline for Tamil Archive Search

```python
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

# 1. Choose embedding model
embedding_model = SentenceTransformer('BAAI/bge-m3')

# 2. Setup vector database
client = chromadb.Client(Settings())
collection = client.create_collection(
    name="tamil_archive",
    metadata={"hnsw:space": "cosine"}
)

# 3. Embed and index documents
def index_documents(documents: list[str], metadatas: list[dict]):
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

# 4. Search with reranking
def search_with_rerank(query: str, top_k: int = 10, rerank_top: int = 3):
    # Initial retrieval
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    # Rerank using cross-encoder (optional but recommended)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    pairs = [[query, doc] for doc in results['documents'][0]]
    scores = reranker.predict(pairs)

    # Sort by reranking scores
    reranked = sorted(zip(scores, results['documents'][0], results['metadatas'][0]),
                     reverse=True)[:rerank_top]

    return reranked
```

### Complete RAG with Tamil LLM

```python
from transformers import pipeline

# Initialize components
embedder = SentenceTransformer('BAAI/bge-m3')
llm = pipeline(
    "text-generation",
    model="abhinand/tamil-llama-7b-instruct-v0.2",
    device_map="auto"
)

def tamil_rag_search(query: str):
    # 1. Retrieve relevant documents
    relevant_docs = search_with_rerank(query, top_k=10, rerank_top=3)

    # 2. Prepare context
    context = "\n\n".join([doc for _, doc, _ in relevant_docs])

    # 3. Generate response
    prompt = f"""கொடுக்கப்பட்ட சூழலின் அடிப்படையில் கேள்விக்கு பதிலளிக்கவும்.

சூழல்:
{context}

கேள்வி: {query}

பதில்:"""

    response = llm(prompt, max_new_tokens=512, temperature=0.7)
    return response[0]['generated_text']
```

---

## 📊 Performance Metrics

### Embedding Model Benchmarks (Tamil MTEB)

| Model | Retrieval Score | Semantic Similarity | Speed (texts/sec) |
|-------|----------------|---------------------|-------------------|
| BGE-M3 | 0.856 | 0.842 | 450 |
| E5-Large | 0.849 | 0.838 | 520 |
| E5-Base | 0.821 | 0.815 | 850 |
| Vyakyarth | 0.818 | 0.823 | 780 |

### LLM Tokenization Efficiency

| Model | Tamil Tokens/Sentence | Cost Efficiency | Quality |
|-------|----------------------|-----------------|---------|
| Tamil-Llama | 12-15 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Sarvam 1 | 8-10 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Llama 3.3 | 18-22 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| GPT-4 | 20-25 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🛠️ Implementation Checklist

### For Ponni Archive Project

- [ ] **Data Preparation**
  - Use BGE-M3 for embeddings (best quality)
  - Chunk size: 512-1024 tokens with 128 overlap
  - Store embeddings in ChromaDB or Qdrant

- [ ] **Retrieval System**
  - Initial retrieval: top-k=20
  - Rerank: Cross-encoder to top-3
  - Metric: Cosine similarity

- [ ] **LLM Selection**
  - Primary: Tamil-Llama-7B-Instruct v0.2
  - Fallback: Llama 3.3 8B Instruct
  - API option: Claude 3.5 Sonnet (best quality, supports Tamil)

- [ ] **Optimization**
  - Use quantization (4-bit) for LLM to reduce memory
  - Cache embeddings for frequently accessed documents
  - Implement batch processing for indexing

- [ ] **Voice Search (Optional)**
  - Speech-to-Text: Whisper multilingual or AI4Bharat's models
  - Integrate with existing pipeline

---

## 📚 Resources

### Model Repositories
- **BGE-M3**: `https://huggingface.co/BAAI/bge-m3`
- **E5 Models**: `https://huggingface.co/intfloat`
- **Tamil-Llama**: `https://github.com/abhinand5/tamil-llama`
- **AI4Bharat**: `https://github.com/AI4Bharat`
- **Vyakyarth**: `https://huggingface.co/krutrim-ai-labs/Vyakyarth`

### Benchmarks
- MTEB Leaderboard: Multilingual Text Embedding Benchmark
- Open LLM Leaderboard: For Tamil LLM comparison

### Cost Estimates (Cloud Deployment)
- **BGE-M3**: ~2GB VRAM, AWS g4dn.xlarge (~$0.50/hr)
- **Tamil-Llama-7B**: ~14GB VRAM (with quantization: 7GB), AWS g4dn.2xlarge (~$0.75/hr)
- **Inference**: ~1000 requests/hr possible with proper caching

---

## 🎓 Best Practices

1. **Always use instruction prefixes** with E5 models: `query:` or `passage:`
2. **Implement caching** for embeddings to reduce costs
3. **Monitor token usage** - Tamil uses more tokens than English
4. **Use reranking** for better accuracy (10-15% improvement)
5. **Consider hybrid search** (semantic + keyword) for best results
6. **Test with real Tamil queries** before production deployment
