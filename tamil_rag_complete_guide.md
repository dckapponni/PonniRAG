# Complete Tamil RAG Search System - Implementation Guide

**A Production-Ready Retrieval-Augmented Generation System for Tamil Documents**

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Application Code](#application-code)
6. [Configuration Files](#configuration-files)
7. [Qdrant Database Setup](#qdrant-database-setup)
8. [Usage Guide](#usage-guide)
9. [Document Format](#document-format)
10. [Advanced Features](#advanced-features)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)
13. [Production Deployment](#production-deployment)
14. [API Reference](#api-reference)
15. [Best Practices](#best-practices)

---

## 1. Overview {#overview}

### 1.1 What This System Does

This Tamil RAG (Retrieval-Augmented Generation) system enables intelligent search and question-answering over Tamil documents using state-of-the-art AI models. It combines:

- **Semantic search** using multilingual embeddings
- **Generative AI** for natural language answers
- **Vector database** for efficient retrieval
- **Bilingual support** for Tamil and English

### 1.2 Technology Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Embedding Model** | `intfloat/multilingual-e5-large` | Best performance for Tamil cross-lingual retrieval |
| **LLM** | `abhinand/tamil-llama-7b-instruct-v0.2` | Optimized for Tamil with 16K Tamil tokens |
| **Vector DB** | Qdrant | High-performance, open-source, production-ready |
| **Web Framework** | Streamlit | Rapid development, interactive UI |
| **Acceleration** | 4-bit quantization | 75% memory reduction, minimal quality loss |

### 1.3 Key Features

✅ **Bilingual Support**: Tamil and English  
✅ **Fast Search**: <10ms vector search with Qdrant  
✅ **Smart Filtering**: Category-based document filtering  
✅ **Flexible Storage**: In-memory or persistent modes  
✅ **Production Ready**: Docker, Kubernetes, Cloud support  
✅ **Rich Metadata**: Tags, categories, language detection  
✅ **Scalable**: Horizontal scaling with sharding  

---

## 2. System Architecture {#system-architecture}

### 2.1 High-Level Architecture

```
┌─────────────────┐
│   User Query    │
│  (Tamil/English)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Tamil RAG System Core           │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │  Embedding   │  │   Tamil-Llama   │ │
│  │  (E5-Large)  │  │  (Generation)   │ │
│  └──────┬───────┘  └────────┬────────┘ │
│         │                   │          │
│         ▼                   ▼          │
│  ┌──────────────────────────────────┐ │
│  │       Qdrant Vector DB           │ │
│  │  • Cosine Similarity Search      │ │
│  │  • Metadata Filtering            │ │
│  │  • 1024-dim Vectors              │ │
│  └──────────────────────────────────┘ │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Tamil Documents│
│  • Literature   │
│  • History      │
│  • Culture      │
└─────────────────┘
```

### 2.2 Data Flow

1. **Document Ingestion**
   ```
   JSON Documents → Chunking → Embedding → Qdrant Storage
   ```

2. **Query Processing**
   ```
   User Query → Query Embedding → Vector Search → Top-K Results → Context Building → LLM Generation → Answer
   ```

### 2.3 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Streamlit UI** | User interface, input handling, result display |
| **Embedding Model** | Convert text to 1024-dim vectors |
| **Qdrant** | Store vectors, perform similarity search, filter by metadata |
| **Tamil-Llama** | Generate natural language answers from context |
| **RAG System** | Orchestrate all components, manage workflow |

---

## 3. Prerequisites {#prerequisites}

### 3.1 System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB free
- Python: 3.9+
- OS: Linux/Mac/Windows

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: 8GB+ VRAM (NVIDIA with CUDA)
- Storage: 50GB+ SSD
- Python: 3.10+

### 3.2 Software Dependencies

- Python 3.9 or higher
- pip (latest version)
- Git
- Docker (optional, for Qdrant server)
- CUDA Toolkit 11.8+ (optional, for GPU)

### 3.3 Knowledge Requirements

- Basic Python programming
- Understanding of embeddings and vectors
- Familiarity with command line
- (Optional) Docker basics for deployment

---

## 4. Installation {#installation}

### 4.1 Step-by-Step Installation

#### Step 1: Create Project Directory

```bash
# Create and navigate to project directory
mkdir tamil-rag-system
cd tamil-rag-system
```

#### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

#### Step 4: Create Requirements File

Create a file named `requirements.txt` with the following content:

```txt
# Core ML/AI Libraries
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.2
accelerate>=0.24.0
bitsandbytes>=0.41.0

# Vector Database - Qdrant
qdrant-client>=1.7.0

# Web Framework
streamlit>=1.28.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0

# Optional: For better performance
sentencepiece>=0.1.99
protobuf>=3.20.0
```

#### Step 5: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU support (if you have NVIDIA GPU with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 6: Verify Installation

```bash
# Check Python packages
pip list | grep -E "torch|transformers|qdrant|streamlit"

# Check if GPU is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4.2 GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## 5. Application Code {#application-code}

### 5.1 Main Application File

Create a file named `app.py` with the complete application code:

```python
"""
Tamil RAG Search Application with Qdrant Vector Database
A Retrieval-Augmented Generation system for Tamil documents using state-of-the-art models.

Models Used:
- Embedding: intfloat/multilingual-e5-large (best for Tamil)
- LLM: abhinand/tamil-llama-7b-instruct-v0.2 (bilingual Tamil-English)
- Vector DB: Qdrant (high-performance, scalable)
"""

import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue
)
import numpy as np
import json
from typing import List, Dict, Tuple
import time
import uuid

# Page configuration
st.set_page_config(
    page_title="Tamil RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        background-color: #4CAF50;
        color: white;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .metadata-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        background-color: #e3f2fd;
        color: #1976d2;
        font-size: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    """Load multilingual E5 embedding model - best for Tamil"""
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    return model


@st.cache_resource
def load_llm_model(use_4bit: bool = True):
    """
    Load Tamil-Llama model for generation
    Uses 4-bit quantization for efficiency
    """
    model_name = "abhinand/tamil-llama-7b-instruct-v0.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if use_4bit and torch.cuda.is_available():
        # 4-bit quantization for lower memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    
    return model, tokenizer


@st.cache_resource
def initialize_qdrant_client(use_memory: bool = True, persist_path: str = "./qdrant_storage"):
    """
    Initialize Qdrant client
    - use_memory=True: In-memory mode (fast, non-persistent)
    - use_memory=False: Persistent storage mode
    """
    if use_memory:
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(path=persist_path)
    
    return client


class TamilRAGSystem:
    """RAG system optimized for Tamil documents using Qdrant"""
    
    def __init__(self, embedding_model, llm_model, tokenizer, qdrant_client):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.client = qdrant_client
        self.collection_name = "tamil_documents"
        self.vector_size = 1024  # multilingual-e5-large dimension
        
    def create_embeddings(self, texts: List[str], prefix: str = "passage: ") -> np.ndarray:
        """
        Create embeddings with E5 prefix for better retrieval
        E5 models require task-specific prefixes
        """
        prefixed_texts = [prefix + text for text in texts]
        embeddings = self.embedding_model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text with overlap for better context preservation
        Tamil sentences can be longer, so we use generous chunk size
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                return True
            return False
        except Exception as e:
            st.error(f"Error creating collection: {str(e)}")
            return False
    
    def index_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Index documents in Qdrant
        documents: [{"title": "...", "content": "...", "category": "..."}, ...]
        """
        # Create collection
        self.create_collection()
        
        # Prepare points for Qdrant
        points = []
        point_id = 0
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc['content'])
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create embedding
                embedding = self.create_embeddings([chunk])[0]
                
                # Create point with payload
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Use UUID for unique IDs
                    vector=embedding.tolist(),
                    payload={
                        "title": doc['title'],
                        "content": chunk,
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "category": doc.get('category', 'general'),
                        "language": doc.get('language', 'mixed')
                    }
                )
                points.append(point)
                point_id += 1
        
        # Upload to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        return len(points)
    
    def retrieve(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k relevant chunks from Qdrant
        Supports optional category filtering
        """
        try:
            # Create query embedding with query prefix
            query_embedding = self.create_embeddings([query], prefix="query: ")[0]
            
            # Build filter if category specified
            query_filter = None
            if category_filter and category_filter != "All":
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category_filter)
                        )
                    ]
                )
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append((result.payload, result.score))
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context: str, language: str = "tamil") -> str:
        """
        Generate answer using Tamil-Llama with proper chat formatting
        """
        # Format prompt in ChatML style used by Tamil-Llama
        if language.lower() == "tamil":
            system_msg = "நீங்கள் ஒரு உதவிகரமான AI உதவியாளர். வழங்கப்பட்ட சூழலின் அடிப்படையில் கேள்விகளுக்கு பதிலளிக்கவும்."
        else:
            system_msg = "You are a helpful AI assistant. Answer questions based on the provided context."
        
        prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
Context: {context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        return response
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the Qdrant collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }
        except:
            return {}


def main():
    st.title("🔍 Tamil RAG Search System")
    st.markdown("### Powered by Tamil-Llama, Multilingual E5 & Qdrant")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Qdrant settings
        st.subheader("🗄️ Vector Database")
        use_memory = st.checkbox("Use in-memory mode", value=True, 
                                 help="In-memory: Fast but non-persistent. Uncheck for persistent storage.")
        
        if not use_memory:
            persist_path = st.text_input("Storage path", value="./qdrant_storage")
        else:
            persist_path = None
        
        st.markdown("---")
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Number of results", 1, 10, 3)
        language = st.selectbox("Response language", ["Tamil", "English", "Bilingual"])
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.markdown("""
        **Embedding Model:**  
        `intfloat/multilingual-e5-large`
        - 1024 dimensions
        - Best for Tamil cross-lingual retrieval
        
        **LLM Model:**  
        `abhinand/tamil-llama-7b-instruct-v0.2`
        - 7B parameters
        - Bilingual (Tamil + English)
        
        **Vector DB:**  
        `Qdrant`
        - High-performance vector search
        - COSINE similarity
        - Supports filtering & metadata
        """)
        
        st.markdown("---")
        use_4bit = st.checkbox("Use 4-bit quantization", value=True, 
                               help="Reduces memory usage, slight quality trade-off")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.indexed = False
        st.session_state.qdrant_client = None
    
    # Initialize Qdrant client if needed
    if st.session_state.qdrant_client is None:
        with st.spinner("Initializing Qdrant client..."):
            st.session_state.qdrant_client = initialize_qdrant_client(
                use_memory=use_memory,
                persist_path=persist_path if not use_memory else "./qdrant_storage"
            )
    
    # Document upload/input section
    st.header("📄 Document Management")
    
    tab1, tab2, tab3 = st.tabs(["Sample Documents", "Upload Custom", "Collection Stats"])
    
    with tab1:
        # Sample Tamil documents with categories
        sample_docs = [
            {
                "title": "தமிழ் இலக்கியம்",
                "content": """தமிழ் இலக்கியம் உலகின் பழமையான இலக்கியங்களில் ஒன்றாகும். சங்க இலக்கியம், திருக்குறள், சிலப்பதிகாரம், மணிமேகலை போன்ற பல சிறந்த படைப்புகள் தமிழில் உள்ளன. 
                இவை தமிழ் மக்களின் வாழ்க்கை முறை, கலாச்சாரம், மற்றும் தத்துவங்களை விளக்குகின்றன. சங்க கால கவிதைகள் காதல், போர், மற்றும் அறநெறி பற்றி பேசுகின்றன. 
                திருக்குறள் உலகளாவிய நெறிமுறைகளை எளிமையாக விளக்குகிறது. சிலப்பதிகாரம் ஒரு காவியமாகும், இது காதல் மற்றும் நீதி பற்றி பேசுகிறது.""",
                "category": "literature",
                "language": "tamil"
            },
            {
                "title": "Tamil Nadu History",
                "content": """Tamil Nadu has a rich historical heritage spanning over 2000 years. The Chola, Chera, and Pandya dynasties ruled the region and contributed significantly to art, architecture, and literature. 
                The temples built during these periods showcase magnificent Dravidian architecture. The Brihadeeswarar Temple in Thanjavur is a UNESCO World Heritage site. 
                The state has been a center of Tamil culture and learning throughout history. Ancient trade routes connected Tamil Nadu to Rome, Greece, and Southeast Asia.""",
                "category": "history",
                "language": "english"
            },
            {
                "title": "தமிழ் மொழியின் சிறப்பு",
                "content": """தமிழ் மொழி ஒரு செம்மொழி என்று அங்கீகரிக்கப்பட்டுள்ளது. இது 2000 ஆண்டுகளுக்கு மேலான தொடர்ச்சியான இலக்கிய வரலாற்றைக் கொண்டுள்ளது. 
                தமிழில் உள்ள 247 எழுத்துக்கள் மிகவும் அறிவியல் பூர்வமாக வடிவமைக்கப்பட்டுள்ளன. உலகம் முழுவதும் சுமார் 8 கோடி மக்கள் தமிழ் பேசுகின்றனர். 
                தமிழ் இலக்கணம் தொல்காப்பியம் மூலம் விளக்கப்படுகிறது. இது உலகின் பழமையான இலக்கண நூல்களில் ஒன்றாகும்.""",
                "category": "language",
                "language": "tamil"
            },
            {
                "title": "தமிழ் கலாச்சாரம்",
                "content": """தமிழ் கலாச்சாரம் மிகவும் பண்டைமை வாய்ந்தது. பொங்கல், தீபாவளி, போன்ற திருவிழாக்கள் சிறப்பாக கொண்டாடப்படுகின்றன. 
                பரதநாட்டியம், கர்நாடக சங்கீதம் போன்ற கலை வடிவங்கள் தமிழகத்தில் வளர்ந்தன. தமிழ் உணவு வகைகள் மிகவும் பிரபலமானவை. 
                இட்லி, தோசை, சாம்பார் போன்றவை உலகம் முழுவதும் பிரபலமாகியுள்ளன.""",
                "category": "culture",
                "language": "tamil"
            }
        ]
        
        if st.button("📚 Load Sample Documents"):
            with st.spinner("Loading models and indexing documents..."):
                try:
                    # Load models
                    if st.session_state.rag_system is None:
                        embedding_model = load_embedding_model()
                        llm_model, tokenizer = load_llm_model(use_4bit=use_4bit)
                        st.session_state.rag_system = TamilRAGSystem(
                            embedding_model, llm_model, tokenizer, st.session_state.qdrant_client
                        )
                    
                    # Index documents
                    num_chunks = st.session_state.rag_system.index_documents(sample_docs)
                    st.session_state.indexed = True
                    
                    st.success(f"✅ Indexed {len(sample_docs)} documents ({num_chunks} chunks) in Qdrant")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("Upload your Tamil documents (JSON format)")
        st.markdown("""
        **Expected format:**
        ```json
        [
          {
            "title": "Document Title",
            "content": "Document content...",
            "category": "optional_category",
            "language": "tamil/english/mixed"
          }
        ]
        ```
        """)
        
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
        
        if uploaded_file is not None:
            try:
                custom_docs = json.load(uploaded_file)
                if st.button("Index Custom Documents"):
                    with st.spinner("Indexing in Qdrant..."):
                        if st.session_state.rag_system is None:
                            embedding_model = load_embedding_model()
                            llm_model, tokenizer = load_llm_model(use_4bit=use_4bit)
                            st.session_state.rag_system = TamilRAGSystem(
                                embedding_model, llm_model, tokenizer, st.session_state.qdrant_client
                            )
                        
                        num_chunks = st.session_state.rag_system.index_documents(custom_docs)
                        st.session_state.indexed = True
                        st.success(f"✅ Indexed {len(custom_docs)} documents ({num_chunks} chunks)")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab3:
        if st.session_state.indexed and st.session_state.rag_system:
            st.subheader("📊 Collection Statistics")
            stats = st.session_state.rag_system.get_collection_stats()
            
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Vectors", stats.get('total_points', 0))
                with col2:
                    st.metric("Vector Dimensions", stats.get('vector_size', 0))
                with col3:
                    st.metric("Distance Metric", stats.get('distance', 'N/A'))
            else:
                st.info("No collection statistics available yet.")
        else:
            st.info("Load documents first to view statistics.")
    
    # Search section
    st.header("🔍 Search")
    
    if not st.session_state.indexed:
        st.info("👆 Please load or upload documents first to start searching")
        return
    
    # Category filter
    categories = ["All", "literature", "history", "language", "culture", "general"]
    selected_