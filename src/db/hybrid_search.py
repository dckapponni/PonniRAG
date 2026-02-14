"""
Hybrid Search Module for Tamil Document Processing - OPTIMIZED FOR STREAMLIT
Provides vector and keyword-based search with LLM-powered answer generation.

KEY FIXES:
1. Uses @st.cache_resource for persistent model caching
2. Loads models on CPU to avoid CUDA OOM
3. Models persist across Streamlit reruns
4. Silent loading during queries (spinners only on first load)
5. Faster response times (< 5 seconds after initial load)
6. FIXED: Increased LLM response size to prevent truncation
"""
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
import os
import logging
from pathlib import Path
import streamlit as st

import torch
torch.set_grad_enabled(False)
from qdrant_client import models
import pandas as pd
import requests
import json
import asyncio
import threading
import httpx
import time
import torch
torch.set_grad_enabled(False)


USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "summary.csv"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = "tamil-llama"
SCORE_THRESHOLD = 0.8  # Minimum cosine similarity for dense vector search

_embed_lock = threading.Lock()
_author_system_cache = {}
_author_system_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_files():
    """Verify existence of critical files (CSV)."""
    logger.info("Verifying critical files...")
    if CSV_PATH.exists():
        logger.info(f"CSV found: {CSV_PATH}")
        logger.info(f"CSV size: {CSV_PATH.stat().st_size / 1024:.2f} KB")
    else:
        logger.error(f"CSV NOT FOUND: {CSV_PATH}")


verify_files()


# ============================================================================
# STREAMLIT-CACHED MODEL LOADERS (PERSISTENT ACROSS RERUNS)
# ============================================================================

@st.cache_resource(show_spinner=False)  # ✅ No spinner during queries
def get_embed_model():
    """
    Load and cache embedding model using Streamlit's cache_resource.
    This ensures the model loads ONCE and persists across all reruns.
    Spinner is disabled - will only show during preload_models().
    """
    logger.info("🔄 Loading embedding model (this happens only once)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    logger.info("✅ Embedding model loaded and cached")
    return model



@st.cache_resource(show_spinner=False) 
def get_qdrant_client() -> QdrantClient:
    """
    Get or initialize Qdrant client (singleton pattern with Streamlit caching).
    Spinner is disabled - will only show during preload_models().
    """
    logger.info("🔄 Connecting to Qdrant SERVER...")
    
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=False,
        timeout=30.0
    )
    
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Connected: {collection_info.points_count} points")
    
    return client
@st.cache_resource(show_spinner=False)
def get_csv_dataframe():
    """Load CSV once and cache it."""
    if not CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource(show_spinner=False)
def get_csv_embeddings():
    """
    Precompute embeddings for CSV rows.
    Cached permanently like embedding model.
    """
    df = get_csv_dataframe()
    model = get_embed_model()

    if df.empty:
        return []

    texts = []
    for _, row in df.iterrows():
        text = " | ".join([str(v) for v in row.values if pd.notna(v)])
        texts.append(text)

    embeddings = model.encode(
        [f"passage: {t}" for t in texts],
        show_progress_bar=False
    )

    return list(zip(texts, embeddings))


# ============================================================================
# EMBEDDING FUNCTIONS (USE CACHED MODELS)
# ============================================================================

def dense_embed_query(text: str):
    """Generate dense embedding for query text (thread-safe)."""
    model = get_embed_model()  # Gets cached model silently
    with _embed_lock:
        return model.encode(f"query: {text}").tolist()


def sparse_embed(text: str):
    """Generate sparse BM25-style embedding for text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    indices, values = [], []
    for token, freq in counts.items():
        indices.append(abs(hash(token)) % (2**31))
        values.append(float(freq))
    return models.SparseVector(indices=indices, values=values)

def search_csv_semantic(question: str, top_k: int = 5):
    """Semantic search over CSV rows."""
    csv_data = get_csv_embeddings()
    model = get_embed_model()

    if not csv_data:
        return []

    query_emb = model.encode(f"query: {question}")

    scored = []
    for text, emb in csv_data:
        score = float(torch.tensor(query_emb) @ torch.tensor(emb))
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [text for text, _ in scored[:top_k]]


# ============================================================================
# REST OF YOUR CODE (UNCHANGED)
# ============================================================================

def check_qdrant_health() -> Dict:
    """Check Qdrant database health and connectivity."""
    try:
        client = get_qdrant_client()  # Uses cached client silently
        
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            return {
                "healthy": True,
                "collection": COLLECTION_NAME,
                "points_count": collection_info.points_count,
                "message": "Qdrant server is healthy"
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": "collection_not_found",
                "message": f"Collection '{COLLECTION_NAME}' not found",
                "details": str(e),
                "action": "Create the collection on the server"
            }
    except Exception as e:
        return {
            "healthy": False,
            "error": "connection_failed",
            "message": "Failed to connect to Qdrant server",
            "details": str(e),
            "action": "Ensure Qdrant server is running on port 6333"
        }


def normalize_author_name(name: str) -> str:
    """Normalize author names by removing prefixes and common variations."""
    if not name:
        return ""
    
    prefixes_to_remove = [
        r'மு\.,?\s*',
        r'டாக்டர்\.?\s*',
        r'திரு\.,?\s*',
        r'திருமதி\.?\s*',
        r'Dr\.?\s*',
        r'Mr\.?\s*',
        r'Mrs\.?\s*',
    ]
    
    cleaned = name.strip()
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def flexible_author_match(search_name: str, csv_name: str) -> bool:
    """Flexible matching between search query and CSV author name."""
    search_normalized = normalize_author_name(search_name).lower()
    csv_normalized = normalize_author_name(csv_name).lower()
    
    if search_normalized == csv_normalized:
        return True
    
    if search_normalized in csv_normalized:
        return True
    
    if csv_normalized in search_normalized:
        return True
    
    special_cases = {
        'கலைஞர்': ['கருணாநிதி', 'மு.,கருணாநிதி', 'மு. கருணாநிதி'],
        'கருணாநிதி': ['கலைஞர்', 'மு.,கருணாநிதி', 'மு. கருணாநிதி'],
        'அண்ணா': ['அண்ணாதுரை', 'சி.என்.அண்ணாதுரை'],
        'அண்ணாதுரை': ['அண்ணா', 'சி.என்.அண்ணாதுரை'],
    }
    
    for key, variations in special_cases.items():
        if key in search_normalized:
            for variant in variations:
                if normalize_author_name(variant).lower() in csv_normalized:
                    return True
        if key in csv_normalized:
            for variant in variations:
                if normalize_author_name(variant).lower() in search_normalized:
                    return True
    
    return False


class EnhancedAuthorQuerySystem:
    """Robust author query system with improved CSV parsing and query detection."""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()
    
    def _load_csv(self):
        """Load CSV with multiple fallback strategies for robustness."""
        try:
            if not self.csv_path.exists():
                logger.error(f"CSV not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return
            
            logger.info(f"Loading CSV: {self.csv_path}")
            
            strategies = [
                ("Standard", {"encoding": "utf-8", "skipinitialspace": True}),
                ("Skip bad lines", {"encoding": "utf-8", "on_bad_lines": "skip", "skipinitialspace": True}),
                ("Python engine", {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"}),
                ("No quoting", {"encoding": "utf-8", "engine": "python", "quoting": 3}),
            ]
            
            loaded = False
            for strategy_name, kwargs in strategies:
                try:
                    self.df = pd.read_csv(self.csv_path, **kwargs)
                    logger.info(f"Loaded with: {strategy_name}")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"{strategy_name} failed: {str(e)[:60]}")
                    continue
            
            if not loaded or self.df is None or self.df.empty:
                logger.error("All loading strategies failed")
                self.df = pd.DataFrame()
                return
            
            self.df.columns = self.df.columns.str.strip()
            logger.info(f"Columns: {list(self.df.columns)}")
            logger.info(f"Rows: {len(self.df)}")
            
            self._fix_column_names()
            
            if 'ஆசிரியர்' in self.df.columns:
                self.df['ஆசிரியர்'] = self.df['ஆசிரியர்'].fillna('').astype(str).str.strip()
                before = len(self.df)
                self.df = self.df[self.df['ஆசிரியர்'] != '']
                logger.info(f"Authors: {before} to {len(self.df)} rows")
            else:
                logger.error("Missing 'ஆசிரியர்' column")
                return
            
            if 'தலைப்பு' in self.df.columns:
                self.df['தலைப்பு'] = self.df['தலைப்பு'].fillna('').astype(str).str.strip()
            
            if len(self.df) > 0:
                sample = self.df.iloc[0]
                logger.info(f"Sample: {sample.get('ஆசிரியர்', 'N/A')[:30]}")
            
        except Exception as e:
            logger.error(f"CSV error: {e}", exc_info=True)
            self.df = pd.DataFrame()
    
    def _fix_column_names(self):
        """Map alternate column names to standard Tamil names."""
        mappings = {
            'author': 'ஆசிரியர்',
            'Author': 'ஆசிரியர்',
            'title': 'தலைப்பு',
            'Title': 'தலைப்பு',
            'heading': 'தலைப்பு',
        }
        
        for old, new in mappings.items():
            if old in self.df.columns and new not in self.df.columns:
                self.df.rename(columns={old: new}, inplace=True)
                logger.info(f"Renamed '{old}' to '{new}'")

    def detect_query_type(self, question: str) -> str:
        """
        Enhanced detection with focus on preventing LLM hallucination.
        These patterns should trigger CSV-only responses.
        """
        q = question.lower()

        ponni_variations = [
            'பொன்னி', 'பொன்னியில்', 'பொன்னியின்', 'பொன்னிக்கு', 
            'பொன்னியை', 'பொன்னியால்', 'பொன்னியுடன்',
            'ponni', 'ponni magazine', 'ponni issue'
        ]
        
        has_ponni = any(variation in q for variation in ponni_variations)

        # ✅ TOPIC AUTHOR PATTERNS - These ask "who wrote X?"
        topic_patterns = [
            'யார் எழுதிய',
            'யார் எழுதினார்',
            'யார் எழுதியது',
            'யார் எழுதியவர்',
            'யார் இயற்றியவர்',
            'எழுதியவர் யார்',
            'எழுதியது யார்',
            'ஆசிரியர் யார்',
            'எழுத்தாளர் யார்',
            'who wrote',
            'who is the author',
            'author of',
            'written by',
        ]

        if any(p in q for p in topic_patterns):
            return 'topic_author'

        # ✅ AUTHOR TOPICS PATTERNS - These ask "what did X write?"
        known_authors = [
            'கலைஞர்', 'கருணாநிதி', 'பெரியார்', 'அண்ணா', 'அண்ணாதுரை',
            'நக்கீரன்', 'பாரதிதாசன்', 'புதுமைப்பித்தன்', 'அகிலன்', 
            'கண்ணதாசன்', 'தங்கமணி', 'நாச்சியப்பன்', 'நாரா',  # ✅ Added common authors
        ]
        has_author = any(author in q for author in known_authors)
        
        author_action_patterns = [
            'என்ன எழுதினார்',
            'என்னென்ன எழுதினார்',
            'எழுதியது என்ன',
            'எழுதியவை',
            'எழுதிய தலைப்பு',
            'எழுதிய தலைப்புகள்',
            'எழுதிய கட்டுரைகள்',
            'எழுதிய தொடர்',  # ✅ NEW: Detect series queries
            'எழுதிய தொடரின்',  # ✅ NEW
            'தொடரின் பெயர்',  # ✅ NEW: "series name" pattern
            'தொடர் பெயர்',  # ✅ NEW
            'பற்றி எழுதினார்',
            'எந்த தலைப்புகள்',
            'அவர் எழுதிய',
            'படைப்புகள்',  # ✅ CRITICAL: "works" pattern
            'படைப்புகளை',  # ✅ CRITICAL
            'படைப்புகளைப் பட்டியலிடுக',  # ✅ CRITICAL: "list works"
            'பட்டியலிடுக',  # ✅ CRITICAL: "list"
            'பட்டியல்',  # ✅ CRITICAL
            'list of writings',
            'articles by',
            'works of',
            'what did write',
            'list works',
            'படைப்பு'  # ✅ CRITICAL: singular "work"
        ]

        if has_author or any(p in q for p in author_action_patterns):
            return 'author_topics'
        
        # LIST ALL AUTHORS PATTERNS
        list_patterns = [
            'எழுத்தாளர்கள் யார்',
            'ஆசிரியர்கள் யார்',
            'எழுத்தாளர்கள் பட்டியல்',
            'ஆசிரியர்கள் பட்டியல்',
            'அனைத்து எழுத்தாளர்கள்',
            'எழுதியவர்கள்',
            'எழுதிய ஆசிரியர்கள்',
            'who are the authors',
            'list all authors',
        ]

        if has_ponni and any(p in q for p in list_patterns):
            return 'list_all_authors'
        
        if any(p in q for p in list_patterns):
            return 'list_all_authors'

        return 'none'
        
    def extract_entity(self, question: str, query_type: str) -> str:
        """
        Enhanced entity extraction with better noise word removal.
        ✅ FIXED: Now properly removes "தொடரின்", "கதையின்" etc. from extracted topics
        """
        q = question.strip()
        
        # ✅ Normalize Ponni variations
        ponni_variations = [
            'பொன்னியில்', 'பொன்னியின்', 'பொன்னிக்கு', 
            'பொன்னியை', 'பொன்னியால்', 'பொன்னியுடன்',
            'ponni magazine', 'ponni issue'
        ]
        
        for variation in ponni_variations:
            q = q.replace(variation, 'பொன்னி')
        q = q.replace('ponni', 'பொன்னி')
        
        if query_type == 'author_topics':
            known_authors = {
                'கலைஞர்': 'கருணாநிதி',
                'கருணாநிதி': 'கருணாநிதி',
                'பெரியார்': 'பெரியார்',
                'அண்ணா': 'அண்ணாதுரை',
                'அண்ணாதுரை': 'அண்ணாதுரை',
                'நக்கீரன்': 'நக்கீரன்',
                'பாரதிதாசன்': 'பாரதிதாசன்',
                'புதுமைப்பித்தன்': 'புதுமைப்பித்தன்',
                'அகிலன்': 'அகிலன்',
                'கண்ணதாசன்': 'கண்ணதாசன்',
                'நாச்சியப்பன்': 'நாச்சியப்பன்',  # ✅ Added
                'நாரா': 'நாச்சியப்பன்',  # ✅ Maps நாரா to full name
            }
            
            for pattern, canonical in known_authors.items():
                if pattern in q:
                    return canonical
            
            patterns = [
                r'(கலைஞர்\s*கருணாநிதி)',
                r'(மு\.,?\s*கருணாநிதி)',
                r'([\u0B80-\u0BFF]+\s+[\u0B80-\u0BFF]+)(?=\s+எழுதி)',
                r'([\u0B80-\u0BFF]+)(?=\s+எழுதி)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, q, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    return normalize_author_name(extracted)
            
            noise = [
                'யார் எழுதினார்',
                'எழுதியவர் யார்',
                'ஆசிரியர் யார்',
                'என்ற நூலை',
                'எழுதியவர்',
                'பற்றி',
                'இதழில்',
                'இதழ்',
                'பொன்னி',
                'என்ற',
                'கட்டுரையை',
                'கதையை',
                'கட்டுரை',
                'எழுதிய',
                'யார்',
                'படைப்புகளைப்',  # ✅ NEW: Remove list keywords
                'படைப்புகளை',  # ✅ NEW
                'படைப்புகள்',  # ✅ NEW
                'பட்டியலிடுக',  # ✅ NEW
                'பட்டியல்',  # ✅ NEW
                'ன்',  # ✅ NEW: possessive marker
                'இன்',  # ✅ NEW
            ]
            
            for word in noise:
                q = re.sub(word, '', q, flags=re.IGNORECASE)
            
            words = re.findall(r'[\u0B80-\u0BFF]+', q)
            words = [normalize_author_name(w) for w in words if len(w) > 2]
            
            return ' '.join(words) if words else ''
        
        elif query_type == 'topic_author':
            # ✅ ENHANCED: Better noise word removal for topic extraction
            noise = [
                'யார் எழுதினார்',
                'எழுதியவர் யார்',
                'ஆசிரியர் யார்',
                'என்ற நூலை',
                'எழுதியவர்',
                'பற்றி',
                'இதழில்',
                'இதழ்',
                'பொன்னி',
                'என்ற',
                'கட்டுரையை',
                'கதையை',
                'கட்டுரை',
                'எழுதிய',
                'யார்',
                # ✅ NEW: Add common title suffixes to noise
                'தொடரின்',
                'தொடர்',
                'கதையின்',
                'நாவலின்',
                'கவிதையின்',
            ]
            
            for word in noise:
                q = re.sub(word, '', q, flags=re.IGNORECASE)
            
            # Remove quotes
            q = q.replace("'", "").replace('"', '')
            
            q = re.sub(r'\s+', ' ', q).strip()
            words = re.findall(r'[\u0B80-\u0BFF]+|[a-zA-Z]+', q)
            words = [w for w in words if len(w) > 2]
            
            return ' '.join(words) if words else ''
        
        return ''

            
    def list_all_authors(self) -> Dict:
        """List all unique authors with article counts."""
        if self.df is None or self.df.empty:
            return {
                "type": "list_all_authors",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "authors": []
            }
        
        if 'ஆசிரியர்' not in self.df.columns:
            return {
                "type": "list_all_authors",
                "success": False,
                "message": "ஆசிரியர் column இல்லை",
                "authors": []
            }
        
        counts = self.df['ஆசிரியர்'].value_counts()
        
        authors = [
            {"name": author, "count": int(count)}
            for author, count in counts.items()
        ]
        
        return {
            "type": "list_all_authors",
            "success": True,
            "total_authors": len(authors),
            "total_articles": len(self.df),
            "authors": authors
        }
    
    def get_topics_by_author(self, author_name: str) -> Dict:
        """Get all articles/topics by a specific author."""
        if self.df is None or self.df.empty:
            return {
                "type": "author_topics",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }
        
        matches = self.df[
            self.df['ஆசிரியர்'].apply(lambda x: flexible_author_match(author_name, str(x)))
        ]
        
        if matches.empty:
            return {
                "type": "author_topics",
                "success": False,
                "author": author_name,
                "message": f"'{author_name}' கண்டுபிடிக்க முடியவில்லை",
                "articles": []
            }
        
        articles = []
        for _, row in matches.iterrows():
            article = {
                "title": row.get('தலைப்பு', ''),
                "author": row.get('ஆசிரியர்', '')
            }
            
            for col in ['ஆண்டு', 'இதழ்', 'ச.எ.', 'வ.எ.']:
                if col in row and pd.notna(row[col]):
                    try:
                        article[col] = int(row[col]) if col in ['ஆண்டு', 'ச.எ.', 'வ.எ.'] else str(row[col])
                    except:
                        article[col] = str(row[col])
            
            articles.append(article)
        
        return {
            "type": "author_topics",
            "success": True,
            "author": author_name,
            "matched_author": matches.iloc[0]['ஆசிரியர்'],
            "count": len(articles),
            "articles": articles
        }
    
    def get_author_by_topic(self, topic: str) -> Dict:
        """
        Find authors who wrote about a specific topic.
        ✅ ENHANCED: Now uses partial matching and handles common suffixes
        """
        if self.df is None or self.df.empty:
            return {
                "type": "topic_author",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }
        
        # ✅ NEW: Clean up common suffixes that don't appear in CSV titles
        topic_cleaned = topic
        suffixes_to_remove = [
            'தொடரின்',
            'தொடர்',
            'கதையின்',
            'கதை',
            'நாவலின்',
            'நாவல்',
            'கட்டுரையின்',
            'கட்டுரை',
            'கவிதையின்',
            'கவிதை',
            'series',
            'story',
            'novel',
            'article',
            'poem'
        ]
        
        for suffix in suffixes_to_remove:
            # Remove suffix if it appears at the end
            if topic_cleaned.endswith(suffix):
                topic_cleaned = topic_cleaned[:-len(suffix)].strip()
            # Also try with space before suffix
            if topic_cleaned.endswith(' ' + suffix):
                topic_cleaned = topic_cleaned[:-(len(suffix)+1)].strip()
        
        # Remove quotes and extra spaces
        topic_cleaned = topic_cleaned.replace("'", "").replace('"', '').strip()
        
        # ✅ STRATEGY 1: Try exact match first (after cleaning)
        matches = self.df[
            self.df['தலைப்பு'].str.contains(topic_cleaned, case=False, na=False, regex=False)
        ]
        
        # ✅ STRATEGY 2: If no match, try word-by-word partial matching
        if matches.empty and len(topic_cleaned.split()) > 1:
            # Split topic into words and search for all words present
            words = topic_cleaned.split()
            
            def contains_all_words(title):
                if pd.isna(title):
                    return False
                title_lower = str(title).lower()
                return all(word.lower() in title_lower for word in words)
            
            matches = self.df[self.df['தலைப்பு'].apply(contains_all_words)]
        
        # ✅ STRATEGY 3: If still no match, try each word individually (most lenient)
        if matches.empty:
            words = topic_cleaned.split()
            if words:  # Try matching with the longest/most significant word
                main_word = max(words, key=len)  # Get longest word
                matches = self.df[
                    self.df['தலைப்பு'].str.contains(main_word, case=False, na=False, regex=False)
                ]
        
        if matches.empty:
            return {
                "type": "topic_author",
                "success": False,
                "topic": topic,
                "cleaned_topic": topic_cleaned,  # ✅ Show what was actually searched
                "message": f"'{topic}' தலைப்பு கண்டுபிடிக்க முடியவில்லை. தேடிய சொல்: '{topic_cleaned}'",
                "articles": [],
                "suggestion": "தலைப்பை முழுமையாக குறிப்பிடவும் அல்லது முக்கிய சொற்களை மட்டும் பயன்படுத்தவும்"
            }
        
        articles = []
        for _, row in matches.iterrows():
            article = {
                "title": row.get('தலைப்பு', ''),
                "author": row.get('ஆசிரியர்', '')
            }
            
            for col in ['ஆண்டு', 'இதழ்', 'ச.எ.', 'வ.எ.']:
                if col in row and pd.notna(row[col]):
                    try:
                        article[col] = int(row[col]) if col in ['ஆண்டு', 'ச.எ.', 'வ.எ.'] else str(row[col])
                    except:
                        article[col] = str(row[col])
            
            articles.append(article)
        
        return {
            "type": "topic_author",
            "success": True,
            "topic": topic,
            "cleaned_topic": topic_cleaned,  # ✅ Show what was searched
            "count": len(articles),
            "articles": articles
        }


def format_author_list(result: Dict) -> str:
    """Format list of all authors in simple numbered list format."""
    if not result['success']:
        return f"Error: {result['message']}"
    
    lines = [
        "பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்",
        f"மொத்த எழுத்தாளர்கள்: {result['total_authors']} | மொத்த கட்டுரைகள்: {result['total_articles']}",
        ""
    ]
    
    sorted_authors = sorted(result['authors'], key=lambda x: x['count'], reverse=True)
    
    for idx, author in enumerate(sorted_authors, 1):
        lines.append(f"{idx}. {author['name']} ({author['count']} கட்டுரைகள்)")
    
    return '\n'.join(lines)


def format_author_topics(result: Dict) -> str:
    """
    Format topics by author in simple numbered list format.
    """
    if not result['success']:
        return f"Error: {result['message']}"
    
    lines = [
        f"எழுத்தாளர்: {result.get('matched_author', result['author'])}",
        f"மொத்த படைப்புகள்: {result['count']}",
        ""
    ]
    
    # Sort articles by year, then issue
    sorted_articles = sorted(
        result['articles'], 
        key=lambda x: (x.get('ஆண்டு', 9999), str(x.get('இதழ்', '')))
    )
    
    for idx, article in enumerate(sorted_articles, 1):
        parts = [f"தலைப்பு: {article.get('title', '-')}"]
        parts.append(f"ஆசிரியர்: {article.get('author', '-')}")
        
        if 'ஆண்டு' in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")
        
        if 'இதழ்' in article:
            parts.append(f"இதழ்: {article['இதழ்']}")
        
        # Removed வ.எ. and ச.எ. (மலர்) from display
        
        lines.append(f"{idx}. {' | '.join(parts)}")
    
    return '\n'.join(lines)

def format_topic_authors(result: Dict) -> str:
    """
    Format topic authors in simple numbered list format.
    """
    if not result['success']:
        msg = f"Error: {result['message']}"
        if 'suggestion' in result:
            msg += f"\n\nசிபாரிசு: {result['suggestion']}"
        return msg
    
    lines = [
        f"தலைப்பு: '{result['topic']}' பற்றிய கட்டுரைகள்",
    ]
    
    # Show cleaned topic if different from original
    if result.get('cleaned_topic') and result['cleaned_topic'] != result['topic']:
        lines.append(f"(தேடிய சொல்: '{result['cleaned_topic']}')")
    
    lines.append(f"கண்டுபிடிக்கப்பட்டவை: {result['count']}")
    lines.append("")
    
    for idx, article in enumerate(result['articles'], 1):
        parts = [f"தலைப்பு: {article.get('title', '-')}"]
        parts.append(f"ஆசிரியர்: {article.get('author', '-')}")
        
        if 'ஆண்டு' in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")
        
        if 'இதழ்' in article:
            parts.append(f"இதழ்: {article['இதழ்']}")
        
        # Removed வ.எ. and ச.எ. (மலர்) from display
        
        lines.append(f"{idx}. {' | '.join(parts)}")
    
    return '\n'.join(lines)

def detect_issue_count_query(question: str) -> bool:
    """Detect if user is asking about issue count."""
    q = question.lower()
    
    patterns = [
        'இதழ் எண்ணிக்கை',
        'எத்தனை இதழ்',
        'இதழ்கள் எத்தனை',
        'மொத்த இதழ்',
        'இதழ் count',
        'issue count',
        'how many issues',
        'number of issues',
        'total issues',
        'இதழ் பட்டியல்',
        'இதழ்களின் பட்டியல்'
    ]
    
    return any(pattern in q for pattern in patterns)


def get_issue_count(csv_path: str) -> Dict:
    """Get unique issue count from CSV with article statistics."""
    try:
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            return {
                "success": False,
                "message": "CSV கோப்பு கிடைக்கவில்லை",
                "count": 0,
                "issues": []
            }
        
        logger.info(f"Loading CSV for issue count: {csv_path}")
        
        df = None
        strategies = [
            {"encoding": "utf-8", "skipinitialspace": True},
            {"encoding": "utf-8", "on_bad_lines": "skip"},
            {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
        ]
        
        for kwargs in strategies:
            try:
                df = pd.read_csv(csv_path, **kwargs)
                break
            except:
                continue
        
        if df is None or df.empty:
            return {
                "success": False,
                "message": "CSV தரவை படிக்க முடியவில்லை",
                "count": 0,
                "issues": []
            }
        
        df.columns = df.columns.str.strip()
        
        issue_col = None
        for col in ['இதழ்', 'issue', 'Issue', 'doc_issue']:
            if col in df.columns:
                issue_col = col
                break
        
        if not issue_col:
            return {
                "success": False,
                "message": "இதழ் column கிடைக்கவில்லை",
                "count": 0,
                "issues": []
            }
        
        df[issue_col] = df[issue_col].fillna('').astype(str).str.strip()
        df_filtered = df[df[issue_col] != '']
        
        unique_issues = df_filtered[issue_col].unique()
        issue_counts = df_filtered[issue_col].value_counts()
        
        try:
            sorted_issues = sorted(unique_issues, key=lambda x: int(x) if x.isdigit() else x)
        except:
            sorted_issues = sorted(unique_issues)
        
        issues_list = []
        for issue in sorted_issues:
            article_count = int(issue_counts[issue])
            issues_list.append({
                "issue_number": issue,
                "article_count": article_count
            })
        
        return {
            "success": True,
            "count": len(unique_issues),
            "total_articles": len(df_filtered),
            "issues": issues_list
        }
        
    except Exception as e:
        logger.error(f"Error getting issue count: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"பிழை: {str(e)}",
            "count": 0,
            "issues": []
        }


def format_issue_count(result: Dict) -> str:
    """Format issue count result in simple numbered list format."""
    if not result['success']:
        return f"Error: {result['message']}"
    
    lines = [
        "பொன்னி இதழ்கள் விவரம்",
        f"மொத்த இதழ்கள்: {result['count']}",
        f"மொத்த கட்டுரைகள்: {result['total_articles']}",
        ""
    ]
    
    issues = result['issues']
    
    if len(issues) <= 20:
        for idx, issue_info in enumerate(issues, 1):
            lines.append(f"{idx}. இதழ்: {issue_info['issue_number']} | கட்டுரைகள்: {issue_info['article_count']}")
    else:
        lines.append("முதல் 10 இதழ்கள்:")
        for idx, issue_info in enumerate(issues[:10], 1):
            lines.append(f"{idx}. இதழ்: {issue_info['issue_number']} | கட்டுரைகள்: {issue_info['article_count']}")
        
        lines.append("")
        lines.append(f"... (மேலும் {len(issues) - 20} இதழ்கள்)")
        lines.append("")
        
        lines.append("கடைசி 10 இதழ்கள்:")
        for idx, issue_info in enumerate(issues[-10:], len(issues) - 9):
            lines.append(f"{idx}. இதழ்: {issue_info['issue_number']} | கட்டுரைகள்: {issue_info['article_count']}")
    
    lines.append("")
    
    if result['issues']:
        article_counts = [i['article_count'] for i in result['issues']]
        lines.append("புள்ளிவிவரம்:")
        lines.append(f"  • சராசரி கட்டுரைகள் ஒரு இதழுக்கு: {sum(article_counts) / len(article_counts):.1f}")
        lines.append(f"  • குறைந்தபட்ச கட்டுரைகள்: {min(article_counts)}")
        lines.append(f"  • அதிகபட்ச கட்டுரைகள்: {max(article_counts)}")
    
    return '\n'.join(lines)

def detect_start_year_query(question: str) -> bool:
    q = question.lower()
    patterns = [
        "எந்த ஆண்டு தொடங்கியது",
        "முதல் ஆண்டு",
        "தொடங்கிய ஆண்டு",
        "எப்போது தொடங்கியது",
        "start year",
        "when did ponni start",
    ]
    return any(p in q for p in patterns)

def get_start_year() -> str:
    df = get_csv_dataframe()

    if df.empty or 'ஆண்டு' not in df.columns:
        return "CSV தரவில் ஆண்டு தகவல் இல்லை."

    try:
        df['ஆண்டு'] = pd.to_numeric(df['ஆண்டு'], errors='coerce')
        min_year = int(df['ஆண்டு'].min())

        return (
            f"பொன்னி இதழ் {min_year} ஆம் ஆண்டு தொடங்கியது.\n\n"
        )
    except Exception as e:
        logger.error(f"Start year calculation error: {e}")
        return "ஆண்டு தகவலை கணக்கிட முடியவில்லை."

def handle_author_query(question: str, csv_path: str) -> Tuple[bool, str]:
    """
    ✅ ENHANCED: Now returns CSV data directly without LLM processing
    This prevents hallucinations for author/topic queries
    """
    if detect_start_year_query(question):
        logger.info("Detected start year query")
        return True, get_start_year()
    if detect_issue_count_query(question):
        logger.info("Detected issue count query")
        result = get_issue_count(csv_path)
        return True, format_issue_count(result)

    with _author_system_lock:
        if csv_path not in _author_system_cache:
            _author_system_cache[csv_path] = EnhancedAuthorQuerySystem(csv_path)
        system = _author_system_cache[csv_path]
    
    if system.df is None or system.df.empty:
        return False, ""
    
    query_type = system.detect_query_type(question)
    logger.info(f"Query type: {query_type}")
    
    if query_type == 'none':
        return False, ""
    
    # ✅ All these queries return CSV data directly - NO LLM
    if query_type == 'list_all_authors':
        result = system.list_all_authors()
        return True, format_author_list(result)
    
    elif query_type == 'author_topics':
        entity = system.extract_entity(question, 'author_topics')
        logger.info(f"Extracted author: '{entity}'")
        
        if not entity:
            return True, "Warning: எழுத்தாளர் பெயரை தெளிவாக குறிப்பிடவும்"
        
        result = system.get_topics_by_author(entity)
        return True, format_author_topics(result)
    
    elif query_type == 'topic_author':
        entity = system.extract_entity(question, 'topic_author')
        logger.info(f"Extracted topic: '{entity}'")
        
        if not entity:
            return True, "Warning: தலைப்பை தெளிவாக குறிப்பிடவும்"
        
        result = system.get_author_by_topic(entity)
        return True, format_topic_authors(result)
    
    return False, ""


def _csv_source() -> List[Dict]:
    """Return a source entry indicating the answer came from the article database."""
    return [{
        "volume": "பொன்னி கட்டுரை தரவுத்தளம்",
        "heading": "Article Database",
        "doc_issue": "",
        "content": "இந்த பதில் பொன்னி இதழ் கட்டுரை அட்டவணையில் இருந்து பெறப்பட்டது. "
                   "இது எழுத்தாளர், தலைப்பு மற்றும் இதழ் தகவல்களைக் கொண்டுள்ளது.",
        "word_count": 0,
        "chunks_merged": 0,
        "score": 1.0,
    }]


def is_author_question(question: str) -> bool:
    """Check if question is about authors."""
    keywords = ["author", "authors", "list", "எழுத்தாளர்", "எழுத்தாளர்கள்", "பட்டியல்"]
    return any(k in question.lower() for k in keywords)


def fetch_all_authors(client: QdrantClient) -> List[str]:
    """Fetch all unique authors from Qdrant database."""
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=10000, with_payload=True)
    authors = set()
    for p in points:
        payload = p.payload or {}
        if payload.get("type") == "author":
            author = payload.get("content", "").strip()
            if author:
                authors.add(author)
    return sorted(authors)


def format_authors_tamil(authors: List[str]) -> str:
    """Format author list in Tamil."""
    if not authors:
        return "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
    lines = ["பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்:\n"]
    for author in authors:
        lines.append(f"• {author}")
    return "\n".join(lines)


class HybridQdrantSearch:
    """Hybrid search combining dense and sparse vectors for optimal results."""
    
    def __init__(self, client: QdrantClient):
        self.client = client

    def search(self, query: str, limit: int = 30, score_threshold: float = SCORE_THRESHOLD):
        """
        Perform hybrid search using dense and sparse vectors.

        Executes a two-stage search combining dense embeddings for semantic similarity
        and sparse embeddings for keyword matching, then fuses results using RRF.

        Args:
            query (str): Search query string in Tamil or English
            limit (int, optional): Maximum number of results to return. Defaults to 30.
            score_threshold (float, optional): Minimum cosine similarity for dense
                vector results. Defaults to SCORE_THRESHOLD (0.8).

        Returns:
            List[ScoredPoint]: List of scored points from Qdrant with fused relevance scores
        """
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_embed_query(query),
                    using="dense",
                    filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="article"))]),
                    score_threshold=score_threshold,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_embed(query),
                    using="sparse",
                    filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="article"))]),
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points


def retrieve_all_chunks_for_document(client: QdrantClient, doc_id: str, doc_issue: str, volume: str) -> List[Dict]:
    """Retrieve all chunks for a specific document."""
    all_chunks = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="article")),
                    models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="metadata.doc_issue", match=models.MatchValue(value=doc_issue)),
                    models.FieldCondition(key="metadata.volume", match=models.MatchValue(value=volume)),
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
        )
        all_chunks.extend(points)
        if offset is None:
            break
    return all_chunks


def merge_consecutive_chunks(client: QdrantClient, points) -> List[Dict]:
    """Merge consecutive chunks from the same document."""
    seen_docs = set()
    merged_docs = []
    
    for p in points:
        payload = p.payload or {}
        if payload.get("type") != "article":
            continue
        
        metadata = payload.get("metadata", {})
        doc_key = (metadata.get("volume"), metadata.get("doc_id"), metadata.get("doc_issue"))
        
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        
        all_chunks = retrieve_all_chunks_for_document(
            client, 
            doc_id=metadata.get("doc_id"), 
            doc_issue=metadata.get("doc_issue"), 
            volume=metadata.get("volume")
        )
        
        if not all_chunks:
            continue
        
        chunk_data = []
        for chunk_point in all_chunks:
            chunk_payload = chunk_point.payload or {}
            chunk_data.append({
                "chunk_id": chunk_payload.get("chunk_id", 0),
                "content": chunk_payload.get("content", "").strip()
            })
        
        chunk_data.sort(key=lambda x: x["chunk_id"])
        full_content = " ".join(chunk["content"] for chunk in chunk_data)
        full_content = re.sub(r'\s+', ' ', full_content).strip()
        
        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', full_content))
        
        if word_count < 50:
            continue
        
        merged_docs.append({
            "volume": metadata.get("doc_id", "unknown"),
            "doc_id": metadata.get("doc_id", "unknown"),
            "doc_issue": metadata.get("doc_issue", "unknown"),
            "heading": metadata.get("title", ""),
            "author_name": metadata.get("author_name", ""),
            "content": full_content,
            "word_count": word_count,
            "chunk_count": len(chunk_data),
            "score": p.score,
        })
    
    merged_docs.sort(key=lambda x: x["score"], reverse=True)
    return merged_docs


def extract_key_facts(docs: List[Dict], question: str) -> List[Dict]:
    """Extract key facts from documents relevant to question."""
    facts = []
    q_keywords = set(re.findall(r'[\u0B80-\u0BFF]{2,}', question.lower()))
    
    for doc in docs[:15]:
        content = doc["content"]
        sentences = re.split(r'[.।!?]+', content)
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30:
                continue
            
            sent_lower = sent.lower()
            relevance = 0
            
            for kw in q_keywords:
                if kw in sent_lower:
                    relevance += 3
            
            if relevance >= 5:
                facts.append({
                    'sentence': sent,
                    'score': relevance,
                    'source_issue': doc.get('doc_issue', 'NA'),
                    'source_volume': doc.get('volume', 'NA')
                })
    
    facts.sort(key=lambda x: x['score'], reverse=True)
    return facts[:20]

TAMIL_ANSWER_SYSTEM_PROMPT = """நீங்கள் பொன்னி இதழ் தொடர்பான கேள்விகளுக்கு பதிலளிக்கும் ஒரு தமிழ் நிபுணர்.

உங்கள் பணி:
1. கொடுக்கப்பட்ட சூழல் (context) மற்றும் கேள்வியின் அடிப்படையில் விரிவான பதில் எழுதுக
2. பதில் 200 முதல் 500 சொற்கள் வரை இருக்க வேண்டும்
3. தெளிவான, எளிமையான, நடைமுறை தமிழில் எழுதுக
4. சூழலில் உள்ள தகவல்களை மட்டுமே பயன்படுத்துக – கற்பனையாக எதையும் சேர்க்காதீர்கள்
5. பதில் வாசிப்பதற்கு மிகவும் எளிதாகவும், நன்கு கட்டமைக்கப்பட்டதாகவும் இருக்க வேண்டும்
6. பதில் தொடங்கும் போதும் முடியும் போதும் எந்தச் சொலும் துண்டிக்கப்பட்டதாக இருக்கக் கூடாது

மிக முக்கியமான வடிவமைப்பு விதிகள் (Formatting Rules):
- கேள்வி **புள்ளிவாரியான (points-wise)** பதிலை எதிர்பார்க்குமானால்:
  * ஒவ்வொரு புள்ளியும் தனித்தனி வரியில் எழுதப்பட வேண்டும்
  * ஒரு புள்ளி முடிந்தவுடன் அடுத்த புள்ளி புதிய வரியில் தொடங்க வேண்டும்
  * புள்ளிகளுக்கிடையே சரியான வரி இடைவெளி இருக்க வேண்டும்
  * ஒரே புள்ளியில் பல கருத்துகளை கலக்கக் கூடாது

- கேள்வி **பத்திவாரியான (paragraph-wise)** பதிலை எதிர்பார்க்குமானால்:
  * ஒவ்வொரு பத்தியும் தனித்தனி வரியில் இருக்க வேண்டும்
  * ஒவ்வொரு பத்தியின் முன்பும் பின்பும் ஒரு காலி வரி (spacing) இருக்க வேண்டும்
  * மிக நீளமான ஒரே பத்தியாக எழுதக்கூடாது
  * ஒவ்வொரு பத்தியும் ஒரு முக்கிய கருத்தை மட்டும் விளக்க வேண்டும்

எழுதும் முறை:
- முதல் வாக்கியத்தில் கேள்விக்கான நேரடியான பதிலை தெளிவாக கூறுக
- அதன் பின்னர் விவரங்கள், விளக்கங்கள், எடுத்துக்காட்டுகளை ஒழுங்காக எழுதுக
- தேவையான இடங்களில் துணைத்தலைப்புகளை பயன்படுத்தலாம்
- இறுதியில் சுருக்கமான முடிவுரை எழுதலாம்

கவனிக்க வேண்டியவை:
- CSV உள்ளடக்கத்தை பயன்படுத்தாமல் பதில் எழுதக்கூடாது.
- CSV தகவல் தொடர்பில்லையெனில் அதனை தெளிவாக குறிப்பிட வேண்டும்.
- சூழலில் இல்லாத தகவல்களை எதையும் எழுதாதீர்கள்
- "சூழலின் படி", "ஆதாரத்தின் படி" போன்ற சொற்களை பயன்படுத்த வேண்டாம்
- வாசிப்பவரின் கண்களுக்கு சோர்வு வராத வகையில் பதிலை அமைக்க வேண்டும்

இப்போது, கீழே கொடுக்கப்பட்ட கேள்வி மற்றும் சூழலின் அடிப்படையில், மேலுள்ள அனைத்து விதிகளையும் கட்டாயமாக பின்பற்றி, தெளிவாகவும் வாசிக்க எளிதாகவும் விரிவான பதிலை எழுதுக."""


def generate_llm_answer(question: str, context: str, csv_context: str, max_words: int = 500) -> str:
    """
    Generate LLM answer using Ollama API.
    """
    try:
        prompt = f"""{TAMIL_ANSWER_SYSTEM_PROMPT}

கேள்வி:
{question}

========================
CSV உள்ளடக்கம்:
{csv_context}
========================

========================
ஆவண சூழல்:
{context}
========================

விரிவான பதில் (200-500 சொற்கள்):
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 1200,
                "num_ctx": 4096,
                "num_gpu": 999,
            }
        }

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()

        answer = re.sub(r'\s+', ' ', answer)

        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        logger.info(f"Ollama answer generated: {word_count} words")

        return answer

    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return ""


async def generate_llm_answer_async(question: str, context: str, csv_context: str, max_words: int = 500) -> str:
    """
    Async version of generate_llm_answer using httpx.

    Non-blocking LLM call that frees the event loop while waiting for Ollama,
    allowing other requests to be served concurrently.
    """
    try:
        prompt = f"""{TAMIL_ANSWER_SYSTEM_PROMPT}

கேள்வி:
{question}

========================
CSV உள்ளடக்கம்:
{csv_context}
========================

========================
ஆவண சூழல்:
{context}
========================

விரிவான பதில் (200-500 சொற்கள்):
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 1200,
                "num_ctx": 4096,
                "num_gpu": 999,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=300.0,
            )
            response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()

        answer = re.sub(r'\s+', ' ', answer)

        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        logger.info(f"Ollama async answer generated: {word_count} words")

        return answer

    except Exception as e:
        logger.error(f"Ollama async generation failed: {e}")
        return ""


def generate_llm_answer_stream(question: str, context: str, csv_context: str):
    """
    Generate LLM answer using Ollama API with streaming.
    Yields individual token strings as they arrive from Ollama.
    """
    try:
        prompt = f"""{TAMIL_ANSWER_SYSTEM_PROMPT}

கேள்வி:
{question}

========================
CSV உள்ளடக்கம்:
{csv_context}
========================

========================
ஆவண சூழல்:
{context}
========================

விரிவான பதில் (200-500 சொற்கள்):
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 1200,
                "num_ctx": 4096,
                "num_gpu": 999,
            }
        }

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=300,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    except Exception as e:
        logger.error(f"Ollama streaming generation failed: {e}")
        yield ""


def generate_extractive_answer(facts: List[Dict], question: str) -> str:
    """Generate fallback extractive answer if LLM fails."""
    if not facts:
        return ""
    
    answer_sentences = []
    for fact in facts[:5]:
        sent = fact['sentence'].strip()
        sent = re.sub(r'__.*?__|பொன்னி களஞ்சியம்', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()
        if len(sent) > 30:
            answer_sentences.append(sent)
    
    if not answer_sentences:
        return ""
    
    answer = '. '.join(answer_sentences)
    if answer and answer[-1] not in '.!?।':
        answer += '.'
    
    return answer


def format_sources(merged_docs: List[Dict]) -> List[Dict]:
    """
    Format source documents for display.

    Prepares a list of source documents with truncated content and metadata
    for display to the user, removing duplicates and limiting content length.
    All documents that passed the score threshold are included.

    Args:
        merged_docs (List[Dict]): List of merged document dictionaries

    Returns:
        List[Dict]: List of formatted source dictionaries containing:
            - volume: Document volume number
            - heading: Article title/heading
            - doc_issue: Issue number
            - content: Truncated content (max 1500 chars)
            - word_count: Total word count of full document
            - chunks_merged: Number of chunks merged
            - score: Relevance score
    """
    sources = []
    seen_hashes = set()

    for doc in merged_docs:
        content_hash = hash(doc["content"][:200])
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        
        content = doc["content"]
        if len(content) > 1500:
            content = content[:1500].rsplit(' ', 1)[0] + "..."
        
        sources.append({
            "volume": doc["volume"],
            "heading": doc["heading"],
            "doc_issue": doc["doc_issue"],
            "content": content,
            "word_count": doc["word_count"],
            "chunks_merged": doc["chunk_count"],
            "score": doc["score"],
        })
    
    return sources


def format_answer_output(answer: str, sources: List[Dict]) -> str:
    """Format complete answer with sources for display."""
    lines = []
    
    lines.append("பதில்:")
    lines.append(answer)
    lines.append("")
    
    if sources:
        lines.append(f"\nஆதாரங்கள் ({len(sources)} ஆவணங்கள்):")
        lines.append("")
        
        for idx, source in enumerate(sources, 1):
            lines.append(f"ஆதாரம் {idx}")
            
            header_parts = [f"இதழ்: {source['doc_issue']}", f"மலர்: {source['volume']}"]
            if source['heading']:
                header_parts.append(f"தலைப்பு: {source['heading']}")
            lines.append(" • ".join(header_parts))
            
            lines.append(f"சொற்கள்: {source['word_count']} | பொருத்தம்: {source['score']:.3f}")
            lines.append("")
            lines.append(source['content'])
            lines.append("")
    
    return '\n'.join(lines)

def ask_question(question: str, return_formatted: bool = False, use_llm: bool = True) -> Dict:
    """
    Main question answering function with database health check and hybrid search.

    Primary entry point for processing user queries. Handles database health checks,
    author queries, vector search, document merging, LLM generation, and source formatting.
    Results are filtered by score threshold rather than a fixed top_k count.
    CSV queries are checked BEFORE vector search to return direct data without LLM.

    Args:
        question (str): User's question in Tamil or English.
        return_formatted (bool, optional): If True, return formatted string; if False,
                                        return dict. Defaults to False.
        use_llm (bool, optional): If True, use LLM for answer generation; if False,
                                use extractive fallback only. Defaults to True.

    Returns:
        dict or str: Depending on return_formatted:
            - If False (default): Dict with keys:
                - answer (str): Generated answer text
                - sources (list): List of source document dicts
                - query_type (str): Type of query handled (optional)
                - error (str/dict): Error details if failed (optional)
            - If True: Formatted string with answer and sources
    """
    t_start = time.time()
    health_status = check_qdrant_health()
    logger.info(f"[TIMING] health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        error_message = f"""
Database Error: {health_status['message']}

Details: {health_status.get('details', 'No additional details')}
Action: {health_status.get('action', 'Contact administrator')}

Error Type: {health_status['error']}
"""
        if return_formatted:
            return error_message
        return {
            "answer": error_message,
            "sources": [],
            "error": health_status
        }
    
    logger.info("Database is healthy - proceeding with query")
    
    # ✅ 2. CHECK CSV QUERIES FIRST (BEFORE VECTOR SEARCH)
    # This is CRITICAL - CSV queries should return immediately
    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, response = handle_author_query(question, str(CSV_PATH))
            
            if is_handled:
                logger.info("✅ Handled as CSV author query - returning direct CSV data")
                if return_formatted:
                    return response
                return {
                    "answer": response,
                    "sources": _csv_source(),
                    "query_type": "author_csv",
                    "csv_direct": True,
                }
        except Exception as e:
            logger.error(f"CSV query error: {e}")
    
    # ✅ 3. IF NOT CSV QUERY, PROCEED WITH VECTOR SEARCH + LLM
    try:
        client = get_qdrant_client()

        if is_author_question(question):
            authors = fetch_all_authors(client)
            answer = format_authors_tamil(authors) if authors else "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, _csv_source())
            return {"answer": answer, "sources": _csv_source()}

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=50, score_threshold=SCORE_THRESHOLD)
        logger.info(f"[TIMING] hybrid_search: {time.time() - t0:.2f}s ({len(results)} results)")

        if not results:
            answer = "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        t0 = time.time()
        merged_docs = merge_consecutive_chunks(client, results)
        logger.info(f"[TIMING] merge_chunks: {time.time() - t0:.2f}s ({len(merged_docs)} docs)")

        if not merged_docs:
            answer = "போதுமான தகவல்கள் இல்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}
        
        logger.info(f"Merged into {len(merged_docs)} documents")
        
        context_parts = []
        for idx, doc in enumerate(merged_docs[:3], 1):
            context_parts.append(f"ஆவணம் {idx}: {doc['content'][:400]}")

        # CSV semantic context
        csv_results = search_csv_semantic(question, top_k=3)
        logger.info("------ CSV Rows Sent To LLM ------")
        for row in csv_results:
            logger.info(row)
        logger.info("----------------------------------")
        
        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])
        else:
            csv_context = "தொடர்புடைய CSV தகவல் இல்லை."

        context = "\n\n".join(context_parts)

        answer = ""
        if use_llm:
            t0 = time.time()
            answer = generate_llm_answer(question, context, csv_context)
            logger.info(f"[TIMING] ollama_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < 150:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            logger.info(f"Extracted {len(facts)} facts")
            answer = generate_extractive_answer(facts, question)
        
        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs)
        logger.info(f"[TIMING] TOTAL ask_question: {time.time() - t_start:.2f}s | {len(sources)} sources")

        if return_formatted:
            return format_answer_output(answer, sources)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        error_msg = f"Query error: {str(e)}"
        logger.error(error_msg)
        if return_formatted:
            return f"Error: {error_msg}"
        return {"answer": error_msg, "sources": [], "error": str(e)}


async def ask_question_async(question: str, return_formatted: bool = False, use_llm: bool = True) -> Dict:
    """
    Async version of ask_question for FastAPI concurrent request handling.

    Uses asyncio.to_thread for sync I/O operations (Qdrant, embeddings) and
    httpx async client for Ollama LLM calls. This allows multiple user requests
    to be processed concurrently without blocking the event loop.
    """
    t_start = time.time()
    health_status = await asyncio.to_thread(check_qdrant_health)
    logger.info(f"[TIMING] async health_check: {time.time() - t_start:.2f}s")

    if not health_status["healthy"]:
        error_message = f"""
Database Error: {health_status['message']}

Details: {health_status.get('details', 'No additional details')}
Action: {health_status.get('action', 'Contact administrator')}

Error Type: {health_status['error']}
"""
        if return_formatted:
            return error_message
        return {
            "answer": error_message,
            "sources": [],
            "error": health_status
        }

    logger.info("Database is healthy - proceeding with async query")

    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, response = await asyncio.to_thread(
                handle_author_query, question, str(CSV_PATH)
            )

            if is_handled:
                logger.info("Handled as CSV author query")
                if return_formatted:
                    return response
                return {
                    "answer": response,
                    "sources": _csv_source(),
                    "query_type": "author_csv",
                }
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    try:
        client = await asyncio.to_thread(get_qdrant_client)

        if is_author_question(question):
            authors = await asyncio.to_thread(fetch_all_authors, client)
            answer = format_authors_tamil(authors) if authors else "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, _csv_source())
            return {"answer": answer, "sources": _csv_source()}

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = await asyncio.to_thread(
            searcher.search, question, 50, SCORE_THRESHOLD
        )
        logger.info(f"[TIMING] async hybrid_search: {time.time() - t0:.2f}s ({len(results)} results)")

        if not results:
            answer = "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        t0 = time.time()
        merged_docs = await asyncio.to_thread(
            merge_consecutive_chunks, client, results
        )
        logger.info(f"[TIMING] async merge_chunks: {time.time() - t0:.2f}s ({len(merged_docs)} docs)")

        if not merged_docs:
            answer = "போதுமான தகவல்கள் இல்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        context_parts = []
        for idx, doc in enumerate(merged_docs[:3], 1):
            context_parts.append(f"ஆவணம் {idx}: {doc['content'][:400]}")

        # CSV semantic context
        csv_results = await asyncio.to_thread(search_csv_semantic, question, 3)
        logger.info("------ CSV Rows Sent To LLM (async) ------")
        for row in csv_results:
            logger.info(row)
        logger.info("------------------------------------------")

        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])
        else:
            csv_context = "தொடர்புடைய CSV தகவல் இல்லை."

        context = "\n\n".join(context_parts)

        answer = ""
        if use_llm:
            t0 = time.time()
            answer = await generate_llm_answer_async(question, context, csv_context)
            logger.info(f"[TIMING] async ollama_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < 100:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)

        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs)
        logger.info(f"[TIMING] TOTAL ask_question_async: {time.time() - t_start:.2f}s | {len(sources)} sources")

        if return_formatted:
            return format_answer_output(answer, sources)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        error_msg = f"Query error: {str(e)}"
        logger.error(error_msg)
        if return_formatted:
            return f"Error: {error_msg}"
        return {"answer": error_msg, "sources": [], "error": str(e)}


def ask_question_stream(question: str):
    """
    Streaming version of ask_question.
    Yields dicts: {"type": "token", "content": str} for answer tokens,
    and {"type": "sources", "sources": list} at the end.
    For non-streamable responses (CSV queries, errors), yields complete answer as single token.
    """
    # 1. Check health
    health_status = check_qdrant_health()
    if not health_status["healthy"]:
        yield {"type": "token", "content": f"Database Error: {health_status['message']}"}
        yield {"type": "sources", "sources": []}
        return

    # 2. Check CSV queries first
    if CSV_PATH.exists():
        try:
            is_handled, response = handle_author_query(question, str(CSV_PATH))
            if is_handled:
                yield {"type": "token", "content": response}
                yield {"type": "sources", "sources": _csv_source()}
                return
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. Vector search + LLM streaming
    try:
        client = get_qdrant_client()

        if is_author_question(question):
            authors = fetch_all_authors(client)
            answer = format_authors_tamil(authors) if authors else "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
            yield {"type": "token", "content": answer}
            yield {"type": "sources", "sources": _csv_source()}
            return

        logger.info(f"Streaming search: {question[:60]}...")
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=50)

        if not results:
            yield {"type": "token", "content": "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."}
            yield {"type": "sources", "sources": []}
            return

        merged_docs = merge_consecutive_chunks(client, results)
        if not merged_docs:
            yield {"type": "token", "content": "போதுமான தகவல்கள் இல்லை."}
            yield {"type": "sources", "sources": []}
            return

        logger.info(f"Merged into {len(merged_docs)} documents")

        context_parts = []
        for idx, doc in enumerate(merged_docs[:3], 1):
            context_parts.append(f"ஆவணம் {idx}: {doc['content'][:400]}")

        csv_results = search_csv_semantic(question, top_k=3)
        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])
        else:
            csv_context = "தொடர்புடைய CSV தகவல் இல்லை."

        context = "\n\n".join(context_parts)

        # Stream LLM tokens
        token_count = 0
        for token in generate_llm_answer_stream(question, context, csv_context):
            token_count += 1
            yield {"type": "token", "content": token}

        # If streaming produced too few tokens, fall back to extractive
        if token_count < 10:
            logger.warning("Streaming produced too few tokens, using extractive fallback")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)
            if answer:
                yield {"type": "token", "content": answer}

        sources = format_sources(merged_docs)
        yield {"type": "sources", "sources": sources}

    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        yield {"type": "token", "content": f"Query error: {str(e)}"}
        yield {"type": "sources", "sources": []}


def preload_ollama_model():
    """
    Warm up the Ollama LLM by sending a minimal prompt.
    Loads the model into GPU memory so the first real query is fast.
    No Streamlit dependency - safe to call from FastAPI.
    """
    logger.info("Preloading Ollama model into memory...")
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": "hello",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": 1,
                "num_gpu": 99,
            }
        }
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        logger.info("Ollama model preloaded successfully")
    except Exception as e:
        logger.warning(f"Ollama model preload failed (non-fatal): {e}")


def preload_models():
    """
    Preload all models at app startup with visible spinners.
    Call this ONCE in your Streamlit app's initialization section.
    After this runs, all subsequent queries will be fast and silent.
    """
    logger.info("=" * 60)
    logger.info("PRELOADING MODELS FOR STREAMLIT")
    logger.info("=" * 60)

    with st.spinner("Loading embedding model..."):
        _ = get_embed_model()
        st.success("Embedding model loaded")

    with st.spinner("Connecting to Qdrant database..."):
        _ = get_qdrant_client()
        st.success("Qdrant connected")

    with st.spinner("Loading Ollama LLM..."):
        preload_ollama_model()
        st.success("Ollama LLM loaded")

    logger.info("=" * 60)
    logger.info("ALL MODELS READY - APP IS READY TO SERVE")
    logger.info("=" * 60)

    st.success("All models loaded successfully! Ready to answer queries.")