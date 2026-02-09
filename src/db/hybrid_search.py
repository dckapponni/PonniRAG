from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient,models
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
import os
import logging
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import json
import torch
torch.set_grad_enabled(False)

import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "summary.csv"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = "tamil-llama"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_files():
    """
    Verify existence of critical files (CSV).
    Checks if the summary CSV file exists and logs its size for diagnostic purposes.
    Called automatically on module import.
    """
    logger.info("Verifying critical files...")
    if CSV_PATH.exists():
        logger.info(f"CSV found: {CSV_PATH}")
        logger.info(f"CSV size: {CSV_PATH.stat().st_size / 1024:.2f} KB")
    else:
        logger.error(f"CSV NOT FOUND: {CSV_PATH}")


verify_files()


@st.cache_resource(show_spinner=False) 
def get_embed_model():
    """
    Load and cache embedding model using Streamlit's cache_resource.
    This ensures the model loads ONCE and persists across all reruns.
    Spinner is disabled - will only show during preload_models().
    """
    logger.info("Loading embedding model (this happens only once)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    logger.info(" Embedding model loaded and cached")
    return model



@st.cache_resource(show_spinner=False) 
def get_qdrant_client() -> QdrantClient:
    """
    Get or initialize Qdrant client (singleton pattern with Streamlit caching).

    Creates a connection to the Qdrant vector database server and caches the client
    instance for the entire Streamlit session. Subsequent calls reuse the connection.

    Returns:
        QdrantClient: Cached Qdrant client instance connected to server.
    """
    logger.info(" Connecting to Qdrant SERVER...")
    
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=False,
        timeout=30.0
    )
    
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f" Connected: {collection_info.points_count} points")
    
    return client


@st.cache_resource(show_spinner=False)  # ✅ No spinner during queries
def get_llm():
    """
    Generate dense embedding for query text.

    Creates a dense vector representation of query text using the cached E5 embedding
    model. Prefixes text with "query:" for E5 model's query-aware encoding.

    Args:
        text (str): Query text to embed.

    Returns:
        list: Dense embedding vector as list of floats (1024 dimensions for E5-large).

    Processing:
        - Retrieves cached embedding model silently
        - Prefixes text with "query:" for E5 model
        - Generates embedding on CPU
    """
    model = get_embed_model()  
    return model.encode(f"query: {text}").tolist()


@st.cache_resource(show_spinner=False)  # ✅ No spinner during queries
def get_qdrant_client() -> QdrantClient:
    """
    Generate sparse BM25-style embedding for text.

    Creates a sparse vector representation using term frequency (similar to BM25).
    Each unique token gets a hash-based index and frequency-based value.

    Args:
        text (str): Text to embed sparsely.

    Returns:
        models.SparseVector: Qdrant sparse vector with indices and values.
            - indices: Hash-based token IDs (0 to 2^31)
            - values: Term frequencies as floats

    Processing:
        1. Tokenizes text using word boundaries
        2. Converts to lowercase
        3. Counts term frequencies
        4. Hashes tokens to indices (modulo 2^31)
        5. Uses frequencies as values
    """
    logger.info("🔄 Connecting to Qdrant SERVER...")
    
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=False,
        timeout=30.0
    )
    
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"✅ Connected: {collection_info.points_count} points")
    
    return client


# ============================================================================
# EMBEDDING FUNCTIONS (USE CACHED MODELS)
# ============================================================================

def dense_embed_query(text: str):
    """Generate dense embedding for query text."""
    model = get_embed_model()  # Gets cached model silently
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



def check_qdrant_health() -> Dict:
    """
    Check Qdrant database health and connectivity.

    Verifies that the Qdrant server is running, accessible, and contains the required
    collection with data.

    Returns:
        dict: Health status containing:
            - healthy (bool): True if all checks pass
            - collection (str): Collection name if successful
            - points_count (int): Number of vectors if successful
            - message (str): Status message
            - error (str): Error type if failed ('collection_not_found', 'connection_failed')
            - details (str): Detailed error message if failed
            - action (str): Recommended action if failed
    """
    try:
        client = get_qdrant_client()  
        
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
    """
    Normalize author names by removing prefixes and common variations.

    Strips common prefixes (titles, honorifics) from Tamil and English author names
    to enable flexible matching.

    Args:
        name (str): Author name to normalize.

    Returns:
        str: Normalized name with prefixes removed and whitespace cleaned.
            Returns empty string if input is None or empty.
    """
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
    """
    Flexible matching between search query and CSV author name.

    Performs multi-level matching including exact match, substring match, and
    special case aliases for common Tamil authors.

    Args:
        search_name (str): Author name from search query.
        csv_name (str): Author name from CSV record.

    Returns:
        bool: True if names match (by any matching rule), False otherwise.
    """
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
    
    def __init__(self, csv_path: str):
        """
        Initialize enhanced author query system with CSV data loading.

        Creates an instance and loads the CSV file containing article metadata
        using multiple fallback strategies for robustness.
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()
    
    def _load_csv(self):
        """     
        Load CSV with multiple fallback strategies for robustness.

        Attempts to load CSV file using progressively more permissive parsing strategies
        until successful. Validates and cleans the loaded data.

        Loading Strategies (tried in order):
            1. Standard UTF-8 encoding
            2. Skip bad lines
            3. Python engine with error skip
            4. Python engine without quoting
        """
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
        """
        Map alternate column names to standard Tamil names.

        Renames English or alternate column names to standard Tamil column names
        for consistent access across different CSV formats.

        Column Mappings:
            - 'author', 'Author' → 'ஆசிரியர்'
            - 'title', 'Title', 'heading' → 'தலைப்பு'
        """
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
        Detect the type of query based on keywords and patterns.

        Analyzes the user's question to categorize it into one of four query types
        for appropriate handling.

        Args:
            question (str): User's question text.

        Returns:
            str: Query type identifier:
                - 'list_all_authors': List all authors in magazine
                - 'author_topics': Get articles by specific author
                - 'topic_author': Find author who wrote about topic
                - 'none': Not an author/topic query
        """
        q = question.lower()
        
        known_authors = [
            'கலைஞர்', 'கருணாநிதி', 'பெரியார்', 'அண்ணா', 'அண்ணாதுரை',
            'நக்கீரன்', 'பாரதிதாசன்', 'புதுமைப்பித்தன்', 'அகிலன்', 'கண்ணதாசன்'
        ]
        has_author = any(author in q for author in known_authors)
        
        list_patterns = [
            'எழுத்தாளர்கள் யார்', 'ஆசிரியர்கள் யார்',
            'எழுத்தாளர்களின் பட்டியல்', 'எழுத்தாளர்கள் பட்டியல்',
            'அனைத்து எழுத்தாளர்', 'எழுத்தாளர்கள் எல்லாம்',
            'list all authors', 'இதழில் எழுதிய ஆசிரியர்கள்',
            'இதழில் எழுதிய எழுத்தாளர்கள்','எழுத்தாளர்'
        ]
        
        if any(p in q for p in list_patterns) and not has_author:
            return 'list_all_authors'
        
        author_action_patterns = [
            'என்ன எழுதினார்', 'எழுதிய தலைப்பு', 'எழுதியது',
            'எந்த தலைப்பு', 'பற்றி எழுதினார்', 'எழுதியவற்றை',
            'எழுதிய கட்டுரைகள்', 'wrote what', 'articles by',
            'குறிப்பிடுக'
        ]
        
        if has_author or any(p in q for p in author_action_patterns):
            return 'author_topics'
        
        topic_patterns = [
            'யார் எழுதினார்', 'எழுதியவர் யார்', 'ஆசிரியர் யார்',
            'என்ற நூலை எழுதியவர்', 'என்ற கட்டுரை','கட்டுரை யார் எழுதினார்', 'தலைப்பு யார் எழுதினார்',
            'who wrote', 'author of'
        ]
        
        if any(p in q for p in topic_patterns):
            return 'topic_author'
        
        return 'none'
    
    def extract_entity(self, question: str, query_type: str) -> str:
        """
        Extract author name or topic from question based on query type.

        Parses the user's question to extract the relevant entity (author name or topic)
        using regex patterns and noise word removal.

        Args:
            question (str): User's question text.
            query_type (str): Pre-detected query type ('author_topics' or 'topic_author').

        Returns:
            str: Extracted entity (author name or topic), or empty string if extraction fails.
        """
        q = question.strip()
        
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
                'என்ன எழுதினார்', 'எழுதியது', 'எழுதிய', 'தலைப்பு',
                'பற்றி', 'இதழில்', 'பொன்னி', 'குறிப்பிடுக', 'என்ன',
                'எழுதியவற்றை', 'எழுதியவற்றைக்'
            ]
            
            for word in noise:
                q = re.sub(word, '', q, flags=re.IGNORECASE)
            
            words = re.findall(r'[\u0B80-\u0BFF]+', q)
            words = [normalize_author_name(w) for w in words if len(w) > 2]
            
            return ' '.join(words) if words else ''
        
        elif query_type == 'topic_author':
            noise = [
                'யார் எழுதினார்', 'எழுதியவர் யார்', 'ஆசிரியர் யார்',
                'என்ற நூலை', 'எழுதியவர்', 'பற்றி', 'இதழில்',
                'பொன்னி', 'என்ற'
            ]
            
            for word in noise:
                q = re.sub(word, '', q, flags=re.IGNORECASE)
            
            words = re.findall(r'[\u0B80-\u0BFF]+|[a-zA-Z]+', q)
            words = [w for w in words if len(w) > 2]
            
            return ' '.join(words) if words else ''
        
        return ''
    
    def list_all_authors(self) -> Dict:
        """
        List all unique authors with article counts.
        Retrieves all unique author names from the CSV and counts how many articles
        each author wrote.

        Returns:
            dict: Result containing:
                - type (str): "list_all_authors"
                - success (bool): True if successful
                - total_authors (int): Number of unique authors
                - total_articles (int): Total article count
                - authors (list): List of dicts with 'name' and 'count'
                - message (str): Error message if failed
        """
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
        """
        Get all articles/topics by a specific author.
        Searches CSV for all articles written by the specified author using flexible
        name matching.
        Args:

            author_name (str): Author name to search for (allows variations/aliases).
        Returns:
            dict: Result containing:
                - type (str): "author_topics"
                - success (bool): True if author found
                - author (str): Search author name
                - matched_author (str): Actual CSV author name (if found)
                - count (int): Number of articles
                - articles (list): List of article dicts with title, author, metadata
                - message (str): Error message if not found
        """
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
        Searches CSV for articles whose titles contain the specified topic/keyword.

        Args:
            topic (str): Topic or keyword to search in article titles.

        Returns:
            dict: Result containing:
                - type (str): "topic_author"
                - success (bool): True if topic found
                - topic (str): Search topic
                - count (int): Number of matching articles
                - articles (list): List of article dicts with title, author, metadata
                - message (str): Error message if not found
        """
        if self.df is None or self.df.empty:
            return {
                "type": "topic_author",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }
        
        matches = self.df[
            self.df['தலைப்பு'].str.contains(topic, case=False, na=False, regex=False)
        ]
        
        if matches.empty:
            return {
                "type": "topic_author",
                "success": False,
                "topic": topic,
                "message": f"'{topic}' தலைப்பு கண்டுபிடிக்க முடியவில்லை",
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
            "type": "topic_author",
            "success": True,
            "topic": topic,
            "count": len(articles),
            "articles": articles
        }


def format_author_list(result: Dict) -> str:
    """ Format list of all authors for display.
        Creates a formatted Tamil text output showing all authors and their article counts,
        sorted by article count (most prolific first).

        Args:
            result (dict): Result from list_all_authors().

        Returns:
            str: Formatted multi-line Tamil text with:
                - Header: "பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்"
                - Statistics: Total authors and articles
                - Numbered list: Each author with article count
    """
    if not result['success']:
        return f"Error: {result['message']}"
    
    lines = [
        f"பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்",
        f"மொத்த எழுத்தாளர்கள்: {result['total_authors']} | கட்டுரைகள்: {result['total_articles']}",
        ""
    ]
    
    sorted_authors = sorted(result['authors'], key=lambda x: x['count'], reverse=True)
    
    for idx, author in enumerate(sorted_authors, 1):
        lines.append(f"{idx}. {author['name']} ({author['count']} கட்டுரைகள்)")
    
    return '\n'.join(lines)

def format_author_topics(result: Dict) -> str:
    """
    Format topics by author for display.
    
    Organizes and formats articles written by a specific author, grouped by year,
    for display in Tamil language.
    
    Args:
        result (Dict): Dictionary containing author query results with keys:
            - success (bool): Whether the query was successful
            - message (str): Error message if unsuccessful
            - matched_author (str): The matched author name
            - author (str): Original author name queried
            - count (int): Total number of articles
            - articles (List[Dict]): List of article dictionaries
    
    Returns:
        str: Formatted string containing author information and articles organized by year,
             or error message if query failed.
    """
    if not result['success']:
        return f"Error: {result['message']}"
    
    lines = [
        f"எழுத்தாளர்: {result.get('matched_author', result['author'])}",
        f"மொத்த கட்டுரைகள்: {result['count']}",
        ""
    ]
    
    by_year = {}
    for article in result['articles']:
        year = article.get('ஆண்டு', 'Unknown')
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(article)
    
    for year in sorted(by_year.keys()):
        if year != 'Unknown':
            lines.append(f"\n{year}ம் ஆண்டு ({len(by_year[year])} கட்டுரைகள்)")
        else:
            lines.append(f"\nஆண்டு குறிப்பிடப்படவில்லை ({len(by_year[year])} கட்டுரைகள்)")
        
        for article in by_year[year]:
            parts = [f"தலைப்பு: {article['title']}"]
            if 'இதழ்' in article:
                parts.append(f"இதழ்: {article['இதழ்']}")
            if 'ச.எ.' in article:
                parts.append(f"ச.எ.: {article['ச.எ.']}")
            
            lines.append(f"  • {' | '.join(parts)}")
    
    return '\n'.join(lines)


def format_topic_authors(result: Dict) -> str:
    """
    Format authors by topic for display.
    
    Organizes and formats articles related to a specific topic, showing all authors
    who have written about that topic.
    
    Args:
        result (Dict): Dictionary containing topic query results with keys:
            - success (bool): Whether the query was successful
            - message (str): Error message if unsuccessful
            - topic (str): The topic that was queried
            - count (int): Total number of articles found
            - articles (List[Dict]): List of article dictionaries containing
              title, author, year, and issue information
    
    Returns:
        str: Formatted string containing topic information and all related articles
             with their authors, or error message if query failed.
    """
    if not result['success']:
        return f"Error: {result['message']}"
    
    lines = [
        f"தலைப்பு: '{result['topic']}' பற்றிய கட்டுரைகள்",
        f"கண்டுபிடிக்கப்பட்டவை: {result['count']}",
        ""
    ]
    
    for idx, article in enumerate(result['articles'], 1):
        parts = [
            f"தலைப்பு: {article['title']}",
            f"ஆசிரியர்: {article['author']}"
        ]
        
        if 'ஆண்டு' in article:
            parts.append(f"ஆண்டு: {article['ஆண்டு']}")
        if 'இதழ்' in article:
            parts.append(f"இதழ்: {article['இதழ்']}")
        
        lines.append(f"{idx}. {' | '.join(parts)}\n")
    
    return '\n'.join(lines)


def detect_issue_count_query(question: str) -> bool:
    """
    Detect if user is asking about issue count.
    
    Analyzes the user's question to determine if they are asking about the number
    of issues or requesting an issue count/list.
    
    Args:
        question (str): User's question in Tamil or English
    
    Returns:
        bool: True if the question is about issue count, False otherwise
    """
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
    """
    Get unique issue count from CSV with article statistics.
    
    Reads a CSV file and extracts unique issue numbers along with article counts
    for each issue. Includes error handling for various CSV reading scenarios.
    
    Args:
        csv_path (str): Path to the CSV file containing issue and article data
    
    Returns:
        Dict: Dictionary containing:
            - success (bool): Whether the operation was successful
            - message (str): Error message if unsuccessful
            - count (int): Number of unique issues
            - total_articles (int): Total number of articles across all issues
            - issues (List[Dict]): List of dictionaries with issue_number and article_count
    """
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
    """
    Format issue count result for display.
    
    Creates a formatted, readable display of issue statistics including total issues,
    total articles, detailed issue listings, and statistical summaries.
    
    Args:
        result (Dict): Dictionary containing issue count data with keys:
            - success (bool): Whether the query was successful
            - message (str): Error message if unsuccessful
            - count (int): Total number of unique issues
            - total_articles (int): Total number of articles
            - issues (List[Dict]): List of issue information dictionaries
    
    Returns:
        str: Formatted string with comprehensive issue statistics, either as a full list
             (if 20 or fewer issues) or as a summary with first 5, last 5, and statistics
             (if more than 20 issues).
    """
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
        lines.append("இதழ் எண் | கட்டுரைகள்")
        lines.append("-" * 30)
        for issue_info in issues:
            lines.append(f"{issue_info['issue_number']:>8} | {issue_info['article_count']} கட்டுரைகள்")
    else:
        lines.append("இதழ் விவரங்கள்:")
        lines.append("")
        
        lines.append("முதல் 5 இதழ்கள்:")
        for issue_info in issues[:5]:
            lines.append(f"  • இதழ் {issue_info['issue_number']}: {issue_info['article_count']} கட்டுரைகள்")
        
        lines.append("")
        lines.append(f"... (மேலும் {len(issues) - 10} இதழ்கள்)")
        lines.append("")
        
        lines.append("கடைசி 5 இதழ்கள்:")
        for issue_info in issues[-5:]:
            lines.append(f"  • இதழ் {issue_info['issue_number']}: {issue_info['article_count']} கட்டுரைகள்")
    
    lines.append("")
    
    if result['issues']:
        article_counts = [i['article_count'] for i in result['issues']]
        lines.append("புள்ளிவிவரம்:")
        lines.append(f"  • சராசரி கட்டுரைகள் ஒரு இதழுக்கு: {sum(article_counts) / len(article_counts):.1f}")
        lines.append(f"  • குறைந்தபட்ச கட்டுரைகள்: {min(article_counts)}")
        lines.append(f"  • அதிகபட்ச கட்டுரைகள்: {max(article_counts)}")
    
    return '\n'.join(lines)


def handle_author_query(question: str, csv_path: str) -> Tuple[bool, str]:
    """
    Handle author-related queries and issue count queries.
    
    Main dispatcher function that routes different types of author and issue queries
    to appropriate handlers and returns formatted results.
    
    Args:
        question (str): User's question in Tamil or English
        csv_path (str): Path to the CSV file containing article data
    
    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if query was handled, False otherwise
            - str: Formatted response string or empty string if not handled
    """
    if detect_issue_count_query(question):
        logger.info("Detected issue count query")
        result = get_issue_count(csv_path)
        return True, format_issue_count(result)
    
    system = EnhancedAuthorQuerySystem(csv_path)
    
    if system.df is None or system.df.empty:
        return False, ""
    
    query_type = system.detect_query_type(question)
    logger.info(f"Query type: {query_type}")
    
    if query_type == 'none':
        return False, ""
    
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


def is_author_question(question: str) -> bool:
    """
    Check if question is about authors.
    
    Performs keyword matching to determine if the user's question is related
    to authors or author listings.
    
    Args:
        question (str): User's question in Tamil or English
    
    Returns:
        bool: True if question contains author-related keywords, False otherwise
    """
    keywords = ["author", "authors", "list", "எழுத்தாளர்", "எழுத்தாளர்கள்", "பட்டியல்", "யார்"]
    return any(k in question.lower() for k in keywords)


def fetch_all_authors(client: QdrantClient) -> List[str]:
    """
    Fetch all unique authors from Qdrant database.
    
    Retrieves all points from the Qdrant collection and extracts unique author names
    from the payload data.
    
    Args:
        client (QdrantClient): Initialized Qdrant client instance
    
    Returns:
        List[str]: Sorted list of unique author names
    """
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
    """
    Format author list in Tamil.
    
    Creates a formatted, bulleted list of authors with a Tamil header.
    
    Args:
        authors (List[str]): List of author names
    
    Returns:
        str: Formatted string with Tamil header and bulleted author list,
             or a message indicating no authors found
    """
    if not authors:
        return "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
    lines = ["பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்:\n"]
    for author in authors:
        lines.append(f"• {author}")
    return "\n".join(lines)


class HybridQdrantSearch:
    """
    Hybrid search combining dense and sparse vectors for optimal results.
    
    Implements a fusion-based search strategy that combines dense embeddings
    (semantic search) with sparse embeddings (keyword-based search) using
    Reciprocal Rank Fusion (RRF) for improved retrieval accuracy.
    
    Attributes:
        client (QdrantClient): Initialized Qdrant client instance
    
    Methods:
        search: Perform hybrid search with dense and sparse vector fusion
    """
    
    def __init__(self, client: QdrantClient):
        """
        Initialize the HybridQdrantSearch.
        
        Args:
            client (QdrantClient): Initialized Qdrant client instance
        """
        self.client = client

    def search(self, query: str, limit: int = 30):
        """
        Perform hybrid search using dense and sparse vectors.
        
        Executes a two-stage search combining dense embeddings for semantic similarity
        and sparse embeddings for keyword matching, then fuses results using RRF.
        
        Args:
            query (str): Search query string in Tamil or English
            limit (int, optional): Maximum number of results to return. Defaults to 30.
        
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
    """
    Retrieve all chunks for a specific document.
    
    Fetches all text chunks belonging to a single document identified by its
    doc_id, doc_issue, and volume from the Qdrant database.
    
    Args:
        client (QdrantClient): Initialized Qdrant client instance
        doc_id (str): Document identifier
        doc_issue (str): Issue number containing the document
        volume (str): Volume number containing the document
    
    Returns:
        List[Dict]: List of all chunk points for the specified document

    """
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
    """
    Merge consecutive chunks from the same document.
    
    Takes search results that may contain multiple chunks from the same document
    and merges them into complete documents. Filters out very short documents
    (less than 50 words) and sorts by relevance score.
    
    Args:
        client (QdrantClient): Initialized Qdrant client instance
        points: List of scored points from Qdrant search results
    
    Returns:
        List[Dict]: List of merged document dictionaries containing:
            - volume: Document volume number
            - doc_id: Document identifier
            - doc_issue: Issue number
            - heading: Article title/heading
            - author_name: Author of the article
            - content: Complete merged content from all chunks
            - word_count: Total word count
            - chunk_count: Number of chunks merged
            - score: Relevance score from search
    """
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
    """
    Extract key facts from documents relevant to question.
    
    Analyzes documents and extracts individual sentences that are highly relevant
    to the user's question based on keyword matching and relevance scoring.
    
    Args:
        docs (List[Dict]): List of document dictionaries with 'content' field
        question (str): User's question in Tamil or English
    
    Returns:
        List[Dict]: List of up to 20 most relevant facts, each containing:
            - sentence: The extracted sentence
            - score: Relevance score based on keyword matches
            - source_issue: Issue number of the source document
            - source_volume: Volume number of the source document
    """
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
- சூழலில் இல்லாத தகவல்களை எதையும் எழுதாதீர்கள்
- "சூழலின் படி", "ஆதாரத்தின் படி" போன்ற சொற்களை பயன்படுத்த வேண்டாம்
- வாசிப்பவரின் கண்களுக்கு சோர்வு வராத வகையில் பதிலை அமைக்க வேண்டும்

இப்போது, கீழே கொடுக்கப்பட்ட கேள்வி மற்றும் சூழலின் அடிப்படையில், மேலுள்ள அனைத்து விதிகளையும் கட்டாயமாக பின்பற்றி, தெளிவாகவும் வாசிக்க எளிதாகவும் விரிவான பதிலை எழுதுக."""


def generate_llm_answer(question: str, context: str, max_words: int = 500) -> str:
    """
    Generate an LLM-based answer using Ollama.
    
    Sends the question and context to an Ollama language model to generate
    a comprehensive answer in Tamil following specific formatting guidelines.
    
    Args:
        question (str): User's question in Tamil or English
        context (str): Relevant context extracted from documents
        max_words (int, optional): Maximum word count for the answer. Defaults to 500.
    
    Returns:
        str: Generated answer in Tamil (200-500 words), or empty string if generation fails
    """
    try:
        prompt = f"""{TAMIL_ANSWER_SYSTEM_PROMPT}

கேள்வி: {question}

சூழல்:
{context}

விரிவான பதில் (200-500 சொற்கள்):"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 800
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

def generate_extractive_answer(facts: List[Dict], question: str) -> str:
    """
    Generate fallback extractive answer if LLM fails.
    
    Creates an answer by concatenating the most relevant extracted sentences
    when LLM generation is unavailable or fails. Used as a backup strategy.
    
    Args:
        facts (List[Dict]): List of extracted fact dictionaries with 'sentence' field
        question (str): User's question (not currently used in logic)
    
    Returns:
        str: Concatenated answer from top 5 relevant sentences, or empty string
             if no suitable facts are available
    """
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


def format_sources(merged_docs: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Format source documents for display.
    
    Prepares a list of source documents with truncated content and metadata
    for display to the user, removing duplicates and limiting content length.
    
    Args:
        merged_docs (List[Dict]): List of merged document dictionaries
        limit (int, optional): Maximum number of sources to return. Defaults to 10.
    
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
    
    for doc in merged_docs[:limit]:
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
    """
    Format complete answer with sources for display.
    
    Creates a comprehensive formatted output combining the generated answer
    with detailed source citations and metadata.
    
    Args:
        answer (str): Generated answer text in Tamil
        sources (List[Dict]): List of formatted source dictionaries
    
    Returns:
        str: Complete formatted output with answer section and sources section
    """
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

def ask_question(question: str, top_k: int = 10, return_formatted: bool = False, use_llm: bool = True) -> Dict:
    """
    Main question answering function with database health check and hybrid search.

    Primary entry point for processing user queries. Handles database health checks,
    author queries, vector search, document merging, LLM generation, and source formatting.

    Args:
        question (str): User's question in Tamil or English.
        top_k (int, optional): Number of source documents to return. Defaults to 10.
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
    health_status = check_qdrant_health()
    
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
    
    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, response = handle_author_query(question, str(CSV_PATH))
            
            if is_handled:
                logger.info("Handled as CSV author query")
                if return_formatted:
                    return response
                return {
                    "answer": response,
                    "sources": [],
                    "query_type": "author_csv"
                }
        except Exception as e:
            logger.error(f"CSV query error: {e}")
            
    try:
        client = get_qdrant_client()  # Uses cached client silently

        if is_author_question(question):
            authors = fetch_all_authors(client)
            answer = format_authors_tamil(authors) if authors else "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        logger.info(f"Searching: {question[:60]}...")
        
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=50)

        if not results:
            answer = "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}

        logger.info(f"Found {len(results)} chunks")

        merged_docs = merge_consecutive_chunks(client, results)
        
        if not merged_docs:
            answer = "போதுமான தகவல்கள் இல்லை."
            if return_formatted:
                return format_answer_output(answer, [])
            return {"answer": answer, "sources": []}
        
        logger.info(f"Merged into {len(merged_docs)} documents")

        context_parts = []
        for idx, doc in enumerate(merged_docs[:5], 1):
            context_parts.append(f"ஆவணம் {idx}: {doc['content'][:800]}")
        
        context = "\n\n".join(context_parts)
        
        answer = ""
        if use_llm:
            answer = generate_llm_answer(question, context)
        
        if not answer or len(answer) < 100:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            logger.info(f"Extracted {len(facts)} facts")
            answer = generate_extractive_answer(facts, question)
        
        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs, limit=top_k)
        logger.info(f"Ready with {len(sources)} sources")

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


# ============================================================================
# PRELOAD FUNCTION FOR STREAMLIT APP INITIALIZATION
# ============================================================================

def preload_models():
    """
    Preload all models at app startup with visible spinners.

    Initializes all required models (embedding, LLM, Qdrant client) with visible
    progress indicators. Should be called ONCE during Streamlit app initialization.
    After this runs, all subsequent queries will be fast and silent.
    """
    logger.info("=" * 60)
    logger.info("PRELOADING MODELS FOR STREAMLIT")
    logger.info("=" * 60)
    
    # Show spinners ONLY during initial preload
    with st.spinner(" Loading embedding model..."):
        _ = get_embed_model()
        st.success(" Embedding model loaded")
    
    with st.spinner(" Loading LLM model (this may take 1-2 minutes)..."):
        _ = get_llm()
        st.success(" LLM model loaded")
    
    with st.spinner(" Connecting to Qdrant database..."):
        _ = get_qdrant_client()
        st.success(" Qdrant connected")
    
    logger.info("=" * 60)
    logger.info(" ALL MODELS READY - APP IS READY TO SERVE")
    logger.info("=" * 60)
    
    st.success(" All models loaded successfully! Ready to answer queries.")