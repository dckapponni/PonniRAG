"""
Hybrid Search Module for Tamil Document Processing.
Provides vector and keyword-based search with Gemini LLM-powered answer generation.
Embedding model loads on GPU if available, falls back to optimized CPU.
"""
from typing import List, Dict, Tuple, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict, OrderedDict
from difflib import SequenceMatcher
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
import hashlib

from google import genai
from google.genai import types as genai_types

FUZZY_THRESHOLD = 0.95
TYPO_DISTANCE   = 1

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    # CPU optimizations: use all but one core for torch, 2 for interop
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # Already configured by another module

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "summary.csv"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_client = None
SCORE_THRESHOLD = 0.8  # Minimum cosine similarity for dense vector search

_embed_lock = threading.Lock()
_author_system_cache = {}
_author_system_lock = threading.Lock()


class ResponseCache:
    """Thread-safe TTL + LRU cache for ask_question results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(question: str) -> str:
        normalized = re.sub(r'\s+', ' ', question.strip().lower())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        key = self._make_key(question)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            if time.time() - self._timestamps.get(key, 0) > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                logger.info(f"[CACHE] TTL expired for {key[:12]}...")
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            total = self._hits + self._misses
            logger.info(f"[CACHE] HIT (hits={self._hits}, misses={self._misses}, rate={self._hits / total:.0%})")
            return self._cache[key]

    def put(self, question: str, result: Dict) -> None:
        key = self._make_key(question)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = result
                self._timestamps[key] = time.time()
                return
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(evicted_key, None)
            self._cache[key] = result
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info("[CACHE] Cache cleared")

    def stats(self) -> Dict:
        with self._lock:
            total = max(1, self._hits + self._misses)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.0%}",
            }


_response_cache = ResponseCache(max_size=100, ttl_seconds=3600)

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


def fuzzy_match_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _edit_distance_one(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > TYPO_DISTANCE:
        return False
    if a == b:
        return True
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[lb] <= TYPO_DISTANCE


# ============================================================================
# TAMIL-SAFE SUFFIX STRIPPING HELPERS
# NOTE: \b (word boundary) does NOT work with Tamil Unicode characters.
# We use (?=\s|$) or explicit end-of-string anchors instead.
# ============================================================================

def _strip_tamil_possessive_suffixes(text: str) -> str:
    """
    Strip common Tamil possessive/genitive suffixes that appear at word boundaries.
    Uses (?=\\s|$) instead of \\b for Tamil Unicode compatibility.

    IMPORTANT: \\b (ASCII word boundary) does NOT work with Tamil Unicode — always
    use (?=\\s|$) or explicit end-of-string anchors for Tamil text.

    Examples:
        பாரதிதாசனின்        -> பாரதிதாசன்      (னின் suffix)
        கிருஷ்ணாமூர்த்தியின் -> கிருஷ்ணாமூர்த்தி (யின் suffix)
        திறக்குமாவின்        -> திறக்குமா        (வின் suffix)
        பாவேந்தரின்          -> பாவேந்தர்        (ரின் suffix)
        கண்ணதாசனால்         -> கண்ணதாசன்        (னால் suffix)
        கதையை               -> கதை              (யை suffix)
    """
    # Order matters: longer/more-specific suffixes first to avoid partial stripping
    suffix_rules = [
        # னின் → ன்  (e.g. பாரதிதாசனின் → பாரதிதாசன்)
        (r'னின்(?=\s|$)', 'ன்'),
        # ரின் → ர்  (e.g. பாவேந்தரின் → பாவேந்தர், கவிஞரின் → கவிஞர்)
        (r'ரின்(?=\s|$)', 'ர்'),
        # யின் → ''  (e.g. கிருஷ்ணாமூர்த்தியின் → கிருஷ்ணாமூர்த்தி)
        (r'யின்(?=\s|$)', ''),
        # வின் → ''  (e.g. திறக்குமாவின் → திறக்குமா)
        (r'வின்(?=\s|$)', ''),
        # னால் → ன்  (e.g. கண்ணதாசனால் → கண்ணதாசன்)
        (r'னால்(?=\s|$)', 'ன்'),
        # ரால் → ர்  (e.g. கவிஞரால் → கவிஞர்)
        (r'ரால்(?=\s|$)', 'ர்'),
        # யால் → ''  (e.g. கதையால் → கதை)
        (r'யால்(?=\s|$)', ''),
        # னை → ன்   (e.g. கண்ணதாசனை → கண்ணதாசன்)
        (r'னை(?=\s|$)', 'ன்'),
        # ரை → ர்   (e.g. கவிஞரை → கவிஞர்)
        (r'ரை(?=\s|$)', 'ர்'),
        # யை → ''   (e.g. கதையை → கதை)
        (r'யை(?=\s|$)', ''),
        # வை → ''   (e.g. படைப்பை → படைப்பு — not perfect but acceptable)
        (r'வை(?=\s|$)', ''),
        # Generic ஆல் at word end
        (r'([\u0B80-\u0BFF])ஆல்(?=\s|$)', r'\1'),
        # Generic ஐ at word end
        (r'([\u0B80-\u0BFF])ஐ(?=\s|$)', r'\1'),
    ]
    result = text
    for pattern, replacement in suffix_rules:
        result = re.sub(pattern, replacement, result)
    return result


def _strip_tamil_possessive_suffix_word(word: str) -> str:
    """
    Strip Tamil possessive suffix from a single word token.
    Used when processing individual extracted tokens.
    """
    return _strip_tamil_possessive_suffixes(word).strip()


# ============================================================================
# STREAMLIT-CACHED MODEL LOADERS (PERSISTENT ACROSS RERUNS)
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_embed_model():
    """
    Load and cache embedding model using Streamlit's cache_resource.
    This ensures the model loads ONCE and persists across all reruns.
    Spinner is disabled - will only show during preload_models().
    """
    logger.info(f"Loading embedding model on {DEVICE} (this happens only once)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    logger.info(f"Embedding model loaded on {DEVICE} and cached")
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
    model = get_embed_model()
    with _embed_lock:
        return model.encode(f"query: {text}").tolist()


def _deterministic_token_hash(token: str) -> int:
    """Deterministic token hash using MD5, consistent across processes.

    Python's built-in hash() is randomized per process (PYTHONHASHSEED),
    which causes sparse vectors at query time to mismatch those created
    at indexing time.  MD5 is deterministic and fast for this use case.
    """
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**31)


def sparse_embed(text: str):
    """Generate sparse BM25-style embedding for text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    indices, values = [], []
    for token, freq in counts.items():
        indices.append(_deterministic_token_hash(token))
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
    if not name:
        return ""
    prefixes_to_remove = [
        r'மு\.,?\s*', r'டாக்டர்\.?\s*', r'திரு\.,?\s*',
        r'திருமதி\.?\s*', r'Dr\.?\s*', r'Mr\.?\s*', r'Mrs\.?\s*',
        r'கவிஞர்\.?\s*', r'பேராசிரியர்\.?\s*', r'அறிஞர்\.?\s*',
        r'புலவர்\.?\s*', r'கவியரசு\.?\s*', r'பாவேந்தர்\.?\s*',
    ]
    cleaned = name.strip()
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def flexible_author_match(search_name: str, csv_name: str) -> bool:
    """
    Flexible matching between search query and CSV author name.

    Priority:
    1. Exact match after normalisation
    2. One is a substring of the other
    3. Known alias / special-case mapping
    4. Token-level subset match  (handles split/joined name variants)
    5. Fuzzy >= FUZZY_THRESHOLD (95%)
    6. Edit-distance <= TYPO_DISTANCE
    """
    search_normalized = normalize_author_name(search_name).lower().strip()
    csv_normalized    = normalize_author_name(csv_name).lower().strip()

    if not search_normalized or not csv_normalized:
        return False

    # Dot-normalized versions: collapse "நா. " -> "நா." to handle initials spacing
    def _dot_norm(s: str) -> str:
        return re.sub(r'\s*\.\s*', '.', s)

    search_dot = _dot_norm(search_normalized)
    csv_dot    = _dot_norm(csv_normalized)

    # 1. Exact (with and without dot normalization)
    if search_normalized == csv_normalized:
        return True
    if search_dot == csv_dot:
        logger.info(f"[DOT-EXACT] '{search_normalized}' ~ '{csv_normalized}' ✅")
        return True

    # 2. Substring (with and without dot normalization)
    if search_normalized in csv_normalized:
        return True
    if csv_normalized in search_normalized:
        return True
    if search_dot in csv_dot:
        logger.info(f"[DOT-SUBSTR] '{search_normalized}' ~ '{csv_normalized}' ✅")
        return True
    if csv_dot in search_dot:
        logger.info(f"[DOT-SUBSTR] '{search_normalized}' ~ '{csv_normalized}' ✅")
        return True

    # 3. Special-case aliases
    # CRITICAL BUG FIX: Do NOT use normalize_author_name() on variant strings.
    # Several variants like 'பாவேந்தர்' and 'கவியரசு' are listed as honorific
    # prefixes inside normalize_author_name() and get stripped to '' (empty string).
    # An empty string is always `in` any other string, causing EVERY csv author
    # to match — producing 1271-result responses for பாரதிதாசன் / கண்ணதாசன் queries.
    # Fix: use variant.lower() directly, and guard against empty variant strings.
    special_cases = {
        'கலைஞர்':     ['கருணாநிதி', 'மு.கருணாநிதி', 'மு. கருணாநிதி'],
        'கருணாநிதி':  ['கலைஞர்',    'மு.கருணாநிதி', 'மு. கருணாநிதி'],
        'அண்ணா':      ['அண்ணாதுரை', 'சி.என்.அண்ணாதுரை'],
        'அண்ணாதுரை':  ['அண்ணா',     'சி.என்.அண்ணாதுரை'],
        'பாரதிதாசன்': ['பாவேந்தர்', 'பாவேந்தர் பாரதிதாசன்'],
        'பாவேந்தர்':  ['பாரதிதாசன்', 'பாவேந்தர் பாரதிதாசன்'],
        'கண்ணதாசன்':  ['கவியரசு', 'கவியரசு கண்ணதாசன்'],
        'கவியரசு':    ['கண்ணதாசன்', 'கவியரசு கண்ணதாசன்'],
    }
    for key, variations in special_cases.items():
        if key in search_normalized:
            for variant in variations:
                variant_lower = variant.lower().strip()
                # Guard: skip empty variants — empty string matches everything
                if variant_lower and variant_lower in csv_normalized:
                    return True
        if key in csv_normalized:
            for variant in variations:
                variant_lower = variant.lower().strip()
                if variant_lower and variant_lower in search_normalized:
                    return True

    # 4. Token-level subset match (exact token equality, min 4 chars)
    # FIXED: use exact token equality (not substring containment) and min
    # length 4 to prevent short common Tamil substrings from causing false matches.
    search_tokens = [t for t in re.findall(r'[\u0B80-\u0BFF]+', search_normalized) if len(t) >= 4]
    csv_tokens    = [t for t in re.findall(r'[\u0B80-\u0BFF]+', csv_normalized)    if len(t) >= 4]
    if search_tokens and csv_tokens:
        if all(any(st == ct for ct in csv_tokens) for st in search_tokens):
            logger.info(f"[TOKEN-AUTHOR] '{search_normalized}' ~ '{csv_normalized}' ✅")
            return True

    # 5. Fuzzy >= 95%
    score = fuzzy_match_score(search_normalized, csv_normalized)
    if score >= FUZZY_THRESHOLD:
        logger.info(f"[FUZZY-AUTHOR] '{search_normalized}' ~ '{csv_normalized}' score={score:.2f} ✅")
        return True

    # 6. Edit-distance <= 1
    if _edit_distance_one(search_normalized, csv_normalized):
        logger.info(f"[TYPO-AUTHOR] '{search_normalized}' ≈ '{csv_normalized}' ✅")
        return True

    return False


# ============================================================================
# PATTERN BANK — central store for all Tamil/English phrase patterns
# Extend any list here to support new question phrasings without
# touching any logic elsewhere.
# ============================================================================

class _PatternBank:
    """Central store for every Tamil / English phrase pattern."""

    # ── Ponni magazine name variants ─────────────────────────────────────────
    PONNI = [
        'பொன்னி', 'பொன்னியில்', 'பொன்னியின்', 'பொன்னிக்கு',
        'பொன்னியை', 'பொன்னியால்', 'பொன்னியுடன்', 'பொன்னி இதழ்',
        'பொன்னி இதழில்', 'பொன்னி மாத இதழ்', 'ponni', 'ponni magazine',
        'ponni issue', 'ponni journal',
    ]

    # ── "List all authors" triggers ───────────────────────────────────────────
    LIST_ALL_AUTHORS = [
        'எழுத்தாளர்கள் யார்', 'ஆசிரியர்கள் யார்',
        'எழுத்தாளர்கள் பட்டியல்', 'எழுத்தாளர்களின் பட்டியல்',
        'ஆசிரியர்கள் பட்டியல்', 'ஆசிரியர்களின் பட்டியல்',
        'அனைத்து எழுத்தாளர்கள்', 'அனைத்து ஆசிரியர்கள்',
        'எழுதியவர்கள்', 'எழுதிய ஆசிரியர்கள்',
        'படைப்பாளிகள் பட்டியல்', 'படைப்பாளர்கள் பட்டியல்',
        'கவிஞர்கள் பட்டியல்', 'கதாசிரியர்கள் பட்டியல்',
        'பங்களிப்பாளர்கள்', 'பங்களிப்பாளர் பட்டியல்',
        'பொன்னியில் யார் யார் எழுதினார்கள்',
        'பொன்னியில் எழுதியவர்கள் யார்',
        'பொன்னியில் பங்கேற்றவர்கள்',
        'இதழில் எழுதியவர்கள்',
        'who are the authors', 'list all authors', 'list authors',
        'all writers', 'list writers', 'list contributors',
        'contributors list', 'who contributed', 'all contributors',
        'who wrote in ponni', 'writers in ponni',
    ]

    # ── "Topic → Author" triggers  (who wrote X?) ────────────────────────────
    TOPIC_AUTHOR = [
        # Classic who-wrote
        'யார் எழுதிய', 'யார் எழுதினார்', 'யார் எழுதியது',
        'யார் எழுதியவர்', 'யார் இயற்றியவர்',
        'எழுதியவர் யார்', 'எழுதியது யார்', 'எழுதியவத் யார்',
        'யார் படைத்தார்', 'படைத்தவர் யார்',
        # Author-of phrasing
        'ஆசிரியர் யார்', 'எழுத்தாளர் யார்', 'ஆசிரியர்?',
        'அசிரியர் யார்',   # common typo
        'கட்டுரையின் ஆசிரியர்', 'கதையின் ஆசிரியர்',
        'கவிதையின் ஆசிரியர்', 'பாடலின் ஆசிரியர்',
        'தொடரின் ஆசிரியர்', 'நாவலின் ஆசிரியர்',
        'தலைப்பின் ஆசிரியர்', 'படைப்பின் ஆசிரியர்',
        'பகுதியின் ஆசிரியர்', 'பகுதி ஆசிரியர்',
        'நூலின் ஆசிரியர்', 'நூலை எழுதியவர்',
        # Sang / composed
        'பாடியவரின் பெயர்', 'பாடியவர் யார்', 'பாடியவர் பெயர்',
        'இயற்றியவர் யார்', 'இயற்றியவர் பெயர்', 'இயற்றியவரின் பெயர்',
        'இயற்றியவர் யாவர்', 'இயற்றியவர் எவர்',
        'இசையமைத்தவர்', 'இசையமைத்தவர் யார்',
        'எழுதியவரின் பெயர்', 'எழுதியவர்', 'எழுதியவரின்',
        # "name" questions
        'ஆசிரியரின் பெயர்', 'ஆசிரியரின் பெயர் என்ன',
        'எழுத்தாளரின் பெயர்', 'எழுத்தாளரின் பெயர் என்ன',
        'படைத்தவரின் பெயர்', 'படைத்தவரின் பெயர் என்ன',
        'இயற்றியவரின் பெயர் என்ன',
        'பாடியவரின் பெயர் என்ன',
        'யார் இந்த கதை எழுதினார்', 'யார் இந்த கவிதை எழுதினார்',
        'யார் இந்த கட்டுரை எழுதினார்',
        # Section / column author
        'பகுதியை எழுதியவர்', 'பகுதியை இயற்றியவர்',
        'நெடுவரிசையின் ஆசிரியர்', 'நெடுவரிசை ஆசிரியர்',
        'தொடரை எழுதியவர்', 'தொடரை இயற்றியவர்',
        # "X எனும் பகுதியில் எழுதிய ஆசிரியர்" style section queries
        'எனும் பகுதியில் எழுதிய ஆசிரியர்',
        'என்னும் பகுதியில் எழுதிய ஆசிரியர்',
        'பகுதியில் எழுதிய ஆசிரியர்',
        'பகுதியில் எழுதியவர்களைப் பட்டியலிடுக',
        'பகுதியில் எழுதியவர்கள்',
        'பகுதியில் எழுதியவர் பெயர்',
        'பகுதியில் எழுதிய எழுத்தாளர்',
        'பகுதியில் எழுதிய ஆசிரியர் பெயர்',
        'பகுதி எழுதிய ஆசிரியர்',
        # Implicit trailing ?
        'ஆசிரியர் என்ன', 'ஆசிரியர் எவர்',
        # "அதிகம்/அதிகமாக + இயற்றியவர்/எழுதியவர்" - most prolific author in a section
        'அதிகம் இயற்றியவர்', 'அதிகமாக இயற்றியவர்',
        'அதிகம் எழுதியவர்', 'அதிகமாக எழுதியவர்',
        'அதிகம் படைத்தவர்', 'அதிகமாக படைத்தவர்',
        'அதிக படைப்புகள் உள்ளவர்', 'அதிக கட்டுரை எழுதியவர்',
        # "குறித்து/பற்றி + கட்டுரை/article + எழுதிய + ஆசிரியர்" style queries
        'குறித்து கட்டுரை எழுதிய ஆசிரியர்',
        'குறித்து கட்டுரை எழுதியவர் யார்',
        'குறித்து கட்டுரை எழுதியவர்',
        'குறித்து எழுதிய ஆசிரியர்',
        'குறித்து எழுதியவர் யார்',
        'குறித்து எழுதியவர்',
        'பற்றி கட்டுரை எழுதிய ஆசிரியர்',
        'பற்றி கட்டுரை எழுதியவர் யார்',
        'பற்றி கட்டுரை எழுதியவர்',
        'பற்றி எழுதிய ஆசிரியர்',
        'பற்றி எழுதியவர் யார்',
        'பற்றி எழுதியவர்',
        'பற்றிய கட்டுரை எழுதியவர்',
        'பற்றிய கட்டுரை எழுதிய ஆசிரியர்',
        'கட்டுரை எழுதிய ஆசிரியர்',
        'கட்டுரை எழுதியவர் யார்',
        'கட்டுரை எழுதியவர்',
        'பற்றிய படைப்புகள்',
        'பற்றிய கட்டுரைகள்',
        'தொடர்பான படைப்புகள்',
        'தொடர்பான கட்டுரைகள்',
        'பற்றி படைப்புகள்',
        # English
        'who wrote', 'who is the author', 'author of', 'written by',
        'who composed', 'composer of', 'who penned', 'penned by',
        'who created', 'creator of', 'who authored',
        'name of the author', 'name of the writer',
        'who is the writer', 'writer of',
    ]

    # ── "Author → Topics" action words ───────────────────────────────────────
    AUTHOR_ACTION = [
        'என்ன எழுதினார்', 'என்னென்ன எழுதினார்',
        'எழுதியது என்ன', 'எழுதியவை என்ன', 'எழுதியவை யாவை',
        'எழுதிய தலைப்பு', 'எழுதிய தலைப்புகள்',
        'எழுதிய கட்டுரைகள்', 'எழுதிய கட்டுரை',
        'எழுதிய கதைகள்', 'எழுதிய கதை',
        'எழுதிய கவிதைகள்', 'எழுதிய கவிதை',
        'எழுதிய நாவல்', 'எழுதிய நாவல்கள்',
        'எழுதிய படைப்புகள்', 'எழுதிய படைப்பு',
        'எழுதிய தொடர்', 'எழுதிய தொடர்கள்',
        'எழுதிய தொடரின்', 'எழுதிய பகுதி',
        'எழுதிய பாடல்', 'எழுதிய பாடல்கள்',
        'இயற்றிய படைப்புகள்', 'இயற்றிய கவிதை',
        'இயற்றிய கவிதைகள்', 'இயற்றிய பாடல்',
        'படைத்த படைப்புகள்', 'படைத்தவை என்ன',
        'பங்களிப்புகள்', 'பங்களிப்பு என்ன',
        'பங்கு என்ன', 'பங்களிப்பு யாவை',
        'பற்றி எழுதினார்', 'எந்த தலைப்புகள்',
        'எந்தெந்த தலைப்புகள்', 'என்னென்ன தலைப்புகள்',
        'என்னென்ன கதைகள்', 'எந்தெந்த கதைகள்',
        'என்னென்ன கட்டுரைகள்',
        'அவர் எழுதிய', 'அவரது படைப்புகள்',
        'அவரின் படைப்புகள்', 'அவர்கள் எழுதிய',
        'படைப்புகள்', 'படைப்புகளை', 'படைப்பு என்ன',
        'படைப்புகளைப் பட்டியலிடுக', 'பட்டியலிடுக',
        'தொடரின் பெயர்', 'தொடர் பெயர்',
        'அவர் எழுதிய தொடர்', 'எழுதிய தொடரின் பெயர்',
        'list of writings', 'articles by', 'works of', 'works by',
        'what did write', 'list works', 'writings by',
        'contributions by', 'what has written', 'what wrote',
        'poems by', 'stories by', 'articles written by',
        'written works', 'literary works', 'what authored',
        # ── NEW: யாவை / யாவை? pattern for "படைப்புகள் யாவை" ──
        'படைப்புகள் யாவை',
    ]

    # ── Known author keywords (for has_author check) ──────────────────────────
    KNOWN_AUTHORS = [
        'கலைஞர்', 'கருணாநிதி', 'பெரியார்', 'அண்ணா', 'அண்ணாதுரை',
        'நக்கீரன்', 'பாரதிதாசன்', 'பாவேந்தர்', 'புதுமைப்பித்தன்',
        'அகிலன்', 'கண்ணதாசன்', 'கண்ணாதாசன்', 'கவியரசு', 'தங்கமணி', 'நாச்சியப்பன்',
        'நாரா', 'சீனிவாசன்', 'டி கே சீனிவாசன்', 'டி.கே.சீனிவாசன்',
        'இராமநாதன்', 'கிருஷ்ணாமூர்த்தி', 'நா.கிருஷ்ணாமூர்த்தி',
        'கமலா', 'விருத்தாசலம்', 'கமலா விருத்தாசலம்',
        'வாணிதாசன்', 'சுரதா', 'திரு.வி.க', 'முருகு',
        'பெரியண்ணன்', 'அப்பாதுரை', 'மு. அண்ணாமலை','அப்பாதுரை', 'அப்பாத்துரை', 'கா.அப்பாத்துரை',
        'மு.வ', 'மு வ', 'தி.க.சீனிவாசன்',
    ]

    # ── Noise phrases stripped from around the topic title ────────────────────
    TOPIC_NOISE_PHRASES = [
        'யார் எழுதினார்', 'யார் எழுதியது', 'யார் இயற்றியவர்',
        'எழுதியவர் யார்', 'எழுதியது யார்', 'எழுதியவத் யார்',
        'ஆசிரியர் யார்', 'எழுத்தாளர் யார்', 'ஆசிரியர்?',
        'அசிரியர் யார்',
        'பாடியவரின் பெயர் என்ன', 'பாடியவரின் பெயர்',
        'பாடியவர் யார்', 'பாடியவர் பெயர்',
        'இயற்றியவரின் பெயர் என்ன', 'இயற்றியவரின் பெயர்',
        'இயற்றியவர் யார்', 'இயற்றியவர் பெயர்',
        'ஆசிரியரின் பெயர் என்ன', 'ஆசிரியரின் பெயர்',
        'எழுத்தாளரின் பெயர் என்ன', 'எழுத்தாளரின் பெயர்',
        'படைத்தவரின் பெயர் என்ன', 'படைத்தவரின் பெயர்',
        'யார் படைத்தார்', 'படைத்தவர் யார்',
        'என்பதின்', 'என்பதன்', 'என்பதை', 'என்பது',
        'என்னும்', 'என்ற', 'எனும்', 'என்பதில்',
        'ஆசிரியர் என்ன', 'ஆசிரியர் எவர்',
        'கட்டுரையின் ஆசிரியர்', 'கதையின் ஆசிரியர்',
        'கவிதையின் ஆசிரியர்', 'பாடலின் ஆசிரியர்',
        'தொடரின் ஆசிரியர்', 'நாவலின் ஆசிரியர்',
        'தலைப்பின் ஆசிரியர்', 'படைப்பின் ஆசிரியர்',
        'பகுதியின் ஆசிரியர்', 'பகுதி ஆசிரியர்',
        'நூலின் ஆசிரியர்', 'நூலை எழுதியவர்',
        'பகுதியை எழுதியவர்', 'பகுதியை இயற்றியவர்',
        'நெடுவரிசையின் ஆசிரியர்', 'நெடுவரிசை ஆசிரியர்',
        'தொடரை எழுதியவர்', 'தொடரை இயற்றியவர்',
        'யார் இந்த கதை எழுதினார்', 'யார் இந்த கவிதை எழுதினார்',
        'யார் இந்த கட்டுரை எழுதினார்',
        'யார் எழுதிய', 'எழுதியவர்', 'எழுதியவரின்',
        'பற்றிய படைப்புகள்',
        'பற்றிய கட்டுரைகள்',
        'பற்றி படைப்புகள்',
        # Section-listing noise
        'எனும் பகுதியில் எழுதிய ஆசிரியர் பெயர்களைப் பட்டியலிடுக',
        'என்னும் பகுதியில் எழுதிய ஆசிரியர் பெயர்களைப் பட்டியலிடுக',
        'எனும் பகுதியில் எழுதிய ஆசிரியர் பெயர்களை',
        'என்னும் பகுதியில் எழுதிய ஆசிரியர் பெயர்களை',
        'எனும் பகுதியில் எழுதிய ஆசிரியர்',
        'என்னும் பகுதியில் எழுதிய ஆசிரியர்',
        'பகுதியில் எழுதிய ஆசிரியர் பெயர்களைப் பட்டியலிடுக',
        'பகுதியில் எழுதிய ஆசிரியர் பெயர்களை',
        'பகுதியில் எழுதிய ஆசிரியர்',
        'பெயர்களைப் பட்டியலிடுக', 'பெயர்களை',
        'who wrote', 'who is the author', 'author of', 'written by',
        'who composed', 'composer of', 'who penned', 'penned by',
        'who created', 'creator of', 'who authored',
        'name of the author', 'name of the writer',
        'who is the writer', 'writer of',
        # "அதிகம்/அதிகமாக + இயற்றியவர்" noise - strip to get section name
        'அதிகம் இயற்றியவர்', 'அதிகமாக இயற்றியவர்',
        'அதிகம் எழுதியவர்', 'அதிகமாக எழுதியவர்',
        'அதிகம் படைத்தவர்', 'அதிகமாக படைத்தவர்',
        'அதிக படைப்புகள் உள்ளவர்', 'அதிக கட்டுரை எழுதியவர்',
        # "குறித்து/பற்றி + கட்டுரை + எழுதிய + ஆசிரியர்" noise (strip whole phrase, keep topic)
        'குறித்து கட்டுரை எழுதிய ஆசிரியர்',
        'குறித்து கட்டுரை எழுதியவர் யார்',
        'குறித்து கட்டுரை எழுதியவர்',
        'குறித்து எழுதிய ஆசிரியர்',
        'குறித்து எழுதியவர் யார்',
        'குறித்து எழுதியவர்',
        'பற்றி கட்டுரை எழுதிய ஆசிரியர்',
        'பற்றி கட்டுரை எழுதியவர் யார்',
        'பற்றி கட்டுரை எழுதியவர்',
        'பற்றி எழுதிய ஆசிரியர்',
        'பற்றி எழுதியவர் யார்',
        'பற்றி எழுதியவர்',
        'பற்றிய கட்டுரை எழுதியவர்',
        'பற்றிய கட்டுரை எழுதிய ஆசிரியர்',
        'கட்டுரை எழுதிய ஆசிரியர்',
        'கட்டுரை எழுதியவர் யார்',
        'கட்டுரை எழுதியவர்',
        # Context prefixes
        'பொன்னி இதழில்', 'பொன்னியில்', 'பொன்னி',
        'இதழில்', 'இதழ்',
    ]

    # Title-type suffix words (stripped only at END of extracted title)
    TITLE_SUFFIXES = [
        'தொடரின்', 'தொடர்', 'கதையின்', 'கதை', 'நாவலின்', 'கட்டுரையின்',
        'கவிதையின்', 'பாடலின்', 'பகுதியின்', 'நூலின்',
        'series', 'story', 'novel', 'article', 'poem', 'part',
    ]

    # ── Noise phrases stripped from around the author name ────────────────────
    AUTHOR_NOISE_PHRASES = [
        'பொன்னி இதழில்', 'பொன்னியில்', 'பொன்னி',
        'இதழில்', 'இதழ்',
        'என்ன எழுதினார்', 'என்னென்ன எழுதினார்',
        'எழுதியது என்ன', 'எழுதியவை என்ன', 'எழுதியவை யாவை',
        'எழுதிய தலைப்புகள்', 'எழுதிய கட்டுரைகள்',
        'எழுதிய கதைகள்', 'எழுதிய கவிதைகள்',
        'எழுதிய படைப்புகள் யாவை', 'எழுதிய படைப்புகள்',
        'படைப்புகளைப் பட்டியலிடுக', 'படைப்புகளை', 'படைப்புகள் யாவை', 'படைப்புகள்',
        'பட்டியலிடுக', 'பட்டியல்', 'யாவை', 'என்ன', 'யார்',
        'எழுதிய', 'இயற்றிய', 'படைத்த',
        'பங்களிப்புகள்', 'பங்களிப்பு',
        'அவர் எழுதிய', 'அவரது', 'அவரின்',
        'list of writings', 'articles by', 'works of', 'works by',
        'writings by', 'contributions by', 'written works',
        'poems by', 'stories by',
    ]

    # Noise single tokens to drop after phrase removal (author extraction)
    AUTHOR_NOISE_TOKENS = {
        'பொன்னி', 'இதழ்', 'இல்', 'கட்டுரை', 'கதை', 'கவிதை',
        'பாடல்', 'தொடர்', 'நாவல்', 'படைப்பு', 'பகுதி',
        'என', 'யார்', 'என்ன', 'யாவை', 'எவர்', 'பட்டியல்',
        'எழுதிய', 'இயற்றிய',
    }

    # Noise single tokens to drop (topic extraction)
    TOPIC_NOISE_TOKENS = {
        'என', 'யார்', 'இல்', 'பற்றி', 'குறித்து', 'பற்றிய',
        'எழுதிய', 'இயற்றிய', 'யாவை', 'என்ன', 'என்னும்','என்பதின்', 'என்பதன்', 'என்பதை', 'என்பது','என்பதில்',
        'என்ற', 'என்பதை', 'கதையை', 'கவிதையை', 'பாடலை',
        # Standalone words that leak after phrase-stripping
        'கட்டுரை', 'ஆசிரியர்', 'எழுத்தாளர்',
        # Standalone "most" words that leak
        'அதிகம்', 'அதிகமாக', 'அதிக',
    }

    # Known author canonical mapping
    AUTHOR_CANONICAL = {
        'கலைஞர்':              'கருணாநிதி',
        'கருணாநிதி':           'கருணாநிதி',
        'பெரியார்':            'பெரியார்',
        'அண்ணா':               'அண்ணாதுரை',
        'அண்ணாதுரை':           'அண்ணாதுரை',
        'அப்பாத்துரை':   'கா.அப்பாத்துரை',
        'அப்பாதுரை':    'கா.அப்பாத்துரை',
        'கா.அப்பாத்துரை': 'கா.அப்பாத்துரை',
        'கா அப்பாத்துரை': 'கா.அப்பாத்துரை',
        'நக்கீரன்':            'நக்கீரன்',
        'பாரதிதாசன்':          'பாரதிதாசன்',
        'பாவேந்தர்':           'பாரதிதாசன்',
        'புதுமைப்பித்தன்':     'புதுமைப்பித்தன்',
        'அகிலன்':              'அகிலன்',
        'கண்ணதாசன்':          'கண்ணதாசன்',
        'கண்ணாதாசன்':         'கண்ணதாசன்',   # common typo (extra ா)
        'கவியரசு':             'கண்ணதாசன்',
        'நாச்சியப்பன்':        'நாச்சியப்பன்',
        'நாரா':                'நாச்சியப்பன்',
        'டி கே சீனிவாசன்':     'டி. கே. சீனிவாசன்',
        'டி.கே.சீனிவாசன்':     'டி. கே. சீனிவாசன்',
        'தி.க.சீனிவாசன்':      'டி. கே. சீனிவாசன்',
        'சீனிவாசன்':           'டி. கே. சீனிவாசன்',
        'நா.கிருஷ்ணாமூர்த்தி': 'நா. கிருஷ்ணாமூர்த்தி',
        'கிருஷ்ணாமூர்த்தி':    'நா. கிருஷ்ணாமூர்த்தி',
        'வாணிதாசன்':           'வாணிதாசன்',
        'சுரதா':               'சுரதா',
        'முருகு':              'முருகு. சுப்பிரமணியம்',
        'பெரியண்ணன்':          'அரு. பெரியண்ணன்',
    }


# Compiled regex helpers
_RE_INITIALS_NAME = re.compile(
    r'((?:[\u0B80-\u0BFF]\.?\s*){1,4}[\u0B80-\u0BFF]{3,}(?:\s+[\u0B80-\u0BFF]{3,})*)'
)
_RE_TAMIL_WORD = re.compile(r'[\u0B80-\u0BFF][\u0B80-\u0BFF\.]*')


def _find_closest_author(df: pd.DataFrame, search_name: str) -> tuple:
    """Return (best_author_name, best_score) from the author column."""
    best_name, best_score = "", 0.0
    search_norm = normalize_author_name(search_name).lower()
    for name in df['ஆசிரியர்'].dropna().unique():
        score = fuzzy_match_score(search_norm, normalize_author_name(str(name)).lower())
        if score > best_score:
            best_score = score
            best_name = str(name)
    return best_name, best_score


def _find_closest_title(df: pd.DataFrame, topic: str) -> tuple:
    """Return (best_title, best_score) from the title column."""
    best_title, best_score = "", 0.0
    topic_lower = topic.lower()
    for title in df['தலைப்பு'].dropna().unique():
        score = fuzzy_match_score(topic_lower, str(title).lower())
        if score > best_score:
            best_score = score
            best_title = str(title)
    return best_title, best_score


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

    # -------------------------------------------------------------------------
    def detect_query_type(self, question: str) -> str:
        """
        Comprehensive query-type detection.
        Returns: 'list_all_authors' | 'topic_author' | 'author_topics' | 'none'
        Evaluation order matters — broader patterns checked last.
        """
        q = question.lower().strip()

        # 1. List all authors
        if any(p in q for p in _PatternBank.LIST_ALL_AUTHORS):
            return 'list_all_authors'

        # 2. Topic → Author  (who wrote X?)
        if any(p in q for p in _PatternBank.TOPIC_AUTHOR):
            return 'topic_author'

        # 3. Author → Topics  (what did X write?)
        has_known_author  = any(a in q for a in _PatternBank.KNOWN_AUTHORS)
        has_initials_name = bool(_RE_INITIALS_NAME.search(q))
        has_author_action = any(p in q for p in _PatternBank.AUTHOR_ACTION)

        if (has_known_author or has_initials_name) and has_author_action:
            return 'author_topics'

        # Softer fallback: name present + writing-related keyword
        if (has_known_author or has_initials_name) and (
            'எழுதிய' in q or 'படைப்பு' in q or 'இயற்றிய' in q or 'படைத்த' in q
        ):
            return 'author_topics'

        # 4. Topic → Author fallback: "X பற்றிய படைப்புகள்/கட்டுரைகள்" style
        #    No author name present, but topic + content-type word => topic_author
        topic_content_patterns = [
            'பற்றிய படைப்புகள்', 'பற்றிய கட்டுரைகள்', 'பற்றிய கவிதைகள்',
            'பற்றிய கதைகள்', 'பற்றி படைப்புகள்', 'பற்றி கட்டுரைகள்',
            'தொடர்பான படைப்புகள்', 'தொடர்பான கட்டுரைகள்',
            'குறித்த படைப்புகள்', 'குறித்த கட்டுரைகள்',
            'சார்ந்த படைப்புகள்', 'சார்ந்த கட்டுரைகள்',
        ]
        if any(p in q for p in topic_content_patterns):
            return 'topic_author'

        return 'none'

    # -------------------------------------------------------------------------
    def extract_entity(self, question: str, query_type: str) -> str:
        """
        Robust entity extraction.
        - query_type='author_topics' -> returns the AUTHOR name
        - query_type='topic_author'  -> returns the TITLE / TOPIC string

        KEY FIX: Uses Tamil-safe suffix stripping via _strip_tamil_possessive_suffixes()
        instead of regex with \\b word boundaries, which do NOT work with Tamil Unicode.
        """
        q = question.strip()

        # Normalise Ponni variants
        for variation in _PatternBank.PONNI:
            if variation not in ('பொன்னி', 'ponni'):
                q = q.replace(variation, 'பொன்னி')
        q = re.sub(r'ponni\b', 'பொன்னி', q, flags=re.IGNORECASE)

        # ── AUTHOR TOPICS: extract AUTHOR NAME ───────────────────────────────
        if query_type == 'author_topics':

            # 0. Canonical map lookup BEFORE suffix stripping
            #    (some names like அப்பாத்துரை get wrongly stripped by யை rule)
            q_lower_raw = q.lower()
            for pattern, canonical in _PatternBank.AUTHOR_CANONICAL.items():
                if pattern in q_lower_raw:
                    logger.info(f"[CANONICAL-PRESUFFIX] matched '{pattern}' → '{canonical}'")
                    return canonical

            # 1. Strip Tamil possessive/genitive suffixes
            q_stripped = _strip_tamil_possessive_suffixes(q)

            # 2. Canonical map lookup again (post-strip, for நின்/ரின்/யின் suffixed names)
            q_lower = q_stripped.lower()
            for pattern, canonical in _PatternBank.AUTHOR_CANONICAL.items():
                if pattern in q_lower:
                    logger.info(f"[CANONICAL] matched '{pattern}' → '{canonical}'")
                    return canonical

            # ... rest of extraction unchanged

            # 2. Strip all noise phrases (longest first)
            noise = sorted(_PatternBank.AUTHOR_NOISE_PHRASES, key=len, reverse=True)
            q_clean = q_stripped
            for nw in noise:
                q_clean = re.sub(re.escape(nw), ' ', q_clean, flags=re.IGNORECASE)
            q_clean = re.sub(r'\s+', ' ', q_clean).strip()

            # 3. Try initials-style name:  e.g. "ஆ. வ. இராமநாதன்"
            m = _RE_INITIALS_NAME.search(q_clean)
            if m:
                candidate = m.group(1).strip()
                # Strip possessive suffixes from candidate too (Tamil-safe)
                candidate = _strip_tamil_possessive_suffixes(candidate).strip()
                bad = _PatternBank.AUTHOR_NOISE_TOKENS
                if not any(bw in candidate for bw in bad) and len(candidate) > 2:
                    return normalize_author_name(candidate)

            # 4. Collect remaining Tamil tokens (first 3) as name
            tokens = _RE_TAMIL_WORD.findall(q_clean)
            # Strip possessive suffixes from each token (Tamil-safe)
            tokens = [_strip_tamil_possessive_suffix_word(t) for t in tokens]
            tokens = [
                normalize_author_name(t) for t in tokens
                if len(t) > 2 and t not in _PatternBank.AUTHOR_NOISE_TOKENS
            ]
            if tokens:
                return ' '.join(tokens[:3])

            return ''

        # ── TOPIC AUTHOR: extract TITLE / TOPIC ──────────────────────────────
        elif query_type == 'topic_author':

            # Strip noise phrases (longest first).
            # Use DOUBLE SPACE as a replacement marker so we can later detect
            # whether a trailing யார்?/என்ன? was adjacent to removed noise
            # (= question suffix, strip it) or was part of the actual title
            # (= keep it, e.g. "தமிழர் யார்?").
            noise = sorted(_PatternBank.TOPIC_NOISE_PHRASES, key=len, reverse=True)
            q_title = q
            for np_phrase in noise:
                q_title = re.sub(re.escape(np_phrase), '  ', q_title, flags=re.IGNORECASE)

            # Strip title-type suffix words at END only
            for sw in _PatternBank.TITLE_SUFFIXES:
                q_title = re.sub(
                    r'\s*' + re.escape(sw) + r'\s*$', '', q_title.strip(),
                    flags=re.IGNORECASE
                )

            # Strip possessive suffixes that may be on the last word of a title.
            # FIXED: replaced \b with Tamil-safe (?=\s|$) via helper function.
            # e.g. "திறக்குமாவின்" -> "திறக்குமா"
            q_title = _strip_tamil_possessive_suffixes(q_title)

            # Strip Tamil locative suffixes from each word (e.g. section names):
            # "வளரும் இலக்கியத்தில்" -> "வளரும் இலக்கியம்"
            # "கவிதையில்" -> "கவிதை"
            # Rules: த்தில் -> ம், த்தின் -> ம், யில் -> '', இல் at word end -> ''
            _locative_rules = [
                (r'த்தில்(?=\s|$)', 'ம்'),
                (r'த்தின்(?=\s|$)', 'ம்'),
                (r'த்திற்கு(?=\s|$)', 'ம்'),
                (r'யில்(?=\s|$)', ''),
                (r'வில்(?=\s|$)', ''),
            ]
            for _pat, _repl in _locative_rules:
                q_title = re.sub(_pat, _repl, q_title)

            # Remove quotes but keep ? when part of title
            q_title = q_title.replace('"', '').replace("'", '')
            # Strip trailing "யார்?"/"என்ன?" ONLY when preceded by double-space,
            # meaning noise was removed just before it (question suffix, not title).
            # Preserves "தமிழர் யார்?" but strips "சமூக நீதி  [noise removed]  யார்?"
            q_title = re.sub(r'  +(யார்|என்ன|யாவை|எவர்)\??$', '', q_title)
            q_title = re.sub(r'\s+', ' ', q_title).strip()

            # Collect Tamil + latin words
            words = re.findall(r'[\u0B80-\u0BFF!?,।]+|[a-zA-Z]+', q_title)
            words = [
                w for w in words
                # Strip trailing punctuation before noise-token check,
                # BUT if the word ends with '?' (e.g. 'யார்?' as part of title),
                # keep it — only strip 'யார்' etc when standalone without '?'
                if len(w.rstrip('?!,')) > 1
                and not (w.rstrip('?!,').lower() in _PatternBank.TOPIC_NOISE_TOKENS
                         and not w.endswith('?'))
                and w not in {'?', '!', ',', '।'}
            ]

            result = ' '.join(words) if words else ''
            # Post-guard: if entire extracted topic is just a noise word like 'யார்?'
            # (meaning no real topic was found), return empty so caller can handle gracefully
            if result.strip().lower().rstrip('?!') in {'யார்', 'என்ன', 'யாவை', 'எவர்', ''}:
                return ''
            return result

        return ''

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def get_topics_by_author(self, author_name: str) -> Dict:
        """
        Get all articles/topics by a specific author.
        Uses flexible_author_match (fuzzy + edit-distance + token-level).
        """
        if self.df is None or self.df.empty:
            return {
                "type": "author_topics",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }

        matches = self.df[
            self.df['ஆசிரியர்'].apply(
                lambda x: flexible_author_match(author_name, str(x))
            )
        ]

        if matches.empty:
            best_name, best_score = _find_closest_author(self.df, author_name)
            suggestion = ""
            if best_score >= 0.70:
                suggestion = (
                    f"நீங்கள் '{best_name}' என்பவரை குறிப்பிட்டீர்களா? "
                    f"(ஒற்றுமை: {best_score:.0%})"
                )
            return {
                "type": "author_topics",
                "success": False,
                "author": author_name,
                "message": f"'{author_name}' கண்டுபிடிக்க முடியவில்லை",
                "suggestion": suggestion,
                "articles": []
            }

        articles = []
        for _, row in matches.iterrows():
            article = {
                "title":  row.get('தலைப்பு', ''),
                "author": row.get('ஆசிரியர்', '')
            }
            for col in ['ஆண்டு', 'இதழ்', 'ச.எ.', 'வ.எ.']:
                if col in row and pd.notna(row[col]):
                    try:
                        article[col] = (
                            int(row[col]) if col in ['ஆண்டு', 'ச.எ.', 'வ.எ.']
                            else str(row[col])
                        )
                    except Exception:
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

# -------------------------------------------------------------------------
    def get_author_by_topic(self, topic: str) -> Dict:
        """
        Find authors who wrote about a specific topic.
        Matching stages:
        1.  Exact substring (after suffix cleaning)
        2.  All words present
        2b. Majority-word match (>= 60%, min 2 meaningful words, result set <= 8)
        3.  Longest Tamil word (>= 6 chars, result set <= 10)
        4.  Fuzzy title match >= 0.80  (looser than global FUZZY_THRESHOLD=0.95)
        5.  ALL long tokens (>= 5 chars) must match — AND logic
        """
        if self.df is None or self.df.empty:
            return {
                "type": "topic_author",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }

        # Clean common suffixes
        topic_cleaned = topic
        for suffix in _PatternBank.TITLE_SUFFIXES:
            topic_cleaned = re.sub(
                r'\s*' + re.escape(suffix) + r'\s*$', '',
                topic_cleaned.strip(), flags=re.IGNORECASE
            )
        # Strip Tamil possessive suffixes
        topic_cleaned = _strip_tamil_possessive_suffixes(topic_cleaned)
        topic_cleaned = (topic_cleaned
                         .replace("'", "").replace('"', '')
                         .replace('?', '').strip())

        def _build_articles(matches):
            articles = []
            for _, row in matches.iterrows():
                article = {
                    "title":  row.get('தலைப்பு', ''),
                    "author": row.get('ஆசிரியர்', '')
                }
                for col in ['ஆண்டு', 'இதழ்', 'ச.எ.', 'வ.எ.']:
                    if col in row and pd.notna(row[col]):
                        try:
                            article[col] = (
                                int(row[col]) if col in ['ஆண்டு', 'ச.எ.', 'வ.எ.']
                                else str(row[col])
                            )
                        except Exception:
                            article[col] = str(row[col])
                articles.append(article)
            return articles

        # ── Stage 1: exact substring ─────────────────────────────────────────
        matches = self.df[
            self.df['தலைப்பு'].str.contains(
                topic_cleaned, case=False, na=False, regex=False
            )
        ]

        # ── Stage 2: all words present ───────────────────────────────────────
        if matches.empty and len(topic_cleaned.split()) > 1:
            words = topic_cleaned.split()
            matches = self.df[
                self.df['தலைப்பு'].apply(
                    lambda t: pd.notna(t) and all(w.lower() in str(t).lower() for w in words)
                )
            ]

        # ── Stage 2b: majority-word match ────────────────────────────────────
        # Only counts meaningful words (>= 3 chars) so short particles like
        # "இது", "தான்" don't drive false matches.
        # Result-set cap of 8 prevents broad hits from common words.
        if matches.empty and len(topic_cleaned.split()) > 2:
            meaningful_words = [w for w in topic_cleaned.split() if len(w) >= 3]
            if len(meaningful_words) >= 2:
                min_matches = max(2, int(len(meaningful_words) * 0.6))
                majority_rows = []
                for _, row in self.df.iterrows():
                    title = str(row.get('தலைப்பு', ''))
                    if not title:
                        continue
                    # Split title into tokens — prevents "இது" matching "இதுதான்"
                    title_tokens = set(re.findall(r'[\u0B80-\u0BFF]+|[a-zA-Z]+', title.lower()))
                    matched_count = sum(
                        1 for w in meaningful_words
                        if w.lower() in title_tokens  # exact token, not substring
                    )
                    if matched_count >= min_matches:
                        majority_rows.append(row)
                if majority_rows and len(majority_rows) <= 8:
                    matches = pd.DataFrame(majority_rows)
                    logger.info(
                        f"[MAJORITY-MATCH] {len(majority_rows)} rows "
                        f"(min_words={min_matches}, words={meaningful_words})"
                    )

        # ── Stage 3: longest Tamil word (>= 6 chars), result set <= 10 ───────
        # FIXED from min 4 chars: common words like "பிறந்த" (6 chars) were
        # matching hundreds of titles. Now requires truly specific Tamil words
        # AND caps the result set to confirm specificity.
        if matches.empty:
            words = topic_cleaned.split()
            if words:
                # Only consider words made of Tamil Unicode characters, min 6 chars
                tamil_words = [w for w in words if re.search(r'[\u0B80-\u0BFF]{6,}', w)]
                if tamil_words:
                    main_word = max(tamil_words, key=len)
                    candidate = self.df[
                        self.df['தலைப்பு'].str.contains(
                            main_word, case=False, na=False, regex=False
                        )
                    ]
                    if not candidate.empty and len(candidate) <= 10:
                        matches = candidate
                        logger.info(
                            f"[LONGEST-WORD] '{main_word}' → {len(candidate)} results"
                        )

        # ── Stage 4: fuzzy title match >= 0.80 ──────────────────────────────
        # Uses 0.80 (not the global FUZZY_THRESHOLD of 0.95) so that near-miss
        # titles like "காதல் பிறந்த கதை" (89%) match "காதில் பிறந்த கதை",
        # and "இது தான் புராணம்" (≈97%) matches "இது தான் பூராணம்".
        TOPIC_FUZZY_THRESHOLD = 0.80
        if matches.empty:
            fuzzy_rows = []
            for _, row in self.df.iterrows():
                title = str(row.get('தலைப்பு', ''))
                if not title:
                    continue
                score = fuzzy_match_score(topic_cleaned, title)
                if score >= TOPIC_FUZZY_THRESHOLD:
                    logger.info(
                        f"[FUZZY-TOPIC] '{topic_cleaned}' ~ '{title}' score={score:.2f} ✅"
                    )
                    fuzzy_rows.append((row, score))
            if fuzzy_rows:
                # Sort by score descending so best matches appear first
                fuzzy_rows.sort(key=lambda x: x[1], reverse=True)
                matches = pd.DataFrame([r for r, _ in fuzzy_rows])

        # ── Stage 5: ALL long tokens (>= 5 chars) must match — AND logic ─────
        # FIXED from OR logic: previously the first matching token won, letting
        # common words like "பிறந்த" match everything. Now ALL long tokens must
        # be present simultaneously.
        if matches.empty:
            tokens = [t for t in topic_cleaned.split() if len(t) >= 5]
            if len(tokens) >= 2:
                candidate = self.df[
                    self.df['தலைப்பு'].apply(
                        lambda t: pd.notna(t) and all(
                            tok.lower() in str(t).lower() for tok in tokens
                        )
                    )
                ]
                if not candidate.empty:
                    matches = candidate
                    logger.info(f"[MULTI-TOKEN-AND] tokens={tokens}")
            elif len(tokens) == 1 and len(tokens[0]) >= 7:
                # Single very long token — only accept if result set is small (specific)
                candidate = self.df[
                    self.df['தலைப்பு'].str.contains(
                        tokens[0], case=False, na=False, regex=False
                    )
                ]
                if not candidate.empty and len(candidate) <= 5:
                    matches = candidate
                    logger.info(
                        f"[SINGLE-LONG-TOKEN] '{tokens[0]}' → {len(candidate)} results"
                    )

        # ── Nothing found ────────────────────────────────────────────────────
        if matches.empty:
            best_title, best_score = _find_closest_title(self.df, topic_cleaned)
            suggestion = ""
            if best_score >= 0.70:
                suggestion = (
                    f"நீங்கள் '{best_title}' என்ற தலைப்பை குறிப்பிட்டீர்களா? "
                    f"(ஒற்றுமை: {best_score:.0%})"
                )
            return {
                "type": "topic_author",
                "success": False,
                "topic": topic,
                "cleaned_topic": topic_cleaned,
                "message": (
                    f"'{topic}' தலைப்பு கண்டுபிடிக்க முடியவில்லை. "
                    f"தேடிய சொல்: '{topic_cleaned}'"
                ),
                "suggestion": suggestion,
                "articles": [],
            }

        return {
            "type": "topic_author",
            "success": True,
            "topic": topic,
            "cleaned_topic": topic_cleaned,
            "count": len(matches),
            "articles": _build_articles(matches)
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
    """Format topics by author in simple numbered list format."""
    if not result['success']:
        return f"Error: {result['message']}"

    lines = [
        f"எழுத்தாளர்: {result.get('matched_author', result['author'])}",
        f"மொத்த படைப்புகள்: {result['count']}",
        ""
    ]

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

        lines.append(f"{idx}. {' | '.join(parts)}")

    return '\n'.join(lines)


def format_topic_authors(result: Dict) -> str:
    """Format topic authors in simple numbered list format."""
    if not result['success']:
        msg = f"Error: {result['message']}"
        if 'suggestion' in result:
            msg += f"\n\nசிபாரிசு: {result['suggestion']}"
        return msg

    lines = [
        f"தலைப்பு: '{result['topic']}' பற்றிய கட்டுரைகள்",
    ]

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
    Returns raw CSV data for author/topic queries.
    The caller passes this data to the LLM for summarization.
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


def _csv_source(csv_data: str) -> List[Dict]:
    """Return a source entry with the raw CSV-retrieved data as evidence."""
    return [{
        "volume": "பொன்னி கட்டுரை தரவுத்தளம்",
        "heading": "Article Database",
        "doc_issue": "",
        "content": csv_data,
        "word_count": len(csv_data.split()),
        "chunks_merged": 0,
        "score": 1.0,
    }]


def _csv_data_suffix(csv_data: str) -> str:
    """Format the raw CSV data as a suffix to append after the LLM summary."""
    return f"\n\n---\n\n**தரவுத்தள தகவல்:**\n\n{csv_data}"


def _combine_csv_answer(llm_summary: str, csv_data: str) -> str:
    """Combine LLM gist with raw CSV data appended below.

    If the LLM summary is empty (failed), prepends a fallback header
    before the raw CSV data so the separator is always visible.
    """
    suffix = _csv_data_suffix(csv_data)
    if llm_summary and llm_summary.strip():
        return llm_summary.strip() + suffix
    # Fallback: no gist available, still show data with separator
    return "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:" + suffix





class HybridQdrantSearch:
    """Hybrid search combining dense and sparse vectors for optimal results."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def search(self, query: str, limit: int = 30, score_threshold: float = SCORE_THRESHOLD):
        """
        Perform hybrid search using dense and sparse vectors.

        Executes a two-stage search combining dense embeddings for semantic similarity
        and sparse embeddings for keyword matching, then fuses results using RRF.

        Dense prefetch uses a cosine score_threshold (default 0.8) to gate
        semantic quality.  Sparse prefetch is capped at `limit` (not limit*2)
        to avoid flooding the RRF pool with weak keyword matches.

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
                    limit=limit,
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


PONNI_ABOUT_CONTEXT = """பொன்னி இதழ் பற்றிய பின்னணி தகவல்:
பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்ற கலை இலக்கிய இதழ். 1947 முதல் 1955 வரை இயங்கியது. முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது.

காலவரிசை (Timeline): பொன்னி 1947 பிப்ரவரியில் தொடங்கியது. 1947ல் மாதம் ஒரு இதழ், 1948 முதல் மாதம் இரண்டு இதழ்கள். மொத்தம் 8 தொகுதிகள் (volumes) வெளியாகின: தொகுதி 1 (1947), தொகுதி 2 (1948), தொகுதி 3 (1949), தொகுதி 4 (1950), தொகுதி 5 (1951), தொகுதி 6 (1952), தொகுதி 7 (1953), தொகுதி 8 (1954-1955). 1955ல் இதழ் நிறுத்தப்பட்டது.

திராவிட இதழ்களின் வரிசையில் வைத்து போற்றத்தக்க இதழாகப் பொன்னி திகழ்கிறது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகியவற்றை மிகத் தீவிரமாக எடுத்துரைக்கும் இதழாக இவ்விதழ் வெளிவந்தது. வண்ண அட்டைப்படத்தில் மிக நேர்த்தியாக வடிவமைக்கப்பட்டு வெளியிடப்பெற்றது.

கவிஞர் கண்ணதாசன் தன் வனவாசம் புத்தகத்தில் 'திராவிடர் கழகத்தை ஆதரிக்கும் ஏடுகள் மிகக் குறைவாக இருந்த காலம். அந்த நேரத்தில் வண்ண முகப்பு அட்டை போட்டு அழகாக நடந்த இதழ் பொன்னி தான். பத்திரிக்கை துறையில் மற்றவர்கள் செய்துகாட்டாத புதுமை எல்லாம் அவர்கள் செய்து காட்டினார்கள்' என்று குறிப்பிட்டுள்ளார்.

தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர் பாரதிதாசன், திரு.வி.க., கலைஞர் மு. கருணாநிதி, கா. அப்பாதுரையார், கவியரசு கண்ணதாசன், டி. கே. சீனிவாசன், மு. வ., மு. அண்ணாமலை, கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்ற பல முதன்மையான இலக்கிய, அரசியல் ஆளுமைகள் பொன்னியில் எழுதியுள்ளனர்.

கவிதைகள், சிறுகதைகள், தொடர்கதைகள், நொடிக் கதைகள், நாடகங்கள், பொதுக் கட்டுரைகள், ஆய்வுக் கட்டுரைகள், ஒப்பாய்வுக் கட்டுரைகள், தொடர் கட்டுரைகள், செய்திப் பாட்டு போன்ற இலக்கிய வகைமைகளில் படைப்புகள் வெளியாகியுள்ளன.

தமிழ் இலக்கியம் மட்டுமன்றி சீனம், பாரசீகம், ரஷ்ய, கன்னடம், உருது, வடமொழி, தெலுங்கு போன்ற உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர்.

மாநில சுயாட்சி, இந்தித் திணிப்பு, தனித்தமிழ் பற்று, விடுதலை, அரசியல், சமூகம் சார்ந்த திராவிடச் சிந்தனைகள் கட்டுரைகளாக, கதைகளாக, கவிதைகளாக வெளிவந்தன. பாரதிதாசன் அவரின் 'குயில்' இதழ் அரசால் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதினார். 'பாரதிதாசன் பரம்பரை கவிஞர்கள்' என்று அறிமுகப்படுத்தியது பொன்னி இதழ்.

பொன்னி இதழ் ஒரு கலை இலக்கிய இதழாக மட்டுமின்றி புரட்சி இதழாகவே இருந்திருக்கிறது. 1947 முதல் 1955 வரையிலான தமிழகத்தின் காலக் கண்ணாடியாகப் பொன்னி இதழ் விளங்குகிறது."""

TAMIL_ANSWER_SYSTEM_PROMPT = """நீங்கள் பொன்னி இதழ் தொடர்பான கேள்விகளுக்கு பதிலளிக்கும் ஒரு தமிழ் நிபுணர்.

உங்கள் பணி:
1. கொடுக்கப்பட்ட சூழல் (context) மற்றும் கேள்வியின் அடிப்படையில் விரிவான பதில் எழுதுக
2. பதில் 200 முதல் 500 சொற்கள் வரை இருக்க வேண்டும்
3. தெளிவான, எளிமையான, நடைமுறை தமிழில் எழுதுக
4. சூழலில் உள்ள தகவல்களை மட்டுமே பயன்படுத்துக – கற்பனையாக எதையும் சேர்க்காதீர்கள்
5. பதில் வாசிப்பதற்கு மிகவும் எளிதாகவும், நன்கு கட்டமைக்கப்பட்டதாகவும் இருக்க வேண்டும்
6. பதில் தொடங்கும் போதும் முடியும் போதும் எந்தச் சொலும் துண்டிக்கப்பட்டதாக இருக்கக் கூடாது
7. பதிலை எழுதி முடிக்கும் போது, கட்டாயமாக ஒரு முடிவு வாக்கியத்துடன் நிறுத்த வேண்டும். பதில் நடுவில் திடீரென நிற்கக் கூடாது

விவரிப்பு வகை கேள்விகளுக்கான விதி:
- பயனர் விவரிப்பு வகையான கேள்வி கேட்டால் (எ.கா. "விளக்குக", "விவரி", "என்ன?", "எப்படி?"), அனைத்து ஆவணங்களின் உள்ளடக்கத்தையும் ஒருங்கிணைத்து, சுருக்கமாக விவரித்து, ஆதாரங்களுடன் பதிலளிக்கவும்
- ஒவ்வொரு ஆவணத்தின் முக்கிய கருத்துகளையும் தெளிவாக சுருக்கி, முழுமையான பதிலை எழுதுக

சுருக்கம் / தலைப்பு சார்ந்த கேள்விகளுக்கான விதி (மிக முக்கியம்):
- கேள்வி ஒரு குறிப்பிட்ட தலைப்பு, கட்டுரை, அல்லது கருத்தை சுருக்கமாகக் கூற கேட்டால், ஆவண சூழலில் கொடுக்கப்பட்ட அனைத்து ஆவணங்களின் உள்ளடக்கத்தையும் பகுப்பாய்வு செய்து சுருக்கமாக எழுதுக
- சூழலில் உள்ள அனைத்து முக்கிய கருத்துகள், வாதங்கள், மற்றும் தகவல்களை உள்ளடக்கிய முழுமையான சுருக்கத்தை வழங்குக
- "போதுமான தகவல் இல்லை" என்று கூறாதீர்கள் — சூழலில் உள்ள தகவல்களைக் கொண்டு எவ்வளவு முடியுமோ அவ்வளவு விரிவாக பதிலளிக்கவும்
- அனைத்து ஆவணங்களிலிருந்தும் சம அளவில் தகவல்கள் கொடுக்கப்பட்டிருக்கும், ஒவ்வொரு ஆவணத்தின் முக்கிய கருத்துகளையும் பயன்படுத்துக

பொன்னி இதழ் பற்றிய கேள்விகளுக்கான விதி (மிக முக்கியம்):
- கேள்வி பொன்னி இதழைப் பற்றியதாக இருந்தால், கீழே கொடுக்கப்பட்ட "பொன்னி பின்னணி தகவல்" பகுதியை **முதன்மையாக** பயன்படுத்தி பதிலளிக்கவும்
- பின்வரும் வகையான கேள்விகள் அனைத்தும் பொன்னி இதழ் பற்றிய கேள்விகள்:
  * பொன்னி இதழ் என்ன? / பொன்னி பற்றி கூறுக / பொன்னி இதழின் வரலாறு
  * பொன்னி காலவரிசை / timeline / எப்போது தொடங்கியது / எப்போது நிறுத்தப்பட்டது
  * பொன்னி நிறுவனர் / யார் தொடங்கினர் / founder / who started ponni
  * பொன்னி எவ்வளவு காலம் இயங்கியது / how long did ponni run
  * பொன்னி தொகுதிகள் / volumes / எத்தனை இதழ்கள்
  * பொன்னி முக்கியத்துவம் / significance / importance / சிறப்பு
  * பொன்னியில் யார் எழுதினர் / who wrote in ponni / contributors / எழுத்தாளர்கள்
  * பொன்னி உள்ளடக்கம் / content types / என்ன வகையான படைப்புகள்
  * பொன்னி திராவிட இயக்கம் / Dravidian movement / பங்களிப்பு
  * what is ponni / about ponni / ponni magazine / ponni history
  * ponni timeline / ponni founding / ponni duration / ponni volumes
- இந்த வகையான கேள்விகளுக்கு, ஆவண சூழலை (document context) விட பொன்னி பின்னணி தகவலுக்கு முன்னுரிமை கொடுக்கவும்
- பொன்னி பின்னணி தகவலுடன் ஆவண சூழலையும் இணைத்து முழுமையான பதிலை எழுதுக
- கேள்வி "செய்திகள்", "பற்றி", "விவரம்" போன்றதாக இருந்தால்,மொழி பகுப்பாய்வு அல்லது இலக்கண விளக்கம் எழுதக்கூடாது.ஆவணங்களில் உள்ள தகவலை மட்டும் சுருக்கமாக விளக்க வேண்டும்.
========================
பொன்னி பின்னணி தகவல்:
""" + PONNI_ABOUT_CONTEXT + """
========================

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

_CSV_SYSTEM_PROMPT = """நீங்கள் பொன்னி இதழ் கட்டுரை தரவுத்தளத்தின் தகவல்களை சுருக்கமாக விளக்கும் தமிழ் உதவியாளர்.

உங்கள் பணி:
1. கொடுக்கப்பட்ட தரவுத்தள தகவலை பகுப்பாய்வு செய்து, ஒரு சுருக்கமான விளக்கத்தை (gist) எழுதுக
2. தரவை அப்படியே பட்டியலிடாதீர்கள் — மொத்த எண்ணிக்கை, முக்கிய பெயர்கள், பொதுவான போக்குகள் போன்ற உயர்நிலை நுண்ணறிவுகளை மட்டும் குறிப்பிடவும்
3. உதாரணம்: "25 எழுத்தாளர்கள் கண்டறியப்பட்டுள்ளனர்" என்று எழுதுக, அனைத்து 25 பெயர்களையும் பட்டியலிடாதீர்கள்
4. பதிலை இயல்பான தமிழ் உரைநடையில் எழுதுக — புள்ளிவாரியாக அல்ல
5. பதில் 50 முதல் 150 சொற்கள் வரை இருக்க வேண்டும்
6. பதிலை ஒரு முடிவு வாக்கியத்துடன் நிறுத்த வேண்டும்"""


def _truncate_at_sentence_boundary(text: str) -> str:
    """Truncate text at the last complete sentence if it ends mid-sentence."""
    if not text:
        return text
    stripped = text.rstrip()
    if stripped and stripped[-1] in '.?!।':
        return stripped
    last_boundary = max(stripped.rfind('. '), stripped.rfind('.'), stripped.rfind('? '), stripped.rfind('?'),
                        stripped.rfind('! '), stripped.rfind('!'), stripped.rfind('।'))
    if last_boundary > len(stripped) * 0.5:
        return stripped[:last_boundary + 1].rstrip()
    return stripped


def _get_gemini_client():
    """Get or create the Gemini API client (singleton)."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _gemini_generation_config(system_instruction: str = None):
    """Return Gemini generation config with the given system instruction."""
    return genai_types.GenerateContentConfig(
        system_instruction=system_instruction or TAMIL_ANSWER_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        top_p=0.9,
    )


def _build_user_content(question: str, context: str, csv_context: str) -> str:
    """Build the user prompt for vector-search queries. Omits empty sections."""
    parts = [f"கேள்வி:\n{question}\n"]

    if csv_context and csv_context.strip():
        parts.append(f"========================\nCSV உள்ளடக்கம்:\n{csv_context}\n========================\n")

    if context and context.strip():
        parts.append(f"========================\nஆவண சூழல்:\n{context}\n========================\n")

    parts.append("விரிவான பதில் (200-500 சொற்கள்):")
    return "\n".join(parts)


def _build_csv_user_content(question: str, csv_data: str) -> str:
    """Build a focused user prompt for CSV-only queries.

    Instructs the LLM to produce a brief gist/summary of the CSV data
    without repeating the raw data verbatim.  The raw data is appended
    to the answer separately after LLM generation.
    """
    return f"""கேள்வி:
{question}

========================
கட்டுரை தரவுத்தள தகவல்:
{csv_data}
========================

மேலே கொடுக்கப்பட்ட தரவுத்தள தகவலை சுருக்கமாக விளக்கவும் (50-150 சொற்கள்).
முக்கியம்: தரவை அப்படியே திரும்ப எழுதாதீர்கள். எண்ணிக்கைகள், முக்கிய பெயர்கள், மற்றும் பொதுவான போக்குகளை மட்டும் சுருக்கமாக குறிப்பிடவும்.
"""


def generate_llm_answer(
    question: str, context: str, csv_context: str, max_words: int = 500,
    user_content: str = None, system_prompt: str = None,
) -> str:
    """
    Generate LLM answer using Gemini API (synchronous).
    Pass user_content/system_prompt to override defaults (e.g. for CSV queries).
    """
    try:
        client = _get_gemini_client()
        if user_content is None:
            user_content = _build_user_content(question, context, csv_context)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_content,
            config=_gemini_generation_config(system_prompt),
        )

        answer = (response.text or "").strip()

        answer = re.sub(r'[^\S\n]+', ' ', answer)
        answer = re.sub(r'\n{3,}', '\n\n', answer)

        # Check if response was truncated due to token limit
        if (response.candidates and
                response.candidates[0].finish_reason and
                str(response.candidates[0].finish_reason) == "MAX_TOKENS"):
            answer = _truncate_at_sentence_boundary(answer)
            logger.info("Response hit token limit — truncated at sentence boundary")

        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        logger.info(f"Gemini answer generated: {word_count} words")

        return answer

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return ""


async def generate_llm_answer_async(
    question: str, context: str, csv_context: str, max_words: int = 500,
    user_content: str = None, system_prompt: str = None,
) -> str:
    """
    Async version of generate_llm_answer using Gemini API.
    Pass user_content/system_prompt to override defaults (e.g. for CSV queries).
    """
    try:
        client = _get_gemini_client()
        if user_content is None:
            user_content = _build_user_content(question, context, csv_context)

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_content,
            config=_gemini_generation_config(system_prompt),
        )

        answer = (response.text or "").strip()

        answer = re.sub(r'[^\S\n]+', ' ', answer)
        answer = re.sub(r'\n{3,}', '\n\n', answer)

        if (response.candidates and
                response.candidates[0].finish_reason and
                str(response.candidates[0].finish_reason) == "MAX_TOKENS"):
            answer = _truncate_at_sentence_boundary(answer)
            logger.info("Async response hit token limit — truncated at sentence boundary")

        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        logger.info(f"Gemini async answer generated: {word_count} words")

        return answer

    except Exception as e:
        logger.error(f"Gemini async generation failed: {e}")
        return ""


def generate_llm_answer_stream(
    question: str, context: str, csv_context: str,
    user_content: str = None, system_prompt: str = None,
):
    """
    Generate LLM answer using Gemini API with streaming.
    Yields individual token strings as they arrive.
    Pass user_content/system_prompt to override defaults (e.g. for CSV queries).
    """
    try:
        client = _get_gemini_client()
        if user_content is None:
            user_content = _build_user_content(question, context, csv_context)

        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=user_content,
            config=_gemini_generation_config(system_prompt),
        ):
            token = chunk.text
            if token:
                yield token

    except Exception as e:
        logger.error(f"Gemini streaming generation failed: {e}")
        return


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


def _select_relevant_docs(merged_docs: List[Dict]) -> List[Dict]:
    """
    Select relevant documents from merged results using score-gap filtering.

    Three-layer relevance filtering:
    1. **Absolute floor**: score >= 35% of the top document's score
    2. **Gap detection**: stop when a doc scores < 40% of the *previous* doc
       (indicates a sharp relevance drop-off between consecutive results)
    3. **Diminishing returns**: after 20 docs, tighten the gap ratio to 50%
       to prevent long tails of marginally relevant results

    Caps at MAX_SOURCES (100), always returns at least MIN_SOURCES (1).
    De-duplicates by content prefix (using deterministic hash).
    """
    MIN_SOURCES = 1
    MAX_SOURCES = 100
    FLOOR_RATIO = 0.35       # must score >= 35% of top doc
    GAP_RATIO = 0.4          # stop if doc scores < 40% of previous doc
    TIGHT_GAP_RATIO = 0.5    # tighter gap after TIGHT_GAP_AFTER docs
    TIGHT_GAP_AFTER = 20     # tighten gap ratio after this many docs

    if not merged_docs:
        return []

    top_score = merged_docs[0]["score"]
    floor_cutoff = top_score * FLOOR_RATIO

    selected = []
    seen_hashes = set()
    prev_score = top_score

    for doc in merged_docs:
        if len(selected) >= MAX_SOURCES:
            break

        score = doc["score"]

        # After minimum satisfied, check cutoffs
        if len(selected) >= MIN_SOURCES:
            # Hard floor: below 35% of top → stop
            if score < floor_cutoff:
                break
            # Gap: sharp drop from previous doc → stop
            # Tighten gap after TIGHT_GAP_AFTER docs to prevent long tails
            gap = TIGHT_GAP_RATIO if len(selected) >= TIGHT_GAP_AFTER else GAP_RATIO
            if prev_score > 0 and score < prev_score * gap:
                break

        # Deterministic dedup by content prefix
        content_hash = hashlib.md5(doc["content"][:200].encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        selected.append(doc)
        prev_score = score

    if merged_docs:
        logger.info(
            f"[RELEVANCE] {len(selected)}/{len(merged_docs)} docs selected "
            f"(top={top_score:.4f}, floor={floor_cutoff:.4f}, "
            f"last={selected[-1]['score']:.4f})"
        )

    return selected


def build_context_from_docs(
    relevant_docs: List[Dict],
    max_context_chars: int = 15000,
    max_context_docs: int = 15,
) -> str:
    """
    Build LLM context using excerpts from the top relevant documents.

    The evidence set (for user display) can contain up to 100 docs, but the
    LLM context is capped at max_context_docs (default 15) to ensure each
    document gets enough characters (~1000 each) for meaningful analysis.
    Documents are already sorted by score, so the top N are the most relevant.
    """
    if not relevant_docs:
        return ""

    # Cap docs sent to LLM — evidence can be larger, context must be focused
    context_docs = relevant_docs[:max_context_docs]
    n = len(context_docs)
    per_doc_limit = max(500, max_context_chars // n)

    context_parts = []
    for idx, doc in enumerate(context_docs, 1):
        excerpt = doc["content"][:per_doc_limit]
        title = doc.get("heading", "")
        header = f"ஆவணம் {idx}"
        if title:
            header += f" — {title}"
        context_parts.append(f"{header}:\n{excerpt}")

    return "\n\n".join(context_parts)


def format_sources(merged_docs: List[Dict]) -> List[Dict]:
    """
    Format source documents for display.

    Uses _select_relevant_docs() to dynamically determine which documents
    are relevant enough to show as evidence.
    """
    relevant = _select_relevant_docs(merged_docs)

    sources = []
    for doc in relevant:
        sources.append({
            "volume":        doc["volume"],
            "heading":       doc["heading"],
            "doc_issue":     doc["doc_issue"],
            "author_name":   doc.get("author_name", ""),
            "content":       doc["content"],
            "word_count":    doc["word_count"],
            "chunks_merged": doc["chunk_count"],
            "score":         doc["score"],
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

    # --- RESPONSE CACHE LOOKUP ---
    cached = _response_cache.get(question)
    if cached is not None:
        logger.info(f"[TIMING] TOTAL ask_question (CACHED): {time.time() - t_start:.2f}s")
        if return_formatted:
            return format_answer_output(cached["answer"], cached["sources"])
        return cached
    # --- END CACHE LOOKUP ---

    # 2. CHECK CSV QUERIES FIRST (BEFORE VECTOR SEARCH)
    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, csv_response = handle_author_query(question, str(CSV_PATH))

            if is_handled and csv_response and csv_response.strip():
                logger.info("CSV query matched - passing to LLM for summarization")
                t0 = time.time()
                csv_content = _build_csv_user_content(question, csv_response)
                llm_summary = generate_llm_answer(
                    question, context="", csv_context="",
                    user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                )
                logger.info(f"[TIMING] gemini_llm (csv): {time.time() - t0:.2f}s")
                logger.info(f"CSV LLM gist length: {len(llm_summary or '')} chars")

                if not llm_summary or len(llm_summary) < 20:
                    llm_summary = ""

                # Combine: LLM gist + raw CSV data appended
                combined_answer = _combine_csv_answer(llm_summary, csv_response)

                result = {
                    "answer": combined_answer,
                    "sources": [],
                    "query_type": "author_csv",
                }
                _response_cache.put(question, result)

                if return_formatted:
                    return format_answer_output(combined_answer, [])
                return result
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. IF NOT CSV QUERY, PROCEED WITH VECTOR SEARCH + LLM
    try:
        client = get_qdrant_client()

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=200, score_threshold=SCORE_THRESHOLD)
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

        # Select relevant docs (same set used for context + evidence)
        relevant_docs = _select_relevant_docs(merged_docs)
        logger.info(f"Selected {len(relevant_docs)} relevant documents for context")

        # Build context with equal excerpts from all relevant docs
        context = build_context_from_docs(relevant_docs)

        # CSV semantic context
        csv_results = search_csv_semantic(question, top_k=3)
        logger.info("------ CSV Rows Sent To LLM ------")
        for row in csv_results:
            logger.info(row)
        logger.info("----------------------------------")

        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])

        answer = ""
        if use_llm:
            t0 = time.time()
            answer = generate_llm_answer(question, context, csv_context)
            logger.info(f"[TIMING] gemini_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < 150:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            logger.info(f"Extracted {len(facts)} facts")
            answer = generate_extractive_answer(facts, question)

        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs)
        logger.info(f"[TIMING] TOTAL ask_question: {time.time() - t_start:.2f}s | {len(sources)} sources")

        result = {"answer": answer, "sources": sources}
        _response_cache.put(question, result)

        if return_formatted:
            return format_answer_output(answer, sources)

        return result

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
    Gemini async LLM calls. This allows multiple user requests
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

    # --- RESPONSE CACHE LOOKUP ---
    cached = _response_cache.get(question)
    if cached is not None:
        logger.info(f"[TIMING] TOTAL ask_question_async (CACHED): {time.time() - t_start:.2f}s")
        if return_formatted:
            return format_answer_output(cached["answer"], cached["sources"])
        return cached
    # --- END CACHE LOOKUP ---

    if CSV_PATH.exists():
        try:
            logger.info("Checking author query...")
            is_handled, csv_response = await asyncio.to_thread(
                handle_author_query, question, str(CSV_PATH)
            )

            if is_handled and csv_response and csv_response.strip():
                logger.info("CSV query matched - passing to LLM for summarization")
                t0 = time.time()
                csv_content = _build_csv_user_content(question, csv_response)
                llm_summary = await generate_llm_answer_async(
                    question, context="", csv_context="",
                    user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                )
                logger.info(f"[TIMING] async gemini_llm (csv): {time.time() - t0:.2f}s")
                logger.info(f"CSV LLM gist length: {len(llm_summary or '')} chars")

                if not llm_summary or len(llm_summary) < 20:
                    llm_summary = ""

                # Combine: LLM gist + raw CSV data appended
                combined_answer = _combine_csv_answer(llm_summary, csv_response)

                result = {
                    "answer": combined_answer,
                    "sources": [],
                    "query_type": "author_csv",
                }
                _response_cache.put(question, result)

                if return_formatted:
                    return format_answer_output(combined_answer, [])
                return result
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    try:
        client = await asyncio.to_thread(get_qdrant_client)

        logger.info(f"Searching: {question[:60]}...")

        t0 = time.time()
        searcher = HybridQdrantSearch(client)
        results = await asyncio.to_thread(
            searcher.search, question, 200, SCORE_THRESHOLD
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

        # Select relevant docs (same set used for context + evidence)
        relevant_docs = _select_relevant_docs(merged_docs)
        logger.info(f"Selected {len(relevant_docs)} relevant documents for context")

        # Build context with equal excerpts from all relevant docs
        context = build_context_from_docs(relevant_docs)

        # CSV semantic context
        csv_results = await asyncio.to_thread(search_csv_semantic, question, 3)
        logger.info("------ CSV Rows Sent To LLM (async) ------")
        for row in csv_results:
            logger.info(row)
        logger.info("------------------------------------------")

        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])

        answer = ""
        if use_llm:
            t0 = time.time()
            answer = await generate_llm_answer_async(question, context, csv_context)
            logger.info(f"[TIMING] async gemini_llm: {time.time() - t0:.2f}s")

        if not answer or len(answer) < 100:
            logger.warning("LLM failed, using extractive answer")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)

        if not answer or len(answer) < 50:
            answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

        sources = format_sources(merged_docs)
        logger.info(f"[TIMING] TOTAL ask_question_async: {time.time() - t_start:.2f}s | {len(sources)} sources")

        result = {"answer": answer, "sources": sources}
        _response_cache.put(question, result)

        if return_formatted:
            return format_answer_output(answer, sources)

        return result

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

    # --- RESPONSE CACHE LOOKUP ---
    cached = _response_cache.get(question)
    if cached is not None:
        logger.info("[CACHE] Streaming cache hit - yielding full cached answer")
        yield {"type": "token", "content": cached["answer"]}
        yield {"type": "sources", "sources": cached["sources"]}
        return
    # --- END CACHE LOOKUP ---

    # 2. Check CSV queries first — stream LLM summary of CSV data
    if CSV_PATH.exists():
        try:
            is_handled, csv_response = handle_author_query(question, str(CSV_PATH))
            if is_handled and csv_response and csv_response.strip():
                logger.info("CSV query matched - streaming LLM summary")
                csv_content = _build_csv_user_content(question, csv_response)
                accumulated = []
                for token in generate_llm_answer_stream(
                    question, context="", csv_context="",
                    user_content=csv_content, system_prompt=_CSV_SYSTEM_PROMPT,
                ):
                    accumulated.append(token)
                    yield {"type": "token", "content": token}

                llm_summary = "".join(accumulated)
                logger.info(f"CSV streaming gist length: {len(llm_summary)} chars")
                if len(llm_summary) < 20:
                    # Gist too short/empty — emit fallback header
                    fallback = "கட்டுரை தரவுத்தளத்திலிருந்து பெறப்பட்ட தகவல்கள்:"
                    yield {"type": "token", "content": fallback}
                    llm_summary = ""

                # Append raw CSV data separator + data after the streamed summary
                csv_suffix = _csv_data_suffix(csv_response)
                yield {"type": "token", "content": csv_suffix}

                combined_answer = _combine_csv_answer(llm_summary, csv_response)
                _response_cache.put(question, {"answer": combined_answer, "sources": []})
                yield {"type": "sources", "sources": []}
                return
        except Exception as e:
            logger.error(f"CSV query error: {e}")

    # 3. Vector search + LLM streaming
    try:
        client = get_qdrant_client()

        logger.info(f"Streaming search: {question[:60]}...")
        searcher = HybridQdrantSearch(client)
        results = searcher.search(question, limit=200, score_threshold=SCORE_THRESHOLD)

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

        # Select relevant docs (same set used for context + evidence)
        relevant_docs = _select_relevant_docs(merged_docs)
        logger.info(f"Selected {len(relevant_docs)} relevant documents for context")

        # Build context with equal excerpts from all relevant docs
        context = build_context_from_docs(relevant_docs)

        csv_results = search_csv_semantic(question, top_k=3)
        csv_context = ""
        if csv_results:
            csv_context = "\n".join([f"{idx}. {row}" for idx, row in enumerate(csv_results, 1)])

        # Stream LLM tokens and accumulate for caching
        token_count = 0
        accumulated_tokens = []
        for token in generate_llm_answer_stream(question, context, csv_context):
            token_count += 1
            accumulated_tokens.append(token)
            yield {"type": "token", "content": token}

        # If streaming produced too few tokens, fall back to extractive
        if token_count < 10:
            logger.warning("Streaming produced too few tokens, using extractive fallback")
            facts = extract_key_facts(merged_docs, question)
            answer = generate_extractive_answer(facts, question)
            if answer:
                accumulated_tokens = [answer]
                yield {"type": "token", "content": answer}

        sources = format_sources(merged_docs)

        # --- CACHE STORE ---
        full_answer = "".join(accumulated_tokens)
        if full_answer and len(full_answer) >= 50:
            _response_cache.put(question, {"answer": full_answer, "sources": sources})
        # --- END CACHE STORE ---

        yield {"type": "sources", "sources": sources}

    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        yield {"type": "token", "content": f"Query error: {str(e)}"}
        yield {"type": "sources", "sources": []}


def validate_gemini_api():
    """
    Validate that the Gemini API key is configured and working.
    Called at startup to fail fast if misconfigured.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — LLM generation will be unavailable")
        return
    logger.info(f"Validating Gemini API key (model: {GEMINI_MODEL})...")
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="hello",
            config=genai_types.GenerateContentConfig(max_output_tokens=5),
        )
        logger.info("Gemini API validated successfully")
    except Exception as e:
        logger.warning(f"Gemini API validation failed (non-fatal): {e}")


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

    with st.spinner("Validating Gemini API..."):
        validate_gemini_api()
        st.success("Gemini API validated")

    logger.info("=" * 60)
    logger.info("ALL MODELS READY - APP IS READY TO SERVE")
    logger.info("=" * 60)

    st.success("All models loaded successfully! Ready to answer queries.")