from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.set_grad_enabled(False)
from qdrant_client import models
from transformers import pipeline
import pandas as pd
from pathlib import Path


QDRANT_PATH = "/home/ubuntu/Ponni_Rag/PonniRAG/src/db/qdrant_data"
COLLECTION_NAME = "qdrant_indexer"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CSV_PATH = Path("/home/ubuntu/Ponni_Rag/PonniRAG/src/db/summary.csv")


_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("🔄 Loading embedding model...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        print("✓ Embedding model cached")
    return _embed_model

def dense_embed_query(text: str):
    return get_embed_model().encode(f"query: {text}").tolist()

def sparse_embed(text: str):
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    indices, values = [], []
    for token, freq in counts.items():
        indices.append(abs(hash(token)) % (2**31))
        values.append(float(freq))
    return models.SparseVector(indices=indices, values=values)


_llm = None
_llm_lock = False

def get_llm():
    global _llm, _llm_lock
    if _llm is not None:
        return _llm
    if _llm_lock:
        import time
        while _llm_lock:
            time.sleep(0.5)
        return _llm
    _llm_lock = True
    try:
        print("🔄 Loading LLM model...")
        _llm = pipeline(
            "text-generation",
            model="abhinand/tamil-llama-7b-instruct-v0.2",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("✓ LLM model cached")
    finally:
        _llm_lock = False
    return _llm


# ============================================================================
# AUTHOR NAME NORMALIZATION AND MATCHING
# ============================================================================

def normalize_author_name(name: str) -> str:
    """
    Normalize author names by removing prefixes and common variations
    Examples:
        'மு.,கருணாநிதி' -> 'கருணாநிதி'
        'மு. கருணாநிதி' -> 'கருணாநிதி'
        'டாக்டர். பெரியார்' -> 'பெரியார்'
    """
    if not name:
        return ""
    
    # Remove common prefixes
    prefixes_to_remove = [
        r'மு\.,?\s*',           # மு., மு.
        r'டாக்டர்\.?\s*',      # டாக்டர்.
        r'திரு\.,?\s*',         # திரு., திரு
        r'திருமதி\.?\s*',      # திருமதி.
        r'Dr\.?\s*',           # Dr.
        r'Mr\.?\s*',           # Mr.
        r'Mrs\.?\s*',          # Mrs.
    ]
    
    cleaned = name.strip()
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def flexible_author_match(search_name: str, csv_name: str) -> bool:
    """
    Flexible matching between search query and CSV author name
    Returns True if they match
    """
    # Normalize both names
    search_normalized = normalize_author_name(search_name).lower()
    csv_normalized = normalize_author_name(csv_name).lower()
    
    # Exact match after normalization
    if search_normalized == csv_normalized:
        return True
    
    # Check if search term is contained in CSV name
    if search_normalized in csv_normalized:
        return True
    
    # Check if CSV name is contained in search term
    if csv_normalized in search_normalized:
        return True
    
    # Handle special cases for common names
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


# ============================================================================
# ENHANCED CSV-BASED AUTHOR QUERY SYSTEM
# ============================================================================

class EnhancedAuthorQuerySystem:
    """
    Robust author query system with improved CSV parsing and query detection
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_csv()
    
    def _load_csv(self):
        """Load CSV with multiple fallback strategies"""
        try:
            if not self.csv_path.exists():
                print(f"❌ CSV not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return
            
            print(f"📂 Loading CSV: {self.csv_path}")
            
            # Try multiple parsing strategies
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
                    print(f"✓ Loaded with: {strategy_name}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠ {strategy_name} failed: {str(e)[:60]}")
                    continue
            
            if not loaded or self.df is None or self.df.empty:
                print("❌ All loading strategies failed")
                self.df = pd.DataFrame()
                return
            
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            print(f"📋 Columns: {list(self.df.columns)}")
            print(f"📊 Rows: {len(self.df)}")
            
            # Check and fix column names
            self._fix_column_names()
            
            # Clean data
            if 'ஆசிரியர்' in self.df.columns:
                self.df['ஆசிரியர்'] = self.df['ஆசிரியர்'].fillna('').astype(str).str.strip()
                before = len(self.df)
                self.df = self.df[self.df['ஆசிரியர்'] != '']
                print(f"✓ Authors: {before} → {len(self.df)} rows")
            else:
                print(f"⚠ Missing 'ஆசிரியர்' column!")
                return
            
            if 'தலைப்பு' in self.df.columns:
                self.df['தலைப்பு'] = self.df['தலைப்பு'].fillna('').astype(str).str.strip()
            
            # Show sample
            if len(self.df) > 0:
                sample = self.df.iloc[0]
                print(f"📄 Sample: {sample.get('ஆசிரியர்', 'N/A')[:30]}")
            
        except Exception as e:
            print(f"❌ CSV error: {e}")
            import traceback
            traceback.print_exc()
            self.df = pd.DataFrame()
    
    def _fix_column_names(self):
        """Map alternate column names"""
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
                print(f"✓ Renamed '{old}' → '{new}'")
    
    def detect_query_type(self, question: str) -> str:
        """Enhanced query type detection"""
        q = question.lower()
        
        # Known authors
        known_authors = [
            'கலைஞர்', 'கருணாநிதி', 'பெரியார்', 'அண்ணா', 'அண்ணாதுரை',
            'நக்கீரன்', 'பாரதிதாசன்', 'புதுமைப்பித்தன்', 'அகிலன்', 'கண்ணதாசன்'
        ]
        has_author = any(author in q for author in known_authors)
        
        # List all authors (highest priority if no specific author)
        list_patterns = [
            'எழுத்தாளர்கள் யார்', 'ஆசிரியர்கள் யார்',
            'எழுத்தாளர்களின் பட்டியல்', 'எழுத்தாளர்கள் பட்டியல்',
            'அனைத்து எழுத்தாளர்', 'எழுத்தாளர்கள் எல்லாம்',
            'list all authors', 'இதழில் எழுதிய ஆசிரியர்கள்',
            'இதழில் எழுதிய எழுத்தாளர்கள்'
        ]
        
        if any(p in q for p in list_patterns) and not has_author:
            return 'list_all_authors'
        
        # What did author X write? (author_topics)
        author_action_patterns = [
            'என்ன எழுதினார்', 'எழுதிய தலைப்பு', 'எழுதியது',
            'எந்த தலைப்பு', 'பற்றி எழுதினார்', 'எழுதியவற்றை',
            'எழுதிய கட்டுரைகள்', 'wrote what', 'articles by',
            'குறிப்பிடுக'
        ]
        
        if has_author or any(p in q for p in author_action_patterns):
            return 'author_topics'
        
        # Who wrote about topic X? (topic_author)
        topic_patterns = [
            'யார் எழுதினார்', 'எழுதியவர் யார்', 'ஆசிரியர் யார்',
            'என்ற நூலை எழுதியவர்', 'என்ற கட்டுரை',
            'who wrote', 'author of'
        ]
        
        if any(p in q for p in topic_patterns):
            return 'topic_author'
        
        return 'none'
    
    def extract_entity(self, question: str, query_type: str) -> str:
        """Extract author name or topic with improved recognition"""
        q = question.strip()
        
        if query_type == 'author_topics':
            # Known author patterns (including variations)
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
            
            # Check for known authors first
            for pattern, canonical in known_authors.items():
                if pattern in q:
                    return canonical
            
            # Extract from patterns like "கலைஞர் கருணாநிதி எழுதியவற்றை"
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
                    # Normalize the extracted name
                    return normalize_author_name(extracted)
            
            # Remove noise words
            noise = [
                'என்ன எழுதினார்', 'எழுதியது', 'எழுதிய', 'தலைப்பு',
                'பற்றி', 'இதழில்', 'பொன்னி', 'குறிப்பிடுக', 'என்ன',
                'எழுதியவற்றை', 'எழுதியவற்றைக்'
            ]
            
            for word in noise:
                q = re.sub(word, '', q, flags=re.IGNORECASE)
            
            # Extract remaining Tamil words
            words = re.findall(r'[\u0B80-\u0BFF]+', q)
            words = [normalize_author_name(w) for w in words if len(w) > 2]
            
            return ' '.join(words) if words else ''
        
        elif query_type == 'topic_author':
            # Remove question markers
            noise = [
                'யார் எழுதினார்', 'எழுதியவர் யார்', 'ஆசிரியர் யார்',
                'என்ற நூலை', 'எழுதியவர்', 'பற்றி', 'இதழில்',
                'பொன்னி', 'என்ற'
            ]
            
            for word in noise:
                q = re.sub(word, '', q, flags=re.IGNORECASE)
            
            # Extract topic
            words = re.findall(r'[\u0B80-\u0BFF]+|[a-zA-Z]+', q)
            words = [w for w in words if len(w) > 2]
            
            return ' '.join(words) if words else ''
        
        return ''
    
    def list_all_authors(self) -> Dict:
        """List all unique authors"""
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
        """Get topics by author with enhanced flexible matching"""
        if self.df is None or self.df.empty:
            return {
                "type": "author_topics",
                "success": False,
                "message": "CSV தரவு கிடைக்கவில்லை",
                "articles": []
            }
        
        # Use flexible matching
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
            
            # Optional fields
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
        """Find who wrote about topic"""
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
    """Format list of all authors - simplified without decorative lines"""
    if not result['success']:
        return f"❌ {result['message']}"
    
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
    """Format topics by author - simplified without decorative lines"""
    if not result['success']:
        return f"❌ {result['message']}"
    
    lines = [
        f"எழுத்தாளர்: {result.get('matched_author', result['author'])}",
        f"மொத்த கட்டுரைகள்: {result['count']}",
        ""
    ]
    
    # Group by year if available
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
    """Format authors by topic - simplified without decorative lines"""
    if not result['success']:
        return f"❌ {result['message']}"
    
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


def handle_author_query(question: str, csv_path: str) -> Tuple[bool, str]:
    """Main handler for author queries"""
    system = EnhancedAuthorQuerySystem(csv_path)
    
    if system.df is None or system.df.empty:
        return False, ""
    
    query_type = system.detect_query_type(question)
    print(f"🔍 Query type: {query_type}")
    
    if query_type == 'none':
        return False, ""
    
    if query_type == 'list_all_authors':
        result = system.list_all_authors()
        return True, format_author_list(result)
    
    elif query_type == 'author_topics':
        entity = system.extract_entity(question, 'author_topics')
        print(f"📝 Extracted author: '{entity}'")
        
        if not entity:
            return True, "⚠ எழுத்தாளர் பெயரை தெளிவாக குறிப்பிடவும்"
        
        result = system.get_topics_by_author(entity)
        return True, format_author_topics(result)
    
    elif query_type == 'topic_author':
        entity = system.extract_entity(question, 'topic_author')
        print(f"📝 Extracted topic: '{entity}'")
        
        if not entity:
            return True, "⚠ தலைப்பை தெளிவாக குறிப்பிடவும்"
        
        result = system.get_author_by_topic(entity)
        return True, format_topic_authors(result)
    
    return False, ""


# ============================================================================
# QDRANT SEARCH FUNCTIONS
# ============================================================================

def is_author_question(question: str) -> bool:
    keywords = ["author", "authors", "list", "எழுத்தாளர்", "எழுத்தாளர்கள்", "பட்டியல்", "யார்"]
    return any(k in question.lower() for k in keywords)

def fetch_all_authors(client: QdrantClient) -> List[str]:
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
    if not authors:
        return "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
    lines = ["பொன்னி இதழில் எழுதிய எழுத்தாளர்கள்:\n"]
    for author in authors:
        lines.append(f"• {author}")
    return "\n".join(lines)


class HybridQdrantSearch:
    def __init__(self, client: QdrantClient):
        self.client = client

    def search(self, query: str, limit: int = 30):
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_embed_query(query),
                    using="dense",
                    filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="intro"))]),
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_embed(query),
                    using="sparse",
                    filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="intro"))]),
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points


def retrieve_all_chunks_for_document(client: QdrantClient, doc_id: str, doc_issue: str, volume: str) -> List[Dict]:
    all_chunks = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="intro")),
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
    seen_docs = set()
    merged_docs = []
    
    for p in points:
        payload = p.payload or {}
        if payload.get("type") != "intro":
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
            "volume": metadata.get("volume"),
            "doc_id": metadata.get("doc_id"),
            "doc_issue": metadata.get("doc_issue"),
            "heading": metadata.get("heading", ""),
            "content": full_content,
            "word_count": word_count,
            "chunk_count": len(chunk_data),
            "score": p.score,
        })
    
    merged_docs.sort(key=lambda x: x["score"], reverse=True)
    return merged_docs


def extract_key_facts(docs: List[Dict], question: str) -> List[Dict]:
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


# ============================================================================
# LLM GENERATION WITH ENHANCED SYSTEM PROMPT
# ============================================================================

TAMIL_ANSWER_SYSTEM_PROMPT = """நீங்கள் பொன்னி இதழ் பற்றிய கேள்விகளுக்கு பதிலளிக்கும் ஒரு தமிழ் நிபுணர்.

உங்கள் பணி:
1. கொடுக்கப்பட்ட சூழல் (context) மற்றும் கேள்வியின் அடிப்படையில் விரிவான பதில் எழுதுக
2. பதில் 200 முதல் 500 சொற்கள் வரை இருக்க வேண்டும்
3. தெளிவான, எளிமையான தமிழில் எழுதுக
4. சூழலில் உள்ள தகவல்களை மட்டுமே பயன்படுத்துக - கற்பனையாக எதையும் சேர்க்காதீர்கள்
5. பதில் நன்கு கட்டமைக்கப்பட்டதாகவும், படிக்க எளிதாகவும் இருக்க வேண்டும்

எழுதும் முறை:
- முதல் வாக்கியத்தில் கேள்விக்கான நேரடி பதிலை தெளிவாக கூறுக
- பின்னர் கூடுதல் விவரங்கள், சூழல், எடுத்துக்காட்டுகள் போன்றவற்றை விளக்குக
- முக்கியமான புள்ளிகளை தனித்தனியாக விளக்குக
- இறுதியில் சுருக்கமான முடிவுரை கொடுக்கலாம்

கவனிக்க வேண்டியவை:
- சூழலில் இல்லாத தகவல்களை எழுதாதீர்கள்
- "சூழலின் படி" அல்லது "ஆதாரத்தின் படி" என்று குறிப்பிட வேண்டாம் - நேரடியாக எழுதுக
- எளிய, நடைமுறை தமிழ் பயன்படுத்துக - வாசிப்பதற்கு எளிதாக இருக்க வேண்டும்

இப்போது கீழே கொடுக்கப்பட்ட கேள்வி மற்றும் சூழலின் அடிப்படையில் விரிவான பதில் எழுதுக."""


def generate_llm_answer(question: str, context: str, max_words: int = 500) -> str:
    """
    Generate a detailed answer using LLM (200-500 words)
    """
    try:
        llm = get_llm()
        
        # Prepare prompt
        prompt = f"""{TAMIL_ANSWER_SYSTEM_PROMPT}

கேள்வி: {question}

சூழல்:
{context}

விரிவான பதில் (200-500 சொற்கள்):"""
        
        # Generate response
        print("🤖 Generating LLM answer...")
        outputs = llm(
            prompt,
            max_new_tokens=800,  # Increased for 500 words
            temperature=0.0,
            top_p=0.9,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=llm.tokenizer.eos_token_id,
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Extract only the answer part (after the prompt)
        answer = generated_text.split("விரிவான பதில் (200-500 சொற்கள்):")[-1].strip()
        
        # Clean up
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Count words
        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        print(f"✓ Generated answer: {word_count} words")
        
        return answer
        
    except Exception as e:
        print(f"⚠ LLM generation failed: {e}")
        return ""


def generate_extractive_answer(facts: List[Dict], question: str) -> str:
    """
    Fallback extractive answer if LLM fails
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
    lines = []
    
    lines.append("📝 பதில்:")
    lines.append(answer)
    lines.append("")
    
    if sources:
        lines.append(f"\n📚 ஆதாரங்கள் ({len(sources)} ஆவணங்கள்):")
        lines.append("")
        
        for idx, source in enumerate(sources, 1):
            lines.append(f"📄 ஆதாரம் {idx}")
            
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
    Enhanced ask_question with CSV-based author query support and LLM generation
    
    Args:
        question: User's question
        top_k: Number of sources to return
        return_formatted: Return formatted string or dict
        use_llm: Use LLM for answer generation (200-500 words)
    """
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    QDRANT_PATH = str(BASE_DIR / "qdrant_data")
    
    # Check CSV-based author queries first
    print(f"🔍 CSV path: {CSV_PATH}")
    print(f"📁 Exists: {CSV_PATH.exists()}")
    
    if CSV_PATH.exists():
        try:
            print(f"🔄 Checking author query...")
            is_handled, response = handle_author_query(question, str(CSV_PATH))
            
            if is_handled:
                print(f"✓ Handled as CSV author query")
                if return_formatted:
                    return response
                return {
                    "answer": response,
                    "sources": [],
                    "query_type": "author_csv"
                }
            else:
                print(f"✓ Not author query, using Qdrant")
        except Exception as e:
            print(f"⚠ CSV query error: {e}")
    else:
        print(f"⚠ CSV not found at: {CSV_PATH}")
    
    # Regular Qdrant search
    client = QdrantClient(path=QDRANT_PATH)

    if is_author_question(question):
        authors = fetch_all_authors(client)
        answer = format_authors_tamil(authors) if authors else "எழுத்தாளர் தகவல்கள் கிடைக்கவில்லை."
        if return_formatted:
            return format_answer_output(answer, [])
        return {"answer": answer, "sources": []}

    print(f"🔍 Searching: {question[:60]}...")
    
    searcher = HybridQdrantSearch(client)
    results = searcher.search(question, limit=50)

    if not results:
        answer = "மன்னிக்கவும், தகவல்கள் கிடைக்கவில்லை."
        if return_formatted:
            return format_answer_output(answer, [])
        return {"answer": answer, "sources": []}

    print(f"✓ Found {len(results)} chunks")

    merged_docs = merge_consecutive_chunks(client, results)
    
    if not merged_docs:
        answer = "போதுமான தகவல்கள் இல்லை."
        if return_formatted:
            return format_answer_output(answer, [])
        return {"answer": answer, "sources": []}
    
    print(f"✓ Merged into {len(merged_docs)} documents")

    # Prepare context for LLM
    context_parts = []
    for idx, doc in enumerate(merged_docs[:5], 1):
        context_parts.append(f"ஆவணம் {idx}: {doc['content'][:800]}")
    
    context = "\n\n".join(context_parts)
    
    # Try LLM generation first
    answer = ""
    if use_llm:
        answer = generate_llm_answer(question, context)
    
    # Fallback to extractive if LLM fails
    if not answer or len(answer) < 100:
        print("⚠ LLM failed, using extractive answer")
        facts = extract_key_facts(merged_docs, question)
        print(f"✓ Extracted {len(facts)} facts")
        answer = generate_extractive_answer(facts, question)
    
    if not answer or len(answer) < 50:
        answer = "கேள்விக்கான தகவல்கள் ஆதாரங்களில் உள்ளன."

    sources = format_sources(merged_docs, limit=top_k)
    print(f"✓ Ready with {len(sources)} sources")

    if return_formatted:
        return format_answer_output(answer, sources)
    
    return {
        "answer": answer,
        "sources": sources
    }


def preload_models():
    print("=" * 60)
    print("PRELOADING MODELS")
    print("=" * 60)
    _ = get_embed_model()
    _ = get_llm()
    print("=" * 60)
    print("✓ MODELS READY")
    print("=" * 60)