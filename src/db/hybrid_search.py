from typing import List, Dict
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


QDRANT_PATH = "/home/ubuntu/Ponni_Rag/PonniRAG/src/db/qdrant_data"
COLLECTION_NAME = "qdrant_indexer"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"   # 🔥 IMPORTANT
        )
    return _embed_model

def dense_embed_query(text: str):
    model = get_embed_model()
    return model.encode(f"query: {text}").tolist()


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

def get_llm():
    global _llm
    if _llm is None:
        _llm = pipeline(
            "text-generation",
            model="abhinand/tamil-llama-7b-instruct-v0.2",
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return _llm



def is_author_question(question: str) -> bool:
    keywords = [
        "author", "authors", "list",
        "எழுத்தாளர்", "எழுத்தாளர்கள்", "பட்டியல்", "யார்"
    ]
    q = question.lower()
    return any(k in q for k in keywords)

def fetch_all_authors(client: QdrantClient) -> List[str]:
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
    )

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
        """Retrieve more chunks initially for merging"""
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_embed_query(query),
                    using="dense",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type",
                                match=models.MatchValue(value="intro")
                            )
                        ]
                    ),
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_embed(query),
                    using="sparse",
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type",
                                match=models.MatchValue(value="intro")
                            )
                        ]
                    ),
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points


def build_context(points) -> str:
    """
    Build context for LLM WITHOUT metadata markers.
    Just clean text separated by newlines.
    """
    blocks = []

    for p in points:
        payload = p.payload or {}
        content = payload.get("content", "").strip()

        # Only include intro type with substantial content
        if payload.get("type") == "intro" and len(content) >= 50:
            # Just add the content, no metadata markers
            blocks.append(content)

    if not blocks:
        return ""

    # Join with double newline for separation
    context = "\n\n".join(blocks)
    return context[:4000]


def retrieve_all_chunks_for_document(client: QdrantClient, doc_id: str, doc_issue: str, volume: str) -> List[Dict]:
    """
    Retrieve ALL chunks for a specific document from Qdrant.
    This fetches the complete document by getting all its chunks.
    """
    all_chunks = []
    offset = None
    
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="intro")
                    ),
                    models.FieldCondition(
                        key="metadata.doc_id",
                        match=models.MatchValue(value=doc_id)
                    ),
                    models.FieldCondition(
                        key="metadata.doc_issue",
                        match=models.MatchValue(value=doc_issue)
                    ),
                    models.FieldCondition(
                        key="metadata.volume",
                        match=models.MatchValue(value=volume)
                    ),
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
    For each document found in search results, retrieve ALL its chunks
    from Qdrant and merge them to show complete content.
    """
    seen_docs = set()
    merged_docs = []
    
    for p in points:
        payload = p.payload or {}
        if payload.get("type") != "intro":
            continue
            
        metadata = payload.get("metadata", {})
        
        doc_key = (
            metadata.get("volume"),
            metadata.get("doc_id"),
            metadata.get("doc_issue"),
        )
        
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
                "content": chunk_payload.get("content", "").strip(),
            })
        
        chunk_data.sort(key=lambda x: x["chunk_id"])
        
        full_content = " ".join(chunk["content"] for chunk in chunk_data)
        full_content = re.sub(r'\s+', ' ', full_content).strip()
        
        word_count = len(full_content.split())
        
        if word_count < 50:
            continue
        
        merged_docs.append({
            "volume": metadata.get("volume"),
            "doc_id": metadata.get("doc_id"),
            "doc_issue": metadata.get("doc_issue"),
            "heading": metadata.get("heading"),
            "content": full_content,
            "word_count": word_count,
            "chunk_count": len(chunk_data),
            "score": p.score,
        })
    
    merged_docs.sort(key=lambda x: x["score"], reverse=True)
    
    return merged_docs


def format_sources(client: QdrantClient, points, limit: int = 10) -> List[Dict]:
    """
    Format sources by retrieving complete documents from Qdrant.
    Returns up to 'limit' unique, complete documents with 100+ words.
    """
    merged_docs = merge_consecutive_chunks(client, points)
    
    sources = []
    for doc in merged_docs:
        sources.append({
            "volume": doc["volume"],
            "heading": doc["heading"],
            "doc_issue": doc["doc_issue"],
            "content": doc["content"],
            "word_count": doc["word_count"],
            "chunks_merged": doc["chunk_count"],
        })
        
        if len(sources) >= limit:
            break
    
    return sources


def call_llm(question: str, context: str) -> str:
    """
    Call LLM with improved prompt to generate original answer, not copy text
    """
    prompt = f"""நீங்கள் ஒரு தமிழ் உதவியாளர். கொடுக்கப்பட்ட தகவல்களைப் படித்து, கேள்விக்கு உங்கள் சொந்த வார்த்தைகளில் சுருக்கமாக பதிலளிக்கவும்.

முக்கியம்: தகவலை அப்படியே எழுத வேண்டாம். முக்கிய கருத்துகளை மட்டும் சுருக்கி கூறவும்.

தகவல்கள்:
{context}

கேள்வி: {question}

பதில்:"""

    try:
        llm = get_llm()

        output = llm(
            prompt,
            max_new_tokens=128,   # 🔥 no quality loss
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            eos_token_id=llm.tokenizer.eos_token_id,
        )[0]["generated_text"]


        # Extract answer after "பதில்:"
        if "பதில்:" in output:
            answer = output.split("பதில்:")[-1].strip()
        else:
            answer = output.strip()
        
        # Clean up metadata markers aggressively
        answer = re.sub(r'\[TEXT[^\]]*\]', '', answer)
        answer = re.sub(r'Volume=\w+', '', answer)
        
        # Remove incomplete sentences at the end
        # If answer ends with incomplete word (no punctuation), remove last sentence
        if answer and not answer[-1] in '.!?।':
            # Find last complete sentence
            sentences = re.split(r'[.!?।]+', answer)
            if len(sentences) > 1:
                answer = '.'.join(sentences[:-1]) + '.'
        
        # Clean up extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # If answer is too short or seems like it's copying, return a generic response
        if len(answer) < 20 or answer.startswith('[TEXT'):
            return "கொடுக்கப்பட்ட கேள்விக்கு தகவல்கள் கிடைத்துள்ளன. மூலங்களைப் பார்க்கவும்."
        
        return answer
        
    except Exception as e:
        return "பதில் உருவாக்குவதில் சிக்கல் ஏற்பட்டது. மூலங்களைப் பார்க்கவும்."
    finally:
        torch.cuda.empty_cache()


def ask_question(question: str, top_k: int = 10) -> Dict:
    """
    Ask a question and get answer with complete, merged sources.
    
    Args:
        question: User's question
        top_k: Number of unique documents to return (default 10)
    
    Returns:
        Dict with 'answer' and 'sources' keys containing complete documents
    """
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent
    QDRANT_PATH = str(BASE_DIR / "qdrant_data")
    
    client = QdrantClient(path=QDRANT_PATH)

    if is_author_question(question):
        authors = fetch_all_authors(client)
        return {
            "answer": format_authors_tamil(authors) if authors else "தகவல் இல்லை",
            "sources": []
        }

    searcher = HybridQdrantSearch(client)
    results = searcher.search(question, limit=30)

    if not results:
        return {"answer": "தகவல் இல்லை", "sources": []}

    context = build_context(results)
    if not context:
        return {"answer": "தகவல் இல்லை", "sources": []}

    sources = format_sources(client, results, limit=top_k)

    return {
        "answer": call_llm(question, context),
        "sources": sources
    }