"""
Production-Ready Tamil RAG Pipeline
====================================
Complete implementation for Tamil archive search with best practices
Uses: BGE-M3 for embeddings + Tamil-Llama/Claude for generation
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
import logging
from dataclasses import dataclass
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structure for search results"""
    document: str
    score: float
    metadata: Dict
    rank: int


class TamilRAGPipeline:
    """
    Production-grade RAG pipeline optimized for Tamil text
    
    Features:
    - BGE-M3 embeddings (best for Tamil)
    - Cross-encoder reranking
    - Caching for performance
    - Batch processing support
    - Error handling
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        collection_name: str = "tamil_archive",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_model: Model for document embeddings
            reranker_model: Cross-encoder for reranking
            collection_name: ChromaDB collection name
            persist_directory: Directory to persist vector DB
        """
        logger.info(f"Initializing Tamil RAG Pipeline...")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Load reranker
        logger.info(f"Loading reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        
        # Initialize vector database
        if persist_directory:
            self.client = chromadb.Client(
                Settings(persist_directory=persist_directory)
            )
        else:
            self.client = chromadb.Client()
        
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        self.collection_name = collection_name
        
    @lru_cache(maxsize=1000)
    def _cached_embed(self, text: str) -> np.ndarray:
        """Cache embeddings for frequently queried texts"""
        return self.embedder.encode([text], convert_to_numpy=True)[0]
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Index documents into vector database with batching
        
        Args:
            documents: List of Tamil documents to index
            metadatas: Optional metadata for each document
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Dictionary with indexing statistics
        """
        if not documents:
            logger.warning("No documents to index")
            return {"indexed": 0}
        
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(documents))]
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.embedder.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings)
        
        # Add to vector database
        self.collection.add(
            embeddings=[emb.tolist() for emb in all_embeddings],
            documents=documents,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        
        logger.info(f"✓ Indexed {len(documents)} documents")
        return {"indexed": len(documents)}
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        rerank_top: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents with reranking
        
        Args:
            query: Tamil search query
            top_k: Number of documents to retrieve initially
            rerank_top: Number of top documents after reranking
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self._cached_embed(query)
        
        # Initial retrieval
        where_clause = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        if not results['documents'][0]:
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Rerank using cross-encoder
        pairs = [[query, doc] for doc in results['documents'][0]]
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine and sort
        combined = list(zip(
            rerank_scores,
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ))
        combined.sort(reverse=True, key=lambda x: x[0])
        
        # Create SearchResult objects
        search_results = [
            SearchResult(
                document=doc,
                score=float(score),
                metadata=meta,
                rank=i + 1
            )
            for i, (score, doc, meta, _) in enumerate(combined[:rerank_top])
        ]
        
        return search_results
    
    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_length: int = 2048
    ) -> str:
        """
        Generate answer using retrieved context
        
        Note: This is a template. Integrate with your LLM:
        - Tamil-Llama-7B-Instruct for local deployment
        - Claude API for best quality
        - Llama 3.3 for multilingual support
        
        Args:
            query: User query in Tamil
            search_results: Retrieved documents
            max_context_length: Maximum context tokens
            
        Returns:
            Generated answer
        """
        # Prepare context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(
                f"[ஆதாரம் {result.rank}]: {result.document}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Truncate if needed (rough approximation)
        if len(context) > max_context_length * 4:  # ~4 chars per token
            context = context[:max_context_length * 4]
        
        # Template prompt for Tamil LLM
        prompt = f"""கொடுக்கப்பட்ட ஆதாரங்களின் அடிப்படையில் கேள்விக்கு விரிவாக பதிலளிக்கவும்.

ஆதாரங்கள்:
{context}

கேள்வி: {query}

வழிகாட்டுதல்கள்:
1. ஆதாரங்களில் உள்ள தகவல்களை மட்டுமே பயன்படுத்தவும்
2. குறிப்பிட்ட ஆதாரங்களை [ஆதாரம் X] எனக் குறிப்பிடவும்
3. தெளிவாகவும் சுருக்கமாகவும் பதிலளிக்கவும்

பதில்:"""
        
        # TODO: Integrate with your chosen LLM
        # Example with Tamil-Llama:
        # from transformers import pipeline
        # llm = pipeline("text-generation", 
        #                model="abhinand/tamil-llama-7b-instruct-v0.2")
        # response = llm(prompt, max_new_tokens=512)
        # return response[0]['generated_text']
        
        return f"TEMPLATE RESPONSE - Integrate with LLM\nPrompt:\n{prompt}"
    
    def rag_query(
        self,
        query: str,
        top_k: int = 20,
        rerank_top: int = 5
    ) -> Dict:
        """
        Complete RAG query: search + generate
        
        Args:
            query: Tamil query string
            top_k: Initial retrieval count
            rerank_top: Documents for context
            
        Returns:
            Dictionary with answer and sources
        """
        # Search
        search_results = self.search(
            query=query,
            top_k=top_k,
            rerank_top=rerank_top
        )
        
        if not search_results:
            return {
                "query": query,
                "answer": "மன்னிக்கவும், உங்கள் கேள்விக்கான பதிலை காணவில்லை.",
                "sources": []
            }
        
        # Generate answer
        answer = self.generate_answer(query, search_results)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "rank": r.rank,
                    "document": r.document,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in search_results
            ]
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dimension": 1024  # BGE-M3
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """Example: Basic RAG pipeline usage"""
    print("="*70)
    print("EXAMPLE 1: Basic Tamil RAG Pipeline")
    print("="*70)
    
    # Initialize pipeline
    rag = TamilRAGPipeline()
    
    # Sample Tamil documents (from archive)
    documents = [
        "தமிழ் இலக்கியத்தின் தொன்மையான நூல் தொல்காப்பியம் ஆகும். இது மொழியியல் மற்றும் இலக்கண விதிகளை விளக்குகிறது.",
        "சங்க காலத்தில் தமிழகம் மூன்று பெரிய அரசுகளால் ஆளப்பட்டது - சேர, சோழ மற்றும் பாண்டிய.",
        "திருக்குறள் திருவள்ளுவரால் இயற்றப்பட்ட உலகப் பொதுமறை. இது 1330 குறள்களைக் கொண்டது.",
        "கம்பராமாயணம் கம்பரால் இயற்றப்பட்ட காவியம். இது வால்மீகி ராமாயணத்தின் தமிழ் வடிவம்.",
        "சிலப்பதிகாரம் இளங்கோவடிகளால் எழுதப்பட்ட ஐம்பெரும் காப்பியங்களில் ஒன்று.",
    ]
    
    metadata = [
        {"source": "இலக்கியம்", "period": "சங்க காலம்"},
        {"source": "வரலாறு", "period": "சங்க காலம்"},
        {"source": "இலக்கியம்", "period": "சங்க காலம்"},
        {"source": "இலக்கியம்", "period": "இடைக்காலம்"},
        {"source": "இலக்கியம்", "period": "சங்க காலம்"},
    ]
    
    # Index documents
    print("\n📚 Indexing documents...")
    stats = rag.index_documents(documents, metadata)
    print(f"Indexed: {stats['indexed']} documents")
    
    # Query
    query = "சங்க காலத்தில் தமிழகத்தை யார் ஆண்டனர்?"
    print(f"\n🔍 Query: {query}")
    
    # Search only
    print("\n🎯 Search Results:")
    results = rag.search(query, top_k=10, rerank_top=3)
    for result in results:
        print(f"\nRank {result.rank} | Score: {result.score:.4f}")
        print(f"Document: {result.document}")
        print(f"Metadata: {result.metadata}")
    
    # Full RAG
    print("\n💬 Complete RAG Response:")
    response = rag.rag_query(query, top_k=10, rerank_top=3)
    print(f"Answer: {response['answer'][:200]}...")
    
    # Stats
    print("\n📊 Collection Statistics:")
    stats = rag.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_with_filtering():
    """Example: Search with metadata filtering"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Search with Filtering")
    print("="*70)
    
    rag = TamilRAGPipeline()
    
    # Query with metadata filter
    query = "இலக்கியம் பற்றி சொல்"
    filter_metadata = {"source": "இலக்கியம்"}
    
    print(f"\n🔍 Query: {query}")
    print(f"Filter: {filter_metadata}")
    
    results = rag.search(
        query=query,
        top_k=10,
        rerank_top=3,
        filter_metadata=filter_metadata
    )
    
    print(f"\nFound {len(results)} results")


def example_batch_indexing():
    """Example: Batch indexing for large datasets"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Indexing")
    print("="*70)
    
    rag = TamilRAGPipeline()
    
    # Simulate large dataset
    large_documents = [
        f"தமிழ் ஆவணம் {i}: இது தமிழ் மொழியில் உள்ள உள்ளடக்கம்."
        for i in range(100)
    ]
    
    print(f"\n📚 Indexing {len(large_documents)} documents in batches...")
    stats = rag.index_documents(
        documents=large_documents,
        batch_size=32,
        show_progress=True
    )
    
    print(f"\n✓ Successfully indexed {stats['indexed']} documents")
    print(f"Collection stats: {rag.get_collection_stats()}")


# ============================================================================
# INTEGRATION EXAMPLES FOR DIFFERENT LLMS
# ============================================================================

def integrate_tamil_llama():
    """Example: Integration with Tamil-Llama"""
    code = '''
# Install: pip install transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Tamil-Llama model
model_id = "abhinand/tamil-llama-7b-instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,  # Use FP16 for speed
    load_in_4bit=True  # Optional: 4-bit quantization
)

def generate_with_tamil_llama(prompt: str, max_tokens: int = 512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Modify TamilRAGPipeline.generate_answer() to use this function
'''
    print("\n" + "="*70)
    print("INTEGRATION: Tamil-Llama")
    print("="*70)
    print(code)


def integrate_claude_api():
    """Example: Integration with Claude API"""
    code = '''
# Install: pip install anthropic
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

def generate_with_claude(prompt: str, max_tokens: int = 1024):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

# Claude has excellent Tamil support and understanding
# Best quality but requires API calls
'''
    print("\n" + "="*70)
    print("INTEGRATION: Claude API (Best Quality)")
    print("="*70)
    print(code)


if __name__ == "__main__":
    print("\n🚀 Tamil RAG Pipeline - Production Examples\n")
    
    # Run examples
    example_basic_usage()
    example_with_filtering()
    example_batch_indexing()
    
    # Show integration examples
    integrate_tamil_llama()
    integrate_claude_api()
    
    print("\n" + "="*70)
    print("✅ Examples completed successfully!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Install dependencies: pip install sentence-transformers chromadb")
    print("2. Choose your LLM (Tamil-Llama for local, Claude for quality)")
    print("3. Index your OCR data")
    print("4. Integrate with Streamlit UI")
    print("="*70)
