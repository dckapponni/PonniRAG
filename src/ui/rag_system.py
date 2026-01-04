import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

class RAGSystem:
    def __init__(self):
        """Initialize RAG system with embedding model and Qdrant client."""
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Qdrant client
        self.client = QdrantClient(path=QDRANT_PATH)
        
        print(f"✓ Qdrant client initialized")
        
    def _clean_chunk_boundaries(self, chunk: str) -> str:
        """
        Clean chunk to avoid incomplete words at start and end.
        Removes partial words from boundaries.
        """
        # Remove leading partial word (if starts mid-word)
        chunk = re.sub(r'^\S*\s+', '', chunk)
        
        # Remove trailing partial word (if ends mid-word)
        chunk = re.sub(r'\s+\S*$', '', chunk)
        
        return chunk.strip()
    
    def _merge_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Merge multiple chunks intelligently to create coherent content.
        Handles overlapping and sequential chunks.
        """
        if not chunks:
            return ""
        
        # Sort chunks by chunk_id to maintain order
        sorted_chunks = sorted(chunks, key=lambda x: x['chunk_id'])
        
        merged_text = []
        for chunk in sorted_chunks:
            content = chunk['content']
            # Clean boundaries for each chunk
            cleaned = self._clean_chunk_boundaries(content)
            if cleaned:
                merged_text.append(cleaned)
        
        # Join with space and clean up multiple spaces
        result = ' '.join(merged_text)
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Search for relevant documents and generate response.
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve (default: 10)
            
        Returns:
            Dictionary containing generated text and sources
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [f"query: {query}"]
            )[0].tolist()
            
            print(f"Query embedding generated, dimension: {len(query_embedding)}")
            
            # Use the correct Qdrant client method - try different API versions
            try:
                # Try modern API (qdrant-client >= 1.7.0)
                search_results = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_embedding,
                    limit=top_k
                ).points
            except AttributeError:
                try:
                    # Try older search API
                    search_results = self.client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_embedding,
                        limit=top_k
                    )
                except AttributeError:
                    # Fallback to scroll with filter (oldest API)
                    from qdrant_client.models import PointIdsList, ScrollRequest
                    scroll_result = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False
                    )
                    search_results = scroll_result[0] if scroll_result else []
            
            print(f"✓ Search completed, found {len(search_results)} results")
            
            if not search_results:
                return {
                    "text": "மன்னிக்கவும், உங்கள் கேள்விக்கு பொருந்தக்கூடிய தகவல் கிடைக்கவில்லை. (Sorry, no relevant information found for your query.)",
                    "sources": []
                }
            
            # Group chunks by document
            doc_groups = {}
            for hit in search_results:
                payload = hit.payload
                # Default score if not available
                score = getattr(hit, 'score', 1.0)
                
                doc_id = payload['metadata'].get('doc_id', 'unknown')
                
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = {
                        'chunks': [],
                        'metadata': payload['metadata'],
                        'type': payload['type'],
                        'max_score': score
                    }
                
                doc_groups[doc_id]['chunks'].append({
                    'content': payload['content'],
                    'chunk_id': payload['chunk_id'],
                    'score': score
                })
            
            # Generate response and sources
            sources = []
            all_content = []
            
            for doc_id, doc_data in doc_groups.items():
                # Merge all chunks for this document
                merged_content = self._merge_chunks(doc_data['chunks'])
                
                if merged_content:
                    all_content.append(merged_content)
                
                # Create source metadata
                metadata = doc_data['metadata']
                source = {
                    'doc_id': doc_id,
                    'type': doc_data['type'],
                    'score': round(doc_data['max_score'] * 100, 2),
                    'volume': metadata.get('volume', 'unknown'),
                    'content_preview': merged_content[:200] + '...' if len(merged_content) > 200 else merged_content
                }
                
                # Add type-specific metadata
                if doc_data['type'] == 'article':
                    source['article_no'] = metadata.get('article_no', '')
                    source['article_heading'] = metadata.get('article_heading', '')
                    source['author'] = metadata.get('article_author_name', '')
                else:  # intro
                    source['heading'] = metadata.get('heading', '')
                
                sources.append(source)
            
            # Sort sources by score (highest first)
            sources.sort(key=lambda x: x['score'], reverse=True)
            
            # Generate final response text
            if all_content:
                response_text = self._generate_response(query, all_content)
            else:
                response_text = "தகவல் கிடைத்தது ஆனால் செயலாக்க முடியவில்லை. (Information found but could not be processed.)"
            
            return {
                "text": response_text,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Search error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": f"பிழை ஏற்பட்டது: {str(e)}",
                "sources": []
            }
    
    def _generate_response(self, query: str, contents: List[str]) -> str:
    """
    Generate final answer using Tamil LLaMA + retrieved context
    """

    context = "\n\n".join(contents)

    prompt = f"""
### வழிமுறை (Instruction):
நீங்கள் ஒரு தமிழ் அறிவார்ந்த உதவியாளர்.
கீழே கொடுக்கப்பட்டுள்ள தகவல்களை மட்டுமே பயன்படுத்தி
பயனரின் கேள்விக்கு தெளிவாகவும் சுருக்கமாகவும் பதிலளிக்கவும்.

### தகவல் (Context):
{context}

### கேள்வி (Question):
{query}

### பதில் (Answer in Tamil):
"""

    inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(self.llm.device)

    with torch.no_grad():
        output = self.llm.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    response = self.tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    # Extract only the answer part
    if "### பதில்" in response:
        response = response.split("### பதில்")[-1]

    return response.strip()

    def search_by_filters(
        self, 
        query: str = None,
        volume: str = None,
        doc_type: str = None,
        author: str = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Search with additional filters.
        
        Args:
            query: Optional search query
            volume: Filter by volume (e.g., 'vol_1')
            doc_type: Filter by type ('article' or 'intro')
            author: Filter by author name
            top_k: Number of results
            
        Returns:
            Dictionary with results and sources
        """
        try:
            # Build filter conditions
            must_conditions = []
            
            if volume:
                must_conditions.append(
                    FieldCondition(
                        key="metadata.volume",
                        match=MatchValue(value=volume)
                    )
                )
            
            if doc_type:
                must_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=doc_type)
                    )
                )
            
            if author:
                must_conditions.append(
                    FieldCondition(
                        key="metadata.article_author_name",
                        match=MatchValue(value=author)
                    )
                )
            
            # Create filter object
            query_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Generate query embedding if query provided
            if query:
                query_embedding = self.embedding_model.encode(
                    [f"query: {query}"]
                )[0].tolist()
            else:
                # If no query, use zero vector for filtering only
                dimension = 1024  # e5-large dimension
                query_embedding = [0.0] * dimension
            
            # Perform search with filters - try different API versions
            try:
                search_results = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=top_k
                ).points
            except AttributeError:
                try:
                    search_results = self.client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_embedding,
                        query_filter=query_filter,
                        limit=top_k
                    )
                except AttributeError:
                    # Fallback to scroll with filter
                    scroll_result = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=query_filter,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False
                    )
                    search_results = scroll_result[0] if scroll_result else []
            
            # Process results same as regular search
            return self._process_results(search_results, query or "filtered search")
            
        except Exception as e:
            print(f"Filter search error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": f"வடிகட்டல் பிழை: {str(e)}",
                "sources": []
            }
    
    def _process_results(self, search_results, query: str) -> Dict[str, Any]:
        """Helper method to process search results."""
        if not search_results:
            return {
                "text": "மன்னிக்கவும், உங்கள் கேள்விக்கு பொருந்தக்கூடிய தகவல் கிடைக்கவில்லை.",
                "sources": []
            }
        
        # Group chunks by document
        doc_groups = {}
        for hit in search_results:
            payload = hit.payload
            score = getattr(hit, 'score', 1.0)
            doc_id = payload['metadata'].get('doc_id', 'unknown')
            
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    'chunks': [],
                    'metadata': payload['metadata'],
                    'type': payload['type'],
                    'max_score': score
                }
            
            doc_groups[doc_id]['chunks'].append({
                'content': payload['content'],
                'chunk_id': payload['chunk_id'],
                'score': score
            })
        
        # Generate response and sources
        sources = []
        all_content = []
        
        for doc_id, doc_data in doc_groups.items():
            merged_content = self._merge_chunks(doc_data['chunks'])
            
            if merged_content:
                all_content.append(merged_content)
            
            metadata = doc_data['metadata']
            source = {
                'doc_id': doc_id,
                'type': doc_data['type'],
                'score': round(doc_data['max_score'] * 100, 2),
                'volume': metadata.get('volume', 'unknown'),
                'content_preview': merged_content[:200] + '...' if len(merged_content) > 200 else merged_content
            }
            
            if doc_data['type'] == 'article':
                source['article_no'] = metadata.get('article_no', '')
                source['article_heading'] = metadata.get('article_heading', '')
                source['author'] = metadata.get('article_author_name', '')
            else:
                source['heading'] = metadata.get('heading', '')
            
            sources.append(source)
        
        sources.sort(key=lambda x: x['score'], reverse=True)
        
        if all_content:
            response_text = self._generate_response(query, all_content)
        else:
            response_text = "தகவல் கிடைத்தது ஆனால் செயலாக்க முடியவில்லை."
        
        return {
            "text": response_text,
            "sources": sources
        }
    
    def check_collection(self):
        """Check if collection exists and is accessible."""
        try:
            collection_info = self.client.get_collection(collection_name=COLLECTION_NAME)
            print(f"✓ Collection '{COLLECTION_NAME}' found")
            print(f"  - Points count: {collection_info.points_count}")
            return True
        except Exception as e:
            print(f"✗ Collection check error: {e}")
            import traceback
            traceback.print_exc()
            return False


# Convenience function for quick searches
def search_ponni(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Quick search function.
    
    Args:
        query: Search query
        top_k: Number of results (default: 10)
        
    Returns:
        Dictionary with text and sources
    """
    rag = RAGSystem()
    return rag.search(query, top_k)


# Debug function
def debug_qdrant_connection():
    """Debug function to check Qdrant connection."""
    try:
        print("\n=== Qdrant Client Debug ===")
        rag = RAGSystem()
        
        print("\n=== Collection Check ===")
        rag.check_collection()
        
        print("\n=== Test Search ===")
        result = rag.search("test query", top_k=10)
        print(f"Search returned {len(result.get('sources', []))} sources")
        
        return rag
    except Exception as e:
        print(f"✗ Debug error: {e}")
        import tracebackk
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run debug when executed directly
    print("Running Qdrant connection debug...")
    debug_qdrant_connection()