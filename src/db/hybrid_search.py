import logging
from typing import Any, Dict, List, Optional, Union

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridQdrantSearch:
    """
    A module to perform Hybrid Search in Qdrant using RRF (Reciprocal Rank Fusion).
    Combines Dense Vector Search and Sparse Keyword Search.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_embedding_func: callable,
        sparse_embedding_func: callable,
        sparse_vector_name: str = "sparse",
        dense_vector_name: str = "dense",
    ):
        """
        Initialize the Hybrid Searcher.

        Args:
            client: An instance of QdrantClient.
            collection_name: Name of the collection to search.
            dense_embedding_func: Function that takes text -> list of floats (dense vector).
            sparse_embedding_func: Function that takes text -> two lists (indices, values) or dictionary of {index: value}.
                                   Should format correctly for Qdrant sparse vectors.
            sparse_vector_name: Name of the sparse vector in your Qdrant config (default: "sparse").
            dense_vector_name: Name of the dense vector in your Qdrant config (default: "dense").
        """
        self.client = client
        self.collection_name = collection_name
        self.dense_func = dense_embedding_func
        self.sparse_func = sparse_embedding_func
        self.sparse_name = sparse_vector_name
        self.dense_name = dense_vector_name

        if not self.client or not models:
            logger.warning("QdrantClient not available. Hybrid search will not work.")

    def build_filter(self, user_filters: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Converts a simple dictionary of user choices into a Qdrant Filter.

        Args:
            user_filters: Dict where keys are field names and values are exact match values.
                          Example: {"category": "history", "year": 2023}

        Returns:
            models.Filter or None
        """
        if not user_filters:
            return None

        must_conditions = []
        for key, value in user_filters.items():
            # Validate value types: Qdrant MatchValue supports int, str, bool
            if value is None:
                continue
            
            if not isinstance(value, (str, int, float, bool)):
                logger.warning(f"Filter value for '{key}' is {type(value)}, which may not be supported directly. Skipping.")
                continue

            # Simple Exact Match logic
            must_conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def search(
        self,
        query_text: str,
        user_filters: Dict[str, Any] = None,
        limit: int = 10,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Performs the hybrid search using RRF Fusion.

        Args:
            query_text: The user's search query.
            user_filters: Dictionary of filters (e.g. {"category": "sports"}).
            limit: Number of results to return.
            rrf_k: The 'k' constant in RRF formula (1 / (k + rank)). Default 60.

        Returns:
            List of payloads with 'score' and 'fusion_score'.
        """
        if not self.client:
            logger.error("QdrantClient is not initialized.")
            return []

        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            logger.warning("Empty or invalid query text provided. Returning empty results.")
            return []
        
        # Sanitize limit
        if limit <= 0:
            logger.warning(f"Invalid limit {limit}. Setting to default 10.")
            limit = 10

        # 1. Generate Embeddings
        try:
            dense_vector = self.dense_func(query_text)
            sparse_vector = self.sparse_func(query_text)
            
            if dense_vector is None or sparse_vector is None:
                 logger.error("Embedding function returned None.")
                 return []

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

        # 2. Build Filter
        qdrant_filter = self.build_filter(user_filters)

        # 3. Define Prefetch Operations (The two sub-searches)
        # We fetch more candidates than 'limit' for better re-ranking fusion (e.g., limit * 2)
        prefetch_limit = max(limit * 2, 20) # Ensure we fetch at least a reasonable amount

        try:
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using=self.dense_name,
                    filter=qdrant_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=self.sparse_name,
                    filter=qdrant_filter,
                    limit=prefetch_limit,
                ),
            ]

            # 4. Execute Query with RRF Fusion
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.Fusion(fusion=models.FusionType.RRF),
                limit=limit,
                with_payload=True,
            )

            # 5. Format Results
            results = []
            if not response or not response.points:
                return []

            for point in response.points:
                item = point.payload or {}
                # RRF score is small (0.0-1.0 roughly), we keep it as is
                item["score"] = point.score
                results.append(item)

            return results

        except Exception as e:
            logger.error(f"Qdrant Search Error: {e}")
            return []

# Example Usage helper (commented out/docstring)
"""
# Assuming you have a client and models set up:

def my_dense_embed(text):
    return model.encode(text).tolist()

def my_sparse_embed(text):
    # Use SPLADE or BM25 encoder here to return qdrant sparse format
    # models.SparseVector(indices=[1, 10], values=[0.5, 0.8])
    return ...

searcher = HybridQdrantSearch(client, "my_collection", my_dense_embed, my_sparse_embed)
results = searcher.search("Tamil History", user_filters={"type": "article"})
"""
