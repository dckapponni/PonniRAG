import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantHybridIndexer:
    """
    A module to manage Indexing (Upsert, Update, Delete) in Qdrant for Hybrid Search.
    Compatible with the HybridQdrantSearch module.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_embedding_func: Callable[[str], List[float]],
        sparse_embedding_func: Callable[[str], Any],
        sparse_vector_name: str = "sparse",
        dense_vector_name: str = "dense",
    ):
        """
        Initialize the Indexer.

        Args:
            client: QdrantClient instance.
            collection_name: Target collection.
            dense_embedding_func: Function (text) -> dense_vector (List[float]).
            sparse_embedding_func: Function (text) -> sparse_vector (models.SparseVector or dict).
            sparse_vector_name: Name of the sparse vector in config.
            dense_vector_name: Name of the dense vector in config.
        """
        self.client = client
        self.collection_name = collection_name
        self.dense_func = dense_embedding_func
        self.sparse_func = sparse_embedding_func
        self.sparse_name = sparse_vector_name
        self.dense_name = dense_vector_name

        if not self.client or not models:
            logger.warning("QdrantClient not available. Indexing will not work.")

    def create_collection(self, dense_vector_size: int = 768, force: bool = False):
        """
        Creates the collection with the required Hybrid (Dense + Sparse) configuration.

        Args:
            dense_vector_size: Dimension of the dense vector (e.g., 768 for BERT/E5).
            force: If True, deletes the existing collection before creating.
        """
        if force:
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_name: models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.sparse_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def _generate_uuid(self, text: str) -> str:
        """Generates a deterministic UUID based on content to avoid duplicates."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
        update_only_new: bool = False,
    ):
        """
        Indexes documents into Qdrant.

        Args:
            documents: List of dicts, each containing:
                       - 'text' (str): The content to embed.
                       - 'metadata' (dict): Payload fields.
                       - 'id' (optional): ID for the point. If missing, generated from text.
            batch_size: Number of points to upload in one request.
            update_only_new: If True, checks if ID exists and skips it.
                             Requires 'id' to be stable or provided.
        """
        if not documents:
            logger.warning("Received empty document list. Nothing to index.")
            return

        # Check if collection exists to avoid silent failures in batch processing
        if not self.client.collection_exists(self.collection_name):
             logger.error(f"Collection '{self.collection_name}' does not exist. Call create_collection() first.")
             return

        points = []
        
        # Pre-process batch
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if not text or not isinstance(text, str):
                logger.warning(f"Document at index {i} has missing or invalid 'text'. Skipping.")
                continue

            # Determine ID
            doc_id = doc.get("id") or self._generate_uuid(text)

            # Check existence if requested
            if update_only_new:
                # We do a quick check. For bulk efficiency, Scroll or Retrieve is better,
                # but valid for incremental updates.
                try:
                    existing = self.client.retrieve(
                        collection_name=self.collection_name, 
                        ids=[doc_id],
                        with_payload=False,
                        with_vectors=False
                    )
                    if existing:
                        logger.debug(f"Skipping existing document ID: {doc_id}")
                        continue
                except Exception as e:
                    logger.error(f"Error checking existence for ID {doc_id}: {e}")
                    # Decide whether to proceed or skip. Safe default: proceed to upsert (which overwrites).
                    pass

            try:
                # Generate Embeddings
                dense_vector = self.dense_func(text)
                sparse_vector = self.sparse_func(text)
                
                if dense_vector is None or sparse_vector is None:
                    logger.error(f"Embedding function returned None for doc {doc_id}. Skipping.")
                    continue

                # Prepare Point
                metadata = doc.get("metadata", {})
                if not isinstance(metadata, dict):
                     metadata = {}

                point = models.PointStruct(
                    id=doc_id,
                    vector={
                        self.dense_name: dense_vector,
                        self.sparse_name: sparse_vector,
                    },
                    payload={
                        "text": text,
                        **metadata
                    },
                )
                points.append(point)

            except Exception as e:
                logger.error(f"Failed to embed document '{doc_id}': {e}")
                continue

        # Upload in batches
        if points:
            total_uploaded = 0
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                    )
                    total_uploaded += len(batch)
                    logger.info(f"Indexed batch of {len(batch)} documents. Total: {total_uploaded}")
                except Exception as e:
                     logger.error(f"Failed to upload batch starting at index {i}: {e}")
        else:
            logger.info("No valid new documents to index after processing.")

    def delete_documents(self, doc_ids: List[Union[str, int]]):
        """
        Deletes documents by their IDs.
        """
        if not doc_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=doc_ids),
        )
        logger.info(f"Deleted {len(doc_ids)} documents.")

    def delete_by_filter(self, filter_conditions: Dict[str, Any]):
        """
        Deletes documents that match specific metadata conditions.
        
        Args:
            filter_conditions: Dict of {field_name: value}. 
                               Example: {"author": "Thiruvalluvar"}
        """
        if not filter_conditions:
            return

        must_conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filter_conditions.items()
        ]
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=must_conditions)
            ),
        )
        logger.info(f"Deleted documents matching filter: {filter_conditions}")

