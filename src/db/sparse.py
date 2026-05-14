"""Sparse BM25 embeddings via fastembed (Qdrant/bm25).

Replaces the legacy tf-hash sparse encoder that suffered from index/query
tokenizer mismatch. Uses fastembed's ``Qdrant/bm25`` model for both
indexing (``sparse_embed_doc``) and querying (``sparse_embed_query``).

Index-time embeddings carry BM25 term weights; query-time embeddings
carry unit weights. Server-side IDF is applied by Qdrant when the
sparse vector is configured with ``Modifier.IDF`` (see indexer).

The model is loaded lazily so importing this module does not pay the
fastembed startup cost.
"""

import logging
import threading
import unicodedata

from qdrant_client import models

logger = logging.getLogger(__name__)

BM25_MODEL_NAME = "Qdrant/bm25"

_lock = threading.Lock()
_model = None


def get_bm25_model():
    """Lazily load and cache the fastembed BM25 sparse encoder (singleton).

    Uses a double-checked locking pattern with ``_lock`` to ensure the
    ``SparseTextEmbedding`` model is initialized at most once in
    multi-threaded environments. Importing this module does not trigger a
    load; the first call to :func:`sparse_embed_doc` or
    :func:`sparse_embed_query` pays the fastembed startup cost.

    Returns:
        SparseTextEmbedding: Cached fastembed ``Qdrant/bm25`` model instance.
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from fastembed import SparseTextEmbedding

        logger.info(f"Loading sparse BM25 model: {BM25_MODEL_NAME}")
        _model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
        logger.info("Sparse BM25 model loaded")
    return _model


def _to_sparse_vector(emb) -> models.SparseVector:
    """Convert a fastembed SparseEmbedding object to a Qdrant SparseVector.

    fastembed exposes ``indices`` and ``values`` as NumPy arrays; Qdrant
    requires plain Python lists. Converts both attributes via ``.tolist()``
    when the method is available.

    If the resulting ``indices`` list is empty, returns a no-op placeholder
    ``SparseVector(indices=[0], values=[0.0])`` because Qdrant rejects
    empty sparse vectors. The placeholder contributes nothing to scoring.

    Args:
        emb: A fastembed ``SparseEmbedding`` object exposing ``indices``
            and ``values`` attributes (NumPy arrays or plain lists).

    Returns:
        models.SparseVector: Qdrant-compatible sparse vector with Python
        list indices and values, or a zero-weight placeholder if the
        embedding is empty.
    """
    indices = emb.indices
    values = emb.values
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not indices:
        # Qdrant rejects empty sparse vectors — emit a no-op placeholder
        # that contributes nothing to scoring.
        return models.SparseVector(indices=[0], values=[0.0])
    return models.SparseVector(indices=list(indices), values=list(values))


def sparse_embed_doc(text: str) -> models.SparseVector:
    """Generate a BM25 sparse embedding for a document at index time.

    Applies Unicode NFC normalization before encoding. Uses
    ``model.embed`` which applies document-side BM25 term weighting.
    Server-side IDF is applied by Qdrant when the sparse vector field is
    configured with ``Modifier.IDF``.

    Args:
        text (str): Raw document text to encode.

    Returns:
        models.SparseVector: BM25 sparse vector with document-side term
        weights, ready for upsert into the Qdrant collection.
    """
    text = unicodedata.normalize("NFC", text)
    model = get_bm25_model()
    emb = next(iter(model.embed([text])))
    return _to_sparse_vector(emb)


def sparse_embed_query(text: str) -> models.SparseVector:
    """Generate a BM25 sparse embedding for a search query at retrieval time.

    Applies Unicode NFC normalization before encoding. Uses
    ``model.query_embed`` which applies query-side BM25 unit-tf weighting,
    matching the weighting scheme Qdrant expects when the sparse vector
    field is configured with ``Modifier.IDF``. Using the same
    ``Qdrant/bm25`` model for both index time (:func:`sparse_embed_doc`)
    and query time ensures token vocabularies are identical.

    Args:
        text (str): Raw user query string to encode.

    Returns:
        models.SparseVector: BM25 sparse vector with query-side unit
        weights, ready for use in a Qdrant hybrid search request.
    """
    text = unicodedata.normalize("NFC", text)
    model = get_bm25_model()
    # query_embed uses BM25 query-side weighting (unit-tf), matching
    # what Qdrant expects when the sparse vector has Modifier.IDF.
    emb = next(iter(model.query_embed([text])))
    return _to_sparse_vector(emb)
