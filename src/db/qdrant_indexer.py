"""Qdrant vector indexer for PonniRAG Tamil documents.

Loads articles and authors from S3, generates dense and sparse
embeddings, and upserts them into a Qdrant collection with
snapshot-based change detection.
"""

import argparse
import json
import logging
import os
import re
import unicodedata
import uuid
from typing import Dict, List

import boto3
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config.config import (
    BATCH_SIZE,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    S3_BUCKET,
    S3_PREFIX,
    S3_SUFFIX,
)
from src.db.article_tagger import ArticleTagger
from src.db.retry import with_qdrant_retry
from src.db.snapshot_manager import (
    build_index_metadata,
    compute_source_data_hash,
    needs_reindex,
    restore_snapshot_from_s3,
    save_index_metadata,
    save_snapshot_to_s3,
)

COLLECTION_NAME = "qdrant_indexer"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # Already configured by another module

logger.info(f"Using device: {DEVICE}")

s3 = boto3.client("s3")

dense_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def dense_embed_doc(text: str) -> List[float]:
    """Generate a dense embedding vector for a document passage at index time.

    Applies Unicode NFC normalization, prepends the ``"passage: "`` prefix
    required by the E5 multilingual model, and encodes the text using the
    module-level ``dense_model`` with L2 normalization.

    Args:
        text (str): Raw document passage text to embed.

    Returns:
        List[float]: Normalized dense embedding vector of length
        ``EMBEDDING_DIM``.
    """
    text = unicodedata.normalize("NFC", text)
    return dense_model.encode(f"passage: {text}", normalize_embeddings=True).tolist()


def dense_embed_query(text: str) -> List[float]:
    """Generate a dense embedding vector for a search query at retrieval time.

    Applies Unicode NFC normalization, prepends the ``"query: "`` prefix
    required by the E5 multilingual model, and encodes the text using the
    module-level ``dense_model`` with L2 normalization. The asymmetric
    ``query:`` / ``passage:`` prefix pairing ensures cosine similarity scores
    are comparable across the index.

    Args:
        text (str): Raw user query string to embed.

    Returns:
        List[float]: Normalized dense embedding vector of length
        ``EMBEDDING_DIM``.
    """
    text = unicodedata.normalize("NFC", text)
    return dense_model.encode(f"query: {text}", normalize_embeddings=True).tolist()


def sparse_embed(text: str) -> models.SparseVector:
    """Generate a BM25 sparse embedding for a text string via fastembed.

    Delegates to :func:`src.db.sparse.sparse_embed_doc`, which uses the
    ``Qdrant/bm25`` fastembed model. Using the same model at both index time
    (this function) and query time (``sparse.sparse_embed_query``) ensures
    that token vocabularies are identical, so BM25 term-match scores are
    meaningful.

    Args:
        text (str): Text to encode into a sparse BM25 vector.

    Returns:
        models.SparseVector: Qdrant sparse vector containing non-zero term
        indices and their corresponding BM25 weights.
    """
    from src.db.sparse import sparse_embed_doc

    return sparse_embed_doc(text)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving Tamil and English punctuation.

    Splits on Tamil ``।`` and standard English punctuation (``.``, ``!``,
    ``?``), then re-attaches the punctuation mark to the preceding sentence
    fragment so that sentence boundaries are preserved. Sentence fragments
    shorter than or equal to 10 characters are discarded.

    Args:
        text (str): Raw article or passage text in Tamil and/or English.

    Returns:
        List[str]: List of sentence strings, each ending with its original
        punctuation mark and containing more than 10 characters.
    """
    sentences = re.split(r"([।.!?]+)", text)

    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()

        if not sentence:
            i += 1
            continue

        if i + 1 < len(sentences) and re.match(r"^[।.!?]+$", sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1

        if len(sentence) > 10:
            result.append(sentence.strip())

    return result


def chunk_text(text: str, size: int) -> List[str]:
    """Split text into sentence-aligned chunks of approximately ``size`` characters.

    Converts ``size`` (a character budget) to a target word count
    (``size // 6``) and a hard ceiling (``target_words * 1.5``). Sentences
    are accumulated greedily until the target word count is reached, at which
    point the current chunk is flushed. Individual sentences that exceed the
    ceiling are emitted as standalone chunks. The final partial chunk is kept
    only if it contains at least 10 words.

    Args:
        text (str): Raw article or passage text to chunk.
        size (int): Approximate character budget per chunk. Controls the
            target and maximum word counts via the ``size // 6`` heuristic.

    Returns:
        List[str]: List of text chunk strings, each respecting sentence
        boundaries. Returns an empty list if ``text`` is empty or whitespace.
    """
    if not text or not text.strip():
        return []

    target_words = size // 6
    max_words = int(target_words * 1.5)

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", sentence))

        if words > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue

        if current_word_count + words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += words

        if current_word_count >= target_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

    if current_chunk:
        last_chunk = " ".join(current_chunk)
        words_in_chunk = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", last_chunk))
        if words_in_chunk >= 10:
            chunks.append(last_chunk)

    return chunks


def validate_chunk(chunk: str) -> bool:
    r"""

    Validate a text chunk for minimum quality before indexing.

    Applies three progressive quality gates:

    1. **Length gate** — the stripped chunk must be at least 20 characters.
    2. **Word count gate** — the chunk must contain at least 10 words
       (Tamil Unicode tokens or ASCII ``\\w+`` tokens).
    3. **Tamil content gate** — the chunk must contain at least 20 Tamil
       Unicode characters (``U+0B80``–``U+0BFF``), ensuring non-Tamil noise
       fragments are excluded.

    Args:
        chunk (str): Text chunk string to evaluate.

    Returns:
        bool: ``True`` if the chunk passes all three quality gates;
        ``False`` otherwise.
    """
    if not chunk or len(chunk.strip()) < 20:
        return False

    # if not re.match(r"^[a-zA-Zஅ-ஹ]", chunk.strip()):
    #     return False

    words = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", chunk))
    if words < 10:
        return False

    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", chunk))
    if tamil_chars < 20:
        return False

    return True


def list_s3_json_files(bucket: str, prefix: str, suffix: str) -> List[str]:
    """List all S3 object keys matching a given prefix and suffix.

    Uses the S3 list paginator to enumerate all objects under ``prefix`` in
    ``bucket`` and returns only those keys whose name ends with ``suffix``.

    Args:
        bucket (str): S3 bucket name to scan.
        prefix (str): Key prefix to filter the listing (e.g. ``"data/ponni/"``).
        suffix (str): File suffix that qualifying keys must end with
            (e.g. ``".json"``).

    Returns:
        List[str]: Sorted list of matching S3 object key strings. Returns an
        empty list if no keys match.
    """
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                keys.append(key)
                logger.debug(f"Found: {key}")
    return keys


def extract_volume_from_s3_key(s3_key: str) -> str:
    """Extract the volume directory identifier from an S3 object key path.

    Splits the key on ``"/"`` and returns the first path component whose
    lowercased value starts with ``"vol_"``.

    Args:
        s3_key (str): Full S3 object key (e.g.
            ``"ponni/vol_3/issue_5/articles.json"``).

    Returns:
        str: Volume identifier string (e.g. ``"vol_3"``), or ``"unknown"``
        if no path component matching the ``vol_*`` pattern is found.
    """
    for part in s3_key.split("/"):
        if part.lower().startswith("vol_"):
            return part
    return "unknown"


def is_author_file(s3_key: str) -> bool:
    """Return True if the S3 key points to an authors metadata file.

    Args:
        s3_key (str): Full S3 object key string.

    Returns:
        bool: ``True`` if the lowercased key ends with ``"authors.json"``;
        ``False`` otherwise.
    """
    return s3_key.lower().endswith("authors.json")


def is_remaining_file(s3_key: str) -> bool:
    """Return True if the S3 key points to a remaining-context supplement file.

    Remaining files contain supplementary article text loaded at query time
    by :func:`llm.load_remaining_context_s3` and should be skipped during
    indexing to avoid duplicate content in the vector store.

    Args:
        s3_key (str): Full S3 object key string.

    Returns:
        bool: ``True`` if the lowercased key contains ``"remaining_vol"``;
        ``False`` otherwise.
    """
    return "remaining_vol" in s3_key.lower()


def load_documents_from_s3() -> List[Dict]:
    """Load, tag, chunk, and prepare article documents from S3 for indexing.

    Executes a two-pass pipeline:

    **Pass 1 — Corpus collection**: Iterates all JSON files under
    ``S3_PREFIX`` in ``S3_BUCKET`` (excluding author files and remaining
    files), parses each ``articles`` array, and accumulates all non-empty
    articles together with their originating S3 key and volume identifier.

    **Pass 2 — Tagging, chunking, and document preparation**: Trains an
    :class:`ArticleTagger` on the full article corpus, then for each article:

    - Assigns content tags and Tamil-script tag labels via the tagger.
    - Splits the article content into chunks using :func:`chunk_text` with
      ``CHUNK_SIZE``.
    - Validates each chunk with :func:`validate_chunk`; invalid chunks are
      skipped and counted.
    - Builds a document dict with a deterministic UUID5 ID, the chunk text,
      and a full metadata payload including ``doc_id``, ``doc_issue``,
      ``article_no``, ``author_name``, ``title``, ``year``, ``chunk_id``,
      ``total_chunks``, ``volume``, ``tags``, and ``tags_tamil``.

    Chunk quality statistics (total, valid, invalid, min/max/avg word count)
    are logged at the end of the run.

    Returns:
        List[Dict]: List of document dicts, each containing:

        - ``"id"`` (str): Deterministic UUID5 string.
        - ``"text"`` (str): Validated chunk text.
        - ``"metadata"`` (dict): Full article and chunk metadata.

        Returns an empty list if no valid articles are found.
    """
    documents = []
    chunk_stats = {"total": 0, "valid": 0, "invalid": 0, "word_counts": []}

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Found {len(keys)} JSON files in S3")

    # --- Pass 1: Collect all articles for tagger training ---
    all_articles = []  # list of (article_dict, key, volume)
    for key in keys:
        if is_author_file(key):
            continue
        if is_remaining_file(key):
            logger.info(f"Skipping remaining file: {key}")
            continue
        logger.info(f"Loading: {key}")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))

            if not isinstance(data, dict) or "articles" not in data:
                logger.info(f"Skipping non-articles JSON: {key}")
                continue

            articles = data.get("articles", [])
            if not isinstance(articles, list):
                logger.warning(f"'articles' is not a list in {key}")
                continue

            volume = extract_volume_from_s3_key(key)

            for article in articles:
                content = article.get("content", "").strip()
                if not content:
                    continue
                all_articles.append((article, key, volume))

        except Exception as e:
            logger.error(f"Failed to load {key}: {e}")

    logger.info(f"Loaded {len(all_articles)} articles from S3")

    # --- Train article tagger on full corpus ---
    tagger = ArticleTagger()
    tagger.train_tfidf([a for a, _, _ in all_articles])
    logger.info("Article tagger trained")

    # --- Pass 2: Tag, chunk, and prepare documents ---
    for article, key, volume in all_articles:
        content = article.get("content", "").strip()

        # Tag at article level
        article_tags = tagger.tag_article(article)
        article_tags_tamil = tagger.get_tamil_tags(article_tags)

        chunks = chunk_text(content, CHUNK_SIZE)

        if "PONGAL" in key.upper():
            print("\n" + "=" * 80)
            print("TITLE:", article.get("title"))
            print("TOTAL CHUNKS:", len(chunks))

            for idx, chunk in enumerate(chunks):
                valid = validate_chunk(chunk)

                print(f"\nCHUNK {idx}")
                print("VALID:", valid)

                words = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", chunk))
                print("WORDS:", words)

                print("START:")
                print(repr(chunk[:120]))

        logger.info(
            f"  doc_id={article.get('doc_id')}, "
            f"article_no={article.get('article_no')}: "
            f"{len(chunks)} chunks, tags={article_tags}"
        )

        for idx, chunk in enumerate(chunks):
            chunk_stats["total"] += 1

            if not validate_chunk(chunk):
                chunk_stats["invalid"] += 1
                logger.debug(f"    Skipped invalid chunk {idx}")
                continue

            chunk_stats["valid"] += 1
            word_count = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", chunk))
            chunk_stats["word_counts"].append(word_count)

            doc_id = article.get("doc_id")
            art_no = article.get("article_no")
            uuid_name = f"{key}-{doc_id}-{art_no}-{idx}"
            documents.append(
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            uuid_name,
                        )
                    ),
                    "text": chunk,
                    "metadata": {
                        "doc_id": article.get("doc_id"),
                        "doc_issue": article.get("doc_issue"),
                        "article_no": article.get("article_no"),
                        "author_name": (
                            ", ".join(article.get("author_name", []))
                            if isinstance(article.get("author_name"), list)
                            else article.get("author_name", "")
                        ),
                        "title": article.get("title", ""),
                        "year": article.get("year", ""),
                        "source_document": article.get("source_document", ""),
                        "source": "s3",
                        "s3_key": key,
                        "chunk_id": idx,
                        "total_chunks": len(chunks),
                        "volume": volume,
                        "tags": article_tags,
                        "tags_tamil": article_tags_tamil,
                    },
                }
            )

    if chunk_stats["word_counts"]:
        logger.info(f"\n{'='*60}")
        logger.info("CHUNK STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total chunks: {chunk_stats['total']}")
        logger.info(f"Valid chunks: {chunk_stats['valid']}")
        logger.info(f"Invalid chunks: {chunk_stats['invalid']}")
        logger.info(
            f"Words -> min={min(chunk_stats['word_counts'])}, "
            f"max={max(chunk_stats['word_counts'])}, "
            f"avg={sum(chunk_stats['word_counts'])//len(chunk_stats['word_counts'])}"
        )

    logger.info(f"Prepared {len(documents)} valid chunks")
    return documents


def load_authors_from_s3() -> List[Dict]:
    """Load author metadata records from S3 ``authors.json`` files for indexing.

    Scans all JSON files under ``S3_PREFIX`` in ``S3_BUCKET``, processes
    only those identified as author files by :func:`is_author_file`, and
    parses each file as a list of issue-level author records. For each
    non-empty author name in each record, builds a document dict with a
    deterministic UUID5 ID derived from the S3 key, ``doc_id``,
    ``doc_issue``, and author name.

    Args:
        None

    Returns:
        List[Dict]: List of author document dicts, each containing:

        - ``"id"`` (str): Deterministic UUID5 string.
        - ``"text"`` (str): Author name string (used as the embedding input).
        - ``"metadata"`` (dict): Keys ``"author"``, ``"doc_id"``,
          ``"doc_issue"``, ``"volume"``, and ``"source"``
          (fixed value ``"authors_json"``).

        Returns an empty list if no author files are found or all fail to
        parse.
    """
    documents = []

    keys = list_s3_json_files(S3_BUCKET, S3_PREFIX, S3_SUFFIX)
    logger.info(f"Scanning {len(keys)} JSON files for authors.json")

    for key in keys:
        if not is_author_file(key):
            continue

        logger.info(f"Processing authors file: {key}")

        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))

            if not isinstance(data, list):
                continue

            volume = extract_volume_from_s3_key(key)

            for item in data:
                doc_id = item.get("doc_id")
                doc_issue = item.get("doc_issue")

                for author in item.get("authors", []):
                    author = author.strip()
                    if not author:
                        continue

                    documents.append(
                        {
                            "id": str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_DNS,
                                    f"{key}-{doc_id}-{doc_issue}-{author}",
                                )
                            ),
                            "text": author,
                            "metadata": {
                                "author": author,
                                "doc_id": doc_id,
                                "doc_issue": doc_issue,
                                "volume": volume,
                                "source": "authors_json",
                            },
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")

    logger.info(f"Prepared {len(documents)} authors")
    return documents


# Payload fields used in scroll/search filters. Without keyword indexes,
# every filtered scroll is a full collection scan (the dominant query latency
# on slower hosts). Indexing turns each into an O(log n) lookup.
_FILTER_INDEX_FIELDS = (
    "type",
    "metadata.doc_id",
    "metadata.doc_issue",
    "metadata.volume",
    "metadata.title",
    "metadata.tags",
)


def ensure_payload_indexes(client):
    """Create keyword payload indexes on all filtered fields, idempotently.

    Safe to call on an existing, populated collection — Qdrant builds the
    indexes in the background without re-indexing vectors. Re-creating an
    index that already exists is a no-op (errors are logged and ignored).

    Args:
        client (QdrantClient): Connected Qdrant client instance.

    Returns:
        None
    """
    for field in _FILTER_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Payload index ensured: {field}")
        except Exception as e:
            logger.warning(f"Payload index for {field} not created: {e}")


def _do_full_index(client):
    """Recreate the Qdrant collection and index all articles and authors.

    Deletes any existing collection named ``COLLECTION_NAME``, creates a
    fresh collection with one dense cosine-similarity vector space
    (``"dense"``, dimension ``EMBEDDING_DIM``) and one IDF-modified sparse
    vector space (``"sparse"``), then indexes documents in two phases:

    **Phase 1 — Articles**: loads and chunks all article documents via
    :func:`load_documents_from_s3`, generates dense and sparse embeddings
    for each chunk, and upserts them to Qdrant in batches of ``BATCH_SIZE``
    with payload type ``"article"``.

    **Phase 2 — Authors**: loads all author records via
    :func:`load_authors_from_s3`, embeds and upserts them in the same
    batched manner with payload type ``"author"``.

    Failed individual points are logged and skipped; the overall indexing
    continues. Collection statistics (total points, article chunks, author
    count) are logged on completion.

    Args:
        client (QdrantClient): Connected Qdrant client instance.

    Returns:
        None
    """
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    logger.info(f"Created collection: {COLLECTION_NAME}")

    ensure_payload_indexes(client)

    logger.info(f"\n{'-'*60}")
    logger.info("INDEXING ARTICLE DOCUMENTS")
    logger.info(f"{'-'*60}")

    documents = load_documents_from_s3()

    if not documents:
        logger.warning("No documents found")
        return

    points = []
    indexed = 0

    for doc in documents:
        try:
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_embed_doc(doc["text"]),
                    "sparse": sparse_embed(doc["text"]),
                },
                payload={
                    "content": doc["text"],
                    "chunk_id": doc["metadata"]["chunk_id"],
                    "type": "article",
                    "metadata": doc["metadata"],
                },
            )
            points.append(point)

            if len(points) >= BATCH_SIZE:
                with_qdrant_retry(
                    client.upsert,
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                indexed += len(points)
                logger.info(f"  Indexed {indexed}/{len(documents)}")
                points.clear()

        except Exception as e:
            logger.error(f"Failed to index: {e}")

    if points:
        with_qdrant_retry(
            client.upsert,
            collection_name=COLLECTION_NAME,
            points=points,
        )
        indexed += len(points)
        logger.info(f"  Indexed {indexed}/{len(documents)}")

    logger.info(f"\n{'-'*60}")
    logger.info("INDEXING AUTHORS")
    logger.info(f"{'-'*60}")

    author_documents = load_authors_from_s3()

    points = []
    indexed_authors = 0

    for doc in author_documents:
        try:
            point = models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": dense_embed_doc(doc["text"]),
                    "sparse": sparse_embed(doc["text"]),
                },
                payload={
                    "content": doc["text"],
                    "chunk_id": 0,
                    "type": "author",
                    "metadata": doc["metadata"],
                },
            )
            points.append(point)

            if len(points) >= BATCH_SIZE:
                with_qdrant_retry(
                    client.upsert,
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                indexed_authors += len(points)
                logger.info(
                    f"  Indexed {indexed_authors}/{len(author_documents)} authors"
                )
                points.clear()

        except Exception as e:
            logger.error(f"Failed to index author: {e}")

    if points:
        with_qdrant_retry(
            client.upsert,
            collection_name=COLLECTION_NAME,
            points=points,
        )
        indexed_authors += len(points)
        logger.info(f"  Indexed {indexed_authors}/{len(author_documents)} authors")

    info = client.get_collection(COLLECTION_NAME)

    logger.info(f"\n{'='*60}")
    logger.info("INDEXING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total points: {info.points_count}")
    logger.info(f"  Article chunks: {indexed}")
    logger.info(f"  Authors: {indexed_authors}")


def main(force_reindex: bool = False):
    """Run smart indexing with change detection and S3 snapshot management.

    Orchestrates the full indexing lifecycle:

    1. **Force reindex path** (``force_reindex=True``): Skips all change
       detection, runs :func:`_do_full_index` unconditionally, then saves
       a fresh snapshot and metadata via :func:`_save_snapshot_and_metadata`.

    2. **Change detection path** (default): Calls :func:`needs_reindex` to
       compare the current S3 source data hash against the stored metadata.

       - **No reindex needed + collection exists**: Logs the existing point
         count and returns immediately.
       - **No reindex needed + collection missing**: Attempts to restore the
         collection from an S3 snapshot via :func:`restore_snapshot_from_s3`.
         Falls through to full reindex if restoration fails.
       - **Reindex needed**: Runs :func:`_do_full_index` and saves snapshot
         and metadata.

    Args:
        force_reindex (bool): If ``True``, bypass change detection and always
            perform a full reindex. Defaults to ``False``.

    Returns:
        None
    """
    logger.info(f"Connecting to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    s3_client = s3

    if force_reindex:
        logger.info("--force-reindex: skipping change detection, running full index")
        _do_full_index(client)
        _save_snapshot_and_metadata(client, s3_client)
        return

    # Check if re-indexing is needed
    reindex_needed, reason = needs_reindex(s3_client, S3_BUCKET, S3_PREFIX, S3_SUFFIX)

    if not reindex_needed:
        logger.info(f"No reindex needed: {reason}")

        # Check if collection already exists in Qdrant (e.g. Docker volume survived)
        if client.collection_exists(COLLECTION_NAME):
            info = client.get_collection(COLLECTION_NAME)
            # Self-heal indexes for collections created before they existed
            # (idempotent; no reindex of vectors).
            ensure_payload_indexes(client)
            logger.info(
                f"Collection '{COLLECTION_NAME}' exists with "
                f"{info.points_count} points — nothing to do"
            )
            return

        # Collection missing — try to restore from S3 snapshot
        logger.info("Collection not found in Qdrant — attempting S3 snapshot restore")
        restored = restore_snapshot_from_s3(
            client, s3_client, S3_BUCKET, COLLECTION_NAME
        )
        if restored:
            info = client.get_collection(COLLECTION_NAME)
            logger.info(
                f"Restored collection with {info.points_count} points from S3 snapshot"
            )
            return

        logger.info("Snapshot restore failed — falling through to full reindex")

    else:
        logger.info(f"Reindex needed: {reason}")

    # Full re-index
    _do_full_index(client)
    _save_snapshot_and_metadata(client, s3_client)


def _save_snapshot_and_metadata(client, s3_client):
    """Save a Qdrant collection snapshot to S3 and write updated index metadata.

    Called after a successful full index run to persist the collection state
    for future snapshot-based restores and change-detection comparisons.
    Performs three operations in sequence:

    1. Fetches current collection info to obtain the live ``points_count``.
    2. Creates a Qdrant snapshot and uploads it to S3 via
       :func:`save_snapshot_to_s3`.
    3. Computes a hash of the S3 source data via
       :func:`compute_source_data_hash`, builds an index metadata record via
       :func:`build_index_metadata`, and writes it to S3 via
       :func:`save_index_metadata`.

    All errors are caught and logged; a failure here does not corrupt the
    already-completed Qdrant index.

    Args:
        client (QdrantClient): Connected Qdrant client instance, used to
            fetch collection info and create the snapshot.
        s3_client: Boto3 S3 client instance, used for all S3 read/write
            operations.

    Returns:
        None
    """
    try:
        info = client.get_collection(COLLECTION_NAME)
        snapshot_key = save_snapshot_to_s3(
            client, s3_client, S3_BUCKET, COLLECTION_NAME
        )
        source_hash = compute_source_data_hash(
            s3_client, S3_BUCKET, S3_PREFIX, S3_SUFFIX
        )
        metadata = build_index_metadata(
            snapshot_s3_key=snapshot_key,
            points_count=info.points_count,
            source_data_hash=source_hash,
        )
        save_index_metadata(s3_client, S3_BUCKET, metadata)
        logger.info("Snapshot and metadata saved to S3")
    except Exception as e:
        logger.error(f"Failed to save snapshot/metadata to S3: {e}")
        logger.info("Index is available in Qdrant but not persisted to S3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PonniRAG Qdrant Indexer")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force full re-index ignoring S3 state",
    )
    parser.add_argument(
        "--ensure-indexes",
        action="store_true",
        help="Create payload indexes on the existing collection and exit "
        "(no reindex; safe on populated data)",
    )
    args = parser.parse_args()
    if args.ensure_indexes:
        logger.info(f"Connecting to Qdrant server at {QDRANT_HOST}:{QDRANT_PORT}")
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        ensure_payload_indexes(_client)
    else:
        main(force_reindex=args.force_reindex)
