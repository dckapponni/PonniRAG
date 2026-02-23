"""
Update summary.csv with article tags from Qdrant.

Run after re-indexing to add a 'வகை' (category) column to the CSV.
Queries Qdrant for each article's tags by doc_id + doc_issue.

Usage:
    cd src/db && python update_csv_tags.py
"""
import os
import sys
import logging
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models

sys.path.append(str(Path(__file__).resolve().parents[1]))

from db.article_tagger import TAXONOMY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "summary.csv"


def get_tags_from_qdrant(client: QdrantClient, doc_id: str, doc_issue: str):
    """Query Qdrant for tags of a specific article (chunk_id=0)."""
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="type", match=models.MatchValue(value="article")),
            models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="metadata.doc_issue", match=models.MatchValue(value=doc_issue)),
            models.FieldCondition(key="metadata.chunk_id", match=models.MatchValue(value=0)),
        ]),
        limit=1,
        with_payload=True,
    )
    if points:
        metadata = (points[0].payload or {}).get("metadata", {})
        return metadata.get("tags_tamil", [])
    return []


def main():
    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        return

    logger.info(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Determine doc_id and doc_issue column names
    doc_id_col = None
    doc_issue_col = None
    for col in df.columns:
        if "doc_id" in col.lower():
            doc_id_col = col
        if "doc_issue" in col.lower() or "இதழ்" in col:
            doc_issue_col = col

    if not doc_id_col or not doc_issue_col:
        logger.error(f"Could not find doc_id/doc_issue columns. Columns: {list(df.columns)}")
        return

    logger.info(f"Using columns: doc_id={doc_id_col}, doc_issue={doc_issue_col}")

    tags_column = []
    updated = 0

    for idx, row in df.iterrows():
        doc_id = str(row.get(doc_id_col, "")).strip()
        doc_issue = str(row.get(doc_issue_col, "")).strip()

        if not doc_id or not doc_issue:
            tags_column.append("")
            continue

        tamil_tags = get_tags_from_qdrant(client, doc_id, doc_issue)
        tag_str = ", ".join(tamil_tags) if tamil_tags else ""
        tags_column.append(tag_str)

        if tamil_tags:
            updated += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} rows ({updated} tagged)")

    df["வகை"] = tags_column

    output_path = CSV_PATH
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved updated CSV: {output_path}")
    logger.info(f"Tagged {updated}/{len(df)} articles")


if __name__ == "__main__":
    main()
