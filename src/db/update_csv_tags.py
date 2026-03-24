"""
Update summary.csv with article tags from Qdrant.

Run after re-indexing to add a 'வகை' (category) column to the CSV.

CSV columns:  வ.எ., ஆண்டு, மலர், இதழ், தலைப்பு, ஆசிரியர்
Qdrant keys:  metadata.doc_id = மலர் (volume)
              metadata.doc_issue = இதழ் (issue)
              metadata.article_no = வ.எ. (article serial number)

Each CSV row maps to one article in Qdrant. We query by doc_id + doc_issue
+ article_no to get the exact article's tags (chunk_id=0).

Usage:
    cd src/db && python update_csv_tags.py
"""

import logging
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

sys.path.append(str(Path(__file__).resolve().parents[1]))
# load_csv handles multi-author rows like [பாண்டியன், நா. வேத்தரசன், வணங்காமுடி]
# that contain commas inside brackets — which pd.read_csv silently drops.
sys.path.append(str(Path(__file__).resolve().parents[1] / "data_extraction"))

from csv_fuzzy_matcher import load_csv  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "qdrant_indexer"
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "summary.csv"

# CSV column names (Tamil headers)
COL_SERIAL = "வ.எ."  # article serial number
COL_YEAR = "ஆண்டு"
COL_VOLUME = "மலர்"  # doc_id in Qdrant
COL_ISSUE = "இதழ்"  # doc_issue in Qdrant
COL_TITLE = "தலைப்பு"
COL_AUTHOR = "ஆசிரியர்"
COL_TAGS = "வகை"  # output column we write


def get_tags_from_qdrant(
    client: QdrantClient,
    doc_id: str,
    doc_issue: str,
    article_no: str,
) -> list:
    """Query Qdrant for tags of a specific article."""
    # Primary: match by doc_id + doc_issue + article_no
    filter_conditions = [
        models.FieldCondition(key="type", match=models.MatchValue(value="article")),
        models.FieldCondition(
            key="metadata.doc_id", match=models.MatchValue(value=doc_id)
        ),
        models.FieldCondition(
            key="metadata.doc_issue", match=models.MatchValue(value=doc_issue)
        ),
        models.FieldCondition(
            key="metadata.chunk_id", match=models.MatchValue(value=0)
        ),
    ]

    # Include article_no if available
    if article_no:
        # article_no may be stored as int or string in Qdrant — try int first
        try:
            filter_conditions_with_no = filter_conditions + [
                models.FieldCondition(
                    key="metadata.article_no",
                    match=models.MatchValue(value=int(article_no)),
                )
            ]
            points, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(must=filter_conditions_with_no),
                limit=1,
                with_payload=True,
            )
            if points:
                metadata = (points[0].payload or {}).get("metadata", {})
                return metadata.get("tags_tamil", [])
        except Exception as e:
            logger.debug(f"article_no int match failed: {e}")

    # Fallback: doc_id + doc_issue only (takes first article in that issue)
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=filter_conditions),
        limit=1,
        with_payload=True,
    )
    if points:
        metadata = (points[0].payload or {}).get("metadata", {})
        return metadata.get("tags_tamil", [])

    return []


def main():
    """Update summary.csv with article tags from Qdrant."""
    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        return

    logger.info(f"Reading CSV: {CSV_PATH}")
    df = load_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # Validate required columns exist
    missing = [c for c in [COL_VOLUME, COL_ISSUE] if c not in df.columns]
    if missing:
        logger.error(
            f"Required columns missing: {missing}. Available: {list(df.columns)}"
        )
        return

    has_serial = COL_SERIAL in df.columns
    if not has_serial:
        logger.warning(
            f"Column '{COL_SERIAL}' not found — will match by மலர்+இதழ் only"
        )

    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    tags_column = []
    updated = 0
    not_found = 0

    for idx, row in df.iterrows():
        # doc_id = மலர் (volume number as string)
        doc_id = str(row.get(COL_VOLUME, "")).strip()
        # doc_issue = இதழ் (issue number as string)
        doc_issue = str(row.get(COL_ISSUE, "")).strip()
        # article_no = வ.எ. (serial number)
        article_no = str(row.get(COL_SERIAL, "")).strip() if has_serial else ""

        # Normalise: remove trailing .0 from numeric values read as float
        # e.g. pandas reads "1" as 1.0 → "1.0" → we want "1"
        if doc_id.endswith(".0"):
            doc_id = doc_id[:-2]
        if doc_issue.endswith(".0"):
            doc_issue = doc_issue[:-2]
        if article_no.endswith(".0"):
            article_no = article_no[:-2]

        if not doc_id or not doc_issue:
            tags_column.append("")
            continue

        tamil_tags = get_tags_from_qdrant(client, doc_id, doc_issue, article_no)
        tag_str = ", ".join(tamil_tags) if tamil_tags else ""
        tags_column.append(tag_str)

        if tamil_tags:
            updated += 1
        else:
            not_found += 1

        if (idx + 1) % 100 == 0:
            logger.info(
                f"  Processed {idx + 1}/{len(df)} rows "
                f"({updated} tagged, {not_found} not found)"
            )

    df[COL_TAGS] = tags_column

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved updated CSV: {CSV_PATH}")
    logger.info(
        f"Tagged {updated}/{len(df)} articles ({not_found} not found in Qdrant)"
    )


if __name__ == "__main__":
    main()
