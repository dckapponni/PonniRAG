"""Article separation pipeline for Tamil documents.

Orchestrates the end-to-end extraction of structured articles from
raw Tamil OCR text files stored in S3. The pipeline runs in four
sequential steps for each document:

1. **Filename parsing** — extracts மலர் (volume), இதழ் (issue), and
   year from the S3 key or falls back to document text.
2. **TOC processing** — locates the பொருளடக்கம் / ஆகியோரின் section,
   extracts author names for pattern matching, and marks TOC lines
   as processed. Used exclusively by pattern extraction; CSV
   extraction does not depend on the TOC.
3. **CSV extraction** — searches the full document for article titles
   listed in ``summary.csv``, matching by fuzzy title similarity
   rather than TOC position.
4. **Pattern extraction** — runs Pattern A, B, and C extractors on
   all remaining unprocessed lines, followed by intro-section and
   remaining-content passes as safety nets.

After all files are processed, per-volume ``remaining_vol{X}.json``
files are written to S3 grouping leftover content from CSV-paired
issues.
"""

import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from article_patterns import extract_pattern_a_forward  # noqa: E402
from article_patterns import extract_pattern_b_forward, extract_pattern_c_reverse
from content_extraction import extract_intro_content_phase1  # noqa: E402
from content_extraction import extract_remaining_content
from csv_fuzzy_matcher import extract_articles_from_csv  # noqa: E402
from csv_fuzzy_matcher import (
    extract_malar_ithal_from_filename,
    extract_malar_ithal_from_text,
    load_csv,
    normalize_csv_value,
)
from doc_utils import count_content_lines  # noqa: E402
from doc_utils import (
    extract_authors_alternative,
    extract_authors_from_toc,
    get_shared_authors,
    is_valid_author_name,
)
from s3_utils import read_text_from_s3  # noqa: E402
from s3_utils import file_exists, list_files, upload_json
from shared_author import build_shared_authors_dict_s3  # noqa: E402
from text_processing import get_intro_keywords, normalize_text  # noqa: E402

from config.config import EXTRACTED_OUTPUT  # noqa: E402
from config.config import BUCKET_NAME, CSV_PATH, OUTPUT_PREFIX

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


def setup_logging(log_file="tamil_doc_processing.log"):
    """
    Configure logging with file and console handlers.

    Args:
        log_file (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("TamilDocProcessor")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_file}"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")
    return logger


logger = setup_logging()


def extract_year_from_s3_key(s3_key):
    """
    Extract year (YYYY) from S3 key string.

    Args:
        s3_key (str): S3 file path.

    Returns:
        str: Extracted year or "Unknown".
    """
    try:
        year_match = re.search(r"(19|20)\d{2}", s3_key)
        if year_match:
            return year_match.group(0)
    except Exception:
        pass
    return "Unknown"


def is_file_already_processed(bucket, output_key):
    """
    Check if a file has already been processed in S3.

    Args:
        bucket (str): S3 bucket name.
        output_key (str): S3 output file key.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        return file_exists(bucket, output_key)
    except Exception as e:
        logger.warning(f"Error checking file: {e}")
        return False


def validate_processed_file(bucket, output_key):
    """
    Validate structure of processed JSON file in S3.

    Args:
        bucket (str): S3 bucket name.
        output_key (str): Output file key.

    Returns:
        bool: True if valid, else False.
    """
    try:
        from s3_utils import read_json_from_s3

        data = read_json_from_s3(bucket, output_key)
        if not isinstance(data, dict):
            return False
        if "articles" not in data:
            return False
        if not isinstance(data["articles"], list):
            return False
        return True
    except Exception:
        return False


def load_csv_from_local(csv_path):
    """
    Load CSV file from local path.

    Args:
        csv_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded CSV dataframe.
    """
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        logger.info(f"Loading CSV: {csv_path}")
        csv_df = load_csv(csv_path)
        logger.info(
            f"CSV loaded: {len(csv_df)} rows, " f"columns: {list(csv_df.columns)}"
        )
        return csv_df
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error loading CSV: {e}", exc_info=True)
        raise


def save_authors_to_s3(bucket, output_key_prefix, doc_id, doc_issue, authors_list):
    """
    Save or update authors list JSON in S3.

    Args:
        bucket (str): S3 bucket name.
        output_key_prefix (str): Output folder prefix.
        doc_id (str): Document ID (malar).
        doc_issue (str): Document issue (ithal).
        authors_list (list): List of authors.
    """
    try:
        authors_key = f"{output_key_prefix}authors.json"
        author_names = []

        if authors_list:
            if isinstance(authors_list[0], dict) and "author_name" in authors_list[0]:
                author_names = [a["author_name"] for a in authors_list]
            elif isinstance(authors_list[0], str):
                author_names = authors_list
            else:
                author_names = [str(a) for a in authors_list]

        existing_data = []
        if file_exists(bucket, authors_key):
            try:
                from s3_utils import read_json_from_s3

                existing_data = read_json_from_s3(bucket, authors_key)
            except Exception:
                pass

        doc_exists = False
        for entry in existing_data:
            if entry.get("doc_id") == doc_id and entry.get("doc_issue") == doc_issue:
                entry["authors"] = author_names
                doc_exists = True
                break

        if not doc_exists:
            existing_data.append(
                {"doc_id": doc_id, "doc_issue": doc_issue, "authors": author_names}
            )

        upload_json(bucket, authors_key, existing_data)
        logger.info(f"Authors saved: {len(author_names)} " f"for {doc_id}/{doc_issue}")

    except Exception as e:
        logger.error(f"Error saving authors: {e}", exc_info=True)
        raise


def find_toc_boundaries(lines):
    """Find TOC start and end line indices.

    Used ONLY for pattern extraction — NOT for CSV.

    Args:
        lines (list): document lines

    Returns:
        tuple: (toc_start, toc_end) or (-1, -1)
    """
    toc_start = -1
    toc_end = -1

    try:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "பொருளடக்கம்" in stripped:
                toc_start = i
                logger.debug(f"TOC start at line {i}")
            if toc_start != -1 and (
                "ஆகியோரின் எழுத்தோவியங்கள்" in stripped or "ஆகியோரின்" in stripped
            ):
                toc_end = i
                logger.debug(f"TOC end at line {i}")
                break

        if toc_start != -1 and toc_end != -1:
            logger.info(f"TOC: lines {toc_start} to {toc_end}")
        elif toc_start != -1:
            logger.warning(f"TOC start at {toc_start} but no end marker")
        else:
            logger.warning("No TOC found")

    except Exception as e:
        logger.error(f"Error finding TOC: {e}")

    return toc_start, toc_end


def extract_authors_from_toc_section(lines, toc_start, toc_end):
    """Extract author names from TOC section.

    Used ONLY for pattern extraction — NOT for CSV.

    Args:
        lines (list): document lines
        toc_start (int): TOC start index
        toc_end (int): TOC end index

    Returns:
        tuple: (authors_original, authors_normalized)
    """
    authors_original = []
    authors_normalized = []

    try:
        for i in range(toc_start + 1, toc_end):
            if i >= len(lines):
                break
            name = lines[i].strip()
            if (
                name
                and not name.isdigit()
                and len(name) > 2
                and not re.match(r"^[\d.\s…]+$", name)
                and not re.match(r"^[.\s…,]+$", name)
            ):
                authors_original.append(name)
                authors_normalized.append(normalize_text(name))

        logger.info(f"TOC section gave {len(authors_original)} authors")

    except Exception as e:
        logger.error(f"Error extracting TOC authors: {e}")

    return authors_original, authors_normalized


def extract_authors_from_content(lines):
    """Extract author names using dash pattern and position detection.

    Used as last fallback for pattern extraction when no TOC exists.

    Args:
        lines (list): document lines

    Returns:
        tuple: (authors_original, authors_normalized)
    """
    authors_original = []
    authors_normalized = []

    try:
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Dash pattern: — அம்மான் or - நாரா நாச்சியப்பன்
            if (
                stripped.startswith("—")
                or stripped.startswith("-")
                or stripped.startswith("–")
            ):
                clean = re.sub(r"^[—\-–]\s*", "", stripped).strip()
                if clean and 3 <= len(clean) <= 40 and clean not in authors_original:
                    authors_original.append(clean)
                    authors_normalized.append(normalize_text(clean))

            # Position pattern: short line right after another short line
            # (title → author pattern)
            elif (
                i > 0
                and len(lines[i - 1].strip()) <= 25
                and lines[i - 1].strip()
                and 3 <= len(stripped) <= 40
                and not stripped.isdigit()
                and stripped not in authors_original
            ):
                if is_valid_author_name(stripped):
                    authors_original.append(stripped)
                    authors_normalized.append(normalize_text(stripped))

        if authors_original:
            logger.info(
                "Content-based author extraction: " f"{len(authors_original)} authors"
            )

    except Exception as e:
        logger.error(f"Error in content author extraction: {e}")

    return authors_original, authors_normalized


def run_pattern_extraction(
    lines,
    content_start_idx,
    processed_lines,
    authors_original,
    authors_normalized,
    doc_id,
    doc_issue,
    year,
    source_document,
    start_article_no=1,
):
    """
    Run all pattern extraction on unprocessed lines only.

    பொருளடக்கம் and ஆகியோரின் are handled BEFORE this function
    in parse_tamil_document. By this point TOC lines are already
    marked as processed so patterns naturally skip them.

    For pattern articles:
    - doc_id, doc_issue → from filename
    - author, title     → from document text
    - year              → from filename
    - content           → from document text

    Args:
        lines (list): document lines
        content_start_idx (int): where content starts (after TOC)
        processed_lines (list): bool list
        authors_original (list): known authors
        authors_normalized (list): normalized authors
        doc_id (str): மலர் from filename
        doc_issue (str): இதழ் from filename
        year (str): year from filename
        source_document (str): filename
        start_article_no (int): starting article number

    Returns:
        tuple: (all_pattern_articles, remaining_only_articles)
            - all_pattern_articles: every article from Pattern A/B/C,
              intro sections, and remaining content extractor combined
            - remaining_only_articles: only articles from the remaining
              content extractor (final safety-net pass), tracked
              separately so the caller can group them into
              remaining_vol{X}.json files
    """
    articles = []
    remaining_articles = []
    article_no = start_article_no

    # Count remaining unprocessed lines
    remaining_count = sum(
        1
        for i in range(content_start_idx, len(lines))
        if not processed_lines[i] and lines[i].strip()
    )

    logger.info(f"Unprocessed content lines: {remaining_count}")

    if remaining_count == 0:
        logger.info("No remaining lines — patterns skipped")
        return articles, remaining_articles

    intro_keywords = get_intro_keywords()

    # Pattern A: HEADING → AUTHOR → CONTENT
    logger.info("Running Pattern A...")
    pattern_a = extract_pattern_a_forward(
        lines,
        content_start_idx,
        len(lines),
        authors_normalized,
        authors_original,
        processed_lines,
        intro_keywords,
    )
    for article in pattern_a:
        articles.append(
            {
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "article_no": article_no,
                "author_name": article["author"],
                "title": article["heading"],
                "content": article["content"],
                "year": year,
                "source_document": source_document,
            }
        )
        article_no += 1
    logger.info(f"Pattern A: {len(pattern_a)} articles")

    # Pattern B: AUTHOR → HEADING → CONTENT
    logger.info("Running Pattern B...")
    pattern_b = extract_pattern_b_forward(
        lines,
        content_start_idx,
        len(lines),
        authors_normalized,
        authors_original,
        processed_lines,
        intro_keywords,
    )
    for article in pattern_b:
        articles.append(
            {
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "article_no": article_no,
                "author_name": article["author"],
                "title": article["heading"],
                "content": article["content"],
                "year": year,
                "source_document": source_document,
            }
        )
        article_no += 1
    logger.info(f"Pattern B: {len(pattern_b)} articles")

    # Pattern C: HEADING → CONTENT → AUTHOR (reverse/embedded)
    logger.info("Running Pattern C...")
    pattern_c = extract_pattern_c_reverse(
        lines,
        content_start_idx,
        len(lines),
        authors_normalized,
        authors_original,
        processed_lines,
        intro_keywords,
    )
    for article in pattern_c:
        articles.append(
            {
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "article_no": article_no,
                "author_name": article["author"],
                "title": article["heading"],
                "content": article["content"],
                "year": year,
                "source_document": source_document,
            }
        )
        article_no += 1
    logger.info(f"Pattern C: {len(pattern_c)} articles")

    # Intro sections
    logger.info("Running intro extraction...")
    intro_count = 0
    i = content_start_idx

    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue

        line = lines[i].strip()
        matched_keyword = None

        for keyword in intro_keywords:
            if keyword in line and (i == 0 or not lines[i - 1].strip()):
                matched_keyword = keyword
                break

        if matched_keyword:
            try:
                content, end_idx_content, has_author = extract_intro_content_phase1(
                    lines,
                    i,
                    processed_lines,
                    authors_normalized,
                    authors_original,
                    intro_keywords,
                )
                if content.strip() and count_content_lines(content) >= 4:
                    articles.append(
                        {
                            "doc_id": doc_id,
                            "doc_issue": doc_issue,
                            "article_no": article_no,
                            "author_name": has_author if has_author else "NA",
                            "title": matched_keyword,
                            "content": content,
                            "year": year,
                            "source_document": source_document,
                        }
                    )
                    article_no += 1
                    intro_count += 1

                    for j in range(i, end_idx_content):
                        if j < len(lines):
                            processed_lines[j] = True
                    i = end_idx_content
                else:
                    i += 1
            except Exception as e:
                logger.warning(f"Intro error at line {i}: {e}")
                i += 1
        else:
            i += 1

    logger.info(f"Intro sections: {intro_count} articles")

    # ------------------------------------------------------------------
    # Remaining content — final safety-net pass.
    # These articles are added to the main list AND tracked separately
    # so process_s3_files() can group them into remaining_vol{X}.json.
    # ------------------------------------------------------------------
    logger.info("Running remaining content extraction...")
    remaining = extract_remaining_content(lines, content_start_idx, processed_lines)

    for article in remaining:
        built = {
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "author_name": article["author"],
            "title": article["heading"],
            "content": article["content"],
            "year": year,
            "source_document": source_document,
        }
        articles.append(built)
        remaining_articles.append(built)  # ← tracked separately
        article_no += 1

    logger.info(f"Remaining: {len(remaining)} articles")

    return articles, remaining_articles


def find_first_csv_title_line(lines, csv_df, malar, ithal):
    """
    Find the line index of the first article title from CSV in the document.

    Used when ஆகியோரின் end-marker is missing so we can skip the
    header/TOC region by stopping one line above the first CSV title.

    Strategy mirrors find_article_boundary_fuzzy: prefer the standalone
    (near-exact, shorter) occurrence over the TOC occurrence.

    Args:
        lines (list): document lines
        csv_df (pd.DataFrame): loaded CSV dataframe
        malar (str): மலர் value
        ithal (str): இதழ் value

    Returns:
        int: line index of the first CSV title found, or -1 if not found
    """
    try:
        from csv_fuzzy_matcher import (
            calculate_similarity,
            match_csv_rows,
            remove_symbols,
        )

        matched_rows = match_csv_rows(csv_df, malar, ithal)
        if matched_rows.empty:
            logger.warning("find_first_csv_title_line: no CSV rows matched")
            return -1

        import pandas as pd

        # Use the very first title in the CSV for this volume/issue
        first_title_raw = None
        for _, row in matched_rows.iterrows():
            t = str(row["தலைப்பு"]) if pd.notna(row["தலைப்பு"]) else ""
            if t:
                first_title_raw = t
                break

        if not first_title_raw:
            logger.warning("find_first_csv_title_line: first CSV title is empty")
            return -1

        title_clean = remove_symbols(first_title_raw)
        candidates = []

        for i, line in enumerate(lines):
            line_clean = remove_symbols(line.strip())
            if not line_clean:
                continue
            sim = calculate_similarity(title_clean, line_clean)
            if sim >= 70:
                candidates.append((sim, i, len(line.strip())))

        if not candidates:
            logger.warning(
                f"find_first_csv_title_line: "
                f"no candidates for '{first_title_raw[:40]}'"
            )
            return -1

        # Prefer standalone (near-exact) occurrence — same logic as CSV extractor
        near_exact = [c for c in candidates if c[0] >= 95]
        if near_exact:
            best = min(near_exact, key=lambda c: c[2])
        else:
            best = max(candidates, key=lambda c: (c[0], -c[2]))

        logger.info(
            f"First CSV title '{first_title_raw[:40]}' "
            f"found at line {best[1]} (sim={best[0]:.1f}%)"
        )
        return best[1]

    except Exception as e:
        logger.error(f"Error in find_first_csv_title_line: {e}", exc_info=True)
        return -1


def mark_varappetrrem_section(
    lines,
    processed_lines,
    start_search_from=0,
    authors_normalized=None,
    authors_original=None,
    csv_df=None,
    malar=None,
    ithal=None,
):
    """
    Find every standalone வரப்பெற்றேம் line and mark it + all content below processed.

    stopping when the next CSV title or a known author name is encountered.

    'Standalone' means the line contains வரப்பெற்றேம் and is short
    (≤ 60 chars), i.e. not embedded mid-sentence in an article.

    Stop conditions (whichever comes first):
      1. A line that fuzzy-matches a known author name.
      2. A line that fuzzy-matches any CSV title for this volume/issue.
      3. End of document.

    Args:
        lines (list): document lines
        processed_lines (list): bool list — modified in place
        start_search_from (int): line index to begin searching from
        authors_normalized (list): normalized author names (may be None)
        authors_original (list): original author names (may be None)
        csv_df (pd.DataFrame): loaded CSV dataframe (may be None)
        malar (str): மலர் value for CSV lookup (may be None)
        ithal (str): இதழ் value for CSV lookup (may be None)

    Returns:
        int: number of வரப்பெற்றேம் sections found and marked
    """
    import pandas as pd
    from csv_fuzzy_matcher import calculate_similarity, remove_symbols
    from text_processing import extract_author_from_line

    # Build a set of cleaned CSV titles for this volume/issue
    csv_titles_clean = set()
    if csv_df is not None and malar and ithal:
        try:
            from csv_fuzzy_matcher import match_csv_rows

            matched_rows = match_csv_rows(csv_df, malar, ithal)
            if not matched_rows.empty:
                for _, row in matched_rows.iterrows():
                    t = str(row["தலைப்பு"]) if pd.notna(row["தலைப்பு"]) else ""
                    if t:
                        csv_titles_clean.add(remove_symbols(t))
        except Exception as e:
            logger.warning(f"mark_varappetrrem_section: CSV title load failed: {e}")

    _authors_norm = authors_normalized or []
    _authors_orig = authors_original or []

    def _is_stop_line(line_text):
        """Return True if this line signals end of வரப்பெற்றேம் section."""
        stripped = line_text.strip()
        if not stripped:
            return False

        # Stop if it matches a known author
        if _authors_norm and _authors_orig:
            if extract_author_from_line(stripped, _authors_norm, _authors_orig):
                return True

        # Stop if it matches a CSV title (≥ 85% similarity)
        if csv_titles_clean:
            line_clean = remove_symbols(stripped)
            for title_clean in csv_titles_clean:
                if calculate_similarity(line_clean, title_clean) >= 85:
                    return True

        return False

    found = 0

    for i in range(start_search_from, len(lines)):
        stripped = lines[i].strip()

        # Must be a standalone line containing the keyword
        if "வரப்பெற்றேம்" not in stripped:
            continue
        if len(stripped) > 60:
            continue

        logger.info(f"Found standalone வரப்பெற்றேம் at line {i}: '{stripped}'")

        # Mark the keyword line itself
        processed_lines[i] = True

        # Mark everything below until next CSV title or author
        j = i + 1
        while j < len(lines):
            if _is_stop_line(lines[j]):
                logger.info(
                    f"வரப்பெற்றேம் section ended at line {j}: "
                    f"'{lines[j].strip()[:50]}' "
                    "(CSV title or author found)"
                )
                break
            processed_lines[j] = True
            j += 1

        logger.info(f"Marked வரப்பெற்றேம் section lines {i}–{j - 1} as processed")
        found += 1

    return found


def parse_tamil_document(lines, shared_authors_dict, csv_df, s3_key):
    """Parse a Tamil document into structured articles.

    IMPORTANT: பொருளடக்கம் and ஆகியோரின் are used ONLY
    for pattern extraction (author extraction + line marking).
    CSV extraction does NOT use TOC at all.

    Flow:
    ┌─────────────────────────────────────────────────────────┐
    │ STEP 1: Extract மலர்/இதழ்/year from FILENAME           │
    │         Fallback to document text if filename fails     │
    ├─────────────────────────────────────────────────────────┤
    │ STEP 2: [PATTERN PREP ONLY]                             │
    │         Find TOC (பொருளடக்கம் → ஆகியோரின்)            │
    │         Extract authors from TOC for pattern matching   │
    │         Mark TOC lines as processed                     │
    ├─────────────────────────────────────────────────────────┤
    │ STEP 3: CSV extraction (if மலர்/இதழ் found)            │
    │         Search titles in FULL document                  │
    │         No TOC dependency                               │
    │         Mark extracted lines as processed               │
    ├─────────────────────────────────────────────────────────┤
    │ STEP 4: Pattern extraction on REMAINING lines only      │
    │         TOC already marked → patterns skip it           │
    └─────────────────────────────────────────────────────────┘

    Returns:
        dict with keys:
            articles           - all articles (CSV + pattern + remaining)
            remaining_articles - only articles from remaining content extractor,
                                 populated only when CSV extraction ran,
                                 used to build remaining_vol{X}.json
            authors_list       - author metadata
            doc_id             - மலர் value
            doc_issue          - இதழ் value
            csv_ran            - True if CSV extraction produced articles
    """
    try:
        source_document = s3_key.split("/")[-1]

        logger.info("=" * 80)
        logger.info(f"PARSING: {source_document}")
        logger.info("=" * 80)

        class S3Path:
            def __init__(self, key):
                self.name = key.split("/")[-1]

        file_path = S3Path(s3_key)

        logger.info("STEP 1: மலர்/இதழ் from filename")

        malar, ithal, year_from_file = extract_malar_ithal_from_filename(
            source_document
        )

        # Fallback to document text if filename fails
        if not malar or not ithal:
            logger.warning("Filename parse failed — trying document text")
            malar, ithal, year_from_file = extract_malar_ithal_from_text(lines)

        # Year priority: filename/text → S3 key path
        if year_from_file:
            year = year_from_file
        else:
            year = extract_year_from_s3_key(s3_key)
            logger.info(f"Year from S3 key fallback: {year}")

        doc_id = normalize_csv_value(malar) if malar else "NA"
        doc_issue = normalize_csv_value(ithal) if ithal else "NA"

        logger.info(f"doc_id={doc_id}, doc_issue={doc_issue}, year={year}")

        logger.info("STEP 2: TOC processing (for pattern extraction only)")

        processed_lines = [False] * len(lines)

        # Find TOC boundaries — patterns only
        toc_start, toc_end = find_toc_boundaries(lines)

        authors_original = []
        authors_normalized = []

        if toc_start != -1 and toc_end != -1:
            # Extract authors from TOC for pattern matching
            authors_original, authors_normalized = extract_authors_from_toc_section(
                lines, toc_start, toc_end
            )
            # Mark TOC lines as processed so patterns skip them
            for i in range(0, toc_end + 1):
                if i < len(lines):
                    processed_lines[i] = True
            content_start_idx = toc_end + 1
            logger.info(
                f"TOC lines 0-{toc_end} marked processed. "
                f"Pattern content starts at {content_start_idx}"
            )

        elif toc_start != -1:
            # TOC found but ஆகியோரின் end marker is missing.
            #
            # Strategy (in priority order):
            #   1. Find the first CSV title in the document.
            #      Mark lines 0 .. (csv_title_line - 2) as processed
            #      so content extraction starts one line above the title.
            #   2. If no CSV title found, fall back to 15 lines from
            #      பொருளடக்கம் (same safe default as before).

            first_csv_line = -1

            if malar and ithal:
                first_csv_line = find_first_csv_title_line(lines, csv_df, malar, ithal)

            if first_csv_line > toc_start:
                # Mark from line 0 up to (csv_title_line - 1) as processed.
                # Example: title at line 18 → mark lines 0–17 as processed,
                # content_start_idx = 18 so extraction starts at the title.
                for i in range(0, first_csv_line):
                    if i < len(lines):
                        processed_lines[i] = True
                content_start_idx = first_csv_line
                logger.warning(
                    f"No ஆகியோரின் end marker. "
                    f"First CSV title at line {first_csv_line}. "
                    f"Marking lines 0–{first_csv_line - 1} as processed. "
                    f"Content starts at line {content_start_idx} "
                    f"(the CSV title line itself)."
                )
            else:
                # Fallback: 15 lines from பொருளடக்கம்
                end_fallback = min(toc_start + 15, len(lines))
                for i in range(0, end_fallback):
                    if i < len(lines):
                        processed_lines[i] = True
                content_start_idx = end_fallback
                logger.warning(
                    f"No ஆகியோரின் end marker and no CSV title found. "
                    f"Falling back to 15 lines from பொருளடக்கம் "
                    f"(lines 0–{end_fallback} marked as processed)."
                )

        else:
            # No TOC at all
            content_start_idx = 0
            logger.warning("No TOC — patterns start from line 0")

        # ------------------------------------------------------------------
        # Mark all standalone வரப்பெற்றேம் sections as processed.
        # Stops at the next CSV title or known author — runs across the
        # full document from content_start_idx onward.
        # ------------------------------------------------------------------
        varappetrrem_count = mark_varappetrrem_section(
            lines,
            processed_lines,
            start_search_from=content_start_idx,
            authors_normalized=authors_normalized,
            authors_original=authors_original,
            csv_df=csv_df,
            malar=malar,
            ithal=ithal,
        )
        if varappetrrem_count:
            logger.info(
                f"Marked {varappetrrem_count} வரப்பெற்றேம் " "section(s) as processed."
            )

        # Author fallback methods for patterns
        if not authors_original:
            logger.info("Trying TOC parse method for authors")
            _, _, toc_orig, toc_norm, _ = extract_authors_from_toc(lines)
            if toc_orig:
                authors_original = toc_orig
                authors_normalized = toc_norm
                logger.info(f"TOC parse: {len(authors_original)} authors")

        if not authors_original and doc_id != "NA" and doc_issue != "NA":
            logger.info("Trying shared authors")
            authors_original, authors_normalized = get_shared_authors(
                doc_id, doc_issue, shared_authors_dict
            )
            if authors_original:
                logger.info(f"Shared authors: {len(authors_original)}")

        if not authors_original:
            logger.info("Trying alternative author extraction")
            authors_original, authors_normalized = extract_authors_alternative(lines)
            if authors_original:
                logger.info(f"Alternative: {len(authors_original)} authors")

        if not authors_original:
            logger.info(
                "Trying content-based author extraction " "(dash/position patterns)"
            )
            authors_original, authors_normalized = extract_authors_from_content(lines)
            if authors_original:
                logger.info(f"Content-based: {len(authors_original)} authors")

        logger.info("Total authors for pattern matching: " f"{len(authors_original)}")

        articles = []
        article_no = 1
        csv_ran = False

        if doc_id != "NA" and doc_issue != "NA":
            logger.info(
                "STEP 3: CSV extraction "
                "(independent of TOC — searches full document)"
            )

            csv_result = extract_articles_from_csv(lines, csv_df, file_path)

            if csv_result and len(csv_result["articles"]) > 0:
                csv_ran = True

                for article in csv_result["articles"]:
                    article["source_document"] = source_document
                    articles.append(article)
                    article_no += 1

                # Mark CSV-extracted lines as processed
                # so patterns don't re-extract them
                for start_line, end_line in csv_result.get("extracted_line_ranges", []):
                    for k in range(start_line, min(end_line, len(lines))):
                        processed_lines[k] = True

                logger.info(
                    f"CSV: {len(csv_result['articles'])} articles. "
                    "Remaining lines go to patterns."
                )

                # Merge CSV authors into pattern authors list
                if csv_result.get("authors_list"):
                    csv_authors = [
                        a["author_name"]
                        for a in csv_result["authors_list"]
                        if a["author_name"] not in authors_original
                    ]
                    if csv_authors:
                        authors_original += csv_authors
                        authors_normalized += [normalize_text(a) for a in csv_authors]
                        logger.info(
                            f"Added {len(csv_authors)} CSV authors "
                            "for pattern matching"
                        )
            else:
                logger.warning(
                    "CSV returned no articles — " "all content goes to patterns"
                )
        else:
            logger.info(
                "STEP 3: CSV skipped " "(no மலர்/இதழ் found). All content → patterns."
            )

        logger.info(
            "STEP 4: Pattern extraction on remaining lines "
            "(TOC and CSV lines already marked as processed)"
        )

        pattern_articles, remaining_articles = run_pattern_extraction(
            lines=lines,
            content_start_idx=content_start_idx,
            processed_lines=processed_lines,
            authors_original=authors_original,
            authors_normalized=authors_normalized,
            doc_id=doc_id,
            doc_issue=doc_issue,
            year=year,
            source_document=source_document,
            start_article_no=article_no,
        )

        articles.extend(pattern_articles)

        # Final authors list
        authors_list = [
            {"doc_id": doc_id, "doc_issue": doc_issue, "author_name": a}
            for a in authors_original
        ]

        logger.info("=" * 80)
        logger.info(f"TOTAL: {len(articles)} articles")
        logger.info(f"  CSV: {len(articles) - len(pattern_articles)}")
        logger.info(
            f"Patterns (A+B+C+intro):{len(pattern_articles)-len(remaining_articles)}"
        )
        logger.info(f"  Remaining content:      {len(remaining_articles)}")
        logger.info("=" * 80)

        return {
            "articles": articles,
            # Only expose remaining_articles when CSV ran so that
            # files without CSV don't pollute remaining_vol*.json
            "remaining_articles": remaining_articles if csv_ran else [],
            "authors_list": authors_list,
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "csv_ran": csv_ran,
        }

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        raise


def save_remaining_per_volume(bucket, output_prefix, vol_remaining_map):
    """
    Save per-volume remaining content as remaining_vol{X}.json in S3.

    Called once after ALL files are processed.
    Each JSON groups remaining articles from every issue of that
    volume that used CSV extraction, so nothing is overwritten
    mid-run.

    Output structure:
    {
        "doc_id": "1",
        "issues": [
            {
                "doc_issue": "1",
                "source_document": "VOL1-1-1948.txt",
                "article_count": 3,
                "articles": [ { ...article dict... }, ... ]
            },
            {
                "doc_issue": "4",
                "source_document": "VOL1-4-1948.txt",
                "article_count": 2,
                "articles": [ ... ]
            }
        ],
        "total_remaining": 5
    }

    Args:
        bucket (str): S3 bucket name.
        output_prefix (str): S3 output prefix folder.
        vol_remaining_map (dict): Mapping of
            doc_id (str) → list of dicts, each dict being:
                {
                    "doc_issue":        str,
                    "source_document":  str,
                    "articles":         list[dict]
                }
    """
    for doc_id, issue_entries in vol_remaining_map.items():
        if not issue_entries:
            continue

        # Attach article_count to each issue entry for readability
        annotated_issues = []
        for entry in issue_entries:
            annotated_issues.append(
                {
                    "doc_issue": entry["doc_issue"],
                    "source_document": entry["source_document"],
                    "article_count": len(entry["articles"]),
                    "articles": entry["articles"],
                }
            )

        total = sum(e["article_count"] for e in annotated_issues)

        payload = {
            "doc_id": doc_id,
            "issues": annotated_issues,
            "total_remaining": total,
        }

        output_key = f"{output_prefix}remaining_vol{doc_id}.json"

        try:
            upload_json(bucket, output_key, payload)
            logger.info(
                f"Saved remaining_vol{doc_id}.json — "
                f"{len(annotated_issues)} issue(s), "
                f"{total} remaining article(s) "
                f"→ s3://{bucket}/{output_key}"
            )
        except Exception as e:
            logger.error(
                f"Failed saving remaining_vol{doc_id}.json: {e}", exc_info=True
            )


def process_s3_files(force_reprocess=False):
    """
    Orchestrate processing of all TXT files from S3.

    After ALL files are processed the function groups remaining
    content articles (from issues that used CSV extraction) by
    volume and saves them as remaining_vol{X}.json files.

    Example output files produced:
        output/remaining_vol1.json   ← issues 1, 4, 6 of vol 1
        output/remaining_vol2.json   ← issues 2, 5    of vol 2

    Args:
        force_reprocess (bool): Reprocess all files if True.
    """
    try:
        logger.info("=" * 80)
        logger.info("Tamil Document Processing")
        logger.info(f"Bucket: {BUCKET_NAME}")
        logger.info(f"Input:  {EXTRACTED_OUTPUT}")
        logger.info(f"Output: {OUTPUT_PREFIX}")
        logger.info(f"CSV:    {CSV_PATH}")
        logger.info(f"Force:  {force_reprocess}")
        logger.info("=" * 80)

        # Load CSV
        try:
            csv_df = load_csv_from_local(CSV_PATH)
        except FileNotFoundError:
            logger.error(f"CSV not found: {CSV_PATH}")
            return
        except Exception as e:
            logger.error(f"CSV load failed: {e}")
            return

        # List TXT files
        txt_files = list_files(BUCKET_NAME, EXTRACTED_OUTPUT, suffix=".txt")
        if not txt_files:
            logger.warning("No TXT files found")
            return
        logger.info(f"Found {len(txt_files)} TXT files")

        # Build shared authors
        logger.info("Building shared authors dictionary...")
        shared_authors_dict = build_shared_authors_dict_s3(
            BUCKET_NAME, EXTRACTED_OUTPUT
        )
        logger.info(f"Shared authors: {len(shared_authors_dict)} groups")

        processed = failed = skipped = 0

        # ---------------------------------------------------------------
        # vol_remaining_map accumulates remaining articles grouped by
        # volume (doc_id) across all processed issues.
        #
        # Structure:
        #   {
        #     "1": [
        #       { "doc_issue": "1", "source_document": "VOL1-1-1948.txt",
        #         "articles": [...] },
        #       { "doc_issue": "4", "source_document": "VOL1-4-1948.txt",
        #         "articles": [...] },
        #     ],
        #     "2": [ ... ]
        #   }
        # ---------------------------------------------------------------
        vol_remaining_map = defaultdict(list)

        for idx, txt_key in enumerate(txt_files, 1):

            relative_key = txt_key[len(EXTRACTED_OUTPUT) :]
            output_key = f"{OUTPUT_PREFIX}{relative_key.rsplit('.', 1)[0]}.json"

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[{idx}/{len(txt_files)}] {txt_key}")
            logger.info("=" * 80)

            try:
                # Check if already processed
                if not force_reprocess:
                    if is_file_already_processed(BUCKET_NAME, output_key):
                        if validate_processed_file(BUCKET_NAME, output_key):
                            logger.info("SKIPPED: already processed")
                            skipped += 1
                            continue
                        else:
                            logger.warning("Invalid file — reprocessing")

                # Read and parse
                text_content = read_text_from_s3(BUCKET_NAME, txt_key)
                lines = text_content.splitlines()
                logger.debug(f"Read {len(lines)} lines")

                result = parse_tamil_document(
                    lines, shared_authors_dict, csv_df, txt_key
                )

                # ----------------------------------------------------------
                # Save main articles JSON — CSV + all pattern articles
                # (this file is unchanged from before)
                # ----------------------------------------------------------
                upload_json(
                    BUCKET_NAME,
                    output_key,
                    {"articles": result["articles"]},
                )
                logger.info(f"Saved {len(result['articles'])} articles → {output_key}")

                # Save authors JSON
                output_key_prefix = (
                    output_key.rsplit("/", 1)[0] + "/" if "/" in output_key else ""
                )
                save_authors_to_s3(
                    BUCKET_NAME,
                    output_key_prefix,
                    result["doc_id"],
                    result["doc_issue"],
                    result["authors_list"],
                )

                # ----------------------------------------------------------
                # Collect remaining articles into per-volume map.
                # Condition: CSV must have run AND remaining articles exist.
                # This ensures only CSV-paired issues are grouped.
                # ----------------------------------------------------------
                if result["csv_ran"] and result["remaining_articles"]:
                    doc_id = result["doc_id"]
                    doc_issue = result["doc_issue"]
                    source_doc = txt_key.split("/")[-1]

                    vol_remaining_map[doc_id].append(
                        {
                            "doc_issue": doc_issue,
                            "source_document": source_doc,
                            "articles": result["remaining_articles"],
                        }
                    )
                    logger.info(
                        f"Queued {len(result['remaining_articles'])} remaining "
                        f"article(s) from issue {doc_issue} "
                        f"into remaining_vol{doc_id}.json"
                    )

                logger.info(f"SUCCESS: {len(result['articles'])} total articles")
                processed += 1

            except Exception as e:
                logger.error(f"FAILED: {txt_key}: {e}", exc_info=True)
                failed += 1

        # ------------------------------------------------------------------
        # After ALL issues are processed, write one remaining_vol{X}.json
        # per volume that had CSV-extracted issues with leftover content.
        # ------------------------------------------------------------------
        if vol_remaining_map:
            logger.info("")
            logger.info("=" * 80)
            logger.info(
                f"Writing remaining_vol*.json for "
                f"{len(vol_remaining_map)} volume(s)..."
            )
            logger.info("=" * 80)
            save_remaining_per_volume(BUCKET_NAME, OUTPUT_PREFIX, vol_remaining_map)
        else:
            logger.info(
                "No remaining articles to save — "
                "no remaining_vol*.json files created."
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info(f"  Total:     {len(txt_files)}")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Skipped:   {skipped}")
        logger.info(f"  Failed:    {failed}")
        if len(txt_files) > 0:
            logger.info(
                "  Success: " f"{((processed + skipped) / len(txt_files) * 100):.1f}%"
            )
        logger.info("=" * 80)

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        force_reprocess = "--force" in sys.argv or "-f" in sys.argv

        if force_reprocess:
            logger.info("FORCE REPROCESS MODE")
        else:
            logger.info("INCREMENTAL MODE — use --force to reprocess all")

        process_s3_files(force_reprocess=force_reprocess)
        logger.info("Done!")

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed: {e}")
        sys.exit(1)
