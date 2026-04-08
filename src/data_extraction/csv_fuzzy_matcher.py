"""CSV fuzzy matching utilities for Tamil document article extraction.

New CSV format (2025):
  - வ.எ.              : row serial number
  - ஆண்டு             : year
  - மலர்              : volume
  - இதழ்              : issue
  - தலைப்பு           : title
  - ஆசிரியர்          : author (brackets = main-topic, plain = subtopic)
  - பெற்றோர்_வ.எ.    : parent row number (NaN = main topic, int = subtopic)

Output JSON shape
-----------------
{
  "articles": [
    {
      "doc_id": "1",
      "doc_issue": "6",
      "article_no": 1,
      "title": "வளரும் இலக்கியம்",           ← main topic
      "author_name": [],                       ← [] when no author
      "year": "1947",
      "source_document": "VOL1-6-1947.txt",
      "subtopics": [
        {
          "subtopic_no": 1,
          "title": "அழகு",
          "author_name": "தோழர் அறிவழகன்",
          "content": "..."
        },
        ...
      ]
    },
    {
      "doc_id": "1",
      "doc_issue": "6",
      "article_no": 2,
      "title": "கண் திறக்குமா?",              ← standalone (no subtopics)
      "author_name": ["நக்கீரன்"],
      "content": "...",
      "year": "1947",
      "source_document": "VOL1-6-1947.txt",
      "subtopics": []
    }
  ]
}
"""

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

logger = logging.getLogger("TamilDocProcessor.csv_fuzzy_matcher")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_csv(csv_path):
    """Load CSV file.

    Multi-author fields wrapped in double quotes are handled
    by pandas quotechar setting.

    "[பாண்டியன், நா. வேத்தரசன், வணங்காமுடி]" → single field.
    """
    df = pd.read_csv(csv_path, encoding="utf-8", quotechar='"')
    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def calculate_similarity(str1, str2):
    """Return similarity percentage (0-100) between two strings."""
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio() * 100


def remove_symbols(text):
    """Remove punctuation/symbols; keep Unicode letters and spaces."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_csv_value(val):
    """Normalize CSV values: '1.0' → '1', strip whitespace."""
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if "." in val_str:
        try:
            float_val = float(val_str)
            if float_val == int(float_val):
                return str(int(float_val))
        except (ValueError, OverflowError):
            pass
    return val_str


# ---------------------------------------------------------------------------
# Author field parsing
# ---------------------------------------------------------------------------


def parse_author_field(author_val):
    """Parse author field from either format.

    Main-topic rows  : "[நக்கீரன்]" or "[A, B, C]" or "[]" or NA
    Subtopic rows    : "தோழர் அறிவழகன்"  (plain, no brackets)

    Returns:
        list[str]: list of author name strings (may be empty)
    """
    if pd.isna(author_val):
        return []
    raw = str(author_val).strip()
    if not raw or raw == "NA":
        return []

    # Bracket-wrapped (main topic)
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        authors = [a.strip() for a in inner.split(",") if a.strip()]
        return authors

    # Plain (subtopic) — treat whole value as single author
    return [raw]


def is_subtopic_row(row):
    """Return True when the row has a parent row number (it is a subtopic)."""
    parent_col = "பெற்றோர்_வ.எ."
    if parent_col not in row.index:
        return False
    val = row[parent_col]
    if pd.isna(val):
        return False
    try:
        int(float(val))
        return True
    except (ValueError, TypeError):
        return False


def get_parent_row_no(row):
    """Return the parent வ.எ. as int, or None."""
    try:
        return int(float(row["பெற்றோர்_வ.எ."]))
    except (ValueError, TypeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Filename / text extraction
# ---------------------------------------------------------------------------


def extract_malar_ithal_from_filename(filename):
    """Extract மலர் and இதழ் from filename.

    Handles: VOL5-7-1951 | VOL5 - 7 - 1951 | VOL1-PONGAL-1948
             VOL_5_7_1951 | 5-7-1951

    Returns:
        tuple: (malar, ithal, year) as strings, or (None, None, None)
    """
    try:
        name = Path(filename).stem
        logger.info(f"Extracting மலர்/இதழ் from filename: {name}")

        # Pattern 1: VOL5-7-1951 (numeric இதழ்)
        vol_match = re.search(
            r"VOL\s*[-_]?\s*(\d+)\s*[-_]\s*(\d+)\s*[-_]\s*(\d{4})",
            name,
            re.IGNORECASE,
        )
        if vol_match:
            malar = str(int(vol_match.group(1)))
            ithal = str(int(vol_match.group(2)))
            year = vol_match.group(3)
            logger.info(f"VOL pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        # Pattern 2: VOL1-PONGAL-1948 (text இதழ்)
        vol_text_match = re.search(
            r"VOL\s*[-_]?\s*(\d+)\s*[-_]\s*([A-Za-z]+)\s*[-_]\s*(\d{4})",
            name,
            re.IGNORECASE,
        )
        if vol_text_match:
            malar = str(int(vol_text_match.group(1)))
            ithal_text = vol_text_match.group(2).upper()
            year = vol_text_match.group(3)
            ITHAL_MAP = {"PONGAL": "பொங்கல் மலர்"}
            ithal = ITHAL_MAP.get(ithal_text, ithal_text)
            logger.info(f"VOL-TEXT pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        # Pattern 3: vol_X_issue_Y_YYYY
        vol_issue_match = re.search(
            r"vol[-_\s]*(\d+)[-_\s]*(?:issue)?[-_\s]*(\d+)[-_\s]*(\d{4})?",
            name,
            re.IGNORECASE,
        )
        if vol_issue_match:
            malar = str(int(vol_issue_match.group(1)))
            ithal = str(int(vol_issue_match.group(2)))
            year = vol_issue_match.group(3) if vol_issue_match.group(3) else None
            logger.info(f"vol_issue pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        # Pattern 4: plain X-Y-YYYY
        plain_match = re.search(r"(\d+)\s*[-_]\s*(\d+)\s*[-_]\s*(\d{4})", name)
        if plain_match:
            malar = str(int(plain_match.group(1)))
            ithal = str(int(plain_match.group(2)))
            year = plain_match.group(3)
            logger.info(f"Plain pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        logger.warning(f"Could not extract மலர்/இதழ் from filename: {name}")
        return None, None, None

    except Exception as e:
        logger.error(f"Error extracting from filename {filename}: {e}")
        return None, None, None


def extract_malar_ithal_from_text(lines):
    """Extract மலர் and இதழ் from document text (fallback).

    Args:
        lines (list): document lines

    Returns:
        tuple: (malar, ithal, year) or (None, None, None)
    """
    try:
        search_lines = lines[:50]
        text = "\n".join(search_lines)

        malar = None
        ithal = None
        year = None

        pongal_match = re.search(r"பொங்கல்\s*மலர்", text)
        if pongal_match:
            malar = "பொங்கல்"
            ithal = "பொங்கல் மலர்"
            logger.info("Found பொங்கல் மலர் in text")

        year_match = re.search(r"(19|20)\d{2}", text)
        if year_match:
            year = year_match.group(0)

        if not malar:
            malar_match = re.search(r"மலர்\s*[:—\-]?\s*(\d+)", text)
            if malar_match:
                malar = str(int(malar_match.group(1)))

        if not ithal:
            ithal_match = re.search(r"இதழ்\s*[:—\-]?\s*(\d+)", text)
            if ithal_match:
                ithal = str(int(ithal_match.group(1)))

        if malar and ithal:
            logger.info(f"Text extraction: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        logger.warning("Could not extract மலர்/இதழ் from document text")
        return None, None, None

    except Exception as e:
        logger.error(f"Error extracting மலர்/இதழ் from text: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# CSV row matching
# ---------------------------------------------------------------------------


def match_csv_rows(csv_df, malar, ithal):
    """Match CSV rows by மலர் (exact) and இதழ் (fuzzy + contains).

    Returns:
        pd.DataFrame: matched rows (all columns preserved)
    """
    malar_norm = normalize_csv_value(malar).strip()
    ithal_norm = normalize_csv_value(ithal).strip()

    df = csv_df.copy()
    df["_மலர்_norm"] = df["மலர்"].apply(lambda x: normalize_csv_value(x).strip())
    df["_இதழ்_norm"] = df["இதழ்"].apply(lambda x: normalize_csv_value(x).strip())

    exact = df[(df["_மலர்_norm"] == malar_norm) & (df["_இதழ்_norm"] == ithal_norm)]
    if not exact.empty:
        logger.info(
            f"Exact CSV match: {len(exact)} rows for "
            f"மலர்={malar_norm}, இதழ்={ithal_norm}"
        )
        return exact

    def ithal_similar(csv_ithal):
        if ithal_norm in csv_ithal or csv_ithal in ithal_norm:
            return True
        return calculate_similarity(ithal_norm, csv_ithal) >= 80

    fuzzy = df[
        (df["_மலர்_norm"] == malar_norm) & (df["_இதழ்_norm"].apply(ithal_similar))
    ]
    if not fuzzy.empty:
        logger.info(
            f"Fuzzy CSV match: {len(fuzzy)} rows ('{ithal_norm}' ~ CSV இதழ் values)"
        )
        return fuzzy

    malar_only = df[df["_மலர்_norm"] == malar_norm]
    if not malar_only.empty:
        unique_ithal = list(malar_only["_இதழ்_norm"].unique())
        logger.warning(
            f"No இதழ் match. மலர்={malar_norm} exists with "
            f"இதழ் values: {unique_ithal}. Looking for: '{ithal_norm}'"
        )
    else:
        logger.warning(f"மலர்={malar_norm} not found in CSV at all")

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Topic / subtopic tree building
# ---------------------------------------------------------------------------


def build_topic_tree(matched_df):
    """Organise matched CSV rows into a parent→children structure.

    A row is a **main topic** when பெற்றோர்_வ.எ. is NaN/empty.
    A row is a **subtopic** when பெற்றோர்_வ.எ. holds the வ.எ. of its parent.

    Returns:
        list[dict]: ordered list of topic dicts:
            {
                "row_no": int,
                "title": str,
                "authors": list[str],
                "year": str,
                "subtopics": [
                    {"row_no": int, "title": str, "author": str}, ...
                ]
            }
    """
    # Build lookup: row_no → row
    row_lookup = {}
    for _, row in matched_df.iterrows():
        rno = normalize_csv_value(row.get("வ.எ.", ""))
        try:
            rno_int = int(float(rno))
        except (ValueError, TypeError):
            rno_int = None
        if rno_int is not None:
            row_lookup[rno_int] = row

    topics = []  # ordered main topics
    topic_index = {}  # row_no → position in topics list

    for _, row in matched_df.iterrows():
        if is_subtopic_row(row):
            continue  # handled below

        rno = normalize_csv_value(row.get("வ.எ.", ""))
        try:
            rno_int = int(float(rno))
        except (ValueError, TypeError):
            rno_int = None

        year_raw = row.get("ஆண்டு", "")
        year = (
            str(int(float(year_raw)))
            if not pd.isna(year_raw) and str(year_raw).strip()
            else "Unknown"
        )

        topic = {
            "row_no": rno_int,
            "title": str(row.get("தலைப்பு", "")).strip(),
            "authors": parse_author_field(row.get("ஆசிரியர்", "")),
            "year": year,
            "subtopics": [],
        }
        topic_index[rno_int] = len(topics)
        topics.append(topic)

    # Attach subtopics to their parents
    for _, row in matched_df.iterrows():
        if not is_subtopic_row(row):
            continue

        parent_no = get_parent_row_no(row)
        if parent_no is None or parent_no not in topic_index:
            logger.warning(
                f"Subtopic '{row.get('தலைப்பு', '')}' "
                f"has unknown parent {parent_no} — skipped"
            )
            continue

        subtopic = {
            "row_no": normalize_csv_value(row.get("வ.எ.", "")),
            "title": str(row.get("தலைப்பு", "")).strip(),
            # Plain author for subtopic (no brackets)
            "author": parse_author_field(row.get("ஆசிரியர்", "")),
        }
        topics[topic_index[parent_no]]["subtopics"].append(subtopic)

    logger.info(
        f"Topic tree: {len(topics)} main topics, "
        f"{sum(len(t['subtopics']) for t in topics)} subtopics"
    )
    return topics


# ---------------------------------------------------------------------------
# Content extraction (fuzzy title search)
# ---------------------------------------------------------------------------


def find_title_in_document(lines, title, search_from=0):
    """Find the line index where *title* appears standalone in the document.

    Strategy:
    - Collect ALL candidate lines with >= 70 % similarity.
    - Prefer near-exact (>= 95 %) matches with shortest line length
      (standalone titles are short; TOC lines are long).
    - Fall back to best fuzzy match.

    Returns:
        int: line index of best match, or -1 if not found.
    """
    title_clean = remove_symbols(title)
    if not title_clean:
        return -1

    candidates = []
    for i in range(search_from, len(lines)):
        stripped = lines[i].strip()
        line_clean = remove_symbols(stripped)
        if not line_clean:
            continue
        sim = calculate_similarity(title_clean, line_clean)
        if sim >= 70:
            candidates.append((sim, i, len(stripped)))

    if not candidates:
        logger.debug(f"Title not found: '{title[:50]}'")
        return -1

    near_exact = [c for c in candidates if c[0] >= 95]
    if near_exact:
        best = min(near_exact, key=lambda c: c[2])
    else:
        best = max(candidates, key=lambda c: (c[0], -c[2]))

    logger.debug(
        f"Title '{title[:40]}' → line {best[1]} " f"(sim={best[0]:.1f}%, len={best[2]})"
    )
    return best[1]


def extract_content_between(lines, start_line, end_line):
    """Return stripped content text from lines[start_line:end_line].

    Removes leading/trailing blank lines.
    """
    content_lines = [i.rstrip() for i in lines[start_line:end_line]]
    # Strip leading blanks
    while content_lines and not content_lines[0].strip():
        content_lines.pop(0)
    # Strip trailing blanks
    while content_lines and not content_lines[-1].strip():
        content_lines.pop()
    return "\n".join(content_lines)


# ---------------------------------------------------------------------------
# Main subtopic content extractor
# ---------------------------------------------------------------------------


def extract_subtopic_contents(lines, subtopics):
    """Extract content for each subtopic.

    Boundary rule:
      - A subtopic's content starts right after its title line.
      - It ends when the next subtopic's title line is found,
        OR when the next MAIN topic's title line is found,
        whichever comes first.

    Args:
        lines (list[str]): full document lines
        subtopics (list[dict]): subtopic dicts with at least {"title": str}
            (may also have pre-located "title_line" from caller)

    Returns:
        list[dict]: same list with "content" and "title_line" added.
    """
    if not subtopics:
        return subtopics

    # Locate each subtopic title in the document (sequential search)
    search_from = 0
    for st in subtopics:
        idx = find_title_in_document(lines, st["title"], search_from=search_from)
        st["title_line"] = idx
        if idx != -1:
            search_from = idx + 1

    # Build content using next boundary
    for i, st in enumerate(subtopics):
        start = st.get("title_line", -1)
        if start == -1:
            st["content"] = ""
            continue

        # Find end: next subtopic title OR document end
        end = len(lines)
        for j in range(i + 1, len(subtopics)):
            next_line = subtopics[j].get("title_line", -1)
            if next_line != -1 and next_line > start:
                end = next_line
                break

        # Content starts AFTER the title line
        st["content"] = extract_content_between(lines, start + 1, end)
        logger.debug(
            f"Subtopic '{st['title'][:30]}' "
            f"lines {start+1}–{end}: "
            f"{len(st['content'])} chars"
        )

    return subtopics


# ---------------------------------------------------------------------------
# Main-topic content extractor (standalone articles without subtopics)
# ---------------------------------------------------------------------------


def extract_main_topic_content(lines, topic, next_topic_line):
    """Extract content for a standalone main topic (no subtopics).

    Content begins after the title line and ends at next_topic_line.

    Args:
        lines (list[str]): document lines
        topic (dict): topic dict with "title_line" set
        next_topic_line (int): line index of the next topic (or len(lines))

    Returns:
        str: extracted content
    """
    start = topic.get("title_line", -1)
    if start == -1:
        return ""
    return extract_content_between(lines, start + 1, next_topic_line)


# ---------------------------------------------------------------------------
# Top-level extraction entry point
# ---------------------------------------------------------------------------


def extract_articles_from_csv(lines, csv_df, file_path):
    """Extract articles (with subtopic support) using CSV metadata.

    Flow:
    1. Extract மலர்/இதழ் from filename (fallback: document text).
    2. Match CSV rows for this volume/issue.
    3. Build parent→children topic tree.
    4. Locate each topic title in the document.
    5. For topics WITH subtopics → extract per-subtopic content.
       For topics WITHOUT subtopics → extract content up to next topic.
    6. Return structured result.

    JSON shape per article
    ----------------------
    {
      "doc_id": "1",
      "doc_issue": "6",
      "article_no": 2,
      "title": "வளரும் இலக்கியம்",
      "author_name": [],
      "year": "1947",
      "source_document": "VOL1-6-1947.txt",
      "subtopics": [
        {
          "subtopic_no": 1,
          "title": "அழகு",
          "author_name": ["தோழர் அறிவழகன்"],
          "content": "..."
        }
      ]
    }

    Standalone article (no subtopics) additionally has:
      "content": "..."
      "subtopics": []

    Args:
        lines (list[str]): full document lines
        csv_df (pd.DataFrame): loaded via load_csv()
        file_path: path-like with .name attribute

    Returns:
        dict or None
    """
    logger.info("=" * 80)
    logger.info("CSV EXTRACTION (topic-subtopic mode)")
    logger.info(f"File: {file_path.name}")
    logger.info("=" * 80)

    if csv_df is None:
        logger.error("CSV DataFrame is None")
        return None

    source_document = file_path.name

    # ----------------------------------------------------------------
    # STEP 1: மலர்/இதழ் from filename, fallback to text
    # ----------------------------------------------------------------
    malar, ithal, year_from_file = extract_malar_ithal_from_filename(source_document)

    if not malar or not ithal:
        logger.warning("Filename parse failed — trying document text")
        malar, ithal, year_from_file = extract_malar_ithal_from_text(lines)

    if not malar or not ithal:
        logger.error("Cannot find மலர்/இதழ். CSV extraction skipped.")
        return None

    logger.info(f"மலர்={malar}, இதழ்={ithal}, year={year_from_file}")

    malar_norm = normalize_csv_value(malar).strip()
    ithal_norm = normalize_csv_value(ithal).strip()

    # ----------------------------------------------------------------
    # STEP 2: Match CSV rows
    # ----------------------------------------------------------------
    matched_df = match_csv_rows(csv_df, malar, ithal)
    if matched_df.empty:
        logger.warning("No CSV rows matched — extraction skipped")
        return None

    logger.info(f"Matched {len(matched_df)} CSV rows")

    # ----------------------------------------------------------------
    # STEP 3: Build topic tree
    # ----------------------------------------------------------------
    topics = build_topic_tree(matched_df)

    # ----------------------------------------------------------------
    # STEP 4: Locate each main-topic title in the document
    # ----------------------------------------------------------------
    search_from = 0
    for topic in topics:
        idx = find_title_in_document(lines, topic["title"], search_from=search_from)
        topic["title_line"] = idx
        if idx != -1:
            search_from = idx + 1
            logger.debug(f"Main topic '{topic['title'][:40]}' at line {idx}")
        else:
            logger.warning(f"Main topic not found in doc: '{topic['title'][:40]}'")

    # ----------------------------------------------------------------
    # STEP 5: Extract content
    # ----------------------------------------------------------------
    articles = []
    article_no = 1
    extracted_line_ranges = []

    for t_idx, topic in enumerate(topics):
        title_line = topic.get("title_line", -1)

        # Determine where the NEXT main topic starts
        next_topic_line = len(lines)
        for future in topics[t_idx + 1 :]:
            fl = future.get("title_line", -1)
            if fl != -1:
                next_topic_line = fl
                break

        year = topic.get("year") or year_from_file or "Unknown"

        if topic["subtopics"]:
            # ── Topics WITH subtopics ───────────────────────────────
            # Restrict subtopic search to the region of this main topic
            region_lines = lines[
                (title_line if title_line != -1 else 0) : next_topic_line
            ]
            region_offset = title_line if title_line != -1 else 0

            subtopics = topic["subtopics"]
            subtopics = extract_subtopic_contents(region_lines, subtopics)

            # Adjust title_line back to global indices
            for st in subtopics:
                if st.get("title_line", -1) != -1:
                    st["title_line"] += region_offset

            built_subtopics = []
            st_no = 1
            for st in subtopics:
                # author field from subtopic: list from parse_author_field
                author_val = st.get("author", [])
                if isinstance(author_val, list):
                    author_names = author_val
                else:
                    author_names = [str(author_val)] if author_val else []

                built_subtopics.append(
                    {
                        "subtopic_no": st_no,
                        "title": st["title"],
                        "author_name": author_names,
                        "content": st.get("content", ""),
                    }
                )
                st_no += 1

            article = {
                "doc_id": malar_norm,
                "doc_issue": ithal_norm,
                "article_no": article_no,
                "title": topic["title"],
                "author_name": topic["authors"],
                "year": year,
                "source_document": source_document,
                "subtopics": built_subtopics,
            }

            if title_line != -1:
                extracted_line_ranges.append((title_line, next_topic_line))

            logger.info(
                f"[{article_no}] '{topic['title'][:40]}' "
                f"— {len(built_subtopics)} subtopics"
            )

        else:
            # ── Standalone topic (no subtopics) ────────────────────
            content = (
                extract_main_topic_content(lines, topic, next_topic_line)
                if title_line != -1
                else ""
            )

            article = {
                "doc_id": malar_norm,
                "doc_issue": ithal_norm,
                "article_no": article_no,
                "title": topic["title"],
                "author_name": topic["authors"],
                "content": content,
                "year": year,
                "source_document": source_document,
                "subtopics": [],
            }

            if title_line != -1:
                extracted_line_ranges.append((title_line, next_topic_line))

            logger.info(
                f"[{article_no}] '{topic['title'][:40]}' "
                f"— standalone ({len(content)} chars)"
            )

        articles.append(article)
        article_no += 1

    # ----------------------------------------------------------------
    # Build authors_list (unique authors across all rows)
    # ----------------------------------------------------------------
    authors_list = []
    seen_authors = set()
    for _, row in matched_df.iterrows():
        for name in parse_author_field(row.get("ஆசிரியர்", "")):
            if name and name not in seen_authors:
                seen_authors.add(name)
                authors_list.append(
                    {
                        "doc_id": malar_norm,
                        "doc_issue": ithal_norm,
                        "author_name": name,
                    }
                )

    logger.info(
        f"CSV extraction complete: {len(articles)} articles "
        f"({sum(len(a['subtopics']) for a in articles)} subtopics total)"
    )

    return {
        "articles": articles,
        "authors_list": authors_list,
        "doc_id": malar_norm,
        "doc_issue": ithal_norm,
        "extracted_line_ranges": extracted_line_ranges,
    }
