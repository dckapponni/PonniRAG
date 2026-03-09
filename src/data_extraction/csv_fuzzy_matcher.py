# csv_fuzzy_matcher.py

import re
import logging
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger('TamilDocProcessor.csv_fuzzy_matcher')


def calculate_similarity(str1, str2):
    """
    Calculate similarity percentage between two strings.

    Args:
        str1 (str): First string
        str2 (str): Second string

    Returns:
        float: Similarity 0-100
    """
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio() * 100


def remove_symbols(text):
    """
    Remove punctuation and symbols, keep Unicode letters and spaces.

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_csv_value(val):
    """
    Normalize CSV values for consistent comparison.
    Converts "1.0" -> "1", strips whitespace.

    Args:
        val: CSV cell value

    Returns:
        str: Normalized string
    """
    if pd.isna(val):
        return ''
    val_str = str(val).strip()
    if '.' in val_str:
        try:
            float_val = float(val_str)
            if float_val == int(float_val):
                return str(int(float_val))
        except:
            pass
    return val_str


def extract_malar_ithal_from_filename(filename):
    """
    Extract மலர் and இதழ் from filename.

    Handles:
    - VOL5-7-1951
    - VOL5 - 7 - 1951
    - VOL1-PONGAL-1948  (text இதழ்)
    - VOL_5_7_1951
    - 5-7-1951

    Args:
        filename (str): filename with or without extension

    Returns:
        tuple: (malar, ithal, year) as strings, or (None, None, None)
    """
    try:
        name = Path(filename).stem
        logger.info(f"Extracting மலர்/இதழ் from filename: {name}")

        # Pattern 1: VOL5-7-1951 (numeric இதழ்)
        vol_match = re.search(
            r'VOL\s*[-_]?\s*(\d+)\s*[-_]\s*(\d+)\s*[-_]\s*(\d{4})',
            name, re.IGNORECASE
        )
        if vol_match:
            malar = str(int(vol_match.group(1)))
            ithal = str(int(vol_match.group(2)))
            year  = vol_match.group(3)
            logger.info(f"VOL pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        # Pattern 2: VOL1-PONGAL-1948 (text இதழ்)
        vol_text_match = re.search(
            r'VOL\s*[-_]?\s*(\d+)\s*[-_]\s*([A-Za-z]+)\s*[-_]\s*(\d{4})',
            name, re.IGNORECASE
        )
        if vol_text_match:
            malar      = str(int(vol_text_match.group(1)))
            ithal_text = vol_text_match.group(2).upper()
            year       = vol_text_match.group(3)

            ITHAL_MAP = {
                'PONGAL':    'பொங்கல் மலர்',
                'DEEPAVALI': 'தீபாவளி மலர்',
                'SPECIAL':   'சிறப்பு மலர்',
                'ANNUAL':    'ஆண்டு மலர்',
            }
            ithal = ITHAL_MAP.get(ithal_text, ithal_text)
            logger.info(f"VOL-TEXT pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        # Pattern 3: vol_X_issue_Y_YYYY
        vol_issue_match = re.search(
            r'vol[-_\s]*(\d+)[-_\s]*(?:issue)?[-_\s]*(\d+)[-_\s]*(\d{4})?',
            name, re.IGNORECASE
        )
        if vol_issue_match:
            malar = str(int(vol_issue_match.group(1)))
            ithal = str(int(vol_issue_match.group(2)))
            year  = vol_issue_match.group(3) if vol_issue_match.group(3) else None
            logger.info(f"vol_issue pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        # Pattern 4: plain X-Y-YYYY
        plain_match = re.search(
            r'(\d+)\s*[-_]\s*(\d+)\s*[-_]\s*(\d{4})',
            name
        )
        if plain_match:
            malar = str(int(plain_match.group(1)))
            ithal = str(int(plain_match.group(2)))
            year  = plain_match.group(3)
            logger.info(f"Plain pattern: மலர்={malar}, இதழ்={ithal}, year={year}")
            return malar, ithal, year

        logger.warning(f"Could not extract மலர்/இதழ் from filename: {name}")
        return None, None, None

    except Exception as e:
        logger.error(f"Error extracting from filename {filename}: {e}")
        return None, None, None


def extract_malar_ithal_from_text(lines):
    """
    Extract மலர் and இதழ் from document text as fallback.
    Used only when filename parsing fails.

    Args:
        lines (list): document lines

    Returns:
        tuple: (malar, ithal, year) or (None, None, None)
    """
    try:
        search_lines = lines[:50]
        text = '\n'.join(search_lines)

        malar = None
        ithal = None
        year  = None

        # Check for பொங்கல் மலர் (special issue)
        pongal_match = re.search(r'பொங்கல்\s*மலர்', text)
        if pongal_match:
            malar = 'பொங்கல்'
            ithal = 'பொங்கல் மலர்'
            logger.info("Found பொங்கல் மலர் in text")

        # Extract year
        year_match = re.search(r'(19|20)\d{2}', text)
        if year_match:
            year = year_match.group(0)

        # Extract numeric மலர்
        if not malar:
            malar_match = re.search(r'மலர்\s*[:—\-]?\s*(\d+)', text)
            if malar_match:
                malar = str(int(malar_match.group(1)))

        # Extract இதழ்
        if not ithal:
            ithal_match = re.search(r'இதழ்\s*[:—\-]?\s*(\d+)', text)
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


def match_csv_rows(csv_df, malar, ithal):
    """
    Match CSV rows by மலர் (exact) and இதழ் (fuzzy + contains).

    Handles:
    - "பொங்கல் மலர்" vs "பொங்கல்"
    - "பொங்கல் மலர் " (trailing space)
    - numeric: "7" vs "7.0"

    Args:
        csv_df (pd.DataFrame): CSV dataframe
        malar (str): மலர் value
        ithal (str): இதழ் value

    Returns:
        pd.DataFrame: matched rows
    """
    malar_norm = normalize_csv_value(malar).strip()
    ithal_norm = normalize_csv_value(ithal).strip()

    csv_df = csv_df.copy()
    csv_df['மலர்_norm'] = csv_df['மலர்'].apply(
        lambda x: normalize_csv_value(x).strip()
    )
    csv_df['இதழ்_norm'] = csv_df['இதழ்'].apply(
        lambda x: normalize_csv_value(x).strip()
    )

    # Exact match first
    exact = csv_df[
        (csv_df['மலர்_norm'] == malar_norm) &
        (csv_df['இதழ்_norm'] == ithal_norm)
    ]
    if not exact.empty:
        logger.info(f"Exact CSV match: {len(exact)} rows for "
                    f"மலர்={malar_norm}, இதழ்={ithal_norm}")
        return exact

    # Fuzzy match on இதழ் — handles பொங்கல் vs பொங்கல் மலர்
    def ithal_similar(csv_ithal):
        if ithal_norm in csv_ithal or csv_ithal in ithal_norm:
            return True
        return calculate_similarity(ithal_norm, csv_ithal) >= 80

    fuzzy = csv_df[
        (csv_df['மலர்_norm'] == malar_norm) &
        (csv_df['இதழ்_norm'].apply(ithal_similar))
    ]
    if not fuzzy.empty:
        logger.info(
            f"Fuzzy CSV match: {len(fuzzy)} rows "
            f"('{ithal_norm}' ~ CSV இதழ் values)"
        )
        return fuzzy

    # Debug: show what the CSV actually has for this மலர்
    malar_only = csv_df[csv_df['மலர்_norm'] == malar_norm]
    if not malar_only.empty:
        unique_ithal = list(malar_only['இதழ்_norm'].unique())
        logger.warning(
            f"No இதழ் match. மலர்={malar_norm} exists in CSV with "
            f"இதழ் values: {unique_ithal}. Looking for: '{ithal_norm}'"
        )
    else:
        logger.warning(f"மலர்={malar_norm} not found in CSV at all")

    return pd.DataFrame()


def find_article_boundary_fuzzy(lines, title, next_title=None):
    """
    Find article content by fuzzy matching title in FULL document.

    KEY RULE:
    A title appears in TWO places in the document:
      1. Inside TOC  — title is embedded in a longer line
                       with author name + page number
                       e.g. "அணியணியாக வாரீர்! நாரா நாச்சியப்பன் 3"
      2. Outside TOC — title appears ALONE on its own line
                       e.g. "அணியணியாக வாரீர்!"

    We always want the OUTSIDE-TOC occurrence — the standalone line.

    Strategy:
    - Collect ALL candidate lines that match >= 70% similarity
    - Among candidates, PREFER lines where the cleaned line text
      matches the cleaned title at >= 95% (near-exact / standalone)
    - If no near-exact match, fall back to best fuzzy match

    This naturally picks the standalone occurrence outside TOC
    because TOC lines are longer (title + author + page) and
    therefore have lower similarity to just the title.

    Args:
        lines (list): ALL document lines (full document, including TOC)
        title (str): article title from CSV
        next_title (str): next article title for boundary detection

    Returns:
        tuple: (content, start_line, end_line)
               content is None if not found
    """
    try:
        title_clean = remove_symbols(title)

        if not title_clean:
            logger.warning(f"Empty title after cleaning: '{title}'")
            return None, -1, -1

        # --- PASS 1: collect all candidates ---
        # candidate = (similarity, line_index, line_length)
        candidates = []

        for i, line in enumerate(lines):
            stripped   = line.strip()
            line_clean = remove_symbols(stripped)

            if not line_clean:
                continue

            similarity = calculate_similarity(title_clean, line_clean)

            if similarity >= 70:
                candidates.append((similarity, i, len(stripped)))
                logger.debug(
                    f"Candidate line {i}: sim={similarity:.1f}% "
                    f"len={len(stripped)} '{stripped[:60]}'"
                )

        if not candidates:
            logger.warning(
                f"Title not found (no candidates): '{title[:50]}'"
            )
            return None, -1, -1

        # --- PASS 2: prefer standalone (near-exact) match ---
        # Near-exact = similarity >= 95%
        # These are lines that contain ONLY the title text,
        # i.e., the actual article heading outside TOC.
        # TOC lines are longer (title + author + page number)
        # so they score lower similarity.

        near_exact = [c for c in candidates if c[0] >= 95]

        if near_exact:
            # Among near-exact matches, pick the one with
            # shortest line (most standalone = least extra text)
            best = min(near_exact, key=lambda c: c[2])
            logger.info(
                f"Title '{title[:50]}' → standalone match "
                f"at line {best[1]} "
                f"(sim={best[0]:.1f}%, len={best[2]})"
            )
            start_line = best[1]
        else:
            # No near-exact — fall back to highest similarity,
            # preferring shorter lines as tiebreak
            best = max(candidates, key=lambda c: (c[0], -c[2]))
            logger.info(
                f"Title '{title[:50]}' → fuzzy match "
                f"at line {best[1]} "
                f"(sim={best[0]:.1f}%, len={best[2]})"
            )
            start_line = best[1]

        # --- Find end boundary using next title ---
        end_line = len(lines)

        if next_title:
            next_title_clean = remove_symbols(next_title)
            next_candidates  = []

            for i in range(start_line + 1, len(lines)):
                stripped   = lines[i].strip()
                line_clean = remove_symbols(stripped)

                if not line_clean:
                    continue

                similarity = calculate_similarity(next_title_clean, line_clean)
                if similarity >= 70:
                    next_candidates.append((similarity, i, len(stripped)))

            if next_candidates:
                # Same rule: prefer standalone (near-exact) for next title
                near_exact_next = [c for c in next_candidates if c[0] >= 95]
                if near_exact_next:
                    next_best = min(near_exact_next, key=lambda c: c[2])
                else:
                    next_best = max(next_candidates, key=lambda c: (c[0], -c[2]))

                if next_best[0] >= 80:
                    end_line = next_best[1]
                    logger.debug(
                        f"Next title boundary at line {end_line} "
                        f"(sim={next_best[0]:.1f}%)"
                    )

        # Extract content
        content_lines = list(lines[start_line:end_line])

        # Remove trailing blank lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()

        content = '\n'.join(content_lines).strip()
        return content, start_line, end_line

    except Exception as e:
        logger.error(
            f"Error in find_article_boundary_fuzzy "
            f"for '{title[:40]}': {e}"
        )
        return None, -1, -1


def extract_articles_from_csv(lines, csv_df, file_path):
    """
    Extract articles using filename-based மலர்/இதழ் matching.

    CSV logic is INDEPENDENT of TOC section.
    பொருளடக்கம் and ஆகியோரின் are NOT used here.
    TOC is only used by pattern extraction in main.py.

    Title search covers the FULL document including TOC region.
    The find_article_boundary_fuzzy function automatically picks
    the standalone occurrence (outside TOC) because:
    - TOC lines contain title + author + page → lower similarity
    - Standalone lines contain only the title → near-exact match

    Flow:
    1. Extract மலர்/இதழ் from filename
    2. If filename fails → try document text
    3. Match CSV rows (exact then fuzzy இதழ் matching)
    4. For each CSV title → find standalone title in full document
    5. Return articles + extracted line ranges

    All metadata (doc_id, doc_issue, author, title, year) from CSV.
    Only content is extracted from document text.

    Args:
        lines (list): document lines (full document)
        csv_df (pd.DataFrame): CSV with columns மலர், இதழ், தலைப்பு,
                               ஆசிரியர், ஆண்டு
        file_path (Path-like): file path with .name attribute

    Returns:
        dict or None
    """
    logger.info("=" * 80)
    logger.info("CSV EXTRACTION")
    logger.info(f"File: {file_path.name}")
    logger.info("NOTE: Searches FULL document. Standalone title")
    logger.info("      occurrence preferred over TOC occurrence.")
    logger.info("=" * 80)

    if csv_df is None:
        logger.error("CSV DataFrame is None")
        return None

    source_document = file_path.name

    # ----------------------------------------------------------------
    # STEP 1: Extract மலர்/இதழ் from FILENAME
    # ----------------------------------------------------------------
    malar, ithal, year_from_file = extract_malar_ithal_from_filename(
        source_document
    )

    # ----------------------------------------------------------------
    # STEP 1b: Fallback to document TEXT if filename fails
    # ----------------------------------------------------------------
    if not malar or not ithal:
        logger.warning(
            "Filename parse failed — trying document text for மலர்/இதழ்"
        )
        malar, ithal, year_from_file = extract_malar_ithal_from_text(lines)

    if not malar or not ithal:
        logger.error(
            "Cannot find மலர்/இதழ் from filename or text. "
            "CSV extraction skipped."
        )
        return None

    logger.info(f"மலர்={malar}, இதழ்={ithal}, year={year_from_file}")

    # ----------------------------------------------------------------
    # STEP 2: Match CSV rows (exact then fuzzy)
    # ----------------------------------------------------------------
    matched_articles = match_csv_rows(csv_df, malar, ithal)

    if matched_articles.empty:
        return None

    malar_norm = normalize_csv_value(malar).strip()
    ithal_norm = normalize_csv_value(ithal).strip()

    logger.info(f"Found {len(matched_articles)} CSV rows to extract")

    # ----------------------------------------------------------------
    # STEP 3: Extract content for each CSV title
    # Search FULL document — standalone title preferred over TOC
    # ----------------------------------------------------------------
    articles              = []
    article_no            = 1
    extracted_line_ranges = []

    for idx_pos, (idx, row) in enumerate(matched_articles.iterrows()):

        # All metadata from CSV
        title  = str(row['தலைப்பு']) if pd.notna(row['தலைப்பு']) else ""
        author = str(row['ஆசிரியர்']) if pd.notna(row['ஆசிரியர்']) else "NA"
        year   = (
            str(int(row['ஆண்டு']))
            if pd.notna(row['ஆண்டு'])
            else (year_from_file or "Unknown")
        )

        if not title:
            logger.warning(f"Empty title at CSV row {idx}, skipping")
            continue

        # Get next title for boundary detection
        next_title = None
        if idx_pos + 1 < len(matched_articles):
            next_row   = matched_articles.iloc[idx_pos + 1]
            next_title = (
                str(next_row['தலைப்பு'])
                if pd.notna(next_row['தலைப்பு'])
                else None
            )

        logger.info(
            f"[{idx_pos + 1}/{len(matched_articles)}] "
            f"Searching: '{title[:50]}'"
        )

        # Search FULL document — standalone title auto-preferred
        content, start_line, end_line = find_article_boundary_fuzzy(
            lines, title, next_title
        )

        if content and len(content) > 100:
            articles.append({
                "doc_id":          malar_norm,
                "doc_issue":       ithal_norm,
                "article_no":      article_no,
                "author_name":     author,
                "title":           title,
                "content":         content,
                "year":            year,
                "source_document": source_document
            })
            article_no += 1

            if start_line != -1 and end_line != -1:
                extracted_line_ranges.append((start_line, end_line))
                logger.info(
                    f"  ✓ Extracted '{title[:40]}' "
                    f"lines {start_line}-{end_line} "
                    f"({len(content)} chars)"
                )
        else:
            logger.warning(
                f"  ✗ Could not extract '{title[:40]}' — "
                f"{'not found' if not content else f'only {len(content)} chars (too short)'}"
            )

    # Build authors list from CSV
    authors_list   = []
    unique_authors = set()
    for _, row in matched_articles.iterrows():
        if pd.notna(row['ஆசிரியர்']):
            author_name = str(row['ஆசிரியர்']).strip()
            if author_name not in ("NA", "", "nan") and \
               author_name not in unique_authors:
                unique_authors.add(author_name)
                authors_list.append({
                    "doc_id":      malar_norm,
                    "doc_issue":   ithal_norm,
                    "author_name": author_name
                })

    logger.info(
        f"CSV extraction complete: "
        f"{len(articles)}/{len(matched_articles)} articles extracted"
    )

    return {
        "articles":              articles,
        "authors_list":          authors_list,
        "doc_id":                malar_norm,
        "doc_issue":             ithal_norm,
        "extracted_line_ranges": extracted_line_ranges
    }