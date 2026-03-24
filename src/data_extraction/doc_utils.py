"""Document utility functions for Tamil document processing."""

import logging
import re

from text_processing import normalize_text

logger = logging.getLogger("TamilDocProcessor.doc_utils")

LINES_PER_PAGE = 50

SECTION_TYPES = {
    "கார்ட்டூன்",
    "தலையங்கம்",
    "விமரிசனம்",
    "கடிதங்கள்",
    "சிறுவர் அரங்கம்",
    "பொதுமேடை",
    "கலையுலகம்",
    "சொல்லாராய்ச்சி",
    "வளரும் இலக்கியம்",
    "சிறுவர்",
    ",,",
    "...",
}

SECTION_HEADERS = {
    "காலமும் கருத்தும்",
    "கருத்தும் காலமும்",
    "பொது மேடை",
    "வளரும் இலக்கியம்",
    "உங்களுக்குத் தெரியுமா?",
    "சொல்லாராய்ச்சி அரங்கு",
    "பொதுமேடை",
}


def extract_doc_info(lines):
    """Extract document ID and issue number from the document.

    Searches both header and footer areas for malar and ithal markers.

    Args:
        lines (list): List of text lines from document

    Returns:
        tuple: (doc_id, doc_issue) as strings, or ("NA", "NA") if not found
    """
    logger.debug("Extracting document info")
    doc_id = "NA"
    doc_issue = "NA"

    search_areas = [
        range(0, min(LINES_PER_PAGE * 3, len(lines))),
        range(max(0, len(lines) - LINES_PER_PAGE), len(lines)),
    ]

    try:
        for search_range in search_areas:
            for i in search_range:
                if i >= len(lines):
                    continue

                line_stripped = lines[i].strip()

                if not line_stripped:
                    continue

                if doc_id == "NA":
                    match = re.search(r"மலர்\s*[:—-]?\s*(\d{1,3})", line_stripped)
                    if match:
                        doc_id = match.group(1)
                        logger.debug(f"Found மலர் at line {i}: ID={doc_id}")

                if doc_issue == "NA":
                    match = re.search(r"இதழ்\s*[:—-]?\s*(\d{1,3})", line_stripped)
                    if match:
                        doc_issue = match.group(1)
                        logger.debug(f"Found இதழ் at line {i}: Issue={doc_issue}")

                if doc_id != "NA" and doc_issue != "NA":
                    logger.info(
                        f"Successfully extracted: மலர்={doc_id}, இதழ்={doc_issue}"
                    )
                    return (doc_id, doc_issue)

        if doc_id == "NA" or doc_issue == "NA":
            logger.warning(f"Incomplete doc info - மலர்={doc_id}, இதழ்={doc_issue}")

        return (doc_id, doc_issue)

    except Exception as e:
        logger.error(f"Error extracting doc info: {e}", exc_info=True)
        return ("NA", "NA")


def is_section_type(text):
    """
    Check if text is a section type (not an author).

    Args:
        text (str): Text to check

    Returns:
        bool: True if it's a section type, False otherwise
    """
    if not text:
        return False

    text_clean = text.strip()

    if text_clean in SECTION_TYPES:
        return True

    if re.match(r"^[.\s…]+$", text_clean):
        return True

    return False


def is_section_header(text):
    """
    Check if text is a known section header (title without author).

    Args:
        text (str): Text to check

    Returns:
        bool: True if it's a section header, False otherwise
    """
    if not text:
        return False

    text_clean = text.strip()

    if text_clean in SECTION_HEADERS:
        return True

    section_starters = [
        "காலமும்",
        "கருத்தும்",
        "வளரும்",
        "உங்களுக்கு",
        "சொல்லாராய்ச்சி",
    ]
    if any(text_clean.startswith(word) for word in section_starters):
        return True

    return False


def is_valid_author_name(text):
    """Validate if text could be an author name.

    Applies strict validation to avoid false positives
    including section types and headers.

    Args:
        text (str): Text fragment to check

    Returns:
        bool: True if valid author name, False otherwise
    """
    if not text or len(text) < 2:
        return False

    text_clean = text.strip()

    if is_section_type(text_clean):
        logger.debug(f"Rejected (section type): '{text_clean}'")
        return False

    if is_section_header(text_clean):
        logger.debug(f"Rejected (section header): '{text_clean}'")
        return False

    if re.match(r"^[.\s…,]+$", text_clean):
        return False

    if re.match(r"^\d+[,\-]?\s*\d*[ம்]?$", text_clean):
        return False

    if len(text_clean) > 40:
        logger.debug(f"Rejected (too long): '{text_clean}'")
        return False

    if not re.search(r"[\u0B80-\u0BFF]", text_clean):
        return False

    dot_count = text_clean.count(".")
    comma_count = text_clean.count(",")
    ellipsis_count = text_clean.count("…")

    if dot_count > 3 or comma_count > 2 or ellipsis_count > 0:
        logger.debug(f"Rejected (too much punctuation): '{text_clean}'")
        return False

    if re.search(r"[a-zA-Z]{3,}", text_clean):
        logger.debug(f"Rejected (contains English): '{text_clean}'")
        return False

    return True


def parse_toc_line_robust(line):
    """
    Parse TOC line handling multiple formats and edge cases.

    Handles:
    - Ellipsis (…) as placeholder for no author
    - Section types (கார்ட்டூன், தலையங்கம்)
    - Section headers (காலமும் கருத்தும்)
    - Variable spacing (2+ spaces, 3+ spaces, tabs)
    - Names with initials (ரா. நீலமேகம்)

    Args:
        line (str): TOC line to parse

    Returns:
        dict or None: Dictionary with 'title', 'author', 'page' keys, or None if invalid
    """
    try:
        line_original = line
        line = line.strip()

        if not line or len(line) < 3:
            return None

        if re.match(r"^[.\s…]+$", line):
            return None

        page_match = re.search(r"[.\s…]*(\d{1,3})\s*$", line)

        if not page_match:
            return None

        page_no = page_match.group(1)

        content_before_page = line_original[: page_match.start()]

        content_before_page = re.sub(r"[.\s…]+$", "", content_before_page)

        if not content_before_page.strip():
            return None

        parts = None

        parts_extreme = re.split(r"(?:\s{4,}|\t+)", content_before_page)
        if len(parts_extreme) >= 2:
            parts = parts_extreme
            logger.debug(f"Using 4+ space split: {len(parts)} parts")

        if not parts or len(parts) < 2:
            parts_very_wide = re.split(r"(?:\s{3,}|\t+)", content_before_page)
            if len(parts_very_wide) >= 2:
                parts = parts_very_wide
                logger.debug(f"Using 3+ space split: {len(parts)} parts")

        if not parts or len(parts) < 2:
            parts_wide = re.split(r"(?:\s{2,}|\t+)", content_before_page)
            if len(parts_wide) >= 2:
                parts = parts_wide
                logger.debug(f"Using 2+ space split: {len(parts)} parts")

        if not parts or len(parts) < 2:
            words = content_before_page.split()
            if len(words) >= 3:
                if is_valid_author_name(words[-1]) and not is_section_type(words[-1]):
                    if len(words) >= 2 and re.match(r"^[\u0B80-\u0BFF]\.$", words[-2]):
                        parts = [" ".join(words[:-2]), " ".join(words[-2:])]
                        logger.debug(f"Single-space split with initial: {parts[-1]}")
                    else:
                        parts = [" ".join(words[:-1]), words[-1]]
                        logger.debug(f"Single-space split: {parts[-1]}")
                else:
                    parts = [content_before_page]
            else:
                parts = [content_before_page]

        if not parts:
            parts = [content_before_page]

        segments = []
        for part in parts:
            cleaned = part.strip()
            cleaned = re.sub(r"^[.\s…]+", "", cleaned)
            cleaned = re.sub(r"[.\s…]+$", "", cleaned)
            cleaned = cleaned.strip()

            if cleaned and not re.match(r"^[.,…\s]+$", cleaned):
                segments.append(cleaned)

        if len(segments) == 0:
            return None

        if len(segments) == 1:
            title = segments[0]
            if is_section_header(title):
                logger.debug(f"Title is section header: '{title}' (no author expected)")
            return {"title": title, "author": None, "page": page_no}

        potential_author = segments[-1]
        title_parts = segments[:-1]

        if is_valid_author_name(potential_author) and not is_section_type(
            potential_author
        ):
            title = " ".join(title_parts).strip()
            logger.info(
                f"Parsed - Title: '{title[:35]}...', "
                f"Author: '{potential_author}', "
                f"Page: {page_no}"
            )
            return {"title": title, "author": potential_author, "page": page_no}
        else:
            title = " ".join(segments).strip()
            if is_section_type(potential_author):
                logger.debug(
                    "Last segment is section type "
                    f"'{potential_author}', "
                    "treating as title-only"
                )
            return {"title": title, "author": None, "page": page_no}

    except Exception as e:
        logger.warning(f"Error parsing TOC line '{line[:50]}...': {e}")
        return None


def find_toc_boundaries(lines):
    """
    Find TOC start and end boundaries with multiple detection methods.

    Args:
        lines (list): List of text lines

    Returns:
        tuple: (toc_start, toc_end) or (-1, -1) if not found
    """
    toc_start = -1
    toc_end = -1

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if "பொருளடக்கம்" in line_stripped:
            toc_start = i + 1
            logger.debug(f"Found பொருளடக்கம் at line {i}")

            for j in range(i + 1, min(i + 100, len(lines))):
                check_line = lines[j].strip()

                if "ஆகியோரின்" in check_line or "எழுத்தோவியங்கள்" in check_line:
                    toc_end = j
                    logger.info(f"Found end marker at line {j}")
                    break

                if j > i + 5:  # ← THIS IS THE KEY CONDITION!
                    if re.match(r"^மலர்\s+\d+", check_line) or re.match(
                        r"^விலை\s+\d+", check_line
                    ):
                        toc_end = j
                        logger.info(f"Found section boundary at line {j}")
                        break

            if toc_end == -1:
                toc_end = min(i + 41, len(lines))
                logger.info(
                    "No end marker, using 40-line limit: "
                    f"lines {toc_start} to {toc_end}"
                )

            break

    return (toc_start, toc_end)


def extract_authors_from_toc(lines):
    """
    Extract authors from TOC with comprehensive format handling.

    Args:
        lines (list): List of text lines from document

    Returns:
        tuple: (doc_id, doc_issue, authors_original,
            authors_normalized, title_author_pairs)
    """
    logger.debug("Extracting authors from TOC (ROBUST VERSION)")

    try:
        doc_id, doc_issue = extract_doc_info(lines)

        authors_set = set()
        authors_original = []
        authors_normalized = []
        title_author_pairs = []

        toc_start, toc_end = find_toc_boundaries(lines)

        if toc_start == -1:
            logger.warning("not found in document")
            return (doc_id, doc_issue, [], [], [])

        logger.debug(f"Parsing TOC lines {toc_start} to {toc_end}")

        for k in range(toc_start, toc_end):
            if k >= len(lines):
                break

            line_content = lines[k]
            line_stripped = line_content.strip()

            if not line_stripped:
                continue

            if "ஆகியோரின்" in line_stripped:
                continue

            if re.match(r"^[.\s…]+$", line_stripped):
                continue

            entry = parse_toc_line_robust(line_content)

            if entry and entry["author"]:
                author = entry["author"]
                title = entry["title"]

                author_norm = normalize_text(author)

                if author_norm not in authors_set:
                    authors_set.add(author_norm)
                    authors_original.append(author)
                    authors_normalized.append(author_norm)
                    logger.info(f"Author: '{author}' (from: '{title[:30]}...')")

                title_author_pairs.append((title, author))
            else:
                if entry:
                    logger.debug(f"No author: '{entry['title'][:40]}...'")

        logger.info(f"EXTRACTED {len(authors_original)} UNIQUE AUTHORS")

        return (
            doc_id,
            doc_issue,
            authors_original,
            authors_normalized,
            title_author_pairs,
        )

    except Exception as e:
        logger.error(f"Error extracting authors from TOC: {e}", exc_info=True)
        return ("NA", "NA", [], [], [])


def count_content_lines(text):
    """
    Count non-empty lines in text.

    Args:
        text (str): Text to count lines in

    Returns:
        int: Number of non-empty lines
    """
    if not text:
        return 0
    return len([line for line in text.split("\n") if line.strip()])


def get_shared_authors(doc_id, doc_issue, all_files_authors):
    """
    Retrieve shared authors for a document from the shared authors dictionary.

    Args:
        doc_id (str): Document ID
        doc_issue (str): Document issue number
        all_files_authors (dict): Dictionary mapping (doc_id, doc_issue) to authors

    Returns:
        tuple: (authors_original, authors_normalized) or ([], []) if not found
    """
    try:
        key = (doc_id, doc_issue)
        authors = all_files_authors.get(key, ([], []))

        if authors[0]:
            logger.debug(
                f"Retrieved {len(authors[0])} shared authors for {doc_id}/{doc_issue}"
            )

        return authors

    except Exception as e:
        logger.error(f"Error getting shared authors: {e}")
        return ([], [])


def extract_authors_alternative(lines):
    """Extract authors by detecting names after blank lines.

    Alternative method when TOC parsing fails.

    Args:
        lines (list): List of text lines from document

    Returns:
        tuple: (authors_original, authors_normalized)
    """
    logger.debug("Using alternative author extraction")
    authors_original = []
    authors_normalized = []
    authors_set = set()

    try:
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                blank_count = 0
                while i < len(lines) and not lines[i].strip():
                    blank_count += 1
                    i += 1

                if blank_count >= 2 and i < len(lines):
                    potential_author = lines[i].strip()

                    if potential_author and 3 <= len(potential_author) <= 30:
                        has_content = False
                        for offset in [1, 2, 3]:
                            check_idx = i - blank_count - offset
                            if check_idx >= 0 and len(lines[check_idx].strip()) > 30:
                                has_content = True
                                break

                        if has_content and is_valid_author_name(potential_author):
                            normalized = normalize_text(potential_author)
                            if normalized not in authors_set:
                                authors_original.append(potential_author)
                                authors_normalized.append(normalized)
                                authors_set.add(normalized)
                                logger.debug(f"Alternative: '{potential_author}'")
            i += 1

        logger.info(f"Alternative extraction: {len(authors_original)} authors")
        return (authors_original, authors_normalized)

    except Exception as e:
        logger.error(f"Error in alternative extraction: {e}", exc_info=True)
        return ([], [])
