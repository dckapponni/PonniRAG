"""Content extraction utilities for Tamil document processing.

Provides low-level helpers for collecting article body text from
raw document line lists after article boundaries have been located
by the pattern extractors or the CSV fuzzy matcher:

- **Blank-line counting** — :func:`count_consecutive_blanks` measures
  separator runs used as section boundaries by all extractors.
- **Keyword lookahead** — :func:`check_keyword_ahead` detects
  upcoming intro section markers so content collection stops before
  the next editorial section begins.
- **Intro section extraction** — :func:`extract_intro_content_phase1`
  collects body text for named editorial sections (தலையங்கம்,
  காலமும் கருத்தும், etc.), handling embedded author attribution
  within the keyword line itself.
- **Remaining content extraction** — :func:`extract_remaining_content`
  provides a final safety-net pass that groups any unprocessed lines
  by blank-line separators and emits sections with valid headings.
"""

import logging

from doc_utils import count_content_lines
from shared_author_local import check_author_ahead  # CHANGED FROM shared_author
from text_processing import extract_author_from_line, is_valid_heading

logger = logging.getLogger("TamilDocProcessor.content_extraction")


def check_keyword_ahead(lines, current_idx, intro_keywords, lookback=2):
    """
    Check if an intro keyword appears in the next few lines.

    Searches forward from the current position to detect if any intro section keyword
    (like தலையங்கம், படைப்புகள், etc.) appears within a specified lookahead window.

    Args:
        lines (list): List of text lines from the document.
        current_idx (int): Current line index to start searching from.
        intro_keywords (list): List of intro section keywords to search for.
        lookback (int, optional): Number of lines to look ahead. Defaults to 2.

    Returns:
        int or None: Index of the line containing the keyword
            if found within lookahead window, None if no
            keyword found or error occurred.
    """
    try:
        for i in range(current_idx, min(current_idx + lookback + 1, len(lines))):
            line = lines[i].strip()
            for keyword in intro_keywords:
                if keyword in line:
                    logger.debug(f"Found keyword '{keyword}' ahead at line {i}")
                    return i
        return None

    except IndexError as e:
        logger.warning(f"Index error in check_keyword_ahead at line {current_idx}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in check_keyword_ahead at line {current_idx}: {e}")
        return None


def count_consecutive_blanks(lines, start_idx):
    """
    Count consecutive blank lines starting from a given index.

    Iterates forward from the starting index and counts how many consecutive lines
    are blank (contain only whitespace or are empty).

    Args:
        lines (list): List of text lines from the document.
        start_idx (int): Starting index to begin counting blank lines.

    Returns:
        int: Number of consecutive blank lines found. Returns 0 if no blank lines
            or if an error occurs.
    """
    try:
        count = 0
        i = start_idx
        while i < len(lines) and not lines[i].strip():
            count += 1
            i += 1

        if count > 0:
            logger.debug(f"Found {count} consecutive blank lines at line {start_idx}")

        return count

    except IndexError as e:
        logger.warning(
            f"Index error in count_consecutive_blanks at line {start_idx}: {e}"
        )
        return 0
    except Exception as e:
        logger.error(f"Error in count_consecutive_blanks at line {start_idx}: {e}")
        return 0


def extract_remaining_content(lines, start_idx, processed_lines):
    """
    Extract remaining unprocessed content using blank space logic.

    Groups unprocessed document content by blank line separators and extracts sections
    that have valid headings. Uses blank line count (3+ consecutive blanks) as primary
    section separator. Filters out sections ending with author dashes and validates
    content length.

    Args:
        lines (list): List of text lines from the document.
        start_idx (int): Starting index for extraction.
        processed_lines (list): Boolean list tracking which lines have been processed
                            (True = already processed, False = available).

    Returns:
        list: List of dictionaries, each containing:
            - heading (str): First line used as heading
            - author (str): Always "NA" (no author detection in this function)
            - content (str): Extracted content text
            - start_idx (int): Starting line index of this section
            - end_idx (int): Ending line index of this section

    Processing Logic:
        1. Skips already processed lines
        2. Groups content separated by 3+ blank lines
        3. Validates first line as heading (< 25 chars, valid format)
        4. Filters content with minimum 4 lines
        5. Marks processed lines to avoid re-extraction
    """
    logger.debug(f"Starting remaining content extraction from line {start_idx}")
    groups = []
    i = start_idx
    total_lines = len(lines)

    try:
        while i < total_lines:
            if processed_lines[i]:
                i += 1
                continue

            if not lines[i].strip():
                blank_count = count_consecutive_blanks(lines, i)
                if blank_count >= 3:
                    for j in range(i, min(i + blank_count, total_lines)):
                        processed_lines[j] = True
                    i += blank_count
                else:
                    i += 1
                continue

            group_start = i
            content_lines = []
            logger.debug(f"Starting new content group at line {group_start}")

            while i < total_lines:
                if processed_lines[i]:
                    break

                line = lines[i]
                stripped = line.strip()

                if not stripped:
                    blank_count = count_consecutive_blanks(lines, i)
                    if blank_count >= 3:
                        while content_lines and not content_lines[-1].strip():
                            content_lines.pop()
                        logger.debug(
                            f"Stopped at {blank_count} blank lines at line {i}"
                        )
                        break
                    else:
                        content_lines.append(line.rstrip())
                        i += 1
                else:
                    content_lines.append(line.rstrip())
                    i += 1

            if content_lines:
                last_line = content_lines[-1].strip() if content_lines else ""

                if last_line and (
                    last_line.startswith("—")
                    or last_line.startswith("-")
                    or last_line.startswith("–")
                ):
                    logger.debug(
                        f"Skipping group ending with author dash at line {group_start}"
                    )
                    for j in range(group_start, i):
                        if j < total_lines:
                            processed_lines[j] = True
                    continue

                first_line = content_lines[0].strip() if content_lines else ""

                if first_line and len(first_line) < 25 and is_valid_heading(first_line):
                    heading = first_line
                    remaining = content_lines[1:]

                    while remaining and not remaining[0].strip():
                        remaining.pop(0)

                    content = "\n".join(remaining)

                    if content.strip() and count_content_lines(content) >= 4:
                        groups.append(
                            {
                                "heading": heading,
                                "author": "NA",
                                "content": content,
                                "start_idx": group_start,
                                "end_idx": i,
                            }
                        )
                        logger.info(
                            "Remaining: Extracted content "
                            f"with heading '{heading}' "
                            f"({count_content_lines(content)} lines)"
                        )
                    else:
                        logger.debug(
                            "Remaining: Content too short "
                            f"({count_content_lines(content)} "
                            f"lines) at line {group_start}"
                        )
                else:
                    logger.debug(
                        "Remaining: No valid heading at "
                        f"line {group_start}, skipping group"
                    )

                for j in range(group_start, i):
                    if j < total_lines:
                        processed_lines[j] = True

            if i < total_lines and not lines[i].strip():
                blank_count = count_consecutive_blanks(lines, i)
                for j in range(i, min(i + blank_count, total_lines)):
                    if j < total_lines:
                        processed_lines[j] = True
                i += blank_count

        logger.info(f"Remaining content extraction: Found {len(groups)} sections")
        return groups

    except IndexError as e:
        logger.error(
            f"Index error in extract_remaining_content at line {i}: {e}", exc_info=True
        )
        return groups
    except Exception as e:
        logger.error(
            f"Unexpected error in extract_remaining_content at line {i}: {e}",
            exc_info=True,
        )
        return groups


def extract_intro_content_phase1(
    lines,
    keyword_idx,
    processed_lines,
    authors_normalized,
    authors_original,
    intro_keywords,
):
    """
    Extract content for intro sections identified by keywords.

    Handles detection of author names within keyword lines
    and proper content boundaries.

    Args:
        lines (list): List of text lines from document
        keyword_idx (int): Index of the intro keyword line
        processed_lines (list): Boolean list tracking which lines have been processed
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        intro_keywords (list): List of intro keywords for boundary detection

    Returns:
        tuple: (content_text, end_index, author_if_found)
            - content_text (str): Extracted content as string
            - end_index (int): Index where extraction stopped
            - author_if_found (str or None): Author name if found in keyword line
    """
    try:
        logger.debug(f"Extracting intro content starting at keyword line {keyword_idx}")

        content_lines = []
        i = keyword_idx + 1

        if i < len(lines) and not lines[i].strip():
            i += 1

        keyword_line = lines[keyword_idx].strip()
        has_author_in_keyword = extract_author_from_line(
            keyword_line, authors_normalized, authors_original
        )

        if has_author_in_keyword:
            logger.debug(f"Found author '{has_author_in_keyword}' in keyword line")

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            author_idx = check_author_ahead(
                lines, i, authors_normalized, authors_original, lookback=3
            )
            if author_idx is not None:
                stop_at = max(i, author_idx - 3)
                while len(content_lines) > stop_at - (keyword_idx + 1):
                    content_lines.pop()
                logger.debug(f"Stopped before author at line {author_idx}")
                break

            keyword_idx_found = check_keyword_ahead(
                lines, i, intro_keywords, lookback=2
            )
            if keyword_idx_found is not None:
                stop_at = max(i, keyword_idx_found - 2)
                while len(content_lines) > stop_at - (keyword_idx + 1):
                    content_lines.pop()
                logger.debug(f"Stopped before next keyword at line {keyword_idx_found}")
                break

            if not stripped:
                blank_count = count_consecutive_blanks(lines, i)
                if blank_count >= 4:
                    while content_lines and not content_lines[-1].strip():
                        content_lines.pop()
                    logger.debug(f"Stopped at {blank_count} blank lines at line {i}")
                    break
                else:
                    content_lines.append(line.rstrip())
            else:
                content_lines.append(line.rstrip())

            i += 1

        content = "\n".join(content_lines)
        logger.debug(
            "Intro content extracted: "
            f"{count_content_lines(content)} lines, "
            f"author={has_author_in_keyword}"
        )

        return (content, i, has_author_in_keyword)

    except IndexError as e:
        logger.error(
            f"Index error in extract_intro_content_phase1 at line {i}: {e}",
            exc_info=True,
        )
        return ("", keyword_idx + 1, None)
    except Exception as e:
        logger.error(
            "Unexpected error in "
            "extract_intro_content_phase1 at "
            f"keyword line {keyword_idx}: {e}",
            exc_info=True,
        )
        return ("", keyword_idx + 1, None)
