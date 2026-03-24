"""Text processing utilities for Tamil document analysis."""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger("TamilDocProcessor.text_processing")


def normalize_text(text):
    """
    Normalize Tamil text by handling Unicode variations and removing punctuation.

    Args:
        text (str): Text to normalize

    Returns:
        str: Normalized text
    """
    try:
        if not text:
            return ""

        text = text.replace("ண", "ண").replace("णु", " णु")
        text = re.sub(r"\.", "", text)
        text = re.sub(r"\s+", "", text.strip().lower())

        return text

    except Exception as e:
        text_str = str(text) if text else ""
        text_preview = text_str[:50] if len(text_str) > 50 else text_str
        logger.warning(f"Error normalizing text '{text_preview}...': {e}")
        return str(text) if text else ""


def is_valid_heading(heading):
    """
    Validate if a line qualifies as a proper article heading.

    Args:
        heading (str): Heading text to validate

    Returns:
        bool: True if valid heading, False otherwise
    """
    try:
        if not heading or not heading.strip():
            return False

        stripped = heading.strip()

        if stripped.isdigit():
            return False

        if len(stripped) >= 25:
            return False

        if (
            stripped.startswith("—")
            or stripped.startswith("-")
            or stripped.startswith("–")
        ):
            return False

        return True

    except Exception as e:
        heading_str = str(heading) if heading else ""
        heading_preview = heading_str[:30] if len(heading_str) > 30 else heading_str
        logger.warning(f"Error validating heading '{heading_preview}...': {e}")
        return False


def fuzzy_match_author(line, authors_normalized, authors_original, threshold=0.8):
    """
    Match a line against known authors using fuzzy string matching.

    Args:
        line (str): Line to match against authors
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        threshold (float): Minimum similarity score (0.0 to 1.0)

    Returns:
        tuple: (matched_author_name or None, similarity_score)
    """
    try:
        if not line.strip():
            return (None, 0)

        line_normalized = normalize_text(line)
        best_match = None
        best_similarity = 0

        for i, author_norm in enumerate(authors_normalized):
            try:
                similarity = SequenceMatcher(None, line_normalized, author_norm).ratio()
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = authors_original[i]
            except Exception as e:
                logger.debug(f"Error comparing with author {i}: {e}")
                continue

        if best_match:
            logger.debug(
                f"Fuzzy matched '{line[:30]}...' to "
                f"'{best_match}' "
                f"(similarity: {best_similarity:.2f})"
            )

        return (best_match, best_similarity)

    except Exception as e:
        line_str = str(line) if line else ""
        line_preview = line_str[:50] if len(line_str) > 50 else line_str
        logger.warning(f"Error in fuzzy_match_author for line '{line_preview}...': {e}")
        return (None, 0)


def extract_author_from_line(line, authors_normalized, authors_original):
    """
    Extract author name from a line by cleaning and matching against known authors.

    Args:
        line (str): Line to extract author from
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names

    Returns:
        str or None: Matched author name, or None if no match found
    """
    try:
        clean_line = line.strip()

        if not clean_line:
            return None

        if (
            clean_line.startswith("—")
            or clean_line.startswith("-")
            or clean_line.startswith("–")
        ):
            clean_line = re.sub(r"^[—\-–]\s*", "", clean_line)

        if "[" in clean_line and "]" in clean_line:
            bracket_match = re.search(r"\[(.*?)\]", clean_line)
            if bracket_match:
                clean_line = bracket_match.group(1)

        if "ஆசிரியர்" in clean_line or ":" in clean_line:
            parts = clean_line.split(":")
            if len(parts) > 1:
                clean_line = parts[-1].strip()

        clean_line = re.sub(r'[,.\]"]+$', "", clean_line)
        clean_line = re.sub(r'^["]+', "", clean_line)

        matched_author, similarity = fuzzy_match_author(
            clean_line, authors_normalized, authors_original
        )

        return matched_author

    except Exception as e:
        line_str = str(line) if line else ""
        line_preview = line_str[:50] if len(line_str) > 50 else line_str
        logger.warning(f"Error extracting author from line '{line_preview}...': {e}")
        return None


def get_intro_keywords():
    """Get list of known introductory section keywords.

    These are true intro/editorial sections, not article headings.

    Returns:
        list: List of Tamil keywords for intro sections
    """
    keywords = [
        "எங்கள் எண்ணம்",
        "காலமும் கருத்தும்",
        "கருத்தும் காலமும்",
        "பாரதிதாசன் பரம்பரை",
        "வள்ளுவர் விருந்து",
        "பொது மேடை",
        "பொதுமேடை",
        "அட்டைப் படம்",
        "செய்திப் பாட்டு",
        "செய்திப்பாட்டு",
        "செய்திபாட்டு",
        "கலையுலகம்",
        "கலை உலகம்",
        "வளரும் இலக்கியம்",
        "இந்தி வேண்டாம்!",
        "இந்தி வந்தது, இந்தி!",
        "உயர்திரு உல்லாசம் அவர்கட்கு",
    ]

    logger.debug(f"Loaded {len(keywords)} intro keywords")
    return keywords
