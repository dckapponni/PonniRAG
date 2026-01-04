"""
Text processing utilities with comprehensive logging.
FIXED: Correct intro_keywords list - removed article headings
"""
import re
import logging
from difflib import SequenceMatcher

# Get logger
logger = logging.getLogger('TamilDocProcessor.text_processing')


def normalize_text(text):
    """
    Normalize Tamil text by handling Unicode variations and removing punctuation.
    """
    try:
        if not text:
            return ""
        
        text = text.replace('ண', 'ண').replace('णு', ' णு')
        text = re.sub(r'\.', '', text)
        text = re.sub(r'\s+', '', text.strip().lower())
        
        return text
        
    except Exception as e:
        logger.warning(f"Error normalizing text '{text[:50] if text else ''}...': {e}")
        return text if text else ""


def normalize_title(title):
    """Normalize title/heading for matching across documents."""
    try:
        if not title:
            return ""
        
        title = re.sub(r'[.,!?;:"\'\-—–()[\]{}]', '', title)
        title = title.replace('ண', 'ண').replace('णு', ' णு')
        title = re.sub(r'\s+', '', title.strip().lower())
        
        return title
        
    except Exception as e:
        logger.warning(f"Error normalizing title '{title[:50] if title else ''}...': {e}")
        return title if title else ""


def is_valid_heading(heading):
    """Validate if a line qualifies as a proper article heading."""
    try:
        if not heading or not heading.strip():
            return False
        
        stripped = heading.strip()
        
        if stripped.isdigit():
            return False
        
        if len(stripped) >= 25:
            return False
        
        if stripped.startswith('—') or stripped.startswith('-') or stripped.startswith('–'):
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating heading '{heading[:30] if heading else ''}...': {e}")
        return False


def fuzzy_match_author(line, authors_normalized, authors_original, threshold=0.8):
    """Match a line against known authors using fuzzy string matching."""
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
            logger.debug(f"Fuzzy matched '{line[:30]}...' to '{best_match}' (similarity: {best_similarity:.2f})")
        
        return (best_match, best_similarity)
        
    except Exception as e:
        logger.warning(f"Error in fuzzy_match_author for line '{line[:50] if line else ''}...': {e}")
        return (None, 0)


def extract_author_from_line(line, authors_normalized, authors_original):
    """Extract author name from a line by cleaning and matching against known authors."""
    try:
        clean_line = line.strip()
        
        if not clean_line:
            return None
        
        # Remove leading dashes
        if clean_line.startswith('—') or clean_line.startswith('-') or clean_line.startswith('–'):
            clean_line = re.sub(r'^[—\-–]\s*', '', clean_line)
        
        # Extract from brackets
        if '[' in clean_line and ']' in clean_line:
            bracket_match = re.search(r'\[(.*?)\]', clean_line)
            if bracket_match:
                clean_line = bracket_match.group(1)
        
        # Remove prefix like "ஆசிரியர்:"
        if 'ஆசிரியர்' in clean_line or ':' in clean_line:
            parts = clean_line.split(':')
            if len(parts) > 1:
                clean_line = parts[-1].strip()
        
        # Clean trailing punctuation
        clean_line = re.sub(r'[,.\]"]+$', '', clean_line)
        clean_line = re.sub(r'^["]+', '', clean_line)
        
        # Match against known authors
        matched_author, similarity = fuzzy_match_author(clean_line, authors_normalized, authors_original)
        
        return matched_author
        
    except Exception as e:
        logger.warning(f"Error extracting author from line '{line[:50] if line else ''}...': {e}")
        return None


def find_author_in_range(lines, start_idx, end_idx, authors_normalized, authors_original):
    """Search for an author name within a specified range of lines."""
    try:
        logger.debug(f"Searching for author from line {start_idx} to {end_idx}")
        
        for i in range(start_idx, min(end_idx, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            
            author = extract_author_from_line(line, authors_normalized, authors_original)
            if author:
                logger.debug(f"Found author '{author}' at line {i}")
                return (author, i)
        
        logger.debug(f"No author found in range {start_idx}-{end_idx}")
        return (None, -1)
        
    except IndexError as e:
        logger.warning(f"Index error in find_author_in_range ({start_idx}-{end_idx}): {e}")
        return (None, -1)
    except Exception as e:
        logger.error(f"Error in find_author_in_range: {e}")
        return (None, -1)


def get_intro_keywords():
    """
    Get list of known introductory section keywords.
    ✅ FIXED: Removed article headings, only true intro sections remain
    
    Returns:
        list: List of Tamil keywords for intro sections
    """
    keywords = [
        # True intro/editorial sections
        "எங்கள் எண்ணம்",
        "காலமும் கருத்தும்",
        "கருத்தும் காலமும்",
        "பாரதிதாசன் பரம்பரை",
        "வள்ளுவர் விருந்து",
        "பொது மேடை",
        "பொதுமேடை",
        
        # Cover page descriptions (these are intro, not articles)
        "அட்டைப் படம்",
        
        # News/announcement sections
        "செய்திப் பாட்டு",
        "செய்திப்பாட்டு",
        "செய்திபாட்டு",
        
        # Special sections
        "கலையுலகம்",
        "கலை உலகம்",
        "வளரும் இலக்கியம்",
        
        # Specific editorial pieces
        "இந்தி வேண்டாம்!",
        "இந்தி வந்தது, இந்தி!",
        "உயர்திரு உல்லாசம் அவர்கட்கு"
    ]
    
    logger.debug(f"Loaded {len(keywords)} intro keywords")
    return keywords