import re
from difflib import SequenceMatcher

def normalize_text(text):
    """
    Normalize Tamil text by handling Unicode variations and removing punctuation.
    
    Args:
        text (str): Input text to normalize
        
    Returns:
        str: Normalized text in lowercase without spaces or periods
    """
    text = text.replace('ண', 'ண').replace('णు', ' णु')
    text = re.sub(r'\.', '', text)
    text = re.sub(r'\s+', '', text.strip().lower())
    return text


def is_valid_heading(heading):
    """
    Validate if a line qualifies as a proper article heading.
    
    Args:
        heading (str): Potential heading text
        
    Returns:
        bool: True if valid heading, False otherwise
    """
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


def fuzzy_match_author(line, authors_normalized, authors_original, threshold=0.8):
    """
    Match a line against known authors using fuzzy string matching.
    
    Args:
        line (str): Line to match
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        threshold (float): Minimum similarity score (0-1)
        
    Returns:
        tuple: (matched_author_name, similarity_score)
    """
    if not line.strip():
        return (None, 0)
    
    line_normalized = normalize_text(line)
    best_match = None
    best_similarity = 0
    
    for i, author_norm in enumerate(authors_normalized):
        similarity = SequenceMatcher(None, line_normalized, author_norm).ratio()
        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = authors_original[i]
    
    return (best_match, best_similarity)


def extract_author_from_line(line, authors_normalized, authors_original):
    """
    Extract author name from a line by cleaning and matching against known authors.
    
    Args:
        line (str): Line potentially containing author name
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        
    Returns:
        str or None: Matched author name or None
    """
    clean_line = line.strip()
    
    if clean_line.startswith('—') or clean_line.startswith('-') or clean_line.startswith('–'):
        clean_line = re.sub(r'^[—\-–]\s*', '', clean_line)
    
    if '[' in clean_line and ']' in clean_line:
        bracket_match = re.search(r'\[(.*?)\]', clean_line)
        if bracket_match:
            clean_line = bracket_match.group(1)
    
    if 'ஆசிரியர்' in clean_line or ':' in clean_line:
        parts = clean_line.split(':')
        if len(parts) > 1:
            clean_line = parts[-1].strip()
    
    clean_line = re.sub(r'[,.\]"]+$', '', clean_line)
    clean_line = re.sub(r'^["]+', '', clean_line)
    
    matched_author, similarity = fuzzy_match_author(clean_line, authors_normalized, authors_original)
    return matched_author


def find_author_in_range(lines, start_idx, end_idx, authors_normalized, authors_original):
    """
    Search for an author name within a specified range of lines.
    
    Args:
        lines (list): List of text lines
        start_idx (int): Start index for search
        end_idx (int): End index for search
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        
    Returns:
        tuple: (author_name, line_index) or (None, -1)
    """
    for i in range(start_idx, min(end_idx, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        author = extract_author_from_line(line, authors_normalized, authors_original)
        if author:
            return (author, i)
    return (None, -1)


def get_intro_keywords():
    """
    Get list of known introductory section keywords.
    
    Returns:
        list: List of Tamil keywords for intro sections
    """
    return [
        "எங்கள் எண்ணம்", "கலையுலகம்", "வளரும் இலக்கியம்", "காலமும் கருத்தும்",
        "பாரதிதாசன் பரம்பரை", "வள்ளுவர் விருந்து", "பொது மேடை", "செய்திப் பாட்டு",
        "கலை உலகம்", "அட்டைப் படம்", "இந்தி வேண்டாம்!", "இந்தி வந்தது, இந்தி!",
        "உயர்திரு உல்லாசம் அவர்கட்கு"
    ]


