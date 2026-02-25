import logging

logger = logging.getLogger('TamilDocProcessor.shared_author_local')


def check_author_ahead(lines, current_idx, authors_normalized, authors_original, lookback=3):
    """
    Check if an author name appears in the next few lines.
    Searches forward from the current position to detect if any known author name
    appears within a specified lookahead window. Used to stop content extraction
    before reaching the next author's section.

    Args:

        lines (list): List of text lines from the document.

        current_idx (int): Current line index to start searching from.

        authors_normalized (list): List of normalized author names for matching.

        authors_original (list): List of original author names (parallel to normalized).

        lookback (int, optional): Number of lines to look ahead. Defaults to 3.

    Returns:

        int or None: Line index where author name was found, or None if no author

                    found within lookahead window or on error.
    """
    try:
        from text_processing import extract_author_from_line
        
        for i in range(current_idx, min(current_idx + lookback + 1, len(lines))):
            if i >= len(lines):
                break
                
            line = lines[i].strip()
            if not line:
                continue
            
            author = extract_author_from_line(line, authors_normalized, authors_original)
            if author:
                logger.debug(f"Found author '{author}' ahead at line {i}")
                return i
        
        return None
        
    except Exception as e:
        logger.error(f"Error in check_author_ahead: {e}")
        return None


