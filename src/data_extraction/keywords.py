def check_keyword_ahead(lines, current_idx, intro_keywords, lookback=2):
    """
    Check if an intro keyword appears in the next few lines.
    
    Args:
        lines (list): List of text lines
        current_idx (int): Current line index
        intro_keywords (list): List of intro keywords
        lookback (int): Number of lines to look ahead
        
    Returns:
        int or None: Index of keyword line if found, None otherwise
    """
    for i in range(current_idx, min(current_idx + lookback + 1, len(lines))):
        line = lines[i].strip()
        for keyword in intro_keywords:
            if keyword in line:
                return i
    return None
