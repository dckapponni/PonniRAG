"""
Content extraction utilities with comprehensive logging.
Handles extraction of intro sections, remaining content, and keyword detection.
"""
import logging
from text_processing import is_valid_heading, extract_author_from_line
from doc_utils import count_content_lines 
from shared_author_local import check_author_ahead  # CHANGED FROM shared_author

logger = logging.getLogger('TamilDocProcessor.content_extraction')


def check_keyword_ahead(lines, current_idx, intro_keywords, lookback=2):
    """
    Check if an intro keyword appears in the next few lines.
    
    Args:
        lines (list): List of text lines
        current_idx (int): Current line index
        intro_keywords (list): List of intro keywords to search for
        lookback (int): Number of lines to look ahead (default: 2)
        
    Returns:
        int or None: Index of keyword line if found, None otherwise
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
    
    Args:
        lines (list): List of text lines
        start_idx (int): Starting index to begin counting
        
    Returns:
        int: Number of consecutive blank lines
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
        logger.warning(f"Index error in count_consecutive_blanks at line {start_idx}: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error in count_consecutive_blanks at line {start_idx}: {e}")
        return 0


def extract_remaining_content(lines, start_idx, processed_lines):
    """
    Extract remaining unprocessed content using blank space logic.
    Groups content by blank line separators and extracts sections with valid headings.
    
    Args:
        lines (list): List of text lines from document
        start_idx (int): Starting index for extraction
        processed_lines (list): Boolean list tracking which lines have been processed
        
    Returns:
        list: List of extracted content sections, each with heading, author, content, and indices
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
                        logger.debug(f"Stopped at {blank_count} blank lines at line {i}")
                        break
                    else:
                        content_lines.append(line.rstrip())
                        i += 1
                else:
                    content_lines.append(line.rstrip())
                    i += 1
            
            if content_lines:
                last_line = content_lines[-1].strip() if content_lines else ""
                
                if last_line and (last_line.startswith('—') or last_line.startswith('-') or last_line.startswith('–')):
                    logger.debug(f"Skipping group ending with author dash at line {group_start}")
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
                    
                    content = '\n'.join(remaining)
                    
                    if content.strip() and count_content_lines(content) >= 4:
                        groups.append({
                            "heading": heading,
                            "author": "NA",
                            "content": content,
                            "start_idx": group_start,
                            "end_idx": i
                        })
                        logger.info(f"Remaining: Extracted content with heading '{heading}' ({count_content_lines(content)} lines)")
                    else:
                        logger.debug(f"Remaining: Content too short ({count_content_lines(content)} lines) at line {group_start}")
                else:
                    logger.debug(f"Remaining: No valid heading at line {group_start}, skipping group")
                
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
        logger.error(f"Index error in extract_remaining_content at line {i}: {e}", exc_info=True)
        return groups
    except Exception as e:
        logger.error(f"Unexpected error in extract_remaining_content at line {i}: {e}", exc_info=True)
        return groups


def extract_intro_content_phase1(lines, keyword_idx, processed_lines, authors_normalized, 
                                  authors_original, intro_keywords):
    """
    Extract content for intro sections identified by keywords in Phase 1.
    Handles detection of author names within keyword lines and proper content boundaries.
    
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
        has_author_in_keyword = extract_author_from_line(keyword_line, authors_normalized, authors_original)
        
        if has_author_in_keyword:
            logger.debug(f"Found author '{has_author_in_keyword}' in keyword line")
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            author_idx = check_author_ahead(lines, i, authors_normalized, authors_original, lookback=3)
            if author_idx is not None:
                stop_at = max(i, author_idx - 3)
                while len(content_lines) > stop_at - (keyword_idx + 1):
                    content_lines.pop()
                logger.debug(f"Stopped before author at line {author_idx}")
                break
            
            keyword_idx_found = check_keyword_ahead(lines, i, intro_keywords, lookback=2)
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
        
        content = '\n'.join(content_lines)
        logger.debug(f"Intro content extracted: {count_content_lines(content)} lines, author={has_author_in_keyword}")
        
        return (content, i, has_author_in_keyword)
        
    except IndexError as e:
        logger.error(f"Index error in extract_intro_content_phase1 at line {i}: {e}", exc_info=True)
        return ('', keyword_idx + 1, None)
    except Exception as e:
        logger.error(f"Unexpected error in extract_intro_content_phase1 at keyword line {keyword_idx}: {e}", exc_info=True)
        return ('', keyword_idx + 1, None)