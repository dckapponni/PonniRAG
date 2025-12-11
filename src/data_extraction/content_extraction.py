from text_processing import is_valid_heading, extract_author_from_line
from doc_utils import count_content_lines 
from shared_author import check_author_ahead

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

def count_consecutive_blanks(lines, start_idx):
    """
    Count consecutive blank lines starting from a given index.
    
    Args:
        lines (list): List of text lines
        start_idx (int): Starting index
        
    Returns:
        int: Number of consecutive blank lines
    """
    count = 0
    i = start_idx
    while i < len(lines) and not lines[i].strip():
        count += 1
        i += 1
    return count


def extract_remaining_content(lines, start_idx, processed_lines):
    """
    Extract remaining unprocessed content using blank space logic.
    
    Args:
        lines (list): List of text lines
        start_idx (int): Starting index for extraction
        processed_lines (list): Boolean list tracking processed lines
        
    Returns:
        list: List of extracted content sections
    """
    groups = []
    i = start_idx
    
  
    total_lines = len(lines)
    
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
                   
               
            for j in range(group_start, i):
                if j < total_lines:
                    processed_lines[j] = True
        
        if i < total_lines and not lines[i].strip():
            blank_count = count_consecutive_blanks(lines, i)
            for j in range(i, min(i + blank_count, total_lines)):
                if j < total_lines:
                    processed_lines[j] = True
            i += blank_count
    
   
    return groups


def extract_intro_content_phase1(lines, keyword_idx, processed_lines, authors_normalized, 
                                authors_original, intro_keywords):
    """
    Extract content for intro sections identified by keywords in Phase 1.
    
    Args:
        lines (list): List of text lines
        keyword_idx (int): Index of keyword line
        processed_lines (list): Boolean list tracking processed lines
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        intro_keywords (list): List of intro keywords
        
    Returns:
        tuple: (content_text, end_index, author_if_found)
    """
    content_lines = []
    i = keyword_idx + 1
    
    if i < len(lines) and not lines[i].strip():
        i += 1
    
    keyword_line = lines[keyword_idx].strip()
    has_author_in_keyword = extract_author_from_line(keyword_line, authors_normalized, authors_original)
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        author_idx = check_author_ahead(lines, i, authors_normalized, authors_original, lookback=3)
        if author_idx is not None:
            stop_at = max(i, author_idx - 3)
            while len(content_lines) > stop_at - (keyword_idx + 1):
                content_lines.pop()
            break
        
        keyword_idx_found = check_keyword_ahead(lines, i, intro_keywords, lookback=2)
        if keyword_idx_found is not None:
            stop_at = max(i, keyword_idx_found - 2)
            while len(content_lines) > stop_at - (keyword_idx + 1):
                content_lines.pop()
            break
        
        if not stripped:
            blank_count = count_consecutive_blanks(lines, i)
            if blank_count >= 4:
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                break
            else:
                content_lines.append(line.rstrip())
        else:
            content_lines.append(line.rstrip())
        
        i += 1
    
    return ('\n'.join(content_lines), i, has_author_in_keyword)
