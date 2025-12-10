from text_processing  import extract_author_from_line, is_valid_heading
from utils  import  count_content_lines
from content_extraction import count_consecutive_blanks


def extract_pattern_b_forward(lines, start_idx, end_idx, authors_normalized, authors_original, processed_lines, intro_keywords):
    """
    Extract articles following Pattern B: AUTHOR → HEADING → CONTENT.
    
    Args:
        lines (list): List of text lines
        start_idx (int): Start index for extraction
        end_idx (int): End index for extraction
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        processed_lines (list): Boolean list tracking processed lines
        intro_keywords (list): List of intro keywords
        
    Returns:
        list: List of extracted article dictionaries
    """
    articles = []
    i = start_idx
    
   
    
    while i < end_idx:
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        matched_author = extract_author_from_line(line, authors_normalized, authors_original)
        
        if matched_author:
            author_idx = i
            
            heading = None
            heading_idx = -1
            
            for check_offset in [1, 2]:
                check_idx = author_idx + check_offset
                if check_idx < end_idx and not processed_lines[check_idx]:
                    potential_heading = lines[check_idx].strip()
                    if potential_heading and len(potential_heading) < 25 and is_valid_heading(potential_heading):
                        if not extract_author_from_line(potential_heading, authors_normalized, authors_original):
                            heading = potential_heading
                            heading_idx = check_idx
                            break
            
            if not heading:
                i += 1
                continue
            
           
            
            content_start = heading_idx + 1
            while content_start < end_idx and not lines[content_start].strip() and (content_start - heading_idx) <= 3:
                content_start += 1
            
            if content_start >= end_idx:
                i += 1
                continue
            
            content_lines = []
            j = content_start
            
            while j < end_idx:
                if processed_lines[j]:
                    break
                
                current_line = lines[j]
                stripped = current_line.strip()
                
                if stripped:
                    next_author = extract_author_from_line(current_line, authors_normalized, authors_original)
                    if next_author:
                        break
                
                if stripped:
                    for keyword in intro_keywords:
                        if keyword in stripped:
                            while content_lines and not content_lines[-1].strip():
                                content_lines.pop()
                            break
                    if any(kw in stripped for kw in intro_keywords):
                        break
                
                if not stripped:
                    blank_count = count_consecutive_blanks(lines, j)
                    if blank_count >= 4:
                        break
                
                content_lines.append(current_line.rstrip())
                j += 1
            
            while content_lines and not content_lines[-1].strip():
                content_lines.pop()
            
            content = '\n'.join(content_lines)
            
            if content.strip() and count_content_lines(content) >= 4:
                for k in range(author_idx, j):
                    if k < len(lines):
                        processed_lines[k] = True
                
                articles.append({
                    "heading": heading,
                    "author": matched_author,
                    "content": content,
                    "start_idx": author_idx,
                    "end_idx": j
                })
              
                i = j
                continue
        
        i += 1
    
  
    return articles
