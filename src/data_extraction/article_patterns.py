from text_processing  import extract_author_from_line, is_valid_heading
from doc_utils  import  count_content_lines
from content_extraction import count_consecutive_blanks


def extract_pattern_a_forward(lines, start_idx, end_idx, authors_normalized, authors_original, processed_lines, intro_keywords):
    """
    Extract articles following Pattern A: HEADING → AUTHOR → CONTENT.
    
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
            
            for check_offset in [3, 2, 1]:
                check_idx = author_idx - check_offset
                if check_idx >= start_idx and not processed_lines[check_idx]:
                    potential_heading = lines[check_idx].strip()
                    if potential_heading and len(potential_heading) < 25 and is_valid_heading(potential_heading):
                        if not extract_author_from_line(potential_heading, authors_normalized, authors_original):
                            heading = potential_heading
                            heading_idx = check_idx
                            break
            
            if not heading:
                i += 1
                continue
            
            
            
            content_start = author_idx + 1
            while content_start < end_idx and not lines[content_start].strip() and (content_start - author_idx) <= 3:
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
                            j -= 1
                            break
                    if j < end_idx and any(kw in stripped for kw in intro_keywords):
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
                for k in range(heading_idx, j):
                    if k < len(lines):
                        processed_lines[k] = True
                
                articles.append({
                    "heading": heading,
                    "author": matched_author,
                    "content": content,
                    "start_idx": heading_idx,
                    "end_idx": j
                })
               
                
                i = j
                continue
        
        i += 1
    
   
    return articles


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


def extract_pattern_c_reverse(lines, start_idx, end_idx, authors_normalized, authors_original, processed_lines, intro_keywords):
    """
    Extract articles following Pattern C: HEADING ← CONTENT ← AUTHOR (reverse extraction).
    
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
        
        matched_author = None
        
        if line.startswith('—') or line.startswith('-') or line.startswith('–'):
            matched_author = extract_author_from_line(line, authors_normalized, authors_original)
        else:
            temp_author = extract_author_from_line(line, authors_normalized, authors_original)
            if temp_author and len(line) <= 30:
                matched_author = temp_author
        
        if matched_author:
            author_idx = i
          
            
            content_end = author_idx - 1
            
            while content_end >= start_idx and not lines[content_end].strip():
                content_end -= 1
            
            if content_end < start_idx:
                i += 1
                continue
            
            content_lines = []
            j = content_end
            
            while j >= start_idx:
                if processed_lines[j]:
                    break
                
                current_line = lines[j]
                stripped = current_line.strip()
                
                if stripped and (stripped.startswith('—') or stripped.startswith('-') or stripped.startswith('–')):
                    prev_author = extract_author_from_line(current_line, authors_normalized, authors_original)
                    if prev_author:
                        break
                
                if stripped:
                    found_keyword = False
                    for keyword in intro_keywords:
                        if keyword in stripped:
                            found_keyword = True
                            break
                    if found_keyword:
                        break
                
                if not stripped:
                    blank_count = 0
                    k = j
                    while k >= start_idx and not lines[k].strip():
                        blank_count += 1
                        k -= 1
                        if blank_count >= 4:
                            break
                    if blank_count >= 4:
                        j = k
                        break
                
                content_lines.insert(0, current_line.rstrip())
                j -= 1
            
            content_start = j + 1
            
            while content_lines and not content_lines[0].strip():
                content_lines.pop(0)
                content_start += 1
            
            while content_lines and not content_lines[-1].strip():
                content_lines.pop()
            
            if not content_lines or count_content_lines('\n'.join(content_lines)) < 4:
             
                i += 1
                continue
            
            heading_idx = content_start - 1
            
            while heading_idx >= start_idx and not lines[heading_idx].strip():
                heading_idx -= 1
            
            if heading_idx < start_idx:
              
                i += 1
                continue
            
            potential_heading = lines[heading_idx].strip()
            
            if not potential_heading:
               
                i += 1
                continue
            
            if len(potential_heading) >= 25:
              
                i += 1
                continue
            
            if potential_heading.startswith('—') or potential_heading.startswith('-') or potential_heading.startswith('–'):
             
                i += 1
                continue
            
            if extract_author_from_line(potential_heading, authors_normalized, authors_original):
              
                i += 1
                continue
            
            two_blank_above = False
            if heading_idx >= 2:
                if not lines[heading_idx - 1].strip() and not lines[heading_idx - 2].strip():
                    two_blank_above = True
            
            two_blank_below = False
            if heading_idx + 2 < len(lines):
                if not lines[heading_idx + 1].strip() and not lines[heading_idx + 2].strip():
                    two_blank_below = True
            
            if not (two_blank_above and two_blank_below):
              
                i += 1
                continue
            
            heading = potential_heading
            content = '\n'.join(content_lines)
            
          
            for k in range(heading_idx, author_idx + 1):
                if k < len(lines):
                    processed_lines[k] = True
            
            articles.append({
                "heading": heading,
                "author": matched_author,
                "content": content,
                "start_idx": heading_idx,
                "end_idx": author_idx + 1
            })
            
            i = author_idx + 1
            continue
        
        i += 1
    
    
    return articles
