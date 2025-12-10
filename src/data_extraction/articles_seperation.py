import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict


def count_words(text):
    """
    Count the number of words in the given text.
    
    Args:
        text (str): Input text string
        
    Returns:
        int: Number of words in the text
    """
    if not text:
        return 0
    return len([word for word in text.split() if word.strip()])


def count_content_lines(text):
    """
    Count the number of non-empty lines in the text.
    
    Args:
        text (str): Input text string
        
    Returns:
        int: Number of lines with content
    """
    if not text:
        return 0
    return len([line for line in text.split('\n') if line.strip()])


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


def extract_doc_info(lines):
    """
    Extract document ID and issue number from the document header.
    
    Args:
        lines (list): List of text lines from document
        
    Returns:
        tuple: (doc_id, doc_issue) as strings
    """
    doc_id = "NA"
    doc_issue = "NA"
    
    for line in lines[:50]:
        line_stripped = line.strip()
        if 'மலர்' in line_stripped:
            match = re.search(r'மலர்\s*[:—-]?\s*(\d+)', line_stripped)
            if match and len(match.group(1)) <= 2:
                doc_id = match.group(1)
        if 'இதழ்' in line_stripped:
            match = re.search(r'இதழ்\s*[:—-]?\s*(\d+)', line_stripped)
            if match and len(match.group(1)) <= 2:
                doc_issue = match.group(1)
    
    return (doc_id, doc_issue)


def extract_authors_alternative(lines):
    """
    Extract authors by detecting names after multiple blank lines.
    
    Args:
        lines (list): List of text lines from document
        
    Returns:
        tuple: (authors_original, authors_normalized)
    """
    authors_original = []
    authors_normalized = []
    authors_set = set()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            blank_count = 0
            while i < len(lines) and not lines[i].strip():
                blank_count += 1
                i += 1
            
            if blank_count >= 2 and i < len(lines):
                potential_author = lines[i].strip()
                
                if potential_author and 3 <= len(potential_author) <= 30:
                    has_content = False
                    for offset in [1, 2, 3]:
                        check_idx = i - offset
                        if check_idx >= 0 and len(lines[check_idx].strip()) > 30:
                            has_content = True
                            break
                    
                    if has_content:
                        normalized = normalize_text(potential_author)
                        if normalized not in authors_set:
                            authors_original.append(potential_author)
                            authors_normalized.append(normalized)
                            authors_set.add(normalized)
        i += 1
    return (authors_original, authors_normalized)


def get_shared_authors(doc_id, doc_issue, all_files_authors):
    """
    Retrieve shared authors for a specific document from the global dictionary.
    
    Args:
        doc_id (str): Document ID
        doc_issue (str): Document issue number
        all_files_authors (dict): Dictionary mapping (doc_id, doc_issue) to authors
        
    Returns:
        tuple: (authors_original, authors_normalized)
    """
    key = (doc_id, doc_issue)
    return all_files_authors.get(key, ([], []))


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


def check_author_ahead(lines, current_idx, authors_normalized, authors_original, lookback=3):
    """
    Check if an author name appears in the next few lines.
    
    Args:
        lines (list): List of text lines
        current_idx (int): Current line index
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        lookback (int): Number of lines to look ahead
        
    Returns:
        int or None: Index of author line if found, None otherwise
    """
    for i in range(current_idx, min(current_idx + lookback + 1, len(lines))):
        author, idx = find_author_in_range(lines, i, i + 1, authors_normalized, authors_original)
        if author:
            return i
    return None


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


def parse_tamil_document(file_path, shared_authors_dict):
    """
    Parse a Tamil document file and extract intro sections, articles, and authors.
    
    Args:
        file_path (str): Path to the text file
        shared_authors_dict (dict): Dictionary of shared authors across documents
        
    Returns:
        dict: Dictionary containing intro, articles, authors_list, doc_id, and doc_issue
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    doc_id, doc_issue = extract_doc_info(lines)
    
    authors_original = []
    authors_normalized = []
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "பொருளடக்கம்" in line:
            start_idx = i
        if "ஆகியோரின் எழுத்தோவியங்கள்" in line:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
      
        authors_original, authors_normalized = get_shared_authors(doc_id, doc_issue, shared_authors_dict)
        if not authors_original:
           
            authors_original, authors_normalized = extract_authors_alternative(lines)
    else:
        for i in range(start_idx + 1, end_idx):
            author_name = lines[i].strip()
            if author_name:
                authors_original.append(author_name)
                authors_normalized.append(normalize_text(author_name))
    
   
    
    parse_start_idx = end_idx + 1 if end_idx != -1 else 0
    processed_lines = [False] * len(lines)
    
    if start_idx != -1 and end_idx != -1:
        for i in range(start_idx, end_idx + 1):
            processed_lines[i] = True
    
    intro_keywords = get_intro_keywords()
    articles = []
    intro = []
    article_no = 1
    
   
    i = parse_start_idx
    phase1_count = 0
    
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        matched_keyword = None
        
        for keyword in intro_keywords:
            if keyword in line:
                if i == 0 or not lines[i - 1].strip():
                    matched_keyword = keyword
                    break
        
        if matched_keyword:
            content, content_end_idx, has_author = extract_intro_content_phase1(
                lines, i, processed_lines, authors_normalized, authors_original, intro_keywords
            )
            
            for j in range(i, content_end_idx):
                if j < len(lines):
                    processed_lines[j] = True
            
            if content.strip() and count_content_lines(content) >= 4:
                if has_author:
                    articles.append({
                        "doc_id": doc_id,
                        "doc_issue": doc_issue,
                        "article_no": article_no,
                        "article_heading": matched_keyword,
                        "article_author_name": has_author,
                        "article_content": content
                    })
                    article_no += 1
                   
                else:
                    intro.append({
                        "doc_id": doc_id,
                        "doc_issue": doc_issue,
                        "heading": matched_keyword,
                        "content": content
                    })
                    phase1_count += 1
                   
            
            i = content_end_idx
        else:
            i += 1
    
    
    pattern_a_articles = extract_pattern_a_forward(
        lines, parse_start_idx, len(lines), 
        authors_normalized, authors_original, processed_lines, intro_keywords
    )
    
    for article in pattern_a_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    

    pattern_b_articles = extract_pattern_b_forward(
        lines, parse_start_idx, len(lines),
        authors_normalized, authors_original, processed_lines, intro_keywords
    )
    
    for article in pattern_b_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    
   
    pattern_c_articles = extract_pattern_c_reverse(
        lines, parse_start_idx, len(lines),
        authors_normalized, authors_original, processed_lines, intro_keywords
    )
    
    for article in pattern_c_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    
   
    remaining_articles = extract_remaining_content(
        lines, parse_start_idx, processed_lines
    )
    
    for article in remaining_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    
    authors_list = []
    for author in authors_original:
        authors_list.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "author_name": author
        })
    
    
    
    return {
        "intro": intro,
        "articles": articles,
        "authors_list": authors_list,
        "doc_id": doc_id,
        "doc_issue": doc_issue
    }


def build_shared_authors_dict(root_folder_path):
    """
    Build a dictionary mapping document IDs to their authors across all files.
    
    Args:
        root_folder_path (Path): Root folder containing text files
        
    Returns:
        dict: Dictionary mapping (doc_id, doc_issue) to (authors_original, authors_normalized)
    """
    
    shared_authors = {}
    
    txt_files = list(root_folder_path.rglob("*.txt"))
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            doc_id, doc_issue = extract_doc_info(lines)
            start_idx = -1
            end_idx = -1
            
            for i, line in enumerate(lines):
                if "பொருளடக்கம்" in line:
                    start_idx = i
                if "ஆகியோரின் எழுத்தோவியங்கள்" in line:
                    end_idx = i
                    break
            
            if start_idx != -1 and end_idx != -1:
                authors_original = []
                authors_normalized = []
                
                for i in range(start_idx + 1, end_idx):
                    author_name = lines[i].strip()
                    if author_name:
                        authors_original.append(author_name)
                        authors_normalized.append(normalize_text(author_name))
                
                key = (doc_id, doc_issue)
                if key not in shared_authors and authors_original:
                    shared_authors[key] = (authors_original, authors_normalized)
                   
        except Exception as e:
            
            continue
    
    
    return shared_authors


def process_folder(root_folder_path, output_root_folder="output_json"):
    """
    Process all text files in a folder and extract structured content.
    
    Args:
        root_folder_path (str): Path to input folder containing text files
        output_root_folder (str): Path to output folder for JSON files
    """
    root_path = Path(root_folder_path)
    output_root = Path(output_root_folder)
    
    if not root_path.exists():
        
        return
    
    output_root.mkdir(parents=True, exist_ok=True)
    txt_files = list(root_path.rglob("*.txt"))
    
    if not txt_files:
       
        return
    
    
    
    shared_authors_dict = build_shared_authors_dict(root_path)
    folder_authors = defaultdict(lambda: defaultdict(set))
    
    for txt_file in txt_files:
        try:
           
            
            result = parse_tamil_document(str(txt_file), shared_authors_dict)
            
            relative_path = txt_file.relative_to(root_path)
            output_folder = output_root / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            base_name = txt_file.stem
            
            content_json_path = output_folder / f"{base_name}.json"
            with open(content_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "intro": result["intro"],
                    "articles": result["articles"]
                }, f, ensure_ascii=False, indent=2)
            
          
            
            folder_key = relative_path.parent
            doc_key = (result["doc_id"], result["doc_issue"])
            for author_info in result["authors_list"]:
                folder_authors[folder_key][doc_key].add(author_info["author_name"])
            
        except Exception as e:
          
            continue

   
    for folder_key, doc_authors_dict in folder_authors.items():
        output_folder = output_root / folder_key
        folder_name = folder_key.name if folder_key.name else "root"
        
        consolidated_authors = []
        for (doc_id, doc_issue), authors_set in doc_authors_dict.items():
            consolidated_authors.append({
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "authors": sorted(list(authors_set))
            })
        
        authors_json_path = output_folder / f"{folder_name}_authors.json"
        with open(authors_json_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_authors, f, ensure_ascii=False, indent=2)
        
       

if __name__ == "__main__":
   
    print("Processing....")
    root_folder = "extracted_texts"
    output_folder = "output"
    process_folder(root_folder, output_folder)
    print("completed.")