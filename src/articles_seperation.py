import json
import re
import os
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict


def count_words(text):
    """
    Count words in text.
    
    Args:
        text (str): Input text
    
    Returns:
        int: Number of words
    """
    if not text:
        return 0
    words = [word for word in text.split() if word.strip()]
    return len(words)


def normalize_text(text):
    """
    Normalize text for fuzzy matching.
    
    Operations:
    - Replace specific Tamil characters
    - Remove dots
    - Remove extra spaces
    - Convert to lowercase
    
    Args:
        text (str): Input text
    
    Returns:
        str: Normalized text
    """
    text = text.replace('ண', 'ण').replace('ணு', ' णु')
    text = re.sub(r'\.', '', text)
    text = re.sub(r'\s+', '', text.strip().lower())
    return text


def is_valid_heading(heading):
    """
    Validate heading - must not be empty or pure number.
    
    Rules:
    - NOT empty
    - NOT pure number (can be char + number)
    
    Args:
        heading (str): Heading text
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not heading or not heading.strip():
        return False
    
    # Check if pure number
    if heading.strip().isdigit():
        return False
    
    return True


def fuzzy_match_author(line, authors_normalized, authors_original, threshold=0.8):
    """
    Fuzzy match line against authors with 80-100% similarity.
    
    Args:
        line (str): Line to match
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
        threshold (float): Minimum similarity threshold (default: 0.8)
    
    Returns:
        tuple: (matched_author_name, similarity) or (None, 0) if no match
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


def extract_author_from_brackets(line, authors_normalized, authors_original):
    """
    Extract author from content within brackets [...].
    
    Args:
        line (str): Line containing brackets
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
    
    Returns:
        str or None: Matched author name or None
    """
    if '[' in line and ']' in line:
        bracket_content = re.search(r'\[(.*?)\]', line)
        if bracket_content:
            content = bracket_content.group(1)
            
            parts = content.split('"')
            if len(parts) > 0:
                potential_author = parts[0].strip()
                matched_author, similarity = fuzzy_match_author(
                    potential_author, authors_normalized, authors_original
                )
                if matched_author:
                    return matched_author
            
            if 'ஆசிரியர்' in content or ':' in content:
                parts = content.split(':')
                if len(parts) > 1:
                    potential_author = parts[-1].strip()
                    potential_author = re.sub(r'[,.\]]+$', '', potential_author)
                    matched_author, similarity = fuzzy_match_author(
                        potential_author, authors_normalized, authors_original
                    )
                    if matched_author:
                        return matched_author
    return None


def extract_author_from_pattern(line, authors_normalized, authors_original):
    """
    Extract author from patterns like 'ஆசிரியர் : author_name'.
    
    Args:
        line (str): Line to search
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
    
    Returns:
        str or None: Matched author name or None
    """
    if 'ஆசிரியர்' in line or ':' in line:
        parts = line.split(':')
        if len(parts) > 1:
            potential_author = parts[-1].strip()
            matched_author, similarity = fuzzy_match_author(
                potential_author, authors_normalized, authors_original
            )
            if matched_author:
                return matched_author
    return None


def find_author_in_range(lines, start_idx, end_idx, authors_normalized, authors_original):
    """
    Search for author name within a range of lines.
    
    Args:
        lines (list): Document lines
        start_idx (int): Start index
        end_idx (int): End index
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
    
    Returns:
        tuple: (author_name, line_index) or (None, -1)
    """
    for i in range(start_idx, min(end_idx, len(lines))):
        line = lines[i].strip()
        
        author = extract_author_from_brackets(line, authors_normalized, authors_original)
        if author:
            return (author, i)
        
        author = extract_author_from_pattern(line, authors_normalized, authors_original)
        if author:
            return (author, i)
        
        matched_author, similarity = fuzzy_match_author(
            line, authors_normalized, authors_original
        )
        if matched_author:
            return (matched_author, i)
    
    return (None, -1)


def extract_doc_info(lines):
    """
    Extract மலர் (doc_id) and இதழ் (doc_issue) from document.
    
    Args:
        lines (list): Document lines
    
    Returns:
        tuple: (doc_id, doc_issue)
    """
    doc_id = "Unknown"
    doc_issue = "Unknown"
    
    for line in lines[:50]:
        line_stripped = line.strip()
        
        if 'மலர்' in line_stripped:
            match = re.search(r'மலர்\s*[:—-]?\s*(\d+)', line_stripped)
            if match:
                doc_id = match.group(1)
        
        if 'இதழ்' in line_stripped:
            match = re.search(r'இதழ்\s*[:—-]?\s*(\d+)', line_stripped)
            if match:
                doc_issue = match.group(1)
    
    return (doc_id, doc_issue)


def extract_authors_alternative(lines):
    """
    Alternative method to extract authors when no author list section exists.
    
    Pattern: 2+ empty lines → author name → content above (30+ chars)
    
    Args:
        lines (list): Document lines
    
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
                potential_author_idx = i
                potential_author = lines[potential_author_idx].strip()
                
                if potential_author and 3 <= len(potential_author) <= 30:
                    has_content = False
                    for offset in [1, 2, 3]:
                        check_idx = potential_author_idx - offset
                        if check_idx >= 0:
                            check_line = lines[check_idx].strip()
                            if len(check_line) > 30:
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
    Get author list from other files with same doc_id and doc_issue.
    
    Args:
        doc_id (str): Document ID
        doc_issue (str): Document issue
        all_files_authors (dict): Dictionary mapping (doc_id, doc_issue) to author lists
    
    Returns:
        tuple: (authors_original, authors_normalized) or ([], [])
    """
    key = (doc_id, doc_issue)
    if key in all_files_authors:
        return all_files_authors[key]
    return ([], [])


def get_intro_keywords():
    """
    Get list of intro section keywords.
    
    Returns:
        list: Intro keywords
    """
    return [
        "எங்கள் எண்ணம்",
        "கலையுலகம்",
        "வளரும் இலக்கியம்",
        "காலமும் கருத்தும்",
        "பாரதிதாசன் பரம்பரை",
        "வள்ளுவர் விருந்து",
        "பொது மேடை",
        "செய்திப் பாட்டு",
        "கலை உலகம்",
        "அட்டைப் படம்",
        "இந்தி வேண்டாம்!",
        "இந்தி வந்தது, இந்தி!",
        "உயர்திரு உல்லாசம் அவர்கட்கு",
        "உயர்த்த உல்லாசம் அவர்கட்கு",
        "வம்புமடம்",
        "விமரிசனம்",
        "கண்ட பயன்"
    ]


def count_consecutive_blanks(lines, start_idx):
    """
    Count consecutive blank lines starting from start_idx.
    
    Args:
        lines (list): Document lines
        start_idx (int): Start index
    
    Returns:
        int: Count of consecutive blank lines
    """
    count = 0
    i = start_idx
    while i < len(lines) and not lines[i].strip():
        count += 1
        i += 1
    return count


def check_author_ahead(lines, current_idx, authors_normalized, authors_original, lookback=3):
    """
    Check if author name appears within next few lines.
    
    Args:
        lines (list): Document lines
        current_idx (int): Current index
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
        lookback (int): Number of lines to check ahead (default: 3)
    
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
    Check if intro keyword appears within next few lines.
    
    Args:
        lines (list): Document lines
        current_idx (int): Current index
        intro_keywords (list): List of intro keywords
        lookback (int): Number of lines to check ahead (default: 2)
    
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
    PHASE 1: Extract intro content with strict stopping rules.
    
    Content stops when:
    1. Author name found (stop 3 lines before)
    2. Another keyword found (stop 2 lines before)
    3. If both fail: 4+ consecutive blank lines
    
    Args:
        lines (list): Document lines
        keyword_idx (int): Keyword line index
        processed_lines (list): Boolean array of processed lines
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
        intro_keywords (list): List of intro keywords
    
    Returns:
        tuple: (content_text, end_index)
    """
    content_lines = []
    i = keyword_idx + 1
    
    # Skip initial blank line
    if i < len(lines) and not lines[i].strip():
        i += 1
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check 1: Author ahead (3 lines)
        author_idx = check_author_ahead(lines, i, authors_normalized, authors_original, lookback=3)
        if author_idx is not None:
            # Stop 3 lines before author
            stop_at = author_idx - 3
            if stop_at < i:
                stop_at = i
            while len(content_lines) > stop_at - (keyword_idx + 1):
                content_lines.pop()
            break
        
        # Check 2: Keyword ahead (2 lines)
        keyword_idx_found = check_keyword_ahead(lines, i, intro_keywords, lookback=2)
        if keyword_idx_found is not None:
            # Stop 2 lines before keyword
            stop_at = keyword_idx_found - 2
            if stop_at < i:
                stop_at = i
            while len(content_lines) > stop_at - (keyword_idx + 1):
                content_lines.pop()
            break
        
        # Check 3: 4+ blank lines
        if not stripped:
            blank_count = count_consecutive_blanks(lines, i)
            if blank_count >= 4:
                # Remove trailing blanks
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                break
            else:
                content_lines.append(line.rstrip())
        else:
            content_lines.append(line.rstrip())
        
        i += 1
    
    return ('\n'.join(content_lines), i)


def extract_heading_above_author(lines, author_idx, authors_normalized, authors_original):
    """
    Extract heading from lines above author (≤25 chars).
    
    Two-step approach:
    1. Check 1-2 lines directly above
    2. Search upward with blank separator
    
    Args:
        lines (list): Document lines
        author_idx (int): Author line index
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
    
    Returns:
        tuple: (heading, heading_start_idx) or (None, -1)
    """
    heading_lines = []
    heading_indices = []
    
    # Step 1: Check direct lines above
    if author_idx - 2 >= 0:
        line2 = lines[author_idx - 2].strip()
        matched_author, _ = fuzzy_match_author(line2, authors_normalized, authors_original)
        if line2 and not matched_author and len(line2) <= 25 and '[' not in line2 and ':' not in line2:
            heading_lines.insert(0, line2)
            heading_indices.insert(0, author_idx - 2)
    
    if author_idx - 1 >= 0:
        line1 = lines[author_idx - 1].strip()
        matched_author, _ = fuzzy_match_author(line1, authors_normalized, authors_original)
        if line1 and not matched_author and len(line1) <= 25 and '[' not in line1 and ':' not in line1:
            heading_lines.append(line1)
            heading_indices.append(author_idx - 1)
    
    if heading_lines:
        return (' '.join(heading_lines), min(heading_indices))
    
    # Step 2: Search upward with blank separator
    for i in range(author_idx - 1, -1, -1):
        line_text = lines[i].strip()
        
        if not line_text:
            continue
        
        if len(line_text) <= 25:
            matched_author, _ = fuzzy_match_author(line_text, authors_normalized, authors_original)
            
            if not matched_author and '[' not in line_text and ':' not in line_text:
                if i > 0 and not lines[i - 1].strip():
                    return (line_text, i)
    
    return (None, -1)


def extract_content_phase2_pattern_a(lines, author_idx, heading_start_idx, processed_lines,
                                     authors_normalized, authors_original, intro_keywords):
    """
    PHASE 2 Pattern A: heading → author → content
    
    Extract content after author with stopping rules:
    1. Author ahead (stop 3 lines before)
    2. Keyword ahead (stop 2 lines before)
    3. 4+ blank lines
    
    Args:
        lines (list): Document lines
        author_idx (int): Author line index
        heading_start_idx (int): Heading start index
        processed_lines (list): Boolean array
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
        intro_keywords (list): List of intro keywords
    
    Returns:
        tuple: (content_text, end_index)
    """
    content_lines = []
    i = author_idx + 1
    
    # Skip initial blank
    if i < len(lines) and not lines[i].strip():
        i += 1
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check 1: Author ahead
        author_idx_found = check_author_ahead(lines, i, authors_normalized, authors_original, lookback=3)
        if author_idx_found is not None:
            stop_at = author_idx_found - 3
            if stop_at < i:
                stop_at = i
            while len(content_lines) > stop_at - (author_idx + 1):
                content_lines.pop()
            break
        
        # Check 2: Keyword ahead
        keyword_idx_found = check_keyword_ahead(lines, i, intro_keywords, lookback=2)
        if keyword_idx_found is not None:
            stop_at = keyword_idx_found - 2
            if stop_at < i:
                stop_at = i
            while len(content_lines) > stop_at - (author_idx + 1):
                content_lines.pop()
            break
        
        # Check 3: 4+ blank lines
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
    
    return ('\n'.join(content_lines), i)


def extract_content_phase2_pattern_b(lines, author_idx, authors_normalized, authors_original, intro_keywords):
    """
    PHASE 2 Pattern B: heading → content → author
    
    Extract heading and content before author.
    
    Args:
        lines (list): Document lines
        author_idx (int): Author line index
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
        intro_keywords (list): List of intro keywords
    
    Returns:
        tuple: (heading, content, heading_start_idx) or (None, None, -1)
    """
    if author_idx < 2:
        return (None, None, -1)
    
    # Check if 2-3 lines above have >25 chars
    has_long_content = False
    for offset in [2, 3]:
        check_idx = author_idx - offset
        if check_idx >= 0:
            check_line = lines[check_idx].strip()
            if len(check_line) > 25:
                has_long_content = True
                break
    
    if not has_long_content:
        return (None, None, -1)
    
    # Traverse upward to find heading
    content_lines = []
    heading = None
    heading_idx = -1
    
    for i in range(author_idx - 1, -1, -1):
        line_text = lines[i].strip()
        
        if not line_text:
            continue
        
        if len(line_text) <= 25:
            matched_author, _ = fuzzy_match_author(line_text, authors_normalized, authors_original)
            
            if not matched_author and '[' not in line_text and ':' not in line_text:
                if i > 0 and not lines[i - 1].strip():
                    heading = line_text
                    heading_idx = i
                    break
                elif i == 0:
                    heading = line_text
                    heading_idx = i
                    break
        
        content_lines.insert(0, lines[i].rstrip())
    
    if heading and content_lines:
        return (heading, '\n'.join(content_lines), heading_idx)
    
    return (None, None, -1)


def extract_remaining_content(lines, start_idx, processed_lines, authors_normalized, 
                              authors_original, intro_keywords):
    """
    PHASE 3: Extract remaining unprocessed content.
    
    Apply 4+ empty lines logic as stopping point.
    
    Args:
        lines (list): Document lines
        start_idx (int): Start index
        processed_lines (list): Boolean array
        authors_normalized (list): Normalized author names
        authors_original (list): Original author names
        intro_keywords (list): List of intro keywords
    
    Returns:
        tuple: (heading, content, end_index)
    """
    heading = ""
    content_lines = []
    i = start_idx
    
    # Try to extract heading from first line
    first_line = lines[i].strip()
    if first_line and len(first_line) <= 25:
        matched_author, _ = fuzzy_match_author(first_line, authors_normalized, authors_original)
        if not matched_author and '[' not in first_line and ':' not in first_line:
            if is_valid_heading(first_line):
                heading = first_line
                i += 1
    
    while i < len(lines):
        if processed_lines[i]:
            break
        
        line = lines[i]
        stripped = line.strip()
        
        # Check for 4+ blank lines
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
    
    content = '\n'.join(content_lines)
    return (heading, content, i)


def parse_tamil_document(file_path, shared_authors_dict):
    """
    Parse Tamil TXT document with final enhanced workflow.
    
    Sequential Processing:
    1. Extract metadata and authors
    2. PHASE 1: Extract intro sections (keywords)
    3. PHASE 2: Extract articles with authors (both patterns)
    4. PHASE 3: Extract remaining content
    
    Args:
        file_path (str): Path to TXT file
        shared_authors_dict (dict): Dictionary of shared authors
    
    Returns:
        dict: Parsed document data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Step 2.1: Extract Document Metadata
    doc_id, doc_issue = extract_doc_info(lines)
    
    # Step 2.2: Author List Extraction
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
        print(f"  ℹ No author list section found. Checking shared authors...")
        authors_original, authors_normalized = get_shared_authors(
            doc_id, doc_issue, shared_authors_dict
        )
        
        if not authors_original:
            print(f"  ℹ No shared authors. Using alternative extraction...")
            authors_original, authors_normalized = extract_authors_alternative(lines)
    else:
        for i in range(start_idx + 1, end_idx):
            author_name = lines[i].strip()
            if author_name:
                authors_original.append(author_name)
                authors_normalized.append(normalize_text(author_name))
    
    parse_start_idx = end_idx + 1 if end_idx != -1 else 0
    
    # Initialize processed lines tracker
    processed_lines = [False] * len(lines)
    if start_idx != -1 and end_idx != -1:
        for i in range(start_idx, end_idx + 1):
            processed_lines[i] = True
    
    intro_keywords = get_intro_keywords()
    articles = []
    intro = []
    article_no = 1
    
    # PHASE 1: Extract Intro Sections (Keywords)
    print(f"  → PHASE 1: Extracting intro sections...")
    i = parse_start_idx
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        
        # Check for intro keyword
        matched_keyword = None
        for keyword in intro_keywords:
            if keyword in line:
                if i == 0 or not lines[i - 1].strip():
                    matched_keyword = keyword
                    break
        
        if matched_keyword:
            content, content_end_idx = extract_intro_content_phase1(
                lines, i, processed_lines, authors_normalized, 
                authors_original, intro_keywords
            )
            
            # Mark as processed
            for j in range(i, content_end_idx):
                if j < len(lines):
                    processed_lines[j] = True
            
            content_line_count = len([ln for ln in content.split('\n') if ln.strip()])
            if content.strip() and count_words(content) >= 50 and content_line_count > 2:
                intro.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "heading": matched_keyword,
                    "content": content
                })
            
            i = content_end_idx
        else:
            i += 1
    
    # PHASE 2: Extract Articles with Authors
    print(f"  → PHASE 2: Extracting articles with authors...")
    i = parse_start_idx
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        # Look for author
        author_found, author_idx = find_author_in_range(
            lines, i, i + 1, authors_normalized, authors_original
        )
        
        if author_found:
            # Try Pattern A: heading → author → content
            heading, heading_idx = extract_heading_above_author(
                lines, author_idx, authors_normalized, authors_original
            )
            
            if heading and is_valid_heading(heading):
                # Extract content after author
                content, content_end_idx = extract_content_phase2_pattern_a(
                    lines, author_idx, heading_idx, processed_lines,
                    authors_normalized, authors_original, intro_keywords
                )
                
                # Mark as processed
                if heading_idx >= 0:
                    for j in range(heading_idx, content_end_idx):
                        if j < len(lines):
                            processed_lines[j] = True
                
                content_line_count = len([ln for ln in content.split('\n') if ln.strip()])
                if content.strip() and count_words(content) >= 50 and content_line_count > 2:
                    articles.append({
                        "doc_id": doc_id,
                        "doc_issue": doc_issue,
                        "article_no": article_no,
                        "article_heading": heading,
                        "article_author_name": author_found,
                        "article_content": content
                    })
                    article_no += 1
                
                i = content_end_idx
            else:
                # Try Pattern B: heading → content → author
                heading_b, content_b, heading_idx_b = extract_content_phase2_pattern_b(
                    lines, author_idx, authors_normalized, authors_original, intro_keywords
                )
                
                if heading_b and content_b and is_valid_heading(heading_b):
                    # Mark as processed
                    for j in range(heading_idx_b, author_idx + 1):
                        if j < len(lines):
                            processed_lines[j] = True
                    
                    content_line_count = len([ln for ln in content_b.split('\n') if ln.strip()])
                    if count_words(content_b) >= 50 and content_line_count > 2:
                        articles.append({
                            "doc_id": doc_id,
                            "doc_issue": doc_issue,
                            "article_no": article_no,
                            "article_heading": heading_b,
                            "article_author_name": author_found,
                            "article_content": content_b
                        })
                        article_no += 1
                    
                    i = author_idx + 1
                else:
                    i += 1
        else:
            i += 1
    
    # PHASE 3: Extract Remaining Content
    print(f"  → PHASE 3: Extracting remaining content...")
    i = parse_start_idx
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        
        # Skip large blank sections
        if not line:
            blank_count = count_consecutive_blanks(lines, i)
            if blank_count >= 4:
                for j in range(i, i + blank_count):
                    if j < len(lines):
                        processed_lines[j] = True
                i += blank_count
                continue
        
        if line:
            heading, content, content_end_idx = extract_remaining_content(
                lines, i, processed_lines, authors_normalized, 
                authors_original, intro_keywords
            )
            
            # Mark as processed
            for j in range(i, content_end_idx):
                if j < len(lines):
                    processed_lines[j] = True
            
            content_line_count = len([ln for ln in content.split('\n') if ln.strip()])
            
            # Only add if heading is valid
            if heading and is_valid_heading(heading) and content.strip() and count_words(content) >= 50 and content_line_count > 2:
                articles.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "article_no": article_no,
                    "article_heading": heading,
                    "article_author_name": "NA",
                    "article_content": content
                })
                article_no += 1
            
            i = content_end_idx
        else:
            i += 1
    
    # Prepare authors list
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
    Build dictionary of authors indexed by (doc_id, doc_issue).
    
    Args:
        root_folder_path (Path): Root folder path
    
    Returns:
        dict: Dictionary mapping (doc_id, doc_issue) to author lists
    """
    print("Building shared authors dictionary...")
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
                    print(f"  ✓ Found authors for {doc_id}/{doc_issue}: {len(authors_original)} authors")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    print(f"Dictionary built: {len(shared_authors)} keys\n")
    return shared_authors


def process_folder(root_folder_path, output_root_folder="output_json"):
    """
    Process all TXT files in root folder.
    
    Creates:
    - {filename}.json for each document
    - {folder}_authors.json for each folder (consolidated, no duplicates)
    
    Args:
        root_folder_path (str): Root folder path
        output_root_folder (str): Output folder path
    """
    root_path = Path(root_folder_path)
    output_root = Path(output_root_folder)
    
    if not root_path.exists():
        print(f"Error: Folder '{root_folder_path}' does not exist!")
        return
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(root_path.rglob("*.txt"))
    
    if not txt_files:
        print(f"No TXT files found in '{root_folder_path}'.")
        return
    
    print(f"Found {len(txt_files)} TXT file(s)\n")
    
    # Build shared authors dictionary
    shared_authors_dict = build_shared_authors_dict(root_path)
    
    # Track authors by folder and (doc_id, doc_issue)
    folder_authors = defaultdict(lambda: defaultdict(set))
    
    # Process each file
    for txt_file in txt_files:
        try:
            print(f"Processing: {txt_file.name}")
            
            result = parse_tamil_document(str(txt_file), shared_authors_dict)
            
            relative_path = txt_file.relative_to(root_path)
            output_folder = output_root / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            base_name = txt_file.stem
            
            # Create {filename}.json
            content_json_path = output_folder / f"{base_name}.json"
            with open(content_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "intro": result["intro"],
                    "articles": result["articles"]
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Created: {content_json_path.name}")
            
            # Collect authors for folder (grouped by doc_id/doc_issue)
            folder_key = relative_path.parent
            doc_key = (result["doc_id"], result["doc_issue"])
            for author_info in result["authors_list"]:
                folder_authors[folder_key][doc_key].add(author_info["author_name"])
            
            print(f"  ✓ Processed successfully\n")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
            continue
    
    # Create consolidated authors files per folder
    print("Creating consolidated authors files...")
    for folder_key, doc_authors_dict in folder_authors.items():
        output_folder = output_root / folder_key
        folder_name = folder_key.name if folder_key.name else "root"
        
        # Build consolidated structure
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
        
        print(f"  ✓ Created: {authors_json_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Output folder: {output_root.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tamil TXT Document Parser - Enhanced Version")
    print("=" * 60)
    print()

    # --- Set your folder paths here ---
    root_folder = "extracted_texts"
    output_folder = "output"

    print(f"\nInput folder: {root_folder}")
    print(f"Output folder: {output_folder}\n")

    print("=" * 60)
    print()

    process_folder(root_folder, output_folder)
