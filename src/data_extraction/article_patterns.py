import logging
import re
from text_processing import extract_author_from_line, is_valid_heading
from doc_utils import count_content_lines
from content_extraction import count_consecutive_blanks

logger = logging.getLogger('TamilDocProcessor.article_patterns')


def extract_pattern_a_forward(lines, start_idx, end_idx, authors_normalized, 
                               authors_original, processed_lines, intro_keywords):
    """
    Extract articles following Pattern A: HEADING → AUTHOR → CONTENT (forward scan).
    
    Args:
        lines (list): List of text lines from document
        start_idx (int): Starting index for extraction
        end_idx (int): Ending index for extraction
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        processed_lines (list): Boolean list tracking which lines have been processed
        intro_keywords (list): List of introduction keywords to detect section boundaries
        
    Returns:
        list: List of extracted article dictionaries containing heading, author, content, and indices
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PATTERN A: Starting extraction from line {start_idx} to {end_idx}")
    logger.info("=" * 80)
    
    articles = []
    i = start_idx
    
    try:
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
                logger.debug(f"Pattern A: Found author '{matched_author}' at line {author_idx}")
                
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
                                logger.debug(f"Pattern A: Found heading '{heading}' at line {heading_idx}")
                                break
                
                if not heading:
                    logger.debug(f"Pattern A: No valid heading found before author")
                    i += 1
                    continue
                
                content_start = author_idx + 1
                while content_start < end_idx and not lines[content_start].strip() and (content_start - author_idx) <= 3:
                    content_start += 1
                
                if content_start >= end_idx:
                    logger.debug(f"Pattern A: No content after author")
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
                            logger.debug(f"Pattern A: Stopped at next author at line {j}")
                            break
                    
                    if stripped:
                        found_intro = False
                        for keyword in intro_keywords:
                            if keyword in stripped:
                                while content_lines and not content_lines[-1].strip():
                                    content_lines.pop()
                                found_intro = True
                                break
                        if found_intro:
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
                    
                    logger.info(f"Pattern A: Extracted '{heading}' by {matched_author} ({count_content_lines(content)} lines)")
                    i = j
                    continue
            
            i += 1
        
        logger.info(f"PATTERN A: Extracted {len(articles)} articles")
        logger.info("=" * 80)
        return articles
        
    except Exception as e:
        logger.error(f"Pattern A: Error at line {i}: {e}", exc_info=True)
        return articles


def extract_pattern_b_forward(lines, start_idx, end_idx, authors_normalized, 
                               authors_original, processed_lines, intro_keywords):
    """
    Extract articles following Pattern B: AUTHOR → HEADING → CONTENT (forward scan).
    
    Args:
        lines (list): List of text lines from document
        start_idx (int): Starting index for extraction
        end_idx (int): Ending index for extraction
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        processed_lines (list): Boolean list tracking which lines have been processed
        intro_keywords (list): List of introduction keywords to detect section boundaries
        
    Returns:
        list: List of extracted article dictionaries containing heading, author, content, and indices
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PATTERN B: Starting extraction from line {start_idx} to {end_idx}")
    logger.info("=" * 80)
    
    articles = []
    i = start_idx
    
    try:
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
                logger.debug(f"Pattern B: Found author '{matched_author}' at line {author_idx}")
                
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
                                logger.debug(f"Pattern B: Found heading '{heading}' at line {heading_idx}")
                                break
                
                if not heading:
                    logger.debug(f"Pattern B: No valid heading found after author")
                    i += 1
                    continue
                
                content_start = heading_idx + 1
                while content_start < end_idx and not lines[content_start].strip() and (content_start - heading_idx) <= 3:
                    content_start += 1
                
                if content_start >= end_idx:
                    logger.debug(f"Pattern B: No content after heading")
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
                            logger.debug(f"Pattern B: Stopped at next author at line {j}")
                            break
                    
                    if stripped:
                        found_intro = False
                        for keyword in intro_keywords:
                            if keyword in stripped:
                                while content_lines and not content_lines[-1].strip():
                                    content_lines.pop()
                                found_intro = True
                                break
                        if found_intro:
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
                    
                    logger.info(f"Pattern B: Extracted '{heading}' by {matched_author} ({count_content_lines(content)} lines)")
                    i = j
                    continue
            
            i += 1
        
        logger.info(f"PATTERN B: Extracted {len(articles)} articles")
        logger.info("=" * 80)
        return articles
        
    except Exception as e:
        logger.error(f"Pattern B: Error at line {i}: {e}", exc_info=True)
        return articles


def find_author_in_content_end(content_lines, authors_normalized, authors_original):
    """
    Check if author name appears in the last few lines of content with enhanced matching for Tamil poems.
    
    Args:
        content_lines (list): List of content lines to check
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        
    Returns:
        tuple: (author_name or None, modified_content_lines)
    """
    if not content_lines:
        return (None, content_lines)
    
    check_lines = min(20, len(content_lines))
    
    logger.debug(f"Pattern C: Checking last {check_lines} lines for embedded author")
    logger.debug(f"   Total content lines: {len(content_lines)}")
    
    if len(content_lines) >= 5:
        logger.debug(f"   Last 5 lines preview:")
        for i in range(max(0, len(content_lines) - 5), len(content_lines)):
            logger.debug(f"     [{i}]: '{content_lines[i][:60]}...'")
    
    for i in range(len(content_lines) - 1, max(len(content_lines) - check_lines - 1, -1), -1):
        line = content_lines[i]
        line_stripped = line.strip()
        
        if not line_stripped:
            continue
        
        if line_stripped.startswith('—') or line_stripped.startswith('-') or line_stripped.startswith('–'):
            clean_line = line_stripped.lstrip('—-– \u2014\u2013\u2012').strip()
            logger.debug(f"   Found dash line at index {i}: '{line_stripped}' -> cleaned: '{clean_line}'")
            
            for idx, author_orig in enumerate(authors_original):
                if clean_line == author_orig:
                    logger.info(f"EXACT MATCH: '{author_orig}'")
                    modified_content = content_lines[:i]
                    while modified_content and not modified_content[-1].strip():
                        modified_content.pop()
                    return (author_orig, modified_content)
                
                if author_orig in clean_line or clean_line in author_orig:
                    logger.info(f"PARTIAL MATCH: '{author_orig}' ~ '{clean_line}'")
                    modified_content = content_lines[:i]
                    while modified_content and not modified_content[-1].strip():
                        modified_content.pop()
                    return (author_orig, modified_content)
                
                clean_no_space = clean_line.replace(' ', '')
                author_no_space = author_orig.replace(' ', '')
                if clean_no_space == author_no_space or clean_no_space in author_no_space or author_no_space in clean_no_space:
                    logger.info(f"SPACE-NORMALIZED MATCH: '{author_orig}' ~ '{clean_line}'")
                    modified_content = content_lines[:i]
                    while modified_content and not modified_content[-1].strip():
                        modified_content.pop()
                    return (author_orig, modified_content)
        
        for author_orig in authors_original:
            if line_stripped == author_orig:
                logger.info(f"EXACT STANDALONE: '{author_orig}'")
                modified_content = content_lines[:i]
                while modified_content and not modified_content[-1].strip():
                    modified_content.pop()
                return (author_orig, modified_content)
        
        for author_orig in authors_original:
            if author_orig in line_stripped:
                logger.info(f"FUZZY MATCH: '{author_orig}' in '{line_stripped[:50]}...'")
                modified_content = content_lines[:i]
                while modified_content and not modified_content[-1].strip():
                    modified_content.pop()
                return (author_orig, modified_content)
        
        if 5 <= len(line_stripped) <= 35:
            skip_words = ['என்று', 'என்ற', 'என்பது', 'போன்ற', 'என்றால்', 
                         'என்னும்', 'என்ற', 'என்கிறார்', 'என்கிறது']
            if any(word in line_stripped for word in skip_words):
                continue
            
            for author_orig in authors_original:
                if line_stripped in author_orig or author_orig in line_stripped:
                    match_ratio = len(line_stripped) / len(author_orig)
                    if match_ratio > 0.4:
                        logger.info(f"SHORT LINE MATCH: '{author_orig}' ~ '{line_stripped}'")
                        modified_content = content_lines[:i]
                        while modified_content and not modified_content[-1].strip():
                            modified_content.pop()
                        return (author_orig, modified_content)
    
    logger.debug(f"No embedded author found")
    return (None, content_lines)


def extract_pattern_c_reverse(lines, start_idx, end_idx, authors_normalized, 
                               authors_original, processed_lines, intro_keywords):
    """
    Extract articles following Pattern C with two scenarios:
    1. STANDALONE: HEADING → CONTENT → AUTHOR (reverse scan)
    2. EMBEDDED: HEADING → CONTENT (author at end - common in poems)
    
    Args:
        lines (list): List of text lines from document
        start_idx (int): Starting index for extraction
        end_idx (int): Ending index for extraction
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        processed_lines (list): Boolean list tracking which lines have been processed
        intro_keywords (list): List of introduction keywords to detect section boundaries
        
    Returns:
        list: List of extracted article dictionaries containing heading, author, content, and indices
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PATTERN C: Starting extraction from line {start_idx} to {end_idx}")
    logger.info(f"PATTERN C: {len(authors_original)} authors available")
    if authors_original:
        logger.info(f"PATTERN C: Sample authors: {', '.join(authors_original[:5])}")
    logger.info("=" * 80)
    
    articles = []
    i = start_idx
    
    heading_candidates_checked = 0
    standalone_author_found = 0
    embedded_author_found = 0
    
    try:
        while i < end_idx:
            if processed_lines[i]:
                i += 1
                continue
            
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if this line is a standalone author (PATTERN C - STANDALONE)
            matched_author = extract_author_from_line(line, authors_normalized, authors_original)
            
            if matched_author:
                author_idx = i
                logger.debug(f"Standalone: Found '{matched_author}' at {author_idx}")
                
                # Search backwards for heading, skipping blank lines
                heading_search_start = author_idx - 1
                while heading_search_start >= start_idx and not lines[heading_search_start].strip():
                    heading_search_start -= 1
                
                if heading_search_start < start_idx:
                    i += 1
                    continue
                
                # The last non-blank line before author is potential end of content
                content_end_idx = heading_search_start
                heading = None
                heading_idx = -1
            
                j = content_end_idx
                
                while j >= start_idx:
                    if processed_lines[j]:
                        break
                    
                    current_line = lines[j].strip()
                    if not current_line:
                        j -= 1
                        continue
                    
                    # Stop if we hit another author
                    prev_author = extract_author_from_line(lines[j], authors_normalized, authors_original)
                    if prev_author:
                        break
                    
                    # Stop if we hit an intro keyword
                    is_intro = False
                    for keyword in intro_keywords:
                        if keyword in current_line:
                            is_intro = True
                            break
                    if is_intro:
                        break
                    
                    # Check if this could be a heading (short line)
                    if len(current_line) <= 60:
                        # Check if there's a blank line IMMEDIATELY before (j-1)
                        has_blank_immediately_before = (j > start_idx and not lines[j-1].strip())
                        is_at_start = (j == start_idx)
                        
                        logger.debug(f"   Checking line {j}: '{current_line[:40]}...' blank_before={has_blank_immediately_before}, at_start={is_at_start}")
                        
                        if has_blank_immediately_before or is_at_start:
                            heading = current_line
                            heading_idx = j
                            logger.debug(f"   Heading: '{heading}' at {heading_idx}")
                            break
                    
                    j -= 1
                
                # If we found a heading, extract the content between heading and author
                if heading:
                    content_lines = []
                    for k in range(heading_idx + 1, content_end_idx + 1):
                        if k < len(lines) and not processed_lines[k]:
                            content_lines.append(lines[k].rstrip())
                    
                    # Clean up trailing blank lines
                    while content_lines and not content_lines[-1].strip():
                        content_lines.pop()
                    # Clean up leading blank lines
                    while content_lines and not content_lines[0].strip():
                        content_lines.pop(0)
                    
                    content = '\n'.join(content_lines)
                    
                    # Require at least 2 content lines
                    if content.strip() and count_content_lines(content) >= 2:
                        # Mark all lines from heading to author as processed
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
                        
                        standalone_author_found += 1
                        logger.info(f"STANDALONE: '{heading}' by {matched_author}")
                        i = author_idx + 1
                        continue
            
            # Check if this line could be a heading for EMBEDDED author pattern
            if len(line) <= 60 and len(line) >= 3:
                # Skip lines that look like part of content
                skip_words = ['என்று', 'என்ற', 'என்பது', 'என்றால்', 'என்னும்']
                if any(word in line for word in skip_words):
                    i += 1
                    continue
                
                heading_idx = i
                heading_candidates_checked += 1
                
                logger.debug(f"Checking heading candidate at {i}: '{line[:50]}...'")
                
                # Count blank lines before and after
                ba = sum(1 for k in range(heading_idx-1, max(heading_idx-3, start_idx-1), -1) 
                        if k >= start_idx and not lines[k].strip())
                bb = sum(1 for k in range(heading_idx+1, min(heading_idx+4, len(lines))) 
                        if not lines[k].strip())
                
                logger.debug(f"   Blanks: above={ba}, below={bb}")
                
                # Accept as heading if has blank lines around it OR at document start
                if ba >= 1 or bb >= 1 or heading_idx == start_idx:
                    heading = line
                    content_start_idx = heading_idx + bb + 1
                    
                    # Collect content lines
                    content_lines = []
                    j = content_start_idx
                    
                    while j < end_idx:
                        if processed_lines[j]:
                            break
                        
                        current_line = lines[j]
                        stripped = current_line.strip()
                        
                        # Stop if we hit another author (standalone)
                        if stripped:
                            next_author = extract_author_from_line(current_line, authors_normalized, authors_original)
                            if next_author:
                                break
                        
                        # Stop if we hit what looks like another heading
                        if stripped and len(stripped) <= 60:
                            next_ba = sum(1 for k in range(j-1, max(j-3, start_idx-1), -1) 
                                         if k >= start_idx and not lines[k].strip())
                            if next_ba >= 2:
                                break
                        
                        # Stop if we hit an intro keyword
                        if stripped:
                            found_intro = False
                            for keyword in intro_keywords:
                                if keyword in stripped:
                                    found_intro = True
                                    break
                            if found_intro:
                                # Remove trailing blanks before breaking
                                while content_lines and not content_lines[-1].strip():
                                    content_lines.pop()
                                break
                        
                        # Stop if we hit too many consecutive blank lines
                        if not stripped:
                            blank_count = count_consecutive_blanks(lines, j)
                            if blank_count >= 4:
                                break
                        
                        content_lines.append(current_line.rstrip())
                        j += 1
                    
                    # Clean up trailing blank lines
                    while content_lines and not content_lines[-1].strip():
                        content_lines.pop()
                    
                    # Check for embedded author in content
                    if len(content_lines) >= 3:
                        logger.debug(f"   Content has {len(content_lines)} lines, checking for embedded author...")
                        
                        embedded_author, modified_content = find_author_in_content_end(
                            content_lines, authors_normalized, authors_original
                        )
                        
                        if embedded_author:
                            content = '\n'.join(modified_content)
                            
                            if content.strip() and count_content_lines(content) >= 2:
                                # Mark all lines as processed
                                for k in range(heading_idx, j):
                                    if k < len(lines):
                                        processed_lines[k] = True
                                
                                articles.append({
                                    "heading": heading,
                                    "author": embedded_author,
                                    "content": content,
                                    "start_idx": heading_idx,
                                    "end_idx": j
                                })
                                
                                embedded_author_found += 1
                                logger.info(f"EMBEDDED: '{heading[:40]}...' by {embedded_author}")
                                i = j
                                continue
                        else:
                            logger.debug(f"   No embedded author found for '{heading[:40]}...'")
            
            i += 1
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"PATTERN C SUMMARY:")
        logger.info(f"  Heading candidates: {heading_candidates_checked}")
        logger.info(f"  Standalone: {standalone_author_found}")
        logger.info(f"  Embedded: {embedded_author_found}")
        logger.info(f"  TOTAL: {len(articles)}")
        logger.info("=" * 80)
        
        return articles
        
    except Exception as e:
        logger.error(f"Pattern C: Error at line {i}: {e}", exc_info=True)
        return articles

def extract_articles_main(lines, start_idx, end_idx, authors_normalized, 
                          authors_original, intro_keywords):
    """
    Main extraction function with correct priority order:
    Pattern C → Pattern A → Pattern B → Intro/Remaining
    
    Args:
        lines (list): List of text lines from document
        start_idx (int): Starting index for extraction
        end_idx (int): Ending index for extraction
        authors_normalized (list): List of normalized author names
        authors_original (list): List of original author names
        intro_keywords (list): List of introduction keywords to detect section boundaries
        
    Returns:
        list: List of all extracted articles from all patterns
    """
    logger = logging.getLogger('TamilDocProcessor')
    
    processed_lines = [False] * len(lines)
    all_articles = []
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING ARTICLE EXTRACTION")
    logger.info(f"Range: {start_idx} to {end_idx}")
    logger.info(f"Authors available: {len(authors_original)}")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("STEP 1: Running Pattern C (poems and embedded authors)")
    
    pattern_c_articles = extract_pattern_c_reverse(
        lines, start_idx, end_idx,
        authors_normalized, authors_original,
        processed_lines, intro_keywords
    )
    
    all_articles.extend(pattern_c_articles)
    logger.info(f"Pattern C extracted: {len(pattern_c_articles)} articles")
    
    logger.info("")
    logger.info("STEP 2: Running Pattern A (heading -> author -> content)")
    
    pattern_a_articles = extract_pattern_a_forward(
        lines, start_idx, end_idx,
        authors_normalized, authors_original,
        processed_lines, intro_keywords
    )
    
    all_articles.extend(pattern_a_articles)
    logger.info(f"Pattern A extracted: {len(pattern_a_articles)} articles")
    
    logger.info("")
    logger.info("STEP 3: Running Pattern B (author -> heading -> content)")
    
    pattern_b_articles = extract_pattern_b_forward(
        lines, start_idx, end_idx,
        authors_normalized, authors_original,
        processed_lines, intro_keywords
    )
    
    all_articles.extend(pattern_b_articles)
    logger.info(f"Pattern B extracted: {len(pattern_b_articles)} articles")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTION SUMMARY")
    logger.info(f"  Pattern C (poems): {len(pattern_c_articles)}")
    logger.info(f"  Pattern A: {len(pattern_a_articles)}")
    logger.info(f"  Pattern B: {len(pattern_b_articles)}")
    logger.info(f"  Total articles: {len(all_articles)}")
    logger.info("=" * 80)
    

    return all_articles
