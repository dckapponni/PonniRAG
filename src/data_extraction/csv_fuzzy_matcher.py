import re
import logging
import pandas as pd
from difflib import SequenceMatcher

logger = logging.getLogger('TamilDocProcessor.csv_fuzzy_matcher')


def calculate_similarity(str1, str2):
    """
    Calculate similarity percentage between two strings.

    Uses Python's SequenceMatcher to compute a ratio-based similarity score
    between two strings, useful for fuzzy matching of titles and text.

    Args:
        str1 (str): First string to compare.
        str2 (str): Second string to compare.

    Returns:
        float: Similarity percentage from 0 to 100. Returns 0 if either string
            is empty or None.
    """
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1, str2).ratio() * 100


def remove_symbols(text):
    """
    Remove all punctuation and symbols, keeping only letters and spaces.

    Strips punctuation, special characters, and symbols from text while preserving
    Unicode letters (including Tamil characters) and spaces. Used for normalizing
    text before fuzzy matching.

    Args:
        text (str or any): Text to clean. Can handle pandas NA values and non-string types.

    Returns:
        str: Cleaned text with only letters and single spaces. Returns empty string
            if input is NA or None.

    Processing Steps:
        1. Converts to string and strips whitespace
        2. Removes all non-word, non-space characters (Unicode-aware)
        3. Collapses multiple spaces into single space
        4. Strips leading/trailing whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_malar_issue_from_text(text):
    """
    Extract மலர் (volume) and இதழ் (issue) numbers from document text.

    Searches the first 100 lines of text for மலர் and இதழ் identifiers using
    multiple regex patterns. Handles special cases like பொங்கல் மலர் (Pongal special).

    Args:
        text (str): Full document text (multi-line string).

    Returns:
        tuple: (malar, issue) where both are strings or None.
            - malar (str or None): Volume number or special name like "பொங்கல்"
            - issue (str or None): Issue number

    Search Patterns:
        For மலர்: "மலர்: X", "மலர் எண்: X", "தொகுதி: X", "பொங்கல் மலர்"
        For இதழ்: "இதழ்: X", "எண்: X", "இல. X"
    """
    lines = text.split('\n')[:100]
    first_pages = '\n'.join(lines)
    
    malar = None
    issue = None
    
    # Check for பொங்கல் மலர்
    pongal_match = re.search(r'பொங்கல்\s*மலர்', first_pages)
    if pongal_match:
        malar = "பொங்கல்"
        issue_match = re.search(r'பொங்கல்\s*மலர்.*?இதழ்\s*[:-]?\s*(\d+)', first_pages, re.DOTALL)
        if issue_match:
            issue = issue_match.group(1)
    
    # Extract மலர்
    if not malar:
        malar_patterns = [
            r'மலர்\s*[:-]?\s*(\d+)',
            r'மலர்\s+எண்\s*[:-]?\s*(\d+)',
            r'தொகுதி\s*[:-]?\s*(\d+)'
        ]
        for pattern in malar_patterns:
            match = re.search(pattern, first_pages)
            if match:
                malar = match.group(1).strip()
                break
    
    # Extract இதழ்
    if not issue:
        issue_patterns = [
            r'இதழ்\s*[:-]?\s*(\d+)',
            r'எண்\s*[:-]?\s*(\d+)',
            r'இல\.\s*(\d+)'
        ]
        for pattern in issue_patterns:
            match = re.search(pattern, first_pages)
            if match:
                issue = match.group(1).strip()
                break
    
    return malar, issue


def normalize_csv_value(val):
    """
    Normalize CSV values for consistent comparison.

    Converts float-formatted strings like '1.0' to integer strings like '1',
    and strips whitespace. Ensures consistent formatting between CSV data
    and extracted document metadata.

    Args:
        val (any): Value from CSV cell (can be string, float, int, or pandas NA).

    Returns:
        str: Normalized string value. Returns empty string if input is NA or None.

    Normalization Rules:
        - Strips leading/trailing whitespace
        - Converts "1.0" → "1" (float strings to integers when appropriate)
        - Preserves non-numeric strings as-is
        - Returns "" for NA/None values

    """
    if pd.isna(val):
        return ''
    
    val_str = str(val).strip()
    if '.' in val_str:
        try:
            float_val = float(val_str)
            if float_val == int(float_val):
                return str(int(float_val))
        except:
            pass
    return val_str


def find_article_boundary_fuzzy(lines, title, next_title=None):
    """
    Find article content boundaries using fuzzy title matching.

    Locates an article in document lines by fuzzy-matching its title, then
    extracts content from the matched title to either the next article's title
    or end of document.

    Args:
        lines (list): List of text lines from the document.
        title (str): Article title to search for.
        next_title (str, optional): Title of the next article for boundary detection.
                                If None, extracts to end of document.

    Returns:
        str or None: Extracted article content including title line, or None if:
            - Title match score < 70%
            - No match found

    Matching Logic:
        - Start: Finds line with highest similarity to title (minimum 70% required)
        - End: Stops at line matching next_title (>80% similarity) or document end
        - Returns all lines from start to end as joined string
    """
    title_clean = remove_symbols(title)
    
    start_idx = -1
    best_score = 0
    
    # Find start
    for i, line in enumerate(lines):
        line_clean = remove_symbols(line)
        if not line_clean:
            continue
        
        similarity = calculate_similarity(title_clean, line_clean)
        if similarity > best_score:
            start_idx = i
            best_score = similarity
            if similarity > 90:
                break
    
    if start_idx == -1 or best_score < 70:
        return None
    
    # Find end
    end_idx = len(lines)
    if next_title:
        next_title_clean = remove_symbols(next_title)
        for i in range(start_idx + 1, len(lines)):
            line_clean = remove_symbols(lines[i])
            if line_clean:
                similarity = calculate_similarity(next_title_clean, line_clean)
                if similarity > 80:
                    end_idx = i
                    break
    
    content_lines = lines[start_idx:end_idx]
    content = '\n'.join(content_lines).strip()
    
    return content


def extract_articles_from_csv(lines, csv_df, file_path):
    
    """
    Extract articles using CSV fuzzy matching when மலர்/இதழ் extraction fails.
    Args:
        lines (list): List of text lines from the document.
        csv_df (pd.DataFrame): DataFrame containing article metadata with columns:
                            மலர், இதழ், தலைப்பு, ஆசிரியர், ஆண்டு
        file_path (Path-like): File path object with .name attribute for source document.

    Returns:
        dict or None: Extraction results containing:
            - articles (list): List of article dictionaries with fields:
                - doc_id, doc_issue, article_no, author_name, title, content,
                year, source_document
            - authors_list (list): List of unique author dictionaries
            - doc_id (str): Normalized மலர் identifier
            - doc_issue (str): Normalized இதழ் number
            
            Returns None if:
            - CSV DataFrame is None
            - Cannot extract மலர்/இதழ் from document
            - No matching articles found in CSV
    """
    logger.info("=" * 80)
    logger.info("CSV FUZZY MATCHING FALLBACK ACTIVATED")
    logger.info(f"File: {file_path}")
    logger.info("=" * 80)
    
    if csv_df is None:
        logger.error("CSV DataFrame is None, cannot proceed with fallback")
        return None
    
    text = '\n'.join(lines)
    source_document = file_path.name
    
    # Try to extract மலர்/இதழ் from content
    doc_malar, doc_issue = extract_malar_issue_from_text(text)
    
    if not doc_malar or not doc_issue:
        logger.error("Could not extract மலர்/இதழ் from content for CSV matching")
        return None
    
    logger.info(f"Extracted from content: மலர்={doc_malar}, இதழ்={doc_issue}")
    
    # Normalize for comparison
    doc_malar_norm = normalize_csv_value(doc_malar)
    doc_issue_norm = normalize_csv_value(doc_issue)
    
    # Create normalized columns
    csv_df['மலர்_norm'] = csv_df['மலர்'].apply(normalize_csv_value)
    csv_df['இதழ்_norm'] = csv_df['இதழ்'].apply(normalize_csv_value)
    
    # Find matching articles
    matched_articles = csv_df[
        (csv_df['மலர்_norm'] == doc_malar_norm) & 
        (csv_df['இதழ்_norm'] == doc_issue_norm)
    ]
    
    if matched_articles.empty:
        logger.warning(f"No matching articles in CSV for மலர்={doc_malar_norm}, இதழ்={doc_issue_norm}")
        return None
    
    logger.info(f"Found {len(matched_articles)} matching articles in CSV")
    
    articles = []
    article_no = 1
    
    for idx_pos, (idx, row) in enumerate(matched_articles.iterrows()):
        title = row['தலைப்பு']
        author = row['ஆசிரியர்'] if pd.notna(row['ஆசிரியர்']) else "NA"
        year = str(int(row['ஆண்டு'])) if pd.notna(row['ஆண்டு']) else "Unknown"
        
        # Get next title for boundary detection
        next_title = None
        if idx_pos + 1 < len(matched_articles):
            next_row = matched_articles.iloc[idx_pos + 1]
            next_title = next_row['தலைப்பு']
        
        # Extract content
        content = find_article_boundary_fuzzy(lines, title, next_title)
        
        if content and len(content) > 100:
            # NEW FORMAT: Changed field names to match requirements
            articles.append({
                "doc_id": doc_malar_norm,
                "doc_issue": doc_issue_norm,
                "article_no": article_no,
                "author_name": str(author), 
                "title": str(title), 
                "content": content,  
                "year": year, 
                "source_document": source_document  
            })
            logger.info(f" CSV: {title[:40]}... ({len(content)} chars, year: {year})")
            article_no += 1
        else:
            logger.debug(f" CSV: {title[:40]}... (not found or too short)")
    
    logger.info(f"CSV Fuzzy Matching extracted {len(articles)} articles")
    
    # Build authors list
    authors_list = []
    unique_authors = set()
    for _, row in matched_articles.iterrows():
        if pd.notna(row['ஆசிரியர்']):
            author = str(row['ஆசிரியர்'])
            if author != "NA" and author not in unique_authors:
                unique_authors.add(author)
                authors_list.append({
                    "doc_id": doc_malar_norm,
                    "doc_issue": doc_issue_norm,
                    "author_name": author
                })
    
    return {
        "articles": articles,  
        "authors_list": authors_list,
        "doc_id": doc_malar_norm,
        "doc_issue": doc_issue_norm
    }