import re
from text_processing import normalize_text

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

