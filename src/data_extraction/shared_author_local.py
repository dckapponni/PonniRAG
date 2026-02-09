import logging
import re
from pathlib import Path
from doc_utils import extract_doc_info
from text_processing import normalize_text

logger = logging.getLogger('TamilDocProcessor.shared_author_local')


def check_author_ahead(lines, current_idx, authors_normalized, authors_original, lookback=3):
    """
    Check if an author name appears in the next few lines.
    Searches forward from the current position to detect if any known author name
    appears within a specified lookahead window. Used to stop content extraction
    before reaching the next author's section.

    Args:

        lines (list): List of text lines from the document.

        current_idx (int): Current line index to start searching from.

        authors_normalized (list): List of normalized author names for matching.

        authors_original (list): List of original author names (parallel to normalized).

        lookback (int, optional): Number of lines to look ahead. Defaults to 3.

    Returns:

        int or None: Line index where author name was found, or None if no author

                    found within lookahead window or on error.
    """
    try:
        from text_processing import extract_author_from_line
        
        for i in range(current_idx, min(current_idx + lookback + 1, len(lines))):
            if i >= len(lines):
                break
                
            line = lines[i].strip()
            if not line:
                continue
            
            author = extract_author_from_line(line, authors_normalized, authors_original)
            if author:
                logger.debug(f"Found author '{author}' ahead at line {i}")
                return i
        
        return None
        
    except Exception as e:
        logger.error(f"Error in check_author_ahead: {e}")
        return None


def build_shared_authors_dict_local(input_dir):
    """

    Build dictionary mapping document IDs to their authors from local TXT files.
    Scans all TXT files in a directory (recursively), extracts மலர்/இதழ் and author
    information from each document's TOC section, and builds a lookup dictionary.
    This dictionary is used as a fallback when individual document author extraction fails.

    Args:

        input_dir (Path or str): Directory path containing TXT files to process.

    Returns:

        dict: Dictionary mapping (doc_id, doc_issue) tuples to (authors_original, 

            authors_normalized) tuples. Returns empty dict on error.

            - Key: (doc_id, doc_issue) tuple of strings

            - Value: (authors_original, authors_normalized) tuple of lists

    """
    logger.info(f"Building shared authors dictionary from: {input_dir}")
    
    shared_authors = {}
    
    try:
        txt_files = list(input_dir.rglob("*.txt"))
        logger.info(f"Found {len(txt_files)} TXT files to process")
        
        if not txt_files:
            logger.warning("No TXT files found")
            return {}
        
        processed_count = 0
        error_count = 0
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                lines = content.splitlines()
                
                doc_id, doc_issue = extract_doc_info(lines)
                
                if doc_id == "NA" or doc_issue == "NA":
                    continue
                
                start_idx = -1
                end_idx = -1
                
                for i, line in enumerate(lines):
                    if "பொருளடக்கம்" in line:
                        start_idx = i
                    if "ஆகியோரின் எழுத்தோவியங்கள்" in line or "ஆகியோரின்" in line:
                        end_idx = i
                        break
                
                if start_idx != -1 and end_idx != -1:
                    authors_original = []
                    authors_normalized = []
                    
                    for i in range(start_idx + 1, end_idx):
                        if i >= len(lines):
                            break
                            
                        author_name = lines[i].strip()
                        if author_name and len(author_name) > 2:
                            if not re.match(r'^[\d.\s…]+$', author_name):
                                authors_original.append(author_name)
                                authors_normalized.append(normalize_text(author_name))
                    
                    key = (doc_id, doc_issue)
                    if key not in shared_authors and authors_original:
                        shared_authors[key] = (authors_original, authors_normalized)
                        processed_count += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {txt_file.name}: {e}")
                error_count += 1
        
        logger.info(f"Shared authors dictionary built: {len(shared_authors)} documents")
        logger.info(f"Success: {processed_count}, Errors: {error_count}")
        
        return shared_authors
        
    except Exception as e:
        logger.error(f"Error building shared authors dictionary: {e}", exc_info=True)
        return {}