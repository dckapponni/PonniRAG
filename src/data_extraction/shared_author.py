from difflib import SequenceMatcher
from pathlib import Path
from doc_utils  import extract_doc_info
from text_processing   import find_author_in_range, normalize_text

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


