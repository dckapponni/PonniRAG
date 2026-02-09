import logging
from pathlib import Path

from s3_utils import list_files, read_text_from_s3

logger = logging.getLogger('TamilDocProcessor.shared_author')


def check_author_ahead(lines, current_idx, authors_normalized, authors_original, lookback=3):
    """
    Check if an author name appears in the next few lines.
    
    Args:
        lines (list): List of text lines
        current_idx (int): Current line index to start searching from
        authors_normalized (list): List of normalized author names for matching
        authors_original (list): List of original author names
        lookback (int): Number of lines to look ahead (default: 3)
        
    Returns:
        int or None: Index of author line if found, None otherwise
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
        
    except IndexError as e:
        logger.warning(f"Index error in check_author_ahead at line {current_idx}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in check_author_ahead at line {current_idx}: {e}")
        return None


def build_shared_authors_dict_s3(bucket, input_prefix):
    """
    Build a dictionary mapping document IDs to their authors across all S3 files.
    
    Args:
        bucket (str): S3 bucket name
        input_prefix (str): S3 prefix for input files
        
    Returns:
        dict: Dictionary mapping (doc_id, doc_issue) to (authors_original, authors_normalized)
    """
    logger.info(f"Building shared authors dictionary from S3: s3://{bucket}/{input_prefix}")
    
    from doc_utils import extract_doc_info
    from text_processing import normalize_text
    
    shared_authors = {}
    
    try:
        logger.debug("Listing TXT files from S3 bucket")
        txt_files = list_files(bucket, input_prefix, suffix='.txt')
        
        logger.info(f"Found {len(txt_files)} TXT files to process for shared authors")
        
        if not txt_files:
            logger.warning("No TXT files found in S3 for shared authors extraction")
            return {}
        
        processed_count = 0
        error_count = 0
        
        for idx, txt_key in enumerate(txt_files, 1):
            try:
                logger.debug(f"[{idx}/{len(txt_files)}] Processing: {txt_key}")
                
                content = read_text_from_s3(bucket, txt_key)
                lines = content.splitlines()
                
                logger.debug(f"Read {len(lines)} lines from {txt_key}")
                
                doc_id, doc_issue = extract_doc_info(lines)
                
                if doc_id == "NA" or doc_issue == "NA":
                    logger.debug(f"Skipping {txt_key}: missing doc_id or doc_issue")
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
                    
                    logger.debug(f"Extracting authors from lines {start_idx+1} to {end_idx}")
                    
                    for i in range(start_idx + 1, end_idx):
                        if i >= len(lines):
                            break
                            
                        author_name = lines[i].strip()
                        if author_name and len(author_name) > 2:
                            import re
                            if not re.match(r'^[\d.\s…]+$', author_name):
                                authors_original.append(author_name)
                                authors_normalized.append(normalize_text(author_name))
                    
                    key = (doc_id, doc_issue)
                    if key not in shared_authors and authors_original:
                        shared_authors[key] = (authors_original, authors_normalized)
                        logger.debug(f"Added {len(authors_original)} authors for {doc_id}/{doc_issue}")
                        processed_count += 1
                else:
                    logger.debug(f"No author section found in {txt_key}")
                    
            except UnicodeDecodeError as e:
                logger.warning(f"Unicode decode error in {txt_key}: {e}")
                error_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {txt_key}: {e}", exc_info=True)
                error_count += 1
        
        logger.info(f"Shared authors dictionary built: {len(shared_authors)} documents processed")
        logger.info(f"   Success: {processed_count}, Errors: {error_count}")
        
        if shared_authors:
            sample_keys = list(shared_authors.keys())[:3]
            for key in sample_keys:
                authors = shared_authors[key][0]
                logger.debug(f"   Sample: {key} -> {len(authors)} authors")
        
        return shared_authors
        
    except Exception as e:
        logger.error(f"Unexpected error building shared authors dictionary from S3: {e}", exc_info=True)
        return {}


def build_shared_authors_dict_local(input_dir):
    """
    Build a dictionary mapping document IDs to their authors across all local files.
    
    Note:
        This function is deprecated. Use build_shared_authors_dict_s3 for S3-based processing.
    
    Args:
        input_dir (Path): Input directory containing TXT files
        
    Returns:
        dict: Dictionary mapping (doc_id, doc_issue) to (authors_original, authors_normalized)
    """
    logger.warning("build_shared_authors_dict_local is deprecated. Use build_shared_authors_dict_s3")
    
    from doc_utils import extract_doc_info
    from text_processing import normalize_text
    
    shared_authors = {}
    
    try:
        logger.debug("Listing TXT files from local directory")
        txt_files = list(input_dir.rglob("*.txt"))
        
        logger.info(f"Found {len(txt_files)} TXT files to process for shared authors")
        
        processed_count = 0
        error_count = 0
        
        for txt_file in txt_files:
            try:
                logger.debug(f"Reading file: {txt_file}")
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                lines = content.splitlines()
                
                doc_id, doc_issue = extract_doc_info(lines)
                
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
                        author_name = lines[i].strip()
                        if author_name:
                            authors_original.append(author_name)
                            authors_normalized.append(normalize_text(author_name))
                    
                    key = (doc_id, doc_issue)
                    if key not in shared_authors and authors_original:
                        shared_authors[key] = (authors_original, authors_normalized)
                        logger.debug(f"Added {len(authors_original)} authors for doc {doc_id}/{doc_issue}")
                        processed_count += 1
                else:
                    logger.debug(f"No author section found in {txt_file.name}")
                    
            except UnicodeDecodeError as e:
                logger.warning(f"Unicode decode error in {txt_file.name}: {e}")
                error_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {txt_file.name}: {e}")
                error_count += 1
        
        logger.info(f"Shared authors dictionary built: {len(shared_authors)} documents processed")
        logger.info(f"Success: {processed_count}, Errors: {error_count}")
        
        return shared_authors
        
    except Exception as e:
        logger.error(f"Unexpected error building shared authors dictionary: {e}", exc_info=True)
        return {}