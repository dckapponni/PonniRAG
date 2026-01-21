"""
Tamil Document Processing - Main Module
Processes Tamil documents from S3 and extracts structured content including articles, intros, and authors.
"""
import sys
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    BUCKET_NAME, 
    REGION_NAME, 
    EXTRACTED_OUTPUT, 
    OUTPUT_PREFIX,
)
from s3_utils import (
    list_files,
    read_text_from_s3,
    upload_json,
    upload_text,
    read_json_from_s3,
    file_exists
)
from content_extraction import (
    count_consecutive_blanks,
    extract_remaining_content,
    extract_intro_content_phase1
)
from doc_utils import (
    extract_doc_info,
    get_shared_authors,
    extract_authors_alternative,
    count_content_lines
)
from article_patterns import (
    extract_pattern_a_forward,
    extract_pattern_b_forward,
    extract_pattern_c_reverse
)
from shared_author import build_shared_authors_dict_s3
from text_processing import normalize_text, get_intro_keywords


def setup_logging(log_file='tamil_doc_processing.log'):
    """
    Setup comprehensive logging for S3 operations.
    
    Args:
        log_file (str): Name of the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('TamilDocProcessor')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_file}"
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_path}")
    logger.info(f"S3 Bucket: {BUCKET_NAME}")
    logger.info(f"Input extracted txt: {EXTRACTED_OUTPUT}")
    logger.info(f"Output Prefix: {OUTPUT_PREFIX}")
    return logger

logger = setup_logging()


def get_s3_folder_path(s3_key, base_prefix):
    """
    Extract folder path from S3 key relative to base prefix.
    
    Args:
        s3_key (str): Full S3 key (e.g., "Raw_Proof_Read_Content/folder1/doc.txt")
        base_prefix (str): Base prefix to remove (e.g., "Raw_Proof_Read_Content/")
        
    Returns:
        str: Folder path (e.g., "folder1/")
    """
    try:
        if s3_key.startswith(base_prefix):
            relative_path = s3_key[len(base_prefix):]
        else:
            relative_path = s3_key
        
        if '/' in relative_path:
            folder_path = '/'.join(relative_path.split('/')[:-1]) + '/'
        else:
            folder_path = ''
        
        logger.debug(f"Extracted folder path: '{folder_path}' from key: '{s3_key}'")
        return folder_path
    except Exception as e:
        logger.error(f"Error extracting folder path from {s3_key}: {e}")
        return ''


def save_authors_to_s3_folder(bucket, output_prefix, folder_path, doc_id, doc_issue, authors_list):
    """
    Save or update authors.json in S3 folder.
    
    Args:
        bucket (str): S3 bucket name
        output_prefix (str): Base output prefix
        folder_path (str): Relative folder path within output
        doc_id (str): Document ID
        doc_issue (str): Document issue
        authors_list (list): List of author dictionaries or strings
        
    Raises:
        Exception: If saving to S3 fails
    """
    try:
        authors_key = f"{output_prefix}{folder_path}authors.json"
        
        logger.debug(f"Preparing to save authors to: s3://{bucket}/{authors_key}")
        
        author_names = []
        if authors_list:
            if isinstance(authors_list[0], dict) and "author_name" in authors_list[0]:
                author_names = [a["author_name"] for a in authors_list]
                logger.debug(f"Extracted {len(author_names)} author names from dict format")
            elif isinstance(authors_list[0], str):
                author_names = authors_list
                logger.debug(f"Using {len(author_names)} author names from string format")
            else:
                logger.warning(f"Unexpected authors_list format: {type(authors_list[0])}")
                author_names = [str(a) for a in authors_list]
        
        existing_data = []
        try:
            if file_exists(bucket, authors_key):
                logger.debug(f"Reading existing authors.json from S3")
                existing_data = read_json_from_s3(bucket, authors_key)
                logger.debug(f"Loaded {len(existing_data)} existing entries")
            else:
                logger.debug(f"No existing authors.json found at {authors_key}")
        except Exception as e:
            logger.debug(f"No existing authors.json or error reading: {e}")
            existing_data = []
        
        doc_exists = False
        for entry in existing_data:
            if entry.get("doc_id") == doc_id and entry.get("doc_issue") == doc_issue:
                entry["authors"] = author_names
                doc_exists = True
                logger.info(f"Updated authors for {doc_id} (Issue: {doc_issue}): {len(author_names)} authors")
                break
        
        if not doc_exists:
            new_entry = {
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "authors": author_names
            }
            existing_data.append(new_entry)
            logger.info(f"Added authors for {doc_id} (Issue: {doc_issue}): {len(author_names)} authors")
        
        logger.debug(f"Uploading authors.json with {len(existing_data)} total entries")
        try:
            upload_json(bucket, authors_key, existing_data)
            logger.info(f"Successfully saved authors.json to {authors_key}")
        except Exception as upload_err:
            logger.error(f"Failed to upload authors.json: {upload_err}")
           
        
    except Exception as e:
        logger.error(f"Error saving authors to S3 folder: {e}", exc_info=True)
        raise


def parse_tamil_document(lines, shared_authors_dict):
    """
    Parse Tamil document with Pattern C running before intro extraction.
    
    Args:
        lines (list): List of text lines from document
        shared_authors_dict (dict): Dictionary of shared authors across documents
        
    Returns:
        dict: Parsed content including intro, articles, authors_list, doc_id, and doc_issue
        
    Raises:
        Exception: If critical parsing error occurs
    """
    try:
        doc_id, doc_issue = extract_doc_info(lines)
        logger.info(f"Processing document: {doc_id}, Issue: {doc_issue}")

        authors_original = []
        authors_normalized = []
        start_idx = end_idx = -1

        for i, line in enumerate(lines):
            if "பொருளடக்கம்" in line:
                start_idx = i
            if "ஆகியோரின் எழுத்தோவியங்கள்" in line or "ஆகியோரின்" in line:
                end_idx = i
                break

        if start_idx != -1 and end_idx != -1:
            logger.info("Using METHOD 1: Extracting authors between markers")
            for i in range(start_idx + 1, end_idx):
                name = lines[i].strip()
                if name and not name.isdigit() and len(name) > 2:
                    import re
                    if not re.match(r'^[\d.\s…]+$', name):
                        authors_original.append(name)
                        authors_normalized.append(normalize_text(name))
            logger.info(f"METHOD 1: Extracted {len(authors_original)} authors")
        
        if not authors_original and start_idx != -1:
            logger.info("Using METHOD 2: Parsing TOC structure")
            from doc_utils import extract_authors_from_toc
            _, _, toc_authors_orig, toc_authors_norm, _ = extract_authors_from_toc(lines)
            if toc_authors_orig:
                authors_original = toc_authors_orig
                authors_normalized = toc_authors_norm
                logger.info(f"METHOD 2: Extracted {len(authors_original)} authors from TOC")
        
        if not authors_original:
            logger.info("Using METHOD 3: Retrieving shared authors")
            authors_original, authors_normalized = get_shared_authors(
                doc_id, doc_issue, shared_authors_dict
            )
            if authors_original:
                logger.info(f"METHOD 3: Retrieved {len(authors_original)} shared authors")
            else:
                authors_original, authors_normalized = extract_authors_alternative(lines)
                if authors_original:
                    logger.info(f"METHOD 4: Extracted {len(authors_original)} authors")

        logger.info(f"Final author count: {len(authors_original)} authors")
        if authors_original:
            logger.debug(f"Authors: {', '.join(authors_original[:5])}")

        parse_start_idx = end_idx + 1 if end_idx != -1 else (start_idx + 26 if start_idx != -1 else 0)
        processed_lines = [False] * len(lines)

        if start_idx != -1 and end_idx != -1:
            for i in range(start_idx, end_idx + 1):
                processed_lines[i] = True
        elif start_idx != -1:
            for i in range(start_idx, min(start_idx + 26, len(lines))):
                processed_lines[i] = True

        intro_keywords = get_intro_keywords()
        intro = []
        articles = []
        article_no = 1

        logger.info("=" * 80)
        logger.info("PHASE 1: Running Pattern A")
        logger.info("=" * 80)
        
        pattern_a_articles = extract_pattern_a_forward(
            lines, parse_start_idx, len(lines),
            authors_normalized, authors_original,
            processed_lines, intro_keywords
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
            logger.info(f"Pattern A article {article_no}: '{article['heading'][:40]}...'")
            article_no += 1

        logger.info(f"Pattern A extracted: {len(pattern_a_articles)} articles")

        logger.info("=" * 80)
        logger.info("PHASE 2: Running Pattern B")
        logger.info("=" * 80)
        
        pattern_b_articles = extract_pattern_b_forward(
            lines, parse_start_idx, len(lines),
            authors_normalized, authors_original,
            processed_lines, intro_keywords
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
            logger.info(f"Pattern B article {article_no}: '{article['heading'][:40]}...'")
            article_no += 1

        logger.info(f"Pattern B extracted: {len(pattern_b_articles)} articles")

        logger.info("=" * 80)
        logger.info("PHASE 3: Running Pattern C")
        logger.info("=" * 80)
        
        pattern_c_articles = extract_pattern_c_reverse(
            lines, parse_start_idx, len(lines),
            authors_normalized, authors_original,
            processed_lines, intro_keywords
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
            logger.info(f"Pattern C poem {article_no}: '{article['heading'][:40]}...'")
            article_no += 1

        logger.info(f"Pattern C extracted: {len(pattern_c_articles)} poems")

        logger.info("=" * 80)
        logger.info("PHASE 4: Extracting intro sections")
        logger.info("=" * 80)
        
        i = parse_start_idx
        intro_count = 0
        
        while i < len(lines):
            if processed_lines[i]:
                i += 1
                continue

            line = lines[i].strip()
            matched_keyword = None

            for keyword in intro_keywords:
                if keyword in line and (i == 0 or not lines[i - 1].strip()):
                    matched_keyword = keyword
                    break

            if matched_keyword:
                try:
                    extraction_start = i
                    
                    content, end_idx_content, has_author = extract_intro_content_phase1(
                        lines, i, processed_lines,
                        authors_normalized, authors_original, intro_keywords
                    )

                    extraction_successful = False

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
                            logger.info(f"Phase 4 article {article_no}: '{matched_keyword}'")
                            article_no += 1
                            extraction_successful = True
                        else:
                            intro.append({
                                "doc_id": doc_id,
                                "doc_issue": doc_issue,
                                "heading": matched_keyword,
                                "content": content
                            })
                            intro_count += 1
                            logger.info(f"Phase 4 intro: '{matched_keyword}'")
                            extraction_successful = True

                    if extraction_successful:
                        for j in range(extraction_start, end_idx_content):
                            if j < len(lines):
                                processed_lines[j] = True
                        i = end_idx_content
                    else:
                        i += 1
                    
                except Exception as e:
                    logger.warning(f"Error extracting intro at line {i}: {e}")
                    i += 1
            else:
                i += 1

        logger.info(f"Phase 4 complete: {intro_count} intro sections")

        logger.info("=" * 80)
        logger.info("PHASE 5: Extracting remaining content")
        logger.info("=" * 80)
        
        try:
            remaining = extract_remaining_content(lines, parse_start_idx, processed_lines)
            
            for article in remaining:
                articles.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "article_no": article_no,
                    "article_heading": article["heading"],
                    "article_author_name": article["author"],
                    "article_content": article["content"]
                })
                article_no += 1
                
            logger.info(f"Phase 5 complete: {len(remaining)} remaining articles")
            
        except Exception as e:
            logger.error(f"Error extracting remaining content: {e}")

        authors_list = [
            {"doc_id": doc_id, "doc_issue": doc_issue, "author_name": a}
            for a in authors_original
        ]

        logger.info(f"Document parsing complete: {len(intro)} intro, {len(articles)} articles")
        
        return {
            "intro": intro,
            "articles": articles,
            "authors_list": authors_list,
            "doc_id": doc_id,
            "doc_issue": doc_issue
        }
        
    except Exception as e:
        logger.error(f"Critical error in parse_tamil_document: {e}", exc_info=True)
        raise


def process_s3_files():
    """
    Process all TXT files from S3 and save results back to S3.
    Configuration is read from config.py.
    
    Raises:
        Exception: If critical processing error occurs
    """
    try:
        logger.info("=" * 80)
        logger.info("Tamil Document Processing - S3 MODE WITH AUTHORS STORAGE")
        logger.info(f"S3 Bucket: {BUCKET_NAME}")
        logger.info(f"EXTRACTED OUTPUT: {EXTRACTED_OUTPUT}")
        logger.info(f"Output Prefix: {OUTPUT_PREFIX}")
        logger.info("=" * 80)
        
        logger.info("Listing TXT files from S3...")
        txt_files = list_files(BUCKET_NAME, EXTRACTED_OUTPUT, suffix='.txt')
        
        if not txt_files:
            logger.warning("No TXT files found in S3")
            return

        logger.info(f"Found {len(txt_files)} TXT files in S3")

        logger.info("Building shared authors dictionary from S3...")
        shared_authors_dict = build_shared_authors_dict_s3(BUCKET_NAME, EXTRACTED_OUTPUT)
        logger.info(f"Shared authors dictionary built with {len(shared_authors_dict)} document groups")
        
        processed = failed = 0

        for idx, txt_key in enumerate(txt_files, 1):
            folder_path = get_s3_folder_path(txt_key, EXTRACTED_OUTPUT)
            file_name = txt_key.split('/')[-1]
            file_stem = file_name.rsplit('.', 1)[0]
            
            output_json_key = f"{OUTPUT_PREFIX}{folder_path}{file_stem}.json"

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[{idx}/{len(txt_files)}] Processing: {txt_key}")
            logger.info(f"Output path: {output_json_key}")
            logger.info("=" * 80)

            try:
                logger.debug(f"Reading file from S3: s3://{BUCKET_NAME}/{txt_key}")
                text_content = read_text_from_s3(BUCKET_NAME, txt_key)
                lines = text_content.splitlines()
                logger.debug(f"Read {len(lines)} lines from file")

                logger.info("Starting document parsing...")
                result = parse_tamil_document(lines, shared_authors_dict)

                output_data = {
                    "intro": result["intro"],
                    "articles": result["articles"]
                }
                
                logger.info(f"Uploading main JSON to S3: {output_json_key}")
                upload_json(BUCKET_NAME, output_json_key, output_data)
                logger.info(f"Main JSON uploaded successfully")

                logger.info(f"Saving authors to subfolder: {folder_path}")
                save_authors_to_s3_folder(
                    BUCKET_NAME,
                    OUTPUT_PREFIX,
                    folder_path,
                    result["doc_id"],
                    result["doc_issue"],
                    result["authors_list"]
                )

                logger.info(f"SUCCESS: {len(result['articles'])} articles, {len(result['intro'])} intros, {len(result['authors_list'])} authors")
                processed += 1

            except Exception as e:
                logger.error(f"FAILED processing {txt_key}: {e}", exc_info=True)
                failed += 1

        logger.info("")
        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY:")
        logger.info(f"   Total files: {len(txt_files)}")
        logger.info(f"   Successfully processed: {processed}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Success rate: {(processed/len(txt_files)*100):.1f}%")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical(f"Fatal error in process_s3_files: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting Tamil Document Processing from S3...")
        process_s3_files()
        logger.info("Processing completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Process failed: {e}")
        sys.exit(1)