import sys
import json
import logging
import pandas as pd
from io import StringIO
from datetime import datetime
from pathlib import Path
import re


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent  
sys.path.insert(0, str(project_root))

from config.config import (
    BUCKET_NAME,
    INPUT_PREFIX,
    EXTRACTED_OUTPUT,
    OUTPUT_PREFIX,
    CSV_PATH
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
from csv_fuzzy_matcher import extract_articles_from_csv
from s3_utils import (
    list_files,
    read_text_from_s3,
    upload_json,
    file_exists
)
INPUT_PREFIX = EXTRACTED_OUTPUT

def setup_logging(log_file='tamil_doc_processing.log'):
    """
    Setup comprehensive logging for Tamil document processing.

    Creates a logging configuration with both file and console handlers. File logs include
    DEBUG level details with timestamps and function names, while console logs show INFO
    level messages with simplified formatting.

    Args:
        log_file (str, optional): Base name for the log file. Defaults to 'tamil_doc_processing.log'.
                                Actual filename will be timestamped.

    Returns:
        logging.Logger: Configured logger instance for the TamilDocProcessor.
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
    logger.info(f"Input Prefix: {INPUT_PREFIX}")
    logger.info(f"Output Prefix: {OUTPUT_PREFIX}")
    return logger

logger = setup_logging()


def extract_year_from_s3_key(s3_key):
    """
    Extract the year from an S3 key path.

    Searches for a 4-digit year pattern (19xx or 20xx) within the S3 key string.
    Typically used to identify the publication year from file paths.

    Args:
        s3_key (str): The S3 object key/path to parse.

    Returns:
        str: The extracted 4-digit year if found, otherwise "Unknown".
    """
    try:
        year_match = re.search(r'(19|20)\d{2}', s3_key)
        if year_match:
            return year_match.group(0)
    except:
        pass
    return "Unknown"


def is_file_already_processed(bucket, output_key):
    """
    Check if a file has already been processed by checking if output exists in S3.
    
    Args:
        bucket (str): S3 bucket name
        output_key (str): Expected output JSON key
        
    Returns:
        bool: True if file already processed, False otherwise
    """
    try:
        if file_exists(bucket, output_key):
            logger.debug(f"Output file already exists: s3://{bucket}/{output_key}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking if file exists: {e}")
        return False


def validate_processed_file(bucket, output_key):
    """
    Validate that the processed file contains valid data.
    
    Args:
        bucket (str): S3 bucket name
        output_key (str): Output JSON key to validate
        
    Returns:
        bool: True if file is valid, False if corrupted/incomplete
    """
    try:
        from s3_utils import read_json_from_s3
        
        data = read_json_from_s3(bucket, output_key)
        
        # Check if it has the expected structure
        if not isinstance(data, dict):
            logger.warning(f"Invalid structure in {output_key}: not a dictionary")
            return False
        
        # Check if it has articles
        if "articles" not in data:
            logger.warning(f"Invalid structure in {output_key}: missing 'articles' key")
            return False
        
        # Check if articles is a list
        if not isinstance(data["articles"], list):
            logger.warning(f"Invalid structure in {output_key}: 'articles' is not a list")
            return False
        
        # Check if there's at least some content (allow empty for genuinely empty files)
        article_count = len(data["articles"])
        logger.debug(f"Validated {output_key}: {article_count} articles")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error in {output_key}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error validating {output_key}: {e}")
        return False


def load_csv_from_local(csv_path):
    """
    Load CSV file from local file system.
    
    Args:
        csv_path (str or Path): Path to local CSV file
        
    Returns:
        pd.DataFrame: Loaded CSV data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        Exception: For other errors
    """
    try:
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading CSV from local file: {csv_path}")
        
        csv_df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        
        logger.info(f"✓ Loaded CSV successfully: {len(csv_df)} rows")
        logger.debug(f"CSV columns: {list(csv_df.columns)}")
        
        return csv_df
        
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV file is empty: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}", exc_info=True)
        raise


def save_authors_to_s3(bucket, output_key_prefix, doc_id, doc_issue, authors_list):
    """
    Save or update the authors.json file in S3.

    Maintains a consolidated JSON file containing author information for all processed
    documents. Updates existing entries or appends new ones based on document ID and issue.

    Args:
        bucket (str): S3 bucket name where the authors file is stored.
        output_key_prefix (str): S3 key prefix (directory path) for the authors.json file.
        doc_id (str): Document identifier (மலர் ID).
        doc_issue (str): Document issue number (இதழ் number).
        authors_list (list): List of authors, either as dictionaries with 'author_name' key,
                            strings, or other types that can be converted to strings.

    Returns:
        None

    Side Effects:
        - Creates or updates authors.json in S3
        - Logs information about added or updated author entries

    Raises:
        Exception: Re-raises any errors encountered during S3 operations after logging.
    """
    try:
        authors_key = f"{output_key_prefix}authors.json"
        
        logger.debug(f"Preparing to save authors to: s3://{bucket}/{authors_key}")
        
        author_names = []
        if authors_list:
            if isinstance(authors_list[0], dict) and "author_name" in authors_list[0]:
                author_names = [a["author_name"] for a in authors_list]
            elif isinstance(authors_list[0], str):
                author_names = authors_list
            else:
                author_names = [str(a) for a in authors_list]
        
        existing_data = []
        if file_exists(bucket, authors_key):
            try:
                from s3_utils import read_json_from_s3
                existing_data = read_json_from_s3(bucket, authors_key)
                logger.debug(f"Loaded {len(existing_data)} existing entries")
            except Exception as e:
                logger.debug(f"No existing authors.json or error reading: {e}")
        
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
        
        upload_json(bucket, authors_key, existing_data)
        logger.info(f"Successfully saved authors.json to S3")
        
    except Exception as e:
        logger.error(f"Error saving authors to S3: {e}", exc_info=True)
        raise


def parse_tamil_document_csv_first(lines, shared_authors_dict, csv_df, s3_key):
    """
    Parse Tamil document using CSV-first approach with pattern extraction fallback.

    Primary parsing function that attempts to extract articles from Tamil documents
    using CSV metadata first. If CSV extraction fails, falls back to pattern-based
    extraction using multiple pattern recognition algorithms.

    Args:
        lines (list): List of text lines from the document.
        shared_authors_dict (dict): Dictionary mapping document IDs to shared author lists.
        csv_df (pd.DataFrame): DataFrame containing CSV metadata for article matching.
        s3_key (str): S3 object key for the source document (used for year extraction).

    Returns:
        dict: Parsed document data containing:
            - articles (list): List of article dictionaries with fields:
                - doc_id, doc_issue, article_no, author_name, title, content,
                year, source_document
            - authors_list (list): List of author dictionaries
            - doc_id (str): Document மலர் identifier
            - doc_issue (str): Document இதழ் number
    Processing Flow:
        1. Attempts CSV-based extraction
        2. On CSV failure, extracts document info (மலர்/இதழ்)
        3. Extracts authors using multiple methods (markers, TOC, shared authors)
        4. Applies pattern extraction (Pattern A, B, C)
        5. Extracts intro sections as articles
        6. Collects remaining unprocessed content
    """
    try:
        logger.info("=" * 80)
        logger.info("STEP 1: ATTEMPTING CSV-BASED EXTRACTION")
        logger.info("=" * 80)
        
        # Extract year and source document from S3 key
        year = extract_year_from_s3_key(s3_key)
        source_document = s3_key.split('/')[-1]  # Get filename from S3 key
        
        # Create a pseudo Path object for CSV extraction
        class S3Path:
            def __init__(self, key):
                self.name = key.split('/')[-1]
        
        file_path = S3Path(s3_key)
        
        # TRY CSV EXTRACTION FIRST
        csv_result = extract_articles_from_csv(lines, csv_df, file_path)
        
        if csv_result and len(csv_result['articles']) > 0:
            logger.info("=" * 80)
            logger.info(f" CSV EXTRACTION SUCCESS: {len(csv_result['articles'])} articles")
            logger.info("=" * 80)
            
            # Add year and source_document to each article
            for article in csv_result['articles']:
                article['source_document'] = source_document
                if 'year' not in article or article['year'] == "Unknown":
                    article['year'] = year
            
            return csv_result
        
        # CSV FAILED - FALLBACK TO PATTERN EXTRACTION
        logger.warning("=" * 80)
        logger.warning(" CSV EXTRACTION FAILED - FALLING BACK TO PATTERN EXTRACTION")
        logger.warning("=" * 80)
        
        # Extract doc info for pattern extraction
        doc_id, doc_issue = extract_doc_info(lines)
        logger.info(f"Processing document: {doc_id}, Issue: {doc_issue}")

        if doc_id == "NA" and doc_issue == "NA":
            logger.error("Cannot extract மலர்/இதழ் - Both methods failed")
            return {
                "articles": [],
                "authors_list": [],
                "doc_id": "NA",
                "doc_issue": "NA"
            }

        # NORMAL PATTERN-BASED PROCESSING
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

        parse_start_idx = end_idx + 1 if end_idx != -1 else (start_idx + 26 if start_idx != -1 else 0)
        processed_lines = [False] * len(lines)

        if start_idx != -1 and end_idx != -1:
            for i in range(start_idx, end_idx + 1):
                processed_lines[i] = True
        elif start_idx != -1:
            for i in range(start_idx, min(start_idx + 26, len(lines))):
                processed_lines[i] = True

        intro_keywords = get_intro_keywords()
        articles = []
        article_no = 1

        # PATTERN A
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
                "author_name": article["author"],
                "title": article["heading"],
                "content": article["content"],
                "year": year,
                "source_document": source_document
            })
            article_no += 1

        # PATTERN B
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
                "author_name": article["author"],
                "title": article["heading"],
                "content": article["content"],
                "year": year,
                "source_document": source_document
            })
            article_no += 1

        # PATTERN C
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
                "author_name": article["author"],
                "title": article["heading"],
                "content": article["content"],
                "year": year,
                "source_document": source_document
            })
            article_no += 1

        # INTRO EXTRACTION
        logger.info("=" * 80)
        logger.info("PHASE 4: Extracting intro sections (as articles)")
        logger.info("=" * 80)
        
        i = parse_start_idx
        
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
                    content, end_idx_content, has_author = extract_intro_content_phase1(
                        lines, i, processed_lines,
                        authors_normalized, authors_original, intro_keywords
                    )

                    if content.strip() and count_content_lines(content) >= 4:
                        articles.append({
                            "doc_id": doc_id,
                            "doc_issue": doc_issue,
                            "article_no": article_no,
                            "author_name": has_author if has_author else "NA",
                            "title": matched_keyword,
                            "content": content,
                            "year": year,
                            "source_document": source_document
                        })
                        article_no += 1

                        for j in range(i, end_idx_content):
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

        # REMAINING CONTENT
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
                    "author_name": article["author"],
                    "title": article["heading"],
                    "content": article["content"],
                    "year": year,
                    "source_document": source_document
                })
                article_no += 1
                
        except Exception as e:
            logger.error(f"Error extracting remaining content: {e}")

        authors_list = [
            {"doc_id": doc_id, "doc_issue": doc_issue, "author_name": a}
            for a in authors_original
        ]

        logger.info(f"Pattern extraction complete: {len(articles)} total articles")
        
        return {
            "articles": articles,
            "authors_list": authors_list,
            "doc_id": doc_id,
            "doc_issue": doc_issue
        }
        
    except Exception as e:
        logger.error(f"Critical error in parse_tamil_document_csv_first: {e}", exc_info=True)
        raise


def process_s3_files(force_reprocess=False):
    """
    Process all TXT files from S3 using CSV-first extraction approach.

    Main orchestration function that coordinates the entire document processing pipeline.
    Reads TXT files from S3, processes them using CSV-first extraction, and saves
    results back to S3 as JSON files.

    Args:
        force_reprocess (bool, optional): If True, reprocess all files even if already processed.
                                        If False, skip files with existing valid outputs.
                                        Defaults to False.

    Returns:
        None

    Processing Steps:
        1. Loads CSV metadata from local file system
        2. Lists all TXT files in S3 input location
        3. Builds shared authors dictionary for fallback
        4. For each file:
            - Checks if already processed (unless force_reprocess=True)
            - Validates existing outputs
            - Reads and parses document
            - Saves articles JSON and authors JSON to S3
        5. Logs comprehensive processing summary

    """
    try:
        logger.info("=" * 80)
        logger.info("Tamil Document Processing - S3 CSV FIRST MODE")
        logger.info(f"S3 Bucket: {BUCKET_NAME}")
        logger.info(f"Input Prefix: {INPUT_PREFIX}")
        logger.info(f"Output Prefix: {OUTPUT_PREFIX}")
        logger.info(f"CSV Path (Local): {CSV_PATH}")
        logger.info(f"Force Reprocess: {force_reprocess}")
        logger.info("=" * 80)
        
        # Load CSV from LOCAL file system
        logger.info("Loading CSV file from local file system...")
        try:
            csv_df = load_csv_from_local(CSV_PATH)
        except FileNotFoundError:
            logger.error(f"✗ CSV file not found at: {CSV_PATH}")
            logger.error("Please check the CSV_PATH in config.py")
            return
        except Exception as e:
            logger.error(f"✗ Error loading CSV from local file: {e}")
            logger.error("CSV is required for this mode. Exiting.")
            return
        
        # List TXT files from S3
        logger.info("Listing TXT files from S3...")
        txt_files = list_files(BUCKET_NAME, INPUT_PREFIX, suffix='.txt')
        
        if not txt_files:
            logger.warning("No TXT files found in S3")
            return

        logger.info(f"Found {len(txt_files)} TXT files")

        # Build shared authors (for fallback)
        logger.info("Building shared authors dictionary (for fallback)...")
        shared_authors_dict = build_shared_authors_dict_s3(BUCKET_NAME, INPUT_PREFIX)
        logger.info(f"Shared authors dictionary built with {len(shared_authors_dict)} document groups")
        
        processed = failed = csv_success = pattern_fallback = skipped = 0

        for idx, txt_key in enumerate(txt_files, 1):
            # Determine output key
            relative_key = txt_key[len(INPUT_PREFIX):]  # Remove input prefix
            output_key = f"{OUTPUT_PREFIX}{relative_key.rsplit('.', 1)[0]}.json"

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[{idx}/{len(txt_files)}] Processing: {txt_key}")
            logger.info(f"Output key: {output_key}")
            logger.info("=" * 80)

            try:
                # CHECK IF ALREADY PROCESSED
                if not force_reprocess:
                    if is_file_already_processed(BUCKET_NAME, output_key):
                        # Validate the existing file
                        if validate_processed_file(BUCKET_NAME, output_key):
                            logger.info(f"⏭️  SKIPPED: File already processed and validated")
                            logger.info(f"   Existing output: s3://{BUCKET_NAME}/{output_key}")
                            logger.info(f"   Use --force flag to reprocess")
                            skipped += 1
                            continue
                        else:
                            logger.warning(f"⚠️  Existing file is invalid/corrupted, reprocessing...")
                
                # Read TXT file from S3
                text_content = read_text_from_s3(BUCKET_NAME, txt_key)
                lines = text_content.splitlines()
                logger.debug(f"Read {len(lines)} lines from S3")

                logger.info("Starting document parsing (CSV FIRST)...")
                result = parse_tamil_document_csv_first(lines, shared_authors_dict, csv_df, txt_key)

                # Track which method was used
                if result and len(result['articles']) > 0:
                    first_article = result['articles'][0]
                    if 'year' in first_article and first_article.get('year') != "Unknown":
                        csv_success += 1
                        logger.info("✓ METHOD: CSV EXTRACTION")
                    else:
                        pattern_fallback += 1
                        logger.info("✓ METHOD: PATTERN EXTRACTION (CSV fallback)")

                # Save main JSON to S3
                output_data = {
                    "articles": result["articles"]
                }
                
                upload_json(BUCKET_NAME, output_key, output_data)
                logger.info(f"Main JSON saved to S3 with {len(result['articles'])} articles")

                # Save authors JSON to S3
                output_key_prefix = output_key.rsplit('/', 1)[0] + '/' if '/' in output_key else ''
                save_authors_to_s3(
                    BUCKET_NAME,
                    output_key_prefix,
                    result["doc_id"],
                    result["doc_issue"],
                    result["authors_list"]
                )

                logger.info(f"✅ SUCCESS: {len(result['articles'])} total articles")
                processed += 1

            except Exception as e:
                logger.error(f"❌ FAILED processing {txt_key}: {e}", exc_info=True)
                failed += 1

        logger.info("")
        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY:")
        logger.info(f"   Total files: {len(txt_files)}")
        logger.info(f"   Successfully processed: {processed}")
        logger.info(f"   Skipped (already processed): {skipped}")
        logger.info(f"   CSV Extraction used: {csv_success}")
        logger.info(f"   Pattern Extraction used: {pattern_fallback}")
        logger.info(f"   Failed: {failed}")
        if len(txt_files) > 0:
            logger.info(f"   Success rate: {(processed/len(txt_files)*100):.1f}%")
        if processed > 0:
            logger.info(f"   CSV success rate: {(csv_success/processed*100):.1f}%")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical(f"Fatal error in process_s3_files: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        # Check for command line argument to force reprocessing
        force_reprocess = "--force" in sys.argv or "-f" in sys.argv
        
        if force_reprocess:
            logger.info("⚠️  FORCE REPROCESS MODE ENABLED - Will reprocess all files")
        else:
            logger.info("📋 INCREMENTAL MODE - Will skip already processed files")
            logger.info("   Use --force or -f flag to reprocess all files")
        
        logger.info("Starting Tamil Document Processing (S3 CSV FIRST MODE)...")
        process_s3_files(force_reprocess=force_reprocess)
        logger.info("Processing completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Process failed: {e}")
        sys.exit(1)