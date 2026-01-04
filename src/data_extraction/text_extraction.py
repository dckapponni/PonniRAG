import time
import re
import logging
import sys
from pathlib import Path
import docx2txt

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import BUCKET_NAME, INPUT_PREFIX, EXTRACTED_OUTPUT
from src.data_extraction.s3_utils import list_files, read_bytes, upload_text


# Configure logging - Console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def normalize_key(key: str) -> str:
    """Normalize S3 key by converting to lowercase and replacing spaces."""
    key = key.lower().strip()
    key = re.sub(r"\s+", "_", key)
    key = key.replace("_.", ".")
    return key


def process_all_docx_files():
    """Process all DOCX files from S3 and extract text content."""
    try:
        # List all files
        logger.info("Starting DOCX file processing")
        docx_files = list_files(BUCKET_NAME, INPUT_PREFIX, suffix=".docx")
        existing_txt_files = set(
            normalize_key(k)
            for k in list_files(BUCKET_NAME, EXTRACTED_OUTPUT, suffix=".txt")
        )

        logger.info(f"Found {len(docx_files)} .docx files")
        logger.info(f"Found {len(existing_txt_files)} extracted .txt files")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Failed to list files from S3: {e}", exc_info=True)
        return

    processed = 0
    skipped = 0
    failed = 0
    start_all = time.time()

    for idx, key in enumerate(docx_files, 1):
        docx_file = key.split("/")[-1]

        # Preserve folder structure safely
        relative_path = key.replace(INPUT_PREFIX, "", 1)

        txt_relative_path = re.sub(
            r"\.docx$", ".txt", relative_path, flags=re.IGNORECASE
        )

        output_key = f"{EXTRACTED_OUTPUT}{txt_relative_path}"

        normalized_output_key = normalize_key(output_key)

        if normalized_output_key in existing_txt_files:
            logger.debug(f"[{idx}/{len(docx_files)}] Skipped (already processed): {docx_file}")
            skipped += 1
            continue

        try:
            start = time.time()

            file_bytes = read_bytes(BUCKET_NAME, key)
            text = docx2txt.process(file_bytes)

            upload_text(BUCKET_NAME, output_key, text)

            elapsed = time.time() - start
            logger.info(f"[{idx}/{len(docx_files)}] Successfully extracted: {docx_file} ({elapsed:.2f}s)")
            processed += 1

        except Exception as e:
            logger.error(f"[{idx}/{len(docx_files)}] Error processing {docx_file}: {e}", exc_info=True)
            failed += 1


    total_time = time.time() - start_all
    
    # Summary
    print("=" * 80)
    logger.info(f"Processing complete - Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"Total processing time: {total_time:.2f}s")
    if docx_files:
        logger.info(f"Average time per file: {total_time/len(docx_files):.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    try:
        process_all_docx_files()
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        sys.exit(1)
