"""Text extraction from DOCX files stored in S3."""

import logging
import re
import sys
import time
from pathlib import Path

import docx2txt

from src.config.config import BUCKET_NAME, EXTRACTED_OUTPUT, INPUT_PREFIX  # noqa: E402
from src.data_extraction.s3_utils import read_bytes  # noqa: E402
from src.data_extraction.s3_utils import list_files, upload_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def normalize_key(key: str) -> str:
    """
    Normalize S3 key by converting to lowercase and replacing spaces with underscores.

    Args:
        key (str): S3 key to normalize

    Returns:
        str: Normalized S3 key
    """
    key = key.lower().strip()
    key = re.sub(r"\s+", "_", key)
    key = key.replace("_.", ".")
    return key


def extract_text_from_doc(file_bytes):
    """
    Try to extract text from .doc file using antiword or textract.

    Args:
        file_bytes: BytesIO object containing the file

    Returns:
        str: Extracted text or error message
    """
    try:
        # Try using textract (install: pip install textract)
        import textract

        file_bytes.seek(0)
        text = textract.process(file_bytes, extension="doc").decode("utf-8")
        return text
    except ImportError:
        logger.warning("textract not installed. Cannot extract .doc files.")
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract .doc file: {e}")
        return ""


def extract_text_from_file(file_bytes, filename):
    """
    Extract text from file, handling both .doc and .docx formats.

    Args:
        file_bytes: BytesIO object containing the file
        filename: Original filename

    Returns:
        str: Extracted text
    """
    try:
        file_bytes.seek(0)
        text = docx2txt.process(file_bytes)
        return text
    except Exception as docx_error:
        logger.warning(f"Failed to extract as DOCX: {docx_error}")

        try:
            file_bytes.seek(0)
            text = extract_text_from_doc(file_bytes)
            if text:
                logger.info("Successfully extracted as .doc format")
                return text
        except Exception as doc_error:
            logger.error(f"Failed to extract as DOC: {doc_error}")

        # Both failed
        raise Exception(
            f"Could not extract text from {filename}. File may be corrupted."
        )


def process_all_docx_files():
    """Process all DOCX files from S3 and extract text content.

    Skips files that have already been processed and uploads extracted text to S3.

    Raises:
        Exception: If critical S3 operations fail
    """
    try:
        logger.info("Starting DOCX file processing")
        docx_files = list_files(BUCKET_NAME, INPUT_PREFIX, suffix=".docx")
        existing_txt_files = set(
            normalize_key(k)
            for k in list_files(BUCKET_NAME, EXTRACTED_OUTPUT, suffix=".txt")
        )

        logger.info(f"Found {len(docx_files)} .docx files")
        logger.info(f"Found {len(existing_txt_files)} extracted .txt files")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to list files from S3: {e}", exc_info=True)
        return

    processed = 0
    skipped = 0
    failed = 0
    corrupted_files = []
    start_all = time.time()

    for idx, key in enumerate(docx_files, 1):
        docx_file = key.split("/")[-1]

        relative_path = key.replace(INPUT_PREFIX, "", 1)

        txt_relative_path = re.sub(
            r"\.docx$", ".txt", relative_path, flags=re.IGNORECASE
        )

        output_key = f"{EXTRACTED_OUTPUT}{txt_relative_path}"

        normalized_output_key = normalize_key(output_key)

        if normalized_output_key in existing_txt_files:
            logger.debug(
                f"[{idx}/{len(docx_files)}] Skipped (already processed): {docx_file}"
            )
            skipped += 1
            continue

        try:
            start = time.time()

            file_bytes = read_bytes(BUCKET_NAME, key)

            # Use new extraction function that handles both formats
            text = extract_text_from_file(file_bytes, docx_file)

            upload_text(BUCKET_NAME, output_key, text)

            elapsed = time.time() - start
            logger.info(
                f"[{idx}/{len(docx_files)}] "
                "Successfully extracted: "
                f"{docx_file} ({elapsed:.2f}s)"
            )
            processed += 1

        except Exception as e:
            logger.error(
                f"[{idx}/{len(docx_files)}] Error processing {docx_file}: {e}",
                exc_info=True,
            )
            failed += 1
            corrupted_files.append(docx_file)

    total_time = time.time() - start_all

    logger.info("=" * 80)
    logger.info(
        f"Processing complete - Processed: {processed}, "
        f"Skipped: {skipped}, Failed: {failed}"
    )
    if corrupted_files:
        logger.warning("Corrupted/failed files:")
        for cf in corrupted_files:
            logger.warning(f"  - {cf}")
    logger.info(f"Total processing time: {total_time:.2f}s")
    if docx_files:
        logger.info(f"Average time per file: {total_time/len(docx_files):.2f}s")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        process_all_docx_files()
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        sys.exit(1)
