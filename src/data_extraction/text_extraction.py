import time
import re
import docx2txt

from src.config.config import BUCKET_NAME, INPUT_PREFIX, EXTRACTED_OUTPUT
from src.data_extraction.s3_utils import list_files, read_bytes, upload_text


def normalize_key(key: str) -> str:
    key = key.lower().strip()
    key = re.sub(r"\s+", "_", key)
    key = key.replace("_.", ".")
    return key


def process_all_docx_files():
    docx_files = list_files(BUCKET_NAME, INPUT_PREFIX, suffix=".docx")
    existing_txt_files = set(
        normalize_key(k)
        for k in list_files(BUCKET_NAME, EXTRACTED_OUTPUT, suffix=".txt")
    )

    print(f"Found {len(docx_files)} .docx files")
    print(f"Found {len(existing_txt_files)} extracted .txt files")
    print("=" * 80)

    processed = 0
    skipped = 0
    start_all = time.time()

    for idx, key in enumerate(docx_files, 1):
        docx_file = key.split("/")[-1]
        txt_filename = docx_file.replace(".docx", ".txt")
        output_key = f"{EXTRACTED_OUTPUT}{txt_filename}"

        normalized_output_key = normalize_key(output_key)
        if normalized_output_key in existing_txt_files:
            print(f"[{idx}] Skipped (already processed): {docx_file}")
            skipped += 1
            continue

        try:
            start = time.time()
            text = docx2txt.process(read_bytes(BUCKET_NAME, key))
            upload_text(BUCKET_NAME, output_key, text)

            print(f"[{idx}] Extracted: {docx_file} ({time.time() - start:.2f}s)")
            processed += 1

        except Exception as e:
            print(f"[{idx}] Error: {e}")

    print("=" * 80)
    print(f"Processed (new) : {processed}")
    print(f"Skipped        : {skipped}")
    print(f"Total time     : {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    process_all_docx_files()
