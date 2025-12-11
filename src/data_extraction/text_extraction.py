import time
import docx2txt
from config.config import BUCKET_NAME, INPUT_PREFIX, EXTRACTED_OUTPUT
from PonniRAG.src.data_extraction.s3_utils import list_files, read_bytes, upload_text
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
def process_all_docx_files():
    """Process all .docx files from S3 and upload extracted text back to S3"""
    docx_files = list_files(BUCKET_NAME, INPUT_PREFIX, suffix=".docx")
    print(f"Found {len(docx_files)} .docx files in S3")
    print("=" * 80)

    total_start_time = time.time()

    for idx, key in enumerate(docx_files, 1):
        docx_file = key.split('/')[-1]
        txt_filename = docx_file.replace('.docx', '.txt')
        output_key = f"{EXTRACTED_OUTPUT}{txt_filename}"

        try:
            start_time = time.time()
            text = docx2txt.process(read_bytes(BUCKET_NAME, key))
            upload_text(BUCKET_NAME, output_key, text)

            time_taken = time.time() - start_time
            print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
            print("   Extracted successfully")
            print(f"   Time taken: {time_taken:.3f} sec")
            print(f"   Saved to S3 key: {output_key}")

        except Exception as e:
            print(f"\n[File {idx}/{len(docx_files)}] {docx_file}")
            print(f"   Error: {str(e)}")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"All {len(docx_files)} files processed! Total time: {total_time:.3f} sec")
    print(f"Average per file: {total_time/len(docx_files):.3f} sec")


if __name__ == "__main__":
    process_all_docx_files()
