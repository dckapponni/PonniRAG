import time
import docx2txt
import boto3
from io import BytesIO
from config.config import BUCKET_NAME, REGION_NAME, INPUT_PREFIX, OUTPUT_PREFIX

# Initialize S3 client
s3 = boto3.client('s3', region_name=REGION_NAME)

def list_docx_files(bucket: str, prefix: str):
    """List all .docx files under a given S3 prefix"""
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.docx'):
                files.append(obj['Key'])
    return files

def extract_text_from_s3_docx(bucket: str, key: str):
    """Download a .docx file from S3 and extract its text"""
    s3_object = s3.get_object(Bucket=bucket, Key=key)
    docx_content = BytesIO(s3_object['Body'].read())
    text = docx2txt.process(docx_content)
    return text

def upload_text_to_s3(bucket: str, key: str, text: str):
    """Upload extracted text to S3"""
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode('utf-8'))

def process_all_docx_files():
    """Process all docx files from input prefix and upload extracted text to S3"""
    docx_files = list_docx_files(BUCKET_NAME, INPUT_PREFIX)
    print(f"Found {len(docx_files)} .docx files in S3")
    print("=" * 80)

    total_start_time = time.time()

    for idx, key in enumerate(docx_files, 1):
        docx_file = key.split('/')[-1]
        txt_filename = docx_file.replace('.docx', '.txt')
        output_key = f"{OUTPUT_PREFIX}{txt_filename}"

        try:
            start_time = time.time()
            text = extract_text_from_s3_docx(BUCKET_NAME, key)
            upload_text_to_s3(BUCKET_NAME, output_key, text)

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
    print(f"All {len(docx_files)} files processed!")
    print(f"Total time: {total_time:.3f} sec")
    print(f"Average per file: {total_time/len(docx_files):.3f} sec")


if __name__ == "__main__":
    process_all_docx_files()
