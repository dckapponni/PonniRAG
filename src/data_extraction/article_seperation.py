import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import boto3
from collections import defaultdict

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
from shared_author import build_shared_authors_dict
from text_processing import normalize_text, get_intro_keywords

# S3 Config
from config.config import BUCKET_NAME, REGION_NAME, EXTRACTED_OUTPUT, OUTPUT_PREFIX

s3 = boto3.client("s3", region_name=REGION_NAME)


# ---------------- S3 Utility Functions ---------------- #
def list_files(bucket: str, prefix: str, suffix: str = None):
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if suffix is None or key.endswith(suffix):
                files.append(key)
    return files


def read_txt_from_s3(bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8").splitlines()


def upload_json_to_s3(bucket: str, key: str, data: dict):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    )


# ---------------- Parsing Logic ---------------- #
def parse_tamil_document(lines, shared_authors_dict):
    doc_id, doc_issue = extract_doc_info(lines)

    authors_original = []
    authors_normalized = []
    start_idx = end_idx = -1

    for i, line in enumerate(lines):
        if "பொருளடக்கம்" in line:
            start_idx = i
        if "ஆகியோரின் எழுத்தோவியங்கள்" in line:
            end_idx = i
            break

    if start_idx == -1 or end_idx == -1:
        authors_original, authors_normalized = get_shared_authors(
            doc_id, doc_issue, shared_authors_dict
        )
        if not authors_original:
            authors_original, authors_normalized = extract_authors_alternative(lines)
    else:
        for i in range(start_idx + 1, end_idx):
            name = lines[i].strip()
            if name:
                authors_original.append(name)
                authors_normalized.append(normalize_text(name))

    parse_start_idx = end_idx + 1 if end_idx != -1 else 0
    processed_lines = [False] * len(lines)

    if start_idx != -1 and end_idx != -1:
        for i in range(start_idx, end_idx + 1):
            processed_lines[i] = True

    intro_keywords = get_intro_keywords()
    intro = []
    articles = []
    article_no = 1

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
            content, end_idx, has_author = extract_intro_content_phase1(
                lines, i, processed_lines,
                authors_normalized, authors_original, intro_keywords
            )

            for j in range(i, end_idx):
                if j < len(lines):
                    processed_lines[j] = True

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
                    article_no += 1
                else:
                    intro.append({
                        "doc_id": doc_id,
                        "doc_issue": doc_issue,
                        "heading": matched_keyword,
                        "content": content
                    })
            i = end_idx
        else:
            i += 1

    for pattern_func in (
        extract_pattern_a_forward,
        extract_pattern_b_forward,
        extract_pattern_c_reverse
    ):
        pattern_articles = pattern_func(
            lines, parse_start_idx, len(lines),
            authors_normalized, authors_original,
            processed_lines, intro_keywords
        )
        for article in pattern_articles:
            articles.append({
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "article_no": article_no,
                "article_heading": article["heading"],
                "article_author_name": article["author"],
                "article_content": article["content"]
            })
            article_no += 1

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

    authors_list = [
        {"doc_id": doc_id, "doc_issue": doc_issue, "author_name": a}
        for a in authors_original
    ]

    return {
        "intro": intro,
        "articles": articles,
        "authors_list": authors_list,
        "doc_id": doc_id,
        "doc_issue": doc_issue
    }


# ---------------- Main Processing (DELTA ENABLED) ---------------- #
def process_s3_folder(input_prefix, output_prefix):
    txt_files = list_files(BUCKET_NAME, input_prefix, suffix=".txt")
    if not txt_files:
        print("No TXT files found in S3.")
        return

    # ✅ DELTA: already processed JSONs
    existing_json_files = set(
        list_files(BUCKET_NAME, output_prefix, suffix=".json")
    )

    print(f"Found {len(txt_files)} TXT files")
    print(f"Found {len(existing_json_files)} existing JSON files")
    print("=" * 80)

    shared_authors_dict = build_shared_authors_dict(Path(EXTRACTED_OUTPUT))
    folder_authors = defaultdict(lambda: defaultdict(set))

    processed = 0
    skipped = 0

    for key in txt_files:
        relative_path = key[len(input_prefix):].lstrip("/")
        output_json_key = f"{output_prefix}/{relative_path}".replace(".txt", ".json")

        # ✅ DELTA CHECK
        if output_json_key in existing_json_files:
            print(f"Skipping (already processed): {key}")
            skipped += 1
            continue

        try:
            lines = read_txt_from_s3(BUCKET_NAME, key)
            result = parse_tamil_document(lines, shared_authors_dict)

            upload_json_to_s3(BUCKET_NAME, output_json_key, {
                "intro": result["intro"],
                "articles": result["articles"]
            })

            folder_key = "/".join(relative_path.split("/")[:-1])
            doc_key = (result["doc_id"], result["doc_issue"])
            for author in result["authors_list"]:
                folder_authors[folder_key][doc_key].add(author["author_name"])

            print(f"Processed NEW: {key}")
            processed += 1

        except Exception as e:
            print(f"Error processing {key}: {e}")

    # Upload authors only if new docs processed
    if processed > 0:
        for folder_key, doc_map in folder_authors.items():
            consolidated = []
            for (doc_id, doc_issue), authors in doc_map.items():
                consolidated.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "authors": sorted(list(authors))
                })

            authors_key = (
                f"{output_prefix}/{folder_key}_authors.json"
                if folder_key else f"{output_prefix}/root_authors.json"
            )

            upload_json_to_s3(BUCKET_NAME, authors_key, consolidated)
            print(f"Uploaded authors JSON: {authors_key}")

    print("=" * 80)
    print(f"Processed (new): {processed}")
    print(f"Skipped (old)  : {skipped}")


if __name__ == "__main__":
    print("Processing S3 TXT files (DELTA mode)...")
    process_s3_folder(EXTRACTED_OUTPUT, OUTPUT_PREFIX)
    print("Completed.")
