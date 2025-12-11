import boto3
from io import BytesIO
import json

s3 = boto3.client('s3')

def list_files(bucket, prefix, suffix=None):
    """List files in S3 bucket/prefix, optionally filtered by suffix"""
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if suffix is None or key.endswith(suffix):
                files.append(key)
    return files

def read_bytes(bucket, key):
    """Read a binary file from S3 (like .docx)"""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return BytesIO(obj['Body'].read())

def upload_text(bucket, key, text):
    """Upload text to S3"""
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode('utf-8'))

def upload_json(bucket, key, data):
    """Upload JSON to S3"""
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
