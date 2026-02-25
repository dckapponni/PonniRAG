import boto3
import logging
from io import BytesIO
import json
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger('TamilDocProcessor.s3_utils')

try:
    s3 = boto3.client('s3')
    logger.debug("S3 client initialized successfully in s3_utils")
except NoCredentialsError:
    logger.critical("AWS credentials not found in s3_utils module")
    raise
except Exception as e:
    logger.critical(f"Failed to initialize S3 client in s3_utils: {e}")
    raise


def list_files(bucket, prefix, suffix=None):
    """
    List files in S3 bucket/prefix, optionally filtered by suffix.
    
    Args:
        bucket (str): S3 bucket name
        prefix (str): S3 prefix path
        suffix (str, optional): File extension filter (e.g., '.txt', '.json')
        
    Returns:
        list: List of S3 object keys matching the criteria
        
    Raises:
        ClientError: If S3 operation fails
    """
    try:
        logger.debug(f"Listing files: bucket={bucket}, prefix={prefix}, suffix={suffix}")
        files = []
        paginator = s3.get_paginator('list_objects_v2')
        
        page_count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            page_count += 1
            for obj in page.get('Contents', []):
                key = obj['Key']
                if suffix is None or key.endswith(suffix):
                    files.append(key)
        
        logger.info(f"Listed {len(files)} files from {page_count} pages in s3://{bucket}/{prefix}")
        return files
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError listing files: {error_code} - {e}")
        if error_code == 'NoSuchBucket':
            logger.error(f"Bucket does not exist: {bucket}")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied to bucket: {bucket}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error listing files in s3://{bucket}/{prefix}: {e}", exc_info=True)
        raise


def read_bytes(bucket, key):
    """
    Read a binary file from S3 (like .docx, .pdf).
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        BytesIO: Binary stream of file contents
        
    Raises:
        ClientError: If S3 operation fails
    """
    try:
        logger.debug(f"Reading binary file: s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj['Body'].read()
        logger.debug(f"Read {len(data)} bytes from {key}")
        return BytesIO(data)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"File not found: s3://{bucket}/{key}")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied reading file: s3://{bucket}/{key}")
        else:
            logger.error(f"S3 ClientError reading file: {error_code} - {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error reading binary file {key}: {e}", exc_info=True)
        raise


def upload_text(bucket, key, text):
    """
    Upload text to S3 with UTF-8 encoding.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key where text will be stored
        text (str): Text content to upload
        
    Raises:
        ClientError: If S3 operation fails
        UnicodeEncodeError: If text contains invalid characters
    """
    try:
        logger.debug(f"Uploading text file: s3://{bucket}/{key}")
        
        if not isinstance(text, str):
            logger.warning(f"Expected string for text upload, got {type(text).__name__}")
            text = str(text)
        
        body = text.encode('utf-8')
        logger.debug(f"Encoded text to {len(body)} bytes")
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType='text/plain; charset=utf-8'
        )
        
        logger.info(f"Successfully uploaded text file: s3://{bucket}/{key}")
        
    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error for {key}: {e}")
        raise
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError uploading text: {error_code} - {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error uploading text file {key}: {e}", exc_info=True)
        raise


def upload_json(bucket, key, data):
    """
    Upload JSON to S3 with UTF-8 encoding and pretty formatting.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key where JSON will be stored
        data (dict/list): Data to serialize as JSON
        
    Raises:
        ClientError: If S3 operation fails
        TypeError: If data is not JSON serializable
    """
    try:
        logger.debug(f"Uploading JSON file: s3://{bucket}/{key}")
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        logger.debug(f"Serialized JSON to {len(json_str)} characters")
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json_str.encode('utf-8'),
            ContentType='application/json; charset=utf-8'
        )
        
        logger.info(f"Successfully uploaded JSON file: s3://{bucket}/{key}")
        
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization error for {key}: {e}")
        logger.error(f"Data type: {type(data).__name__}")
        raise
        
    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error for JSON {key}: {e}")
        raise
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError uploading JSON: {error_code} - {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error uploading JSON file {key}: {e}", exc_info=True)
        raise


def read_text_from_s3(bucket, key):
    """
    Read text file from S3 and return as UTF-8 string.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        str: Text content of file
        
    Raises:
        ClientError: If S3 operation fails
        UnicodeDecodeError: If file is not valid UTF-8
    """
    try:
        logger.debug(f"Reading text file: s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        text = obj['Body'].read().decode('utf-8')
        logger.debug(f"Read {len(text)} characters from {key}")
        return text
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"File not found: s3://{bucket}/{key}")
        else:
            logger.error(f"S3 ClientError reading text: {error_code} - {e}")
        raise
        
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error reading {key}: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error reading text file {key}: {e}", exc_info=True)
        raise


def read_json_from_s3(bucket, key):
    """
    Read and parse JSON file from S3.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        dict/list: Parsed JSON data
        
    Raises:
        ClientError: If S3 operation fails
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        logger.debug(f"Reading JSON file: s3://{bucket}/{key}")
        text = read_text_from_s3(bucket, key)
        data = json.loads(text)
        logger.debug(f"Successfully parsed JSON from {key}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {key}: {e}")
        logger.error(f"Error at line {e.lineno}, column {e.colno}")
        raise
        
    except Exception as e:
        logger.error(f"Error reading JSON file {key}: {e}", exc_info=True)
        raise


def file_exists(bucket, key):
    """
    Check if a file exists in S3.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        logger.debug(f"Checking if file exists: s3://{bucket}/{key}")
        s3.head_object(Bucket=bucket, Key=key)
        logger.debug(f"File exists: {key}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.debug(f"File does not exist: {key}")
            return False
        else:
            logger.warning(f"Error checking file existence for {key}: {error_code}")
            return False
            
    except Exception as e:
        logger.warning(f"Unexpected error checking file existence {key}: {e}")
        return False


