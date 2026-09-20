"""Resume PDFs in S3.

The database keeps the extracted text and the analysis; the original file
lives in S3 under a key the `resumes` row points at. Uploading is optional —
with `S3_BUCKET` unset every call here is a no-op, so local development and
the existing flow work with no AWS account.

Resume PDFs are candidate personal data, so the bucket is expected to be
private: what gets stored is the object key, and links handed to the browser
are presigned and short-lived rather than public URLs.
"""

import logging
import os
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "resumes").strip().strip("/")
S3_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "")).strip()
# Set for MinIO or LocalStack; left empty for real S3.
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip() or None
# How long a download link stays valid.
PRESIGN_EXPIRY_SECONDS = int(os.getenv("S3_PRESIGN_EXPIRY", "900"))

_client = None


def enabled() -> bool:
    return bool(S3_BUCKET)


def _get_client():
    """Created on first use so the app still boots without boto3 or creds."""
    global _client
    if _client is None:
        import boto3  # imported late: only needed when S3 is configured
        from botocore.config import Config

        _client = boto3.client(
            "s3",
            region_name=S3_REGION or None,
            endpoint_url=S3_ENDPOINT_URL,
            # Without this, presigned URLs are signed for the bucket's region
            # but addressed to the global s3.amazonaws.com host, which answers
            # with a redirect that breaks the signature outside us-east-1.
            # MinIO and LocalStack need path addressing instead.
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if S3_ENDPOINT_URL else "virtual"},
            ),
        )
    return _client


@dataclass
class StoredFile:
    key: str
    url: str


# What a stored file is. Part of the S3 key, so these strings are durable.
KIND_ORIGINAL = "original"
KIND_JOB_DESCRIPTION = "job-description"
KIND_IMPROVED = "improved"
KIND_TAILORED = "tailored"


def build_key(user_id: int, kind: str, filename: str) -> str:
    """`resumes/<user>/<kind>/<uuid>.pdf`.

    The uploaded name is deliberately not part of the key — it is attacker
    controlled, and the display name is already a column on the row. Keeping
    every kind under `resumes/<user>/` means one prefix sweep still clears an
    entire account.
    """
    extension = (filename or "").rsplit(".", 1)[-1].lower()
    suffix = f".{extension}" if extension and extension.isalnum() else ""
    return f"{S3_PREFIX}/{user_id}/{kind}/{uuid.uuid4().hex}{suffix}"


def object_url(key: str) -> str:
    """The object's canonical address. Not a download link on a private bucket."""
    if S3_ENDPOINT_URL:
        return f"{S3_ENDPOINT_URL.rstrip('/')}/{S3_BUCKET}/{key}"
    if S3_REGION:
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"


def upload(
    user_id: int,
    data: bytes,
    kind: str = KIND_ORIGINAL,
    filename: str = "resume.pdf",
    content_type: str = "application/pdf",
) -> StoredFile | None:
    """Store one file. Returns None when S3 is off or the upload fails.

    A storage failure must not cost the user work they already paid for, so
    this logs and gives up rather than raising into the request.
    """
    if not enabled() or not data:
        return None

    key = build_key(user_id, kind, filename)
    try:
        _get_client().put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
            ServerSideEncryption="AES256",
            Metadata={"original-filename": (filename or "")[:200]},
        )
    except Exception as e:
        logger.warning("S3 upload failed for user %s: %s", user_id, e)
        return None

    return StoredFile(key=key, url=object_url(key))


def presigned_url(key: str, expires_in: int | None = None) -> str | None:
    """A time-limited GET link for a private object."""
    if not enabled() or not key:
        return None
    try:
        return _get_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires_in or PRESIGN_EXPIRY_SECONDS,
        )
    except Exception as e:
        logger.warning("Could not presign %s: %s", key, e)
        return None


def delete(key: str) -> None:
    if not enabled() or not key:
        return
    try:
        _get_client().delete_object(Bucket=S3_BUCKET, Key=key)
    except Exception as e:
        logger.warning("Could not delete %s: %s", key, e)


def delete_user_files(user_id: int) -> int:
    """Every object for one account — the database cascade cannot reach S3.

    Returns how many objects were removed.
    """
    if not enabled():
        return 0

    prefix = f"{S3_PREFIX}/{user_id}/"
    removed = 0
    try:
        client = _get_client()
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=S3_BUCKET, Prefix=prefix
        )
        for page in pages:
            keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if not keys:
                continue
            client.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": keys})
            removed += len(keys)
    except Exception as e:
        logger.warning("Could not clear S3 objects for user %s: %s", user_id, e)
    return removed
