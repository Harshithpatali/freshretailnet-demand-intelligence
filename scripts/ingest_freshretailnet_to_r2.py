"""Download FreshRetailNet-50K Parquet files from Hugging Face and upload to Cloudflare R2.

Required environment variables:
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET

The dataset is public, so HF_TOKEN is not required.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import boto3
from botocore.config import Config
from huggingface_hub import hf_hub_download

DATASET_REPO = "Dingdong-Inc/FreshRetailNet-50K"
REVISION = "main"
FILES = {
    "data/train.parquet": {
        "key": "raw/train.parquet",
        "sha256": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    },
    "data/eval.parquet": {
        "key": "raw/eval.parquet",
        "sha256": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    account_id = required_env("R2_ACCOUNT_ID")
    access_key = required_env("R2_ACCESS_KEY_ID")
    secret_key = required_env("R2_SECRET_ACCESS_KEY")
    bucket = required_env("R2_BUCKET")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )

    for filename, metadata in FILES.items():
        print(f"Downloading {filename} from {DATASET_REPO} ({REVISION})...")
        downloaded = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=filename,
                revision=REVISION,
            )
        )

        actual_hash = sha256_file(downloaded)
        expected_hash = metadata["sha256"]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )

        print(f"Verified {filename}: {downloaded.stat().st_size:,} bytes")
        object_key = metadata["key"]
        print(f"Uploading to s3://{bucket}/{object_key}...")
        s3.upload_file(
            str(downloaded),
            bucket,
            object_key,
            ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
        )

        head = s3.head_object(Bucket=bucket, Key=object_key)
        if head["ContentLength"] != downloaded.stat().st_size:
            raise RuntimeError(f"R2 size verification failed for {object_key}")

        print(f"Uploaded and verified: {object_key} ({head['ContentLength']:,} bytes)")

    print("All FreshRetailNet-50K files were uploaded and verified successfully.")


if __name__ == "__main__":
    main()
