# FreshRetailNet-50K data ingestion

This repository downloads the public Parquet files from the official Hugging Face dataset and uploads them to a **private Cloudflare R2 bucket**. The raw data is not committed to Git.

## Source

- Dataset: [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- Files: `data/train.parquet` and `data/eval.parquet`
- License: CC BY 4.0. Retain attribution to the dataset authors when redistributing or publishing derived work.
- The ingestion script checks the downloaded files against their published SHA-256 checksums before uploading.

## 1. Create R2 credentials

In Cloudflare Dashboard:

1. Open **R2 Object Storage** and create a private bucket (for example, `freshretailnet-50k`).
2. Create an R2 S3 API token with **Object Read & Write** permission scoped to this bucket.
3. Copy the Access Key ID and Secret Access Key when shown.
4. Find your Cloudflare Account ID in the dashboard.

Do not paste credentials into source code, commit them, or put them in repository files.

## 2. Add GitHub Actions secrets

In this GitHub repository, open **Settings → Secrets and variables → Actions → New repository secret**. Add:

| Secret name | Value |
|---|---|
| `R2_ACCOUNT_ID` | Cloudflare Account ID |
| `R2_ACCESS_KEY_ID` | R2 Access Key ID |
| `R2_SECRET_ACCESS_KEY` | R2 Secret Access Key |
| `R2_BUCKET` | Exact R2 bucket name, e.g. `freshretailnet-50k` |

The dataset is public, so a Hugging Face token is not required.

## 3. Run the workflow

1. Open the repository's **Actions** tab.
2. Select **Ingest FreshRetailNet-50K to Cloudflare R2**.
3. Select **Run workflow** on the `main` branch.
4. Open the run and inspect the logs.

The workflow downloads both files on a GitHub-hosted runner, verifies their SHA-256 checksums, uploads them, and checks the stored object sizes with R2 `HeadObject`.

## 4. Expected R2 object keys

```text
raw/train.parquet
raw/eval.parquet
```

The bucket should remain private. Downstream jobs should access the objects using scoped credentials or short-lived presigned URLs; do not make the raw bucket public just to read the files.

## Run locally (optional)

Install dependencies:

```bash
python -m pip install huggingface_hub boto3
```

Set the four environment variables listed above, then run:

```bash
python scripts/ingest_freshretailnet_to_r2.py
```

The script downloads to the Hugging Face cache and uploads directly from the runner or local machine to R2. It does not store the Parquet files in the Git repository.
