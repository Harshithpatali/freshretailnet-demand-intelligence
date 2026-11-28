"""Immutable provenance and versioned publication helpers for R2 experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    workflow_name: str,
    inputs: Mapping[str, str | Path],
    evaluation_start: str | None,
    evaluation_end: str | None,
    configuration: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Create a content-addressed data snapshot and run/config manifest."""
    input_hashes = {
        name: {"sha256": sha256_file(path), "bytes": Path(path).stat().st_size}
        for name, path in sorted(inputs.items())
    }
    snapshot_payload = json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode()
    snapshot_sha = hashlib.sha256(snapshot_payload).hexdigest()
    commit_sha = os.getenv("GITHUB_SHA", "local")
    run_id = os.getenv("GITHUB_RUN_ID", "manual")
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    return {
        "manifest_schema_version": 1,
        "workflow_name": workflow_name,
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "ref": os.getenv("GITHUB_REF"),
            "commit_sha": commit_sha,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "run_url": (
                f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{run_id}"
                if os.getenv("GITHUB_REPOSITORY") and run_id != "manual" else None
            ),
        },
        "data_snapshot": {
            "sha256": snapshot_sha,
            "inputs": input_hashes,
        },
        "evaluation_period": {
            "start": evaluation_start,
            "end": evaluation_end,
        },
        "configuration": configuration,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "versioned_outputs": list(outputs or []),
    }


def versioned_key(key: str, manifest: dict) -> str:
    github = manifest.get("github", {})
    commit = str(github.get("commit_sha") or "unknown")[:12]
    run_id = str(github.get("run_id") or "manual")
    attempt = str(github.get("run_attempt") or "1")
    workflow = str(manifest.get("workflow_name", "experiment")).strip().lower().replace(" ", "-")
    return f"experiment-runs/{commit}/{run_id}-attempt-{attempt}/{workflow}/{key.lstrip('/')}"


def upload_versioned_artifact(
    s3,
    bucket: str,
    local_path: str | Path,
    key: str,
    content_type: str,
    manifest: dict,
) -> list[str]:
    """Publish stable API alias and immutable commit/run-scoped copy; verify both."""
    path = Path(local_path)
    destinations = [key, versioned_key(key, manifest)]
    for destination in destinations:
        s3.upload_file(str(path), bucket, destination, ExtraArgs={"ContentType": content_type})
        remote_size = s3.head_object(Bucket=bucket, Key=destination)["ContentLength"]
        if remote_size != path.stat().st_size:
            raise RuntimeError(f"R2 upload verification failed for {destination}")
        print(f"Verified s3://{bucket}/{destination} ({remote_size:,} bytes)")
    return destinations


def publish_manifest(s3, bucket: str, manifest: dict, workflow_name: str) -> tuple[Path, list[str]]:
    """Write and publish a manifest in stable and immutable locations."""
    import tempfile

    path = Path(tempfile.gettempdir()) / f"{workflow_name.replace(' ', '_')}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    key = f"experiment-metadata/{workflow_name.replace(' ', '-').lower()}/latest_manifest.json"
    destinations = upload_versioned_artifact(
        s3, bucket, path, key, "application/json", manifest
    )
    return path, destinations
