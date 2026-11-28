from __future__ import annotations

import json
import os
from pathlib import Path

import sys

from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from botocore.exceptions import ClientError

from api.main import StakeholderBriefRequest, _client, generate_stakeholder_brief


def diagnose_r2_objects() -> None:
    client = _client()
    bucket = os.environ["R2_BUCKET"]
    client.head_bucket(Bucket=bucket)
    print("R2_BUCKET_ACCESS=success")
    for key in (
        "inventory-optimizer-artifacts/inventory_optimizer_summary.json",
        "forecast-research-artifacts/forecast_research_summary.json",
    ):
        try:
            response = client.head_object(Bucket=bucket, Key=key)
            print(f"R2_OBJECT_FOUND={key} SIZE={response.get('ContentLength')}")
        except ClientError as exc:
            error = exc.response.get("Error", {})
            metadata = exc.response.get("ResponseMetadata", {})
            print(
                f"R2_OBJECT_ERROR={key} CODE={error.get('Code')} "
                f"HTTP={metadata.get('HTTPStatusCode')}"
            )
            raise


def main() -> None:
    audience = os.getenv("STAKEHOLDER_AUDIENCE", "Executive leadership")
    focus = os.getenv(
        "STAKEHOLDER_FOCUS",
        "Explain the inventory service gap, forecast model comparisons, uncertainty experiments and next decisions.",
    )
    diagnose_r2_objects()
    result = generate_stakeholder_brief(
        StakeholderBriefRequest(audience=audience, focus=focus)
    )

    expected = {
        "inventory-optimizer-artifacts/inventory_optimizer_summary.json",
        "forecast-research-artifacts/forecast_research_summary.json",
    }
    sources = set(result.get("source_artifacts", []))
    missing_sources = sorted(expected - sources)
    if missing_sources:
        raise RuntimeError(f"Brief did not use both current R2 summaries: {missing_sources}")

    markdown = result.get("brief_markdown", "")
    required_sections = (
        "## Part 1. Inventory service outcomes",
        "## Part 2. Inventory constraints and distribution experiments",
        "## Part 3. Forecast accuracy and model comparisons",
        "## Part 4. Statistical evidence and experiment decisions",
    )
    absent_sections = [heading for heading in required_sections if heading not in markdown]
    if absent_sections:
        raise RuntimeError(f"Generated report is missing sections: {absent_sections}")

    sections = markdown.split("## Part ")[1:]
    if len(sections) != 4:
        raise RuntimeError(f"Expected three report sections, received {len(sections)}.")
    short_sections = [
        index for index, section in enumerate(sections, start=1)
        if len(section.split()) < 150
    ]
    if short_sections:
        raise RuntimeError(f"Generated sections appear truncated or too short: {short_sections}")
    usage_parts = (result.get("usage") or {}).get("parts") or []
    capped_parts = [
        index for index, item in enumerate(usage_parts, start=1)
        if isinstance(item, dict) and item.get("completion_tokens", 0) >= 630
    ]
    if capped_parts:
        raise RuntimeError(
            f"Groq reached the 650-token completion cap in sections {capped_parts}; "
            "report may be truncated."
        )

    report_path = Path("stakeholder-brief.md")
    report_path.write_text(markdown, encoding="utf-8")
    summary = {
        "status": "success",
        "model": result.get("model"),
        "generation_strategy": result.get("generation_strategy"),
        "generated_at_utc": result.get("generated_at_utc"),
        "source_artifacts": result.get("source_artifacts"),
        "missing_source_artifacts": result.get("missing_source_artifacts"),
        "report_characters": len(markdown),
        "report_bytes": report_path.stat().st_size,
        "usage": result.get("usage"),
    }
    Path("stakeholder-brief-run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("STAKEHOLDER_BRIEF_SMOKE_TEST=SUCCESS")
    print(f"MODEL={summary['model']}")
    print(f"SOURCE_ARTIFACTS={len(summary['source_artifacts'])}")
    print(f"REPORT_CHARACTERS={summary['report_characters']}")
    print(f"REPORT_BYTES={summary['report_bytes']}")
    print(f"USAGE={json.dumps(summary['usage'], default=str)}")


if __name__ == "__main__":
    try:
        main()
    except HTTPException as exc:
        print(f"STAKEHOLDER_BRIEF_HTTP_ERROR={exc.status_code}")
        print(json.dumps(exc.detail, ensure_ascii=False, default=str))
        raise
