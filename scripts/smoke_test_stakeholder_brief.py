from __future__ import annotations

import json
import os
from pathlib import Path

import sys

from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.main import StakeholderBriefRequest, generate_stakeholder_brief


def main() -> None:
    audience = os.getenv("STAKEHOLDER_AUDIENCE", "Executive leadership")
    focus = os.getenv(
        "STAKEHOLDER_FOCUS",
        "Explain the inventory service gap, forecast model comparisons, uncertainty experiments and next decisions.",
    )
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
        "## Part 1. Inventory optimisation and service level",
        "## Part 2. Forecast accuracy and model comparisons",
        "## Part 3. Statistical evidence and experiment decisions",
    )
    absent_sections = [heading for heading in required_sections if heading not in markdown]
    if absent_sections:
        raise RuntimeError(f"Generated report is missing sections: {absent_sections}")

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
