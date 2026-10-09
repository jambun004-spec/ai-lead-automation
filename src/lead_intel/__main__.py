"""CLI: python -m lead_intel --name "Acme Plumbing" --website https://acme.example"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from lead_intel.config import ConfigurationError
from lead_intel.crew import qualify_lead
from lead_intel.models.lead import LeadInput
from lead_intel.models.report import RunStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research and qualify a single lead.")
    parser.add_argument("--name", required=True, help="Business name")
    parser.add_argument("--website")
    parser.add_argument("--location")
    parser.add_argument("--industry")
    args = parser.parse_args(argv)

    try:
        lead = LeadInput(
            business_name=args.name,
            website=args.website,
            location=args.location,
            industry_hint=args.industry,
        )
        result = qualify_lead(lead)
    except (ValidationError, ConfigurationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(result.model_dump_json(indent=2))
    return 0 if result.status is RunStatus.SCORED else 1


if __name__ == "__main__":
    sys.exit(main())
