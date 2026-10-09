"""Extract one enquiry (live LLM call):

python -m lead_intel.inbound --config config/businesses/<business>.json \
    --message "How much is <service>? Any slot this Saturday?"
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import uuid

from pydantic import ValidationError

from lead_intel.config import ConfigurationError
from lead_intel.inbound.business_config import load_business_config
from lead_intel.inbound.extraction import extract_enquiry
from lead_intel.inbound.models import Channel, InboundEnquiry
from lead_intel.inbound.settings import load_inbound_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from one inbound enquiry.")
    parser.add_argument("--message", required=True)
    parser.add_argument("--config", help="Business config JSON (default: LEAD_OPS_BUSINESS_CONFIG_PATH)")
    parser.add_argument("--channel", default=Channel.WEB_FORM.value, choices=[c.value for c in Channel])
    parser.add_argument("--received-at", help="ISO datetime with offset (default: now, UTC)")
    args = parser.parse_args(argv)

    try:
        settings = load_inbound_settings()
        config_path = args.config or settings.business_config_path
        if not config_path:
            raise ConfigurationError("Pass --config or set LEAD_OPS_BUSINESS_CONFIG_PATH.")
        config = load_business_config(config_path)
        enquiry = InboundEnquiry(
            business_id=config.business_id,
            channel=args.channel,
            external_message_id=f"cli-{uuid.uuid4()}",
            message_text=args.message,
            received_at=args.received_at or dt.datetime.now(dt.timezone.utc),
        )
        result = extract_enquiry(enquiry, config, settings=settings)
    except (ValidationError, ConfigurationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(result.model_dump_json(indent=2))
    return 0 if result.actionable else 1


if __name__ == "__main__":
    sys.exit(main())
