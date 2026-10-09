"""Builders for inbound tests and the offline evaluation fixture (no LLM or network)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from lead_intel.inbound.business_config import BusinessConfig, load_business_config
from lead_intel.inbound.models import (
    ContactExtraction,
    EnquiryExtraction,
    ExtractedFlag,
    ExtractedText,
    InboundEnquiry,
    IntentAssessment,
)
from lead_intel.models.evidence import Confidence

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "inbound_enquiries.json"
EXAMPLE_CONFIG_PATH = ROOT / "config" / "businesses" / "example-dental-clinic.json"


@cache
def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def example_config() -> BusinessConfig:
    return load_business_config(EXAMPLE_CONFIG_PATH)


def _quoted(message: str, value: str) -> str:
    if value.casefold() not in message.casefold():
        raise AssertionError(f"fixture value {value!r} is not verbatim in message {message!r}")
    return value


def text(message: str, value: str | None) -> ExtractedText:
    if value is None:
        return ExtractedText(value=None, confidence=Confidence.INSUFFICIENT)
    return ExtractedText(value=value, confidence=Confidence.CONFIRMED, evidence=_quoted(message, value))


def flag(message: str, evidence: str | None) -> ExtractedFlag:
    if evidence is None:
        return ExtractedFlag(value=False, confidence=Confidence.INFERRED)
    return ExtractedFlag(value=True, confidence=Confidence.CONFIRMED, evidence=_quoted(message, evidence))


def build_gold_extraction(case: dict[str, Any]) -> EnquiryExtraction:
    g, msg = case["gold"], case["message"]
    return EnquiryExtraction(
        primary_intent=IntentAssessment(
            intent=g["intent"], confidence=Confidence.CONFIRMED, evidence=_quoted(msg, g["intent_evidence"])
        ),
        secondary_intents=g.get("secondary", []),
        service_requested=text(msg, g.get("service")),
        matched_service_id=g.get("matched_service_id"),
        contact=ContactExtraction(
            name=text(msg, g.get("name")), phone=text(msg, g.get("phone")), email=text(msg, g.get("email"))
        ),
        preferred_datetime_text=text(msg, g.get("datetime")),
        urgency=flag(msg, g.get("urgent")),
        human_requested=flag(msg, g.get("human")),
        scope_concern=flag(msg, g.get("scope")),
        context_summary=g.get("summary", f"Customer enquiry: {g['intent']}"),
        missing_information=g.get("llm_missing", []),
    )


def build_enquiry(case: dict[str, Any], *, message: str | None = None) -> InboundEnquiry:
    return InboundEnquiry(
        business_id="example-dental-clinic",
        channel=case.get("channel", "web_form"),
        external_message_id=case.get("id", "test-message"),
        message_text=message or case["message"],
        received_at=load_fixture()["received_at"],
        channel_contact=case.get("channel_contact"),
    )


def case_by_id(case_id: str) -> dict[str, Any]:
    return next(c for c in load_fixture()["cases"] if c["id"] == case_id)
