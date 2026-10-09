"""Inbound extraction pipeline.

LLM text -> Pydantic validation -> deterministic grounding checks -> date/time
normalization -> missing-information rules. Invalid output yields a FAILED
result with no extraction; nothing downstream may act on it. This module performs
no external actions itself (no replies, bookings or persistence).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from crewai import Crew, Process

from lead_intel.inbound.business_config import BusinessConfig
from lead_intel.inbound.datetime_normalization import normalize_preferred_datetime
from lead_intel.inbound.extraction_task import build_extraction_agent, build_extraction_prompt, build_extraction_task
from lead_intel.inbound.models import (
    DateTimeStatus,
    EnquiryExtraction,
    EnquiryExtractionResult,
    ExtractedText,
    ExtractionStatus,
    InboundEnquiry,
    MissingInfo,
    NormalizedDateTime,
)
from lead_intel.inbound.settings import InboundSettings, load_inbound_settings
from lead_intel.llm_output import validate_llm_json
from lead_intel.models.evidence import Confidence

Runner = Callable[[str], "str | None"]  # prompt -> raw LLM text

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_QUOTES = " \t\"'“”‘’.,!?;:"


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_grounded(evidence: str | None, message: str) -> bool:
    return bool(evidence) and _norm(evidence.strip(_QUOTES)) in _norm(message)


def _ground(extraction: EnquiryExtraction, message: str, notes: list[str]) -> EnquiryExtraction:
    """'confirmed' needs a verbatim quote; otherwise it is scored down to 'inferred'."""
    data = extraction.model_dump()
    paths = [
        ("primary_intent",), ("service_requested",), ("preferred_datetime_text",),
        ("urgency",), ("human_requested",), ("scope_concern",),
        ("contact", "name"), ("contact", "phone"), ("contact", "email"),
    ]  # fmt: skip
    for path in paths:
        item = data
        for key in path:
            item = item[key]
        if item["confidence"] == Confidence.CONFIRMED and not _is_grounded(item["evidence"], message):
            item["confidence"] = Confidence.INFERRED
            notes.append(f"{'.'.join(path)}: evidence not found verbatim in message; treated as inferred")
    return EnquiryExtraction.model_validate(data)


def _confirmed_value(item: ExtractedText) -> str | None:
    return item.value if item.confidence is Confidence.CONFIRMED else None


def _valid_phone(value: str) -> bool:
    return 7 <= len(re.sub(r"\D", "", value)) <= 15


def failed_result(enquiry: InboundEnquiry, errors: list[str]) -> EnquiryExtractionResult:
    return EnquiryExtractionResult(enquiry=enquiry, status=ExtractionStatus.FAILED, errors=errors)


def process_extraction_output(
    enquiry: InboundEnquiry, config: BusinessConfig, raw_text: str | None
) -> EnquiryExtractionResult:
    """The single deterministic gate between LLM text and anything downstream."""
    outcome = validate_llm_json(EnquiryExtraction, raw_text)
    if not outcome.ok:
        return failed_result(enquiry, outcome.errors)

    notes: list[str] = []
    extraction = _ground(outcome.value, enquiry.message_text, notes)
    ambiguities = list(extraction.ambiguities)

    # Service: only configured ids are accepted.
    matched = extraction.matched_service_id
    if matched is not None and matched not in config.service_ids():
        notes.append(f"matched_service_id {matched!r} is not a configured service; ignored")
        matched = None
    if extraction.service_requested.value and matched is None:
        ambiguities.append(f"requested service {extraction.service_requested.value!r} is not a configured service")

    # Contact details: confirmed and well-formed, or supplied by the channel itself.
    contact = extraction.contact
    email, phone = _confirmed_value(contact.email), _confirmed_value(contact.phone)
    if email and not _EMAIL.match(email):
        ambiguities.append(f"email {email!r} does not look valid")
        email = None
    if phone and not _valid_phone(phone):
        ambiguities.append(f"phone {phone!r} does not look valid")
        phone = None
    channel = enquiry.channel_contact
    has_contact = bool(email or phone or (channel and (channel.email or channel.phone)))
    has_name = bool(_confirmed_value(contact.name) or (channel and channel.name))

    # Date/time: only the customer's own (confirmed) wording is resolved.
    dt_item = extraction.preferred_datetime_text
    if dt_item.value is None:
        preferred = NormalizedDateTime(status=DateTimeStatus.NOT_PROVIDED, timezone=config.timezone)
    elif dt_item.confidence is not Confidence.CONFIRMED:
        preferred = NormalizedDateTime(
            status=DateTimeStatus.AMBIGUOUS, raw_text=dt_item.value, timezone=config.timezone,
            reason="date/time wording is not quoted from the message",
        )  # fmt: skip
    else:
        preferred = normalize_preferred_datetime(
            dt_item.value, reference=enquiry.received_at, tz=config.tz, date_order=config.date_order
        )
    if preferred.status in (DateTimeStatus.AMBIGUOUS, DateTimeStatus.UNRECOGNIZED):
        ambiguities.append(f"preferred date/time {preferred.raw_text!r}: {preferred.reason}")

    # Missing information: deterministic, from the configured rules.
    rules = config.qualification_rules
    required = set(rules.required_for_all)
    intents = {extraction.primary_intent.intent, *extraction.secondary_intents}
    if intents & set(rules.appointment_intents):
        required |= set(rules.required_for_appointment)
    present = {
        MissingInfo.CUSTOMER_NAME: has_name,
        MissingInfo.CONTACT_METHOD: has_contact,
        MissingInfo.SERVICE: matched is not None,
        MissingInfo.PREFERRED_DATETIME: preferred.status is DateTimeStatus.RESOLVED,
    }
    missing = [item for item in MissingInfo if item in required and not present[item]]
    extra = [m.value for m in extraction.missing_information if m not in missing]
    if extra:
        notes.append(f"LLM also listed as missing (not required by rules): {', '.join(extra)}")

    return EnquiryExtractionResult(
        enquiry=enquiry,
        status=ExtractionStatus.EXTRACTED,
        extraction=extraction,
        matched_service_id=matched,
        preferred_datetime=preferred,
        missing_information=missing,
        ambiguities=ambiguities,
        notes=notes,
    )


def crew_runner(settings: InboundSettings) -> Runner:
    """Default runner: one CrewAI agent + task with a schema guardrail."""
    agent = build_extraction_agent(settings)  # configuration errors surface here

    def run(prompt: str) -> str | None:
        task = build_extraction_task(prompt, agent, settings)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return crew.kickoff().raw  # no inputs: the prompt contains literal JSON braces

    return run


def extract_enquiry(
    enquiry: InboundEnquiry,
    config: BusinessConfig,
    *,
    settings: InboundSettings | None = None,
    runner: Runner | None = None,
) -> EnquiryExtractionResult:
    if enquiry.business_id != config.business_id:
        raise ValueError(f"enquiry is for {enquiry.business_id!r}, config is for {config.business_id!r}")
    if runner is None:
        runner = crew_runner(settings or load_inbound_settings())
    prompt = build_extraction_prompt(enquiry, config)
    try:
        raw = runner(prompt)
    except Exception as exc:  # LLM/network/guardrail-exhaustion failures fail safely
        return failed_result(enquiry, [f"Extraction failed: {type(exc).__name__}: {exc}"])
    return process_extraction_output(enquiry, config, raw)
