from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from lead_intel.inbound.models import (
    DateTimeStatus,
    EnquiryExtraction,
    EnquiryExtractionResult,
    ExtractedFlag,
    ExtractedText,
    ExtractionStatus,
    InboundEnquiry,
    IntentAssessment,
    NormalizedDateTime,
)
from lead_intel.models.evidence import Confidence
from tests.inbound_support import build_enquiry, build_gold_extraction, case_by_id

C, I, X = Confidence.CONFIRMED, Confidence.INFERRED, Confidence.INSUFFICIENT


def _enquiry(**overrides) -> dict:
    data = {
        "business_id": "example-dental-clinic",
        "channel": "web_form",
        "external_message_id": "m-1",
        "message_text": "Hello",
        "received_at": "2026-10-07T10:00:00+05:30",
    }
    data.update(overrides)
    return data


def test_valid_enquiry():
    enquiry = InboundEnquiry(**_enquiry(message_text="  Hello  "))
    assert enquiry.message_text == "Hello"


@pytest.mark.parametrize(
    "overrides",
    [
        {"received_at": "2026-10-07T10:00:00"},  # naive: relative dates would be unsafe
        {"business_id": "Example Clinic"},
        {"channel": "carrier_pigeon"},
        {"external_message_id": ""},
        {"message_text": "   "},
        {"message_text": "x" * 5001},
        {"priority": "high"},
    ],
)
def test_invalid_enquiry_rejected(overrides):
    with pytest.raises(ValidationError):
        InboundEnquiry(**_enquiry(**overrides))


# --- per-field confidence rules ------------------------------------------------


def test_confirmed_text_requires_evidence():
    with pytest.raises(ValidationError, match="evidence"):
        ExtractedText(value="Priya", confidence=C)


@pytest.mark.parametrize(
    "value, confidence",
    [(None, C), (None, I), ("Priya", X)],
)
def test_value_null_exactly_when_insufficient(value, confidence):
    with pytest.raises(ValidationError, match="insufficient"):
        ExtractedText(value=value, confidence=confidence, evidence="Priya")


def test_inferred_text_without_evidence_is_allowed():
    assert ExtractedText(value="evening", confidence=I).evidence is None


def test_true_flag_cannot_be_insufficient():
    with pytest.raises(ValidationError):
        ExtractedFlag(value=True, confidence=X)
    assert ExtractedFlag(value=False, confidence=X).value is False


def test_confirmed_intent_requires_evidence():
    with pytest.raises(ValidationError, match="evidence"):
        IntentAssessment(intent="pricing_enquiry", confidence=C)


def test_extraction_rejects_scores_and_unknown_fields():
    data = build_gold_extraction(case_by_id("pricing-whitening")).model_dump(mode="json")
    data["lead_score"] = 90
    with pytest.raises(ValidationError, match="lead_score"):
        EnquiryExtraction.model_validate(data)


# --- result invariants ---------------------------------------------------------------


def test_normalized_datetime_invariants():
    with pytest.raises(ValidationError):
        NormalizedDateTime(status=DateTimeStatus.RESOLVED, timezone="UTC")
    with pytest.raises(ValidationError):
        NormalizedDateTime(status=DateTimeStatus.AMBIGUOUS, timezone="UTC", date=dt.date(2026, 10, 10))
    with pytest.raises(ValidationError):
        NormalizedDateTime(
            status=DateTimeStatus.RESOLVED, timezone="UTC", date=dt.date(2026, 10, 10),
            start=dt.datetime(2026, 10, 10, 9, tzinfo=dt.timezone.utc),
        )  # fmt: skip


def test_failed_result_cannot_carry_an_extraction():
    case = case_by_id("pricing-whitening")
    with pytest.raises(ValidationError):
        EnquiryExtractionResult(
            enquiry=build_enquiry(case), status=ExtractionStatus.FAILED,
            extraction=build_gold_extraction(case), errors=["x"],
        )  # fmt: skip
    with pytest.raises(ValidationError):
        EnquiryExtractionResult(enquiry=build_enquiry(case), status=ExtractionStatus.FAILED)


def test_failed_result_is_not_actionable():
    result = EnquiryExtractionResult(
        enquiry=build_enquiry(case_by_id("pricing-whitening")), status=ExtractionStatus.FAILED, errors=["bad"]
    )
    assert result.actionable is False
