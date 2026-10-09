"""Inbound enquiry domain models and the structured LLM extraction contract.

Vertical-neutral: business-specific vocabulary (services, scope, keywords) lives
in ``BusinessConfig``, never here.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from lead_intel.models.evidence import Confidence

BUSINESS_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"


# --- inbound enquiry (input) --------------------------------------------------


class Channel(str, Enum):
    WEB_FORM = "web_form"
    WEB_CHAT = "web_chat"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    SMS = "sms"
    OTHER = "other"


class ChannelContact(BaseModel):
    """Contact details supplied by the channel itself (e.g. WhatsApp sender number)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, max_length=200)
    phone: str | None = Field(default=None, max_length=50)
    email: str | None = Field(default=None, max_length=320)


class InboundEnquiry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_id: str = Field(pattern=BUSINESS_ID_PATTERN)
    channel: Channel
    external_message_id: str = Field(min_length=1, max_length=200)
    message_text: str = Field(min_length=1, max_length=5000)
    received_at: AwareDatetime  # timezone required: relative dates depend on it
    channel_contact: ChannelContact | None = None


# --- LLM extraction contract (output) -----------------------------------------


class Intent(str, Enum):
    PRICING_ENQUIRY = "pricing_enquiry"
    APPOINTMENT_REQUEST = "appointment_request"
    AVAILABILITY_ENQUIRY = "availability_enquiry"
    SERVICE_INFORMATION = "service_information"
    RESCHEDULE_OR_CANCEL = "reschedule_or_cancel"
    COMPLAINT = "complaint"
    GENERAL_QUESTION = "general_question"
    OTHER = "other"


class MissingInfo(str, Enum):
    CUSTOMER_NAME = "customer_name"
    CONTACT_METHOD = "contact_method"
    SERVICE = "service"
    PREFERRED_DATETIME = "preferred_datetime"


class _Grounded(BaseModel):
    """Shared confidence rules for every extracted item.

    - confirmed  -> the message states it; ``evidence`` must quote the message.
    - inferred   -> reasonable reading of indirect wording.
    - insufficient -> not determinable; no value may be asserted.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    confidence: Confidence
    evidence: str | None = Field(default=None, max_length=500, description="Verbatim quote from the message.")

    def _check_confirmed_has_evidence(self) -> None:
        if self.confidence is Confidence.CONFIRMED and not self.evidence:
            raise ValueError("confirmed items require a verbatim evidence quote")


class ExtractedText(_Grounded):
    value: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _consistent(self) -> ExtractedText:
        if (self.value is None) != (self.confidence is Confidence.INSUFFICIENT):
            raise ValueError("value must be null exactly when confidence is 'insufficient'")
        self._check_confirmed_has_evidence()
        return self


class ExtractedFlag(_Grounded):
    value: bool

    @model_validator(mode="after")
    def _consistent(self) -> ExtractedFlag:
        if self.value and self.confidence is Confidence.INSUFFICIENT:
            raise ValueError("a true flag cannot have confidence 'insufficient'")
        self._check_confirmed_has_evidence()
        return self


class IntentAssessment(_Grounded):
    intent: Intent

    @model_validator(mode="after")
    def _consistent(self) -> IntentAssessment:
        self._check_confirmed_has_evidence()
        return self


class ContactExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ExtractedText
    phone: ExtractedText
    email: ExtractedText


class EnquiryExtraction(BaseModel):
    """Exactly what the extraction LLM must return. No dates are resolved here."""

    model_config = ConfigDict(extra="forbid")

    primary_intent: IntentAssessment
    secondary_intents: list[Intent] = Field(default_factory=list, max_length=5)
    service_requested: ExtractedText
    matched_service_id: str | None = Field(default=None, description="Id from the configured service list, or null.")
    contact: ContactExtraction
    preferred_datetime_text: ExtractedText = Field(description="Raw date/time wording, copied not resolved.")
    urgency: ExtractedFlag
    human_requested: ExtractedFlag
    scope_concern: ExtractedFlag = Field(description="Asks for something outside the business's allowed scope.")
    context_summary: str = Field(min_length=1, max_length=600)
    missing_information: list[MissingInfo] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list, max_length=10)


# --- deterministic post-processing results ------------------------------------


class DateTimeStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_PROVIDED = "not_provided"
    UNRECOGNIZED = "unrecognized"


class TimeWindow(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


class NormalizedDateTime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DateTimeStatus
    raw_text: str | None = None
    timezone: str
    date: dt.date | None = None
    time: dt.time | None = None
    time_window: TimeWindow | None = None
    start: AwareDatetime | None = None
    candidates: list[dt.date] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> NormalizedDateTime:
        if (self.status is DateTimeStatus.RESOLVED) != (self.date is not None):
            raise ValueError("a date is present exactly when status is 'resolved'")
        if self.start is not None and (self.date is None or self.time is None):
            raise ValueError("start requires both date and time")
        return self


class ExtractionStatus(str, Enum):
    EXTRACTED = "extracted"
    FAILED = "failed"


class EnquiryExtractionResult(BaseModel):
    """Pipeline output. A FAILED result carries no extraction and is never actionable."""

    enquiry: InboundEnquiry
    status: ExtractionStatus
    extraction: EnquiryExtraction | None = None
    matched_service_id: str | None = None
    preferred_datetime: NormalizedDateTime | None = None
    missing_information: list[MissingInfo] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> EnquiryExtractionResult:
        if self.status is ExtractionStatus.FAILED and (self.extraction is not None or not self.errors):
            raise ValueError("a FAILED result has errors and no extraction")
        if self.status is ExtractionStatus.EXTRACTED and (self.extraction is None or self.errors):
            raise ValueError("an EXTRACTED result has an extraction and no errors")
        return self

    @property
    def actionable(self) -> bool:
        """Later phases may only act on validated extractions."""
        return self.status is ExtractionStatus.EXTRACTED

    @property
    def needs_clarification(self) -> bool:
        return bool(self.missing_information or self.ambiguities)
